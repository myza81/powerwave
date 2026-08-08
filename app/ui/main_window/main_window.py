"""PowerwaveMainWindow — operational Powerwave viewer.

Supports two display modes:
  Standard   — single DisturbanceRecord in Phase 4A two-pane layout
  Grouped    — mixed-source record in stacked multi-panel layout (Phase D2)

File loading runs on a QRunnable background thread.
Grouped synthetic load runs on the UI thread (fast, <100 ms).

Phase D4.4 additions:
  Direct CSV/Excel opens run IntelligenceManager classification on the worker
  thread, route through display_grouped_record(), and show DataReviewDialog
  when timestamp interpretation is ambiguous or any column needs confirmation.
  COMTRADE direct-opens preserve the existing set_record() path.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)

from app.analytics.frequency import FrequencyDisplayMode, FrequencyRegistry
from app.analytics.harmonics import HarmonicRegistry
from app.analytics.harmonics.harmonic_models import HarmonicDisplayMode
from app.analytics.phasors import PhasorDisplayMode, PhasorRegistry
from app.analytics.rms.rms_models import RMSConfig, RMSDisplayMode, RMSWindowMode
from app.analytics.scaling import EngineeringScalingMode, GlobalScalingConfig, ScalingRegistry

from app.intelligence import RuleManager
from app.models import DisturbanceRecord
from app.providers import ProviderManager, ComtradeProvider, CsvProvider, ExcelProvider
from app.visualization.axis.datetime_axis import TimeDisplayMode
from app.visualization.managers.synchronization_manager import SynchronizationManager
from app.visualization.overlays.overlay_colors import sequence_curve_label
from app.visualization.performance import timed_section
from app.ui.dialogs import confirm_destructive_action
from app.ui.import_wizard import ImportWizardWidget
from app.ui.session import SessionCanvasController, SessionPanel
from app.ui.widgets.measurement_panel import MeasurementPanel
from app.ui.widgets.event_list_panel import EventListPanel
from app.ui.widgets.quality_report_panel import QualityReportPanel
from app.analytics.quality import RecordQuality, compute_quality_fingerprint
from app.ui.widgets.fault_summary_panel import FaultSummaryPanel
from app.analytics.fault import classify_fault_from_events
from app.ui.widgets.protection_timing_panel import ProtectionTimingPanel
from app.analytics.protection import extract_protection_timing
from app.ui.widgets.correlation_report_panel import CorrelationReportPanel
from app.analytics.correlation import correlate_all_pairs
from app.ui.widgets.suggestion_bar import SuggestionBar
from app.analytics.suggestions import SuggestionContext, generate_suggestions

_FILE_FILTER = (
    "Supported Files (*.cfg *.comtrade *.csv *.xlsx);;"
    "COMTRADE (*.cfg *.comtrade);;CSV (*.csv);;Excel (*.xlsx);;All Files (*)"
)
_MANIFEST_FILTER = "Event Manifests (*.yaml *.yml);;All Files (*)"

_CSV_EXCEL_SUFFIXES = frozenset({".csv", ".xlsx", ".xls"})


def _provider_type_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".cfg", ".comtrade"):
        return "comtrade"
    if suffix in (".xlsx", ".xls"):
        return "excel"
    if suffix == ".csv":
        return "csv"
    return suffix.lstrip(".") or "unknown"


def _record_source_path(record: DisturbanceRecord) -> Path:
    try:
        source_file = record.metadata.source_file
    except AttributeError:
        source_file = ""
    return Path(str(source_file or ""))


def _is_csv_excel_record(record: DisturbanceRecord) -> bool:
    path = _record_source_path(record)
    provider_type = str(getattr(record.metadata, "provider_type", "") or "").lower()
    return provider_type in {
        "csv",
        "excel",
        "normalized_csv",
        "normalized_excel",
    } or path.suffix.lower() in _CSV_EXCEL_SUFFIXES


def _make_source_record(
    source_id: str,
    record: DisturbanceRecord,
    provider_type: str = "unknown",
):
    """Wrap a DisturbanceRecord in a SourceRecord for multi-source sessions."""
    from app.data.multi_source_session import SourceRecord
    from app.data.signal_metadata import SignalMetadata

    signal_metadata = {
        ch.name: SignalMetadata(name=ch.name, unit=ch.unit, source=source_id)
        for ch in record.analog_channels
    }
    return SourceRecord(
        source_id=source_id,
        provider_type=provider_type,
        record=record,
        signal_metadata=signal_metadata,
        original_start_time=record.timing_info.start_time,
        sampling_rates=list(record.sampling_info.sampling_rates),
    )

# Preferred display order for grouped panels (harmonic/sequence panels start hidden)
_PANEL_ORDER = [
    "voltage_raw",
    "current_raw",
    "sequence_voltage",
    "sequence_current",
    "thd_voltage",
    "thd_current",
    "harmonic_spectrum_voltage",
    "harmonic_spectrum_current",
    "power",
    "frequency",
    "rocof",
    "other",
]

def _log_direct_open_mapping(filename: str, signal_metadata: dict) -> None:
    """Diagnostic log of channel → display group for a direct CSV/Excel open."""
    groups = {name: m.display_group for name, m in signal_metadata.items()}
    print(f"[D4.4] {filename}: {groups}", file=sys.stderr)




def _log_runtime_route(message: str) -> None:
    """Write concise runtime routing evidence for direct-open reconciliation."""
    print(f"[D4.4.3 route] {message}", file=sys.stderr)


def _make_sequence_record(
    source_record: DisturbanceRecord,
    time: object,
    data_dict: dict,
    unit: str,
) -> DisturbanceRecord:
    """Build a synthetic DisturbanceRecord from sequence component magnitude arrays."""
    import numpy as np
    import pandas as pd
    from app.models import AnalogChannel

    df = pd.DataFrame({"time": np.asarray(time, dtype=np.float64)})
    channels = []
    for i, (name, arr) in enumerate(data_dict.items()):
        df[name] = np.asarray(arr, dtype=np.float64)
        channels.append(AnalogChannel(name=name, unit=unit, index=i))

    return DisturbanceRecord(
        metadata=source_record.metadata,
        waveform_data=df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=source_record.sampling_info,
        timing_info=source_record.timing_info,
        disturbance_info=source_record.disturbance_info,
    )


def _make_harmonic_record(
    source_record: DisturbanceRecord,
    time: object,
    data_dict: dict,
    unit: str,
) -> DisturbanceRecord:
    """Build a synthetic DisturbanceRecord from harmonic/THD trend arrays."""
    import numpy as np
    import pandas as pd
    from app.models import AnalogChannel

    df = pd.DataFrame({"time": np.asarray(time, dtype=np.float64)})
    channels = []
    for i, (name, arr) in enumerate(data_dict.items()):
        df[name] = np.asarray(arr, dtype=np.float64)
        channels.append(AnalogChannel(name=name, unit=unit, index=i))

    return DisturbanceRecord(
        metadata=source_record.metadata,
        waveform_data=df,
        analog_channels=channels,
        digital_channels=[],
        sampling_info=source_record.sampling_info,
        timing_info=source_record.timing_info,
        disturbance_info=source_record.disturbance_info,
    )


def _build_provider_manager() -> ProviderManager:
    """Create a ProviderManager with all standard providers registered."""
    manager = ProviderManager()
    manager.register_provider(ComtradeProvider())
    manager.register_provider(CsvProvider())
    manager.register_provider(ExcelProvider())
    return manager


def _format_load_status(record: DisturbanceRecord) -> str:
    """Format a status bar string from a loaded DisturbanceRecord."""
    n_a = len(record.analog_channels)
    n_d = len(record.digital_channels)
    rates = record.sampling_info.sampling_rates
    rate_str = f"{rates[0]:.1f} Hz" if rates else "unknown"
    name = Path(record.metadata.source_file).name
    return f"{name} | {n_a} analog, {n_d} digital | Fs = {rate_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Threading helpers
# ─────────────────────────────────────────────────────────────────────────────


class _WorkerSignals(QObject):
    """Cross-thread signals for load workers."""

    finished: pyqtSignal = pyqtSignal(object)  # emits _DirectOpenResult
    error: pyqtSignal = pyqtSignal(str)         # emits error message


@dataclasses.dataclass
class _DirectOpenResult:
    """Rich result from _IntelligentLoadWorker for any direct file open."""

    record: DisturbanceRecord
    path: Path
    provider_type: str          # "comtrade", "csv", "excel"
    signal_metadata: dict       # dict[str, SignalMetadata] — empty for COMTRADE
    ts_ambiguous: bool
    ts_matrices: dict           # {col_name: TimestampInterpretationMatrix}


class _IntelligentLoadWorker(QRunnable):
    """Loads a file and runs D4.3 intelligence classification for CSV/Excel.

    For COMTRADE files, signal_metadata is empty and ambiguity flags are False.
    For CSV/Excel, IntelligenceManager classifies each channel and detects
    whether the timestamp column interpretation is ambiguous.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        path: Path,
        intelligence_manager,   # IntelligenceManager — avoid import; duck-typed
    ) -> None:
        super().__init__()
        self._manager = provider_manager
        self._path = path
        self._intelligence = intelligence_manager
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            record = self._manager.load(self._path)
            suffix = self._path.suffix.lower()
            provider_type = _provider_type_from_path(self._path)

            if suffix in _CSV_EXCEL_SUFFIXES:
                from app.data.direct_load_intelligence import (
                    build_signal_metadata,
                    detect_timestamp_ambiguity,
                )
                signal_metadata = build_signal_metadata(
                    record, self._intelligence, self._path.stem, provider_type
                )
                ts_ambiguous, ts_matrices = detect_timestamp_ambiguity(
                    self._path, record
                )
            else:
                signal_metadata = {}
                ts_ambiguous = False
                ts_matrices = {}

            self.signals.finished.emit(_DirectOpenResult(
                record=record,
                path=self._path,
                provider_type=provider_type,
                signal_metadata=signal_metadata,
                ts_ambiguous=ts_ambiguous,
                ts_matrices=ts_matrices,
            ))
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────


class PowerwaveMainWindow(QMainWindow):
    """Powerwave viewer — session canvas workspace."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Powerwave")
        self.resize(1400, 900)

        self._provider_manager = _build_provider_manager()
        self._rule_manager = RuleManager()
        self._intelligence_manager = self._rule_manager.intelligence_manager
        self._sync_manager = SynchronizationManager()

        # RMS display state (Phase 5A)
        self._rms_display_mode: RMSDisplayMode = RMSDisplayMode.OFF
        self._rms_config: RMSConfig = RMSConfig()
        self._rms_window_actions: dict[RMSWindowMode, object] = {}
        self._current_signal_metadata: dict = {}
        self._time_display_mode: TimeDisplayMode = TimeDisplayMode.RELATIVE
        self._time_axis_actions: dict[TimeDisplayMode, object] = {}
        # Engineering scaling state (Phase 5B)
        self._scaling_mode: EngineeringScalingMode = EngineeringScalingMode.RAW
        self._scaling_registry: ScalingRegistry = ScalingRegistry()
        self._scaling_mode_actions: dict[EngineeringScalingMode, object] = {}
        # Frequency/ROCOF display state (Phase 5C)
        self._frequency_registry: FrequencyRegistry = FrequencyRegistry()
        self._frequency_display_mode_actions: dict[FrequencyDisplayMode, object] = {}
        # Phasor/sequence display state (Phase 6A)
        self._phasor_registry: PhasorRegistry = PhasorRegistry()
        self._phasor_display_mode_actions: dict[PhasorDisplayMode, object] = {}
        # Harmonic analysis state (Phase 8)
        self._harmonic_registry: HarmonicRegistry = HarmonicRegistry()
        self._harmonic_display_mode_actions: dict[HarmonicDisplayMode, object] = {}
        self._harmonic_panel_cache: object = None
        self._harmonic_panel_cache_record_id: int | None = None
        self._performance_timing_enabled = False
        self._performance_timing_sink = None

        # Session workspace (Phase 9B/9D)
        self._active_session = None                              # EventAnalysisSession | None
        self._session_panel: SessionPanel | None = None
        self._session_canvas_controller: SessionCanvasController | None = None
        self._session_canvas_active: bool = False
        self._session_navigator = None                           # WaveformNavigatorStrip | None
        self._embedded_import_wizard: ImportWizardWidget | None = None
        self._canvas_theme: str = "dark"
        self._crosshair_snap_enabled: bool = False

        # Measurement panel (Phase 1 Enhancement)
        self._measurement_dock = self._build_measurement_dock()
        self._measurement_mode: bool = False

        # Event list panel (Phase 2 Enhancement)
        self._event_dock = self._build_event_dock()

        # Data quality fingerprint (Phase 4 Enhancement)
        self._quality_fingerprint: RecordQuality | None = None
        self._quality_dock = self._build_quality_dock()

        # Fault characterisation (Phase 5 Enhancement)
        self._fault_dock = self._build_fault_dock()

        # Protection timing (Phase 6 Enhancement)
        self._protection_dock = self._build_protection_dock()

        # Cross-source correlation (Phase 7 Enhancement)
        self._correlation_dock = self._build_correlation_dock()

        # Contextual suggestions (Phase 8 Enhancement)
        self._suggestion_dock = self._build_suggestion_dock()
        self._last_suggestion_events: list = []
        self._last_suggestion_fault = None
        self._last_suggestion_protection = None
        self._last_suggestion_correlation: list = []

        self._build_layout()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._measurement_dock)
        self._measurement_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._event_dock)
        self._event_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._quality_dock)
        self._quality_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._fault_dock)
        self._fault_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._protection_dock)
        self._protection_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._correlation_dock)
        self._correlation_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self._suggestion_dock)
        self._suggestion_dock.hide()

        self._build_menu()

    # ─────────────────────────────────────────────────────────────────────────
    # Qt overrides
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._sync_manager.clear()
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        from PyQt6.QtWidgets import QLabel
        placeholder = QLabel("Open a file to begin (File → Open…)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(placeholder)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _clear_sync_before_layout_switch(self) -> None:
        try:
            self._sync_manager.clear()
        except RuntimeError:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Measurement dock (Phase 1 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_measurement_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Measurements", self)
        dock.setObjectName("MeasurementDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._measurement_widget = MeasurementPanel()
        dock.setWidget(self._measurement_widget)
        dock.setMinimumHeight(120)
        return dock

    def _on_toggle_measurement_mode(self, checked: bool) -> None:
        self._measurement_mode = checked
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.set_measurement_mode(
                checked, self._active_session
            )
        if checked:
            self._measurement_dock.show()
        else:
            self._measurement_dock.hide()
        self._measurement_widget.update_measurements(None)

    def _on_measurement_mode_changed_from_canvas(self, enabled: bool) -> None:
        """Sync the View menu action when measurement mode is toggled from a canvas right-click."""
        self._measurement_mode = enabled
        was_blocked = self._measurement_mode_action.blockSignals(True)
        self._measurement_mode_action.setChecked(enabled)
        self._measurement_mode_action.blockSignals(was_blocked)
        if enabled:
            self._measurement_dock.show()
        else:
            self._measurement_dock.hide()
            self._measurement_widget.update_measurements(None)

    def _on_session_measurement_result(self, result) -> None:
        """Forward session measurement result to the measurement panel."""
        self._measurement_widget.update_measurements(result)

    # ─────────────────────────────────────────────────────────────────────────
    # Event detection dock (Phase 2 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_event_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Detected Events", self)
        dock.setObjectName("EventListDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._event_list_widget = EventListPanel()
        self._event_list_widget.event_selected.connect(self._on_event_selected)
        dock.setWidget(self._event_list_widget)
        dock.setMinimumHeight(140)
        return dock

    def _run_event_detection(self, record: "DisturbanceRecord") -> None:
        """Classify channels, detect events, apply markers and populate event list."""
        from app.analytics.events.event_detector import classify_channel_roles, detect_events
        import numpy as np

        try:
            time = record.waveform_data["time"].to_numpy(dtype=np.float64)
            roles = classify_channel_roles(record.analog_channels)
            if not roles:
                self._event_list_widget.clear()
                return

            data_by_channel: dict[str, np.ndarray] = {
                name: record.waveform_data[name].to_numpy(dtype=np.float64)
                for name in roles
                if name in record.waveform_data.columns
            }

            trigger_s = (
                record.timing_info.trigger_time - record.timing_info.start_time
            ).total_seconds()
            nominal = getattr(record.sampling_info, "nominal_frequency", None)
            try:
                nominal = float(nominal) if nominal else 50.0
            except (TypeError, ValueError):
                nominal = 50.0

            events = detect_events(
                time, data_by_channel, roles,
                trigger_time_s=trigger_s,
                nominal_hz=nominal,
            )
        except Exception:
            return

        # Populate list panel and show the dock if events were found
        self._event_list_widget.load_events(events)
        if events:
            self._event_dock.show()

        # Cache events for Phase 8 suggestion context
        self._last_suggestion_events = list(events)

        # Fault characterisation (Phase 5) — runs on the same data already in scope
        self._run_fault_characterisation(record, events, time, data_by_channel, nominal)

        # Protection timing (Phase 6) — same data, adds digital channel extraction
        self._run_protection_timing(record, events, time, data_by_channel, nominal)

    def _on_event_selected(self, t_start: float) -> None:
        """Jump the session canvas viewport to a detected event."""
        if not (self._session_canvas_active and self._session_canvas_controller is not None):
            return
        window = 0.1
        t_lo, t_hi = t_start - window, t_start + window
        for canvas in self._session_canvas_controller.active_canvases():
            canvas.normalize_viewport(t_lo, t_hi)
            break

    # ─────────────────────────────────────────────────────────────────────────
    # Cursor readout dock (Phase 3 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_session_crosshair_moved(self, t: float, values: list) -> None:
        """Forward hover crosshair position + interpolated values to the session panel."""
        if self._session_panel is not None:
            self._session_panel.update_crosshair_readouts(t, values)

    # ─────────────────────────────────────────────────────────────────────────
    # Data quality dock (Phase 4 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_quality_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Data Quality", self)
        dock.setObjectName("QualityReportDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._quality_widget = QualityReportPanel()
        dock.setWidget(self._quality_widget)
        return dock

    def _run_quality_check(self, record) -> None:
        """Compute quality fingerprint for a loaded record and refresh the UI."""
        try:
            time = record.time
            data_by_channel: dict[str, object] = {}
            for ch in record.analog_channels:
                data_by_channel[ch.name] = ch.samples
            if not data_by_channel:
                return
            import numpy as np
            time_arr = np.asarray(time, dtype=float)
            data_arr = {k: np.asarray(v, dtype=float) for k, v in data_by_channel.items()}
            result = compute_quality_fingerprint(time_arr, data_arr)
            self._quality_fingerprint = result
            self._quality_widget.load_quality(result)
            # Auto-show dock if any issues were found
            from app.analytics.quality import QualityGrade
            if result.overall_grade != QualityGrade.OK:
                self._quality_dock.show()
        except Exception:  # noqa: BLE001
            pass
        # Phase 8: generate contextual suggestions after all analytics are done
        QTimer.singleShot(0, self._run_suggestions)

    # ─────────────────────────────────────────────────────────────────────────
    # Fault characterisation dock (Phase 5 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fault_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Fault Characterisation", self)
        dock.setObjectName("FaultCharacterisationDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._fault_summary_widget = FaultSummaryPanel()
        dock.setWidget(self._fault_summary_widget)
        return dock

    def _run_fault_characterisation(
        self,
        record,
        events: list,
        time,
        data_by_channel: dict,
        nominal_hz: float,
    ) -> None:
        """Classify fault type from detected events and show the summary dock."""
        try:
            result = classify_fault_from_events(
                events, time, data_by_channel, record.analog_channels,
                nominal_hz=nominal_hz,
            )
            self._last_suggestion_fault = result
            if result is not None:
                self._fault_summary_widget.load_fault(result)
                self._fault_dock.show()
            else:
                self._fault_summary_widget.clear_fault()
        except Exception:  # noqa: BLE001
            self._last_suggestion_fault = None
            self._fault_summary_widget.clear_fault()

    # ─────────────────────────────────────────────────────────────────────────
    # Protection timing dock (Phase 6 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_protection_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Protection Timing", self)
        dock.setObjectName("ProtectionTimingDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._protection_timing_widget = ProtectionTimingPanel()
        dock.setWidget(self._protection_timing_widget)
        dock.setMinimumHeight(150)
        return dock

    def _run_protection_timing(
        self,
        record,
        events: list,
        time,
        analog_data: dict,
        nominal_hz: float,
    ) -> None:
        """Extract protection relay timings and populate the timing dock."""
        import numpy as np
        try:
            # Build digital channel data dict
            digital_data: dict[str, np.ndarray] = {}
            for ch in record.digital_channels:
                if ch.name in record.waveform_data.columns:
                    digital_data[ch.name] = (
                        record.waveform_data[ch.name].to_numpy(dtype=float)
                    )

            result = extract_protection_timing(
                events,
                np.asarray(time, dtype=float),
                analog_data,
                digital_data,
                record.digital_channels,
                record.analog_channels,
                nominal_hz=nominal_hz,
            )
            if result is not None and len(result.events) > 1:
                self._last_suggestion_protection = result
                self._protection_timing_widget.load_timing(result)
                self._protection_dock.show()
            else:
                self._last_suggestion_protection = None
                self._protection_timing_widget.clear_timing()
        except Exception:  # noqa: BLE001
            self._last_suggestion_protection = None
            self._protection_timing_widget.clear_timing()

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-source correlation dock (Phase 7 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_correlation_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Source Correlation", self)
        dock.setObjectName("CorrelationReportDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._correlation_widget = CorrelationReportPanel()
        dock.setWidget(self._correlation_widget)
        dock.setMinimumHeight(150)
        return dock

    def _run_cross_correlation(self, sources: list) -> None:
        """Compute pairwise cross-correlations for a multi-source set.

        Args:
            sources: list of SourceRecord or SessionSource objects.
                     Each must have .source_id and .record attributes.
        """
        import numpy as np
        from app.analytics.events.event_detector import _infer_role  # noqa: PLC0415

        if len(sources) < 2:
            self._correlation_widget.clear_results()
            return

        def _best_channel(record) -> tuple[str, np.ndarray, np.ndarray] | None:
            """Return (source_id, time, data) for the best voltage or current channel."""
            wf = record.waveform_data
            if "time" not in wf.columns or wf.empty:
                return None
            time = wf["time"].to_numpy(dtype=float)

            # Try voltage first, then current, then highest-RMS analog
            for preferred_role in ("voltage", "current"):
                best_rms = -1.0
                best_name = None
                best_data = None
                for ch in record.analog_channels:
                    role = _infer_role(ch.name, getattr(ch, "unit", "") or "")
                    if role != preferred_role:
                        continue
                    if ch.name not in wf.columns:
                        continue
                    data = wf[ch.name].to_numpy(dtype=float)
                    rms = float(np.sqrt(np.nanmean(data ** 2)))
                    if rms > best_rms:
                        best_rms = rms
                        best_name = ch.name
                        best_data = data
                if best_name is not None:
                    return best_name, time, best_data

            # Fallback: any analog channel with highest RMS
            best_rms = -1.0
            best_name = None
            best_data = None
            for ch in record.analog_channels:
                if ch.name not in wf.columns:
                    continue
                data = wf[ch.name].to_numpy(dtype=float)
                rms = float(np.sqrt(np.nanmean(data ** 2)))
                if rms > best_rms:
                    best_rms = rms
                    best_name = ch.name
                    best_data = data
            return (best_name, time, best_data) if best_name else None

        try:
            sources_data: list[tuple[str, np.ndarray, np.ndarray, str]] = []
            for src in sources:
                result = _best_channel(src.record)
                if result is not None:
                    ch_name, t, d = result
                    sources_data.append((src.source_id, t, d, ch_name))

            if len(sources_data) < 2:
                self._correlation_widget.clear_results()
                return

            results = correlate_all_pairs(sources_data, max_lag_s=10.0)
            if results:
                self._last_suggestion_correlation = results
                self._correlation_widget.load_results(results)
                self._correlation_dock.show()

                # Auto-apply high-confidence offsets to EventAnalysisSession
                if self._active_session is not None:
                    self._apply_correlation_offsets(results)
            else:
                self._last_suggestion_correlation = []
                self._correlation_widget.clear_results()
            # Refresh suggestions now that correlation results are available
            QTimer.singleShot(0, self._run_suggestions)
        except Exception:  # noqa: BLE001
            self._last_suggestion_correlation = []
            self._correlation_widget.clear_results()

    def _apply_correlation_offsets(self, results: list) -> None:
        """Apply high-confidence correlation offsets to the active session."""
        if self._active_session is None:
            return
        for res in results:
            if not res.same_event_likely or res.confidence < 0.70:
                continue
            try:
                self._active_session.set_time_offset(
                    res.source_b_id,
                    res.suggested_offset_s,
                    method="correlation",
                    confidence=res.confidence,
                )
                self._active_session.set_alignment_notes(
                    res.source_b_id,
                    f"Auto-aligned via cross-correlation against '{res.source_a_id}' "
                    f"(coeff={res.correlation_coeff:.3f}, lag={res.lag_ms:+.2f} ms)",
                )
                self._refresh_session_source_row(res.source_b_id)
                if self._session_canvas_active and self._session_canvas_controller is not None:
                    self._session_canvas_controller.on_offset_changed(
                        res.source_b_id,
                        res.suggested_offset_s,
                        self._active_session,
                    )
            except Exception:  # noqa: BLE001
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Contextual suggestion bar (Phase 8 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_suggestion_dock(self):
        from PyQt6.QtWidgets import QDockWidget, QWidget
        dock = QDockWidget("Suggestions", self)
        dock.setObjectName("SuggestionBarDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.TopDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        # Hide the title bar so the dock appears as a seamless banner strip
        dock.setTitleBarWidget(QWidget())
        self._suggestion_widget = SuggestionBar()
        self._suggestion_widget.action_requested.connect(self._on_suggestion_action)
        self._suggestion_widget.all_dismissed.connect(lambda: dock.hide())
        dock.setWidget(self._suggestion_widget)
        dock.setMaximumHeight(48)
        return dock

    def _build_suggestion_action_map(self) -> dict:
        """Return mapping from action_id to callable."""
        return {
            "enable_rms": lambda: self._on_rms_mode_changed(RMSDisplayMode.OVERLAY),
            "enable_phasors": lambda: self._on_phasor_display_mode_changed(
                PhasorDisplayMode.SEQUENCE_COMPONENTS
            ),
            "enable_harmonics": lambda: self._on_harmonic_display_mode_changed(
                HarmonicDisplayMode.SPECTRUM
            ),
            "enable_frequency": lambda: None,  # frequency display not yet toggleable via mode
            "show_quality": lambda: self._quality_dock.show(),
            "show_fault": lambda: self._fault_dock.show(),
            "show_protection": lambda: self._protection_dock.show(),
            "show_correlation": lambda: self._correlation_dock.show(),
            "auto_align": self._on_suggestion_auto_align,
        }

    def _on_suggestion_action(self, action_id: str) -> None:
        action_map = self._build_suggestion_action_map()
        action = action_map.get(action_id)
        if action is not None:
            try:
                action()
            except Exception:  # noqa: BLE001
                pass

    def _on_suggestion_auto_align(self) -> None:
        """Trigger auto-align on the active session (used by suggestion bar)."""
        if self._active_session is None:
            return
        targets = list(self._active_session.sources)
        if len(targets) >= 2:
            QTimer.singleShot(0, lambda t=targets: self._run_cross_correlation(t))

    def _run_suggestions(self) -> None:
        """Build suggestion context from cached analytics results and refresh bar."""
        try:
            ctx = SuggestionContext(
                events=self._last_suggestion_events,
                quality=self._quality_fingerprint,
                fault=self._last_suggestion_fault,
                protection=self._last_suggestion_protection,
                correlation_results=self._last_suggestion_correlation,
                is_multi_source=self._active_session is not None,
                rms_active=(self._rms_display_mode != RMSDisplayMode.OFF),
                phasors_active=(
                    self._phasor_registry.display_mode != PhasorDisplayMode.OFF
                ),
                harmonics_active=(
                    self._harmonic_registry.display_mode != HarmonicDisplayMode.OFF
                ),
                quality_dock_visible=self._quality_dock.isVisible(),
                fault_dock_visible=self._fault_dock.isVisible(),
                protection_dock_visible=self._protection_dock.isVisible(),
                correlation_dock_visible=self._correlation_dock.isVisible(),
            )
            suggestions = generate_suggestions(ctx)
            if suggestions:
                self._suggestion_widget.load_suggestions(suggestions)
                self._suggestion_dock.show()
        except Exception:  # noqa: BLE001
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Menu
    # ─────────────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # S9: unified File menu — single Open entry point for all formats
        file_menu = menu_bar.addMenu("&File")
        open_action = file_menu.addAction("&Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.setToolTip(
            "Open a COMTRADE, CSV, or Excel file — CSV/Excel launches the Import Wizard "
            "automatically. Use 'Add Source' in the Session Panel to compare multiple files."
        )
        open_action.triggered.connect(self._open_unified_file)
        manifest_action = file_menu.addAction("Open &Event Manifest…")
        manifest_action.setShortcut("Ctrl+E")
        manifest_action.triggered.connect(self._open_manifest_dialog)
        self._save_manifest_action = file_menu.addAction("&Save Session as Manifest…")
        self._save_manifest_action.setShortcut("Ctrl+Shift+S")
        self._save_manifest_action.setToolTip(
            "Export the current session to a YAML manifest file that can be reopened later"
        )
        self._save_manifest_action.setEnabled(False)
        self._save_manifest_action.triggered.connect(self._on_save_session_as_manifest)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        view_menu = menu_bar.addMenu("&View")
        # Session Panel toggle (Phase 9B) — created lazily on first use
        self._session_panel_action = view_menu.addAction("Session &Panel")
        self._session_panel_action.setCheckable(True)
        self._session_panel_action.setChecked(False)
        self._session_panel_action.triggered.connect(self._on_toggle_session_panel)

        # Session Canvas (Phase 9D) — disabled until a session is active
        self._session_canvas_action = view_menu.addAction("Session &Canvas")
        self._session_canvas_action.setCheckable(True)
        self._session_canvas_action.setChecked(False)
        self._session_canvas_action.setEnabled(False)
        self._session_canvas_action.setToolTip(
            "Show the multi-source waveform canvas for the active session"
        )
        self._session_canvas_action.triggered.connect(self._on_toggle_session_canvas)

        # Show Legend (Phase 9E) — only meaningful when session canvas is active
        self._show_legend_action = view_menu.addAction("Show &Legend")
        self._show_legend_action.setCheckable(True)
        self._show_legend_action.setChecked(True)
        self._show_legend_action.setToolTip(
            "Show or hide the per-panel channel legend in the session canvas"
        )
        self._show_legend_action.triggered.connect(self._on_toggle_legend)

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Canvas Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        dark_theme = theme_menu.addAction("&Dark")
        dark_theme.setCheckable(True)
        dark_theme.setChecked(True)
        dark_theme.triggered.connect(lambda: self._on_canvas_theme_changed("dark"))
        theme_group.addAction(dark_theme)
        light_theme = theme_menu.addAction("&Light")
        light_theme.setCheckable(True)
        light_theme.triggered.connect(lambda: self._on_canvas_theme_changed("light"))
        theme_group.addAction(light_theme)

        crosshair_menu = view_menu.addMenu("&Crosshair Mode")
        crosshair_group = QActionGroup(self)
        crosshair_group.setExclusive(True)
        free_crosshair = crosshair_menu.addAction("&Free")
        free_crosshair.setCheckable(True)
        free_crosshair.setChecked(True)
        free_crosshair.triggered.connect(
            lambda: self._on_crosshair_snap_changed(False)
        )
        crosshair_group.addAction(free_crosshair)
        snap_crosshair = crosshair_menu.addAction("&Snap to Waveform")
        snap_crosshair.setCheckable(True)
        snap_crosshair.triggered.connect(
            lambda: self._on_crosshair_snap_changed(True)
        )
        crosshair_group.addAction(snap_crosshair)

        view_menu.addSeparator()
        time_axis_menu = view_menu.addMenu("&Time Axis Mode")
        time_axis_group = QActionGroup(self)
        time_axis_group.setExclusive(True)

        relative_time = time_axis_menu.addAction("&Relative Time")
        relative_time.setCheckable(True)
        relative_time.setChecked(True)
        relative_time.triggered.connect(
            lambda: self._on_time_axis_mode_changed(TimeDisplayMode.RELATIVE)
        )
        time_axis_group.addAction(relative_time)
        self._time_axis_actions[TimeDisplayMode.RELATIVE] = relative_time

        absolute_time = time_axis_menu.addAction("&Absolute Timestamp")
        absolute_time.setCheckable(True)
        absolute_time.triggered.connect(
            lambda: self._on_time_axis_mode_changed(TimeDisplayMode.ABSOLUTE)
        )
        time_axis_group.addAction(absolute_time)
        self._time_axis_actions[TimeDisplayMode.ABSOLUTE] = absolute_time

        view_menu.addSeparator()
        self._measurement_mode_action = view_menu.addAction("&Measurement Mode")
        self._measurement_mode_action.setCheckable(True)
        self._measurement_mode_action.setChecked(False)
        self._measurement_mode_action.setShortcut("Ctrl+M")
        self._measurement_mode_action.setToolTip(
            "Enable two-cursor measurement: drag cursor A (yellow) and cursor B (cyan) "
            "to measure Δt, ΔY, RMS, frequency and energy between them"
        )
        self._measurement_mode_action.toggled.connect(self._on_toggle_measurement_mode)
        view_menu.addAction(self._measurement_dock.toggleViewAction())
        view_menu.addAction(self._event_dock.toggleViewAction())
        view_menu.addAction(self._quality_dock.toggleViewAction())
        view_menu.addAction(self._fault_dock.toggleViewAction())
        view_menu.addAction(self._protection_dock.toggleViewAction())
        view_menu.addAction(self._correlation_dock.toggleViewAction())
        view_menu.addAction(self._suggestion_dock.toggleViewAction())

        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addSeparator()
        rms_menu = tools_menu.addMenu("&RMS Display")
        rms_group = QActionGroup(self)
        rms_group.setExclusive(True)

        rms_off = rms_menu.addAction("&Off")
        rms_off.setCheckable(True)
        rms_off.setChecked(True)
        rms_off.triggered.connect(lambda: self._on_rms_mode_changed(RMSDisplayMode.OFF))
        rms_group.addAction(rms_off)

        rms_overlay = rms_menu.addAction("&Overlay")
        rms_overlay.setCheckable(True)
        rms_overlay.triggered.connect(
            lambda: self._on_rms_mode_changed(RMSDisplayMode.OVERLAY)
        )
        rms_group.addAction(rms_overlay)

        rms_only = rms_menu.addAction("RMS &Only")
        rms_only.setCheckable(True)
        rms_only.triggered.connect(
            lambda: self._on_rms_mode_changed(RMSDisplayMode.RMS_ONLY)
        )
        rms_group.addAction(rms_only)

        rms_window_menu = tools_menu.addMenu("RMS &Window")
        rms_window_group = QActionGroup(self)
        rms_window_group.setExclusive(True)

        half_cycle = rms_window_menu.addAction("&Half Cycle")
        half_cycle.setCheckable(True)
        half_cycle.triggered.connect(
            lambda: self._on_rms_window_mode_changed(RMSWindowMode.HALF_CYCLE)
        )
        rms_window_group.addAction(half_cycle)
        self._rms_window_actions[RMSWindowMode.HALF_CYCLE] = half_cycle

        one_cycle = rms_window_menu.addAction("&One Cycle")
        one_cycle.setCheckable(True)
        one_cycle.setChecked(True)
        one_cycle.triggered.connect(
            lambda: self._on_rms_window_mode_changed(RMSWindowMode.ONE_CYCLE)
        )
        rms_window_group.addAction(one_cycle)
        self._rms_window_actions[RMSWindowMode.ONE_CYCLE] = one_cycle

        two_cycle = rms_window_menu.addAction("&Two Cycle")
        two_cycle.setCheckable(True)
        two_cycle.triggered.connect(
            lambda: self._on_rms_window_mode_changed(RMSWindowMode.TWO_CYCLE)
        )
        rms_window_group.addAction(two_cycle)
        self._rms_window_actions[RMSWindowMode.TWO_CYCLE] = two_cycle

        custom_window = rms_window_menu.addAction("&Custom Samples...")
        custom_window.setCheckable(True)
        custom_window.triggered.connect(
            lambda: self._on_rms_window_mode_changed(RMSWindowMode.CUSTOM_SAMPLES)
        )
        rms_window_group.addAction(custom_window)
        self._rms_window_actions[RMSWindowMode.CUSTOM_SAMPLES] = custom_window

        tools_menu.addSeparator()
        scaling_mode_menu = tools_menu.addMenu("&Engineering Scaling")
        scaling_group = QActionGroup(self)
        scaling_group.setExclusive(True)

        _scaling_items = [
            ("&Raw (no scaling)", EngineeringScalingMode.RAW),
            ("&Primary (×PT/CT)", EngineeringScalingMode.PRIMARY),
            ("&Secondary (÷PT/CT)", EngineeringScalingMode.SECONDARY),
            ("Per-&Unit (pu)", EngineeringScalingMode.PER_UNIT),
        ]
        for label, smode in _scaling_items:
            action = scaling_mode_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(smode == EngineeringScalingMode.RAW)
            action.triggered.connect(
                lambda _checked, m=smode: self._on_scaling_mode_changed(m)
            )
            scaling_group.addAction(action)
            self._scaling_mode_actions[smode] = action

        tools_menu.addAction("Scaling &Configuration…").triggered.connect(
            self._on_scaling_config
        )

        tools_menu.addSeparator()
        freq_disp_menu = tools_menu.addMenu("&Frequency Display")
        freq_disp_group = QActionGroup(self)
        freq_disp_group.setExclusive(True)

        _freq_disp_items = [
            ("&Panel Only (default)", FrequencyDisplayMode.PANEL_ONLY),
            ("&Overlay",              FrequencyDisplayMode.OVERLAY),
            ("O&ff",                  FrequencyDisplayMode.OFF),
        ]
        for label, fmode in _freq_disp_items:
            action = freq_disp_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(fmode == FrequencyDisplayMode.PANEL_ONLY)
            action.triggered.connect(
                lambda _checked, m=fmode: self._on_frequency_display_mode_changed(m)
            )
            freq_disp_group.addAction(action)
            self._frequency_display_mode_actions[fmode] = action

        tools_menu.addSeparator()
        phasor_disp_menu = tools_menu.addMenu("&Phasor Display")
        phasor_disp_group = QActionGroup(self)
        phasor_disp_group.setExclusive(True)

        _phasor_disp_items = [
            ("O&ff (default)",           PhasorDisplayMode.OFF),
            ("&Magnitude",               PhasorDisplayMode.MAGNITUDE),
            ("&Angle",                   PhasorDisplayMode.ANGLE),
            ("&Sequence Components",     PhasorDisplayMode.SEQUENCE_COMPONENTS),
        ]
        for label, pmode in _phasor_disp_items:
            action = phasor_disp_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(pmode == PhasorDisplayMode.OFF)
            action.triggered.connect(
                lambda _checked, m=pmode: self._on_phasor_display_mode_changed(m)
            )
            phasor_disp_group.addAction(action)
            self._phasor_display_mode_actions[pmode] = action

        tools_menu.addSeparator()
        harmonic_disp_menu = tools_menu.addMenu("&Harmonic Analysis")
        harmonic_disp_group = QActionGroup(self)
        harmonic_disp_group.setExclusive(True)

        _harmonic_disp_items = [
            ("O&ff (default)",          HarmonicDisplayMode.OFF),
            ("&Magnitude Overlay",      HarmonicDisplayMode.HARMONIC_MAGNITUDE),
            ("&THD Trend",              HarmonicDisplayMode.THD),
            ("&Spectrum Panels",        HarmonicDisplayMode.SPECTRUM),
        ]
        for label, hmode in _harmonic_disp_items:
            action = harmonic_disp_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(hmode == HarmonicDisplayMode.OFF)
            action.triggered.connect(
                lambda _checked, m=hmode: self._on_harmonic_display_mode_changed(m)
            )
            harmonic_disp_group.addAction(action)
            self._harmonic_display_mode_actions[hmode] = action

        tools_menu.addSeparator()
        tools_menu.addAction("&Calculated Signals…").triggered.connect(
            self._on_calculated_signals
        )

    # ─────────────────────────────────────────────────────────────────────────
    # File loading (standard path)
    # ─────────────────────────────────────────────────────────────────────────

    def _confirm_replace_active_session(self) -> bool:
        """Sprint 1D: gate any workflow that is about to discard the active
        session (File > Open, "New Session") behind a confirmation, but
        only when that session actually has something to lose. Returns
        True immediately (no prompt) for an empty/absent session.
        """
        if self._active_session is None or not self._active_session.has_meaningful_work():
            return True
        return confirm_destructive_action(
            self,
            title="Start a new session?",
            message=(
                "The current session contains loaded sources or calculated work.\n\n"
                "Continuing will discard the current in-memory session.\n\n"
                "This action cannot be undone."
            ),
        )

    def _open_unified_file(self) -> None:
        """S9 unified entry point: open any file directly into a fresh session canvas.

        Starts a fresh EventAnalysisSession then delegates to _on_add_to_session()
        which handles the file dialog, Import Wizard auto-fire for CSV/Excel, and
        session canvas activation. The user never needs to know about separate code paths.
        """
        if not self._confirm_replace_active_session():
            return
        self._on_new_session()
        self._on_add_to_session()


    def _on_load_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Load Error", message)

    def _on_multi_source_loaded(self, session) -> None:
        """Route a multi-source load (manifest) through the EventAnalysisSession canvas."""
        from app.sessions.event_session import EventAnalysisSession

        event_session = EventAnalysisSession()
        for src in session.sources:
            origin = str(getattr(src, "origin_path", None) or "")
            event_session.add_source(
                src.record,
                src.source_id,
                getattr(src, "provider_type", "unknown"),
                origin or None,
            )
        event_session.default_layout()

        self._active_session = event_session
        self._time_display_mode = TimeDisplayMode.ABSOLUTE
        self._set_time_axis_action_checked(TimeDisplayMode.ABSOLUTE)
        self._current_signal_metadata = {}
        self._session_canvas_action.setEnabled(True)
        self._save_manifest_action.setEnabled(True)
        panel = self._ensure_session_panel()
        panel.refresh_all(event_session)
        self._activate_session_canvas()

        ids = ", ".join(s.source_id for s in session.sources)
        self.setWindowTitle(f"Powerwave — Multi-Source: {ids}")
        self.statusBar().showMessage(
            f"Session: {session.source_count()} source(s) loaded."
        )
        QTimer.singleShot(
            0, lambda s=event_session: self._run_cross_correlation(s.list_sources())
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Manifest loading (Phase D4)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_manifest_dialog(self) -> None:
        """Open a YAML event manifest via file dialog."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Event Manifest", "samples/manifests", _MANIFEST_FILTER
        )
        if path_str:
            self._load_manifest(Path(path_str))

    # ─────────────────────────────────────────────────────────────────────────
    # RMS display mode (Phase 5A)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_time_axis_action_checked(self, mode: TimeDisplayMode) -> None:
        action = self._time_axis_actions.get(mode)
        if action is not None:
            action.setChecked(True)

    def _on_time_axis_mode_changed(self, mode: TimeDisplayMode) -> None:
        """Switch visible panels between relative and absolute timestamp labels."""
        self._time_display_mode = mode
        self._apply_time_axis_mode_to_visible()
        self._set_time_axis_action_checked(mode)
        label = (
            "Time axis: Absolute Timestamp"
            if mode == TimeDisplayMode.ABSOLUTE
            else "Time axis: Relative Time"
        )
        self.statusBar().showMessage(label)

    def _apply_time_axis_mode_to_visible(self) -> None:
        """Apply current time-axis display policy without changing X data."""
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            self._session_canvas_controller.set_time_axis_mode(
                self._time_display_mode, self._active_session
            )

    def _on_rms_mode_changed(self, mode: RMSDisplayMode) -> None:
        """Apply a new global RMS display mode to all currently active canvases."""
        self._rms_display_mode = mode
        self._apply_rms_mode_to_all_canvases()
        label = {
            RMSDisplayMode.OFF: "RMS: Off",
            RMSDisplayMode.OVERLAY: "RMS: Overlay",
            RMSDisplayMode.RMS_ONLY: "RMS: Only",
        }.get(mode, "RMS: Unknown")
        self.statusBar().showMessage(label)

    def _on_rms_window_mode_changed(self, mode: RMSWindowMode) -> None:
        """Apply a new engineering RMS measurement window globally."""
        custom_samples = self._rms_config.custom_window_samples
        if mode == RMSWindowMode.CUSTOM_SAMPLES:
            value, ok = QInputDialog.getInt(
                self,
                "RMS Window",
                "Custom RMS window samples:",
                int(custom_samples or 100),
                1,
                1_000_000,
                1,
            )
            if not ok:
                action = self._rms_window_actions.get(self._rms_config.window_mode)
                if action is not None:
                    action.setChecked(True)
                return
            custom_samples = int(value)

        self._rms_config = dataclasses.replace(
            self._rms_config,
            window_mode=mode,
            custom_window_samples=custom_samples if mode == RMSWindowMode.CUSTOM_SAMPLES else None,
            cycles_per_window=2 if mode == RMSWindowMode.TWO_CYCLE else 1,
        )
        action = self._rms_window_actions.get(mode)
        if action is not None:
            action.setChecked(True)
        self._rebuild_rms_overlays_for_config_change()
        label = {
            RMSWindowMode.HALF_CYCLE: "RMS window: Half Cycle",
            RMSWindowMode.ONE_CYCLE: "RMS window: One Cycle",
            RMSWindowMode.TWO_CYCLE: "RMS window: Two Cycle",
            RMSWindowMode.CUSTOM_SAMPLES: f"RMS window: {custom_samples} samples",
        }[mode]
        self.statusBar().showMessage(label)

    def _rebuild_rms_overlays_for_config_change(self) -> None:
        """Clear and rebuild per-canvas RMS curves for the selected window."""
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            self._session_canvas_controller.set_rms_mode(
                self._rms_display_mode, self._active_session, config=self._rms_config
            )

    def _apply_rms_mode_to_all_canvases(self) -> None:
        """Push the current RMS mode to the session canvas controller."""
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            self._session_canvas_controller.set_rms_mode(
                self._rms_display_mode, self._active_session, config=self._rms_config
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Engineering scaling (Phase 5B)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_scaling_mode_changed(self, mode: EngineeringScalingMode) -> None:
        """Apply a new global engineering scaling mode to all active canvases."""
        self._scaling_mode = mode
        self._apply_scaling_to_all_canvases()
        action = self._scaling_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)
        labels = {
            EngineeringScalingMode.RAW: "Scaling: Raw",
            EngineeringScalingMode.PRIMARY: "Scaling: Primary",
            EngineeringScalingMode.SECONDARY: "Scaling: Secondary",
            EngineeringScalingMode.PER_UNIT: "Scaling: Per-Unit",
        }
        self.statusBar().showMessage(labels.get(mode, "Scaling changed"))

    def _apply_scaling_to_all_canvases(self) -> None:
        """Push the current scaling mode and registry to the session canvas controller."""
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            self._session_canvas_controller.set_scaling_mode(
                self._scaling_mode, self._active_session,
                registry=self._scaling_registry,
            )

    def _on_scaling_config(self) -> None:
        """Open the Scaling Configuration dialog and apply any changes."""
        from PyQt6.QtWidgets import QDialog
        from app.ui.dialogs.scaling_config_dialog import ScalingConfigDialog

        dlg = ScalingConfigDialog(self, initial_config=self._scaling_registry.global_config)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_config: GlobalScalingConfig = dlg.get_config()
        self._scaling_registry.set_global_config(new_config)
        if self._scaling_mode != EngineeringScalingMode.RAW:
            self._apply_scaling_to_all_canvases()
        self.statusBar().showMessage("Scaling configuration updated.")

    def _on_calculated_signals(self) -> None:
        """Open the Calculated Signals creation/preview dialog for the
        active session. On successful creation, the new signal is synced
        onto the session canvas (Phase 3B) -- the dialog itself never
        renders anything; it only creates and resolves the definition in
        the session (Phase 3A), and this handler wires that completion
        back to the canvas/session panel.
        """
        from PyQt6.QtWidgets import QDialog, QMessageBox
        from app.ui.calculated_signals import CalculatedSignalDialog

        if self._active_session is None or not self._active_session.list_analog_channels(active_only=True):
            QMessageBox.information(
                self,
                "Calculated Signals",
                "Open a session with at least one active analog channel "
                "before creating a calculated signal.",
            )
            return

        dlg = CalculatedSignalDialog(self._active_session, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Calculated signal created.")
            self._sync_calculated_signals_to_canvas()

    def _on_frequency_display_mode_changed(self, mode: FrequencyDisplayMode) -> None:
        """Apply a new frequency/ROCOF panel visibility mode."""
        self._frequency_registry.set_display_mode(mode)
        action = self._frequency_display_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)  # type: ignore[union-attr]
        self._apply_frequency_display_mode()
        labels = {
            FrequencyDisplayMode.PANEL_ONLY: "Frequency: Panel Only",
            FrequencyDisplayMode.OVERLAY:    "Frequency: Overlay",
            FrequencyDisplayMode.OFF:        "Frequency: Off",
        }
        self.statusBar().showMessage(labels.get(mode, "Frequency display changed"))

    def _apply_frequency_display_mode(self) -> None:
        pass  # frequency channels are native session channels — no synthetic panels needed

    def _on_phasor_display_mode_changed(self, mode: PhasorDisplayMode) -> None:
        """Apply a new phasor/sequence component display mode."""
        self._phasor_registry.set_display_mode(mode)
        action = self._phasor_display_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)  # type: ignore[union-attr]
        with timed_section(
            "phasor_mode_switch",
            enabled=self._performance_timing_enabled,
            sink=self._performance_timing_sink,
        ):
            self._apply_phasor_display_mode()
        labels = {
            PhasorDisplayMode.OFF:                 "Phasor: Off",
            PhasorDisplayMode.MAGNITUDE:           "Phasor: Magnitude",
            PhasorDisplayMode.ANGLE:               "Phasor: Angle",
            PhasorDisplayMode.SEQUENCE_COMPONENTS: "Phasor: Sequence Components",
        }
        self.statusBar().showMessage(labels.get(mode, "Phasor display changed"))

    def _apply_phasor_display_mode(self) -> None:
        """Apply phasor display mode: overlays on waveform canvases + sequence panel visibility.

        MAGNITUDE / ANGLE  — calls set_phasor_display_mode() on all non-sequence
                             canvases; phasor overlays appear/disappear on the
                             voltage_raw and current_raw panes.
        SEQUENCE_COMPONENTS — shows sequence_voltage / sequence_current panels and
                              calls set_phasor_display_mode(SEQUENCE_COMPONENTS) on
                              waveform canvases to clear any magnitude/angle overlays.
        OFF                — calls set_phasor_display_mode(OFF) on all waveform
                             canvases and hides sequence panels.
        """
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            mode = self._phasor_registry.display_mode
            seq_visible = (mode == PhasorDisplayMode.SEQUENCE_COMPONENTS)
            self._toggle_synthetic_panels("synthetic:sequence", seq_visible)
            self._session_canvas_controller.set_phasor_mode(
                mode, self._active_session, config=self._phasor_registry.config,
            )
            return


    # ─────────────────────────────────────────────────────────────────────────
    # Harmonic display mode (Phase 8)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_harmonic_display_mode_changed(self, mode: HarmonicDisplayMode) -> None:
        """Apply a new harmonic display mode to all currently active canvases."""
        self._harmonic_registry.set_display_mode(mode)
        action = self._harmonic_display_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)  # type: ignore[union-attr]
        with timed_section(
            "harmonic_mode_switch",
            enabled=self._performance_timing_enabled,
            sink=self._performance_timing_sink,
        ):
            self._apply_harmonic_display_mode()
        labels = {
            HarmonicDisplayMode.OFF:                "Harmonic: Off",
            HarmonicDisplayMode.HARMONIC_MAGNITUDE: "Harmonic: Magnitude Overlay",
            HarmonicDisplayMode.THD:                "Harmonic: THD Trend",
            HarmonicDisplayMode.SPECTRUM:           "Harmonic: Spectrum Panels",
        }
        self.statusBar().showMessage(labels.get(mode, "Harmonic display changed"))

    def _apply_harmonic_display_mode(self) -> None:
        """Apply harmonic display mode: overlays on waveform canvases + panel visibility.

        HARMONIC_MAGNITUDE — calls set_harmonic_display_mode() on all non-harmonic
                             canvases; per-order magnitude envelopes appear on the
                             voltage_raw and current_raw panes.
        THD               — hides magnitude overlays; shows thd_voltage / thd_current
                             panels.
        SPECTRUM          — hides magnitude overlays; shows harmonic_spectrum panels.
        OFF               — removes magnitude overlays; hides all harmonic panels.
        """
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            mode = self._harmonic_registry.display_mode
            thd_visible = (mode == HarmonicDisplayMode.THD)
            spec_visible = (mode == HarmonicDisplayMode.SPECTRUM)
            self._toggle_synthetic_panels("synthetic:thd", thd_visible)
            self._toggle_synthetic_panels("synthetic:harmonic_spectrum", spec_visible)
            self._session_canvas_controller.set_harmonic_mode(
                mode, self._active_session, config=self._harmonic_registry.config,
            )
            return


    # ─────────────────────────────────────────────────────────────────────────
    # Session workspace (Phase 9B)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_session_panel(self) -> SessionPanel:
        """Create the session panel on first use and dock it on the left."""
        if self._session_panel is None:
            self._session_panel = SessionPanel(self)
            self.addDockWidget(
                Qt.DockWidgetArea.LeftDockWidgetArea, self._session_panel
            )
            self._wire_session_panel(self._session_panel)
            self._session_panel.visibilityChanged.connect(
                self._session_panel_action.setChecked
            )
        return self._session_panel

    def _wire_session_panel(self, panel: SessionPanel) -> None:
        panel.source_add_requested.connect(self._on_session_add_source_requested)
        panel.source_remove_requested.connect(self._on_session_remove_source)
        panel.offset_changed.connect(self._on_session_offset_changed)
        panel.offset_edit_finished.connect(self._on_session_offset_edit_finished)
        panel.offset_reset_requested.connect(self._on_session_offset_reset)
        panel.auto_align_requested.connect(self._on_session_auto_align)
        panel.channel_visibility_changed.connect(
            self._on_session_channel_visibility
        )
        panel.channel_colour_change_requested.connect(
            self._on_session_channel_colour
        )
        panel.channel_panel_changed.connect(self._on_session_channel_panel)
        panel.new_panel_requested.connect(self._on_session_new_panel_requested)
        panel.session_cleared.connect(self._on_session_cleared)
        panel.source_active_changed.connect(self._on_session_source_active)
        panel.set_as_reference_requested.connect(self._on_session_set_as_reference)
        panel.calculated_signal_visibility_changed.connect(
            self._on_calc_signal_visibility_changed
        )
        panel.calculated_signal_recalculate_requested.connect(
            self._on_calc_signal_recalculate
        )
        panel.calculated_signal_recalculate_all_requested.connect(
            self._on_calc_signal_recalculate_all
        )
        panel.calculated_signal_delete_requested.connect(
            self._on_calc_signal_delete
        )

    def _on_toggle_session_panel(self, checked: bool) -> None:
        panel = self._ensure_session_panel()
        panel.setVisible(checked)

    def _on_toggle_session_canvas(self, checked: bool) -> None:
        if checked:
            if self._active_session is None:
                self._session_canvas_action.setChecked(False)
                return
            self._activate_session_canvas()
        else:
            self._deactivate_session_canvas()

    def _activate_session_canvas(self) -> None:
        """Build/rebuild the session canvas and make it the central widget."""
        from PyQt6.QtWidgets import QVBoxLayout, QWidget
        from app.visualization.widgets.waveform_navigator import WaveformNavigatorStrip
        if self._active_session is None:
            return
        if self._session_canvas_controller is None:
            self._session_canvas_controller = SessionCanvasController()

        self._clear_sync_before_layout_switch()
        scroll_area = self._session_canvas_controller.rebuild_layout(self._active_session)

        # Compound central widget: navigator strip (top) + scrollable panels (bottom)
        self._session_navigator = WaveformNavigatorStrip()
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._session_navigator)
        central_layout.addWidget(scroll_area, stretch=1)
        self.setCentralWidget(central)
        self._session_canvas_active = True
        self._session_canvas_action.setChecked(True)

        self._session_canvas_controller.register_with_sync(self._sync_manager)
        # Wire navigator viewport ↔ session canvas X range
        self._session_canvas_controller.set_navigator(self._session_navigator)

        # S6: wire measurement results to the measurement panel dock + delta readout bar
        self._session_canvas_controller.set_measurement_callback(
            self._on_session_measurement_result
        )
        self._session_canvas_controller.set_measurement_mode_changed_callback(
            self._on_measurement_mode_changed_from_canvas
        )
        # Hover crosshair → live cursor readout bar (show+update on mouse move)
        self._session_canvas_controller.set_crosshair_readout_callback(
            self._on_session_crosshair_moved
        )
        # Drag-drop channel moves → same rebuild path as "Move to panel" legend action
        self._session_canvas_controller.set_channel_panel_changed_callback(
            self._on_session_channel_panel
        )
        self._session_canvas_controller.set_time_axis_mode(
            self._time_display_mode, self._active_session
        )
        self._session_canvas_controller.set_canvas_theme(self._canvas_theme)
        self._session_canvas_controller.set_crosshair_snap_enabled(
            self._crosshair_snap_enabled
        )
        self._session_canvas_controller.refresh_all(self._active_session)
        self._sync_session_panel_colours()
        self._refresh_timing_assessment()
        # Normalize all panels to the same X range after auto-range settles
        _ctrl = self._session_canvas_controller
        _sess = self._active_session
        QTimer.singleShot(
            0, lambda c=_ctrl, s=_sess: c.normalize_all_to_session_window(s)
        )

        n = len(self._active_session.list_sources())
        self.statusBar().showMessage(f"Session canvas: {n} source(s) loaded.")

    def _deactivate_session_canvas(self) -> None:
        """Exit session canvas mode and show placeholder."""
        from PyQt6.QtWidgets import QLabel
        self._clear_sync_before_layout_switch()
        self._session_canvas_active = False
        self._session_canvas_action.setChecked(False)
        placeholder = QLabel("Open a file to begin (File → Open…)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(placeholder)

    def _on_open_session(self) -> None:
        """Start a fresh session workspace. User adds sources via the Session Panel."""
        if not self._confirm_replace_active_session():
            return
        self._on_new_session()

    def _on_new_session(self) -> None:
        """Start a fresh EventAnalysisSession and show the session panel."""
        from app.sessions import EventAnalysisSession

        self._active_session = EventAnalysisSession()
        panel = self._ensure_session_panel()
        panel.refresh_all(self._active_session)
        panel.show()
        self._session_panel_action.setChecked(True)
        self._session_canvas_action.setEnabled(True)
        if self._session_canvas_active:
            self._deactivate_session_canvas()
        self.statusBar().showMessage("New session started.")

    def _on_add_to_session(self) -> None:
        """Add one source to the active session.

        Shows a single file dialog for all supported types.
        COMTRADE files are loaded directly; CSV/Excel opens the Import Wizard
        with the chosen path pre-filled so the user doesn't pick the file twice.
        """
        if self._active_session is None:
            self._on_new_session()
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Add Source to Session", "", _FILE_FILTER
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() in _CSV_EXCEL_SUFFIXES:
            self._show_embedded_import_wizard(path_str)
        else:
            self.statusBar().showMessage(f"Loading {path.name}…")
            try:
                record = self._provider_manager.load(path)
                self._on_session_import_record_ready(record)
            except Exception as exc:  # noqa: BLE001
                self._on_load_error(str(exc))

    def _show_embedded_import_wizard(self, path_str: str) -> None:
        """Host the CSV/Excel Import Wizard in the main workspace.

        If a wizard is already embedded (e.g. the user triggers File > Open or
        Session Panel > Add Source again before finishing the current one),
        confirm discard through the wizard's own risk check instead of silently
        replacing it — the wizard is non-modal, so this is reachable in a way
        the old exec()-based dialog never allowed.
        """
        if self._embedded_import_wizard is not None:
            self._embedded_import_wizard.request_close()
            if self._embedded_import_wizard is not None:
                return  # user declined to discard the in-progress wizard

        self._clear_sync_before_layout_switch()
        self._session_canvas_active = False
        self._session_canvas_action.setChecked(False)

        wizard = ImportWizardWidget(self)
        self._embedded_import_wizard = wizard
        wizard.set_source_path(path_str)
        wizard.import_completed.connect(self._on_session_import_record_ready)
        wizard.finished.connect(self._clear_embedded_import_wizard_reference)
        wizard.close_requested.connect(self._on_embedded_import_wizard_closed)
        self.setCentralWidget(wizard)
        self.statusBar().showMessage(f"Import wizard: {Path(path_str).name}")
        QTimer.singleShot(0, wizard.profile_selected_file)

    def _clear_embedded_import_wizard_reference(self) -> None:
        self._embedded_import_wizard = None

    def _on_embedded_import_wizard_closed(self) -> None:
        self._embedded_import_wizard = None
        if self._active_session is not None and self._active_session.list_sources():
            self._activate_session_canvas()
        else:
            self._deactivate_session_canvas()

    # -------------------------------------------------------------------------
    # Single-record → session loader (replaces FlexiblePlotCanvas paths)
    # -------------------------------------------------------------------------

    def _load_record_into_session(
        self,
        record: DisturbanceRecord,
        signal_metadata: dict | None = None,
        *,
        time_mode: TimeDisplayMode = TimeDisplayMode.RELATIVE,
    ) -> None:
        """Create a fresh EventAnalysisSession from one record and activate the session canvas."""
        from app.sessions.event_session import EventAnalysisSession

        session = EventAnalysisSession()
        source_path = _record_source_path(record)
        display_name = source_path.stem or record.metadata.station_name or "source"
        provider_type = str(getattr(record.metadata, "provider_type", "") or "unknown")
        origin_path = str(source_path) if source_path.name else None

        session.add_source(record, display_name, provider_type, origin_path)
        session.default_layout()

        # Inject synthetic computed panels (hidden until mode toggled on)
        self._add_sequence_panels_to_session(session, record, signal_metadata)
        self._add_harmonic_panels_to_session(session, record, signal_metadata)

        self._active_session = session
        self._time_display_mode = time_mode
        self._set_time_axis_action_checked(time_mode)
        self._current_signal_metadata = signal_metadata or {}

        self._session_canvas_action.setEnabled(True)
        self._save_manifest_action.setEnabled(True)
        panel = self._ensure_session_panel()
        panel.refresh_all(session)
        self._activate_session_canvas()

    def _add_sequence_panels_to_session(
        self,
        session,
        record: DisturbanceRecord,
        signal_metadata: dict | None,
    ) -> None:
        """Compute sequence components and inject as hidden synthetic panels in the session."""
        import numpy as np
        from app.analytics.phasors.phasor_extraction import extract_phasor
        from app.analytics.phasors.symmetrical_components import (
            compute_sequence_from_phasor_arrays,
        )

        if record is None:
            return
        channel_names = [ch.name for ch in record.analog_channels]
        channel_phases = {ch.name: ch.phase for ch in record.analog_channels if ch.phase}
        groups = self._phasor_registry.detect_three_phase_groups(
            channel_names, signal_metadata or None, channel_phases or None,
        )
        if not groups:
            return
        try:
            time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            return
        rates = [r for r in record.sampling_info.sampling_rates if r > 0]
        if rates:
            sample_rate_hz = float(rates[0])
        elif len(time_col) >= 2:
            diffs = np.diff(time_col[: min(100, len(time_col))])
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            sample_rate_hz = 1.0 / float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            sample_rate_hz = 0.0
        if sample_rate_hz <= 0:
            return

        config = self._phasor_registry.config
        seq_voltage_data: dict[str, np.ndarray] = {}
        seq_current_data: dict[str, np.ndarray] = {}
        seq_time: np.ndarray | None = None

        for group in groups:
            if not group.complete:
                continue
            try:
                ch_a = record.waveform_data[group.phase_a].to_numpy(dtype=np.float64)
                ch_b = record.waveform_data[group.phase_b].to_numpy(dtype=np.float64)
                ch_c = record.waveform_data[group.phase_c].to_numpy(dtype=np.float64)
                pa = extract_phasor(time_col, ch_a, sample_rate_hz, config)
                pb = extract_phasor(time_col, ch_b, sample_rate_hz, config)
                pc = extract_phasor(time_col, ch_c, sample_rate_hz, config)
                seq = compute_sequence_from_phasor_arrays(pa, pb, pc)
            except Exception:  # noqa: BLE001
                continue
            seq_time = seq["time"]
            if group.signal_type == "voltage":
                seq_voltage_data[sequence_curve_label("V", "positive")] = seq["mag_v1"]
                seq_voltage_data[sequence_curve_label("V", "negative")] = seq["mag_v2"]
                seq_voltage_data[sequence_curve_label("V", "zero")] = seq["mag_v0"]
            else:
                seq_current_data[sequence_curve_label("I", "positive")] = seq["mag_v1"]
                seq_current_data[sequence_curve_label("I", "negative")] = seq["mag_v2"]
                seq_current_data[sequence_curve_label("I", "zero")] = seq["mag_v0"]

        if seq_time is None:
            return
        for data_dict, unit, title, ptype in [
            (seq_voltage_data, "V",  "Sequence Voltage",  "synthetic:sequence_voltage"),
            (seq_current_data, "A",  "Sequence Current",  "synthetic:sequence_current"),
        ]:
            if not data_dict:
                continue
            syn_record = _make_sequence_record(record, seq_time, data_dict, unit)
            source_id = session.add_source(syn_record, title, ptype)
            panel_id = session.add_panel(title)
            session.set_panel_visible(panel_id, False)
            for ch_name in data_dict:
                session.set_channel_panel(source_id, ch_name, panel_id)

    def _add_harmonic_panels_to_session(
        self,
        session,
        record: DisturbanceRecord,
        signal_metadata: dict | None,
    ) -> None:
        """Compute harmonic trends and inject as hidden synthetic panels in the session."""
        import numpy as np
        from app.analytics.harmonics.harmonic_cache import HarmonicCache
        from app.analytics.harmonics.harmonic_extraction import (
            compute_harmonic_window_samples,
            extract_harmonics,
        )
        from app.analytics.harmonics.harmonic_metrics import compute_thd_array
        from app.analytics.harmonics.harmonic_models import HarmonicChannelRole
        from app.analytics.harmonics.harmonic_overlay import classify_harmonic_role

        if record is None:
            return
        config = self._harmonic_registry.config
        try:
            time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            return
        rates = [r for r in record.sampling_info.sampling_rates if r > 0]
        if rates:
            sample_rate_hz = float(rates[0])
        elif len(time_col) >= 2:
            diffs = np.diff(time_col[: min(100, len(time_col))])
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            sample_rate_hz = 1.0 / float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            sample_rate_hz = 0.0
        if sample_rate_hz <= 0:
            return

        window = compute_harmonic_window_samples(sample_rate_hz, config)
        overlap_clamped = max(0.0, min(config.overlap, 0.999))
        hop = max(1, int(round(window * (1.0 - overlap_clamped))))
        if (
            self._harmonic_panel_cache is None
            or self._harmonic_panel_cache_record_id != id(record)
        ):
            self._harmonic_panel_cache = HarmonicCache()
            self._harmonic_panel_cache_record_id = id(record)
        cache = self._harmonic_panel_cache
        thd_voltage: dict[str, np.ndarray] = {}
        thd_current: dict[str, np.ndarray] = {}
        spec_voltage: dict[str, np.ndarray] = {}
        spec_current: dict[str, np.ndarray] = {}
        harmonic_time: np.ndarray | None = None
        _SPECTRUM_ORDERS = [3, 5, 7, 11, 13]

        for ch in record.analog_channels:
            meta = (signal_metadata or {}).get(ch.name)
            role = classify_harmonic_role(ch.name, ch.unit, meta).role
            if role == HarmonicChannelRole.UNKNOWN:
                continue
            try:
                raw_data = record.waveform_data[ch.name].to_numpy(dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            cached = cache.get(ch.name, window, hop, config.nominal_hz, config.max_order)
            h_result = cached
            if h_result is None:
                try:
                    h_result = extract_harmonics(raw_data, sample_rate_hz, config, time=time_col)
                except Exception:  # noqa: BLE001
                    continue
                cache.put(ch.name, window, hop, config.nominal_hz, config.max_order, h_result)
            if h_result.n_windows == 0:
                continue
            harmonic_time = h_result.harmonic_time
            thd_arr = compute_thd_array(h_result.magnitudes) * 100.0
            if role == HarmonicChannelRole.VOLTAGE_HARMONIC:
                thd_voltage[ch.name] = thd_arr
                if not spec_voltage:
                    for order in _SPECTRUM_ORDERS:
                        mag = h_result.get_magnitude(order)
                        if mag is not None:
                            spec_voltage[f"H{order}"] = mag
            else:
                thd_current[ch.name] = thd_arr
                if not spec_current:
                    for order in _SPECTRUM_ORDERS:
                        mag = h_result.get_magnitude(order)
                        if mag is not None:
                            spec_current[f"H{order}"] = mag

        if harmonic_time is None:
            return
        for data_dict, unit, title, ptype in [
            (thd_voltage,  "%",     "THD — Voltage (%)",           "synthetic:thd_voltage"),
            (thd_current,  "%",     "THD — Current (%)",           "synthetic:thd_current"),
            (spec_voltage, "V RMS", "Harmonic Spectrum — Voltage", "synthetic:harmonic_spectrum_voltage"),
            (spec_current, "A RMS", "Harmonic Spectrum — Current", "synthetic:harmonic_spectrum_current"),
        ]:
            if not data_dict:
                continue
            syn_record = _make_harmonic_record(record, harmonic_time, data_dict, unit)
            source_id = session.add_source(syn_record, title, ptype)
            panel_id = session.add_panel(title)
            session.set_panel_visible(panel_id, False)
            for ch_name in data_dict:
                session.set_channel_panel(source_id, ch_name, panel_id)

    def _toggle_synthetic_panels(self, provider_prefix: str, visible: bool) -> None:
        """Show/hide session panels whose source provider_type starts with provider_prefix."""
        if self._active_session is None or self._session_canvas_controller is None:
            return
        matching_ids = {
            s.source_id
            for s in self._active_session.list_sources()
            if s.provider_type.startswith(provider_prefix)
        }
        if not matching_ids:
            return
        changed = False
        for panel in self._active_session.list_panels():
            if any(ref[0] in matching_ids for ref in panel.channel_refs):
                if panel.is_visible != visible:
                    self._active_session.set_panel_visible(panel.panel_id, visible)
                    changed = True
        if not changed:
            return
        scroll = self._session_canvas_controller.rebuild_layout(self._active_session)
        central = self.centralWidget()
        if central is not None:
            layout = central.layout()
            if layout is not None and layout.count() >= 2:
                old_item = layout.takeAt(1)
                if old_item and old_item.widget():
                    old_item.widget().setParent(None)
                layout.addWidget(scroll, stretch=1)
        self._session_canvas_controller.refresh_all(self._active_session)
        self._session_canvas_controller.register_with_sync(self._sync_manager)

    def _on_session_import_record_ready(self, record: object) -> None:
        from app.models import DisturbanceRecord

        if not isinstance(record, DisturbanceRecord):
            return
        if self._active_session is None:
            return
        source_path = _record_source_path(record)
        display_name = source_path.stem or "source"
        provider_type = str(
            getattr(record.metadata, "provider_type", "") or "unknown"
        )
        origin_path = str(source_path) if source_path.name else None
        self._active_session.add_source(
            record, display_name, provider_type, origin_path
        )
        self._active_session.default_layout()
        self._session_canvas_action.setEnabled(True)
        self._save_manifest_action.setEnabled(True)
        panel = self._ensure_session_panel()
        panel.refresh_all(self._active_session)
        self._activate_session_canvas()
        # default_layout() rebuilds real-channel panels from scratch and can
        # transiently drop a panel that only existed for a calculated
        # signal (e.g. a Power panel created solely by a calc result, with
        # no real Power channel) -- resync so any existing calculated
        # signals get their panel and curve back immediately.
        self._sync_calculated_signals_to_canvas()
        n = len(self._active_session.list_sources())
        self.statusBar().showMessage(
            f"Session: {n} source(s) loaded — {display_name} added."
        )
        self._notify_if_ambiguous_timestamp(display_name, record)

    def _notify_if_ambiguous_timestamp(self, display_name: str, record) -> None:
        """Sprint 1E: a source loaded directly through a provider (not the
        Import Wizard, which shows its own ambiguity banner mid-workflow --
        see TimestampSelectPage) may have had its date column resolved
        using Powerwave's DD/MM/YYYY ambiguous-date default. Surface that
        once, right after load, since a direct load has no other timestamp
        review step. Never fires for Wizard-produced records: the Import
        Wizard's own RecordingMetadata construction does not set this
        field, so this check is a no-op for that path by construction.
        """
        sample = getattr(record.metadata, "timestamp_ambiguity_sample", None)
        if not sample:
            return
        from app.data.timestamp_disambiguation import format_ambiguous_date_example

        example = format_ambiguous_date_example(sample)
        example_line = f"\n\nExample:\n{example}" if example else ""
        QMessageBox.information(
            self,
            "Ambiguous date format detected",
            (
                f'"{display_name}" was interpreted using DD/MM/YYYY by default.'
                f"{example_line}\n\n"
                "If this source uses a different date order, reopen the file "
                "through the Import Wizard's Timestamp Settings or Advanced "
                "Timestamp Repair to review or override the interpretation."
            ),
        )

    def _on_session_add_source_requested(self) -> None:
        """Handle 'Add Source' button inside the session panel."""
        self._on_add_to_session()

    def _on_session_remove_source(self, source_id: str) -> None:
        if self._active_session is None:
            return
        source = self._active_session.get_source(source_id)
        if source is None:
            return
        dependents = self._active_session.get_calculated_dependents_for_source(source_id)
        if dependents:
            message = (
                f'"{source.display_name}" will be removed from the current session.\n\n'
                "Any calculated signals that depend on this source will become "
                "stale or unavailable.\n\n"
                "This action cannot be undone."
            )
        else:
            message = (
                f'"{source.display_name}" will be removed from the current session.\n\n'
                "This action cannot be undone."
            )
        if not confirm_destructive_action(
            self, title="Remove source?", message=message
        ):
            return
        self._active_session.remove_source(source_id)
        if self._session_panel is not None:
            self._session_panel.remove_source_row(source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_source_removed(source_id)
        self._refresh_timing_assessment()
        n = len(self._active_session.list_sources())
        self.statusBar().showMessage(f"Source removed. Session: {n} source(s).")

    def _on_session_offset_changed(self, source_id: str, offset_s: float) -> None:
        if self._active_session is None:
            return
        self._active_session.set_time_offset(source_id, offset_s, method="manual")
        self._active_session.set_alignment_notes(source_id, "")
        self._refresh_session_source_row(source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_offset_changed(
                source_id, offset_s, self._active_session
            )
        # Calculated signals depending on source_id are marked STALE by
        # set_time_offset() itself (Phase 2C-2) -- repaint their (now-stale)
        # curves so the "(stale)" label appears immediately. This is display
        # only: no recalculation happens on every tick, see
        # _on_session_offset_edit_finished for the committed-edit trigger.
        self._sync_calculated_signals_to_canvas(
            list(self._active_session.get_calculated_dependents_for_source(source_id))
        )
        self._refresh_timing_assessment()

    def _on_session_offset_edit_finished(self, source_id: str) -> None:
        """Offset spinbox editing committed (focus lost / Enter) -- unlike
        _on_session_offset_changed (every valueChanged tick), this fires
        once per edit and is the appropriate place to recalculate
        dependent calculated signals.
        """
        if self._active_session is None:
            return
        self._recalculate_calculated_signals_for_source(source_id)

    def _on_session_offset_reset(self, source_id: str) -> None:
        if self._active_session is None:
            return
        source = self._active_session.get_source(source_id)
        if source is None:
            return
        old_offset = source.time_offset_s
        old_method = source.alignment_method
        self._active_session.set_time_offset(source_id, 0.0, method="none")
        self._active_session.set_alignment_notes(source_id, "")
        self._refresh_session_source_row(source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_offset_changed(
                source_id, 0.0, self._active_session
            )
        self.statusBar().showMessage(f"Offset reset to 0.000 s for {source_id}.")
        # Reset is a single committed action (button click), not a
        # continuous edit -- recalculate immediately, same as
        # editingFinished, but only when Reset actually changed anything
        # (Sprint 1C no-change guard): clicking Reset while already at
        # 0.000s/method="none" must not trigger a no-op recalculation.
        if old_offset != 0.0 or old_method != "none":
            self._recalculate_calculated_signals_for_source(source_id)
        self._refresh_timing_assessment()

    def _on_session_set_as_reference(self, source_id: str) -> None:
        """Set one source as the time reference: its offset becomes 0.0 and all
        other sources are shifted to maintain their relative positions."""
        if self._active_session is None:
            return
        ref_source = self._active_session.get_source(source_id)
        if ref_source is None:
            return
        ref_offset = ref_source.time_offset_s
        changed_source_ids: list[str] = []
        for source in self._active_session.list_sources():
            old_offset = source.time_offset_s
            old_method = source.alignment_method
            if source.source_id == source_id:
                new_offset = 0.0
            else:
                new_offset = source.time_offset_s - ref_offset
            self._active_session.set_time_offset(
                source.source_id, new_offset, method="manual"
            )
            self._active_session.set_alignment_notes(source.source_id, "")
            self._refresh_session_source_row(source.source_id)
            if old_offset != new_offset or old_method != "manual":
                changed_source_ids.append(source.source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.refresh_all(self._active_session)
        self.statusBar().showMessage(
            f"Set '{ref_source.display_name}' as reference (t=0)."
        )
        # Set-as-Reference can move several sources' offsets in one action
        # (Sprint 1C no-op + dedup guard): only sources whose offset/method
        # actually changed are recalculated, and a calculated signal
        # depending on more than one of them is resolved once, not once
        # per matching source.
        self._recalculate_calculated_signals_for_sources(changed_source_ids)
        self._refresh_timing_assessment()

    def _on_session_auto_align(self, source_id: str) -> None:
        if self._active_session is None:
            return
        from app.sessions.alignment_engine import suggest_alignment_offsets

        all_sources = self._active_session.list_sources()
        if source_id == "all":
            targets = [s for s in all_sources if s.is_active]
        else:
            targets = [s for s in all_sources if s.source_id == source_id]

        if not targets:
            self.statusBar().showMessage("Auto-align: no active sources to align.")
            return

        results = suggest_alignment_offsets(targets)
        changed_source_ids: list[str] = []
        for result in results:
            source = self._active_session.get_source(result.source_id)
            old_offset = source.time_offset_s if source is not None else None
            old_method = source.alignment_method if source is not None else None
            self._active_session.set_time_offset(
                result.source_id,
                result.suggested_offset_s,
                method=result.alignment_method,
                confidence=result.alignment_confidence,
            )
            self._active_session.set_alignment_notes(result.source_id, result.notes)
            self._refresh_session_source_row(result.source_id)
            if self._session_canvas_active and self._session_canvas_controller is not None:
                self._session_canvas_controller.on_offset_changed(
                    result.source_id, result.suggested_offset_s, self._active_session
                )
            if old_offset != result.suggested_offset_s or old_method != result.alignment_method:
                changed_source_ids.append(result.source_id)

        n = len(results)
        self.statusBar().showMessage(f"Auto-aligned {n} source(s).")

        # Auto-align is a single committed action, not a continuous edit --
        # recalculate dependent calculated signals for every source whose
        # offset/method actually changed (Sprint 1C no-op guard); "Align
        # All" can touch several sources at once, so a calculated signal
        # depending on more than one of them is resolved once (dedup),
        # not once per matching source.
        self._recalculate_calculated_signals_for_sources(changed_source_ids)
        self._refresh_timing_assessment()

        # Cross-correlation pass (Phase 7) — runs after trigger-based alignment
        if source_id == "all" and len(targets) >= 2:
            QTimer.singleShot(0, lambda t=targets: self._run_cross_correlation(t))

    def _refresh_timing_assessment(self) -> None:
        """Recompute the session-wide timing-reference compatibility banner
        (Sprint 1B). Read-only -- never mutates the session. Called from
        every session-refresh path that can change which sources are
        active or how they're aligned: source add/remove, activation
        change, offset change, and alignment-method change (Reset,
        Auto-align, Set-as-reference all change the alignment method).
        """
        if self._active_session is None or self._session_panel is None:
            return
        self._session_panel.refresh_timing_assessment(self._active_session)

    def _refresh_session_source_row(self, source_id: str) -> None:
        """Pull current state from the session and refresh the corresponding panel row."""
        if self._active_session is None or self._session_panel is None:
            return
        sources = {s.source_id: s for s in self._active_session.list_sources()}
        source = sources.get(source_id)
        if source is None:
            return
        try:
            metrics = self._active_session.get_source_quality_metrics(source_id)
        except Exception:  # noqa: BLE001
            metrics = None
        try:
            notes = self._active_session.get_alignment_notes(source_id)
        except AttributeError:
            notes = ""
        panels = self._active_session.list_panels()
        self._session_panel.refresh_source_row(
            source_id, source, metrics, panels, alignment_notes=notes
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Calculated Signals — canvas/session-panel integration (Phase 3B)
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_calculated_signals_to_canvas(self, calc_ids: list[str] | None = None) -> None:
        """Refresh calculated-signal curves on canvas and in the Session
        Panel's Calculated Signals section, after create/recalculate/
        visibility/delete.

        calc_ids=None syncs every calculated signal; passing a specific
        list scopes the canvas repaint to just those signals (e.g. after
        resolve_for_source() recalculates only the ones depending on one
        source) -- unrelated calculated signals and every original source
        curve are left untouched.
        """
        if self._active_session is None:
            return
        if self._session_panel is not None:
            self._session_panel.refresh_calculated_signals(self._active_session)
        if not self._session_canvas_active or self._session_canvas_controller is None:
            return

        target_ids = (
            calc_ids if calc_ids is not None
            else [e.definition.calc_id for e in self._active_session.list_calculated_signals()]
        )
        needs_new_panel = False
        for calc_id in target_ids:
            placement = self._active_session.ensure_calculated_signal_panel(calc_id)
            if placement is not None and placement[1]:
                needs_new_panel = True
        if needs_new_panel:
            # Rare path: a calculated signal's inferred panel (e.g. its
            # first-ever Power result) does not exist yet -- a full
            # rebuild is required to create the SessionCanvasWidget for it.
            # Ordinary recalculation/visibility/delete never takes this path.
            self._activate_session_canvas()

        self._session_canvas_controller.refresh_calculated_signals(
            self._active_session, calc_ids
        )

    def _recalculate_calculated_signals_for_source(self, source_id: str) -> None:
        """Recalculate only the calculated signals depending on source_id,
        after a COMMITTED alignment edit (offset editingFinished, a fine
        nudge, Reset, Auto-align, Set-as-reference, or reactivation) --
        never on every spinbox tick. A failed recalculation leaves the
        previous OK/STALE result completely untouched
        (CalculatedSignalResolutionService's own contract); the affected
        curves are repainted from current session state either way, so a
        failed recalculation's last-known-good/stale curve remains visible
        rather than disappearing.
        """
        self._recalculate_calculated_signals_for_sources([source_id])

    def _recalculate_calculated_signals_for_sources(self, source_ids: list[str]) -> None:
        """Recalculate the calculated signals depending on ANY of
        *source_ids*, each exactly once -- even when a signal depends on
        more than one of the changed sources (e.g. Set-as-Reference or
        "Align All", which can move several sources' offsets in a single
        committed action). Callers are expected to have already filtered
        *source_ids* down to sources whose offset/method actually changed
        (Sprint 1C no-op guard) -- an empty list is a cheap no-op here.
        """
        if self._active_session is None or not source_ids:
            return
        from app.calculated_signals.resolver import CalculatedSignalResolutionService

        service = CalculatedSignalResolutionService(self._active_session)
        batch = service.resolve_for_sources(source_ids)
        calc_ids = [r.calc_id for r in batch.successful] + [f.calc_id for f in batch.failures]
        if not calc_ids:
            return
        self._sync_calculated_signals_to_canvas(calc_ids)

    def _on_calc_signal_visibility_changed(self, calc_id: str, visible: bool) -> None:
        if self._active_session is None:
            return
        self._active_session.set_calculated_signal_visible(calc_id, visible)
        self._sync_calculated_signals_to_canvas([calc_id])

    def _on_calc_signal_recalculate(self, calc_id: str) -> None:
        if self._active_session is None:
            return
        from app.calculated_signals.resolver import (
            CalculatedSignalResolutionError,
            CalculatedSignalResolutionService,
        )

        service = CalculatedSignalResolutionService(self._active_session)
        try:
            service.resolve_one(calc_id)
        except CalculatedSignalResolutionError as exc:
            self.statusBar().showMessage(f"Recalculation failed: {exc}")
        else:
            self.statusBar().showMessage("Calculated signal recalculated.")
        self._sync_calculated_signals_to_canvas([calc_id])

    def _on_calc_signal_recalculate_all(self) -> None:
        if self._active_session is None:
            return
        from app.calculated_signals.resolver import CalculatedSignalResolutionService

        service = CalculatedSignalResolutionService(self._active_session)
        batch = service.resolve_all()
        self._sync_calculated_signals_to_canvas()
        self.statusBar().showMessage(
            f"Recalculated {len(batch.successful)} calculated signal(s); "
            f"{len(batch.failures)} failed."
        )

    def _on_calc_signal_delete(self, calc_id: str) -> None:
        if self._active_session is None:
            return
        definition = self._active_session.get_calculated_signal_definition(calc_id)
        if definition is None:
            return
        message = (
            f'"{definition.name}" and its current result will be removed.\n\n'
            "The original source channels will not be affected.\n\n"
            "This action cannot be undone."
        )
        if not confirm_destructive_action(
            self, title="Delete calculated signal?", message=message
        ):
            return
        if self._session_canvas_controller is not None:
            self._session_canvas_controller.remove_calculated_signal_curve(calc_id)
        self._active_session.remove_calculated_signal(calc_id)
        if self._session_panel is not None:
            self._session_panel.refresh_calculated_signals(self._active_session)
        self.statusBar().showMessage("Calculated signal deleted.")

    def _on_session_channel_visibility(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        if self._active_session is None:
            return
        self._active_session.set_channel_visibility(source_id, channel_name, visible)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_channel_visibility_changed(
                source_id, channel_name, visible, self._active_session
            )

    def _sync_session_panel_colours(self) -> None:
        """Push controller-computed waveform colours to the session panel swatches.

        Called after refresh_all() so swatches reflect the actual curve colours
        rather than the raw (often None) color_hex values in the session model.
        """
        if self._active_session is None or self._session_canvas_controller is None:
            return
        panel = self._ensure_session_panel()
        for ch in self._active_session.list_analog_channels(active_only=False):
            colour = self._session_canvas_controller.get_channel_colour(
                ch.source_id, ch.channel_name
            )
            panel.update_channel_colour(ch.source_id, ch.channel_name, colour)

    def _on_session_channel_colour(
        self, source_id: str, channel_name: str
    ) -> None:
        if self._active_session is None:
            return
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QColorDialog, QStyleFactory, QWidget
        ch = self._active_session.get_channel(source_id, channel_name)
        initial = QColor(ch.color_hex if (ch and ch.color_hex) else "#aaaaaa")

        dlg = QColorDialog(initial, self)
        dlg.setWindowTitle("Choose curve colour")
        # DontUseNativeDialog forces Qt to render all parts itself so our
        # palette/style can override Windows dark mode on every child widget.
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)

        fusion = QStyleFactory.create("Fusion")
        light = QPalette()
        light.setColor(QPalette.ColorRole.Window,          QColor("#f0f0f0"))
        light.setColor(QPalette.ColorRole.WindowText,      QColor("#1a1a1a"))
        light.setColor(QPalette.ColorRole.Base,            QColor("#ffffff"))
        light.setColor(QPalette.ColorRole.AlternateBase,   QColor("#e8e8e8"))
        light.setColor(QPalette.ColorRole.Text,            QColor("#1a1a1a"))
        light.setColor(QPalette.ColorRole.Button,          QColor("#e0e0e0"))
        light.setColor(QPalette.ColorRole.ButtonText,      QColor("#1a1a1a"))
        light.setColor(QPalette.ColorRole.Highlight,       QColor("#0078d4"))
        light.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

        dlg.setStyle(fusion)
        dlg.setPalette(light)
        for child in dlg.findChildren(QWidget):
            child.setStyle(fusion)
            child.setPalette(light)

        if dlg.exec() != QColorDialog.DialogCode.Accepted:
            return
        chosen = dlg.selectedColor()
        if not chosen.isValid():
            return
        new_hex = chosen.name()
        self._active_session.set_channel_colour(source_id, channel_name, new_hex)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_colour_changed(source_id, channel_name, new_hex)
        self._ensure_session_panel().update_channel_colour(source_id, channel_name, new_hex)

    def _on_toggle_legend(self, checked: bool) -> None:
        if self._session_canvas_controller is not None:
            self._session_canvas_controller.set_legend_visible(checked)

    def _on_canvas_theme_changed(self, theme: str) -> None:
        self._canvas_theme = "light" if str(theme).lower() == "light" else "dark"
        if self._session_canvas_controller is not None:
            self._session_canvas_controller.set_canvas_theme(self._canvas_theme)
        label = "Light" if self._canvas_theme == "light" else "Dark"
        self.statusBar().showMessage(f"Canvas theme: {label}.")

    def _on_crosshair_snap_changed(self, enabled: bool) -> None:
        self._crosshair_snap_enabled = bool(enabled)
        if self._session_canvas_controller is not None:
            self._session_canvas_controller.set_crosshair_snap_enabled(
                self._crosshair_snap_enabled
            )
        label = "snap to waveform" if self._crosshair_snap_enabled else "free movement"
        self.statusBar().showMessage(f"Crosshair mode: {label}.")

    def _on_session_channel_panel(
        self, source_id: str, channel_name: str, panel_id: str
    ) -> None:
        if self._active_session is None:
            return
        self._active_session.set_channel_panel(source_id, channel_name, panel_id)
        self._rebuild_session_canvas_after_panel_change()

    def _on_session_new_panel_requested(
        self, source_id: str, channel_name: str
    ) -> None:
        """User chose '＋ New panel…' in the channel tree — prompt for a name and create it."""
        if self._active_session is None:
            return
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "New Panel", "Panel name:", text="New Panel"
        )
        if not ok or not name.strip():
            return
        panel_id = self._active_session.add_panel(name.strip())
        self._active_session.set_channel_panel(source_id, channel_name, panel_id)
        # Update session panel combos with the new panel list, then select the new panel
        sp = self._ensure_session_panel()
        sp.refresh_all_panel_choices(self._active_session.list_panels())
        sp.update_channel_panel(source_id, channel_name, panel_id)
        self._rebuild_session_canvas_after_panel_change()

    def _rebuild_session_canvas_after_panel_change(self) -> None:
        """Rebuild and refresh the session canvas after any panel-assignment change."""
        if not self._session_canvas_active or self._active_session is None:
            return
        self._activate_session_canvas()

    def _on_session_cleared(self) -> None:
        from app.sessions import EventAnalysisSession

        self._active_session = EventAnalysisSession()
        if self._session_canvas_active:
            self._deactivate_session_canvas()
        self._session_canvas_action.setEnabled(False)
        self._save_manifest_action.setEnabled(False)
        self._refresh_timing_assessment()
        self.statusBar().showMessage("Session cleared.")

    def _on_save_session_as_manifest(self) -> None:
        """Export the active session to a YAML manifest file."""
        if self._active_session is None or not self._active_session.list_sources():
            return
        from datetime import datetime as _dt
        from PyQt6.QtWidgets import QInputDialog, QFileDialog
        from app.data.manifest_generator import generate_manifest

        event_id, ok = QInputDialog.getText(
            self,
            "Save Session as Manifest",
            "Event ID (used as the manifest identifier):",
            text="event_" + _dt.now().strftime("%Y%m%d_%H%M%S"),
        )
        if not ok or not event_id.strip():
            return

        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Event Manifest",
            f"{event_id.strip()}.yaml",
            "Event Manifests (*.yaml *.yml);;All Files (*)",
        )
        if not path_str:
            return

        try:
            generate_manifest(
                self._active_session,
                event_id.strip(),
                Path(path_str),
            )
            self.statusBar().showMessage(f"Manifest saved: {Path(path_str).name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _on_session_source_active(self, source_id: str, is_active: bool) -> None:
        if self._active_session is None:
            return
        self._active_session.set_source_active(source_id, is_active)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.refresh_all(self._active_session)
        if is_active:
            # Reactivation is a sensible existing boundary to attempt
            # recalculation (Phase 3B step 17); deactivation leaves
            # dependents stale without an automatic (doomed) retry.
            self._recalculate_calculated_signals_for_source(source_id)
        else:
            self._sync_calculated_signals_to_canvas(
                list(self._active_session.get_calculated_dependents_for_source(source_id))
            )
        self._refresh_timing_assessment()

    def _load_manifest(self, manifest_path: Path) -> None:
        """Load a YAML manifest, show the data review dialog, then visualize."""
        from PyQt6.QtWidgets import QDialog
        from app.data.manifest_loader import build_session_from_manifest, load_manifest
        from app.data.review_summary import build_event_review_summary
        from app.ui.dialogs.data_review_dialog import DataReviewDialog

        self.statusBar().showMessage(f"Loading manifest: {manifest_path.name} …")
        try:
            manifest_data = load_manifest(manifest_path)
            session = build_session_from_manifest(manifest_path)
        except (FileNotFoundError, ValueError) as exc:
            self._on_load_error(str(exc))
            return

        # Build review summary and show the review dialog before visualization
        summary = build_event_review_summary(session, manifest_data=manifest_data)
        dlg = DataReviewDialog(summary, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Manifest load cancelled.")
            return

        for src_id, confirmed_rows in dlg.confirmed_column_rows.items():
            self._rule_manager.save_confirmed_rows(confirmed_rows, source_id=src_id)

        self._on_multi_source_loaded(session)
