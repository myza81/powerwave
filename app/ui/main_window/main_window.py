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
from app.visualization.axis_management import AxisDisplayMode
from app.visualization.managers.visualization_manager import VisualizationManager
from app.visualization.overlays.overlay_colors import sequence_curve_label
from app.visualization.performance import timed_section
from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline
from app.ui.import_wizard import ImportWizardDialog
from app.ui.session import SessionCanvasController, SessionPanel
from app.ui.widgets import SignalBrowserDock, SignalBrowserEntry
from app.ui.widgets.measurement_panel import MeasurementPanel
from app.ui.widgets.event_list_panel import EventListPanel
from app.ui.widgets.cursor_readout_bar import CursorReadoutBar
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
_SAMPLE_MANIFEST = Path("samples") / "manifests" / "pulu_20260306.yaml"

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

_HARMONIC_PANEL_KEYS: frozenset[str] = frozenset({
    "thd_voltage",
    "thd_current",
    "harmonic_spectrum_voltage",
    "harmonic_spectrum_current",
})


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
    """Powerwave viewer — standard and grouped multi-panel display modes.

    Standard layout (Phase 4A):
      QSplitter(Vertical): FlexiblePlotCanvas (stretch=3) + DigitalEventTimeline (stretch=1)
      Activated by File → Open or on application start.

    Grouped layout (Phase D2):
      QSplitter(Vertical): one FlexiblePlotCanvas per non-empty display group.
      Activated by Tools → Load Synthetic Mixed Disturbance.
      X-axes linked via QTimer.singleShot(0) after layout is shown.

    VisualizationManager is held as an instance attribute to prevent GC of
    cursor_moved signal connections (pyqtSignal weak-ref contract).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Powerwave")
        self.resize(1400, 900)

        self._provider_manager = _build_provider_manager()
        self._rule_manager = RuleManager()
        self._intelligence_manager = self._rule_manager.intelligence_manager
        self._canvas = FlexiblePlotCanvas()
        self._timeline = DigitalEventTimeline()
        self._vis_manager = VisualizationManager(self._canvas, self._timeline)
        self._x_axis_linked: bool = False
        self._panel_canvases: dict = {}
        self._grouped_timeline = None

        # RMS display state (Phase 5A)
        self._rms_display_mode: RMSDisplayMode = RMSDisplayMode.OFF
        self._rms_config: RMSConfig = RMSConfig()
        self._rms_window_actions: dict[RMSWindowMode, object] = {}
        self._current_signal_metadata: dict = {}
        self._time_display_mode: TimeDisplayMode = TimeDisplayMode.RELATIVE
        self._time_axis_actions: dict[TimeDisplayMode, object] = {}
        self._axis_display_mode: AxisDisplayMode = AxisDisplayMode.SHARED
        self._axis_mode_actions: dict[AxisDisplayMode, object] = {}
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
        self._signal_browser = SignalBrowserDock(self)
        self._signal_entry_targets: dict[str, tuple[str, str, str]] = {}

        # Session workspace (Phase 9B/9D)
        self._active_session = None                              # EventAnalysisSession | None
        self._session_panel: SessionPanel | None = None
        self._session_canvas_controller: SessionCanvasController | None = None
        self._session_canvas_active: bool = False

        # Measurement panel (Phase 1 Enhancement)
        self._measurement_dock = self._build_measurement_dock()
        self._measurement_mode: bool = False

        # Event list panel (Phase 2 Enhancement)
        self._event_dock = self._build_event_dock()

        # Cursor readout bar (Phase 3 Enhancement)
        self._cursor_readout_dock = self._build_cursor_readout_dock()

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
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._signal_browser)
        self._signal_browser.visibility_changed.connect(self._on_signal_visibility_changed)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._measurement_dock)
        self._measurement_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._event_dock)
        self._event_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._cursor_readout_dock)
        self._cursor_readout_dock.hide()
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
        self._canvas.measurement_cursors_moved.connect(self._on_measurement_cursors_moved)
        self._canvas.cursor_values_changed.connect(self._on_cursor_values_changed)
        self._build_menu()

    # ─────────────────────────────────────────────────────────────────────────
    # Qt overrides
    # ─────────────────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._x_axis_linked:
            self._vis_manager.link_x_axis()
            self._x_axis_linked = True

    def closeEvent(self, event) -> None:
        self._vis_manager.synchronization_manager.clear()
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self._restore_standard_layout()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _qt_widget_alive(widget) -> bool:
        """Return False when a Python Qt wrapper points at a deleted C++ object."""
        if widget is None:
            return False
        try:
            return not sip.isdeleted(widget)
        except RuntimeError:
            return False

    def _detach_if_alive(self, widget) -> None:
        if self._qt_widget_alive(widget):
            widget.setParent(None)

    def _clear_sync_before_layout_switch(self) -> None:
        try:
            self._vis_manager.synchronization_manager.clear()
        except RuntimeError:
            pass

    def _ensure_standard_widgets_alive(self) -> None:
        """Recreate standard widgets if Qt deleted their underlying objects."""
        canvas_alive = self._qt_widget_alive(self._canvas)
        timeline_alive = self._qt_widget_alive(self._timeline)
        if canvas_alive and timeline_alive:
            return

        self._clear_sync_before_layout_switch()
        if not canvas_alive:
            self._canvas = FlexiblePlotCanvas()
        if not timeline_alive:
            self._timeline = DigitalEventTimeline()
        self._vis_manager = VisualizationManager(self._canvas, self._timeline)
        self._x_axis_linked = False

    def _link_standard_x_axis(self) -> None:
        if not (
            self._qt_widget_alive(self._canvas)
            and self._qt_widget_alive(self._timeline)
        ):
            return
        right_axes = self._canvas.right_axis_count()
        self._canvas.reserve_grouped_axis_columns(right_axes)
        self._timeline.reserve_grouped_axis_columns(right_axes)
        self._timeline.match_viewbox_geometry(self._canvas._primary_plot.getViewBox())
        self._vis_manager.link_x_axis()
        t_start, t_end = self._canvas._primary_plot.getViewBox().viewRange()[0]
        self._timeline.normalize_viewport(t_start, t_end)
        self._x_axis_linked = True
        QTimer.singleShot(0, self._match_standard_timeline_geometry)

    def _match_standard_timeline_geometry(self) -> None:
        if not (
            self._qt_widget_alive(self._canvas)
            and self._qt_widget_alive(self._timeline)
        ):
            return
        self._timeline.match_viewbox_geometry(self._canvas._primary_plot.getViewBox())

    def _restore_standard_layout(self) -> None:
        """Restore the two-pane standard layout (canvas + timeline)."""
        self._clear_sync_before_layout_switch()
        self._ensure_standard_widgets_alive()
        self._timeline.show()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._timeline)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self._panel_canvases = {}
        self._grouped_timeline = None
        self._quality_fingerprint = None
        self._fault_summary_widget.clear_fault()
        self._protection_timing_widget.clear_timing()
        self._correlation_widget.clear_results()
        self._suggestion_widget.clear_suggestions()
        self._last_suggestion_events = []
        self._last_suggestion_fault = None
        self._last_suggestion_protection = None
        self._last_suggestion_correlation = []
        self._refresh_signal_browser()
        if self.isVisible() and not self._x_axis_linked:
            QTimer.singleShot(0, self._link_standard_x_axis)

    def _rebuild_grouped_layout(
        self,
        panel_canvases: dict,
        record: DisturbanceRecord,
    ) -> None:
        """Replace central widget with grouped stacked display.

        Panels are inserted in _PANEL_ORDER; any unrecognised groups follow.
        Synchronization registration is deferred via QTimer.singleShot(0) so
        PyQtGraph scenes are fully initialised before signal wiring is attached.

        Each panel canvas is given a minimum height (180 px) so the QSplitter
        cannot collapse any panel to zero.  Initial splitter sizes are set
        explicitly so all panels share space equally on first show — without
        this the splitter may give the top panel all available space.
        """
        self._clear_sync_before_layout_switch()
        self._ensure_standard_widgets_alive()
        self._detach_if_alive(self._canvas)
        if not record.digital_channels:
            self._detach_if_alive(self._timeline)

        splitter = QSplitter(Qt.Orientation.Vertical)

        ordered_keys = [k for k in _PANEL_ORDER if k in panel_canvases]
        known = set(ordered_keys)
        for k in panel_canvases:
            if k not in known:
                ordered_keys.append(k)

        self._panel_canvases = {}
        for i, key in enumerate(ordered_keys):
            canvas = panel_canvases[key]
            canvas.setMinimumHeight(180)
            splitter.addWidget(canvas)
            splitter.setStretchFactor(i, 1)
            self._panel_canvases[key] = canvas

        self._grouped_timeline = None
        if record.digital_channels:
            self._timeline.show()
            self._timeline.setMinimumHeight(80)
            splitter.addWidget(self._timeline)
            splitter.setStretchFactor(len(ordered_keys), 1)
            self._grouped_timeline = self._timeline
        else:
            self._timeline.clear()
            self._timeline.hide()

        # Seed equal initial sizes so the splitter distributes space evenly
        # before any window resize.  Each panel gets 300 px; Qt will scale
        # proportionally to the real window height on the first paint event.
        n_panels = len(ordered_keys) + (1 if record.digital_channels else 0)
        splitter.setSizes([300] * n_panels)

        self.setCentralWidget(splitter)
        self._refresh_signal_browser()
        QTimer.singleShot(0, self._link_panel_x_axes)

    def _link_panel_x_axes(self) -> None:
        """Register grouped panels for synchronized X interaction.

        Called via QTimer.singleShot(0) from _rebuild_grouped_layout to ensure
        PyQtGraph items are parented into a visible scene before signal wiring.

        Before synchronization, all canvases reserve the same left/right axis
        columns so identical X values map to identical horizontal pixels even
        when one panel has more Y axes than another.
        """
        if not self._panel_canvases:
            return
        canvases = [
            canvas
            for canvas in self._panel_canvases.values()
            if self._qt_widget_alive(canvas)
        ]
        if not canvases:
            self._panel_canvases = {}
            return
        master = canvases[0]

        max_right_axes = max((canvas.right_axis_count() for canvas in canvases), default=0)
        for canvas in canvases:
            canvas.reserve_grouped_axis_columns(max_right_axes)
        timeline = (
            self._grouped_timeline
            if self._qt_widget_alive(self._grouped_timeline)
            else None
        )
        if timeline is not None:
            timeline.reserve_grouped_axis_columns(max_right_axes)
            timeline.match_viewbox_geometry(master._primary_plot.getViewBox())

        self._vis_manager.register_synchronized_panels(
            canvases,
            timeline=timeline,
        )

        t_start, t_end = master._primary_plot.getViewBox().viewRange()[0]
        for canvas in canvases[1:]:
            canvas.normalize_viewport(t_start, t_end)
        if timeline is not None:
            timeline.normalize_viewport(t_start, t_end)
            QTimer.singleShot(
                0,
                lambda: (
                    timeline.match_viewbox_geometry(master._primary_plot.getViewBox())
                    if self._qt_widget_alive(timeline) and self._qt_widget_alive(master)
                    else None
                ),
            )
        # Re-pin Y ranges on ALL canvases after synchronization registration;
        # range propagation can trigger auto-range on sparse panels.
        for canvas in canvases:
            if canvas._sparse_mode:
                canvas._force_y_ranges()

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
        self._canvas.set_measurement_mode(checked)
        for canvas in self._panel_canvases.values():
            if self._qt_widget_alive(canvas):
                canvas.set_measurement_mode(checked)
                canvas.measurement_cursors_moved.connect(
                    self._on_measurement_cursors_moved
                )
        if checked:
            self._measurement_dock.show()
            self._measurement_widget.update_measurements(None)
        else:
            self._measurement_dock.hide()
            self._measurement_widget.update_measurements(None)

    def _on_measurement_cursors_moved(self, t_a: float, t_b: float) -> None:
        result = self._canvas.compute_current_measurements()
        if result is None:
            sender_canvas = self.sender()
            if hasattr(sender_canvas, "compute_current_measurements"):
                result = sender_canvas.compute_current_measurements()
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
                self._canvas.clear_event_markers()
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

        # Apply markers to all active canvases
        self._canvas.add_event_markers(events)
        for canvas in self._panel_canvases.values():
            if self._qt_widget_alive(canvas):
                canvas.add_event_markers(events)

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
        """Jump the canvas viewport to a detected event."""
        window = 0.1  # ±100 ms around the event start
        t_lo = t_start - window
        t_hi = t_start + window
        if self._panel_canvases:
            for canvas in self._panel_canvases.values():
                if self._qt_widget_alive(canvas):
                    canvas._primary_plot.setXRange(t_lo, t_hi, padding=0)
                    break
        elif self._qt_widget_alive(self._canvas):
            self._canvas._primary_plot.setXRange(t_lo, t_hi, padding=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Cursor readout dock (Phase 3 Enhancement)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_cursor_readout_dock(self):
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Cursor Readout", self)
        dock.setObjectName("CursorReadoutDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self._cursor_readout_widget = CursorReadoutBar()
        dock.setWidget(self._cursor_readout_widget)
        dock.setMaximumHeight(60)
        return dock

    def _on_cursor_values_changed(self, t: float, values: list) -> None:
        self._cursor_readout_widget.update_values(t, values)

    def _connect_canvas_readout(self, canvas) -> None:
        """Connect cursor_values_changed from a grouped-layout panel canvas."""
        if hasattr(canvas, "cursor_values_changed"):
            canvas.cursor_values_changed.connect(self._on_cursor_values_changed)

    def _show_cursor_readout(self) -> None:
        """Make the cursor readout dock visible (called after a record loads)."""
        self._cursor_readout_dock.show()

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
            # Refresh browser to apply grade colours
            self._refresh_signal_browser()
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

        file_menu = menu_bar.addMenu("&File")
        open_action = file_menu.addAction("&Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.setToolTip("Open a COMTRADE, CSV, or Excel file")
        open_action.triggered.connect(self._open_file_dialog)
        import_action = file_menu.addAction("&Import Wizard…")
        import_action.setShortcut("Ctrl+I")
        import_action.setToolTip(
            "Open the Import Wizard for CSV or Excel files — "
            "configure columns, timestamps and units before loading"
        )
        import_action.triggered.connect(self._open_import_wizard)
        open_session_action = file_menu.addAction("&Multi-Source Viewer…")
        open_session_action.setShortcut("Ctrl+Shift+N")
        open_session_action.setToolTip(
            "Open a workspace to load and compare multiple recordings side by side"
        )
        open_session_action.triggered.connect(self._on_open_session)
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
        view_menu.addAction(self._signal_browser.toggleViewAction())
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

        axis_mode_menu = view_menu.addMenu("&Axis Mode")
        axis_mode_group = QActionGroup(self)
        axis_mode_group.setExclusive(True)

        shared_axis = axis_mode_menu.addAction("&Shared Axis")
        shared_axis.setCheckable(True)
        shared_axis.setChecked(True)
        shared_axis.triggered.connect(
            lambda: self._on_axis_display_mode_changed(AxisDisplayMode.SHARED)
        )
        axis_mode_group.addAction(shared_axis)
        self._axis_mode_actions[AxisDisplayMode.SHARED] = shared_axis

        dedicated_axis = axis_mode_menu.addAction("&Dedicated Axis")
        dedicated_axis.setCheckable(True)
        dedicated_axis.triggered.connect(
            lambda: self._on_axis_display_mode_changed(AxisDisplayMode.DEDICATED)
        )
        axis_mode_group.addAction(dedicated_axis)
        self._axis_mode_actions[AxisDisplayMode.DEDICATED] = dedicated_axis

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
        view_menu.addAction(self._cursor_readout_dock.toggleViewAction())
        view_menu.addAction(self._quality_dock.toggleViewAction())
        view_menu.addAction(self._fault_dock.toggleViewAction())
        view_menu.addAction(self._protection_dock.toggleViewAction())
        view_menu.addAction(self._correlation_dock.toggleViewAction())
        view_menu.addAction(self._suggestion_dock.toggleViewAction())

        tools_menu = menu_bar.addMenu("&Tools")
        synthetic_action = tools_menu.addAction("Load &Synthetic Mixed Disturbance")
        synthetic_action.setShortcut("Ctrl+T")
        synthetic_action.triggered.connect(self._on_load_synthetic_mixed)
        pulu_action = tools_menu.addAction("Load Sample &PULU Event")
        pulu_action.triggered.connect(self._on_load_sample_pulu)

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

    # ─────────────────────────────────────────────────────────────────────────
    # File loading (standard path)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Disturbance File", "", _FILE_FILTER
        )
        if path_str:
            self._load_file(Path(path_str))

    def _open_import_wizard(self) -> None:
        dlg = ImportWizardDialog(self)
        dlg.import_completed.connect(self._on_import_wizard_record_ready)
        dlg.exec()

    def _on_import_wizard_record_ready(self, record: object) -> None:
        if not isinstance(record, DisturbanceRecord):
            QMessageBox.warning(
                self,
                "Import Wizard",
                "Import did not return a waveform record.",
            )
            return
        self._display_imported_record(record)

    def _display_imported_record(self, record: DisturbanceRecord) -> None:
        """Render an Import Wizard record through existing visualization paths."""
        source_path = _record_source_path(record)
        provider_type = str(getattr(record.metadata, "provider_type", "") or "import")
        source_id = source_path.stem or record.metadata.station_name or "imported_record"

        if self._panel_canvases:
            self._restore_standard_layout()

        self._time_display_mode = TimeDisplayMode.ABSOLUTE
        self._set_time_axis_action_checked(TimeDisplayMode.ABSOLUTE)
        axis_mode = TimeDisplayMode.ABSOLUTE.value
        self._current_signal_metadata = {}

        panel_canvases = self._vis_manager.display_grouped_record(
            record, None, axis_mode=axis_mode
        )
        for canvas in panel_canvases.values():
            canvas.set_axis_display_mode(self._axis_display_mode)

        if panel_canvases:
            seq_panels = self._build_sequence_panels(record, None, FlexiblePlotCanvas)
            panel_canvases.update(seq_panels)
            harmonic_panels = self._build_harmonic_panels(record, None, FlexiblePlotCanvas)
            panel_canvases.update(harmonic_panels)
            self._rebuild_grouped_layout(panel_canvases, record)
        else:
            self._vis_manager.set_record(record, axis_mode=axis_mode)
            self._canvas.set_axis_display_mode(self._axis_display_mode)
            QTimer.singleShot(0, self._link_standard_x_axis)

        self._refresh_signal_browser()
        if self._rms_display_mode != RMSDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_rms_mode_to_all_canvases)
        if self._phasor_registry.display_mode != PhasorDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_phasor_display_mode)
        if self._harmonic_registry.display_mode != HarmonicDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_harmonic_display_mode)
        self.statusBar().showMessage(_format_load_status(record))
        self.setWindowTitle(f"Powerwave - {source_id} ({provider_type})")

    def _load_file(self, path: Path) -> None:
        self.statusBar().showMessage(f"Loading: {path.name} …")
        worker = _IntelligentLoadWorker(
            self._provider_manager, path, self._intelligence_manager
        )
        worker.signals.finished.connect(self._on_record_loaded)
        worker.signals.error.connect(self._on_load_error)
        QThreadPool.globalInstance().start(worker)

    def _on_record_loaded(self, result: object) -> None:
        if isinstance(result, _DirectOpenResult):
            _log_runtime_route(
                f"_on_record_loaded rich result provider={result.provider_type} "
                f"path={result.path}"
            )
            if result.provider_type in ("csv", "excel"):
                self._handle_direct_csv_excel(result)
            else:
                # COMTRADE — preserve existing set_record() behavior
                record = result.record
                _log_runtime_route("COMTRADE direct open -> standard set_record")
                if self._panel_canvases:
                    self._restore_standard_layout()
                self._time_display_mode = TimeDisplayMode.RELATIVE
                self._set_time_axis_action_checked(TimeDisplayMode.RELATIVE)
                self._vis_manager.set_record(record, axis_mode=TimeDisplayMode.RELATIVE.value)
                self._canvas.set_axis_display_mode(self._axis_display_mode)
                QTimer.singleShot(0, self._link_standard_x_axis)
                self._refresh_signal_browser()
                QTimer.singleShot(0, lambda r=record: self._run_event_detection(r))
                QTimer.singleShot(0, lambda r=record: self._run_quality_check(r))
                self._show_cursor_readout()
                self.statusBar().showMessage(_format_load_status(record))
                title = (
                    record.metadata.station_name
                    or Path(record.metadata.source_file).name
                )
                self.setWindowTitle(f"Powerwave — {title}")
        elif isinstance(result, DisturbanceRecord):
            # Fallback for any legacy path that still emits a plain record
            if _is_csv_excel_record(result):
                _log_runtime_route(
                    "legacy DisturbanceRecord CSV/Excel result -> grouped route"
                )
                self._handle_legacy_csv_excel_record(result)
                return
            _log_runtime_route("legacy DisturbanceRecord non-CSV result -> standard set_record")
            if self._panel_canvases:
                self._restore_standard_layout()
            self._time_display_mode = TimeDisplayMode.RELATIVE
            self._set_time_axis_action_checked(TimeDisplayMode.RELATIVE)
            self._vis_manager.set_record(result, axis_mode=TimeDisplayMode.RELATIVE.value)
            self._canvas.set_axis_display_mode(self._axis_display_mode)
            QTimer.singleShot(0, self._link_standard_x_axis)
            self._refresh_signal_browser()
            QTimer.singleShot(0, lambda r=result: self._run_event_detection(r))
            QTimer.singleShot(0, lambda r=result: self._run_quality_check(r))
            self._show_cursor_readout()
            self.statusBar().showMessage(_format_load_status(result))
            title = result.metadata.station_name or Path(result.metadata.source_file).name
            self.setWindowTitle(f"Powerwave — {title}")

    def _handle_legacy_csv_excel_record(self, record: DisturbanceRecord) -> None:
        """Route plain CSV/Excel records through the D4.4 grouped policy."""
        from app.data.direct_load_intelligence import (
            build_signal_metadata,
            detect_timestamp_ambiguity,
        )

        path = _record_source_path(record)
        provider_type = _provider_type_from_path(path)
        if provider_type not in {"csv", "excel"}:
            provider_type = str(record.metadata.provider_type or "csv").lower()
        source_id = path.stem or "csv_source"
        signal_metadata = build_signal_metadata(
            record, self._intelligence_manager, source_id, provider_type
        )
        ts_ambiguous, ts_matrices = detect_timestamp_ambiguity(path, record)
        self._handle_direct_csv_excel(_DirectOpenResult(
            record=record,
            path=path,
            provider_type=provider_type,
            signal_metadata=signal_metadata,
            ts_ambiguous=ts_ambiguous,
            ts_matrices=ts_matrices,
        ))

    def _handle_direct_csv_excel(self, result: _DirectOpenResult) -> None:
        """Display a directly-opened CSV/Excel file with intelligence-driven grouping.

        Shows DataReviewDialog when the timestamp interpretation is ambiguous or
        any column requires operator confirmation. Auto-applies for clean data.
        Applies the operator-selected timestamp format to rebase start_time before
        visualization, and renders with absolute datetime axis labels.
        """
        from PyQt6.QtWidgets import QDialog
        from app.data.direct_load_intelligence import (
            apply_selected_timestamp_format,
            build_direct_open_diagnostics,
            log_direct_open_diagnostics,
        )
        from app.data.multi_source_session import MultiSourceSession, SourceRecord
        from app.data.review_summary import build_event_review_summary
        from app.ui.dialogs.data_review_dialog import DataReviewDialog

        record = result.record
        source_id = result.path.stem or "csv_source"
        signal_metadata: dict = result.signal_metadata
        selected_formats: dict[str, str] = {}
        _log_runtime_route(
            f"_handle_direct_csv_excel executing provider={result.provider_type} "
            f"path={result.path}"
        )

        needs_review = result.ts_ambiguous or any(
            m.requires_user_confirmation for m in signal_metadata.values()
        )

        if needs_review:
            source = SourceRecord(
                source_id=source_id,
                provider_type=result.provider_type,
                record=record,
                signal_metadata=signal_metadata,
                original_start_time=record.timing_info.start_time,
                sampling_rates=list(record.sampling_info.sampling_rates),
            )
            session = MultiSourceSession()
            session.add_source(source)
            summary = build_event_review_summary(session)
            ts_matrices = {source_id: result.ts_matrices} if result.ts_matrices else {}
            dlg = DataReviewDialog(summary, ts_matrices=ts_matrices, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.statusBar().showMessage("Load cancelled.")
                return
            self._rule_manager.save_confirmed_rows(
                dlg.confirmed_column_rows.get(source_id, []),
                source_id=source_id,
            )
            selected_formats = dlg.selected_timestamp_formats.get(source_id, {})

        # Apply operator-selected timestamp format (rebases start_time only)
        if selected_formats:
            record = apply_selected_timestamp_format(
                record, result.ts_matrices, selected_formats
            )

        self._time_display_mode = TimeDisplayMode.ABSOLUTE
        self._set_time_axis_action_checked(TimeDisplayMode.ABSOLUTE)
        axis_mode = TimeDisplayMode.ABSOLUTE.value

        diag = build_direct_open_diagnostics(
            source_path=str(result.path),
            provider_type=result.provider_type,
            signal_metadata=signal_metadata,
            ts_matrices=result.ts_matrices,
            selected_formats=selected_formats,
            axis_mode=axis_mode,
        )
        log_direct_open_diagnostics(diag)

        if self._panel_canvases:
            self._restore_standard_layout()

        self._current_signal_metadata = signal_metadata or {}

        panel_canvases = self._vis_manager.display_grouped_record(
            record, signal_metadata or None, axis_mode=axis_mode
        )
        for canvas in panel_canvases.values():
            canvas.set_axis_display_mode(self._axis_display_mode)
        _log_runtime_route(
            f"display_grouped_record returned panels={list(panel_canvases.keys())}"
        )
        if not panel_canvases:
            raise RuntimeError(
                "CSV/Excel direct open produced no grouped panels; refusing "
                "to fall back to the standard analog/digital splitter."
            )

        # Build sequence component panels if three-phase groups exist (Phase 6B)
        seq_panels = self._build_sequence_panels(
            record, signal_metadata or None, FlexiblePlotCanvas
        )
        panel_canvases.update(seq_panels)

        # Build harmonic analysis panels (Phase 8)
        harmonic_panels = self._build_harmonic_panels(
            record, signal_metadata or None, FlexiblePlotCanvas
        )
        panel_canvases.update(harmonic_panels)

        self._rebuild_grouped_layout(panel_canvases, record)
        self._refresh_signal_browser()

        # Re-apply RMS mode to the newly created panels if mode is active
        if self._rms_display_mode != RMSDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_rms_mode_to_all_canvases)

        # Re-apply phasor mode to the newly created panels if mode is active
        if self._phasor_registry.display_mode != PhasorDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_phasor_display_mode)

        # Re-apply harmonic mode to the newly created panels if mode is active
        if self._harmonic_registry.display_mode != HarmonicDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_harmonic_display_mode)

        QTimer.singleShot(0, lambda r=record: self._run_event_detection(r))
        QTimer.singleShot(0, lambda r=record: self._run_quality_check(r))
        self._show_cursor_readout()
        for canvas in panel_canvases.values():
            self._connect_canvas_readout(canvas)
        self.statusBar().showMessage(_format_load_status(record))
        self.setWindowTitle(f"Powerwave — {source_id}")

    def _on_load_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Load Error", message)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-source loading (Phase D3)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_multi_source_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Multi-Source Session", "", _FILE_FILTER
        )
        if not paths:
            return
        if len(paths) == 1:
            self._load_file(Path(paths[0]))
            return
        self._load_multi_source([Path(p) for p in paths])

    def _load_multi_source(self, paths: list[Path]) -> None:
        from app.data.multi_source_session import MultiSourceSession

        self.statusBar().showMessage(f"Loading {len(paths)} sources…")
        session = MultiSourceSession()
        errors: list[str] = []
        for path in paths:
            try:
                record = self._provider_manager.load(path)
                source = _make_source_record(
                    path.stem, record, provider_type=path.suffix.lstrip(".")
                )
                session.add_source(source)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}: {exc}")
        if errors:
            QMessageBox.warning(self, "Load Warnings", "\n".join(errors))
        if session.is_empty():
            self.statusBar().showMessage("No sources loaded.")
            return
        self._on_multi_source_loaded(session)

    def _on_multi_source_loaded(self, session) -> None:
        if self._panel_canvases:
            self._restore_standard_layout()
        self._time_display_mode = TimeDisplayMode.ABSOLUTE
        self._set_time_axis_action_checked(TimeDisplayMode.ABSOLUTE)
        panel_canvases = self._vis_manager.display_multi_source_session(
            session,
            axis_mode=TimeDisplayMode.ABSOLUTE.value,
        )
        for canvas in panel_canvases.values():
            canvas.set_axis_display_mode(self._axis_display_mode)
        first_record = session.sources[0].record
        self._rebuild_grouped_layout(panel_canvases, first_record)
        self._refresh_signal_browser()
        n = len(panel_canvases)
        ids = ", ".join(s.source_id for s in session.sources)
        self.setWindowTitle(f"Powerwave — Multi-Source: {ids}")
        self.statusBar().showMessage(
            f"Multi-source session: {n} panel(s) from {session.source_count()} source(s)"
        )
        # Cross-source correlation (Phase 7)
        QTimer.singleShot(0, lambda s=session: self._run_cross_correlation(s.sources))

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic mixed display (Phase D2)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load_synthetic_mixed(self) -> None:
        """Load and display the synthetic mixed-source disturbance (Phase D2 dev action)."""
        from app.data.synthetic import make_mixed_disturbance_record
        self.statusBar().showMessage("Generating synthetic mixed disturbance…")
        result = make_mixed_disturbance_record()
        self._time_display_mode = TimeDisplayMode.RELATIVE
        self._set_time_axis_action_checked(TimeDisplayMode.RELATIVE)
        panel_canvases = self._vis_manager.display_grouped_record(
            result.record, result.signal_metadata
        )
        for canvas in panel_canvases.values():
            canvas.set_axis_display_mode(self._axis_display_mode)

        # Build sequence panels if three-phase groups exist (Phase 6B)
        seq_panels = self._build_sequence_panels(
            result.record, result.signal_metadata, FlexiblePlotCanvas
        )
        panel_canvases.update(seq_panels)

        # Build harmonic analysis panels (Phase 8)
        harmonic_panels = self._build_harmonic_panels(
            result.record, result.signal_metadata, FlexiblePlotCanvas
        )
        panel_canvases.update(harmonic_panels)

        self._rebuild_grouped_layout(panel_canvases, result.record)
        self._refresh_signal_browser()

        if self._phasor_registry.display_mode != PhasorDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_phasor_display_mode)

        if self._harmonic_registry.display_mode != HarmonicDisplayMode.OFF:
            QTimer.singleShot(0, self._apply_harmonic_display_mode)

        n = len(panel_canvases)
        self.setWindowTitle("Powerwave — Synthetic Mixed Disturbance")
        self.statusBar().showMessage(f"Synthetic mixed disturbance: {n} panel(s)")

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

    def _on_load_sample_pulu(self) -> None:
        """Load the built-in pulu_20260306 sample event manifest."""
        self._load_manifest(_SAMPLE_MANIFEST)

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
        self._vis_manager.set_time_axis_mode(self._time_display_mode)
        if (
            self._session_canvas_active
            and self._session_canvas_controller is not None
            and self._active_session is not None
        ):
            self._session_canvas_controller.set_time_axis_mode(
                self._time_display_mode, self._active_session
            )

    def _set_axis_mode_action_checked(self, mode: AxisDisplayMode) -> None:
        action = self._axis_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)

    def _on_axis_display_mode_changed(self, mode: AxisDisplayMode) -> None:
        """Switch visible panels between shared and dedicated Y-axis grouping."""
        self._axis_display_mode = mode
        self._apply_axis_display_mode_to_visible()
        self._set_axis_mode_action_checked(mode)
        self.statusBar().showMessage(
            "Axis mode: Shared" if mode == AxisDisplayMode.SHARED else "Axis mode: Dedicated"
        )

    def _apply_axis_display_mode_to_visible(self) -> None:
        """Apply current Y-axis grouping without reloading records."""
        if self._session_canvas_active:
            if (
                self._session_canvas_controller is not None
                and self._active_session is not None
            ):
                self._session_canvas_controller.set_axis_display_mode(
                    self._axis_display_mode, self._active_session
                )
            return
        canvases = list(self._panel_canvases.values()) if self._panel_canvases else [self._canvas]
        for canvas in canvases:
            if self._qt_widget_alive(canvas):
                canvas.set_axis_display_mode(self._axis_display_mode)
        if self._panel_canvases:
            QTimer.singleShot(0, self._link_panel_x_axes)
        else:
            QTimer.singleShot(0, self._link_standard_x_axis)

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
        canvases = list(self._panel_canvases.values()) if self._panel_canvases else [self._canvas]
        for canvas in canvases:
            if not self._qt_widget_alive(canvas):
                continue
            canvas.set_rms_display_mode(RMSDisplayMode.OFF)
            canvas.set_rms_display_mode(
                self._rms_display_mode,
                config=self._rms_config,
                signal_metadata=self._current_signal_metadata or None,
            )

    def _apply_rms_mode_to_all_canvases(self) -> None:
        """Push the current RMS mode to every visible canvas."""
        canvases = []
        if self._panel_canvases:
            canvases.extend(self._panel_canvases.values())
        else:
            canvases.append(self._canvas)

        sig_meta = self._current_signal_metadata or None
        for canvas in canvases:
            if not self._qt_widget_alive(canvas):
                continue
            canvas.set_rms_display_mode(
                self._rms_display_mode,
                config=self._rms_config,
                signal_metadata=sig_meta,
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
        """Push the current scaling mode and registry to every visible canvas."""
        canvases = list(self._panel_canvases.values()) if self._panel_canvases else [self._canvas]
        for canvas in canvases:
            if self._qt_widget_alive(canvas):
                canvas.set_scaling_mode(self._scaling_mode, registry=self._scaling_registry)

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
        """Show or hide frequency/ROCOF panel canvases based on current mode."""
        if not self._panel_canvases:
            return
        mode = self._frequency_registry.display_mode
        freq_keys = self._frequency_registry.frequency_panel_keys(
            list(self._panel_canvases.keys())
        )
        visible = mode != FrequencyDisplayMode.OFF
        for key in freq_keys:
            canvas = self._panel_canvases.get(key)
            if canvas is not None and self._qt_widget_alive(canvas):
                canvas.setVisible(visible)

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
        mode = self._phasor_registry.display_mode

        def _is_sequence_key(k: str) -> bool:
            return (
                k in ("sequence_voltage", "sequence_current")
                or k.endswith("/sequence_voltage")
                or k.endswith("/sequence_current")
            )

        # Gather waveform canvases (not sequence panels)
        if self._panel_canvases:
            waveform_canvases = [
                c for k, c in self._panel_canvases.items()
                if not _is_sequence_key(k) and self._qt_widget_alive(c)
            ]
        elif self._qt_widget_alive(self._canvas):
            waveform_canvases = [self._canvas]
        else:
            waveform_canvases = []

        # Apply phasor overlay mode to waveform canvases
        sig_meta = self._current_signal_metadata or None
        for canvas in waveform_canvases:
            if hasattr(canvas, "set_performance_timing"):
                canvas.set_performance_timing(
                    self._performance_timing_enabled,
                    self._performance_timing_sink,
                )
            canvas.set_phasor_display_mode(
                mode, signal_metadata=sig_meta
            )

        # Show or hide sequence component panels
        seq_visible = (mode == PhasorDisplayMode.SEQUENCE_COMPONENTS)
        for key, canvas in (self._panel_canvases or {}).items():
            if _is_sequence_key(key) and self._qt_widget_alive(canvas):
                canvas.setVisible(seq_visible)

    # ─────────────────────────────────────────────────────────────────────────
    # Sequence panel construction (Phase 6B)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sequence_panels(
        self,
        record: DisturbanceRecord,
        signal_metadata: dict | None,
        canvas_factory,
    ) -> dict:
        """Build sequence component canvas panels for a single-record grouped layout.

        Detects three-phase groups, computes V1/V2/V0 and I1/I2/I0 magnitudes,
        and returns panel canvases keyed by "sequence_voltage" / "sequence_current".
        Panels start hidden; _apply_phasor_display_mode() controls visibility.

        Returns {} when no complete three-phase groups are found or on any error.
        Multi-source sessions are not handled here (document as Phase 6B limitation).
        """
        import numpy as np
        from app.analytics.phasors.phasor_extraction import (
            compute_phasor_window_samples,
            extract_phasor,
        )
        from app.analytics.phasors.symmetrical_components import (
            compute_sequence_from_phasor_arrays,
        )

        if record is None:
            return {}

        channel_names = [ch.name for ch in record.analog_channels]
        channel_phases = {
            ch.name: ch.phase
            for ch in record.analog_channels
            if ch.phase
        }

        groups = self._phasor_registry.detect_three_phase_groups(
            channel_names,
            signal_metadata or None,
            channel_phases or None,
        )
        if not groups:
            return {}

        # Estimate sample rate
        try:
            time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        except (KeyError, Exception):  # noqa: BLE001
            return {}

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
            return {}

        config = self._phasor_registry.config
        seq_voltage_data: dict[str, np.ndarray] = {}
        seq_current_data: dict[str, np.ndarray] = {}
        seq_time: np.ndarray | None = None

        with timed_section(
            "sequence_rendering",
            enabled=self._performance_timing_enabled,
            sink=self._performance_timing_sink,
        ):
            for group in groups:
                if not group.complete:
                    continue
                try:
                    ch_a = record.waveform_data[group.phase_a].to_numpy(dtype=np.float64)  # type: ignore[index]
                    ch_b = record.waveform_data[group.phase_b].to_numpy(dtype=np.float64)  # type: ignore[index]
                    ch_c = record.waveform_data[group.phase_c].to_numpy(dtype=np.float64)  # type: ignore[index]
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
            return {}

        result: dict = {}
        for panel_key, data_dict, unit, title in [
            ("sequence_voltage", seq_voltage_data, "V", "Sequence Voltage"),
            ("sequence_current", seq_current_data, "A", "Sequence Current"),
        ]:
            if not data_dict:
                continue
            syn_record = _make_sequence_record(record, seq_time, data_dict, unit)
            canvas = canvas_factory()
            canvas.set_record(syn_record)
            canvas.set_panel_title(title)
            canvas.setVisible(False)
            result[panel_key] = canvas

        return result

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
        mode = self._harmonic_registry.display_mode

        # Gather waveform canvases (exclude harmonic analysis panels)
        if self._panel_canvases:
            waveform_canvases = [
                c for k, c in self._panel_canvases.items()
                if k not in _HARMONIC_PANEL_KEYS and self._qt_widget_alive(c)
            ]
        elif self._qt_widget_alive(self._canvas):
            waveform_canvases = [self._canvas]
        else:
            waveform_canvases = []

        # Push harmonic overlay mode to waveform canvases
        sig_meta = self._current_signal_metadata or None
        for canvas in waveform_canvases:
            if hasattr(canvas, "set_performance_timing"):
                canvas.set_performance_timing(
                    self._performance_timing_enabled,
                    self._performance_timing_sink,
                )
            canvas.set_harmonic_display_mode(mode, signal_metadata=sig_meta)

        # Show/hide THD panels
        thd_visible = (mode == HarmonicDisplayMode.THD)
        for key in ("thd_voltage", "thd_current"):
            canvas = (self._panel_canvases or {}).get(key)
            if canvas is not None and self._qt_widget_alive(canvas):
                canvas.setVisible(thd_visible)

        # Show/hide spectrum panels
        spectrum_visible = (mode == HarmonicDisplayMode.SPECTRUM)
        for key in ("harmonic_spectrum_voltage", "harmonic_spectrum_current"):
            canvas = (self._panel_canvases or {}).get(key)
            if canvas is not None and self._qt_widget_alive(canvas):
                canvas.setVisible(spectrum_visible)

    def _build_harmonic_panels(
        self,
        record: DisturbanceRecord,
        signal_metadata: dict | None,
        canvas_factory,
    ) -> dict:
        """Build THD trend and harmonic spectrum panels for a grouped layout.

        Computes harmonic extraction for all eligible channels, then constructs:
          - thd_voltage / thd_current  — time-varying THD% trend panels
          - harmonic_spectrum_voltage / harmonic_spectrum_current
                                       — H3..H13 magnitude trend panels

        Panels start hidden; _apply_harmonic_display_mode() controls visibility.
        Returns {} when no eligible channels are found or on any error.
        """
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
            return {}

        config = self._harmonic_registry.config

        try:
            time_col = record.waveform_data["time"].to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            return {}

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
            return {}

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
        # Spectrum: H3..H13 for the first eligible voltage/current channel only.
        spec_voltage: dict[str, np.ndarray] = {}
        spec_current: dict[str, np.ndarray] = {}
        harmonic_time: np.ndarray | None = None
        _SPECTRUM_ORDERS = [3, 5, 7, 11, 13]

        with timed_section(
            "harmonic_panel_build",
            enabled=self._performance_timing_enabled,
            sink=self._performance_timing_sink,
        ):
            for ch in record.analog_channels:
                name = ch.name
                meta = (signal_metadata or {}).get(name)
                role = classify_harmonic_role(name, ch.unit, meta).role
                if role == HarmonicChannelRole.UNKNOWN:
                    continue

                try:
                    raw_data = record.waveform_data[name].to_numpy(dtype=np.float64)
                except Exception:  # noqa: BLE001
                    continue

                cached = cache.get(
                    name, window, hop, config.nominal_hz, config.max_order
                )
                if cached is not None:
                    h_result = cached
                else:
                    try:
                        h_result = extract_harmonics(
                            raw_data, sample_rate_hz, config, time=time_col
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    cache.put(
                        name, window, hop, config.nominal_hz, config.max_order, h_result
                    )

                if h_result.n_windows == 0:
                    continue

                harmonic_time = h_result.harmonic_time
                thd_arr = compute_thd_array(h_result.magnitudes) * 100.0  # → percent

                if role == HarmonicChannelRole.VOLTAGE_HARMONIC:
                    thd_voltage[name] = thd_arr
                    if not spec_voltage:
                        for order in _SPECTRUM_ORDERS:
                            mag = h_result.get_magnitude(order)
                            if mag is not None:
                                spec_voltage[f"H{order}"] = mag
                else:
                    thd_current[name] = thd_arr
                    if not spec_current:
                        for order in _SPECTRUM_ORDERS:
                            mag = h_result.get_magnitude(order)
                            if mag is not None:
                                spec_current[f"H{order}"] = mag

        if harmonic_time is None:
            return {}

        result: dict = {}
        for panel_key, data_dict, unit, title in [
            ("thd_voltage",              thd_voltage,   "%",     "THD — Voltage (%)"),
            ("thd_current",              thd_current,   "%",     "THD — Current (%)"),
            ("harmonic_spectrum_voltage", spec_voltage, "V RMS", "Harmonic Spectrum — Voltage"),
            ("harmonic_spectrum_current", spec_current, "A RMS", "Harmonic Spectrum — Current"),
        ]:
            if not data_dict:
                continue
            syn_record = _make_harmonic_record(record, harmonic_time, data_dict, unit)
            canvas = canvas_factory()
            canvas.set_record(syn_record)
            canvas.set_panel_title(title)
            canvas.setVisible(False)
            result[panel_key] = canvas

        return result

    def _refresh_signal_browser(self) -> None:
        """Rebuild the dock tree from the current visualization runtime state."""
        entries: list[SignalBrowserEntry] = []
        targets: dict[str, tuple[str, str, str]] = {}

        def source_label(record: DisturbanceRecord | None, fallback: str) -> str:
            if record is None:
                return fallback
            provider = str(getattr(record.metadata, "provider_type", "") or "").upper()
            source_file = Path(str(getattr(record.metadata, "source_file", "") or "")).name
            return provider or source_file or fallback

        def add_entry(
            *,
            kind: str,
            panel_key: str,
            source: str,
            group: str,
            name: str,
            visible: bool,
        ) -> None:
            key = f"signal-{len(targets)}"
            targets[key] = (kind, panel_key, name)
            quality_grade: str | None = None
            quality_tooltip: str = ""
            if self._quality_fingerprint and name in self._quality_fingerprint.channels:
                ch_q = self._quality_fingerprint.channels[name]
                quality_grade = ch_q.grade.value
                quality_tooltip = ch_q.tooltip
            entries.append(SignalBrowserEntry(
                key=key,
                source=source,
                group=group,
                name=name,
                visible=visible,
                kind=kind,
                quality_grade=quality_grade,
                quality_tooltip=quality_tooltip,
            ))

        if self._panel_canvases:
            for panel_key, canvas in self._panel_canvases.items():
                if not self._qt_widget_alive(canvas):
                    continue
                record = getattr(canvas, "_record", None)
                if "/" in panel_key:
                    src, group_key = panel_key.split("/", 1)
                    src_label = src
                else:
                    group_key = panel_key
                    src_label = source_label(record, "Record")
                group_label = getattr(canvas, "_panel_base_title", "") or group_key
                visible = set(canvas.visible_channel_names())
                for name in canvas.all_channel_names():
                    add_entry(
                        kind="analog",
                        panel_key=panel_key,
                        source=src_label,
                        group=group_label,
                        name=name,
                        visible=name in visible,
                    )
        elif self._qt_widget_alive(self._canvas):
            record = getattr(self._canvas, "_record", None)
            visible = set(self._canvas.visible_channel_names())
            for name in self._canvas.all_channel_names():
                add_entry(
                    kind="analog",
                    panel_key="standard",
                    source=source_label(record, "Record"),
                    group="Analog",
                    name=name,
                    visible=name in visible,
                )

        timeline = self._grouped_timeline if self._grouped_timeline is not None else self._timeline
        if self._qt_widget_alive(timeline):
            record = getattr(timeline, "_record", None)
            visible = set(timeline.visible_channel_names())
            for name in timeline.all_channel_names():
                add_entry(
                    kind="digital",
                    panel_key="digital",
                    source=source_label(record, "Record"),
                    group="Digital",
                    name=name,
                    visible=name in visible,
                )

        self._signal_entry_targets = targets
        self._signal_browser.set_entries(entries)

    def _on_signal_visibility_changed(self, entry_key: str, visible: bool) -> None:
        """Apply a Signal Browser checkbox change to live plot widgets."""
        target = self._signal_entry_targets.get(entry_key)
        if target is None:
            return
        kind, panel_key, name = target
        if kind == "analog":
            canvas = self._canvas if panel_key == "standard" else self._panel_canvases.get(panel_key)
            if not self._qt_widget_alive(canvas):
                return
            names = set(canvas.visible_channel_names())
            if visible:
                names.add(name)
            else:
                names.discard(name)
            ordered = [ch_name for ch_name in canvas.all_channel_names() if ch_name in names]
            canvas.set_visible_channels(ordered)
        elif kind == "digital":
            timeline = self._grouped_timeline if self._grouped_timeline is not None else self._timeline
            if not self._qt_widget_alive(timeline):
                return
            names = set(timeline.visible_channel_names())
            if visible:
                names.add(name)
            else:
                names.discard(name)
            ordered = [ch_name for ch_name in timeline.all_channel_names() if ch_name in names]
            timeline.set_visible_channels(ordered)
        else:
            return

        if self._panel_canvases:
            QTimer.singleShot(0, self._link_panel_x_axes)
        elif self._qt_widget_alive(self._canvas):
            QTimer.singleShot(0, self._link_standard_x_axis)

        self._refresh_signal_browser()
        self.statusBar().showMessage(f"{name}: {'visible' if visible else 'hidden'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Session workspace (Phase 9B)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_session_panel(self) -> SessionPanel:
        """Create the session panel on first use and dock it."""
        if self._session_panel is None:
            self._session_panel = SessionPanel(self)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._session_panel
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
        panel.offset_reset_requested.connect(self._on_session_offset_reset)
        panel.auto_align_requested.connect(self._on_session_auto_align)
        panel.channel_visibility_changed.connect(
            self._on_session_channel_visibility
        )
        panel.channel_colour_change_requested.connect(
            self._on_session_channel_colour
        )
        panel.channel_panel_changed.connect(self._on_session_channel_panel)
        panel.session_cleared.connect(self._on_session_cleared)
        panel.source_active_changed.connect(self._on_session_source_active)
        panel.set_as_reference_requested.connect(self._on_session_set_as_reference)

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
        if self._active_session is None:
            return
        if self._session_canvas_controller is None:
            self._session_canvas_controller = SessionCanvasController()

        self._clear_sync_before_layout_switch()
        splitter = self._session_canvas_controller.rebuild_layout(self._active_session)
        self.setCentralWidget(splitter)
        self._panel_canvases = {}
        self._grouped_timeline = None
        self._session_canvas_active = True
        self._session_canvas_action.setChecked(True)

        self._session_canvas_controller.register_with_sync(
            self._vis_manager.synchronization_manager
        )
        self._session_canvas_controller.refresh_all(self._active_session)

        # Sync session canvas to the current time-axis and Y-axis modes so it
        # matches whatever was already set in the single-source viewer.
        self._session_canvas_controller.set_time_axis_mode(
            self._time_display_mode, self._active_session
        )
        self._session_canvas_controller.set_axis_display_mode(
            self._axis_display_mode, self._active_session
        )

        n = len(self._active_session.list_sources())
        self.statusBar().showMessage(f"Session canvas: {n} source(s) loaded.")

    def _deactivate_session_canvas(self) -> None:
        """Restore standard layout and exit session canvas mode."""
        self._session_canvas_active = False
        self._session_canvas_action.setChecked(False)
        self._restore_standard_layout()

    def _on_open_session(self) -> None:
        """Start a fresh session workspace. User adds sources via the Session Panel."""
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
            dlg = ImportWizardDialog(self)
            dlg.load_page.set_path(path_str)
            # Trigger profiling once the dialog is visible so Next is enabled immediately.
            QTimer.singleShot(0, dlg.profile_selected_file)
            dlg.import_completed.connect(self._on_session_import_record_ready)
            dlg.exec()
        else:
            self.statusBar().showMessage(f"Loading {path.name}…")
            try:
                record = self._provider_manager.load(path)
                self._on_session_import_record_ready(record)
            except Exception as exc:  # noqa: BLE001
                self._on_load_error(str(exc))

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
        n = len(self._active_session.list_sources())
        self.statusBar().showMessage(
            f"Session: {n} source(s) loaded — {display_name} added."
        )

    def _on_session_add_source_requested(self) -> None:
        """Handle 'Add Source' button inside the session panel."""
        self._on_add_to_session()

    def _on_session_remove_source(self, source_id: str) -> None:
        if self._active_session is None:
            return
        self._active_session.remove_source(source_id)
        if self._session_panel is not None:
            self._session_panel.remove_source_row(source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_source_removed(source_id)
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

    def _on_session_offset_reset(self, source_id: str) -> None:
        if self._active_session is None:
            return
        self._active_session.set_time_offset(source_id, 0.0, method="none")
        self._active_session.set_alignment_notes(source_id, "")
        self._refresh_session_source_row(source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.on_offset_changed(
                source_id, 0.0, self._active_session
            )
        self.statusBar().showMessage(f"Offset reset to 0.000 s for {source_id}.")

    def _on_session_set_as_reference(self, source_id: str) -> None:
        """Set one source as the time reference: its offset becomes 0.0 and all
        other sources are shifted to maintain their relative positions."""
        if self._active_session is None:
            return
        ref_source = self._active_session.get_source(source_id)
        if ref_source is None:
            return
        ref_offset = ref_source.time_offset_s
        for source in self._active_session.list_sources():
            if source.source_id == source_id:
                new_offset = 0.0
            else:
                new_offset = source.time_offset_s - ref_offset
            self._active_session.set_time_offset(
                source.source_id, new_offset, method="manual"
            )
            self._active_session.set_alignment_notes(source.source_id, "")
            self._refresh_session_source_row(source.source_id)
        if self._session_canvas_active and self._session_canvas_controller is not None:
            self._session_canvas_controller.refresh_all(self._active_session)
        self.statusBar().showMessage(
            f"Set '{ref_source.display_name}' as reference (t=0)."
        )

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
        for result in results:
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

        n = len(results)
        self.statusBar().showMessage(f"Auto-aligned {n} source(s).")

        # Cross-correlation pass (Phase 7) — runs after trigger-based alignment
        if source_id == "all" and len(targets) >= 2:
            QTimer.singleShot(0, lambda t=targets: self._run_cross_correlation(t))

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

    def _on_toggle_legend(self, checked: bool) -> None:
        if self._session_canvas_controller is not None:
            self._session_canvas_controller.set_legend_visible(checked)

    def _on_session_channel_panel(
        self, source_id: str, channel_name: str, panel_id: str
    ) -> None:
        if self._active_session is None:
            return
        self._active_session.set_channel_panel(source_id, channel_name, panel_id)

    def _on_session_cleared(self) -> None:
        from app.sessions import EventAnalysisSession

        self._active_session = EventAnalysisSession()
        if self._session_canvas_active:
            self._deactivate_session_canvas()
        self._session_canvas_action.setEnabled(False)
        self._save_manifest_action.setEnabled(False)
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
