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

from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QMessageBox,
)

from app.data.intelligence import IntelligenceManager
from app.models import DisturbanceRecord
from app.providers import ProviderManager, ComtradeProvider, CsvProvider, ExcelProvider
from app.visualization.managers.visualization_manager import VisualizationManager
from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas
from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline

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
    return provider_type in {"csv", "excel"} or path.suffix.lower() in _CSV_EXCEL_SUFFIXES


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

# Preferred display order for grouped panels
_PANEL_ORDER = [
    "voltage_raw",
    "current_raw",
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
        intelligence_manager: IntelligenceManager,
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
        self._intelligence_manager = IntelligenceManager()
        self._canvas = FlexiblePlotCanvas()
        self._timeline = DigitalEventTimeline()
        self._vis_manager = VisualizationManager(self._canvas, self._timeline)
        self._x_axis_linked: bool = False
        self._panel_canvases: dict = {}
        self._grouped_timeline = None

        self._build_layout()
        self._build_menu()

    # ─────────────────────────────────────────────────────────────────────────
    # Qt overrides
    # ─────────────────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._x_axis_linked:
            self._vis_manager.link_x_axis()
            self._x_axis_linked = True

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self._restore_standard_layout()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _restore_standard_layout(self) -> None:
        """Restore the two-pane standard layout (canvas + timeline)."""
        self._timeline.show()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._timeline)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self._panel_canvases = {}
        self._grouped_timeline = None

    def _rebuild_grouped_layout(
        self,
        panel_canvases: dict,
        record: DisturbanceRecord,
    ) -> None:
        """Replace central widget with grouped stacked display.

        Panels are inserted in _PANEL_ORDER; any unrecognised groups follow.
        X-axis linking is deferred via QTimer.singleShot(0) so PyQtGraph
        scenes are fully initialised before setXLink is called.

        Each panel canvas is given a minimum height (180 px) so the QSplitter
        cannot collapse any panel to zero.  Initial splitter sizes are set
        explicitly so all panels share space equally on first show — without
        this the splitter may give the top panel all available space.
        """
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
        QTimer.singleShot(0, self._link_panel_x_axes)

    def _link_panel_x_axes(self) -> None:
        """Link all grouped panel canvases to the first panel's X-axis.

        Called via QTimer.singleShot(0) from _rebuild_grouped_layout to ensure
        PyQtGraph items are parented into a visible scene before setXLink.

        After linking, each follower is explicitly normalized to the master's
        exact data range.  PyQtGraph's linkedViewChanged() maps ranges via screen
        pixel widths; panels with different secondary axis counts have different
        primary-ViewBox widths, so the initial linked range is asymmetrically
        stretched.  normalize_viewport() overrides that with a uniform data range
        and re-pins Y ranges for sparse records.
        """
        if not self._panel_canvases:
            return
        canvases = list(self._panel_canvases.values())
        master = canvases[0]
        for follower in canvases[1:]:
            follower._primary_plot.setXLink(master._primary_plot)
        if self._grouped_timeline is not None:
            self._grouped_timeline.link_x_to(master._primary_plot)

        t_start, t_end = master._primary_plot.getViewBox().viewRange()[0]
        for canvas in canvases[1:]:
            canvas.normalize_viewport(t_start, t_end)
        # Re-pin Y ranges on ALL canvases after X-linking — setXLink may trigger
        # auto-range or linkedViewChanged which can override previously pinned ranges.
        for canvas in canvases:
            if canvas._sparse_mode:
                canvas._force_y_ranges()

    # ─────────────────────────────────────────────────────────────────────────
    # Menu
    # ─────────────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        open_action = file_menu.addAction("&Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        multi_action = file_menu.addAction("Open &Multi-Source…")
        multi_action.setShortcut("Ctrl+M")
        multi_action.triggered.connect(self._open_multi_source_dialog)
        manifest_action = file_menu.addAction("Open &Event Manifest…")
        manifest_action.setShortcut("Ctrl+E")
        manifest_action.triggered.connect(self._open_manifest_dialog)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        tools_menu = menu_bar.addMenu("&Tools")
        synthetic_action = tools_menu.addAction("Load &Synthetic Mixed Disturbance")
        synthetic_action.setShortcut("Ctrl+T")
        synthetic_action.triggered.connect(self._on_load_synthetic_mixed)
        pulu_action = tools_menu.addAction("Load Sample &PULU Event")
        pulu_action.triggered.connect(self._on_load_sample_pulu)

    # ─────────────────────────────────────────────────────────────────────────
    # File loading (standard path)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Disturbance File", "", _FILE_FILTER
        )
        if path_str:
            self._load_file(Path(path_str))

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
                self._vis_manager.set_record(record)
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
            self._vis_manager.set_record(result)
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
            selected_formats = dlg.selected_timestamp_formats.get(source_id, {})

        # Apply operator-selected timestamp format (rebases start_time only)
        if selected_formats:
            record = apply_selected_timestamp_format(
                record, result.ts_matrices, selected_formats
            )

        axis_mode = "absolute_datetime"

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

        panel_canvases = self._vis_manager.display_grouped_record(
            record, signal_metadata or None, axis_mode=axis_mode
        )
        _log_runtime_route(
            f"display_grouped_record returned panels={list(panel_canvases.keys())}"
        )
        if not panel_canvases:
            raise RuntimeError(
                "CSV/Excel direct open produced no grouped panels; refusing "
                "to fall back to the standard analog/digital splitter."
            )
        self._rebuild_grouped_layout(panel_canvases, record)

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
        panel_canvases = self._vis_manager.display_multi_source_session(session)
        first_record = session.sources[0].record
        self._rebuild_grouped_layout(panel_canvases, first_record)
        n = len(panel_canvases)
        ids = ", ".join(s.source_id for s in session.sources)
        self.setWindowTitle(f"Powerwave — Multi-Source: {ids}")
        self.statusBar().showMessage(
            f"Multi-source session: {n} panel(s) from {session.source_count()} source(s)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic mixed display (Phase D2)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load_synthetic_mixed(self) -> None:
        """Load and display the synthetic mixed-source disturbance (Phase D2 dev action)."""
        from app.data.synthetic import make_mixed_disturbance_record
        self.statusBar().showMessage("Generating synthetic mixed disturbance…")
        result = make_mixed_disturbance_record()
        panel_canvases = self._vis_manager.display_grouped_record(
            result.record, result.signal_metadata
        )
        self._rebuild_grouped_layout(panel_canvases, result.record)
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

        self._on_multi_source_loaded(session)
