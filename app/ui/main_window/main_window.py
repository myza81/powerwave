"""PowerwaveMainWindow — operational Powerwave viewer.

Supports two display modes:
  Standard   — single DisturbanceRecord in Phase 4A two-pane layout
  Grouped    — mixed-source record in stacked multi-panel layout (Phase D2)

File loading runs on a QRunnable background thread.
Grouped synthetic load runs on the UI thread (fast, <100 ms).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QMessageBox,
)

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
    """Cross-thread signals for _LoadWorker."""

    finished: pyqtSignal = pyqtSignal(object)  # emits DisturbanceRecord
    error: pyqtSignal = pyqtSignal(str)         # emits error message


class _LoadWorker(QRunnable):
    """Loads a file on a background thread via QThreadPool."""

    def __init__(self, provider_manager: ProviderManager, path: Path) -> None:
        super().__init__()
        self._manager = provider_manager
        self._path = path
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            record = self._manager.load(self._path)
            self.signals.finished.emit(record)
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
            splitter.addWidget(canvas)
            splitter.setStretchFactor(i, 1)
            self._panel_canvases[key] = canvas

        self._grouped_timeline = None
        if record.digital_channels:
            splitter.addWidget(self._timeline)
            splitter.setStretchFactor(len(ordered_keys), 1)
            self._grouped_timeline = self._timeline

        self.setCentralWidget(splitter)
        QTimer.singleShot(0, self._link_panel_x_axes)

    def _link_panel_x_axes(self) -> None:
        """Link all grouped panel canvases to the first panel's X-axis.

        Called via QTimer.singleShot(0) from _rebuild_grouped_layout to ensure
        PyQtGraph items are parented into a visible scene before setXLink.
        """
        if not self._panel_canvases:
            return
        canvases = list(self._panel_canvases.values())
        master = canvases[0]
        for follower in canvases[1:]:
            follower._primary_plot.setXLink(master._primary_plot)
        if self._grouped_timeline is not None:
            self._grouped_timeline.link_x_to(master._primary_plot)

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
        worker = _LoadWorker(self._provider_manager, path)
        worker.signals.finished.connect(self._on_record_loaded)
        worker.signals.error.connect(self._on_load_error)
        QThreadPool.globalInstance().start(worker)

    def _on_record_loaded(self, record: DisturbanceRecord) -> None:
        # Return to standard layout if grouped display was active
        if self._panel_canvases:
            self._restore_standard_layout()
        self._vis_manager.set_record(record)
        self.statusBar().showMessage(_format_load_status(record))
        title = record.metadata.station_name or Path(record.metadata.source_file).name
        self.setWindowTitle(f"Powerwave — {title}")

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
