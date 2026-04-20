"""
src/main.py

PowerWave Analyst — application entry point.

MainWindow layout (BEN32-style):
  Central widget: QScrollArea containing a horizontal container:
    Left  : LabelPanel (fixed 160px) — channel name / unit strip
    Right : ChannelCanvas            — waveform / trend / digital rows

  Both LabelPanel and ChannelCanvas sit inside the same QScrollArea so
  vertical scrolling keeps them perfectly synchronised.

  Menu: File > Open (Ctrl+O), File > Exit
  Status bar: loaded filename + channel count

File open flow (LAW 2 — never block the UI thread):
  1. QFileDialog picks .cfg / .csv / .xlsx / .xls files
  2. File type detected by extension → correct parser selected
  3. Parser runs in QThreadPool via run_in_thread()
  4. On success: app_state.record_loaded emitted; panels refresh
  5. On error:   QMessageBox shows the error message

Architecture: Presentation layer — may import core/, ui/, parsers/.
              Uses only upward-permitted imports (LAW 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QToolBar,
    QWidget,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from core.app_state import app_state
from core.thread_manager import run_in_thread
from models.disturbance_record import DisturbanceRecord
from parsers.comtrade_parser import ComtradeParser
from parsers.csv_parser import CsvParser
from parsers.excel_parser import ExcelParser
from parsers.parser_exceptions import NeedsMappingDialog, NeedsSheetSelection
from parsers.pmu_csv_parser import PmuCsvParser, is_pmu_csv
from engine.decimator import prepare_display_data
from ui.channel_canvas import ChannelCanvas
from ui.measurement_panel import MeasurementPanel
from ui.rms_converter_dock import RmsConverterDock
from ui.unified_canvas import UnifiedCanvasWidget
from ui.waveform_panel import LabelPanel

# ── Module constants ──────────────────────────────────────────────────────────

WINDOW_TITLE: str   = "PowerWave Analyst"
MIN_WIDTH:    int   = 1280
MIN_HEIGHT:   int   = 800
MEASURE_DOCK_MIN_WIDTH: int = 240

FILE_FILTER: str = (
    "Disturbance Records (*.cfg *.CFG *.csv *.CSV *.xlsx *.xls);;"
    "COMTRADE (*.cfg *.CFG);;"
    "CSV Files (*.csv *.CSV);;"
    "Excel Files (*.xlsx *.xls);;"
    "All Files (*)"
)

_COMTRADE_EXT: frozenset[str] = frozenset({'.cfg', '.dat'})
_CSV_EXT:      frozenset[str] = frozenset({'.csv'})
_EXCEL_EXT:    frozenset[str] = frozenset({'.xlsx', '.xls', '.xlsm'})


class MainWindow(QMainWindow):
    """Top-level application window for PowerWave Analyst.

    Holds the combined LabelPanel + ChannelCanvas scroll view, menu bar,
    and status bar.  File loading is dispatched to a background thread;
    results arrive via app_state signals.
    """

    def __init__(self) -> None:
        """Initialise the window layout, menus, and signal connections."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._record: DisturbanceRecord | None = None

        self._setup_central_widget()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_measurement_dock()
        self._setup_rms_converter_dock()
        self._connect_signals()

        self.statusBar().showMessage("Ready — open a disturbance record to begin.")

    # ── Layout setup ──────────────────────────────────────────────────────────

    def _setup_central_widget(self) -> None:
        """Create the tab widget with Waveform and Unified Canvas tabs.

        Tab 0 — Waveform: original LabelPanel + ChannelCanvas scroll view.
        Tab 1 — Unified Canvas: multi-file, multi-stack overlay view.
        """
        self._label_panel = LabelPanel()
        self._canvas = ChannelCanvas()

        # ── Tab 0: Waveform ───────────────────────────────────────────────────
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)
        h_layout.addWidget(self._label_panel)
        h_layout.addWidget(self._canvas, stretch=1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # ── Tab 1: Unified Canvas ─────────────────────────────────────────────
        self._unified_canvas = UnifiedCanvasWidget()

        # ── Tab widget ────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.addTab(scroll_area, 'Waveform')
        tabs.addTab(self._unified_canvas, 'Unified Canvas')
        self.setCentralWidget(tabs)

    def _setup_menu(self) -> None:
        """Build the File and Tools menus."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a disturbance record file")
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit PowerWave Analyst")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menu_bar.addMenu("&Tools")

        self._rms_toggle_action = QAction("&RMS Converter", self)
        self._rms_toggle_action.setShortcut("Ctrl+R")
        self._rms_toggle_action.setStatusTip("Open / close the RMS Converter (Ctrl+R)")
        self._rms_toggle_action.setCheckable(True)
        self._rms_toggle_action.triggered.connect(self._toggle_rms_dock)
        tools_menu.addAction(self._rms_toggle_action)

    def _setup_toolbar(self) -> None:
        """Add navigation toolbar with zoom buttons.

        Keyboard shortcuts:
          Ctrl+0  — Zoom to Fit (full record)
          Ctrl+F  — Zoom to Fault (trigger ±200 ms)
          +       — Zoom In 50%
          -       — Zoom Out 100%
        """
        toolbar: QToolBar = self.addToolBar("Navigation")
        toolbar.setMovable(False)

        fit_action = QAction("Fit", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.setStatusTip("Zoom to fit full record (Ctrl+0)")
        fit_action.triggered.connect(self._canvas.zoom_to_fit)
        toolbar.addAction(fit_action)

        fault_action = QAction("Fault ±200ms", self)
        fault_action.setShortcut("Ctrl+F")
        fault_action.setStatusTip("Zoom to trigger ±200 ms (Ctrl+F)")
        fault_action.triggered.connect(self._canvas.zoom_to_fault)
        toolbar.addAction(fault_action)

        toolbar.addSeparator()

        zoomin_action = QAction("Zoom In", self)
        zoomin_action.setShortcut("+")
        zoomin_action.setStatusTip("Zoom in 50% (+)")
        zoomin_action.triggered.connect(self._canvas.zoom_in)
        toolbar.addAction(zoomin_action)

        zoomout_action = QAction("Zoom Out", self)
        zoomout_action.setShortcut("-")
        zoomout_action.setStatusTip("Zoom out 100% (-)")
        zoomout_action.triggered.connect(self._canvas.zoom_out)
        toolbar.addAction(zoomout_action)

        toolbar.addSeparator()

        autofit_action = QAction("Auto-fit All", self)
        autofit_action.setStatusTip("Fit all analogue Y-axes to visible data")
        autofit_action.triggered.connect(self._canvas.autofit_all_channels)
        toolbar.addAction(autofit_action)

    def _setup_measurement_dock(self) -> None:
        """Create the right-side measurement dock widget."""
        self._measurement_panel = MeasurementPanel()

        dock = QDockWidget("Measurements", self)
        dock.setWidget(self._measurement_panel)
        dock.setMinimumWidth(MEASURE_DOCK_MIN_WIDTH)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_rms_converter_dock(self) -> None:
        """Create the RMS Converter dock (hidden by default)."""
        self._rms_dock = RmsConverterDock(parent=self)
        self._rms_dock.setMinimumHeight(300)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._rms_dock)
        self._rms_dock.hide()
        self._rms_dock.visibilityChanged.connect(self._rms_toggle_action.setChecked)

    def _toggle_rms_dock(self, checked: bool) -> None:
        """Show or hide the RMS Converter dock.

        Args:
            checked: True = show, False = hide.
        """
        if checked:
            self._rms_dock.show()
        else:
            self._rms_dock.hide()

    def _connect_signals(self) -> None:
        """Wire app_state signals to panel/canvas slots."""
        app_state.record_loaded.connect(self._label_panel.load_record)
        app_state.record_loaded.connect(self._canvas.load_record)
        app_state.record_loaded.connect(self._on_record_loaded_status)
        app_state.record_loaded.connect(self._on_record_stored)
        app_state.channel_toggled.connect(self._canvas.update_channel_visibility)
        app_state.cursor_moved.connect(self._on_cursor_moved)
        self._label_panel.y_scale_requested.connect(self._canvas.scale_y_channel)
        self._label_panel.y_reset_requested.connect(self._canvas.reset_y_channel)
        self._label_panel.y_autofit_requested.connect(self._canvas.autofit_y_channel)

    # ── Cursor / record slots ──────────────────────────────────────────────────

    def _on_record_stored(self, record: DisturbanceRecord) -> None:
        """Store the current record and populate the measurement panel at load time.

        Called after canvas.load_record (signal connection order), so cursor
        initial positions are already set and get_cursor_time() is valid.

        Args:
            record: The freshly loaded DisturbanceRecord.
        """
        self._record = record
        t_a = self._canvas.get_cursor_time(0)
        t_b = self._canvas.get_cursor_time(1)
        self._measurement_panel.refresh(record, t_a, t_b)
        self._label_panel.update_values(record, t_a)

    def _on_cursor_moved(self, cursor_id: int, time_display: float) -> None:
        """Update measurement panel and label panel when a cursor moves.

        Args:
            cursor_id:    0 = cursor A, 1 = cursor B.
            time_display: New cursor position in display units.
        """
        if self._record is None:
            return
        if cursor_id == 0:
            t_a = time_display
            t_b = self._canvas.get_cursor_time(1)
        else:
            t_a = self._canvas.get_cursor_time(0)
            t_b = time_display
        self._measurement_panel.refresh(self._record, t_a, t_b)
        if cursor_id == 0:
            self._label_panel.update_values(self._record, time_display)

    # ── File open flow ─────────────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        """Show the file-open dialog and dispatch parsing to a background thread."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Disturbance Record",
            "",
            FILE_FILTER,
        )
        if not path_str:
            return

        path = Path(path_str)
        self.statusBar().showMessage(f"Loading {path.name}…")
        self._dispatch_load(path)

    def _dispatch_load(self, path: Path) -> None:
        """Select the correct parser and run it off the UI thread.

        Args:
            path: Path to the file selected by the user.
        """
        ext = path.suffix.lower()

        if ext in _COMTRADE_EXT:
            fn = self._load_comtrade
        elif ext in _CSV_EXT:
            fn = self._load_csv
        elif ext in _EXCEL_EXT:
            fn = self._load_excel
        else:
            self._show_error(
                "Unsupported file type",
                f"Cannot open '{path.name}'.\n"
                "Supported formats: COMTRADE (.cfg), CSV (.csv), "
                "Excel (.xlsx / .xls).",
            )
            return

        run_in_thread(
            fn,
            path,
            on_done=self._on_parse_done,
            on_error=self._on_parse_error,
        )

    # ── Parser shims (run off the UI thread) ──────────────────────────────────

    def _load_comtrade(self, path: Path) -> DisturbanceRecord:
        """Load a COMTRADE .cfg / .dat file pair.

        Args:
            path: Path to the .cfg (or .dat) file.

        Returns:
            Parsed DisturbanceRecord with display arrays pre-computed.
        """
        return prepare_display_data(ComtradeParser().load(path))

    def _load_csv(self, path: Path) -> DisturbanceRecord:
        """Load a CSV file, routing PMU CSV files to PmuCsvParser.

        PMU CSV files (first line starts with "ID: NNN, Station Name: ...")
        are handled by PmuCsvParser.  All other CSV files go through
        CsvParser with auto-detection.

        Args:
            path: Path to the CSV file.

        Returns:
            Parsed DisturbanceRecord.

        Raises:
            RuntimeError: When the file requires manual channel mapping.
        """
        if is_pmu_csv(path):
            return prepare_display_data(PmuCsvParser().load(path))

        try:
            return prepare_display_data(CsvParser().load(path))
        except NeedsMappingDialog as exc:
            n = len(exc.columns)
            raise RuntimeError(
                f"'{path.name}' requires manual channel mapping ({n} columns "
                f"could not be auto-detected).\n\n"
                "The Channel Mapping Dialog will be available in a future "
                "milestone.  For now, please load a COMTRADE (.cfg) file."
            ) from exc

    def _load_excel(self, path: Path) -> DisturbanceRecord:
        """Load an Excel file, auto-selecting the first sheet when multiple.

        Args:
            path: Path to the Excel workbook.

        Returns:
            Parsed DisturbanceRecord.

        Raises:
            RuntimeError: When the file requires manual channel mapping.
        """
        parser = ExcelParser()
        try:
            return prepare_display_data(parser.load(path))
        except NeedsSheetSelection as exc:
            first_sheet = exc.sheet_names[0]
            self._auto_selected_sheet = first_sheet
            try:
                return prepare_display_data(parser.load(path, sheet_name=first_sheet))
            except NeedsMappingDialog as exc2:
                n = len(exc2.columns)
                raise RuntimeError(
                    f"'{path.name}' (sheet: '{first_sheet}') requires manual "
                    f"channel mapping ({n} columns could not be auto-detected).\n\n"
                    "The Channel Mapping Dialog will be available in a future "
                    "milestone.  For now, please load a COMTRADE (.cfg) file."
                ) from exc2
        except NeedsMappingDialog as exc:
            n = len(exc.columns)
            raise RuntimeError(
                f"'{path.name}' requires manual channel mapping ({n} columns "
                f"could not be auto-detected).\n\n"
                "The Channel Mapping Dialog will be available in a future "
                "milestone.  For now, please load a COMTRADE (.cfg) file."
            ) from exc

    # ── Callbacks (run on UI thread via Qt signal dispatch) ───────────────────

    def _on_parse_done(self, record: DisturbanceRecord) -> None:
        """Emit app_state.record_loaded after a successful parse.

        Args:
            record: Successfully parsed DisturbanceRecord.
        """
        sheet_note = ""
        if hasattr(self, '_auto_selected_sheet'):
            sheet_note = f" [sheet: {self._auto_selected_sheet}]"
            del self._auto_selected_sheet

        app_state.record_loaded.emit(record)

        if sheet_note:
            current = self.statusBar().currentMessage()
            self.statusBar().showMessage(current + sheet_note)

    def _on_parse_error(self, error_msg: str) -> None:
        """Show a QMessageBox for parse failures.

        Args:
            error_msg: Human-readable error string from the worker.
        """
        self.statusBar().showMessage("Load failed.")
        self._show_error("Failed to load file", error_msg)

    def _on_record_loaded_status(self, record: DisturbanceRecord) -> None:
        """Update the status bar after a record is loaded.

        Args:
            record: The freshly loaded DisturbanceRecord.
        """
        n_a = record.n_analogue
        n_d = record.n_digital
        mode = record.display_mode
        self.statusBar().showMessage(
            f"{record.file_path.name}  —  "
            f"{n_a} analogue / {n_d} digital  [{mode}  {record.sample_rate:.0f} Hz]"
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _show_error(self, title: str, message: str) -> None:
        """Show a modal error dialog.

        Args:
            title:   Dialog window title.
            message: Human-readable description of the error.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(message)
        box.exec()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the PowerWave Analyst application."""
    app = QApplication(sys.argv)
    app.setApplicationName(WINDOW_TITLE)
    app.setOrganizationName("PowerWave")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
