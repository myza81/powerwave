"""
src/main.py

PowerWave Analyst — application entry point.

MainWindow layout:
  Central widget: UnifiedCanvasWidget (multi-file, multi-stack analogue canvas)
  Right dock:     MeasurementPanel (cursor times + per-channel value table)
  Bottom dock:    RmsConverterDock (hidden by default, toggle via Tools menu)

  Menu: File > Open (Ctrl+O), File > Exit
        Edit > Preferences (Ctrl+,)
        Tools > RMS Converter (Ctrl+R)

Architecture: Presentation layer — may import core/, ui/.
              Uses only upward-permitted imports (LAW 1).
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from ui.measurement_panel import MeasurementPanel
from ui.rms_converter_dock import RmsConverterDock
from ui.settings_dialog import SettingsDialog
from ui.unified_canvas import UnifiedCanvasWidget

# ── Module constants ──────────────────────────────────────────────────────────

WINDOW_TITLE: str           = "PowerWave Analyst"
MIN_WIDTH:    int           = 1280
MIN_HEIGHT:   int           = 800
MEASURE_DOCK_MIN_WIDTH: int = 240


class MainWindow(QMainWindow):
    """Top-level application window for PowerWave Analyst.

    Holds the Unified Canvas as the central widget, a right-side
    measurement dock, and an optional RMS Converter bottom dock.
    """

    def __init__(self) -> None:
        """Initialise the window layout, menus, and signal connections."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._setup_central_widget()
        self._setup_menu()
        self._setup_measurement_dock()
        self._setup_rms_converter_dock()
        self._connect_signals()

        self.statusBar().showMessage("Ready — open a disturbance record to begin.")

    # ── Layout setup ──────────────────────────────────────────────────────────

    def _setup_central_widget(self) -> None:
        """Set the Unified Canvas as the sole central widget."""
        self._unified_canvas = UnifiedCanvasWidget()
        self.setCentralWidget(self._unified_canvas)

    def _setup_menu(self) -> None:
        """Build the File, Edit, and Tools menus."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Add a disturbance record file to the canvas")
        open_action.triggered.connect(self._unified_canvas.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit PowerWave Analyst")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("&Edit")

        prefs_action = QAction("&Preferences…", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.setStatusTip("Open application preferences (Ctrl+,)")
        prefs_action.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_action)

        tools_menu = menu_bar.addMenu("&Tools")

        self._rms_toggle_action = QAction("&RMS Converter", self)
        self._rms_toggle_action.setShortcut("Ctrl+R")
        self._rms_toggle_action.setStatusTip("Open / close the RMS Converter (Ctrl+R)")
        self._rms_toggle_action.setCheckable(True)
        self._rms_toggle_action.triggered.connect(self._toggle_rms_dock)
        tools_menu.addAction(self._rms_toggle_action)

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
        """Wire Unified Canvas signals to panel slots."""
        self._unified_canvas.readout_updated.connect(
            self._measurement_panel.update_readout
        )

    # ── Preferences ───────────────────────────────────────────────────────────

    def _open_preferences(self) -> None:
        """Open the global Preferences dialog (Edit → Preferences / Ctrl+,)."""
        dlg = SettingsDialog(parent=self)
        dlg.exec()
        self._measurement_panel.apply_settings()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the PowerWave Analyst application."""
    app = QApplication(sys.argv)
    app.setApplicationName(WINDOW_TITLE)
    app.setOrganizationName("PowerWave")
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QPushButton {"
        "  color: #DDDDDD;"
        "  background: #2E2E3E;"
        "  border: 1px solid #555555;"
        "  border-radius: 3px;"
        "  padding: 4px 10px;"
        "}"
        "QPushButton:hover   { background: #3A3A5A; }"
        "QPushButton:pressed { background: #4A4A7A; }"
        "QPushButton:disabled { color: #666666; background: #252535; }"
    )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
