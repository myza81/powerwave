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
        Help > User Guide (F1), Help > About

Architecture: Presentation layer — may import core/, ui/.
              Uses only upward-permitted imports (LAW 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from core.app_settings import AppSettings
from ui import theme_palette
from ui.help_dialog import HelpDialog
from ui.measurement_panel import MeasurementPanel
from ui.rms_converter_dock import RmsConverterDock
from ui.settings_dialog import SettingsDialog
from ui.unified_canvas import UnifiedCanvasWidget

# ── Module constants ──────────────────────────────────────────────────────────

WINDOW_TITLE: str           = "PowerWave Analyst"
MIN_WIDTH:    int           = 1280
MIN_HEIGHT:   int           = 800
MEASURE_DOCK_MIN_WIDTH: int = 240

_ICONS_DIR = Path(__file__).parent / 'assets' / 'icons'


# ── Theme stylesheet builder ──────────────────────────────────────────────────

def _build_app_stylesheet(theme: str) -> str:
    """Return the complete application-wide QSS for *theme*."""
    p = theme_palette.get(theme)
    # Forward-slash paths required by Qt QSS on all platforms
    check_url = _ICONS_DIR / 'check_white.svg'
    dash_url  = _ICONS_DIR / 'dash_white.svg'
    check_qss = str(check_url).replace('\\', '/')
    dash_qss  = str(dash_url).replace('\\', '/')
    return f"""
/* ── Menus ───────────────────────────────────────────────────────────── */
QMenuBar {{ background-color: {p['bg_sidebar']}; color: {p['text']};
           border-bottom: 1px solid {p['border_dim']}; }}
QMenuBar::item {{ background: transparent; padding: 4px 8px; }}
QMenuBar::item:selected {{ background-color: {p['bg_selected']}; color: #FFFFFF; }}
QMenuBar::item:pressed  {{ background-color: {p['bg_selected']}; }}
QMenu {{ background-color: {p['bg_sidebar']}; color: {p['text']};
         border: 1px solid {p['border_dim']}; }}
QMenu::item {{ padding: 5px 24px 5px 28px; }}
QMenu::item:selected  {{ background-color: {p['bg_selected']}; color: #FFFFFF; }}
QMenu::item:disabled  {{ color: {p['text_dim']}; }}
QMenu::separator      {{ height: 1px; background: {p['border_dim']}; margin: 2px 0; }}

/* ── Dock widgets ────────────────────────────────────────────────────── */
QDockWidget {{ color: {p['text']}; }}
QDockWidget::title {{ background: {p['bg_sidebar']}; color: {p['text']};
                     padding: 4px 6px; border-bottom: 1px solid {p['border_dim']}; }}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent; border: none; padding: 2px; }}

/* ── Status bar ──────────────────────────────────────────────────────── */
QStatusBar {{ background: {p['bg_sidebar']}; color: {p['text_dim']}; }}
QStatusBar::item {{ border: none; }}

/* ── Main window ─────────────────────────────────────────────────────── */
QMainWindow {{ background-color: {p['bg_app']}; }}

/* ── Labels ──────────────────────────────────────────────────────────── */
QLabel {{ color: {p['text']}; background: transparent; }}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{ color: {p['text_bright']}; background: {p['bg_sidebar']};
              border: 1px solid {p['border']}; border-radius: 3px; padding: 4px 10px; }}
QPushButton:hover    {{ background: {p['bg_hover']}; }}
QPushButton:pressed  {{ background: {p['bg_selected']}; color: #FFFFFF; }}
QPushButton:checked  {{ background: {p['bg_selected']}; border-color: {p['accent']};
                        color: #FFFFFF; }}
QPushButton:disabled {{ color: {p['text_dim']}; background: {p['bg_app']};
                        border-color: {p['border_dim']}; }}

/* ── Line / text edits ───────────────────────────────────────────────── */
QLineEdit {{ background: {p['bg_input']}; color: {p['text_input']};
            border: 1px solid {p['border']}; border-radius: 2px; padding: 2px 4px;
            selection-background-color: {p['accent']}; selection-color: #FFFFFF; }}
QLineEdit:focus {{ border-color: {p['accent']}; }}

/* ── Spin boxes ──────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {p['bg_input']}; color: {p['text_input']};
    border: 1px solid {p['border']}; border-radius: 2px; padding: 2px 4px; }}
QSpinBox::up-button,   QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {p['bg_header']}; border: 1px solid {p['border']}; width: 16px; }}
QSpinBox::up-button:hover,   QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {p['bg_hover']}; }}
QSpinBox::up-arrow,   QDoubleSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid {p['text_bright']}; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {p['text_bright']}; }}

/* ── Combo boxes ─────────────────────────────────────────────────────── */
QComboBox {{ background: {p['bg_input']}; color: {p['text_input']};
            border: 1px solid {p['border']}; border-radius: 2px; padding: 2px 6px; }}
QComboBox:hover {{ border-color: {p['accent']}; }}
QComboBox::drop-down {{ border: none; background: {p['bg_header']}; width: 20px; }}
QComboBox QAbstractItemView {{ background: {p['bg_sidebar']}; color: {p['text']};
    selection-background-color: {p['bg_selected']}; color: {p['text']};
    border: 1px solid {p['border']}; outline: none; }}

/* ── Checkboxes ──────────────────────────────────────────────────────── */
QCheckBox {{ color: {p['text']}; spacing: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px;
    border: 1px solid {p['border']}; border-radius: 2px;
    background: {p['bg_input']}; }}
QCheckBox::indicator:checked      {{ background: {p['accent']}; border-color: {p['accent']}; }}
QCheckBox::indicator:indeterminate {{ background: {p['bg_hover']}; border-color: {p['border']}; }}

/* ── Tree / list item checkboxes ─────────────────────────────────────── */
QTreeView::indicator, QListView::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {p['border']}; border-radius: 2px;
    background: {p['bg_input']}; }}
QTreeView::indicator:checked, QListView::indicator:checked {{
    background: {p['accent']}; border-color: {p['accent']};
    image: url({check_qss}); }}
QTreeView::indicator:unchecked, QListView::indicator:unchecked {{
    background: {p['bg_input']}; border-color: {p['border']}; }}
QTreeView::indicator:indeterminate, QListView::indicator:indeterminate {{
    background: {p['bg_selected']}; border-color: {p['accent']};
    image: url({dash_qss}); }}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {{ color: {p['text']}; border: 1px solid {p['border']}; border-radius: 4px;
            margin-top: 8px; font-weight: bold; padding-top: 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px;
                   padding: 0 4px; color: {p['text_dim']}; }}

/* ── Sliders ─────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{ background: {p['bg_header']}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {p['accent']}; border: 1px solid {p['border']};
    width: 12px; height: 12px; border-radius: 6px; margin: -4px 0; }}
QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 2px; }}
QSlider::groove:vertical {{ background: {p['bg_header']}; width: 4px; border-radius: 2px; }}
QSlider::handle:vertical {{ background: {p['accent']}; border: 1px solid {p['border']};
    width: 12px; height: 12px; border-radius: 6px; margin: 0 -4px; }}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical   {{ background: {p['bg_app']}; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: {p['bg_app']}; height: 10px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {p['bg_header']}; border-radius: 4px;
    min-height: 20px; min-width: 20px; }}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{ background: {p['border']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

/* ── Toolbar ─────────────────────────────────────────────────────────── */
QToolBar {{ background: {p['bg_toolbar']}; border: none; spacing: 2px; }}
QToolButton {{ background: transparent; color: {p['text']};
               border: none; padding: 3px; }}
QToolButton:hover {{ background: {p['bg_hover']}; border-radius: 3px; }}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle       {{ background: {p['border_dim']}; }}
QSplitter::handle:hover {{ background: {p['accent']}; }}

/* ── Tab widget ──────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: 1px solid {p['border_dim']}; background: {p['bg_app']}; }}
QTabBar::tab          {{ background: {p['bg_sidebar']}; color: {p['text_dim']};
                        border: 1px solid {p['border_dim']}; padding: 4px 12px; }}
QTabBar::tab:selected {{ background: {p['bg_app']}; color: {p['text_bright']};
                         border-bottom: none; }}
QTabBar::tab:hover    {{ background: {p['bg_hover']}; color: {p['text']}; }}

/* ── Trees and lists ─────────────────────────────────────────────────── */
QListWidget, QTreeWidget {{ background: {p['bg_item']}; color: {p['text']};
                           border: 1px solid {p['border_dim']}; }}
QListWidget::item:selected,
QTreeWidget::item:selected  {{ background: {p['bg_selected']}; color: #FFFFFF; }}
QListWidget::item:hover,
QTreeWidget::item:hover     {{ background: {p['bg_hover']}; }}
QHeaderView::section {{ background: {p['bg_header']}; color: {p['text']};
                        border: none; padding: 2px; }}

/* ── Tables ──────────────────────────────────────────────────────────── */
QTableWidget {{ background: {p['bg_panel']}; color: {p['text_bright']};
               gridline-color: {p['border_dim']}; border: none; }}
QTableWidget::item          {{ padding: 1px; }}
QTableWidget::item:selected {{ background: {p['bg_selected']}; color: #FFFFFF; }}

/* ── Dialogs ─────────────────────────────────────────────────────────── */
QDialog {{ background: {p['bg_dialog']}; color: {p['text']}; }}
"""


def apply_app_theme(theme: str) -> None:
    """Apply the dark/light global stylesheet to the running QApplication."""
    app = QApplication.instance()
    if app:
        app.setStyleSheet(_build_app_stylesheet(theme))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Top-level application window for PowerWave Analyst."""

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
        """Build the File, Edit, Tools, and Help menus."""
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

        help_menu = menu_bar.addMenu("&Help")

        guide_action = QAction("&User Guide", self)
        guide_action.setShortcut("F1")
        guide_action.setStatusTip("Open the PowerWave Analyst user guide (F1)")
        guide_action.triggered.connect(self._open_help)
        help_menu.addAction(guide_action)

        help_menu.addSeparator()

        about_action = QAction("&About PowerWave Analyst", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._open_about)
        help_menu.addAction(about_action)

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
        """Show or hide the RMS Converter dock."""
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

    def _open_help(self, topic: int = 0) -> None:
        """Open the User Guide dialog (Help > User Guide / F1)."""
        dlg = HelpDialog(parent=self, initial_topic=topic)
        dlg.exec()

    def _open_about(self) -> None:
        """Open the User Guide on the Getting Started page (Help > About)."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About PowerWave Analyst",
            "<b>PowerWave Analyst</b><br>"
            "Power system disturbance record analysis tool.<br><br>"
            "Supports COMTRADE, PMU CSV, generic CSV and Excel formats.<br>"
            "100% offline — no network calls, no telemetry.<br><br>"
            "<small>For detailed usage, open Help &gt; User Guide (F1).</small>",
        )

    def _open_preferences(self) -> None:
        """Open the global Preferences dialog (Edit → Preferences / Ctrl+,)."""
        old_threshold = AppSettings.get('calculation.timestamp_grouping_threshold_h', 1.0)
        dlg = SettingsDialog(parent=self)
        dlg.exec()
        theme = AppSettings.get('display.theme', 'dark')
        apply_app_theme(theme)
        self._measurement_panel.apply_settings()
        self._unified_canvas.apply_theme()
        self._rms_dock.apply_theme()
        new_threshold = AppSettings.get('calculation.timestamp_grouping_threshold_h', 1.0)
        if new_threshold != old_threshold:
            self._unified_canvas.regroup_files()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the PowerWave Analyst application."""
    app = QApplication(sys.argv)
    app.setApplicationName(WINDOW_TITLE)
    app.setOrganizationName("PowerWave")
    app.setStyle("Fusion")

    # Apply the saved theme before the window is shown
    apply_app_theme(AppSettings.get('display.theme', 'dark'))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
