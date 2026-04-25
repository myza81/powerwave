"""
src/ui/settings_dialog.py

Global Preferences dialog for PowerWave Analyst.

Layout:
  ┌─ PowerWave Analyst — Preferences ─────────────────────────────────┐
  │ ┌──────────────┬────────────────────────────────────────────────┐  │
  │ │ Calculation  │  ┌─ Calculation ──────────────────────────┐    │  │
  │ │ Display      │  │  Default nominal frequency  [50 Hz ▼]   │   │  │
  │ │ PMU          │  │  RMS merge tolerance        [10.0] ms   │   │  │
  │ │              │  │  PU mode Y-axis range       [2.0]  pu   │   │  │
  │ │              │  └────────────────────────────────────────┘    │  │
  │ └──────────────┴────────────────────────────────────────────────┘  │
  │          [Restore Defaults]        [Cancel]  [Apply]  [OK]        │
  └────────────────────────────────────────────────────────────────────┘

Architecture: Presentation layer (ui/) — reads/writes core.AppSettings only.
              No direct engine/ imports (LAW 1).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import AppSettings
from ui import theme_palette

# ── Constants ──────────────────────────────────────────────────────────────────

DIALOG_MIN_W: int = 720
DIALOG_MIN_H: int = 480
NAV_WIDTH:    int = 170

_CATEGORY_ITEMS: list[tuple[str, str]] = [
    ('Calculation',  '⚙'),
    ('Display',      '🎨'),
    ('PMU',          '📡'),
]

_FREQ_OPTIONS:  list[int]  = [50, 60]
_TZ_OPTIONS:   list[str]  = ['SGT (UTC+8)', 'MYT (UTC+8)', 'UTC']


# ── Colour picker helper ───────────────────────────────────────────────────────

class _ColourButton(QWidget):
    """Small inline widget: a colour swatch button + hex label.

    Clicking the swatch opens QColorDialog.  The ``colour`` property
    returns the currently selected hex string (e.g. ``'#FFD700'``).
    """

    def __init__(self, initial_hex: str, parent: QWidget | None = None) -> None:
        """Initialise the colour button.

        Args:
            initial_hex: Starting colour as ``'#RRGGBB'``.
            parent:      Optional parent widget.
        """
        super().__init__(parent)
        self._colour = initial_hex
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._swatch = QPushButton()
        self._swatch.setFixedSize(28, 22)
        self._swatch.setToolTip('Click to change colour')
        self._swatch.clicked.connect(self._pick_colour)

        self._hex_label = QLabel()
        self._hex_label.setFixedWidth(68)

        layout.addWidget(self._swatch)
        layout.addWidget(self._hex_label)
        layout.addStretch()

        self._refresh_swatch()

    @property
    def colour(self) -> str:
        """Currently selected colour as a ``'#RRGGBB'`` string."""
        return self._colour

    @colour.setter
    def colour(self, value: str) -> None:
        self._colour = value
        self._refresh_swatch()

    def _refresh_swatch(self) -> None:
        px = QPixmap(24, 18)
        px.fill(QColor(self._colour))
        icon = QIcon(px)
        self._swatch.setIcon(icon)
        self._swatch.setIconSize(px.size())
        self._hex_label.setText(self._colour.upper())

    def _pick_colour(self) -> None:
        chosen = QColorDialog.getColor(
            QColor(self._colour), self, 'Choose Colour')
        if chosen.isValid():
            self.colour = chosen.name().upper()


# ── Settings pages ─────────────────────────────────────────────────────────────

def _make_section_label(text: str, p: dict) -> QLabel:
    """Return a bold section heading label styled for *p* (palette)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f'font-weight: bold; font-size: 11pt; color: {p["text_accent"]};'
    )
    return lbl


def _make_separator(p: dict) -> QFrame:
    """Return a horizontal separator line styled for *p* (palette)."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f'color: {p["sep_line"]};')
    return line


class _CalculationPage(QWidget):
    """Settings page: Calculation section."""

    def __init__(
        self,
        data:   dict[str, Any],
        theme:  str = 'dark',
        parent: QWidget | None = None,
    ) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``calculation`` section dict from AppSettings.snapshot().
            theme:  Active theme name (``'dark'`` or ``'light'``).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        p = theme_palette.get(theme)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('Calculation', p))
        layout.addWidget(_make_separator(p))

        group = QGroupBox('Default values for new files')
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(16)

        # Nominal frequency
        self._freq_combo = QComboBox()
        for f in _FREQ_OPTIONS:
            self._freq_combo.addItem(f'{f} Hz', userData=f)
        current_freq = data.get('nominal_frequency', 50)
        idx = self._freq_combo.findData(current_freq)
        self._freq_combo.setCurrentIndex(max(0, idx))
        form.addRow('Default nominal frequency:', self._freq_combo)

        # RMS tolerance
        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(1.0, 500.0)
        self._tol_spin.setDecimals(1)
        self._tol_spin.setSuffix('  ms')
        self._tol_spin.setValue(data.get('rms_tolerance_ms', 10.0))
        self._tol_spin.setToolTip(
            'Maximum time gap (ms) between samples from different files\n'
            'to be considered simultaneous during RMS merge.'
        )
        form.addRow('RMS merge tolerance:', self._tol_spin)

        # PU Y-range
        self._pu_spin = QDoubleSpinBox()
        self._pu_spin.setRange(0.5, 5.0)
        self._pu_spin.setDecimals(2)
        self._pu_spin.setSingleStep(0.1)
        self._pu_spin.setSuffix('  pu')
        self._pu_spin.setValue(data.get('pu_yrange', 2.0))
        self._pu_spin.setToolTip(
            'Symmetric Y-axis range when PU mode is active.\n'
            'E.g. 2.0 → Y-axis shows −2.0 pu to +2.0 pu.'
        )
        form.addRow('PU mode Y-axis range (±):', self._pu_spin)

        layout.addWidget(group)

        # ── Time alignment group ──────────────────────────────────────────────
        ta_group = QGroupBox('Time alignment')
        ta_form = QFormLayout(ta_group)
        ta_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        ta_form.setVerticalSpacing(10)
        ta_form.setHorizontalSpacing(16)

        self._tz_combo = QComboBox()
        _tz_items = [
            ('UTC  (0)',                    0),
            ('MYT / SGT  (UTC+8)',          8),
            ('WIB — Indonesia West  (UTC+7)', 7),
            ('ICT — Thailand / Vietnam  (UTC+7)', 7),
            ('JST / KST  (UTC+9)',          9),
            ('IST — India  (UTC+5:30)',     5.5),
            ('CET  (UTC+1)',                1),
            ('CEST  (UTC+2)',               2),
            ('EST  (UTC-5)',               -5),
            ('PST  (UTC-8)',               -8),
        ]
        for label, offset in _tz_items:
            self._tz_combo.addItem(label, userData=offset)
        stored = data.get('comtrade_tz_offset_h', 0)
        idx = self._tz_combo.findData(stored)
        self._tz_combo.setCurrentIndex(max(0, idx))
        self._tz_combo.setToolTip(
            'Timezone of timestamps stored in COMTRADE CFG files.\n'
            'COMTRADE stores local time with no timezone field.\n\n'
            'PMU CSV files are always converted from SGT to UTC automatically.\n'
            'Set this to match where your COMTRADE files were recorded so\n'
            'COMTRADE and PMU waveforms align on the shared time axis.\n\n'
            'Example: Malaysian BEN32 files → MYT / SGT (UTC+8)'
        )
        ta_form.addRow('COMTRADE timestamp timezone:', self._tz_combo)

        tz_note = QLabel(
            'COMTRADE stores local wall-clock time. Select the timezone of the\n'
            'substation so timestamps align correctly with PMU CSV files (UTC).'
        )
        tz_note.setWordWrap(True)
        tz_note.setStyleSheet(f'color: {p["text_dim"]}; font-size: 8pt;')
        ta_form.addRow('', tz_note)

        layout.addWidget(ta_group)

        # ── File grouping group ───────────────────────────────────────────────
        grp_group = QGroupBox('File grouping')
        grp_form = QFormLayout(grp_group)
        grp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        grp_form.setVerticalSpacing(10)
        grp_form.setHorizontalSpacing(16)

        self._grouping_spin = QDoubleSpinBox()
        self._grouping_spin.setRange(0.1, 48.0)
        self._grouping_spin.setDecimals(1)
        self._grouping_spin.setSingleStep(0.5)
        self._grouping_spin.setSuffix('  h')
        self._grouping_spin.setValue(data.get('timestamp_grouping_threshold_h', 1.0))
        self._grouping_spin.setToolTip(
            'Files whose start timestamps differ by less than this value (hours)\n'
            'are plotted together on a shared time axis.\n'
            'Files outside this window get their own independent canvas section.'
        )
        grp_form.addRow('Timestamp grouping threshold:', self._grouping_spin)

        layout.addWidget(grp_group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'nominal_frequency':              self._freq_combo.currentData(),
            'rms_tolerance_ms':               self._tol_spin.value(),
            'pu_yrange':                      self._pu_spin.value(),
            'comtrade_tz_offset_h':           self._tz_combo.currentData(),
            'timestamp_grouping_threshold_h': self._grouping_spin.value(),
        }


class _DisplayPage(QWidget):
    """Settings page: Display section."""

    def __init__(
        self,
        data:   dict[str, Any],
        theme:  str = 'dark',
        parent: QWidget | None = None,
    ) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``display`` section dict from AppSettings.snapshot().
            theme:  Active theme name (``'dark'`` or ``'light'``).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        p = theme_palette.get(theme)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('Display', p))
        layout.addWidget(_make_separator(p))

        # Cursor colours
        cursor_group = QGroupBox('Cursor colours')
        cursor_form = QFormLayout(cursor_group)
        cursor_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        cursor_form.setVerticalSpacing(10)
        cursor_form.setHorizontalSpacing(16)

        self._c1_btn = _ColourButton(data.get('cursor_c1_colour', '#FFD700'))
        self._c1_btn.setToolTip('Cursor C1 colour (default: gold)')
        cursor_form.addRow('Cursor C1:', self._c1_btn)

        self._c2_btn = _ColourButton(data.get('cursor_c2_colour', '#00E5FF'))
        self._c2_btn.setToolTip('Cursor C2 colour (default: cyan)')
        cursor_form.addRow('Cursor C2:', self._c2_btn)

        layout.addWidget(cursor_group)

        # Panel font size
        font_group = QGroupBox('Panel text')
        font_form = QFormLayout(font_group)
        font_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        font_form.setVerticalSpacing(10)
        font_form.setHorizontalSpacing(16)

        self._font_spin = QSpinBox()
        self._font_spin.setRange(7, 14)
        self._font_spin.setSuffix('  pt')
        self._font_spin.setValue(data.get('panel_font_size', 9))
        self._font_spin.setToolTip(
            'Font size for labels and values in the Measurements panel.\n'
            'Takes effect immediately when you click Apply or OK.'
        )
        font_form.addRow('Measurements panel font size:', self._font_spin)

        layout.addWidget(font_group)

        # Theme
        theme_group = QGroupBox('Theme')
        theme_form = QFormLayout(theme_group)
        theme_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        theme_form.setVerticalSpacing(10)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem('Dark',  userData='dark')
        self._theme_combo.addItem('Light', userData='light')
        idx = self._theme_combo.findData(data.get('theme', 'dark'))
        self._theme_combo.setCurrentIndex(max(0, idx))
        theme_form.addRow('Application theme:', self._theme_combo)

        note = QLabel('Theme change takes effect immediately after Apply / OK.')
        note.setStyleSheet(f'color: {p["text_dim"]}; font-size: 8pt;')
        theme_form.addRow('', note)

        layout.addWidget(theme_group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'theme':            self._theme_combo.currentData(),
            'cursor_c1_colour': self._c1_btn.colour,
            'cursor_c2_colour': self._c2_btn.colour,
            'panel_font_size':  self._font_spin.value(),
        }


class _PmuPage(QWidget):
    """Settings page: PMU section."""

    def __init__(
        self,
        data:   dict[str, Any],
        theme:  str = 'dark',
        parent: QWidget | None = None,
    ) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``pmu`` section dict from AppSettings.snapshot().
            theme:  Active theme name (``'dark'`` or ``'light'``).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        p = theme_palette.get(theme)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('PMU', p))
        layout.addWidget(_make_separator(p))

        group = QGroupBox('PMU CSV Import')
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(16)

        self._tz_combo = QComboBox()
        for tz in _TZ_OPTIONS:
            self._tz_combo.addItem(tz)
        stored_tz = data.get('default_timezone', 'SGT (UTC+8)')
        idx = self._tz_combo.findText(stored_tz)
        self._tz_combo.setCurrentIndex(max(0, idx))
        self._tz_combo.setToolTip(
            'Default timezone assumed when importing a PMU CSV file\n'
            'that does not contain explicit timezone information.\n'
            'The selected timezone is always converted to UTC internally.'
        )
        form.addRow('Default timezone:', self._tz_combo)

        tz_note = QLabel(
            'PMU timestamps are always converted to UTC for time-axis alignment.\n'
            'This setting only affects the pre-populated value in the import dialog.'
        )
        tz_note.setWordWrap(True)
        tz_note.setStyleSheet(f'color: {p["text_dim"]}; font-size: 8pt;')
        form.addRow('', tz_note)

        layout.addWidget(group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'default_timezone': self._tz_combo.currentText(),
        }



# ── Main dialog ────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Two-panel modal Preferences dialog.

    Left panel  — category navigation list.
    Right panel — QStackedWidget with one page per category.

    Changes are staged in memory until the user clicks OK or Apply.
    Cancel discards all changes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Open the dialog, loading current settings from AppSettings."""
        super().__init__(parent)
        self.setWindowTitle('PowerWave Analyst — Preferences')
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)
        self.setModal(True)

        # Load a snapshot; pages edit this in-memory copy
        self._snapshot = AppSettings.snapshot()

        self._setup_ui()
        current_theme = self._snapshot.get('display', {}).get('theme', 'dark')
        self._apply_style(current_theme)

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Build the full dialog layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)

        current_theme = self._snapshot.get('display', {}).get('theme', 'dark')
        p = theme_palette.get(current_theme)

        # ── Main splitter (nav | content) ─────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        # Left nav list
        self._nav = QListWidget()
        self._nav.setFixedWidth(NAV_WIDTH)
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setStyleSheet(
            f'QListWidget {{ background: {p["bg_sidebar"]}; border-right: 1px solid {p["border_dim"]}; font-size: 10pt; }}'
            f'QListWidget::item {{ padding: 10px 14px; color: {p["text"]}; }}'
            f'QListWidget::item:selected {{ background: {p["bg_selected"]}; color: #FFFFFF; border-left: 3px solid {p["accent"]}; }}'
            f'QListWidget::item:hover:!selected {{ background: {p["bg_hover"]}; }}'
        )
        for label, icon in _CATEGORY_ITEMS:
            item = QListWidgetItem(f'  {icon}  {label}')
            item.setSizeHint(item.sizeHint().__class__(NAV_WIDTH, 42))
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_category_changed)

        # Right stacked content
        self._stack = QStackedWidget()

        calc_data    = self._snapshot.get('calculation', {})
        display_data = self._snapshot.get('display', {})
        pmu_data     = self._snapshot.get('pmu', {})

        self._page_calc    = _CalculationPage(calc_data, theme=current_theme)
        self._page_display = _DisplayPage(display_data, theme=current_theme)
        self._page_pmu     = _PmuPage(pmu_data, theme=current_theme)

        # Wrap each page in a scroll area so it handles small windows gracefully
        for page in (self._page_calc, self._page_display, self._page_pmu):
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet(f'background: {p["bg_dialog"]};')
            self._stack.addWidget(scroll)

        splitter.addWidget(self._nav)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

        # ── Bottom button bar ──────────────────────────────────────────────────
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(16, 8, 16, 0)
        btn_bar.setSpacing(8)

        restore_btn = QPushButton('Restore Defaults')
        restore_btn.setToolTip('Reset all settings to factory defaults')
        restore_btn.clicked.connect(self._on_restore_defaults)

        btn_bar.addWidget(restore_btn)
        btn_bar.addStretch()

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply)

        btn_bar.addWidget(box)
        root.addLayout(btn_bar)

        # Select first category
        self._nav.setCurrentRow(0)

    def _apply_style(self, theme: str) -> None:
        """Apply a theme-consistent stylesheet to the dialog."""
        p = theme_palette.get(theme)
        self.setStyleSheet(
            f'QDialog {{ background: {p["bg_dialog"]}; }}'
            f'QLabel   {{ color: {p["text"]}; }}'
            f'QGroupBox {{ color: {p["text"]}; border: 1px solid {p["border"]}; border-radius: 4px; margin-top: 8px; padding-top: 4px; }}'
            f'QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {p["text_dim"]}; }}'
            f'QComboBox, QDoubleSpinBox, QSpinBox {{'
            f'  background: {p["bg_input"]}; color: {p["text_input"]};'
            f'  border: 1px solid {p["border"]}; border-radius: 3px; padding: 2px 4px;'
            f'}}'
            f'QComboBox::drop-down {{ border: none; background: {p["bg_header"]}; width: 20px; }}'
            f'QComboBox QAbstractItemView {{ background: {p["bg_sidebar"]}; color: {p["text"]}; '
            f'  selection-background-color: {p["bg_selected"]}; border: 1px solid {p["border"]}; }}'
            f'QSpinBox::up-button, QDoubleSpinBox::up-button,'
            f'QSpinBox::down-button, QDoubleSpinBox::down-button {{'
            f'  background: {p["bg_header"]}; border: 1px solid {p["border"]}; width: 16px; }}'
            f'QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,'
            f'QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{'
            f'  background: {p["bg_hover"]}; }}'
            f'QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{'
            f'  width: 0; height: 0;'
            f'  border-left: 4px solid transparent; border-right: 4px solid transparent;'
            f'  border-bottom: 5px solid {p["text_bright"]}; }}'
            f'QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{'
            f'  width: 0; height: 0;'
            f'  border-left: 4px solid transparent; border-right: 4px solid transparent;'
            f'  border-top: 5px solid {p["text_bright"]}; }}'
            f'QPushButton {{'
            f'  background: {p["bg_sidebar"]}; color: {p["text_bright"]};'
            f'  border: 1px solid {p["border"]}; border-radius: 4px; padding: 5px 12px;'
            f'}}'
            f'QPushButton:hover   {{ background: {p["bg_hover"]}; }}'
            f'QPushButton:pressed {{ background: {p["bg_selected"]}; color: #FFFFFF; }}'
            f'QScrollArea, QScrollBar {{ background: {p["bg_dialog"]}; }}'
            f'QSplitter::handle {{ background: {p["border_dim"]}; }}'
            f'QDialogButtonBox QPushButton {{ min-width: 72px; }}'
        )

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_category_changed(self, row: int) -> None:
        """Switch the right panel to the page at ``row``.

        Args:
            row: Selected row index in the navigation list.
        """
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    def _collect_all(self) -> dict[str, dict[str, Any]]:
        """Gather current values from all pages into a nested settings dict."""
        return {
            'calculation': self._page_calc.collect(),
            'display':     self._page_display.collect(),
            'pmu':         self._page_pmu.collect(),
        }

    def _on_apply(self) -> None:
        """Write staged values to AppSettings and save to disk."""
        AppSettings.apply_snapshot(self._collect_all())

    def _on_ok(self) -> None:
        """Apply changes and close the dialog."""
        self._on_apply()
        self.accept()

    def _on_restore_defaults(self) -> None:
        """Rebuild all pages with factory-default values."""
        from core.app_settings import _DEFAULTS  # noqa: PLC0415
        defaults = {s: dict(v) for s, v in _DEFAULTS.items()}

        current_theme = self._snapshot.get('display', {}).get('theme', 'dark')
        p = theme_palette.get(current_theme)

        # Rebuild pages in-place
        idx = self._stack.currentIndex()

        self._page_calc    = _CalculationPage(defaults.get('calculation', {}), theme=current_theme)
        self._page_display = _DisplayPage(defaults.get('display', {}), theme=current_theme)
        self._page_pmu     = _PmuPage(defaults.get('pmu', {}), theme=current_theme)

        # Replace scroll-area widgets
        for stack_idx, page in enumerate((
            self._page_calc, self._page_display, self._page_pmu,
        )):
            old_scroll = self._stack.widget(stack_idx)
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet(f'background: {p["bg_dialog"]};')
            self._stack.removeWidget(old_scroll)
            old_scroll.deleteLater()
            self._stack.insertWidget(stack_idx, scroll)

        self._stack.setCurrentIndex(idx)
