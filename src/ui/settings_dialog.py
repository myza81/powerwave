"""
src/ui/settings_dialog.py

Global Preferences dialog for PowerWave Analyst.

Layout:
  ┌─ PowerWave Analyst — Preferences ─────────────────────────────────┐
  │ ┌──────────────┬────────────────────────────────────────────────┐  │
  │ │ Calculation  │  ┌─ Calculation ──────────────────────────┐    │  │
  │ │ Display      │  │  Default nominal frequency  [50 Hz ▼]   │   │  │
  │ │ PMU          │  │  RMS merge tolerance        [10.0] ms   │   │  │
  │ │ About        │  │  PU mode Y-axis range       [2.0]  pu   │   │  │
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

# ── Constants ──────────────────────────────────────────────────────────────────

DIALOG_MIN_W: int = 720
DIALOG_MIN_H: int = 480
NAV_WIDTH:    int = 170

_CATEGORY_ITEMS: list[tuple[str, str]] = [
    ('Calculation',  '⚙'),
    ('Display',      '🎨'),
    ('PMU',          '📡'),
    ('About',        'ℹ'),
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

def _make_section_label(text: str) -> QLabel:
    """Return a bold section heading label."""
    lbl = QLabel(text)
    lbl.setStyleSheet('font-weight: bold; font-size: 11pt; color: #AADDFF;')
    return lbl


def _make_separator() -> QFrame:
    """Return a horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet('color: #444;')
    return line


class _CalculationPage(QWidget):
    """Settings page: Calculation section."""

    def __init__(self, data: dict[str, Any], parent: QWidget | None = None) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``calculation`` section dict from AppSettings.snapshot().
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('Calculation'))
        layout.addWidget(_make_separator())

        group = QGroupBox('Default values for new files')
        group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #CCCCCC; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
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
        ta_group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #CCCCCC; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
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
        tz_note.setStyleSheet('color: #888; font-size: 8pt;')
        ta_form.addRow('', tz_note)

        layout.addWidget(ta_group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'nominal_frequency':    self._freq_combo.currentData(),
            'rms_tolerance_ms':     self._tol_spin.value(),
            'pu_yrange':            self._pu_spin.value(),
            'comtrade_tz_offset_h': self._tz_combo.currentData(),
        }


class _DisplayPage(QWidget):
    """Settings page: Display section."""

    def __init__(self, data: dict[str, Any], parent: QWidget | None = None) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``display`` section dict from AppSettings.snapshot().
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('Display'))
        layout.addWidget(_make_separator())

        # Cursor colours
        cursor_group = QGroupBox('Cursor colours')
        cursor_group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #CCCCCC; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
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

        # Theme (future)
        theme_group = QGroupBox('Theme')
        theme_group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #CCCCCC; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
        theme_form = QFormLayout(theme_group)
        theme_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        theme_form.setVerticalSpacing(10)

        self._theme_combo = QComboBox()
        self._theme_combo.addItem('Dark (default)', userData='dark')
        self._theme_combo.addItem('Light  [coming soon]', userData='light')
        self._theme_combo.model().item(1).setEnabled(False)
        theme_form.addRow('Application theme:', self._theme_combo)

        note = QLabel('Theme change takes effect on next application launch.')
        note.setStyleSheet('color: #888; font-size: 8pt;')
        theme_form.addRow('', note)

        layout.addWidget(theme_group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'theme':            self._theme_combo.currentData(),
            'cursor_c1_colour': self._c1_btn.colour,
            'cursor_c2_colour': self._c2_btn.colour,
        }


class _PmuPage(QWidget):
    """Settings page: PMU section."""

    def __init__(self, data: dict[str, Any], parent: QWidget | None = None) -> None:
        """Populate fields from ``data``.

        Args:
            data:   The ``pmu`` section dict from AppSettings.snapshot().
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('PMU'))
        layout.addWidget(_make_separator())

        group = QGroupBox('PMU CSV Import')
        group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #CCCCCC; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
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
        tz_note.setStyleSheet('color: #888; font-size: 8pt;')
        form.addRow('', tz_note)

        layout.addWidget(group)
        layout.addStretch()

    def collect(self) -> dict[str, Any]:
        """Return the current widget values as a settings dict."""
        return {
            'default_timezone': self._tz_combo.currentText(),
        }


class _AboutPage(QWidget):
    """Settings page: About section."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        layout.addWidget(_make_section_label('About'))
        layout.addWidget(_make_separator())

        title = QLabel('<b>PowerWave Analyst</b>')
        title.setStyleSheet('font-size: 16pt; color: #EEEEFF;')
        layout.addWidget(title)

        subtitle = QLabel('Power System Disturbance Record Analysis')
        subtitle.setStyleSheet('font-size: 10pt; color: #AAAAAA;')
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        info_lines = [
            ('Version',     '2C (Unified Canvas)'),
            ('Platform',    'Windows / macOS'),
            ('Python',      '3.11'),
            ('UI toolkit',  'PyQt6'),
            ('Rendering',   'PyQtGraph + OpenGL'),
            ('Data',        '100 % offline — no network calls ever'),
        ]
        for label, value in info_lines:
            row = QLabel(f'<b>{label}:</b>  {value}')
            row.setStyleSheet('color: #CCCCCC; font-size: 9pt;')
            layout.addWidget(row)

        layout.addStretch()


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
        self._apply_dark_style()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Build the full dialog layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)

        # ── Main splitter (nav | content) ─────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        # Left nav list
        self._nav = QListWidget()
        self._nav.setFixedWidth(NAV_WIDTH)
        self._nav.setFrameShape(QFrame.Shape.NoFrame)
        self._nav.setStyleSheet(
            'QListWidget {'
            '  background: #252535;'
            '  border-right: 1px solid #3A3A4A;'
            '  font-size: 10pt;'
            '}'
            'QListWidget::item {'
            '  padding: 10px 14px;'
            '  color: #BBBBBB;'
            '}'
            'QListWidget::item:selected {'
            '  background: #3A3A5A;'
            '  color: #FFFFFF;'
            '  border-left: 3px solid #6688FF;'
            '}'
            'QListWidget::item:hover:!selected {'
            '  background: #2E2E3E;'
            '}'
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

        self._page_calc    = _CalculationPage(calc_data)
        self._page_display = _DisplayPage(display_data)
        self._page_pmu     = _PmuPage(pmu_data)
        self._page_about   = _AboutPage()

        # Wrap each page in a scroll area so it handles small windows gracefully
        for page in (self._page_calc, self._page_display,
                     self._page_pmu, self._page_about):
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet('background: #1E1E2E;')
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

    def _apply_dark_style(self) -> None:
        """Apply a consistent dark stylesheet to the dialog."""
        self.setStyleSheet(
            'QDialog { background: #1E1E2E; }'
            'QLabel  { color: #CCCCCC; }'
            'QGroupBox { color: #CCCCCC; }'
            'QComboBox, QDoubleSpinBox, QSpinBox {'
            '  background: #2A2A3A; color: #EEEEEE;'
            '  border: 1px solid #555; border-radius: 3px; padding: 2px 4px;'
            '}'
            'QComboBox::drop-down { border: none; }'
            'QComboBox QAbstractItemView { background: #2A2A3A; color: #EEEEEE; '
            '  selection-background-color: #3A3A6A; }'
            'QPushButton {'
            '  background: #2E2E4E; color: #DDDDDD;'
            '  border: 1px solid #555; border-radius: 4px; padding: 5px 12px;'
            '}'
            'QPushButton:hover   { background: #3A3A5E; }'
            'QPushButton:pressed { background: #4A4A7E; }'
            'QScrollArea, QScrollBar { background: #1E1E2E; }'
            'QSplitter::handle { background: #3A3A4A; }'
            'QDialogButtonBox QPushButton { min-width: 72px; }'
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

        # Rebuild pages in-place
        idx = self._stack.currentIndex()

        self._page_calc    = _CalculationPage(defaults.get('calculation', {}))
        self._page_display = _DisplayPage(defaults.get('display', {}))
        self._page_pmu     = _PmuPage(defaults.get('pmu', {}))
        self._page_about   = _AboutPage()

        # Replace scroll-area widgets
        for stack_idx, page in enumerate((
            self._page_calc, self._page_display,
            self._page_pmu, self._page_about,
        )):
            old_scroll = self._stack.widget(stack_idx)
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet('background: #1E1E2E;')
            self._stack.removeWidget(old_scroll)
            old_scroll.deleteLater()
            self._stack.insertWidget(stack_idx, scroll)

        self._stack.setCurrentIndex(idx)
