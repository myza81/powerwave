"""
src/ui/unified_canvas.py

Unified Canvas — multi-file, multi-stack analogue waveform overlay tab.

Layout:
  ┌─ Toolbar ──────────────────────────────────────────────────────────┐
  │ [Add File] [Remove]  [PU Mode]  [Phasor Display]                   │
  ├─ Left (300px) ──────────┬─ Right ─────────────────────────────────┤
  │ File tree                │  Stack 1 — Voltage (L) | Current (R)   │
  │  ▶ FILE1.cfg             │  ─────────────────────────────────────  │
  │    ☑ VR    [Raw   ]      │  Stack 2 — Freq (L) | Power (R)        │
  │    ☑ IY    [Raw   ]      │  ─────────────────────────────────────  │
  │    ☑ FREQ  [Value—]      │  ...                                    │
  │  ▶ PMU.csv               │  ─────────────────────────────────────  │
  │    ☑ V1    [RMS  —]      │  Time axis (s)                          │
  ├──────────────────────────┴─────────────────────────────────────────┤
  │ Offset: FILE1 [-][━━━●━━━][+] 0.0 ms   FILE2 [-][...][+]          │
  └────────────────────────────────────────────────────────────────────┘

Data flow:
  1. User adds files → parsed on background thread → DisturbanceRecord
  2. Channels auto-assigned to stacks by signal role; user can override
     via right-click → "Move to Stack…"
  3. Per-channel toggle (Raw / RMS):
       Raw   = raw_data decimated from record.time_array
       RMS   = cycle-by-cycle, computed on demand on background thread
       Value = locked (PMU, TREND, derived, or non-electrical role)
  4. Per-file time offset applied for manual multi-file alignment
  5. [Phasor Display] → floating moveable QDialog with arrow diagram

Architecture: Presentation layer (ui/) — imports engine/ and models/ only
              (LAW 1).  Heavy computation dispatched via core.thread_manager
              (LAW 2).  Decimation cap MAX_DISPLAY_PTS per channel (LAW 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import AppSettings
from core.thread_manager import run_in_thread
from engine.decimator import decimate_minmax, decimate_uniform
from engine.rms_calculator import compute_cycle_rms
from engine.rms_merger import start_epoch_from_datetime
from models.channel import SignalRole
from models.disturbance_record import DisturbanceRecord, SourceFormat
from parsers.comtrade_parser import ComtradeParser
from parsers.csv_parser import CsvParser
from parsers.excel_parser import ExcelParser
from parsers.pmu_csv_parser import PmuCsvParser, is_pmu_csv
from ui.pmu_import_dialog import PmuImportDialog, SetStartTimeDialog

# ── Module constants ───────────────────────────────────────────────────────────

FILE_PANEL_WIDTH:    int   = 300
SLIDER_RANGE:        int   = 3000
OFFSET_DEBOUNCE_MS:  int   = 50
DEFAULT_STEP_MS:     float = 10.0
CANVAS_BG:           str   = '#1E1E1E'
MAX_DISPLAY_PTS:     int   = 2000      # LAW 3 — never render > 4000 pts/channel

CURSOR1_COLOUR:  str   = '#FFD700'     # gold
CURSOR2_COLOUR:  str   = '#00E5FF'     # cyan
PMU_SAMPLE_RATE: float = 50.0          # Hz — used for xcorr lag conversion
READOUT_MARGIN: int = 8                # px from edge of GLW widget

SQRT3: float = 1.7320508075688772      # √3 for phase-to-earth PU conversion

# Source formats whose channels are already in RMS / phasor form
_PMU_SOURCE_FORMATS: frozenset[str] = frozenset({'PMU_CSV'})

# Signal roles always locked to 'value' (not raw AC waveforms, no RMS toggle)
_LOCKED_VALUE_ROLES: frozenset[str] = frozenset({
    SignalRole.FREQ,
    SignalRole.ROCOF,
    SignalRole.P_MW,
    SignalRole.Q_MVAR,
    SignalRole.DC_FIELD_I,
    SignalRole.DC_FIELD_V,
    SignalRole.MECH_SPEED,
    SignalRole.MECH_VALVE,
})

# Voltage roles eligible for PU conversion
_VOLTAGE_ROLES: frozenset[str] = frozenset({
    SignalRole.V_PHASE,
    SignalRole.V_LINE,
    SignalRole.V_RESIDUAL,
    SignalRole.V1_PMU,
    SignalRole.SEQ_RMS,
})

_LINE_VOLTAGE_ROLES: frozenset[str] = frozenset({SignalRole.V_LINE})

_CURRENT_ROLES: frozenset[str] = frozenset({
    SignalRole.I_PHASE,
    SignalRole.I_EARTH,
    SignalRole.I1_PMU,
})

# ── Stack definitions ─────────────────────────────────────────────────────────
# stack_idx → frozenset of signal roles assigned to left / right axis by default

STACK_LEFT_ROLES: dict[int, frozenset[str]] = {
    0: frozenset({
        SignalRole.V_PHASE, SignalRole.V_LINE, SignalRole.V_RESIDUAL,
        SignalRole.V1_PMU, SignalRole.SEQ_RMS,
    }),
    1: frozenset({SignalRole.FREQ, SignalRole.ROCOF}),
    2: frozenset({SignalRole.DC_FIELD_V}),
    3: frozenset({SignalRole.MECH_SPEED}),
    4: frozenset({SignalRole.ANALOGUE}),
}

STACK_RIGHT_ROLES: dict[int, frozenset[str]] = {
    0: frozenset({SignalRole.I_PHASE, SignalRole.I_EARTH, SignalRole.I1_PMU}),
    1: frozenset({SignalRole.P_MW, SignalRole.Q_MVAR}),
    2: frozenset({SignalRole.DC_FIELD_I}),
    3: frozenset({SignalRole.MECH_VALVE}),
    4: frozenset(),
}

STACK_LEFT_LABELS: dict[int, str] = {
    0: 'Voltage',
    1: 'Frequency',
    2: 'Field Voltage',
    3: 'Speed (RPM)',
    4: 'Value',
}

STACK_RIGHT_LABELS: dict[int, str] = {
    0: 'Current',
    1: 'Power',
    2: 'Field Current',
    3: 'Position (%)',
    4: '',
}

STACK_NAMES: dict[int, str] = {
    0: 'Stack 1 — Voltage / Current',
    1: 'Stack 2 — Freq / Power',
    2: 'Stack 3 — DC Field',
    3: 'Stack 4 — Mechanical',
    4: 'Stack 5 — Generic',
}

# Phasor diagram colour palette (index → hex)
_PHASOR_COLOURS: list[str] = [
    '#FF4444', '#FFCC00', '#4488FF',
    '#44BB44', '#FF8800', '#AA44FF',
    '#00DDDD', '#AAAAAA',
]

_FILE_FILTER = (
    "Disturbance Records (*.cfg *.CFG *.csv *.CSV *.xlsx *.xls);;"
    "COMTRADE (*.cfg *.CFG);;"
    "CSV Files (*.csv *.CSV);;"
    "Excel Files (*.xlsx *.xls);;"
    "All Files (*)"
)
_COMTRADE_EXT: frozenset[str] = frozenset({'.cfg', '.dat'})
_CSV_EXT:      frozenset[str] = frozenset({'.csv'})
_EXCEL_EXT:    frozenset[str] = frozenset({'.xlsx', '.xls', '.xlsm'})

# Tree column indices
TREE_COL_NAME: int = 0   # channel name + checkbox
TREE_COL_MODE: int = 1   # Raw / RMS toggle button (or locked label)
TREE_COL_BASE: int = 2   # base voltage kV spinbox (voltage channels only)

# Digital events strip
STACK_DIGITAL:   int   = 5    # stack index — always rendered last
DIG_CH_HEIGHT:   float = 0.72 # filled band height per digital channel (0..1)

# Roles considered for "Select standard set" on a bay
_STANDARD_SET_ROLES: frozenset[str] = frozenset({
    SignalRole.V_PHASE,
    SignalRole.V_LINE,
    SignalRole.V_RESIDUAL,
    SignalRole.I_PHASE,
    SignalRole.I_EARTH,
})


# ── Draggable readout label ────────────────────────────────────────────────────

class _DraggableLabel(QLabel):
    """QLabel that can be repositioned by mouse drag.

    Emits ``user_moved`` after each drag step so the parent can suppress future
    auto-repositioning.
    """

    user_moved: pyqtSignal = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._drag_origin: Optional[object] = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta = event.position().toPoint() - self._drag_origin
            new_pos = self.pos() + delta
            parent = self.parentWidget()
            if parent:
                new_pos.setX(max(0, min(new_pos.x(), parent.width() - self.width())))
                new_pos.setY(max(0, min(new_pos.y(), parent.height() - self.height())))
            self.move(new_pos)
            self.user_moved.emit()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# ── Digital events ViewBox — wheel scrolls Y instead of zooming ───────────────

class _DigitalViewBox(pg.ViewBox):
    """ViewBox used exclusively for the digital events strip.

    Mouse-wheel scrolls through channels (pans Y) rather than zooming,
    so the user can navigate a tall strip that is cropped by the splitter.
    The X axis is controlled by the shared X-link and is not affected.
    """

    def wheelEvent(self, ev, axis: Optional[int] = None) -> None:  # type: ignore[override]
        """Pan Y by one channel per wheel notch; ignore X."""
        delta = ev.delta() if hasattr(ev, 'delta') else ev.angleDelta().y()
        notches = delta / 120.0                 # standard: 1 notch = 120 units
        self.translateBy(y=notches * 0.8)       # 0.8 channel-heights per notch
        ev.accept()


# ── Time axis — seconds ↔ cycles display ─────────────────────────────────────

class _TimeAxis(pg.AxisItem):
    """AxisItem for the bottom time axis with optional cycles display.

    When ``_cycles_mode`` is True, tick strings are multiplied by the nominal
    frequency so the axis reads in power-system cycles instead of seconds.
    The underlying data coordinates remain in seconds throughout; only the
    displayed labels change.
    """

    def __init__(self, orientation: str = 'bottom', **kwargs) -> None:
        super().__init__(orientation, **kwargs)
        self._cycles_mode:  bool  = False
        self._nominal_freq: float = 50.0

    def set_cycles_mode(self, enabled: bool, nominal_freq: float = 50.0) -> None:
        """Switch between seconds and cycles display and redraw."""
        self._cycles_mode  = enabled
        self._nominal_freq = nominal_freq
        self.picture = None   # invalidate cached paint
        self.update()

    def tickStrings(self, values, scale, spacing) -> list[str]:  # type: ignore[override]
        if not self._cycles_mode:
            return super().tickStrings(values, scale, spacing)
        freq = self._nominal_freq
        cyc  = [v * freq for v in values]
        # Use integer formatting when spacing is at least 1 cycle
        if spacing * freq >= 0.99:
            return [f'{v:.0f}' for v in cyc]
        return [f'{v:.1f}' for v in cyc]


# ── Internal state dataclass ───────────────────────────────────────────────────

@dataclass
class _LoadedFile:
    """Runtime state for one file loaded into the Unified Canvas.

    Attributes:
        file_id:       Unique string key (sequential integer as string).
        path:          Absolute path to the source file.
        record:        Parsed DisturbanceRecord.
        nominal_freq:  Active nominal frequency used for RMS computation (Hz).
        selected_ids:  Set of analogue channel_ids currently checked in the tree.
        start_epoch:   POSIX epoch float for record.start_time (UTC).
        timestamp_ok:  False when the file had a broken timestamp and the user
                       has not yet provided a corrected start time.
        tree_item:     The top-level QTreeWidgetItem for this file.
    """
    file_id:              str
    path:                 Path
    record:               DisturbanceRecord
    nominal_freq:         float
    selected_ids:         set[int]         = field(default_factory=set)
    selected_digital_ids: set[int]         = field(default_factory=set)
    scatter_ids:          set[int]         = field(default_factory=set)
    start_epoch:          float            = 0.0
    timestamp_ok:         bool             = True
    voltage_convention:   str              = 'line_to_line'   # 'line_to_line' | 'line_to_earth'
    tree_item:            Optional[object] = field(default=None, repr=False)


# ── Per-file offset control strip ─────────────────────────────────────────────

class _OffsetRow(QWidget):
    """One horizontal strip controlling the time offset for one file.

    Signals:
        offset_changed: Emitted when the offset value changes.
            Args: (file_id: str, offset_s: float)
        freq_changed: Emitted when the nominal frequency selector changes.
            Args: (file_id: str, freq_hz: float)
    """

    offset_changed: pyqtSignal = pyqtSignal(str, float)
    freq_changed:   pyqtSignal = pyqtSignal(str, float)

    def __init__(
        self,
        file_id:      str,
        display_name: str,
        parent:       Optional[QWidget] = None,
    ) -> None:
        """Initialise an offset row for ``file_id``.

        Args:
            file_id:      The _LoadedFile.file_id this row controls.
            display_name: Short filename shown as the row label.
            parent:       Optional parent widget.
        """
        super().__init__(parent)
        self._file_id  = file_id
        self._step_s   = DEFAULT_STEP_MS / 1000.0
        self._offset_s = 0.0
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        name_lbl = QLabel(display_name)
        name_lbl.setFixedWidth(140)
        name_lbl.setStyleSheet('color: #CCCCCC; font-size: 8pt;')
        layout.addWidget(name_lbl)

        layout.addWidget(QLabel('Freq:'))
        self._freq_combo = QComboBox()
        self._freq_combo.addItems(['50 Hz', '60 Hz'])
        self._freq_combo.setFixedWidth(60)
        self._freq_combo.currentIndexChanged.connect(self._on_freq_changed)
        layout.addWidget(self._freq_combo)

        layout.addWidget(QLabel('  Offset:'))

        btn_minus = QPushButton('−')
        btn_minus.setFixedWidth(24)
        btn_minus.clicked.connect(self._step_minus)
        layout.addWidget(btn_minus)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-SLIDER_RANGE, SLIDER_RANGE)
        self._slider.setValue(0)
        self._slider.setFixedWidth(160)
        self._slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self._slider)

        btn_plus = QPushButton('+')
        btn_plus.setFixedWidth(24)
        btn_plus.clicked.connect(self._step_plus)
        layout.addWidget(btn_plus)

        self._offset_lbl = QLabel('0.0 ms')
        self._offset_lbl.setFixedWidth(70)
        self._offset_lbl.setStyleSheet('color: #AAAAAA; font-size: 8pt;')
        layout.addWidget(self._offset_lbl)

        layout.addWidget(QLabel('Step:'))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.1, 10_000.0)
        self._step_spin.setValue(DEFAULT_STEP_MS)
        self._step_spin.setSuffix(' ms')
        self._step_spin.setFixedWidth(90)
        self._step_spin.valueChanged.connect(self._on_step_changed)
        layout.addWidget(self._step_spin)

        layout.addStretch()

    @property
    def offset_s(self) -> float:
        """Current time offset in seconds."""
        return self._offset_s

    def _on_slider_moved(self, pos: int) -> None:
        if self._updating:
            return
        self._offset_s = pos * self._step_s
        self._offset_lbl.setText(f'{self._offset_s * 1000:.1f} ms')
        self.offset_changed.emit(self._file_id, self._offset_s)

    def _step_minus(self) -> None:
        self._slider.setValue(self._slider.value() - 1)

    def _step_plus(self) -> None:
        self._slider.setValue(self._slider.value() + 1)

    def _on_step_changed(self, new_ms: float) -> None:
        self._updating = True
        self._step_s = new_ms / 1000.0
        new_pos = int(round(self._offset_s / self._step_s)) if self._step_s > 0 else 0
        self._slider.setValue(int(np.clip(new_pos, -SLIDER_RANGE, SLIDER_RANGE)))
        self._updating = False

    def _on_freq_changed(self, index: int) -> None:
        freq = 50.0 if index == 0 else 60.0
        self.freq_changed.emit(self._file_id, freq)


# ── Phasor drawing canvas ──────────────────────────────────────────────────────

class _PhasorCanvas(QWidget):
    """QPainter widget that draws phasor arrows on a unit circle.

    Arrows are drawn from the origin with length proportional to magnitude.
    The reference axis (0°) points to the right; angles are counter-clockwise.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise with an empty phasor list."""
        super().__init__(parent)
        self._phasors: list[tuple[str, float, float, str]] = []
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_phasors(
        self,
        phasors: list[tuple[str, float, float, str]],
    ) -> None:
        """Update phasor data and trigger a repaint.

        Args:
            phasors: List of (name, magnitude, angle_deg, colour_hex).
        """
        self._phasors = phasors
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the unit circle, axes, and phasor arrows."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = int(min(w, h) * 0.42)

        # Background
        painter.fillRect(self.rect(), QColor(CANVAS_BG))

        # Unit circle
        painter.setPen(QPen(QColor('#444444'), 1))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Reference axes
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.drawLine(cx - radius, cy, cx + radius, cy)
        painter.drawLine(cx, cy - radius, cx, cy + radius)

        if not self._phasors:
            painter.setPen(QPen(QColor('#666666'), 1))
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                'No phasor data',
            )
            painter.end()
            return

        max_mag = max(abs(m) for _, m, _, _ in self._phasors)
        scale = radius / max(max_mag, 1e-6)

        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)

        for name, mag, angle_deg, colour in self._phasors:
            # QPainter y-axis is inverted → negate angle for standard convention
            rad = math.radians(-angle_deg)
            dx = mag * scale * math.cos(rad)
            dy = mag * scale * math.sin(rad)
            ex = int(cx + dx)
            ey = int(cy + dy)

            pen_colour = QColor(colour)
            painter.setPen(QPen(pen_colour, 2))
            painter.drawLine(cx, cy, ex, ey)

            # Arrowhead
            arrow_len   = 8
            arrow_half  = math.radians(20)
            back_angle  = math.atan2(dy, dx) + math.pi
            ax1 = int(ex + arrow_len * math.cos(back_angle + arrow_half))
            ay1 = int(ey + arrow_len * math.sin(back_angle + arrow_half))
            ax2 = int(ex + arrow_len * math.cos(back_angle - arrow_half))
            ay2 = int(ey + arrow_len * math.sin(back_angle - arrow_half))
            painter.drawLine(ex, ey, ax1, ay1)
            painter.drawLine(ex, ey, ax2, ay2)

            # Label next to arrow tip
            painter.setPen(QPen(pen_colour, 1))
            painter.drawText(ex + 5, ey - 4, name[:8])

        painter.end()


# ── Phasor display dialog ──────────────────────────────────────────────────────

class PhasorDialog(QDialog):
    """Moveable, non-blocking dialog showing a live phasor diagram.

    Two columns:
      Left  — value table (channel, magnitude, angle) updated as cursor moves.
      Right — phasor arrow canvas (QPainter).

    The dialog is created with ``Qt.WindowType.Window`` so it is non-modal
    and freely moveable independent of the main window.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the dialog layout."""
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('Phasor Display')
        self.setMinimumSize(520, 320)
        self.resize(640, 400)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: live value table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(['Channel', 'Magnitude', 'Angle (°)'])
        self._table.horizontalHeader().setStretchLastSection(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.resizeSection(1, 80)
        hh.resizeSection(2, 72)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            'QTableWidget { background: #1E1E1E; color: #DDDDDD; font-size: 8pt; }'
            'QHeaderView::section { background: #3A3A3A; color: #CCCCCC; }'
            'QTableWidget::item:alternate { background: #252525; }'
        )
        self._table.setMinimumWidth(230)
        layout.addWidget(self._table, stretch=1)

        # Right: phasor arrow canvas
        self._canvas = _PhasorCanvas()
        self._canvas.setMinimumWidth(250)
        layout.addWidget(self._canvas, stretch=2)

    def update_phasors(
        self,
        phasors: list[tuple[str, float, float, str]],
    ) -> None:
        """Refresh the value table and the phasor diagram.

        Args:
            phasors: List of (channel_name, magnitude, angle_deg, colour_hex).
                     magnitude and angle come from the current cursor position.
        """
        self._table.setRowCount(len(phasors))
        for row, (name, mag, angle, colour) in enumerate(phasors):
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(f'{mag:.4f}'))
            self._table.setItem(row, 2, QTableWidgetItem(f'{angle:.1f}'))
        self._canvas.set_phasors(phasors)


# ── Frequency cross-correlation helper (module level — runs on worker thread) ──

def _xcorr_freq_lag(
    target_freq: np.ndarray,
    ref_freq:    np.ndarray,
) -> float:
    """Return the time lag (seconds) of target_freq relative to ref_freq.

    Uses FFT-based cross-correlation (scipy) for speed.  Both arrays are
    de-meaned so that slow DC drift does not dominate the peak.

    Positive return value means target_freq starts AFTER ref_freq
    (i.e. to align, shift target forward by lag_s).
    Negative return value means target_freq starts BEFORE ref_freq.

    Args:
        target_freq: Raw Frequency channel array from the file to align.
        ref_freq:    Raw Frequency channel array from the reference file.

    Returns:
        Lag in seconds at PMU_SAMPLE_RATE (50 fps → 0.020 s resolution).
    """
    from scipy.signal import correlate  # noqa: PLC0415

    # Trim to same length and de-mean
    n = min(len(target_freq), len(ref_freq))
    a = ref_freq[:n]    - ref_freq[:n].mean()
    b = target_freq[:n] - target_freq[:n].mean()

    corr = correlate(a, b, mode='full', method='fft')
    # correlate(ref, target): peak at index n-1-k when target lags ref by k.
    # Negate to get lag as "target is k samples AFTER ref".
    lag_samples = (n - 1) - int(np.argmax(corr))
    return float(lag_samples) / PMU_SAMPLE_RATE


# ── Main widget ────────────────────────────────────────────────────────────────

class UnifiedCanvasWidget(QWidget):
    """Multi-file, multi-stack analogue waveform overlay tab.

    Loads its own files independently of the main waveform panel.
    Each channel can be displayed as raw waveform or cycle-by-cycle RMS.
    Channels are automatically grouped into parameter stacks (Voltage/Current,
    Freq/Power, etc.) with a shared synchronised X time axis.
    """

    # Emitted whenever cursors move or channel selection changes.
    # Carries (t_c1_s, t_c2_s, rows) where rows is a list of
    # (display_name: str, unit: str, val_c1: float, val_c2: float).
    # NaN is used for a disabled cursor's time and its channel values.
    # Connected by main.py to MeasurementPanel.show_unified_mode().
    readout_updated: pyqtSignal = pyqtSignal(float, float, object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the layout and initialise all state."""
        super().__init__(parent)

        # ── State ─────────────────────────────────────────────────────────────
        self._files:        dict[str, _LoadedFile] = {}
        self._file_counter: int = 0
        self._offsets:      dict[str, float] = {}

        # (file_id, ch_id) → 'raw' | 'rms' | 'value'
        self._channel_mode:   dict[tuple[str, int], str]          = {}
        # (file_id, ch_id) → True when mode cannot be changed by user
        self._mode_locked:    dict[tuple[str, int], bool]         = {}
        # (file_id, ch_id) → (stack_idx, 'left' | 'right')
        self._stack_assign:   dict[tuple[str, int], tuple[int, str]] = {}
        # RMS cache: (file_id, ch_id) → (t_centres_s, rms_values)
        self._rms_cache:      dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

        self._pu_mode: bool = False
        # (file_id, ch_id) → base voltage kV (0.0 = not set)
        self._base_kv: dict[tuple[str, int], float] = {}

        # Canvas state — rebuilt on structural changes
        self._stack_widgets:  dict[int, pg.PlotWidget]   = {}
        self._stack_plots:    dict[int, pg.PlotItem]     = {}
        self._stack_vb2s:     dict[int, pg.ViewBox]      = {}
        self._cursor1_lines:  dict[int, pg.InfiniteLine] = {}
        self._cursor2_lines:  dict[int, pg.InfiniteLine] = {}
        self._cursor1_enabled: bool = True   # C1 shown by default
        self._cursor2_enabled: bool = False
        self._xaxis_cycles:   bool = False
        self._time_axes: dict[int, _TimeAxis] = {}
        self._readout:        Optional[_DraggableLabel]  = None
        self._readout_pinned: bool = False
        self._curves:         dict[tuple[str, int], pg.PlotDataItem] = {}
        self._digital_curves: dict[tuple[str, int], pg.PlotDataItem] = {}
        self._digital_row:    dict[tuple[str, int], int]              = {}
        self._ref_plot:       Optional[pg.PlotItem] = None

        # Guard flag — prevents recursive itemChanged during batch propagation
        self._in_tree_change: bool = False

        # Debounce timer — offset slider drag → _update_curves
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(OFFSET_DEBOUNCE_MS)
        self._update_timer.timeout.connect(self._update_curves)

        # Phasor dialog — lazy-created on first open
        self._phasor_dialog: Optional[PhasorDialog] = None

        # ── Build UI ──────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.setSizes([FILE_PANEL_WIDTH, 900])
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_offset_section())

    # ── UI builders ────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        """Build the top toolbar strip."""
        bar = QWidget()
        bar.setStyleSheet('background: #2D2D2D;')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        add_btn = QPushButton('+ Add File')
        add_btn.clicked.connect(self._on_add_file)
        layout.addWidget(add_btn)

        rm_btn = QPushButton('Remove')
        rm_btn.clicked.connect(self._on_remove_file)
        layout.addWidget(rm_btn)

        layout.addWidget(self._make_sep())

        self._pu_btn = QPushButton('PU Mode')
        self._pu_btn.setCheckable(True)
        self._pu_btn.setToolTip(
            'Toggle voltage display between actual values and per-unit (PU).\n'
            'Right-click a voltage channel in the tree to set its nominal kV.'
        )
        self._pu_btn.toggled.connect(self._on_pu_toggled)
        layout.addWidget(self._pu_btn)

        self._cycles_btn = QPushButton('s / cyc')
        self._cycles_btn.setCheckable(True)
        self._cycles_btn.setToolTip(
            'Toggle X-axis between seconds and power-system cycles.\n'
            'Cycle count uses the nominal frequency of the first loaded file.'
        )
        self._cycles_btn.toggled.connect(self._on_cycles_toggled)
        layout.addWidget(self._cycles_btn)

        layout.addWidget(self._make_sep())

        phasor_btn = QPushButton('Phasor Display')
        phasor_btn.setToolTip('Open the phasor diagram (moveable floating window)')
        phasor_btn.clicked.connect(self._show_phasor_dialog)
        layout.addWidget(phasor_btn)

        layout.addStretch()
        return bar

    def _build_file_panel(self) -> QWidget:
        """Build the left file + channel tree panel."""
        panel = QWidget()
        panel.setFixedWidth(FILE_PANEL_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel(' Files & Channels')
        hdr.setStyleSheet(
            'background: #3A3A3A; color: #AAAAAA; font-size: 8pt; padding: 3px;'
        )
        layout.addWidget(hdr)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText('Filter channels…')
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.setStyleSheet(
            'background: #2A2A2A; color: #CCCCCC; font-size: 8pt;'
            ' border: none; border-bottom: 1px solid #444; padding: 3px 6px;'
        )
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_edit)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(False)
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(['Channel', 'Mode', 'Nominal kV'])
        self._tree.setStyleSheet(
            'background: #252525; color: #DDDDDD; font-size: 8pt;'
        )
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(
            TREE_COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        self._tree.header().setSectionResizeMode(
            TREE_COL_MODE, QHeaderView.ResizeMode.Fixed
        )
        self._tree.header().setSectionResizeMode(
            TREE_COL_BASE, QHeaderView.ResizeMode.Fixed
        )
        self._tree.header().resizeSection(TREE_COL_MODE, 58)
        self._tree.header().resizeSection(TREE_COL_BASE, 70)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        layout.addWidget(self._tree)
        return panel

    def _build_canvas_panel(self) -> QWidget:
        """Build the right panel containing the stack splitter.

        Each active stack lives in its own PlotWidget inside a vertical
        QSplitter, giving the user native drag-to-resize handles between
        stacks.  The readout overlay floats above the whole canvas area.
        """
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # ── Canvas container — holds splitter + floating readout overlay ──────
        self._glw = QWidget()                    # alias kept for compat
        self._glw.setStyleSheet(f'background: {CANVAS_BG};')
        glw_layout = QVBoxLayout(self._glw)
        glw_layout.setContentsMargins(0, 0, 0, 0)
        glw_layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setHandleWidth(5)
        self._splitter.setStyleSheet(
            'QSplitter::handle         { background: #2A2A2A; }'
            'QSplitter::handle:hover   { background: #4A7A4A; }'
            'QSplitter::handle:pressed { background: #5A9A5A; }'
        )
        glw_layout.addWidget(self._splitter)
        self._splitter.splitterMoved.connect(self._update_digital_tick_font)
        panel_layout.addWidget(self._glw, stretch=1)

        # ── Draggable readout overlay — floats over the container ─────────────
        self._readout = _DraggableLabel(self._glw)
        self._readout.setStyleSheet(
            'background-color: rgba(255,255,255,210);'
            'color: #111111;'
            'font-family: Monospace;'
            'font-size: 8pt;'
            'padding: 6px 8px;'
            'border-radius: 4px;'
        )
        self._readout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._readout.user_moved.connect(
            lambda: setattr(self, '_readout_pinned', True)
        )
        self._readout.hide()
        self._readout.raise_()

        # Reposition overlay when container is resized
        self._glw.installEventFilter(self)

        return panel

    def _build_offset_section(self) -> QWidget:
        """Build the per-file time offset strip at the bottom."""
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)

        hdr = QLabel(' Time Offset Controls')
        hdr.setStyleSheet(
            'background: #3A3A3A; color: #AAAAAA; font-size: 8pt; padding: 3px;'
        )
        section_layout.addWidget(hdr)

        self._offset_scroll = QScrollArea()
        self._offset_scroll.setWidgetResizable(True)
        self._offset_scroll.setMaximumHeight(120)
        self._offset_scroll.setStyleSheet('background: #222222;')
        self._offset_container = QWidget()
        self._offset_layout = QVBoxLayout(self._offset_container)
        self._offset_layout.setContentsMargins(0, 0, 0, 0)
        self._offset_layout.setSpacing(0)
        self._offset_layout.addStretch()
        self._offset_scroll.setWidget(self._offset_container)
        section_layout.addWidget(self._offset_scroll)

        return section

    # ── Qt overrides ───────────────────────────────────────────────────────────

    def eventFilter(self, obj: object, event: object) -> bool:
        """Reposition the readout overlay when the GLW widget is resized."""
        if obj is self._glw and event.type() == QEvent.Type.Resize:
            self._reposition_readout()
        return super().eventFilter(obj, event)

    # ── File loading ───────────────────────────────────────────────────────────

    def open_file_dialog(self) -> None:
        """Public entry-point so File > Open in the main menu delegates here."""
        self._on_add_file()

    def _on_add_file(self) -> None:
        """Open file dialog and dispatch parsing to the background thread."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add File — Unified Canvas', '', _FILE_FILTER
        )
        for path_str in paths:
            run_in_thread(
                self._parse_file,
                Path(path_str),
                on_done=self._on_file_parsed,
                on_error=lambda exc: QMessageBox.critical(
                    self, 'Load Error', str(exc)
                ),
            )

    def _parse_file(
        self, path: Path
    ) -> tuple[_LoadedFile, object]:
        """Parse one file and return (_LoadedFile, report) (background thread).

        For PMU CSV files, also runs the import validator and returns a
        ParseInspectionReport.  For all other formats, the report is None.

        Args:
            path: Path to the file to parse.

        Returns:
            (loaded, report) where report is a ParseInspectionReport or None.
        """
        ext = path.suffix.lower()
        report = None

        if ext in _COMTRADE_EXT:
            record = ComtradeParser().load(path)
        elif ext in _CSV_EXT:
            if is_pmu_csv(path):
                record, report = PmuCsvParser().load_with_report(path)
            else:
                record = CsvParser().load(path)
        elif ext in _EXCEL_EXT:
            record = ExcelParser().load(path)
        else:
            raise ValueError(f'Unsupported file type: {path.suffix}')

        file_id = str(self._file_counter)
        self._file_counter += 1

        raw_epoch = start_epoch_from_datetime(record.start_time)
        # COMTRADE stores local wall-clock time (no timezone in the standard).
        # PMU CSV is already converted to UTC by the parser.
        # Apply the user-configured UTC offset to bring COMTRADE epochs to UTC.
        if record.source_format not in _PMU_SOURCE_FORMATS and raw_epoch > 86400:
            tz_h = AppSettings.get('calculation.comtrade_tz_offset_h', 0)
            raw_epoch -= float(tz_h) * 3600.0

        loaded = _LoadedFile(
            file_id=file_id,
            path=path,
            record=record,
            nominal_freq=50.0,
            selected_ids=set(),           # all unchecked by default
            selected_digital_ids=set(),
            start_epoch=raw_epoch,
            timestamp_ok=(report is None or not report.has_blockers),
        )
        # LAW 9: TREND records (sample_rate < 200 Hz) default to scatter display
        if record.display_mode == 'TREND':
            loaded.scatter_ids = {ch.channel_id for ch in record.analogue_channels}
        return loaded, report

    def _on_file_parsed(self, result: tuple) -> None:
        """Integrate a freshly parsed file into UI state (UI thread).

        Shows PmuImportDialog when the report contains BLOCKER issues.
        Applies any user-provided start-time anchor before integrating.

        Args:
            result: (loaded, report) tuple from _parse_file.
        """
        loaded, report = result

        # ── Show import dialog for PMU files with broken timestamps ───────────
        if report is not None and report.has_blockers:
            hints = self._build_time_hints(report.first_date_str)
            dlg = PmuImportDialog(report, hints, parent=self)
            if dlg.exec() and dlg.anchor_utc is not None:
                loaded.start_epoch = start_epoch_from_datetime(dlg.anchor_utc)
                loaded.timestamp_ok = True

        # ── Integrate into state ──────────────────────────────────────────────
        self._files[loaded.file_id] = loaded
        self._offsets[loaded.file_id] = 0.0

        for ch in loaded.record.analogue_channels:
            key = (loaded.file_id, ch.channel_id)
            mode, locked = self._detect_mode(ch, loaded.record)
            self._channel_mode[key]  = mode
            self._mode_locked[key]   = locked
            self._stack_assign[key]  = self._default_stack(ch.signal_role)

        self._add_tree_item(loaded)
        self._add_offset_row(loaded)
        self._rebuild_canvas()

    def _build_time_hints(self, date_str: str) -> list[tuple[str, str]]:
        """Return (filename, time_str) hints for already-loaded files.

        Used to populate the hint section of PmuImportDialog so the user can
        see what times other files on the same date were recorded.

        Args:
            date_str: Date string from the broken file (e.g. '10/15/25').

        Returns:
            List of (stem, 'HH:MM:SS SGT') tuples for files whose epoch is
            a valid non-fallback value.
        """
        import datetime as _dt_mod  # noqa: PLC0415
        hints: list[tuple[str, str]] = []
        for f in self._files.values():
            if f.start_epoch <= 86400 or not f.timestamp_ok:
                continue
            try:
                utc = _dt_mod.datetime.utcfromtimestamp(f.start_epoch)
                sgt = utc + _dt_mod.timedelta(hours=8)
                hints.append((f.path.stem[:20], sgt.strftime('%H:%M:%S') + ' SGT'))
            except Exception:
                pass
        return hints

    def _on_remove_file(self) -> None:
        """Remove the currently selected file from tree and state dicts."""
        items = self._tree.selectedItems()
        if not items:
            return
        item = items[0]
        while item.parent() is not None:
            item = item.parent()

        file_id = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        if file_id is None:
            return

        loaded = self._files.pop(file_id, None)
        if loaded:
            for ch in loaded.record.analogue_channels:
                key = (file_id, ch.channel_id)
                self._channel_mode.pop(key, None)
                self._mode_locked.pop(key, None)
                self._stack_assign.pop(key, None)
                self._rms_cache.pop(key, None)
                self._base_kv.pop(key, None)
            for dch in loaded.record.digital_channels:
                key = (file_id, dch.channel_id)
                self._digital_curves.pop(key, None)
                self._digital_row.pop(key, None)

        self._offsets.pop(file_id, None)
        idx = self._tree.indexOfTopLevelItem(item)
        self._tree.takeTopLevelItem(idx)

        self._rebuild_offset_strip()
        self._rebuild_canvas()

    # ── Channel mode detection ─────────────────────────────────────────────────

    @staticmethod
    def _detect_mode(ch: object, record: DisturbanceRecord) -> tuple[str, bool]:
        """Return (mode, locked) for a channel.

        Args:
            ch:     An AnalogueChannel instance.
            record: The owning DisturbanceRecord.

        Returns:
            Tuple of (mode_str, is_locked).
            mode_str ∈ {'raw', 'rms', 'value'}.
            is_locked is True when the user may not toggle the mode.
        """
        if getattr(ch, 'is_derived', False):
            return 'value', True
        if record.source_format in _PMU_SOURCE_FORMATS:
            return 'rms', True
        if record.display_mode == 'TREND':
            return 'value', True
        role = getattr(ch, 'signal_role', SignalRole.ANALOGUE)
        if role in _LOCKED_VALUE_ROLES:
            return 'value', True
        return 'raw', False

    # ── Stack assignment ───────────────────────────────────────────────────────

    @staticmethod
    def _default_stack(role: str) -> tuple[int, str]:
        """Return (stack_idx, side) for a signal role.

        Args:
            role: SignalRole string value.

        Returns:
            Tuple of (stack_index, 'left' | 'right').
        """
        for idx, roles in STACK_LEFT_ROLES.items():
            if role in roles:
                return idx, 'left'
        for idx, roles in STACK_RIGHT_ROLES.items():
            if role in roles:
                return idx, 'right'
        return 4, 'left'

    # ── Tree management ────────────────────────────────────────────────────────

    def _add_tree_item(self, loaded: _LoadedFile) -> None:
        """Add a file node with bay → analogue/digital group → channel hierarchy.

        Tree structure:
          FILE (tristate, unchecked)
            BAY_NAME (tristate, unchecked, collapsed)
              Analogue (N)  (tristate, unchecked)
                ch1 (unchecked) [Mode btn] [kV spin]
                ...
              Digital  (N)  (tristate, unchecked, collapsed)
                dch1 (unchecked)
                ...

        All channels start unchecked — user selects what to display.

        Args:
            loaded: The _LoadedFile to add.
        """
        conv_badge = '[L-L]' if loaded.voltage_convention == 'line_to_line' else '[L-E]'
        file_item = QTreeWidgetItem([f'📄 {loaded.path.stem}  {conv_badge}'])
        file_item.setData(TREE_COL_NAME, Qt.ItemDataRole.UserRole, loaded.file_id)
        file_item.setFlags(
            file_item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        file_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)

        # ── Group channels by bay ─────────────────────────────────────────────
        analogue_by_bay: dict[str, list] = {}
        for ch in loaded.record.analogue_channels:
            bay = ch.bay_name or '(Ungrouped)'
            analogue_by_bay.setdefault(bay, []).append(ch)

        digital_by_bay: dict[str, list] = {}
        for dch in loaded.record.digital_channels:
            bay = dch.bay_name or '(Ungrouped)'
            digital_by_bay.setdefault(bay, []).append(dch)

        # Preserve parser-detected bay order; append any extras alphabetically
        all_bays: set[str] = set(analogue_by_bay) | set(digital_by_bay)
        bay_order: list[str] = list(loaded.record.bay_names) if loaded.record.bay_names else []
        for b in sorted(all_bays):
            if b not in bay_order:
                bay_order.append(b)
        if not bay_order:
            bay_order = ['(All Channels)']

        self._tree.blockSignals(True)

        for bay_name in bay_order:
            a_chs = analogue_by_bay.get(bay_name, [])
            d_chs = digital_by_bay.get(bay_name, [])
            if not a_chs and not d_chs:
                continue

            bay_item = QTreeWidgetItem([bay_name])
            bay_item.setData(
                TREE_COL_NAME, Qt.ItemDataRole.UserRole,
                ('bay', loaded.file_id, bay_name),
            )
            bay_item.setFlags(
                bay_item.flags()
                | Qt.ItemFlag.ItemIsAutoTristate
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            bay_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
            file_item.addChild(bay_item)

            # ── Analogue sub-group ────────────────────────────────────────────
            if a_chs:
                agroup = QTreeWidgetItem([f'Analogue  ({len(a_chs)})'])
                agroup.setData(
                    TREE_COL_NAME, Qt.ItemDataRole.UserRole,
                    ('agroup', loaded.file_id, bay_name),
                )
                agroup.setFlags(
                    agroup.flags()
                    | Qt.ItemFlag.ItemIsAutoTristate
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                agroup.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
                bay_item.addChild(agroup)

                for ch in a_chs:
                    key = (loaded.file_id, ch.channel_id)
                    mode   = self._channel_mode.get(key, 'raw')
                    locked = self._mode_locked.get(key, False)
                    _, side = self._stack_assign.get(key, (4, 'left'))
                    side_tag = '[R]' if side == 'right' else '[L]'

                    ch_item = QTreeWidgetItem(
                        [f'{side_tag} {ch.name}  [{ch.unit or "—"}]', '', '']
                    )
                    ch_item.setData(
                        TREE_COL_NAME, Qt.ItemDataRole.UserRole,
                        ('ach', loaded.file_id, ch.channel_id),
                    )
                    ch_item.setFlags(ch_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    ch_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
                    agroup.addChild(ch_item)

                    btn = QPushButton()
                    btn.setFixedSize(54, 18)
                    btn.setStyleSheet('font-size: 7pt; padding: 1px;')
                    if locked:
                        btn.setText(mode.capitalize())
                        btn.setEnabled(False)
                        btn.setToolTip('Mode is fixed — channel is in its native format.')
                    else:
                        btn.setText('Raw' if mode == 'raw' else 'RMS')
                        btn.clicked.connect(
                            lambda _c, fid=loaded.file_id, cid=ch.channel_id, b=btn:
                                self._on_mode_toggled(fid, cid, b)
                        )
                    self._tree.setItemWidget(ch_item, TREE_COL_MODE, btn)

                    if self._is_voltage_channel(loaded.file_id, ch.channel_id):
                        spin = QDoubleSpinBox()
                        spin.setRange(0.0, 9999.0)
                        spin.setDecimals(1)
                        spin.setSuffix(' kV')
                        spin.setSpecialValueText('—')
                        spin.setFixedWidth(68)
                        spin.setStyleSheet('font-size: 7pt;')
                        spin.setToolTip(
                            'System nominal voltage in line-to-line kV (e.g. 275).\n'
                            'Right-click the file to set the voltage convention.'
                        )
                        spin.setValue(self._base_kv.get(key, 0.0))
                        spin.valueChanged.connect(
                            lambda v, fid=loaded.file_id, cid=ch.channel_id:
                                self._on_base_kv_changed(fid, cid, v)
                        )
                        self._tree.setItemWidget(ch_item, TREE_COL_BASE, spin)

            # ── Digital sub-group ─────────────────────────────────────────────
            if d_chs:
                # Skip secondary channels of complementary pairs (CLOSED side)
                display_d = [
                    dch for dch in d_chs
                    if not (dch.is_complementary and not dch.is_primary_of_pair)
                ]
                if display_d:
                    dgroup = QTreeWidgetItem([f'Digital  ({len(display_d)})'])
                    dgroup.setData(
                        TREE_COL_NAME, Qt.ItemDataRole.UserRole,
                        ('dgroup', loaded.file_id, bay_name),
                    )
                    dgroup.setFlags(
                        dgroup.flags()
                        | Qt.ItemFlag.ItemIsAutoTristate
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    dgroup.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
                    bay_item.addChild(dgroup)
                    dgroup.setExpanded(False)

                    for dch in display_d:
                        pair_tag = '  ⇌' if dch.is_primary_of_pair else ''
                        dch_item = QTreeWidgetItem(
                            [f'{dch.name}{pair_tag}', '', '']
                        )
                        dch_item.setData(
                            TREE_COL_NAME, Qt.ItemDataRole.UserRole,
                            ('dch', loaded.file_id, dch.channel_id),
                        )
                        dch_item.setFlags(dch_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        dch_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
                        dgroup.addChild(dch_item)

        self._tree.blockSignals(False)
        self._tree.addTopLevelItem(file_item)
        file_item.setExpanded(True)
        # Bay nodes stay collapsed — user expands on demand
        loaded.tree_item = file_item

    def _on_tree_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        """Sync selected_ids / selected_digital_ids and rebuild when a checkbox changes.

        Handles four tree levels: file → bay → group → channel.
        File/bay/group nodes propagate their state down to all descendants.
        PartiallyChecked state propagates upward automatically via ItemIsAutoTristate.

        Args:
            item: The item whose checkbox changed.
            col:  Column index (always TREE_COL_NAME).
        """
        if col != TREE_COL_NAME or self._in_tree_change:
            return

        data  = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        state = item.checkState(TREE_COL_NAME)

        # PartiallyChecked propagates upward from children automatically.
        # Just sync selected_ids from the current tree state and rebuild.
        if state == Qt.CheckState.PartiallyChecked:
            file_id = self._get_file_id_for_item(item)
            if file_id and file_id in self._files:
                self._sync_selected_ids(self._files[file_id])
            self._rebuild_canvas()
            return

        file_id = self._get_file_id_for_item(item)
        if file_id is None:
            return
        loaded = self._files.get(file_id)
        if loaded is None:
            return

        is_group_node = (
            isinstance(data, str)
            or (isinstance(data, tuple) and data[0] in ('bay', 'agroup', 'dgroup'))
        )

        if is_group_node:
            # Propagate down with signals blocked, then fix parent tristates
            self._in_tree_change = True
            try:
                self._tree.blockSignals(True)
                self._propagate_check_down(item, state)
                self._tree.blockSignals(False)
                self._update_ancestors(item)
            finally:
                self._in_tree_change = False
            self._sync_selected_ids(loaded)
        else:
            # Channel node — update directly
            if isinstance(data, tuple) and len(data) == 3:
                kind, _fid, ch_id = data
                if kind == 'ach':
                    if state == Qt.CheckState.Checked:
                        loaded.selected_ids.add(ch_id)
                    else:
                        loaded.selected_ids.discard(ch_id)
                elif kind == 'dch':
                    if state == Qt.CheckState.Checked:
                        loaded.selected_digital_ids.add(ch_id)
                    else:
                        loaded.selected_digital_ids.discard(ch_id)

        self._rebuild_canvas()

    # ── Tree helper methods ────────────────────────────────────────────────────

    def _get_file_id_for_item(self, item: QTreeWidgetItem) -> Optional[str]:
        """Walk up to the top-level (file) node and return its file_id.

        Args:
            item: Any QTreeWidgetItem in the tree.

        Returns:
            file_id string, or None if the root node is not a file node.
        """
        root = item
        while root.parent() is not None:
            root = root.parent()
        data = root.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        return data if isinstance(data, str) else None

    def _propagate_check_down(
        self,
        item:  QTreeWidgetItem,
        state: Qt.CheckState,
    ) -> None:
        """Recursively set all descendants to ``state``.

        Must be called with signals blocked to prevent N rebuild calls.

        Args:
            item:  The parent whose subtree to update.
            state: Target check state (Checked or Unchecked).
        """
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(TREE_COL_NAME, state)
            self._propagate_check_down(child, state)

    def _recalculate_tristate(self, item: QTreeWidgetItem) -> None:
        """Set ``item``'s check state to match the aggregate of its children.

        Args:
            item: A group/bay/file node with children.
        """
        n = item.childCount()
        if n == 0:
            return
        checked = sum(
            1 for i in range(n)
            if item.child(i).checkState(TREE_COL_NAME) == Qt.CheckState.Checked
        )
        partial = sum(
            1 for i in range(n)
            if item.child(i).checkState(TREE_COL_NAME) == Qt.CheckState.PartiallyChecked
        )
        if checked == n:
            item.setCheckState(TREE_COL_NAME, Qt.CheckState.Checked)
        elif checked == 0 and partial == 0:
            item.setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
        else:
            item.setCheckState(TREE_COL_NAME, Qt.CheckState.PartiallyChecked)

    def _update_ancestors(self, item: QTreeWidgetItem) -> None:
        """Walk upward from ``item`` and recalculate tristate on each ancestor.

        Called after a blocked batch propagation, so parent nodes didn't
        receive automatic tristate updates.

        Args:
            item: The item whose ancestors to update.
        """
        parent = item.parent()
        while parent is not None:
            self._tree.blockSignals(True)
            self._recalculate_tristate(parent)
            self._tree.blockSignals(False)
            parent = parent.parent()

    def _sync_selected_ids(self, loaded: _LoadedFile) -> None:
        """Rebuild selected_ids and selected_digital_ids from current tree state.

        Scans the entire file subtree; O(N) but N ≤ a few hundred channels.

        Args:
            loaded: The _LoadedFile whose selection to sync.
        """
        if loaded.tree_item is None:
            return
        loaded.selected_ids.clear()
        loaded.selected_digital_ids.clear()

        def _walk(node: QTreeWidgetItem) -> None:
            data = node.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
            st   = node.checkState(TREE_COL_NAME)
            if isinstance(data, tuple) and len(data) == 3:
                kind = data[0]
                if kind == 'ach' and st == Qt.CheckState.Checked:
                    loaded.selected_ids.add(data[2])
                elif kind == 'dch' and st == Qt.CheckState.Checked:
                    loaded.selected_digital_ids.add(data[2])
            for i in range(node.childCount()):
                _walk(node.child(i))

        _walk(loaded.tree_item)

    def _find_bay_item(
        self,
        file_id:  str,
        bay_name: str,
    ) -> Optional[QTreeWidgetItem]:
        """Return the QTreeWidgetItem for a bay node, or None if not found.

        Args:
            file_id:  The file this bay belongs to.
            bay_name: The bay label string.
        """
        loaded = self._files.get(file_id)
        if loaded is None or loaded.tree_item is None:
            return None
        for i in range(loaded.tree_item.childCount()):
            bay = loaded.tree_item.child(i)
            d = bay.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
            if isinstance(d, tuple) and d[0] == 'bay' and d[2] == bay_name:
                return bay
        return None

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click on tree item → context menu.

        File nodes:    Set Start Time / Auto-align / Voltage Convention.
        Bay nodes:     Select standard set / Check all / Uncheck all.
        Analogue chs:  Stack reassignment.
        Digital chs:   (reserved for future actions).

        Args:
            pos: Click position in tree viewport coordinates.
        """
        from PyQt6.QtWidgets import QMenu  # noqa: PLC0415
        item = self._tree.itemAt(pos)
        if item is None:
            return

        global_pos = self._tree.viewport().mapToGlobal(pos)
        data = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)

        # ── File-level context menu ───────────────────────────────────────────
        if isinstance(data, str):
            file_id = data
            loaded_file = self._files.get(file_id)
            if loaded_file is None:
                return

            menu = QMenu(self)
            set_time_act = menu.addAction('Set Start Time…')

            align_menu = menu.addMenu('Auto-align from Frequency to…')
            align_acts: dict[str, object] = {}
            for other_id, other in self._files.items():
                if other_id != file_id and other.timestamp_ok:
                    act = align_menu.addAction(other.path.stem)
                    align_acts[other_id] = act
            if not align_acts:
                align_menu.setEnabled(False)

            menu.addSeparator()
            conv_menu = menu.addMenu('Voltage Convention (channel value format)')
            ll_act = conv_menu.addAction('Channel values are Line-to-Line  (default)')
            le_act = conv_menu.addAction('Channel values are Line-to-Earth  (phase-to-earth)')
            ll_act.setCheckable(True)
            le_act.setCheckable(True)
            ll_act.setChecked(loaded_file.voltage_convention == 'line_to_line')
            le_act.setChecked(loaded_file.voltage_convention == 'line_to_earth')

            chosen = menu.exec(global_pos)
            if chosen is set_time_act:
                self._show_set_start_time_dialog(file_id)
            elif chosen is ll_act:
                self._set_voltage_convention(file_id, 'line_to_line')
            elif chosen is le_act:
                self._set_voltage_convention(file_id, 'line_to_earth')
            else:
                for ref_id, act in align_acts.items():
                    if chosen is act:
                        self._do_auto_align(file_id, ref_id)
            return

        if not isinstance(data, tuple):
            return
        kind = data[0]

        # ── Bay-level context menu ────────────────────────────────────────────
        if kind == 'bay':
            _, file_id, bay_name = data
            menu = QMenu(self)
            std_act  = menu.addAction('Select standard set  (3Ф V + I)')
            menu.addSeparator()
            all_act  = menu.addAction('Check all channels in bay')
            none_act = menu.addAction('Uncheck all channels in bay')
            chosen = menu.exec(global_pos)
            if chosen is std_act:
                self._select_standard_set(file_id, bay_name)
            elif chosen is all_act:
                self._set_bay_check(file_id, bay_name, Qt.CheckState.Checked)
            elif chosen is none_act:
                self._set_bay_check(file_id, bay_name, Qt.CheckState.Unchecked)
            return

        # Group nodes: no menu
        if kind in ('agroup', 'dgroup'):
            return

        # ── Analogue channel — stack reassignment + scatter toggle ────────────
        if kind == 'ach':
            _, file_id, ch_id = data
            key = (file_id, ch_id)
            curr_stack, curr_side = self._stack_assign.get(key, (4, 'left'))
            loaded_ch = self._files.get(file_id)

            menu       = QMenu(self)

            # Scatter / line display toggle
            scatter_act = menu.addAction('Display as Scatter')
            scatter_act.setCheckable(True)
            scatter_act.setChecked(
                loaded_ch is not None and ch_id in loaded_ch.scatter_ids
            )
            menu.addSeparator()

            left_menu  = menu.addMenu('→ Left Axis')
            right_menu = menu.addMenu('→ Right Axis')

            left_actions:  dict[int, object] = {}
            right_actions: dict[int, object] = {}
            for sidx, sname in STACK_NAMES.items():
                la = left_menu.addAction(sname)
                la.setCheckable(True)
                la.setChecked(curr_stack == sidx and curr_side == 'left')
                left_actions[sidx] = la

                ra = right_menu.addAction(sname)
                ra.setCheckable(True)
                ra.setChecked(curr_stack == sidx and curr_side == 'right')
                right_actions[sidx] = ra

            chosen = menu.exec(global_pos)
            if chosen is None:
                return
            if chosen is scatter_act:
                if loaded_ch is not None:
                    if ch_id in loaded_ch.scatter_ids:
                        loaded_ch.scatter_ids.discard(ch_id)
                    else:
                        loaded_ch.scatter_ids.add(ch_id)
                    self._rebuild_canvas()
                return
            for sidx, act in left_actions.items():
                if chosen is act:
                    self._stack_assign[key] = (sidx, 'left')
                    self._refresh_tree_label(item, file_id, ch_id)
                    self._rebuild_canvas()
                    return
            for sidx, act in right_actions.items():
                if chosen is act:
                    self._stack_assign[key] = (sidx, 'right')
                    self._refresh_tree_label(item, file_id, ch_id)
                    self._rebuild_canvas()
                    return

    # ── Bay selection helpers ──────────────────────────────────────────────────

    def _select_standard_set(self, file_id: str, bay_name: str) -> None:
        """Check only the standard 3-phase V + I channels for one bay.

        Checks all V_PHASE/V_LINE/V_RESIDUAL/I_PHASE/I_EARTH analogue channels
        in the bay and unchecks everything else in that bay.

        Args:
            file_id:  The file this bay belongs to.
            bay_name: Bay label string.
        """
        loaded = self._files.get(file_id)
        bay_item = self._find_bay_item(file_id, bay_name)
        if loaded is None or bay_item is None:
            return

        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}

        self._in_tree_change = True
        try:
            self._tree.blockSignals(True)
            for i in range(bay_item.childCount()):
                grp = bay_item.child(i)
                grp_data = grp.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
                if not (isinstance(grp_data, tuple) and grp_data[0] == 'agroup'):
                    continue
                for j in range(grp.childCount()):
                    ch_item = grp.child(j)
                    ch_data = ch_item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
                    if not (isinstance(ch_data, tuple) and ch_data[0] == 'ach'):
                        continue
                    ch = ch_map.get(ch_data[2])
                    target_state = (
                        Qt.CheckState.Checked
                        if ch and ch.signal_role in _STANDARD_SET_ROLES
                        else Qt.CheckState.Unchecked
                    )
                    ch_item.setCheckState(TREE_COL_NAME, target_state)
                self._recalculate_tristate(grp)
            self._recalculate_tristate(bay_item)
            if bay_item.parent():
                self._recalculate_tristate(bay_item.parent())
            self._tree.blockSignals(False)
        finally:
            self._in_tree_change = False

        self._sync_selected_ids(loaded)
        self._rebuild_canvas()

    def _set_bay_check(
        self,
        file_id:  str,
        bay_name: str,
        state:    Qt.CheckState,
    ) -> None:
        """Check or uncheck all channels in a bay.

        Args:
            file_id:  The file this bay belongs to.
            bay_name: Bay label string.
            state:    Target check state.
        """
        loaded   = self._files.get(file_id)
        bay_item = self._find_bay_item(file_id, bay_name)
        if loaded is None or bay_item is None:
            return

        self._in_tree_change = True
        try:
            self._tree.blockSignals(True)
            self._propagate_check_down(bay_item, state)
            bay_item.setCheckState(TREE_COL_NAME, state)
            self._update_ancestors(bay_item)
            self._tree.blockSignals(False)
        finally:
            self._in_tree_change = False

        self._sync_selected_ids(loaded)
        self._rebuild_canvas()

    # ── Start-time anchor (mechanism 2) ───────────────────────────────────────

    def _show_set_start_time_dialog(self, file_id: str) -> None:
        """Open SetStartTimeDialog and apply any corrected epoch.

        Args:
            file_id: The _LoadedFile whose start time to correct.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        dlg = SetStartTimeDialog(loaded.start_epoch, loaded.path.stem, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.anchor_utc is not None:
            loaded.start_epoch = start_epoch_from_datetime(dlg.anchor_utc)
            loaded.timestamp_ok = True
            self._rebuild_canvas()

    # ── Auto-align from frequency cross-correlation (mechanism 3) ─────────────

    def _do_auto_align(self, target_id: str, reference_id: str) -> None:
        """Cross-correlate frequency channels to find alignment offset.

        Dispatches the computation to a background thread.  The UI remains
        responsive while the correlation runs.

        Args:
            target_id:    file_id of the file to align (whose offset to adjust).
            reference_id: file_id of the reference file (fixed, offset = 0).
        """
        target    = self._files.get(target_id)
        reference = self._files.get(reference_id)
        if target is None or reference is None:
            return

        t_freq = self._get_freq_raw(target)
        r_freq = self._get_freq_raw(reference)

        if t_freq is None or r_freq is None:
            QMessageBox.warning(
                self,
                'Auto-align',
                'Both files must have a Frequency channel for auto-alignment.\n'
                'No Frequency channel found in one or both files.',
            )
            return

        # Check epochs are close enough that xcorr makes sense.
        # If they differ by more than 1 day, the user needs to set the start
        # time first via mechanism 1 or 2.
        epoch_gap = abs(target.start_epoch - reference.start_epoch)
        if epoch_gap > 86400:
            QMessageBox.warning(
                self,
                'Auto-align',
                f'The two files are more than 24 hours apart on the time axis\n'
                f'({epoch_gap / 3600:.1f} h gap).\n\n'
                f'Set the approximate start time for "{target.path.stem}" first\n'
                f'via right-click → "Set Start Time…", then retry auto-align.',
            )
            return

        run_in_thread(
            _xcorr_freq_lag,
            t_freq,
            r_freq,
            on_done=lambda lag_s, tid=target_id, rid=reference_id:
                self._on_auto_align_done(tid, rid, lag_s),
            on_error=lambda msg: QMessageBox.critical(
                self, 'Auto-align Error', msg
            ),
        )

    def _on_auto_align_done(
        self,
        target_id:    str,
        reference_id: str,
        lag_s:        float,
    ) -> None:
        """Apply the cross-correlation lag as a per-file offset (UI thread).

        The lag is added to any existing manual offset so the user's prior
        coarse adjustment is preserved.

        Args:
            target_id:    file_id of the file whose offset to update.
            reference_id: file_id of the reference file.
            lag_s:        Time lag in seconds (target lags reference by this much).
        """
        target    = self._files.get(target_id)
        reference = self._files.get(reference_id)
        if target is None or reference is None:
            return

        # Alignment formula:
        #   to make target overlap reference, set target's offset so that its
        #   canvas t[0] matches reference's canvas t[0] + lag_s
        ref_epoch  = self._get_ref_epoch()
        ref_canvas = reference.start_epoch - ref_epoch + self._offsets.get(reference_id, 0.0)
        tgt_canvas = target.start_epoch - ref_epoch

        self._offsets[target_id] = ref_canvas + lag_s - tgt_canvas
        self._update_curves()

        # Reflect in the offset slider label (best-effort — slider range may not
        # cover large offsets, but the curve data is already correct)
        QMessageBox.information(
            self,
            'Auto-align complete',
            f'Applied offset  {self._offsets[target_id] * 1000:+.1f} ms  '
            f'to  "{target.path.stem}"\n'
            f'(cross-correlation lag: {lag_s * 1000:+.1f} ms)',
        )

    @staticmethod
    def _get_freq_raw(loaded: '_LoadedFile') -> Optional[np.ndarray]:
        """Return the raw Frequency channel array for a loaded file, or None.

        Args:
            loaded: The _LoadedFile to search.

        Returns:
            float64 ndarray of frequency values, or None if not found.
        """
        for ch in loaded.record.analogue_channels:
            if getattr(ch, 'signal_role', '') == SignalRole.FREQ:
                d = getattr(ch, 'raw_data', None)
                if d is not None and len(d) > 0:
                    return d.astype(np.float64)
        return None

    def _refresh_tree_label(
        self,
        item:    QTreeWidgetItem,
        file_id: str,
        ch_id:   int,
    ) -> None:
        """Update the axis/stack side tag shown in a channel tree item.

        Args:
            item:    The QTreeWidgetItem to update.
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None:
            return
        _, side = self._stack_assign.get((file_id, ch_id), (4, 'left'))
        side_tag = '[R]' if side == 'right' else '[L]'
        self._tree.blockSignals(True)
        item.setText(TREE_COL_NAME, f'{side_tag} {ch.name}  [{ch.unit or "—"}]')
        self._tree.blockSignals(False)

    def _on_mode_toggled(
        self,
        file_id: str,
        ch_id:   int,
        btn:     QPushButton,
    ) -> None:
        """Toggle a channel between Raw and RMS display mode.

        Switching to RMS triggers background computation if no cached result
        exists.  Switching back to Raw uses the pre-decimated raw_data arrays.

        Args:
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
            btn:     The mode button whose label to update.
        """
        key     = (file_id, ch_id)
        current = self._channel_mode.get(key, 'raw')
        new     = 'rms' if current == 'raw' else 'raw'
        self._channel_mode[key] = new
        btn.setText('RMS' if new == 'rms' else 'Raw')

        if new == 'rms' and key not in self._rms_cache:
            self._compute_rms_for_channel(file_id, ch_id)
        else:
            self._update_curves()

    # ── Offset strip ───────────────────────────────────────────────────────────

    def _add_offset_row(self, loaded: _LoadedFile) -> None:
        """Append one _OffsetRow for ``loaded`` to the offset strip.

        Args:
            loaded: The file whose offset row to add.
        """
        row = _OffsetRow(loaded.file_id, loaded.path.stem[:18])
        row.offset_changed.connect(self._on_offset_changed)
        row.freq_changed.connect(self._on_freq_changed)
        count = self._offset_layout.count()
        self._offset_layout.insertWidget(count - 1, row)

    def _rebuild_offset_strip(self) -> None:
        """Rebuild the offset strip to reflect the current set of loaded files."""
        while self._offset_layout.count() > 1:
            item_w = self._offset_layout.takeAt(0)
            if item_w.widget():
                item_w.widget().deleteLater()
        for loaded in self._files.values():
            self._add_offset_row(loaded)

    def _on_offset_changed(self, file_id: str, offset_s: float) -> None:
        """Store offset and schedule a debounced curve update.

        Args:
            file_id:  The file whose offset changed.
            offset_s: New offset in seconds.
        """
        self._offsets[file_id] = offset_s
        self._update_timer.start()

    def _on_freq_changed(self, file_id: str, freq_hz: float) -> None:
        """Update nominal frequency and invalidate this file's RMS cache.

        Args:
            file_id:  The file whose frequency changed.
            freq_hz:  New nominal frequency in Hz.
        """
        loaded = self._files.get(file_id)
        if loaded:
            loaded.nominal_freq = freq_hz
            for ch in loaded.record.analogue_channels:
                self._rms_cache.pop((file_id, ch.channel_id), None)

    # ── Canvas rebuild ─────────────────────────────────────────────────────────

    def _rebuild_canvas(self) -> None:
        """Clear and rebuild all stacks from scratch.

        Each active stack lives in its own PlotWidget inside self._splitter,
        giving the user native drag-to-resize handles between stacks.
        The digital events strip uses a _DigitalViewBox so mouse-wheel
        scrolls through channels instead of zooming.

        Called on structural changes: file add/remove, checkbox toggle,
        stack assignment change.  Offset / mode changes use _update_curves().
        """
        # ── 1. Disconnect all X-links and remove secondary ViewBoxes ─────────
        # Must happen BEFORE the PlotWidgets are removed from the splitter,
        # otherwise pending resize events reach already-deleted C++ objects.
        for vb2 in self._stack_vb2s.values():
            try:
                vb2.setXLink(None)
            except RuntimeError:
                pass
            try:
                s = vb2.scene()
                if s is not None:
                    s.removeItem(vb2)
            except RuntimeError:
                pass
        for plot in self._stack_plots.values():
            try:
                plot.setXLink(None)
            except RuntimeError:
                pass

        # ── 2. Remove all PlotWidgets from the splitter ───────────────────────
        while self._splitter.count() > 0:
            w = self._splitter.widget(0)
            w.setParent(None)   # detach; Python GC will clean up
            w.deleteLater()

        self._stack_widgets.clear()
        self._stack_plots.clear()
        self._stack_vb2s.clear()
        self._cursor1_lines.clear()
        self._cursor2_lines.clear()
        self._curves.clear()
        self._digital_curves.clear()
        self._digital_row.clear()
        self._time_axes.clear()
        self._ref_plot = None

        active = sorted(self._get_active_stacks())
        if not active:
            self._update_readout()
            return

        ref_plot: Optional[pg.PlotItem] = None

        for row_idx, stack_idx in enumerate(active):
            is_last = row_idx == len(active) - 1

            # ── Digital events strip ──────────────────────────────────────────
            if stack_idx == STACK_DIGITAL:
                n_digital = sum(
                    len(f.selected_digital_ids) for f in self._files.values()
                )
                # _DigitalViewBox: wheel pans Y; X is controlled by X-link
                dig_vb   = _DigitalViewBox()
                dig_t_ax = _TimeAxis(orientation='bottom')
                dig_t_ax.set_cycles_mode(self._xaxis_cycles, self._get_nominal_freq())
                dig_pw   = pg.PlotWidget(
                    background=CANVAS_BG,
                    viewBox=dig_vb,
                    axisItems={'bottom': dig_t_ax},
                )
                self._time_axes[STACK_DIGITAL] = dig_t_ax
                dig_plot: pg.PlotItem = dig_pw.getPlotItem()

                # Height: show up to 8 rows initially; splitter lets user grow
                visible_rows = min(n_digital, 8)
                dig_pw.setMinimumHeight(50)
                self._splitter.addWidget(dig_pw)
                self._splitter.setCollapsible(row_idx, False)

                dig_plot.showGrid(x=True, y=False, alpha=0.15)
                dig_plot.setMenuEnabled(False)
                dig_vb.setMenuEnabled(False)
                dig_vb.setMouseEnabled(x=False, y=True)
                _t_lbl = 'Time (cycles)' if self._xaxis_cycles else 'Time (s)'
                dig_plot.setLabel('bottom', _t_lbl)
                dig_plot.setLabel('left', 'Events')
                dig_plot.hideAxis('right')
                dig_plot.setYRange(
                    -0.3,
                    max(1.0, float(visible_rows)) - 0.3,
                    padding=0,
                )

                if ref_plot is None:
                    ref_plot = dig_plot
                else:
                    dig_plot.setXLink(ref_plot)

                c1d = pg.InfiniteLine(
                    pos=0.0, angle=90, movable=True,
                    pen=pg.mkPen(CURSOR1_COLOUR, width=1, style=Qt.PenStyle.DashLine),
                )
                c1d.setZValue(100)
                c1d.sigDragged.connect(self._on_cursor_dragged)
                if self._cursor1_enabled:
                    dig_plot.addItem(c1d)
                self._cursor1_lines[STACK_DIGITAL] = c1d

                c2d = pg.InfiniteLine(
                    pos=0.0, angle=90, movable=True,
                    pen=pg.mkPen(CURSOR2_COLOUR, width=1, style=Qt.PenStyle.DashLine),
                )
                c2d.setZValue(99)
                c2d.sigDragged.connect(self._on_cursor_dragged)
                if self._cursor2_enabled:
                    dig_plot.addItem(c2d)
                self._cursor2_lines[STACK_DIGITAL] = c2d

                dig_pw.scene().sigMouseClicked.connect(
                    lambda ev, p=dig_plot: self._on_canvas_clicked(ev, p)
                )
                self._stack_widgets[STACK_DIGITAL] = dig_pw
                self._stack_plots[STACK_DIGITAL]   = dig_plot
                # No vb2 for digital strip
                continue

            # ── Analogue stack ────────────────────────────────────────────────
            t_ax = _TimeAxis(orientation='bottom')
            t_ax.set_cycles_mode(self._xaxis_cycles, self._get_nominal_freq())
            pw   = pg.PlotWidget(background=CANVAS_BG, axisItems={'bottom': t_ax})
            self._time_axes[stack_idx] = t_ax
            plot: pg.PlotItem = pw.getPlotItem()
            pw.setMinimumHeight(100)
            self._splitter.addWidget(pw)
            self._splitter.setCollapsible(row_idx, False)

            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.getViewBox().setMenuEnabled(False)

            _t_lbl = 'Time (cycles)' if self._xaxis_cycles else 'Time (s)'
            if is_last:
                plot.setLabel('bottom', _t_lbl)
            else:
                plot.hideAxis('bottom')

            left_label = STACK_LEFT_LABELS[stack_idx]
            if self._pu_mode and stack_idx == 0:
                left_label = 'Voltage (PU)'
            plot.setLabel('left', left_label)

            # Right axis + secondary ViewBox
            plot.showAxis('right')
            right_axis = plot.getAxis('right')
            right_label = STACK_RIGHT_LABELS[stack_idx]
            if right_label:
                right_axis.setLabel(right_label)
            right_axis.hide()

            vb2 = pg.ViewBox()
            plot.scene().addItem(vb2)
            right_axis.linkToView(vb2)
            vb2.setXLink(plot)
            vb2.setMenuEnabled(False)
            plot.getViewBox().sigResized.connect(
                lambda _vb=plot.getViewBox(), _v2=vb2:
                    _v2.setGeometry(_vb.sceneBoundingRect())
            )

            if ref_plot is None:
                ref_plot = plot
            else:
                plot.setXLink(ref_plot)

            c1 = pg.InfiniteLine(
                pos=0.0, angle=90, movable=True,
                pen=pg.mkPen(CURSOR1_COLOUR, width=1, style=Qt.PenStyle.DashLine),
                label='C1', labelOpts={
                    'position': 0.97, 'color': CURSOR1_COLOUR,
                    'fill': pg.mkBrush(40, 40, 40, 180),
                },
            )
            c1.setZValue(100)
            c1.sigDragged.connect(self._on_cursor_dragged)
            if self._cursor1_enabled:
                plot.addItem(c1)
            self._cursor1_lines[stack_idx] = c1

            c2 = pg.InfiniteLine(
                pos=0.0, angle=90, movable=True,
                pen=pg.mkPen(CURSOR2_COLOUR, width=1, style=Qt.PenStyle.DashLine),
                label='C2', labelOpts={
                    'position': 0.90, 'color': CURSOR2_COLOUR,
                    'fill': pg.mkBrush(40, 40, 40, 180),
                },
            )
            c2.setZValue(99)
            c2.sigDragged.connect(self._on_cursor_dragged)
            if self._cursor2_enabled:
                plot.addItem(c2)
            self._cursor2_lines[stack_idx] = c2

            pw.scene().sigMouseClicked.connect(
                lambda ev, p=plot: self._on_canvas_clicked(ev, p)
            )
            self._stack_widgets[stack_idx] = pw
            self._stack_plots[stack_idx]   = plot
            self._stack_vb2s[stack_idx]    = vb2

        self._ref_plot = ref_plot

        # Set equal initial heights; user can resize via splitter handles
        n = self._splitter.count()
        if n > 0:
            self._splitter.setSizes([200] * n)

        self._plot_all_channels()
        self._update_readout()
        self._readout.raise_()   # keep overlay on top after widget rebuild

    def _get_active_stacks(self) -> set[int]:
        """Return stack indices that have at least one checked channel.

        Analogue channels contribute their assigned stack index.
        Any selected digital channel adds STACK_DIGITAL (always last).

        Returns:
            Set of active stack_idx values.
        """
        active: set[int] = set()
        for loaded in self._files.values():
            for ch_id in loaded.selected_ids:
                stack_idx, _ = self._stack_assign.get(
                    (loaded.file_id, ch_id), (4, 'left')
                )
                active.add(stack_idx)
            if loaded.selected_digital_ids:
                active.add(STACK_DIGITAL)
        return active

    def _plot_all_channels(self) -> None:
        """Populate all stack PlotItems and ViewBoxes with channel curves.

        Must be called after _rebuild_canvas has created the stack PlotItems.
        Shows the right axis for stacks that have at least one right-side channel.
        Applies PU Y-range to Stack 0 when PU mode is active.
        """
        has_right: dict[int, bool] = {idx: False for idx in self._stack_plots}
        ref_epoch = self._get_ref_epoch()

        for loaded in self._files.values():
            for ch_id in loaded.selected_ids:
                key = (loaded.file_id, ch_id)
                stack_idx, side = self._stack_assign.get(key, (4, 'left'))
                if stack_idx not in self._stack_plots:
                    continue

                ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
                ch = ch_map.get(ch_id)
                if ch is None:
                    continue

                t_data, y_data = self._channel_xy(loaded, ch, ref_epoch)
                if t_data is None or len(t_data) == 0:
                    continue

                colour = getattr(ch, 'colour', None) or '#AAAAAA'
                if ch.channel_id in loaded.scatter_ids:
                    curve = pg.PlotDataItem(
                        t_data, y_data,
                        pen=None,
                        symbol='o', symbolSize=3,
                        symbolBrush=pg.mkBrush(colour),
                        symbolPen=None,
                        name=f'{loaded.path.stem}/{ch.name}',
                    )
                else:
                    curve = pg.PlotDataItem(
                        t_data, y_data,
                        pen=pg.mkPen(colour, width=1),
                        name=f'{loaded.path.stem}/{ch.name}',
                    )

                if side == 'right':
                    self._stack_vb2s[stack_idx].addItem(curve)
                    has_right[stack_idx] = True
                else:
                    self._stack_plots[stack_idx].addItem(curve)

                self._curves[key] = curve

        for stack_idx, has_r in has_right.items():
            ax = self._stack_plots[stack_idx].getAxis('right')
            if has_r:
                ax.show()
            else:
                ax.hide()

        if self._pu_mode and 0 in self._stack_plots:
            _yr = AppSettings.get('calculation.pu_yrange', 2.0)
            self._stack_plots[0].setYRange(-_yr, _yr, padding=0)

        # ── Digital events strip ──────────────────────────────────────────────
        if STACK_DIGITAL not in self._stack_plots:
            return

        dig_plot = self._stack_plots[STACK_DIGITAL]
        channel_row = 0
        tick_labels: list[tuple[float, str]] = []

        for loaded in self._files.values():
            dch_map = {c.channel_id: c for c in loaded.record.digital_channels}
            for dch_id in sorted(loaded.selected_digital_ids):
                dch = dch_map.get(dch_id)
                if dch is None:
                    continue
                t_data, y_data = self._digital_xy(loaded, dch, ref_epoch, channel_row)
                if t_data is None or len(t_data) == 0:
                    channel_row += 1
                    continue

                colour = getattr(dch, 'colour', '#AAAAAA') or '#AAAAAA'
                curve = pg.PlotDataItem(
                    t_data, y_data,
                    pen=pg.mkPen(colour, width=1.5),
                    fillLevel=float(channel_row),
                    brush=pg.mkBrush(colour + '55'),
                )
                dig_plot.addItem(curve)
                key = (loaded.file_id, dch_id)
                self._digital_curves[key] = curve
                self._digital_row[key]    = channel_row

                label = f'{loaded.path.stem[:6]}/{dch.name[:14]}'
                tick_labels.append((channel_row + DIG_CH_HEIGHT / 2, label))
                channel_row += 1

        if tick_labels:
            dig_plot.getAxis('left').setTicks([tick_labels])
        dig_plot.setYRange(-0.3, max(1.0, float(channel_row)) - 0.3, padding=0)
        self._update_digital_tick_font()

    # ── Digital tick-font auto-size ────────────────────────────────────────────

    def _update_digital_tick_font(self, *_) -> None:
        """Fit Y-axis tick labels in the digital strip without overlap.

        Computes a point size proportional to the pixel height available per
        channel row and applies it to the left axis.  Called once after
        plotting and again whenever the user drags a splitter handle.
        """
        if STACK_DIGITAL not in self._stack_plots or STACK_DIGITAL not in self._stack_widgets:
            return
        n_rows = max(1, len(self._digital_row))
        dig_pw   = self._stack_widgets[STACK_DIGITAL]
        dig_plot = self._stack_plots[STACK_DIGITAL]
        # Usable plot area is roughly 85 % of the widget height
        px_available = dig_pw.height() * 0.85
        if px_available <= 10:
            px_available = 200 * 0.85   # fallback before first layout pass
        px_per_row = px_available / n_rows
        pt = max(6, min(11, int(px_per_row * 0.55)))
        font = QFont()
        font.setPointSize(pt)
        dig_plot.getAxis('left').setStyle(tickFont=font)

    # ── Curve update (no layout rebuild) ──────────────────────────────────────

    def _update_curves(self) -> None:
        """Refresh all curve data arrays without rebuilding the canvas layout.

        Called on: offset slider change, PU mode toggle, RMS result available.
        """
        ref_epoch = self._get_ref_epoch()

        for loaded in self._files.values():
            for ch_id in loaded.selected_ids:
                key = (loaded.file_id, ch_id)
                curve = self._curves.get(key)
                if curve is None:
                    continue

                ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
                ch = ch_map.get(ch_id)
                if ch is None:
                    continue

                t_data, y_data = self._channel_xy(loaded, ch, ref_epoch)
                if t_data is not None and len(t_data) > 0:
                    curve.setData(t_data, y_data)

        if self._pu_mode and 0 in self._stack_plots:
            _yr = AppSettings.get('calculation.pu_yrange', 2.0)
            self._stack_plots[0].setYRange(-_yr, _yr, padding=0)
        elif not self._pu_mode and 0 in self._stack_plots:
            self._stack_plots[0].enableAutoRange(axis='y', enable=True)

        # Update digital channel curves
        for loaded in self._files.values():
            dch_map = {c.channel_id: c for c in loaded.record.digital_channels}
            for dch_id in loaded.selected_digital_ids:
                key = (loaded.file_id, dch_id)
                curve = self._digital_curves.get(key)
                if curve is None:
                    continue
                dch = dch_map.get(dch_id)
                if dch is None:
                    continue
                row = self._digital_row.get(key, 0)
                t_data, y_data = self._digital_xy(loaded, dch, ref_epoch, row)
                if t_data is not None and len(t_data) > 0:
                    curve.setData(t_data, y_data)

    def _channel_xy(
        self,
        loaded:    _LoadedFile,
        ch:        object,
        ref_epoch: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute (t_abs_s, y) display arrays for one channel.

        Time is in absolute seconds relative to ``ref_epoch``.
        Decimation is applied according to LAW 3 (MAX_DISPLAY_PTS).

        Args:
            loaded:    The _LoadedFile owning this channel.
            ch:        The AnalogueChannel.
            ref_epoch: POSIX epoch of the earliest loaded file.

        Returns:
            ``(t_seconds, y_values)`` or ``(None, None)`` if data unavailable.
        """
        key       = (loaded.file_id, getattr(ch, 'channel_id', -1))
        mode      = self._channel_mode.get(key, 'raw')
        offset_s  = self._offsets.get(loaded.file_id, 0.0)
        t_shift   = (loaded.start_epoch - ref_epoch) + offset_s
        record    = loaded.record

        if mode == 'rms':
            cached = self._rms_cache.get(key)
            if cached is not None:
                t_rms, rms_vals = cached
                t_abs = t_rms.astype(np.float64) + t_shift
                y = rms_vals.astype(np.float64)
            else:
                # No computed RMS in cache.  For PMU files (source_format=PMU_CSV)
                # raw_data already contains phasor magnitudes / RMS values — use
                # it directly.  For non-PMU channels the toggle to RMS mode kicks
                # off background computation; until it arrives return nothing.
                if record.source_format not in _PMU_SOURCE_FORMATS:
                    return None, None
                t_raw = record.time_array
                d_raw = getattr(ch, 'raw_data', None)
                if d_raw is None:
                    return None, None
                n = min(len(t_raw), len(d_raw))
                if n == 0:
                    return None, None
                t_abs_raw = t_raw[:n].astype(np.float64) + t_shift
                d_raw_f   = d_raw[:n].astype(np.float64)
                t_abs, y = decimate_uniform(t_abs_raw, d_raw_f, MAX_DISPLAY_PTS)
        else:
            t_raw = record.time_array
            d_raw = getattr(ch, 'raw_data', None)
            if d_raw is None:
                return None, None
            n = min(len(t_raw), len(d_raw))
            if n == 0:
                return None, None
            t_abs_raw = t_raw[:n].astype(np.float64) + t_shift
            d_raw_f   = d_raw[:n].astype(np.float64)
            if record.display_mode == 'TREND':
                t_abs, y = decimate_uniform(t_abs_raw, d_raw_f, MAX_DISPLAY_PTS)
            else:
                t_abs, y = decimate_minmax(t_abs_raw, d_raw_f, MAX_DISPLAY_PTS)

        # PU conversion for voltage channels
        if self._pu_mode:
            divisor = self._get_pu_divisor(loaded.file_id, getattr(ch, 'channel_id', -1))
            if divisor > 0.0:
                y = y / divisor

        return t_abs, y

    # ── RMS computation ────────────────────────────────────────────────────────

    def _compute_rms_for_channel(self, file_id: str, ch_id: int) -> None:
        """Dispatch background RMS computation for one channel.

        Automatically triggered when the user toggles a channel to RMS mode
        and no cached result exists.

        Args:
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None:
            return

        t_raw = loaded.record.time_array.astype(np.float64)
        d_raw = ch.raw_data
        freq  = loaded.nominal_freq

        run_in_thread(
            lambda t=t_raw, d=d_raw, f=freq: compute_cycle_rms(t, d, f),
            on_done=lambda res, fid=file_id, cid=ch_id:
                self._on_rms_done(fid, cid, res),
            on_error=lambda exc: QMessageBox.critical(
                self, 'RMS Error', str(exc)
            ),
        )

    def _on_rms_done(
        self,
        file_id: str,
        ch_id:   int,
        result:  tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Store RMS result and refresh the affected curve.

        Args:
            file_id: The file this result belongs to.
            ch_id:   The channel_id.
            result:  Tuple (t_centres_s, rms_values).
        """
        key = (file_id, ch_id)
        self._rms_cache[key] = result

        loaded = self._files.get(file_id)
        if loaded is None or key not in self._curves:
            return
        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None:
            return

        t_data, y_data = self._channel_xy(loaded, ch, self._get_ref_epoch())
        if t_data is not None and len(t_data) > 0:
            self._curves[key].setData(t_data, y_data)

    # ── Cursors ────────────────────────────────────────────────────────────────

    def _on_cursor_dragged(self, line: pg.InfiniteLine) -> None:
        """Keep all same-colour cursors aligned when any one is dragged.

        Args:
            line: The InfiniteLine that was moved.
        """
        pos = line.pos().x()
        in_c1 = line in self._cursor1_lines.values()
        target = self._cursor1_lines if in_c1 else self._cursor2_lines
        for cursor in target.values():
            if cursor is not line:
                cursor.blockSignals(True)
                cursor.setValue(pos)
                cursor.blockSignals(False)
        self._update_readout()

    def _on_canvas_clicked(self, event, plot: pg.PlotItem) -> None:
        """Right-click within a stack PlotWidget → cursor enable/disable menu.

        Each PlotWidget connects its scene's sigMouseClicked to this slot with
        the owning PlotItem captured via lambda, so we know exactly which
        scene the click came from without cross-scene coordinate confusion.

        Args:
            event: PyQtGraph MouseClickEvent from the PlotWidget's scene.
            plot:  The PlotItem that owns the scene that was clicked.
        """
        if event.button() != Qt.MouseButton.RightButton:
            return
        if not plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            return
        event.accept()

        from PyQt6.QtWidgets import QMenu  # noqa: PLC0415
        menu = QMenu(self)
        fit_act = menu.addAction('Zoom to Fit')
        menu.addSeparator()
        act1 = menu.addAction('Cursor 1  (gold)')
        act1.setCheckable(True)
        act1.setChecked(self._cursor1_enabled)
        act2 = menu.addAction('Cursor 2  (cyan)')
        act2.setCheckable(True)
        act2.setChecked(self._cursor2_enabled)

        chosen = menu.exec(event.screenPos().toPoint())
        if chosen is fit_act:
            self._zoom_to_fit()
        elif chosen is act1:
            self._set_cursor_enabled(1, not self._cursor1_enabled)
        elif chosen is act2:
            self._set_cursor_enabled(2, not self._cursor2_enabled)

    def _set_cursor_enabled(self, n: int, enabled: bool) -> None:
        """Add or remove cursor n (1 or 2) from all active stacks.

        Args:
            n:       Cursor index (1 or 2).
            enabled: True to show, False to hide.
        """
        if n == 1:
            self._cursor1_enabled = enabled
            lines = self._cursor1_lines
        else:
            self._cursor2_enabled = enabled
            lines = self._cursor2_lines

        for stack_idx, plot in self._stack_plots.items():
            cursor = lines.get(stack_idx)
            if cursor is None:
                continue
            if enabled:
                vr = plot.getViewBox().viewRange()
                mid = (vr[0][0] + vr[0][1]) / 2.0
                cursor.setValue(mid)
                plot.addItem(cursor)
            else:
                plot.removeItem(cursor)

        self._update_readout()

    def _update_readout(self) -> None:
        """Emit cursor positions and per-channel values to the measurement panel."""
        _nan = float('nan')
        any_active = self._cursor1_enabled or self._cursor2_enabled
        if not any_active or not self._curves:
            self.readout_updated.emit(_nan, _nan, [])
            return

        # \u2500\u2500 Cursor times \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        c1_line = next(iter(self._cursor1_lines.values()), None)
        c2_line = next(iter(self._cursor2_lines.values()), None)
        t_c1 = c1_line.value() if (self._cursor1_enabled and c1_line) else _nan
        t_c2 = c2_line.value() if (self._cursor2_enabled and c2_line) else _nan

        # \u2500\u2500 Per-channel rows \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        rows: list[tuple[str, str, float, float]] = []
        for loaded in self._files.values():
            ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
            for ch_id in loaded.selected_ids:
                key = (loaded.file_id, ch_id)
                if key not in self._curves:
                    continue
                ch = ch_map.get(ch_id)
                if ch is None:
                    continue
                name    = f'{loaded.path.stem[:8]}/{ch.name}'
                unit    = getattr(ch, 'unit', '') or ''
                val_c1  = self._get_value_at_cursor(t_c1, key) if not math.isnan(t_c1) else _nan
                val_c2  = self._get_value_at_cursor(t_c2, key) if not math.isnan(t_c2) else _nan
                rows.append((name, unit, val_c1, val_c2))

        self.readout_updated.emit(t_c1, t_c2, rows)

    def _reposition_readout(self) -> None:
        """Position the readout overlay within the GLW widget bounds."""
        if self._readout is None or not self._readout.isVisible():
            return
        m  = READOUT_MARGIN
        pw = self._glw.width()
        ph = self._glw.height()
        rw = self._readout.width()
        rh = self._readout.height()
        if not self._readout_pinned:
            self._readout.move(max(0, pw - rw - m), max(0, ph - rh - m))
        else:
            cur = self._readout.pos()
            self._readout.move(
                max(0, min(cur.x(), pw - rw - m)),
                max(0, min(cur.y(), ph - rh - m)),
            )

    def _get_value_at_cursor(self, t: float, key: tuple[str, int]) -> float:
        """Return the nearest display value for ``key`` at time ``t``.

        Reads directly from the live PlotDataItem arrays — no separate snapshot
        dict required.

        Args:
            t:   Absolute time position in seconds (same units as curve X data).
            key: (file_id, ch_id) lookup key into self._curves.

        Returns:
            Nearest Y value, or NaN if the curve has no data.
        """
        curve = self._curves.get(key)
        if curve is None:
            return float('nan')
        x_data, y_data = curve.getData()
        if x_data is None or len(x_data) == 0:
            return float('nan')
        idx = int(np.searchsorted(x_data, t))
        idx = int(np.clip(idx, 0, len(x_data) - 1))
        if idx > 0 and abs(x_data[idx - 1] - t) < abs(x_data[idx] - t):
            idx -= 1
        return float(y_data[idx])

    # ── PU mode ────────────────────────────────────────────────────────────────

    def _on_pu_toggled(self, checked: bool) -> None:
        """Switch PU mode on or off and refresh the voltage stack.

        Args:
            checked: True → PU mode active.
        """
        self._pu_mode = checked
        if self._stack_plots:
            # Update left-axis label on Stack 0
            if 0 in self._stack_plots:
                label = 'Voltage (PU)' if checked else 'Voltage'
                self._stack_plots[0].setLabel('left', label)
            self._update_curves()
        else:
            self._rebuild_canvas()

    def _on_base_kv_changed(self, file_id: str, ch_id: int, value: float) -> None:
        """Store base voltage and refresh curves if PU mode is active.

        Args:
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
            value:   New base voltage in kV (0.0 = not set).
        """
        self._base_kv[(file_id, ch_id)] = value
        if self._pu_mode:
            self._update_curves()

    def _is_voltage_channel(self, file_id: str, ch_id: int) -> bool:
        """Return True if the channel is a voltage type eligible for PU.

        Args:
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return False
        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        return ch is not None and ch.signal_role in _VOLTAGE_ROLES

    def _set_voltage_convention(self, file_id: str, convention: str) -> None:
        """Set the base-kV input convention for a file and refresh.

        Args:
            file_id:    The _LoadedFile to update.
            convention: ``'line_to_line'`` or ``'line_to_earth'``.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        loaded.voltage_convention = convention
        badge = '[L-L]' if convention == 'line_to_line' else '[L-E]'
        if loaded.tree_item is not None:
            self._tree.blockSignals(True)
            loaded.tree_item.setText(
                TREE_COL_NAME, f'📄 {loaded.path.stem}  {badge}')
            self._tree.blockSignals(False)
        if self._pu_mode:
            self._update_curves()

    def _get_pu_divisor(self, file_id: str, ch_id: int) -> float:
        """Return the PU divisor for a voltage channel; 0.0 if not applicable.

        The base kV input is ALWAYS entered in line-to-line terms (e.g. 275 kV).
        The ``voltage_convention`` on the file declares what format the channel
        values are stored in:

        - ``line_to_line``  (default) — channel data is in L-L form:
            divisor = base  (e.g. 275 kV → use 275 directly)

        - ``line_to_earth`` — channel data is in L-E (phase-to-earth) form:
            divisor = base / √3  (e.g. 275 kV → use 275/1.732 = 158.77 kV)

        Args:
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
        """
        base = self._base_kv.get((file_id, ch_id), 0.0)
        if base <= 0.0:
            return 0.0
        loaded = self._files.get(file_id)
        if loaded is None:
            return 0.0
        ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None or ch.signal_role not in _VOLTAGE_ROLES:
            return 0.0
        if loaded.voltage_convention == 'line_to_earth':
            return base / SQRT3   # channel is L-E; derive phase base from L-L input
        return base               # channel is L-L (default); use L-L base directly

    # ── Phasor dialog ──────────────────────────────────────────────────────────

    def _show_phasor_dialog(self) -> None:
        """Create (once) and show / raise the phasor display dialog."""
        if self._phasor_dialog is None:
            self._phasor_dialog = PhasorDialog(self)
        self._phasor_dialog.show()
        self._phasor_dialog.raise_()
        self._phasor_dialog.activateWindow()

    # ── Digital channel XY helper ──────────────────────────────────────────────

    def _digital_xy(
        self,
        loaded:      _LoadedFile,
        dch:         object,
        ref_epoch:   float,
        channel_row: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build step-function display arrays for one digital channel.

        Returns decimated (t_abs, y) where y is the 0/1 state scaled to a
        DIG_CH_HEIGHT band offset by channel_row.  A step-function is created
        by repeating each sample so transitions are vertical.

        Args:
            loaded:      The owning _LoadedFile.
            dch:         A DigitalChannel instance.
            ref_epoch:   Reference POSIX epoch (earliest loaded file).
            channel_row: Y-axis row index for this channel (0, 1, 2, …).

        Returns:
            ``(t_step, y_step)`` ready for PlotDataItem, or ``(None, None)``.
        """
        record   = loaded.record
        offset_s = self._offsets.get(loaded.file_id, 0.0)
        t_shift  = (loaded.start_epoch - ref_epoch) + offset_s

        t_raw = record.time_array
        d_raw = getattr(dch, 'data', None)
        if d_raw is None:
            return None, None
        n = min(len(t_raw), len(d_raw))
        if n == 0:
            return None, None

        t_abs = t_raw[:n].astype(np.float64) + t_shift
        d_f64 = d_raw[:n].astype(np.float64)

        # Decimate before building step data to keep point count manageable
        if len(t_abs) > MAX_DISPLAY_PTS:
            t_abs, d_f64 = decimate_uniform(t_abs, d_f64, MAX_DISPLAY_PTS)
            n = len(t_abs)

        # Build step function: repeat each sample so transitions are vertical
        # Result: [t0, t1, t1, t2, t2, t3, …, tN-1]  (length 2N-1)
        if n > 1:
            t_step = np.empty(2 * n - 1)
            d_step = np.empty(2 * n - 1)
            t_step[0::2] = t_abs
            t_step[1::2] = t_abs[1:]
            d_step[0::2] = d_f64
            d_step[1::2] = d_f64[:-1]
        else:
            t_step = t_abs
            d_step = d_f64

        y_step = d_step * DIG_CH_HEIGHT + channel_row
        return t_step, y_step

    # ── Filter box ─────────────────────────────────────────────────────────────

    def _on_filter_changed(self, text: str) -> None:
        """Show/hide tree items to match ``text`` filter (case-insensitive).

        Parent nodes are shown if any descendant matches.  Matching nodes are
        auto-expanded so results are immediately visible.

        Args:
            text: Current text from the filter QLineEdit.
        """
        text = text.strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            self._apply_filter_recursive(self._tree.topLevelItem(i), text)

    def _apply_filter_recursive(
        self,
        item: QTreeWidgetItem,
        text: str,
    ) -> bool:
        """Recursively show/hide ``item`` based on ``text``.

        Args:
            item: Tree node to evaluate.
            text: Lower-case filter string (empty = show all).

        Returns:
            True if ``item`` should be visible (matches or has matching child).
        """
        if not text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._apply_filter_recursive(item.child(i), text)
            return True

        own_match = text in item.text(TREE_COL_NAME).lower()
        child_visible = False
        for i in range(item.childCount()):
            if self._apply_filter_recursive(item.child(i), text):
                child_visible = True

        visible = own_match or child_visible
        item.setHidden(not visible)
        if child_visible:
            item.setExpanded(True)
        return visible

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_nominal_freq(self) -> float:
        """Return nominal frequency from the first loaded file or AppSettings."""
        for loaded in self._files.values():
            freq = loaded.record.nominal_frequency
            if freq in (50.0, 60.0):
                return freq
        return AppSettings.get('calculation.nominal_frequency', 50.0)

    def _on_cycles_toggled(self, checked: bool) -> None:
        """Switch X-axis tick labels between seconds and cycles without rebuild."""
        self._xaxis_cycles = checked
        freq   = self._get_nominal_freq()
        t_lbl  = 'Time (cycles)' if checked else 'Time (s)'
        for ax in self._time_axes.values():
            ax.set_cycles_mode(checked, freq)
        # Update axis title on every visible bottom axis
        for plot in self._stack_plots.values():
            if plot.getAxis('bottom').isVisible():
                plot.setLabel('bottom', t_lbl)

    def _zoom_to_fit(self) -> None:
        """Reset X range to show the full time span of all rendered curves."""
        if not self._stack_plots:
            return
        t_min, t_max = float('inf'), float('-inf')
        for curve in self._curves.values():
            xd = curve.xData
            if xd is not None and len(xd) > 0:
                t_min = min(t_min, float(xd[0]))
                t_max = max(t_max, float(xd[-1]))
        for curve in self._digital_curves.values():
            xd = curve.xData
            if xd is not None and len(xd) > 0:
                t_min = min(t_min, float(xd[0]))
                t_max = max(t_max, float(xd[-1]))
        if t_min < t_max:
            ref = next(iter(self._stack_plots.values()))
            ref.setXRange(t_min, t_max, padding=0.02)

    def _get_ref_epoch(self) -> float:
        """Return the earliest start_epoch across all loaded files.

        Returns:
            Reference POSIX epoch (float). 0.0 if no files are loaded.
        """
        if not self._files:
            return 0.0
        return min(f.start_epoch for f in self._files.values())

    @staticmethod
    def _make_sep() -> QWidget:
        """Return a thin vertical separator widget for the toolbar."""
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(20)
        sep.setStyleSheet('background: #555555;')
        return sep
