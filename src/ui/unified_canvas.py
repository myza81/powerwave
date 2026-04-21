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

from core.thread_manager import run_in_thread
from engine.decimator import decimate_minmax, decimate_uniform
from engine.rms_calculator import compute_cycle_rms
from engine.rms_merger import start_epoch_from_datetime
from models.channel import SignalRole
from models.disturbance_record import DisturbanceRecord
from parsers.comtrade_parser import ComtradeParser
from parsers.csv_parser import CsvParser
from parsers.excel_parser import ExcelParser
from parsers.pmu_csv_parser import PmuCsvParser, is_pmu_csv

# ── Module constants ───────────────────────────────────────────────────────────

FILE_PANEL_WIDTH:    int   = 300
SLIDER_RANGE:        int   = 3000
OFFSET_DEBOUNCE_MS:  int   = 50
DEFAULT_STEP_MS:     float = 10.0
CANVAS_BG:           str   = '#1E1E1E'
MAX_DISPLAY_PTS:     int   = 2000      # LAW 3 — never render > 4000 pts/channel

CURSOR1_COLOUR: str = '#FFD700'        # gold
CURSOR2_COLOUR: str = '#00E5FF'        # cyan
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


# ── Internal state dataclass ───────────────────────────────────────────────────

@dataclass
class _LoadedFile:
    """Runtime state for one file loaded into the Unified Canvas.

    Attributes:
        file_id:      Unique string key (sequential integer as string).
        path:         Absolute path to the source file.
        record:       Parsed DisturbanceRecord.
        nominal_freq: Active nominal frequency used for RMS computation (Hz).
        selected_ids: Set of analogue channel_ids currently checked in the tree.
        start_epoch:  POSIX epoch float for record.start_time (UTC).
        tree_item:    The top-level QTreeWidgetItem for this file.
    """
    file_id:      str
    path:         Path
    record:       DisturbanceRecord
    nominal_freq: float
    selected_ids: set[int]            = field(default_factory=set)
    start_epoch:  float               = 0.0
    tree_item:    Optional[object]    = field(default=None, repr=False)


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


# ── Main widget ────────────────────────────────────────────────────────────────

class UnifiedCanvasWidget(QWidget):
    """Multi-file, multi-stack analogue waveform overlay tab.

    Loads its own files independently of the main waveform panel.
    Each channel can be displayed as raw waveform or cycle-by-cycle RMS.
    Channels are automatically grouped into parameter stacks (Voltage/Current,
    Freq/Power, etc.) with a shared synchronised X time axis.
    """

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
        self._stack_plots:   dict[int, pg.PlotItem]     = {}
        self._stack_vb2s:    dict[int, pg.ViewBox]      = {}
        self._cursor1_lines: dict[int, pg.InfiniteLine] = {}
        self._cursor2_lines: dict[int, pg.InfiniteLine] = {}
        self._cursor1_enabled: bool = False
        self._cursor2_enabled: bool = False
        self._readout:       Optional[_DraggableLabel]  = None
        self._readout_pinned: bool = False
        self._curves:        dict[tuple[str, int], pg.PlotDataItem] = {}
        self._ref_plot:      Optional[pg.PlotItem] = None

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
            'Right-click a voltage channel in the tree to set its base kV.'
        )
        self._pu_btn.toggled.connect(self._on_pu_toggled)
        layout.addWidget(self._pu_btn)

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

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(False)
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(['Channel', 'Mode', 'Base kV'])
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
        """Build the right panel containing the stacked GraphicsLayoutWidget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(CANVAS_BG)
        layout.addWidget(self._glw, stretch=1)

        # Draggable readout overlay — child of the GLW widget
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

        # Reposition overlay on GLW resize
        self._glw.installEventFilter(self)

        # Right-click on any stack → cursor enable/disable menu
        self._glw.scene().sigMouseClicked.connect(self._on_canvas_clicked)

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

    def _parse_file(self, path: Path) -> _LoadedFile:
        """Parse one file and return a _LoadedFile (runs on background thread).

        Args:
            path: Path to the file to parse.

        Returns:
            _LoadedFile with record populated and all analogue channels selected.
        """
        ext = path.suffix.lower()
        if ext in _COMTRADE_EXT:
            record = ComtradeParser().load(path)
        elif ext in _CSV_EXT:
            record = PmuCsvParser().load(path) if is_pmu_csv(path) else CsvParser().load(path)
        elif ext in _EXCEL_EXT:
            record = ExcelParser().load(path)
        else:
            raise ValueError(f'Unsupported file type: {path.suffix}')

        file_id = str(self._file_counter)
        self._file_counter += 1

        return _LoadedFile(
            file_id=file_id,
            path=path,
            record=record,
            nominal_freq=50.0,
            selected_ids={ch.channel_id for ch in record.analogue_channels},
            start_epoch=start_epoch_from_datetime(record.start_time),
        )

    def _on_file_parsed(self, loaded: _LoadedFile) -> None:
        """Integrate a freshly parsed file into UI state (runs on UI thread).

        Args:
            loaded: The _LoadedFile returned by the background parser.
        """
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
        """Add a file header + one channel row per analogue channel to the tree.

        Args:
            loaded: The _LoadedFile to add.
        """
        file_item = QTreeWidgetItem([f'📄 {loaded.path.stem}'])
        file_item.setData(TREE_COL_NAME, Qt.ItemDataRole.UserRole, loaded.file_id)
        file_item.setFlags(
            file_item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        file_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Checked)

        self._tree.blockSignals(True)
        for ch in loaded.record.analogue_channels:
            key = (loaded.file_id, ch.channel_id)
            mode   = self._channel_mode.get(key, 'raw')
            locked = self._mode_locked.get(key, False)
            _, side = self._stack_assign.get(key, (4, 'left'))
            side_tag = '[R]' if side == 'right' else '[L]'

            ch_item = QTreeWidgetItem([
                f'  {side_tag} {ch.name}  [{ch.unit or "—"}]', '', '',
            ])
            ch_item.setData(TREE_COL_NAME, Qt.ItemDataRole.UserRole, ch.channel_id)
            ch_item.setFlags(ch_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            ch_item.setCheckState(
                TREE_COL_NAME,
                Qt.CheckState.Checked
                if ch.channel_id in loaded.selected_ids
                else Qt.CheckState.Unchecked,
            )
            file_item.addChild(ch_item)

            # Mode toggle button — disabled for locked channels
            btn = QPushButton()
            btn.setFixedSize(54, 18)
            btn.setStyleSheet('font-size: 7pt; padding: 1px;')
            if locked:
                btn.setText(mode.capitalize())
                btn.setEnabled(False)
                btn.setToolTip('Mode is fixed — this channel is in its native format.')
            else:
                btn.setText('Raw' if mode == 'raw' else 'RMS')
                btn.clicked.connect(
                    lambda _checked, fid=loaded.file_id, cid=ch.channel_id, b=btn:
                        self._on_mode_toggled(fid, cid, b)
                )
            self._tree.setItemWidget(ch_item, TREE_COL_MODE, btn)

            # Base kV spinbox for voltage channels
            if self._is_voltage_channel(loaded.file_id, ch.channel_id):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 9999.0)
                spin.setDecimals(1)
                spin.setSuffix(' kV')
                spin.setSpecialValueText('—')
                spin.setFixedWidth(68)
                spin.setStyleSheet('font-size: 7pt;')
                spin.setValue(self._base_kv.get(key, 0.0))
                spin.valueChanged.connect(
                    lambda v, fid=loaded.file_id, cid=ch.channel_id:
                        self._on_base_kv_changed(fid, cid, v)
                )
                self._tree.setItemWidget(ch_item, TREE_COL_BASE, spin)

        self._tree.blockSignals(False)
        self._tree.addTopLevelItem(file_item)
        file_item.setExpanded(True)
        loaded.tree_item = file_item

    def _on_tree_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        """Sync selected_ids and rebuild canvas when a checkbox changes.

        Args:
            item: The item whose checkbox changed.
            col:  Column index (always TREE_COL_NAME).
        """
        parent = item.parent()
        if parent is None:
            # File-level checkbox: propagate to all children in one pass to
            # avoid N separate _rebuild_canvas calls (one per child signal).
            file_id = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
            loaded = self._files.get(file_id)
            if loaded is None:
                return
            state = item.checkState(TREE_COL_NAME)
            self._tree.blockSignals(True)
            if state == Qt.CheckState.Checked:
                loaded.selected_ids = {
                    ch.channel_id for ch in loaded.record.analogue_channels
                }
                for i in range(item.childCount()):
                    item.child(i).setCheckState(TREE_COL_NAME, Qt.CheckState.Checked)
            elif state == Qt.CheckState.Unchecked:
                loaded.selected_ids.clear()
                for i in range(item.childCount()):
                    item.child(i).setCheckState(TREE_COL_NAME, Qt.CheckState.Unchecked)
            # PartiallyChecked is set automatically by ItemIsAutoTristate when
            # individual children change — selected_ids already updated by the
            # per-child path below, nothing more to do here.
            self._tree.blockSignals(False)
            self._rebuild_canvas()
            return

        file_id = parent.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        ch_id   = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        loaded  = self._files.get(file_id)
        if loaded is None or ch_id is None:
            return

        if item.checkState(TREE_COL_NAME) == Qt.CheckState.Checked:
            loaded.selected_ids.add(ch_id)
        else:
            loaded.selected_ids.discard(ch_id)

        self._rebuild_canvas()

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click on a channel item → offer stack reassignment.

        Args:
            pos: Click position in tree viewport coordinates.
        """
        from PyQt6.QtWidgets import QMenu  # noqa: PLC0415
        item = self._tree.itemAt(pos)
        if item is None or item.parent() is None:
            return

        file_item = item.parent()
        file_id = file_item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        ch_id   = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        if file_id is None or ch_id is None:
            return

        key          = (file_id, ch_id)
        curr_stack, curr_side = self._stack_assign.get(key, (4, 'left'))

        menu       = QMenu(self)
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

        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
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
        item.setText(TREE_COL_NAME, f'  {side_tag} {ch.name}  [{ch.unit or "—"}]')
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

        Called on structural changes: file add/remove, channel checkbox toggle,
        stack assignment change.  Offset changes and mode toggles use the
        lighter _update_curves() path instead.
        """
        # Break all X-links before destroying the scene.  PyQtGraph does not
        # disconnect these itself and a pending resize event after clear() will
        # call screenGeometry() on the already-deleted C++ ViewBox objects.
        # vb2 items are added directly to the scene (not to the GLW layout),
        # so glw.clear() does NOT remove them — must be removed explicitly or
        # they persist as ghost ViewBoxes with stale curves still visible.
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

        self._glw.clear()
        self._stack_plots.clear()
        self._stack_vb2s.clear()
        self._cursor1_lines.clear()
        self._cursor2_lines.clear()
        self._curves.clear()
        self._ref_plot = None

        active = sorted(self._get_active_stacks())
        if not active:
            self._update_readout()
            return

        ref_plot: Optional[pg.PlotItem] = None

        for row_idx, stack_idx in enumerate(active):
            is_last = row_idx == len(active) - 1

            plot: pg.PlotItem = self._glw.addPlot(row=row_idx, col=0)
            plot.setMinimumHeight(120)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.getViewBox().setMenuEnabled(False)

            if is_last:
                plot.setLabel('bottom', 'Time (s)')
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

            # Cursor 1 (gold) — z=100
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

            # Cursor 2 (cyan) — z=99
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

            self._stack_plots[stack_idx] = plot
            self._stack_vb2s[stack_idx]  = vb2

        self._ref_plot = ref_plot
        self._plot_all_channels()
        self._update_readout()

    def _get_active_stacks(self) -> set[int]:
        """Return stack indices that have at least one checked channel.

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
                curve = pg.PlotDataItem(
                    t_data,
                    y_data,
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
            self._stack_plots[0].setYRange(-2.0, 2.0, padding=0)

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
            self._stack_plots[0].setYRange(-2.0, 2.0, padding=0)
        elif not self._pu_mode and 0 in self._stack_plots:
            self._stack_plots[0].enableAutoRange(axis='y', enable=True)

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
            if cached is None:
                return None, None
            t_rms, rms_vals = cached
            t_abs = t_rms.astype(np.float64) + t_shift
            y = rms_vals.astype(np.float64)
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

    def _on_canvas_clicked(self, event) -> None:
        """Right-click anywhere on the canvas → cursor enable/disable menu.

        Args:
            event: PyQtGraph MouseClickEvent from the GLW scene.
        """
        if event.button() != Qt.MouseButton.RightButton:
            return
        in_any_plot = any(
            p.getViewBox().sceneBoundingRect().contains(event.scenePos())
            for p in self._stack_plots.values()
        )
        if not in_any_plot:
            return
        event.accept()

        from PyQt6.QtWidgets import QMenu  # noqa: PLC0415
        menu = QMenu(self)
        act1 = menu.addAction('Cursor 1  (gold)')
        act1.setCheckable(True)
        act1.setChecked(self._cursor1_enabled)
        act2 = menu.addAction('Cursor 2  (cyan)')
        act2.setCheckable(True)
        act2.setChecked(self._cursor2_enabled)

        chosen = menu.exec(event.screenPos().toPoint())
        if chosen is act1:
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
        """Rebuild the readout overlay text from active cursor positions."""
        if self._readout is None:
            return
        any_active = self._cursor1_enabled or self._cursor2_enabled
        if not any_active or not self._curves:
            self._readout.hide()
            return

        lines: list[str] = []

        for n, cursor_lines, enabled in (
            (1, self._cursor1_lines, self._cursor1_enabled),
            (2, self._cursor2_lines, self._cursor2_enabled),
        ):
            if not enabled:
                continue
            cursor = next(iter(cursor_lines.values()), None)
            if cursor is None:
                continue
            t = cursor.value()
            lines.append(f'C{n}  {t:.3f} s')
            for loaded in self._files.values():
                for ch_id in loaded.selected_ids:
                    key = (loaded.file_id, ch_id)
                    if key not in self._curves:
                        continue
                    val = self._get_value_at_cursor(t, key)
                    ch_map = {c.channel_id: c for c in loaded.record.analogue_channels}
                    ch = ch_map.get(ch_id)
                    if ch is None:
                        continue
                    name = f'{loaded.path.stem[:8]}/{ch.name}'
                    short = name if len(name) <= 18 else name[:17] + '\u2026'
                    val_str = f'{val:.4f}' if not np.isnan(val) else '---'
                    lines.append(f'  {short:<18}  {val_str}')

        if self._cursor1_enabled and self._cursor2_enabled:
            c1 = next(iter(self._cursor1_lines.values()), None)
            c2 = next(iter(self._cursor2_lines.values()), None)
            if c1 and c2:
                dx = abs(c2.value() - c1.value())
                lines.append(f'\u0394X = {dx:.3f} s')

        self._readout.setText('\n'.join(lines))
        self._readout.adjustSize()
        self._readout.show()
        self._reposition_readout()

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

    def _get_pu_divisor(self, file_id: str, ch_id: int) -> float:
        """Return the PU divisor for a voltage channel; 0.0 if not applicable.

        Phase-to-phase (V_LINE): divisor = V_base.
        Phase-to-earth / residual: divisor = V_base / √3.

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
        return base if ch.signal_role in _LINE_VOLTAGE_ROLES else base / SQRT3

    # ── Phasor dialog ──────────────────────────────────────────────────────────

    def _show_phasor_dialog(self) -> None:
        """Create (once) and show / raise the phasor display dialog."""
        if self._phasor_dialog is None:
            self._phasor_dialog = PhasorDialog(self)
        self._phasor_dialog.show()
        self._phasor_dialog.raise_()
        self._phasor_dialog.activateWindow()

    # ── Helpers ────────────────────────────────────────────────────────────────

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
