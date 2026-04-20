"""
src/ui/rms_converter_dock.py

RMS Converter — standalone dock for multi-file cycle-by-cycle RMS analysis.

Layout:
  ┌─ Toolbar ──────────────────────────────────────────────────────┐
  │ [Add File] [Remove] [Compute RMS]  Tolerance: [10 ms ▼]       │
  ├─ Left (file tree) ──────┬─ Right ─────────────────────────────┤
  │ QTreeWidget              │ pg waveform (RMS trend lines)       │
  │  ▶ JMHE_500kV           ├─────────────────────────────────────┤
  │    ☑ VR                  │ QTableWidget — merged RMS values    │
  │    ☑ IY                 ├─────────────────────────────────────┤
  │  ▶ PMU.csv              │ Per-file offset strip               │
  │    ☑ V1                  │  JMHE: [-][slider][+] 0.0ms step:10│
  └─────────────────────────┴─────────────────────────────────────┘
  │ [Export CSV] [Export Excel]    ⚠ N NaN cells                  │
  └────────────────────────────────────────────────────────────────┘

Data flow:
  1. User adds files → parsed on background thread → DisturbanceRecord
  2. User selects channels, sets freq per file → [Compute RMS]
  3. RMS computation on background thread → rms_results stored per file
  4. Merger builds common time grid (nearest-neighbour, user tolerance)
  5. Waveform + table updated; offset slider re-runs merge only (fast)

Architecture: Presentation layer (ui/) — imports engine/ and models/ only (LAW 1).
              Heavy work dispatched via core.thread_manager (LAW 2).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.thread_manager import run_in_thread
from engine.rms_calculator import compute_rms_for_record
from engine.rms_merger import (
    DEFAULT_TOLERANCE_S,
    MergeResult,
    RmsChannelData,
    format_time_column,
    merge_rms_channels,
    start_epoch_from_datetime,
)
from models.channel import SignalRole
from models.disturbance_record import DisturbanceRecord
from parsers.comtrade_parser import ComtradeParser
from parsers.csv_parser import CsvParser
from parsers.excel_parser import ExcelParser
from parsers.pmu_csv_parser import PmuCsvParser, is_pmu_csv

# ── Module constants ───────────────────────────────────────────────────────────

TREE_COL_NAME:   int = 0    # column index: channel name / file stem
# column index: base voltage kV spinbox (voltage channels)
TREE_COL_BASE:   int = 1
FILE_PANEL_WIDTH: int = 300  # px — fixed left-panel width
SLIDER_RANGE:     int = 3000  # ± slider units (actual offset = units × step_s)
OFFSET_DEBOUNCE_MS: int = 50  # ms — debounce before re-merging on slider drag
DEFAULT_STEP_MS:  float = 10.0
WAVEFORM_BG:      str = '#1E1E1E'
NAN_WARN_COLOUR:  str = '#FF8800'
NAN_OK_COLOUR:    str = '#888888'

CURSOR1_COLOUR:   str = '#FFD700'   # gold
CURSOR2_COLOUR:   str = '#00E5FF'   # cyan
READOUT_MARGIN:   int = 8           # px from bottom-right edge of plot viewport

SQRT3: float = 1.7320508075688772   # √3 for phase-to-earth PU conversion

# Signal roles that default to the right (current) Y-axis
_CURRENT_ROLES: frozenset[str] = frozenset({
    SignalRole.I_PHASE,
    SignalRole.I_EARTH,
    SignalRole.I1_PMU,
})

# Voltage roles eligible for PU conversion
_VOLTAGE_ROLES: frozenset[str] = frozenset({
    SignalRole.V_PHASE,
    SignalRole.V_LINE,
    SignalRole.V_RESIDUAL,
    SignalRole.V1_PMU,
    SignalRole.SEQ_RMS,
})

# Phase-to-phase roles: divide by V_base (not V_base/√3)
_LINE_VOLTAGE_ROLES: frozenset[str] = frozenset({
    SignalRole.V_LINE,
})

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


# ── Draggable readout label ────────────────────────────────────────────────────

class _DraggableLabel(QLabel):
    """QLabel that can be repositioned by mouse drag.

    Emits ``user_moved`` after each drag step so the parent can track that the
    user has manually positioned it (suppressing future auto-repositioning).
    """

    user_moved: pyqtSignal = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # QPoint of mouse-down in label coords
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
                new_pos.setX(
                    max(0, min(new_pos.x(), parent.width() - self.width())))
                new_pos.setY(
                    max(0, min(new_pos.y(), parent.height() - self.height())))
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
    """Runtime state for one file loaded into the RMS Converter.

    Attributes:
        file_id:      Unique string key (sequential integer as string).
        path:         Absolute path to the source file.
        record:       Parsed DisturbanceRecord.
        nominal_freq: Active nominal frequency for this file (Hz).
        selected_ids: Set of analogue channel_ids currently checked.
        rms_results:  Dict channel_id → (t_from_start, rms_values).
                      Empty until Compute RMS is run.
        start_epoch:  POSIX epoch float for the record's start_time.
        tree_item:    The QTreeWidgetItem for this file in the file tree.
    """
    file_id:      str
    path:         Path
    record:       DisturbanceRecord
    nominal_freq: float
    selected_ids: set[int] = field(default_factory=set)
    rms_results:  dict[int, tuple[np.ndarray, np.ndarray]
                       ] = field(default_factory=dict)
    start_epoch:  float = 0.0
    tree_item:    Optional[QTreeWidgetItem] = field(default=None, repr=False)


# ── Offset row widget ──────────────────────────────────────────────────────────

class _OffsetRow(QWidget):
    """One horizontal strip controlling the time offset for one file.

    Signals:
        offset_changed: Emitted when the offset value changes.
            Args: (file_id: str, offset_s: float)
        freq_changed: Emitted when the frequency selector changes.
            Args: (file_id: str, freq_hz: float)
    """

    offset_changed: pyqtSignal = pyqtSignal(str, float)
    freq_changed:   pyqtSignal = pyqtSignal(str, float)

    def __init__(self, file_id: str, display_name: str, parent: Optional[QWidget] = None) -> None:
        """Initialise an offset row for ``file_id``.

        Args:
            file_id:      The _LoadedFile.file_id this row controls.
            display_name: Short filename shown as a label.
            parent:       Optional parent widget.
        """
        super().__init__(parent)
        self._file_id = file_id
        self._step_s = DEFAULT_STEP_MS / 1000.0
        self._offset_s = 0.0
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # File label
        name_lbl = QLabel(display_name)
        name_lbl.setFixedWidth(140)
        name_lbl.setStyleSheet('color: #CCCCCC; font-size: 8pt;')
        layout.addWidget(name_lbl)

        # Frequency selector
        layout.addWidget(QLabel('Freq:'))
        from PyQt6.QtWidgets import QComboBox
        self._freq_combo = QComboBox()
        self._freq_combo.addItems(['50 Hz', '60 Hz'])
        self._freq_combo.setFixedWidth(60)
        self._freq_combo.currentIndexChanged.connect(self._on_freq_changed)
        layout.addWidget(self._freq_combo)

        layout.addWidget(QLabel('  Offset:'))

        # Minus button
        btn_minus = QPushButton('−')
        btn_minus.setFixedWidth(24)
        btn_minus.clicked.connect(self._step_minus)
        layout.addWidget(btn_minus)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-SLIDER_RANGE, SLIDER_RANGE)
        self._slider.setValue(0)
        self._slider.setFixedWidth(160)
        self._slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self._slider)

        # Plus button
        btn_plus = QPushButton('+')
        btn_plus.setFixedWidth(24)
        btn_plus.clicked.connect(self._step_plus)
        layout.addWidget(btn_plus)

        # Offset display label
        self._offset_lbl = QLabel('0.0 ms')
        self._offset_lbl.setFixedWidth(70)
        self._offset_lbl.setStyleSheet('color: #AAAAAA; font-size: 8pt;')
        layout.addWidget(self._offset_lbl)

        # Step size spinbox
        layout.addWidget(QLabel('Step:'))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.1, 10_000.0)
        self._step_spin.setValue(DEFAULT_STEP_MS)
        self._step_spin.setSuffix(' ms')
        self._step_spin.setFixedWidth(90)
        self._step_spin.valueChanged.connect(self._on_step_changed)
        layout.addWidget(self._step_spin)

        layout.addStretch()

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def offset_s(self) -> float:
        """Current time offset in seconds."""
        return self._offset_s

    # ── Private ───────────────────────────────────────────────────────────────

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
        """Adjust slider position to keep actual offset constant when step changes."""
        self._updating = True
        self._step_s = new_ms / 1000.0
        new_pos = int(round(self._offset_s / self._step_s)
                      ) if self._step_s > 0 else 0
        self._slider.setValue(
            int(np.clip(new_pos, -SLIDER_RANGE, SLIDER_RANGE)))
        self._updating = False

    def _on_freq_changed(self, index: int) -> None:
        freq = 50.0 if index == 0 else 60.0
        self.freq_changed.emit(self._file_id, freq)


# ── Main dock widget ───────────────────────────────────────────────────────────

class RmsConverterDock(QDockWidget):
    """Dock widget for multi-file cycle-by-cycle RMS analysis and export.

    Independent from the main canvas — files are loaded fresh into this dock.
    No signals are shared with app_state.

    Usage::

        dock = RmsConverterDock(parent=main_window)
        main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the dock layout and wire all internal signals."""
        super().__init__('RMS Converter', parent)
        self.setObjectName('RmsConverterDock')
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # ── State ────────────────────────────────────────────────────────────
        # file_id → _LoadedFile
        self._files:        dict[str, _LoadedFile] = {}
        self._file_counter: int = 0
        self._offsets:      dict[str, float] = {}  # file_id → offset_s
        self._tolerance_s:  float = DEFAULT_TOLERANCE_S
        self._merge_result: Optional[MergeResult] = None
        # (file_id, ch_id) → display name
        self._col_names:    dict[tuple[str, int], str] = {}
        self._waveform_curves: dict[tuple[str, int], pg.PlotDataItem] = {}
        # (file_id, ch_id) → 'left' | 'right'  — axis assignment per channel
        self._axis_assignment: dict[tuple[str, int], str] = {}
        # Per-unit mode state
        self._pu_mode: bool = False
        # (file_id, ch_id) → base voltage in kV (0.0 = not set)
        self._base_kv: dict[tuple[str, int], float] = {}
        # Secondary ViewBox and axis — initialised in _build_right_panel
        self._vb2:        Optional[pg.ViewBox] = None
        self._right_axis: Optional[pg.AxisItem] = None
        # Crosshair cursor state — initialised in _build_right_panel
        self._cursor1_enabled: bool = False
        self._cursor2_enabled: bool = False
        self._cursor1: Optional[pg.InfiniteLine] = None
        self._cursor2: Optional[pg.InfiniteLine] = None
        self._readout: Optional[_DraggableLabel] = None
        self._readout_pinned: bool = False   # True once user drags it
        # Snapshot of last rendered data for cursor value lookup
        self._t_rel:         Optional[np.ndarray] = None
        self._rms_channels:  list[RmsChannelData] = []

        # Debounce timer for slider → re-merge
        self._merge_timer = QTimer(self)
        self._merge_timer.setSingleShot(True)
        self._merge_timer.setInterval(OFFSET_DEBOUNCE_MS)
        self._merge_timer.timeout.connect(self._run_merge)

        # ── Build UI ─────────────────────────────────────────────────────────
        root = QWidget()
        self.setWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_toolbar())

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.addWidget(self._build_file_panel())
        content_splitter.addWidget(self._build_right_panel())
        content_splitter.setSizes([FILE_PANEL_WIDTH, 900])
        root_layout.addWidget(content_splitter, stretch=1)

        root_layout.addWidget(self._build_bottom_bar())

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        """Reposition the readout overlay when the waveform widget is resized."""
        if obj is self._waveform and event.type() == QEvent.Type.Resize:
            self._reposition_readout()
        return super().eventFilter(obj, event)

    # ── UI builders ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        """Build the top toolbar strip with file controls and tolerance selector."""
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

        layout.addWidget(self._make_separator())

        compute_btn = QPushButton('Compute RMS')
        compute_btn.setStyleSheet('font-weight: bold; color: #44FFAA;')
        compute_btn.clicked.connect(self._on_compute_rms)
        layout.addWidget(compute_btn)

        layout.addWidget(self._make_separator())

        layout.addWidget(QLabel('Tolerance:'))
        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(1.0, 500.0)
        self._tol_spin.setValue(DEFAULT_TOLERANCE_S * 1000.0)
        self._tol_spin.setSuffix(' ms')
        self._tol_spin.setFixedWidth(90)
        self._tol_spin.valueChanged.connect(self._on_tolerance_changed)
        layout.addWidget(self._tol_spin)

        layout.addWidget(self._make_separator())

        self._pu_btn = QPushButton('PU Mode')
        self._pu_btn.setCheckable(True)
        self._pu_btn.setChecked(False)
        self._pu_btn.setToolTip(
            'Toggle between actual RMS values and per-unit (PU) values.\n'
            'Right-click a voltage channel in the file tree to set its base kV.'
        )
        self._pu_btn.toggled.connect(self._on_pu_toggled)
        layout.addWidget(self._pu_btn)

        layout.addStretch()
        return bar

    def _build_file_panel(self) -> QWidget:
        """Build the left file+channel selection tree."""
        panel = QWidget()
        panel.setFixedWidth(FILE_PANEL_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel(' Files & Channels')
        hdr.setStyleSheet(
            'background: #3A3A3A; color: #AAAAAA; font-size: 8pt; padding: 3px;')
        layout.addWidget(hdr)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(False)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(['Channel', 'Base kV'])
        self._tree.setStyleSheet(
            'background: #252525; color: #DDDDDD; font-size: 8pt;')
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(
            TREE_COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        self._tree.header().setSectionResizeMode(
            TREE_COL_BASE, QHeaderView.ResizeMode.Fixed
        )
        self._tree.header().resizeSection(TREE_COL_BASE, 76)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(
            self._on_tree_context_menu)
        layout.addWidget(self._tree)
        return panel

    def _build_right_panel(self) -> QWidget:
        """Build the right panel: waveform + table + offset strip (vertical splitter)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        v_splitter = QSplitter(Qt.Orientation.Vertical)

        # Waveform — primary PlotItem (left Y-axis) + secondary ViewBox (right Y-axis)
        self._waveform = pg.GraphicsLayoutWidget()
        self._waveform.setBackground(WAVEFORM_BG)
        self._waveform.setMinimumHeight(120)
        self._plot = self._waveform.addPlot(row=0, col=0)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel('bottom', 'Time (s)')
        self._plot.setLabel('left', 'Voltage')
        self._plot.getAxis('left').setStyle(tickFont=None)

        # Secondary ViewBox for right (current) Y-axis.
        # Use PlotItem's existing right AxisItem — do NOT add a new one to the
        # layout at (2,2) as that cell is already occupied and triggers a warning.
        self._plot.showAxis('right')
        self._right_axis = self._plot.getAxis('right')
        self._right_axis.setLabel('Current')
        self._right_axis.hide()   # hidden until at least one channel is assigned there

        self._vb2 = pg.ViewBox()
        self._plot.scene().addItem(self._vb2)
        self._right_axis.linkToView(self._vb2)
        self._vb2.setXLink(self._plot)

        # Keep vb2 geometry locked to the primary ViewBox
        self._plot.getViewBox().sigResized.connect(self._sync_vb2)

        # ── Crosshair cursors ─────────────────────────────────────────────────
        # Disable PyQtGraph's default right-click menu so we can show ours
        self._plot.getViewBox().setMenuEnabled(False)
        self._vb2.setMenuEnabled(False)

        self._cursor1 = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(CURSOR1_COLOUR, width=1, style=Qt.PenStyle.DashLine),
            label='C1', labelOpts={'position': 0.97, 'color': CURSOR1_COLOUR,
                                   'fill': pg.mkBrush(40, 40, 40, 180)},
        )
        self._cursor1.sigDragged.connect(self._on_cursor_moved)

        self._cursor2 = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(CURSOR2_COLOUR, width=1, style=Qt.PenStyle.DashLine),
            label='C2', labelOpts={'position': 0.90, 'color': CURSOR2_COLOUR,
                                   'fill': pg.mkBrush(40, 40, 40, 180)},
        )
        self._cursor2.sigDragged.connect(self._on_cursor_moved)

        # Readout panel — draggable QLabel overlay on the waveform widget.
        # Using a child widget (not a scene item) avoids data-coordinate positioning
        # problems and gives reliable semi-transparent rendering on all platforms.
        self._readout = _DraggableLabel(self._waveform)
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
            lambda: setattr(self, '_readout_pinned', True))
        self._readout.hide()

        # Reposition overlay whenever the waveform widget is resized
        self._waveform.installEventFilter(self)

        # Right-click on waveform scene → cursor enable/disable menu
        self._waveform.scene().sigMouseClicked.connect(self._on_waveform_clicked)

        v_splitter.addWidget(self._waveform)

        # Table
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            'QTableWidget { background: #1E1E1E; color: #DDDDDD; font-size: 8pt; }'
            'QHeaderView::section { background: #3A3A3A; color: #CCCCCC; }'
            'QTableWidget::item:alternate { background: #252525; }'
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._table.horizontalHeader().customContextMenuRequested.connect(
            self._on_header_context_menu
        )
        self._table.setMinimumHeight(120)
        v_splitter.addWidget(self._table)

        v_splitter.setSizes([200, 250])
        layout.addWidget(v_splitter, stretch=1)

        # Offset strip (scrollable, one row per file)
        offset_hdr = QLabel(' Time Offset Controls')
        offset_hdr.setStyleSheet(
            'background: #3A3A3A; color: #AAAAAA; font-size: 8pt; padding: 3px;')
        layout.addWidget(offset_hdr)

        self._offset_scroll = QScrollArea()
        self._offset_scroll.setWidgetResizable(True)
        self._offset_scroll.setMaximumHeight(150)
        self._offset_scroll.setStyleSheet('background: #222222;')
        self._offset_container = QWidget()
        self._offset_layout = QVBoxLayout(self._offset_container)
        self._offset_layout.setContentsMargins(0, 0, 0, 0)
        self._offset_layout.setSpacing(0)
        self._offset_layout.addStretch()
        self._offset_scroll.setWidget(self._offset_container)
        layout.addWidget(self._offset_scroll)

        return panel

    def _build_bottom_bar(self) -> QWidget:
        """Build the export bar with CSV/Excel buttons and NaN status."""
        bar = QWidget()
        bar.setStyleSheet('background: #2D2D2D;')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        csv_btn = QPushButton('Export CSV')
        csv_btn.clicked.connect(self._on_export_csv)
        layout.addWidget(csv_btn)

        xls_btn = QPushButton('Export Excel')
        xls_btn.clicked.connect(self._on_export_excel)
        layout.addWidget(xls_btn)

        layout.addWidget(self._make_separator())

        self._nan_label = QLabel('No data')
        self._nan_label.setStyleSheet(
            f'color: {NAN_OK_COLOUR}; font-size: 8pt;')
        layout.addWidget(self._nan_label)

        layout.addStretch()
        return bar

    # ── File loading ──────────────────────────────────────────────────────────

    def _on_add_file(self) -> None:
        """Open file dialog and load selected file on background thread."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add File — RMS Converter', '', _FILE_FILTER
        )
        for path_str in paths:
            path = Path(path_str)
            run_in_thread(
                self._parse_file,
                path,
                on_done=self._on_file_parsed,
                on_error=self._on_parse_error,
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
            if is_pmu_csv(path):
                record = PmuCsvParser().load(path)
            else:
                record = CsvParser().load(path)
        elif ext in _EXCEL_EXT:
            record = ExcelParser().load(path)
        else:
            raise ValueError(f'Unsupported file type: {path.suffix}')

        file_id = str(self._file_counter)
        self._file_counter += 1
        selected = {ch.channel_id for ch in record.analogue_channels}
        start_epoch = start_epoch_from_datetime(record.start_time)

        # Build default column names and axis assignments
        stem = path.stem[:12]
        for ch in record.analogue_channels:
            key = (file_id, ch.channel_id)
            self._col_names[key] = f'{stem}_{ch.name}_RMS'
            # Current roles → right axis; everything else → left axis
            self._axis_assignment[key] = (
                'right' if ch.signal_role in _CURRENT_ROLES else 'left'
            )

        return _LoadedFile(
            file_id=file_id,
            path=path,
            record=record,
            nominal_freq=50.0,
            selected_ids=selected,
            start_epoch=start_epoch,
        )

    def _on_file_parsed(self, loaded: _LoadedFile) -> None:
        """Called on UI thread after background parse completes.

        Args:
            loaded: The freshly parsed _LoadedFile.
        """
        self._files[loaded.file_id] = loaded
        self._offsets[loaded.file_id] = 0.0
        self._add_tree_item(loaded)
        self._add_offset_row(loaded)

    def _on_parse_error(self, exc: Exception) -> None:
        """Show error dialog if file parsing fails.

        Args:
            exc: The exception raised by the parser.
        """
        QMessageBox.critical(self, 'Load Error', str(exc))

    # ── File tree ─────────────────────────────────────────────────────────────

    def _add_tree_item(self, loaded: _LoadedFile) -> None:
        """Add a file+channel tree entry for ``loaded``.

        Args:
            loaded: The _LoadedFile to add to the tree.
        """
        file_item = QTreeWidgetItem([f'📄 {loaded.path.stem}'])
        file_item.setData(
            TREE_COL_NAME, Qt.ItemDataRole.UserRole, loaded.file_id)
        file_item.setFlags(
            file_item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        file_item.setCheckState(TREE_COL_NAME, Qt.CheckState.Checked)

        self._tree.blockSignals(True)
        for ch in loaded.record.analogue_channels:
            axis = self._axis_assignment.get(
                (loaded.file_id, ch.channel_id), 'left')
            axis_tag = '[R]' if axis == 'right' else '[L]'
            ch_item = QTreeWidgetItem(
                [f'  {axis_tag} {ch.name}  [{ch.unit or "—"}]', ''])
            ch_item.setData(
                TREE_COL_NAME, Qt.ItemDataRole.UserRole, ch.channel_id)
            ch_item.setFlags(ch_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            ch_item.setCheckState(
                TREE_COL_NAME,
                Qt.CheckState.Checked if ch.channel_id in loaded.selected_ids
                else Qt.CheckState.Unchecked,
            )
            file_item.addChild(ch_item)

            # Embed base-kV spinbox for voltage channels
            if self._is_voltage_channel(loaded.file_id, ch.channel_id):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 9999.0)
                spin.setDecimals(1)
                spin.setSuffix(' kV')
                spin.setSpecialValueText('—')   # 0.0 → "—" means "not set"
                spin.setFixedWidth(74)
                spin.setStyleSheet('font-size: 7pt;')
                spin.setValue(self._base_kv.get(
                    (loaded.file_id, ch.channel_id), 0.0))
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
        """Sync selected_ids on the _LoadedFile when a checkbox changes.

        Args:
            item: The item that changed.
            col:  Column index (always TREE_COL_NAME here).
        """
        parent = item.parent()
        if parent is None:
            return  # file-level item (tristate handled by Qt automatically)

        file_id = parent.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        ch_id = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        loaded = self._files.get(file_id)
        if loaded is None or ch_id is None:
            return

        if item.checkState(TREE_COL_NAME) == Qt.CheckState.Checked:
            loaded.selected_ids.add(ch_id)
        else:
            loaded.selected_ids.discard(ch_id)

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click on a channel item → offer left/right axis assignment.

        Only channel-level items (children of file items) show this menu.
        File-level items are ignored.

        Args:
            pos: Click position in tree widget coordinates.
        """
        item = self._tree.itemAt(pos)
        if item is None or item.parent() is None:
            return   # clicked on a file item or empty space

        file_item = item.parent()
        file_id = file_item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        ch_id = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        if file_id is None or ch_id is None:
            return

        key = (file_id, ch_id)
        current = self._axis_assignment.get(key, 'left')

        from PyQt6.QtWidgets import QMenu  # noqa: PLC0415
        menu = QMenu(self)
        left_action = menu.addAction('→ Left Axis  (Voltage)')
        right_action = menu.addAction('→ Right Axis (Current)')
        left_action.setCheckable(True)
        right_action.setCheckable(True)
        left_action.setChecked(current == 'left')
        right_action.setChecked(current == 'right')

        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        new_axis = 'right' if chosen is right_action else 'left'
        if new_axis == current:
            return

        self._axis_assignment[key] = new_axis
        self._update_tree_item_label(item, file_id, ch_id, new_axis)
        # Re-draw waveform with updated assignment (merge data unchanged)
        if self._merge_result is not None:
            channels = self._collect_rms_channels()
            self._update_waveform(self._merge_result, channels)

    def _update_tree_item_label(
        self,
        item: QTreeWidgetItem,
        file_id: str,
        ch_id: int,
        axis: str,
    ) -> None:
        """Refresh the axis tag shown on a channel tree item.

        Args:
            item:    The QTreeWidgetItem to update.
            file_id: The file this channel belongs to.
            ch_id:   The channel_id.
            axis:    'left' or 'right'.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        ch_map = {ch.channel_id: ch for ch in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None:
            return
        tag = '[R]' if axis == 'right' else '[L]'
        self._tree.blockSignals(True)
        item.setText(TREE_COL_NAME, f'  {tag} {ch.name}  [{ch.unit or "—"}]')
        self._tree.blockSignals(False)

    def _collect_rms_channels(self) -> list[RmsChannelData]:
        """Build the RmsChannelData list from current file state.

        Returns:
            List of RmsChannelData for all selected channels that have RMS results.
        """
        channels: list[RmsChannelData] = []
        for loaded in self._files.values():
            for ch_id in loaded.selected_ids:
                if ch_id not in loaded.rms_results:
                    continue
                t_from_start, rms_vals = loaded.rms_results[ch_id]
                col_name = self._col_names.get(
                    (loaded.file_id, ch_id),
                    f'{loaded.path.stem}_{ch_id}_RMS',
                )
                channels.append(RmsChannelData(
                    file_id=loaded.file_id,
                    channel_id=ch_id,
                    col_name=col_name,
                    t_from_start=t_from_start,
                    rms=rms_vals,
                    start_epoch=loaded.start_epoch,
                ))
        return channels

    def _on_remove_file(self) -> None:
        """Remove the currently selected file from the tree and state dicts."""
        items = self._tree.selectedItems()
        if not items:
            return
        # Walk up to top-level item
        item = items[0]
        while item.parent() is not None:
            item = item.parent()

        file_id = item.data(TREE_COL_NAME, Qt.ItemDataRole.UserRole)
        if file_id is None:
            return

        self._files.pop(file_id, None)
        self._offsets.pop(file_id, None)

        idx = self._tree.indexOfTopLevelItem(item)
        self._tree.takeTopLevelItem(idx)

        self._rebuild_offset_strip()
        self._run_merge()

    # ── Offset strip ──────────────────────────────────────────────────────────

    def _add_offset_row(self, loaded: _LoadedFile) -> None:
        """Append one _OffsetRow for ``loaded`` to the offset strip.

        Args:
            loaded: The file whose offset row to add.
        """
        row = _OffsetRow(loaded.file_id, loaded.path.stem[:18])
        row.offset_changed.connect(self._on_offset_changed)
        row.freq_changed.connect(self._on_freq_changed)
        # Insert before the trailing stretch
        count = self._offset_layout.count()
        self._offset_layout.insertWidget(count - 1, row)

    def _rebuild_offset_strip(self) -> None:
        """Rebuild the offset strip to match current loaded files."""
        while self._offset_layout.count() > 1:  # keep trailing stretch
            item = self._offset_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for loaded in self._files.values():
            row = _OffsetRow(loaded.file_id, loaded.path.stem[:18])
            row.offset_changed.connect(self._on_offset_changed)
            row.freq_changed.connect(self._on_freq_changed)
            count = self._offset_layout.count()
            self._offset_layout.insertWidget(count - 1, row)

    def _on_offset_changed(self, file_id: str, offset_s: float) -> None:
        """Store updated offset and schedule a re-merge.

        Args:
            file_id:  The file whose offset changed.
            offset_s: New offset in seconds.
        """
        self._offsets[file_id] = offset_s
        self._merge_timer.start()

    def _on_freq_changed(self, file_id: str, freq_hz: float) -> None:
        """Update nominal frequency for a file; mark RMS as stale.

        Args:
            file_id:  The file whose frequency changed.
            freq_hz:  New nominal frequency (50.0 or 60.0).
        """
        loaded = self._files.get(file_id)
        if loaded:
            loaded.nominal_freq = freq_hz
            loaded.rms_results.clear()   # force recompute on next Compute RMS

    def _on_tolerance_changed(self, val_ms: float) -> None:
        """Update snap tolerance and re-merge.

        Args:
            val_ms: New tolerance in milliseconds.
        """
        self._tolerance_s = val_ms / 1000.0
        self._merge_timer.start()

    # ── RMS computation ───────────────────────────────────────────────────────

    def _on_compute_rms(self) -> None:
        """Launch background RMS computation for all files with pending work."""
        files_to_compute = [
            f for f in self._files.values() if not f.rms_results and f.selected_ids
        ]
        if not files_to_compute:
            self._run_merge()
            return

        # Snapshot required info before handing to thread (avoid cross-thread record access)
        for loaded in files_to_compute:
            file_id = loaded.file_id
            record = loaded.record
            selected_ids = list(loaded.selected_ids)
            nominal_freq = loaded.nominal_freq

            run_in_thread(
                lambda r=record, ids=selected_ids, freq=nominal_freq: (
                    compute_rms_for_record(r, ids, freq)
                ),
                on_done=lambda results, fid=file_id: self._on_rms_done(
                    fid, results),
                on_error=lambda exc: QMessageBox.critical(
                    self, 'RMS Error', str(exc)),
            )

    def _on_rms_done(
        self,
        file_id: str,
        results: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Store RMS results and trigger merge (runs on UI thread via signal).

        Args:
            file_id: The file whose RMS computation finished.
            results: Dict channel_id → (t_from_start, rms_values).
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return
        loaded.rms_results = results
        # If all files now have results, merge and display
        if all(f.rms_results for f in self._files.values() if f.selected_ids):
            self._run_merge()

    # ── Merge + display ───────────────────────────────────────────────────────

    def _run_merge(self) -> None:
        """Build RmsChannelData list and run nearest-neighbour merge."""
        channels = self._collect_rms_channels()
        if not channels:
            return

        result = merge_rms_channels(channels, self._offsets, self._tolerance_s)
        self._merge_result = result
        self._update_waveform(result, channels)
        self._update_table(result, channels)
        self._update_nan_label(result)

    def _sync_vb2(self) -> None:
        """Keep the secondary ViewBox geometry aligned with the primary ViewBox."""
        if self._vb2 is not None:
            self._vb2.setGeometry(self._plot.getViewBox().sceneBoundingRect())

    def _update_waveform(
        self,
        result: MergeResult,
        channels: list[RmsChannelData],
    ) -> None:
        """Redraw all RMS trend lines, routing each channel to left or right Y-axis.

        Voltage channels (and all 'left'-assigned channels) go to the primary
        PlotItem ViewBox.  Current channels (and 'right'-assigned channels) go
        to the secondary ViewBox linked to the right AxisItem.

        Args:
            result:   The current MergeResult.
            channels: The channel descriptors used to build result.
        """
        self._plot.clear()
        if self._vb2 is not None:
            self._vb2.clear()
        self._waveform_curves.clear()

        if len(result.t_common) == 0:
            return

        # relative seconds from first point
        t_rel = result.t_common - result.t_common[0]
        self._t_rel = t_rel
        self._rms_channels = list(channels)

        left_label = 'Voltage (PU)' if self._pu_mode else 'Voltage'
        self._plot.setLabel('left', left_label)

        has_right = False
        for col_idx, rms_ch in enumerate(channels):
            colour = self._get_channel_colour(
                rms_ch.file_id, rms_ch.channel_id)
            mask = ~np.isnan(result.data_2d[:, col_idx])
            if not np.any(mask):
                continue

            y_data = result.data_2d[mask, col_idx].copy()
            if self._pu_mode:
                divisor = self._get_pu_divisor(
                    rms_ch.file_id, rms_ch.channel_id)
                if divisor > 0.0:
                    y_data = y_data / divisor

            curve = pg.PlotDataItem(
                t_rel[mask],
                y_data,
                pen=pg.mkPen(colour, width=1),
                name=rms_ch.col_name,
            )

            axis = self._axis_assignment.get(
                (rms_ch.file_id, rms_ch.channel_id), 'left'
            )
            if axis == 'right' and self._vb2 is not None:
                self._vb2.addItem(curve)
                has_right = True
            else:
                self._plot.addItem(curve)

            self._waveform_curves[(rms_ch.file_id, rms_ch.channel_id)] = curve

        # Show right axis only when at least one channel is assigned there
        if self._right_axis is not None:
            self._right_axis.setVisible(has_right)

        # PU mode default Y range: 0 … +1.5 pu for voltage left axis
        if self._pu_mode:
            self._plot.setYRange(0.0, 1.5, padding=0)
        else:
            self._plot.enableAutoRange(axis='y', enable=True)

        self._sync_vb2()
        self._update_readout()

    def _update_table(self, result: MergeResult, channels: list[RmsChannelData]) -> None:
        """Populate the QTableWidget from the MergeResult.

        Uses setUpdatesEnabled(False) during bulk write for performance.
        When PU mode is active, voltage columns are converted to per-unit values.

        Args:
            result:   The current MergeResult.
            channels: The channel descriptors aligned with result.data_2d columns.
        """
        n_rows = len(result.t_common)
        n_cols = len(result.col_names) + 1   # +1 for Time column

        # Pre-compute PU divisors for each column (0.0 = no conversion)
        pu_divisors: list[float] = []
        for rms_ch in channels:
            if self._pu_mode:
                pu_divisors.append(self._get_pu_divisor(
                    rms_ch.file_id, rms_ch.channel_id))
            else:
                pu_divisors.append(0.0)

        self._table.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        try:
            self._table.setRowCount(n_rows)
            self._table.setColumnCount(n_cols)

            headers = ['Time'] + result.col_names
            self._table.setHorizontalHeaderLabels(headers)

            time_strs = format_time_column(
                result.t_common, result.has_timestamps)

            for row in range(n_rows):
                t_item = QTableWidgetItem(time_strs[row])
                t_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, 0, t_item)

                for col_idx in range(len(result.col_names)):
                    val = result.data_2d[row, col_idx]
                    if np.isnan(val):
                        cell = QTableWidgetItem('')
                        cell.setBackground(pg.mkColor('#3A2200'))
                    else:
                        divisor = pu_divisors[col_idx]
                        display_val = val / divisor if divisor > 0.0 else val
                        cell = QTableWidgetItem(f'{display_val:.4f}')
                        cell.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        )
                    self._table.setItem(row, col_idx + 1, cell)
        finally:
            self._table.blockSignals(False)
            self._table.setUpdatesEnabled(True)

        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )

    def _update_nan_label(self, result: MergeResult) -> None:
        """Update the NaN status label in the bottom bar.

        Args:
            result: The current MergeResult.
        """
        n_nan = len(result.nan_cells)
        if n_nan == 0:
            self._nan_label.setText('No missing values')
            self._nan_label.setStyleSheet(
                f'color: {NAN_OK_COLOUR}; font-size: 8pt;')
        else:
            self._nan_label.setText(
                f'⚠ {n_nan} missing cell{"s" if n_nan > 1 else ""}')
            self._nan_label.setStyleSheet(
                f'color: {NAN_WARN_COLOUR}; font-size: 8pt;')

    # ── Header rename ─────────────────────────────────────────────────────────

    def _on_header_context_menu(self, pos) -> None:
        """Right-click on a table column header → rename dialog.

        Args:
            pos: Position of the right-click in header coordinates.
        """
        col = self._table.horizontalHeader().logicalIndexAt(pos)
        if col <= 0:   # col 0 is Time — not renameable
            return
        current = self._table.horizontalHeaderItem(col)
        if current is None:
            return
        name, ok = QInputDialog.getText(
            self, 'Rename Column', 'New column name:', text=current.text()
        )
        if ok and name.strip():
            self._table.setHorizontalHeaderItem(
                col, QTableWidgetItem(name.strip()))
            # Update col_names in merge result and _col_names map
            if self._merge_result and col - 1 < len(self._merge_result.col_names):
                self._merge_result.col_names[col - 1] = name.strip()
            # Sync back to _col_names dict
            ch_idx = col - 1
            all_channel_keys = [
                (fid, chid)
                for fid, loaded in self._files.items()
                for chid in loaded.selected_ids
                if chid in loaded.rms_results
            ]
            if ch_idx < len(all_channel_keys):
                self._col_names[all_channel_keys[ch_idx]] = name.strip()

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export_csv(self) -> None:
        """Export the merged RMS table to a CSV file."""
        if not self._check_export_ready():
            return
        if self._confirm_nan_export():
            path, _ = QFileDialog.getSaveFileName(
                self, 'Export CSV', '', 'CSV Files (*.csv)'
            )
            if path:
                self._write_csv(Path(path))

    def _on_export_excel(self) -> None:
        """Export the merged RMS table to an Excel file."""
        if not self._check_export_ready():
            return
        if self._confirm_nan_export():
            path, _ = QFileDialog.getSaveFileName(
                self, 'Export Excel', '', 'Excel Files (*.xlsx)'
            )
            if path:
                self._write_excel(Path(path))

    def _check_export_ready(self) -> bool:
        """Return True if there is data to export; show warning otherwise."""
        if self._merge_result is None or len(self._merge_result.t_common) == 0:
            QMessageBox.warning(self, 'No Data', 'Run Compute RMS first.')
            return False
        return True

    def _confirm_nan_export(self) -> bool:
        """If NaN cells exist, ask user whether to proceed with blank cells.

        Returns:
            True if user confirms or there are no NaN cells.
        """
        result = self._merge_result
        if result is None or not result.nan_cells:
            return True
        n = len(result.nan_cells)
        reply = QMessageBox.question(
            self,
            'Missing Values',
            f'{n} cell{"s" if n > 1 else ""} have no data and will be exported as blank.\n\n'
            'Proceed with export?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _write_csv(self, path: Path) -> None:
        """Write merged table to CSV.

        Args:
            path: Destination file path.
        """
        result = self._merge_result
        assert result is not None
        time_strs = format_time_column(result.t_common, result.has_timestamps)
        headers = self._collect_current_headers()

        try:
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row_idx, t_str in enumerate(time_strs):
                    row_data = [t_str]
                    for col_idx in range(result.data_2d.shape[1]):
                        val = result.data_2d[row_idx, col_idx]
                        row_data.append('' if np.isnan(val) else f'{val:.4f}')
                    writer.writerow(row_data)
            QMessageBox.information(
                self, 'Export Complete', f'Saved to:\n{path}')
        except OSError as exc:
            QMessageBox.critical(self, 'Export Error', str(exc))

    def _write_excel(self, path: Path) -> None:
        """Write merged table to Excel using openpyxl.

        Args:
            path: Destination file path.
        """
        try:
            import openpyxl  # optional dependency  # noqa: PLC0415
        except ImportError:
            QMessageBox.critical(
                self, 'Missing Dependency',
                'openpyxl is required for Excel export.\n'
                'Install with: pip install openpyxl'
            )
            return

        result = self._merge_result
        assert result is not None
        time_strs = format_time_column(result.t_common, result.has_timestamps)
        headers = self._collect_current_headers()

        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = 'RMS Data'
            ws.append(headers)
            for row_idx, t_str in enumerate(time_strs):
                row_data = [t_str]
                for col_idx in range(result.data_2d.shape[1]):
                    val = result.data_2d[row_idx, col_idx]
                    row_data.append(None if np.isnan(
                        val) else round(float(val), 4))
                ws.append(row_data)
            wb.save(path)
            QMessageBox.information(
                self, 'Export Complete', f'Saved to:\n{path}')
        except OSError as exc:
            QMessageBox.critical(self, 'Export Error', str(exc))

    # ── Crosshair cursors ─────────────────────────────────────────────────────

    def _on_waveform_clicked(self, event) -> None:
        """Show cursor enable/disable context menu on right-click inside the plot.

        Args:
            event: PyQtGraph MouseClickEvent from the scene.
        """
        from PyQt6.QtCore import Qt as _Qt  # noqa: PLC0415
        if event.button() != _Qt.MouseButton.RightButton:
            return
        # Only respond to clicks within the primary ViewBox bounds
        if not self._plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
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
        """Add or remove cursor n (1 or 2) from the plot.

        Args:
            n:       Cursor index (1 or 2).
            enabled: True to show, False to hide.
        """
        cursor = self._cursor1 if n == 1 else self._cursor2
        if cursor is None:
            return
        if n == 1:
            self._cursor1_enabled = enabled
        else:
            self._cursor2_enabled = enabled

        if enabled:
            # Place cursor at the centre of the current X range
            vr = self._plot.getViewBox().viewRange()
            mid = (vr[0][0] + vr[0][1]) / 2.0
            cursor.setValue(mid)
            self._plot.addItem(cursor)
        else:
            self._plot.removeItem(cursor)

        self._update_readout()

    def _on_cursor_moved(self) -> None:
        """Called while a cursor is being dragged — refreshes the readout panel."""
        self._update_readout()

    def _update_readout(self) -> None:
        """Rebuild the readout QLabel text from active cursor positions."""
        if self._readout is None:
            return

        any_active = self._cursor1_enabled or self._cursor2_enabled
        if not any_active or self._merge_result is None or self._t_rel is None:
            self._readout.hide()
            return

        lines: list[str] = []

        for n, cursor, enabled in (
            (1, self._cursor1, self._cursor1_enabled),
            (2, self._cursor2, self._cursor2_enabled),
        ):
            if not enabled or cursor is None:
                continue
            t = cursor.value()
            lines.append(f'C{n}  {t:.3f} s')
            for col_idx, rms_ch in enumerate(self._rms_channels):
                val = self._get_value_at_t(t, col_idx)
                if np.isnan(val):
                    val_str = '---'
                else:
                    if self._pu_mode:
                        div = self._get_pu_divisor(
                            rms_ch.file_id, rms_ch.channel_id)
                        if div > 0.0:
                            val = val / div
                    val_str = f'{val:.4f}'
                name = self._merge_result.col_names[col_idx]
                short = name if len(name) <= 18 else name[:17] + '…'
                lines.append(f'  {short:<18}  {val_str}')

        if self._cursor1_enabled and self._cursor2_enabled:
            dx = abs(self._cursor2.value() - self._cursor1.value())
            lines.append(f'ΔX = {dx:.3f} s')

        self._readout.setText('\n'.join(lines))
        self._readout.adjustSize()
        self._readout.show()
        self._reposition_readout()

    def _reposition_readout(self) -> None:
        """Position the readout QLabel within the waveform widget.

        If the user has not yet dragged the panel, it auto-snaps to the
        bottom-right corner.  After the user drags it, only clamping is applied
        so the panel stays within bounds when the waveform is resized.
        """
        if self._readout is None or not self._readout.isVisible():
            return
        m = READOUT_MARGIN
        pw = self._waveform.width()
        ph = self._waveform.height()
        rw = self._readout.width()
        rh = self._readout.height()

        if not self._readout_pinned:
            # Auto snap to bottom-right
            self._readout.move(max(0, pw - rw - m), max(0, ph - rh - m))
        else:
            # User has positioned it manually — only clamp to keep it in view
            cur = self._readout.pos()
            self._readout.move(
                max(0, min(cur.x(), pw - rw - m)),
                max(0, min(cur.y(), ph - rh - m)),
            )

    def _get_value_at_t(self, t: float, col_idx: int) -> float:
        """Return the nearest RMS value for col_idx at relative time t (seconds).

        Args:
            t:       Cursor X position in relative seconds (t_rel).
            col_idx: Column index into merge_result.data_2d.

        Returns:
            Nearest data value, or NaN if out of range or no data.
        """
        if self._merge_result is None or self._t_rel is None:
            return float('nan')
        t_arr = self._t_rel
        if len(t_arr) == 0:
            return float('nan')
        idx = int(np.searchsorted(t_arr, t))
        idx = int(np.clip(idx, 0, len(t_arr) - 1))
        if idx > 0 and abs(t_arr[idx - 1] - t) < abs(t_arr[idx] - t):
            idx -= 1
        return float(self._merge_result.data_2d[idx, col_idx])

    # ── Per-unit mode ─────────────────────────────────────────────────────────

    def _on_pu_toggled(self, checked: bool) -> None:
        """Switch between actual RMS and per-unit display modes.

        Args:
            checked: True → PU mode; False → actual RMS values.
        """
        self._pu_mode = checked
        if self._merge_result is not None and len(self._merge_result.t_common) > 0:
            channels = self._collect_rms_channels()
            self._update_waveform(self._merge_result, channels)
            self._update_table(self._merge_result, channels)

    def _on_base_kv_changed(self, file_id: str, ch_id: int, value: float) -> None:
        """Store updated base voltage and refresh PU display if active.

        Called by the inline spinbox in the file tree (value 0.0 = not set).

        Args:
            file_id: The _LoadedFile.file_id.
            ch_id:   The analogue channel_id.
            value:   New base voltage in kV (0.0 means not set).
        """
        self._base_kv[(file_id, ch_id)] = value
        if self._pu_mode and self._merge_result is not None:
            channels = self._collect_rms_channels()
            self._update_waveform(self._merge_result, channels)
            self._update_table(self._merge_result, channels)

    def _is_voltage_channel(self, file_id: str, ch_id: int) -> bool:
        """Return True if the channel is a voltage type eligible for PU conversion.

        Args:
            file_id: The _LoadedFile.file_id.
            ch_id:   The analogue channel_id.
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return False
        ch_map = {ch.channel_id: ch for ch in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        return ch is not None and ch.signal_role in _VOLTAGE_ROLES

    def _get_pu_divisor(self, file_id: str, ch_id: int) -> float:
        """Return the PU divisor for a voltage channel; 0.0 if not applicable.

        Phase-to-phase (V_LINE): divisor = V_base.
        Phase-to-earth / residual: divisor = V_base / √3.
        Returns 0.0 if base voltage is not set or channel is not a voltage type.

        Args:
            file_id: The _LoadedFile.file_id.
            ch_id:   The analogue channel_id.
        """
        base = self._base_kv.get((file_id, ch_id), 0.0)
        if base <= 0.0:
            return 0.0
        loaded = self._files.get(file_id)
        if loaded is None:
            return 0.0
        ch_map = {ch.channel_id: ch for ch in loaded.record.analogue_channels}
        ch = ch_map.get(ch_id)
        if ch is None or ch.signal_role not in _VOLTAGE_ROLES:
            return 0.0
        if ch.signal_role in _LINE_VOLTAGE_ROLES:
            return base
        return base / SQRT3

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_current_headers(self) -> list[str]:
        """Read current column headers from the QTableWidget (respects renames).

        Returns:
            List of header strings starting with 'Time'.
        """
        headers = []
        for col in range(self._table.columnCount()):
            item = self._table.horizontalHeaderItem(col)
            headers.append(item.text() if item else f'Col{col}')
        return headers

    def _get_channel_colour(self, file_id: str, channel_id: int) -> str:
        """Return the colour string for a channel, falling back to white.

        Args:
            file_id:    The _LoadedFile.file_id.
            channel_id: The channel_id to look up.

        Returns:
            Hex colour string (e.g. '#FF4444').
        """
        loaded = self._files.get(file_id)
        if loaded is None:
            return '#FFFFFF'
        ch_map = {ch.channel_id: ch for ch in loaded.record.analogue_channels}
        ch = ch_map.get(channel_id)
        return ch.colour if ch and ch.colour else '#FFFFFF'

    @staticmethod
    def _make_separator() -> QWidget:
        """Return a 1px vertical separator widget."""
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet('background: #555555;')
        return sep
