"""
src/ui/channel_canvas.py

ChannelCanvas — main waveform / trend rendering canvas.

Renders all visible channels from a DisturbanceRecord on a stacked
multi-plot layout powered by PyQtGraph + OpenGL.

Layout (BEN32-style, top → bottom):
  For each bay:
    [ROW_HEIGHT_BAY_HEADER px invisible spacer — matches LabelPanel bay header]
    [ROW_HEIGHT_ANALOGUE px analogue PlotItem per visible analogue channel]
    [ROW_HEIGHT_DIGITAL px digital PlotItem per visible digital channel]
  [TIME_AXIS_HEIGHT px time-axis PlotItem — single shared bottom axis]

Row ordering is determined by utils.channel_ordering.get_ordered_rows(),
the single source of truth shared with LabelPanel.  This guarantees
pixel-perfect vertical alignment at all scroll positions.

Y-axes are hidden on all plots — channel names live in LabelPanel.
All X axes are linked to ONE reference plot (star topology, O(n) signals).

Analogue channels:
  WAVEFORM mode (sample_rate >= 200 Hz) — continuous PlotDataItem line
  TREND mode    (sample_rate <  200 Hz) — ScatterPlotItem, size=4

Digital channels:
  Two overlaid PlotDataItems:
    low_line  — dim baseline at y=0.10 (always visible)
    step_line — bold grey bar at y=0.70 when high, y=0.10 when low
  Y range locked to [DIG_Y_MIN, DIG_Y_MAX]; stepMode='right' for correct edges.

Decimation:
  ALL decimation is pre-computed on the background thread by
  engine.decimator.prepare_display_data() before record_loaded is emitted.
  channel_canvas.py reads ch._display_t / ch._display_d and calls
  setData() only — zero computation on the UI thread (LAW 2, LAW 3).

Viewport-aware re-decimation:
  On zoom or pan, _update_viewport() re-decimates the visible time slice
  at full resolution.  A 16ms QTimer debounce prevents re-decimation on
  every pixel of a smooth drag (keeping pan at 60fps).  The mask uses
  record._t_display (display units) — NOT record.time_array (raw seconds).

Architecture: Presentation layer (ui/) — imports core/, models/, pyqtgraph,
              and engine/ (service layer, permitted by LAW 1 upward-only rule).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsSimpleTextItem, QMenu

from core.app_state import app_state
from engine.decimator import (
    MAX_ANALOGUE_POINTS,
    MAX_DIGITAL_POINTS,
    decimate_digital,
    decimate_minmax,
    decimate_uniform,
)
from models.channel import AnalogueChannel, DigitalChannel
from models.disturbance_record import DisturbanceRecord
from utils.channel_ordering import (
    ROW_HEIGHT_ANALOGUE,
    ROW_HEIGHT_BAY_HEADER,
    ROW_HEIGHT_DIGITAL,
    get_ordered_rows,
)

# ── Module-level PyQtGraph configuration ─────────────────────────────────────
pg.setConfigOptions(
    useOpenGL=True,
    antialias=False,
    foreground='w',
    background='#1E1E1E',
)

# ── Module constants ───────────────────────────────────────────────────────────

BACKGROUND_COLOUR: str      = '#1E1E1E'
GRID_ALPHA: float           = 0.15
WAVEFORM_PEN_WIDTH: int     = 1
TREND_SCATTER_SIZE: int     = 4

TIME_AXIS_HEIGHT: int       = 30     # px — bottom shared time axis
TIME_AXIS_LABEL_FALLBACK: str = 'Time'

# Digital rendering style (BEN32)
DIG_BASELINE_Y: float       = 0.1
DIG_Y_MIN: float            = -0.1
DIG_Y_MAX: float            = 1.2
DIG_FILL_PEN: str           = '#BBBBBB'
DIG_FILL_BRUSH: str         = '#888888'
DIG_FILL_PEN_WIDTH: int     = 2
DIG_BASELINE_PEN: str       = '#555555'
DIG_BASELINE_PEN_WIDTH: int = 1

TRIGGER_COLOUR: str         = '#FF4444'
TRIGGER_LINE_WIDTH: int     = 2
TRIGGER_LABEL_POS: float    = 0.95

CURSOR_A_COLOUR: str        = '#FFFF00'
CURSOR_B_COLOUR: str        = '#FF8800'
CURSOR_PEN_WIDTH: float     = 1.5
CURSOR_LABEL_POS: float     = 0.05   # near top of each plot
CURSOR_A_INIT: float        = 0.0    # display units — trigger position
CURSOR_B_INIT_MS: float     = 100.0  # ms offset for WAVEFORM/ms records

AXIS_LABEL_COLOUR: str      = '#AAAAAA'

FAULT_WINDOW_S: float       = 0.200   # ±200 ms around trigger for fault zoom
VP_DEBOUNCE_MS: int         = 16      # 1 frame debounce for viewport updates
_Y_AUTOFIT_PADDING: float   = 0.075  # 7.5 % padding above/below data range


class _SceneCursorLine(QGraphicsLineItem):
    """Full-canvas-height cursor line drawn directly on the QGraphicsScene.

    Replaces one ``pg.InfiniteLine`` per plot (392 objects for JMHE) with
    two objects total — one per cursor — regardless of channel count.

    The line spans from ``scene.sceneRect().top()`` to ``bottom()`` and is
    kept at the correct scene-x coordinate by ``ChannelCanvas._update_scene_cursors()``.
    It is draggable from any scroll position: mouse events are mapped from
    scene coordinates back to ViewBox data coordinates via ``mapSceneToView``.

    Args:
        cursor_id: 0 = cursor A, 1 = cursor B.
        canvas:    Owning ChannelCanvas (for coordinate transforms and callbacks).
        pen:       Pen used to draw the line.
        label:     Text shown near the top of the line ('A' or 'B').
    """

    _HIT_HALF_WIDTH: int = 5   # px hit area either side of the drawn line

    def __init__(
        self,
        cursor_id: int,
        canvas: 'ChannelCanvas',
        pen: QPen,
        label: str,
    ) -> None:
        super().__init__()
        self._cursor_id = cursor_id
        self._canvas    = canvas
        self._dragging  = False
        self.setPen(pen)
        self.setZValue(100)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

        self._label = QGraphicsSimpleTextItem(label, parent=self)
        self._label.setBrush(pen.color())
        self._label.setZValue(101)

    def setLine(self, x1: float, y1: float, x2: float, y2: float) -> None:  # type: ignore[override]
        """Override to keep the 'A'/'B' label pinned near the top of the line."""
        super().setLine(x1, y1, x2, y2)
        self._label.setPos(x1 + 2.0, y1 + 4.0)

    def shape(self) -> QPainterPath:
        """Widen the hit area to ±_HIT_HALF_WIDTH px for easy grabbing."""
        path = QPainterPath()
        ln   = self.line()
        w    = self._HIT_HALF_WIDTH
        y0   = min(ln.y1(), ln.y2())
        h    = abs(ln.y2() - ln.y1()) or 1.0
        path.addRect(QRectF(ln.x1() - w, y0, w * 2.0, h))
        return path

    def mousePressEvent(self, event) -> None:
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging or self._canvas._ref_plot is None:
            return
        vb       = self._canvas._ref_plot.getViewBox()
        view_pos = vb.mapSceneToView(event.scenePos())
        self._canvas._on_cursor_dragged(self._cursor_id, float(view_pos.x()))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        event.accept()


class ChannelCanvas(pg.GraphicsLayoutWidget):
    """Stacked multi-channel waveform canvas (BEN32 style).

    No Y-axis labels on individual plots — channel names live in the
    adjacent LabelPanel.  A single shared time axis sits at the bottom.
    All X axes are linked in star topology to the first analogue plot
    (or first plot of any kind if no analogue channels are present).

    All decimation is pre-computed on the background thread by
    ``engine.decimator.prepare_display_data()``.  On zoom or pan,
    ``_update_viewport()`` re-decimates the visible window slice at full
    resolution, debounced to one frame (16ms) for 60fps performance.

    Usage::

        canvas = ChannelCanvas()
        canvas.load_record(record)          # record already has _display_t/_display_d
        canvas.update_channel_visibility(3, False)
        canvas.zoom_to_fault()              # snap to ±200ms window
    """

    def __init__(self, parent=None) -> None:
        """Initialise an empty canvas with dark engineering theme."""
        super().__init__(parent=parent)
        self.setBackground(BACKGROUND_COLOUR)

        # Pin the scene to the top-left of the QGraphicsView.
        # Default alignment is AlignCenter which shifts rows down when the
        # view is taller than the scene, breaking alignment with LabelPanel.
        self.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        # Zero internal spacing so row pixel heights are exact
        self.ci.layout.setSpacing(0)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)

        self._analogue_plots:  dict[int, pg.PlotItem]                           = {}
        self._digital_plots:   dict[int, pg.PlotItem]                           = {}
        self._analogue_curves: dict[int, pg.PlotDataItem | pg.ScatterPlotItem] = {}
        self._digital_curves:  dict[int, pg.PlotDataItem]                      = {}
        self._trigger_lines: list[pg.InfiniteLine]                     = []
        self._ref_plot: Optional[pg.PlotItem]                          = None   # X-link reference
        self._record: Optional[DisturbanceRecord]                      = None   # current record

        # Two scene-level cursor lines span the full canvas height.
        # _SceneCursorLine handles dragging from any scroll position.
        # This replaces the per-plot InfiniteLine approach (392 objects → 2).
        self._cursor_pos_a:   float                        = CURSOR_A_INIT
        self._cursor_pos_b:   float                        = CURSOR_B_INIT_MS  # updated in load_record
        self._scene_cursor_a: Optional[_SceneCursorLine]  = None
        self._scene_cursor_b: Optional[_SceneCursorLine]  = None

        # Performance: re-entrancy guard and viewport threshold (Fix 2, Fix 3)
        self._updating_viewport: bool                      = False
        self._last_viewport:     tuple                     = (None, None)

        # Viewport debounce timer — fires _on_vp_timer 16ms after last range change
        self._vp_timer: QTimer = QTimer(self)
        self._vp_timer.setSingleShot(True)
        self._vp_timer.setInterval(VP_DEBOUNCE_MS)
        self._vp_timer.timeout.connect(self._on_vp_timer)
        self._pending_range: Optional[tuple[float, float]] = None

        # Update cursor scene positions whenever the scene rect changes
        # (initial layout, window resize, record reload).
        self.scene().sceneRectChanged.connect(
            lambda _rect: self._update_scene_cursors()
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_record(self, record: DisturbanceRecord) -> None:
        """Rebuild the canvas for ``record``.

        All decimated display arrays must already be attached to channels by
        ``engine.decimator.prepare_display_data()`` on the background thread.
        This method only constructs PlotItems and calls setData() — no
        computation occurs here (LAW 2).

        Args:
            record: The DisturbanceRecord to display (with _display_t/_display_d set).
        """
        self._clear_canvas()
        self._record = record

        if not record.time_array.size:
            return

        # Time display array and axis label pre-computed by decimator
        t_display: np.ndarray = getattr(
            record, '_t_display', record.time_array
        )
        time_axis_label: str = getattr(
            record, '_time_axis_label', TIME_AXIS_LABEL_FALLBACK
        )
        is_trend = record.display_mode == 'TREND'

        # Set initial cursor B offset in the correct display unit
        if 'min' in time_axis_label:
            self._cursor_pos_b = CURSOR_B_INIT_MS / 1_000.0 / 60.0
        elif 'ms' in time_axis_label:
            self._cursor_pos_b = CURSOR_B_INIT_MS
        else:
            self._cursor_pos_b = CURSOR_B_INIT_MS / 1_000.0
        self._cursor_pos_a = CURSOR_A_INIT

        # Compute canvas height from row list
        rows = get_ordered_rows(record)
        total_height = TIME_AXIS_HEIGHT
        for row_spec in rows:
            if row_spec['type'] == 'bay_header':
                total_height += ROW_HEIGHT_BAY_HEADER
            elif row_spec['type'] == 'analogue':
                total_height += ROW_HEIGHT_ANALOGUE
            elif row_spec['type'] == 'digital':
                total_height += ROW_HEIGHT_DIGITAL
        self.setFixedHeight(total_height)

        # Build rows — UI work only (setData, addPlot, configure)
        grid_row = 0
        for row_spec in rows:
            if row_spec['type'] == 'bay_header':
                grid_row = self._add_bay_spacer(grid_row)
            elif row_spec['type'] == 'analogue':
                grid_row = self._add_analogue_plot(
                    row_spec['channel'], is_trend, grid_row
                )
            elif row_spec['type'] == 'digital':
                grid_row = self._add_digital_plot(
                    row_spec['channel'], grid_row
                )

        self._add_time_axis(t_display, time_axis_label)

        # Connect X-range changes to viewport re-decimation (star topology ref)
        if self._ref_plot is not None:
            self._ref_plot.getViewBox().sigXRangeChanged.connect(
                lambda vb, rng: self._schedule_viewport(rng)
            )
            self._ref_plot.getViewBox().sigXRangeChanged.connect(
                lambda vb, rng: self._update_scene_cursors()
            )
            self._build_scene_cursors()

        print(
            f'[ChannelCanvas] total_height={total_height}'
            f'  grid_rows={self.ci.layout.rowCount()}'
            f'  channel_plots={len(self._analogue_plots) + len(self._digital_plots)}'
        )

    def scale_y_channel(self, channel_id: int, scale_factor: float) -> None:
        """Scale the Y axis of one analogue channel by ``scale_factor``.

        Centres the zoom on the current mid-point of the Y range and disables
        auto-range so the manual scale is preserved across pans.

        Args:
            channel_id:   The analogue channel whose Y axis to scale.
            scale_factor: >1 zooms out (larger range), <1 zooms in (smaller range).
        """
        plot = self._analogue_plots.get(channel_id)
        if plot is None:
            return
        y_min, y_max = plot.viewRange()[1]
        centre    = (y_min + y_max) / 2.0
        half_span = (y_max - y_min) / 2.0 * scale_factor
        plot.setYRange(centre - half_span, centre + half_span, padding=0)
        plot.enableAutoRange(axis='y', enable=False)

    def reset_y_channel(self, channel_id: int) -> None:
        """Restore auto Y-range for one analogue channel.

        Called when the user double-clicks the channel's label row.

        Args:
            channel_id: The analogue channel to reset.
        """
        plot = self._analogue_plots.get(channel_id)
        if plot is None:
            return
        plot.enableAutoRange(axis='y', enable=True)

    def autofit_y_channel(self, channel_id: int) -> None:
        """Fit the Y axis of one analogue channel tightly to the visible data.

        Uses the current horizontal viewport to determine the visible data
        slice, then applies nanmin/nanmax with 7.5 % padding.  Handles flat
        channels (all values equal) by adding ±0.5 around the constant value.
        Falls back to the full channel array when the viewport slice is empty.

        Args:
            channel_id: The analogue channel whose Y axis to auto-fit.
        """
        plot = self._analogue_plots.get(channel_id)
        record = self._record
        if plot is None or record is None or self._ref_plot is None:
            return

        ch = next(
            (c for c in record.analogue_channels if c.channel_id == channel_id),
            None,
        )
        if ch is None or len(ch.raw_data) == 0:
            return

        # ── Determine visible raw-seconds slice ──────────────────────────────
        x_min, x_max = self._ref_plot.getViewBox().viewRange()[0]
        trigger_offset_s: float = (
            record.trigger_time - record.start_time
        ).total_seconds()
        label: str = getattr(record, '_time_axis_label', 'Time (ms)')
        if 'ms' in label:
            scale = 1_000.0
        elif 'min' in label:
            scale = 1.0 / 60.0
        else:
            scale = 1.0

        t_raw = record.time_array
        t_start_raw = x_min / scale + trigger_offset_s
        t_end_raw   = x_max / scale + trigger_offset_s

        n = min(len(t_raw), len(ch.raw_data))
        mask = (t_raw[:n] >= t_start_raw) & (t_raw[:n] <= t_end_raw)
        data = ch.raw_data[:n][mask]
        if len(data) == 0:
            data = ch.raw_data   # fallback to full channel

        # ── Compute padded range ──────────────────────────────────────────────
        y_lo = float(np.nanmin(data))
        y_hi = float(np.nanmax(data))
        span = y_hi - y_lo
        if span == 0.0:
            span = 1.0
            y_lo -= 0.5
            y_hi += 0.5
        else:
            pad   = span * _Y_AUTOFIT_PADDING
            y_lo -= pad
            y_hi += pad

        plot.setYRange(y_lo, y_hi, padding=0)
        plot.enableAutoRange(axis='y', enable=False)

    def autofit_all_channels(self) -> None:
        """Fit every visible analogue channel to the current viewport.

        Iterates all analogue plots and applies autofit_y_channel().
        Digital channels are untouched (fixed Y range by design).
        """
        for channel_id in list(self._analogue_plots.keys()):
            self.autofit_y_channel(channel_id)

    def update_channel_visibility(self, channel_id: int, visible: bool) -> None:
        """Show or hide the PlotItem for ``channel_id``.

        Searches both analogue and digital plot dicts — their channel_ids
        are independent namespaces and must not share a single dict.

        Args:
            channel_id: The channel to update.
            visible:    True = show, False = hide.
        """
        plot = self._analogue_plots.get(channel_id) or self._digital_plots.get(channel_id)
        if plot is not None:
            plot.setVisible(visible)

    def zoom_to_fit(self) -> None:
        """Reset X range to full record duration; Y axes auto-fit."""
        if self._ref_plot is None or self._record is None:
            return
        t = getattr(self._record, '_t_display', None)
        if t is None or len(t) == 0:
            return
        self._ref_plot.setXRange(float(t[0]), float(t[-1]), padding=0.02)

    def zoom_to_fault(self) -> None:
        """Zoom to trigger ±200 ms (WAVEFORM) or ±0.2 s / ±0.003 min (TREND).

        Trigger is at t=0 in all display units.  The half-window is
        FAULT_WINDOW_S converted to the active display unit.
        """
        if self._ref_plot is None or self._record is None:
            return
        label: str = getattr(self._record, '_time_axis_label', 'Time (ms)')
        if 'min' in label:
            half = FAULT_WINDOW_S / 60.0
        elif 'ms' in label:
            half = FAULT_WINDOW_S * 1000.0
        else:   # seconds
            half = FAULT_WINDOW_S
        self._ref_plot.setXRange(-half, half, padding=0)

    def zoom_in(self) -> None:
        """Reduce X range by 50% centred on current view centre."""
        if self._ref_plot is None:
            return
        vb = self._ref_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        centre = (x_min + x_max) / 2.0
        half = (x_max - x_min) / 4.0   # 50% width reduction → ¼ of current span
        vb.setXRange(centre - half, centre + half, padding=0)

    def zoom_out(self) -> None:
        """Increase X range by 100% centred on current view centre."""
        if self._ref_plot is None:
            return
        vb = self._ref_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        centre = (x_min + x_max) / 2.0
        half = (x_max - x_min)          # 100% increase → current span becomes half
        vb.setXRange(centre - half, centre + half, padding=0)

    def get_cursor_time(self, cursor_id: int) -> float:
        """Return the current display-unit position of cursor A (0) or B (1).

        Args:
            cursor_id: 0 for cursor A, 1 for cursor B.

        Returns:
            Current position in display units, or 0.0 if not yet initialised.
        """
        return self._cursor_pos_a if cursor_id == 0 else self._cursor_pos_b

    # ── Mouse overrides ────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        """Update scene cursor extents when the canvas widget is resized.

        Args:
            event: The QResizeEvent.
        """
        super().resizeEvent(event)
        # Guard: PyQtGraph calls resizeEvent(None) during super().__init__()
        # before our own __init__ has initialised _ref_plot.
        if hasattr(self, '_ref_plot'):
            self._update_scene_cursors()

    def wheelEvent(self, event) -> None:
        """Wheel zooms the X axis; Shift+Wheel propagates to scroll area.

        Plain wheel and Ctrl+Wheel both zoom so the behaviour matches the
        pre-ScrollArea convention familiar to the user.  Hold Shift to
        scroll vertically instead.
        """
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            event.ignore()
        else:
            super().wheelEvent(event)

    def contextMenuEvent(self, event) -> None:
        """Right-click shows a minimal context menu (Zoom to Fit).

        Args:
            event: The context menu QContextMenuEvent.
        """
        menu = QMenu(self)
        fit_action = menu.addAction("Zoom to Fit")
        fit_action.triggered.connect(self.zoom_to_fit)
        menu.exec(event.globalPos())

    # ── Private — canvas lifecycle ─────────────────────────────────────────────

    def _on_cursor_dragged(self, cursor_id: int, new_pos: float) -> None:
        """Update cursor position, sync scene line, and emit cursor_moved.

        Called from ``_SceneCursorLine.mouseMoveEvent`` — no InfiniteLines,
        no per-plot slave sync required.

        Args:
            cursor_id: 0 for cursor A, 1 for cursor B.
            new_pos:   New position in display units.
        """
        if cursor_id == 0:
            self._cursor_pos_a = new_pos
        else:
            self._cursor_pos_b = new_pos
        self._update_scene_cursors()
        app_state.cursor_moved.emit(cursor_id, new_pos)

    def _build_scene_cursors(self) -> None:
        """Create two full-height scene cursor lines and add them to the scene.

        Called once per ``load_record`` after all plots are built and
        ``_ref_plot`` is known.  Removes any previous scene lines first.
        """
        scene = self.scene()
        if scene is None:
            return

        # Remove previous lines (e.g. from a prior load_record call)
        for line in (self._scene_cursor_a, self._scene_cursor_b):
            if line is not None:
                scene.removeItem(line)

        pen_a = QPen(QColor(CURSOR_A_COLOUR))
        pen_a.setWidthF(CURSOR_PEN_WIDTH)
        pen_a.setStyle(Qt.PenStyle.DashLine)

        pen_b = QPen(QColor(CURSOR_B_COLOUR))
        pen_b.setWidthF(CURSOR_PEN_WIDTH)
        pen_b.setStyle(Qt.PenStyle.DashLine)

        self._scene_cursor_a = _SceneCursorLine(0, self, pen_a, 'A')
        self._scene_cursor_b = _SceneCursorLine(1, self, pen_b, 'B')
        scene.addItem(self._scene_cursor_a)
        scene.addItem(self._scene_cursor_b)

        self._update_scene_cursors()

    def _update_scene_cursors(self) -> None:
        """Reposition both scene cursor lines to match current cursor times.

        Called after any event that changes either the cursor time values
        (drag) or the mapping from time → scene-x (zoom, pan, resize).
        """
        if (self._ref_plot is None
                or self._scene_cursor_a is None
                or self._scene_cursor_b is None):
            return
        scene = self.scene()
        if scene is None:
            return

        rect = scene.sceneRect()
        vb   = self._ref_plot.getViewBox()

        x_a = vb.mapViewToScene(pg.Point(self._cursor_pos_a, 0)).x()
        x_b = vb.mapViewToScene(pg.Point(self._cursor_pos_b, 0)).x()

        self._scene_cursor_a.setLine(x_a, rect.top(), x_a, rect.bottom())
        self._scene_cursor_b.setLine(x_b, rect.top(), x_b, rect.bottom())

    def _clear_canvas(self) -> None:
        """Remove all plots, scene cursor lines, and reset tracking state."""
        self._vp_timer.stop()
        self._pending_range   = None
        self._updating_viewport = False
        self._last_viewport   = (None, None)

        # Remove scene cursor lines before clearing plots (scene.clear() is
        # NOT called — ci.clear() only clears the GraphicsLayout, not raw
        # scene items we added directly).
        _scene = self.scene()
        if _scene is not None:
            for line in (self._scene_cursor_a, self._scene_cursor_b):
                if line is not None:
                    _scene.removeItem(line)
        self._scene_cursor_a = None
        self._scene_cursor_b = None

        self.clear()
        self._analogue_plots.clear()
        self._digital_plots.clear()
        self._analogue_curves.clear()
        self._digital_curves.clear()
        self._trigger_lines.clear()
        self._ref_plot = None
        self._record   = None

    # ── Private — viewport re-decimation ──────────────────────────────────────

    def _schedule_viewport(self, view_range: tuple[float, float]) -> None:
        """Debounce viewport updates — store latest range, fire after 16ms idle.

        Args:
            view_range: (t_start, t_end) in display units from sigXRangeChanged.
        """
        self._pending_range = view_range
        self._vp_timer.start()   # restarts countdown if already running

    def _on_vp_timer(self) -> None:
        """Fired by debounce timer — delegates to _update_viewport."""
        if self._pending_range is not None:
            self._update_viewport(self._pending_range)

    def _update_viewport(self, view_range: tuple[float, float]) -> None:
        """Re-decimate all channels for the current viewport.

        Converts view_range (display units) → raw seconds, masks
        ``record.time_array`` directly, reads ``ch.raw_data`` / ``ch.data``
        at those indices, converts the masked time back to display units,
        then decimates and calls setData().

        This guarantees every zoom/pan call decimates from the original raw
        arrays regardless of zoom history — never from pre-decimated data.

        Performance guards applied here:
          Fix 2 — Re-entrancy guard: ``setData()`` triggers autorange which
                   fires ``sigXRangeChanged`` which would re-enter this method;
                   the guard breaks that loop immediately.
          Fix 3a — Viewport threshold: skip if the view moved less than 1% of
                   the current span (scroll inertia / floating-point noise).
          Fix 3b — Static digital channels: channels whose state never changes
                   (flagged by ``prepare_display_data``) are skipped entirely —
                   89% of digital ``setData()`` calls eliminated for JMHE.

        Args:
            view_range: (t_start, t_end) in display units (ms / s / min).
        """
        # Fix 2: re-entrancy guard — setData() → autorange → sigXRangeChanged
        if self._updating_viewport:
            return
        self._updating_viewport = True
        try:
            record = self._record
            if record is None:
                return

            t_start, t_end = view_range

            # Fix 3a: skip sub-pixel / inertia pans (< 1% of visible span)
            last_start, last_end = self._last_viewport
            if last_start is not None:
                span = t_end - t_start
                if (span > 0
                        and abs(t_start - last_start) < span * 0.01
                        and abs(t_end - last_end) < span * 0.01):
                    return
            self._last_viewport = (t_start, t_end)

            t_raw: np.ndarray = record.time_array
            is_trend = record.display_mode == 'TREND'

            # ── Convert display-unit range → raw seconds ──────────────────────
            trigger_offset_s: float = (
                record.trigger_time - record.start_time
            ).total_seconds()
            label: str = getattr(record, '_time_axis_label', 'Time (ms)')
            if 'ms' in label:
                scale = 1_000.0          # raw s → ms
            elif 'min' in label:
                scale = 1.0 / 60.0      # raw s → min
            else:
                scale = 1.0             # raw s → s

            t_start_raw = t_start / scale + trigger_offset_s
            t_end_raw   = t_end   / scale + trigger_offset_s

            # Full-array boolean mask — shared across all channels
            mask: np.ndarray = (t_raw >= t_start_raw) & (t_raw <= t_end_raw)

            for ch in record.analogue_channels:
                curve = self._analogue_curves.get(ch.channel_id)
                if curve is None or not ch.visible:
                    continue
                n = min(len(t_raw), len(ch.raw_data))
                mask_n    = mask[:n]
                t_vis_raw = t_raw[:n][mask_n]         # raw seconds
                d_vis     = ch.raw_data[:n][mask_n]   # always from source
                if len(t_vis_raw) == 0:
                    continue
                t_vis = (t_vis_raw - trigger_offset_s) * scale
                if is_trend:
                    t_dec, d_dec = decimate_uniform(t_vis, d_vis, MAX_ANALOGUE_POINTS)
                else:
                    t_dec, d_dec = decimate_minmax(t_vis, d_vis, MAX_ANALOGUE_POINTS)
                curve.setData(t_dec, d_dec)

            for ch in record.digital_channels:
                # Fix 3b: skip channels whose state never changes (all zeros/ones)
                if getattr(ch, '_display_is_static', False):
                    continue
                curve = self._digital_curves.get(ch.channel_id)
                if curve is None or not ch.visible:
                    continue
                n = min(len(t_raw), len(ch.data))
                mask_n    = mask[:n]
                t_vis_raw = t_raw[:n][mask_n]         # raw seconds
                d_vis     = ch.data[:n][mask_n]       # always from source
                if len(t_vis_raw) == 0:
                    continue
                t_vis = (t_vis_raw - trigger_offset_s) * scale
                t_dec, d_dec = decimate_digital(t_vis, d_vis, MAX_DIGITAL_POINTS)
                curve.setData(t_dec, d_dec)
        finally:
            self._updating_viewport = False

    # ── Private — row builders ─────────────────────────────────────────────────

    def _add_bay_spacer(self, grid_row: int) -> int:
        """Add an invisible ROW_HEIGHT_BAY_HEADER px spacer for bay alignment.

        Args:
            grid_row: Current GridLayout row index.

        Returns:
            Next available row index.
        """
        spacer = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_BAY_HEADER)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_BAY_HEADER)
        self._configure_spacer_plot(spacer)
        # Spacer plots are NOT used as X-link reference — they carry no data
        if self._ref_plot is not None:
            spacer.setXLink(self._ref_plot)
        return grid_row + 1

    def _add_analogue_plot(
        self,
        ch: AnalogueChannel,
        is_trend: bool,
        grid_row: int,
    ) -> int:
        """Add one ROW_HEIGHT_ANALOGUE px PlotItem for an analogue channel.

        Reads pre-decimated arrays from ``ch._display_t`` / ``ch._display_d``.

        Args:
            ch:        The analogue channel to render.
            is_trend:  True when display_mode == 'TREND'.
            grid_row:  Current GridLayout row index.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_ANALOGUE)
        self._configure_analogue_plot(plot)

        # First analogue plot becomes the X-link reference (star topology)
        if self._ref_plot is None:
            self._ref_plot = plot
            plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        else:
            plot.setXLink(self._ref_plot)

        t = getattr(ch, '_display_t', np.array([]))
        d = getattr(ch, '_display_d', np.array([]))

        curve = self._make_analogue_curve(t, d, ch, is_trend)
        plot.addItem(curve)
        self._add_trigger_line(plot, trigger_pos=0.0)

        self._analogue_plots[ch.channel_id]  = plot
        self._analogue_curves[ch.channel_id] = curve
        return grid_row + 1

    def _add_digital_plot(
        self,
        ch: DigitalChannel,
        grid_row: int,
    ) -> int:
        """Add one ROW_HEIGHT_DIGITAL px PlotItem for a digital channel.

        Reads pre-decimated arrays from ``ch._display_t`` / ``ch._display_d``.

        Args:
            ch:       The digital channel to render.
            grid_row: Current GridLayout row index.

        Returns:
            Next available row index.
        """
        plot = self.addPlot(row=grid_row, col=0)
        self.ci.layout.setRowMaximumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self.ci.layout.setRowMinimumHeight(grid_row, ROW_HEIGHT_DIGITAL)
        self._configure_digital_plot(plot)

        if self._ref_plot is None:
            self._ref_plot = plot
            plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        else:
            plot.setXLink(self._ref_plot)

        t = getattr(ch, '_display_t', np.array([]))
        d = getattr(ch, '_display_d', np.array([]))

        baseline, filled = self._make_digital_curves(t, d)
        plot.addItem(baseline)
        plot.addItem(filled)
        self._add_trigger_line(plot, trigger_pos=0.0, show_label=False)

        self._digital_plots[ch.channel_id]  = plot
        self._digital_curves[ch.channel_id] = filled
        return grid_row + 1

    # ── Private — curve factories ──────────────────────────────────────────────

    def _make_analogue_curve(
        self,
        t: np.ndarray,
        d: np.ndarray,
        ch: AnalogueChannel,
        is_trend: bool,
    ) -> pg.PlotDataItem | pg.ScatterPlotItem:
        """Create the waveform item for one analogue channel.

        Args:
            t:         Pre-decimated time array (display units).
            d:         Pre-decimated data array.
            ch:        Analogue channel (for colour).
            is_trend:  True for TREND scatter mode.

        Returns:
            PlotDataItem (WAVEFORM) or ScatterPlotItem (TREND).
        """
        if is_trend:
            return pg.ScatterPlotItem(
                x=t,
                y=d,
                pen=None,
                brush=pg.mkBrush(QColor(ch.colour)),
                size=TREND_SCATTER_SIZE,
            )

        return pg.PlotDataItem(
            t,
            d,
            pen=pg.mkPen(ch.colour, width=WAVEFORM_PEN_WIDTH),
            antialias=False,
        )

    def _make_digital_curves(
        self,
        t: np.ndarray,
        d: np.ndarray,
    ) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
        """Create baseline + filled step items for one digital channel.

        Args:
            t: Pre-decimated time array (display units).
            d: Pre-decimated data array (0.0 / 1.0).

        Returns:
            Tuple (baseline, filled).
        """
        baseline = pg.PlotDataItem(
            t,
            np.full(len(t), DIG_BASELINE_Y, dtype=np.float64),
            pen=pg.mkPen(DIG_BASELINE_PEN, width=DIG_BASELINE_PEN_WIDTH),
            antialias=False,
        )

        filled = pg.PlotDataItem(
            t,
            d,
            stepMode='right',
            pen=pg.mkPen(DIG_FILL_PEN, width=DIG_FILL_PEN_WIDTH),
            fillLevel=0.0,
            brush=pg.mkBrush(DIG_FILL_BRUSH),
            antialias=False,
        )

        return baseline, filled

    # ── Private — time axis ───────────────────────────────────────────────────

    def _add_time_axis(
        self, t_display: np.ndarray, label: str = TIME_AXIS_LABEL_FALLBACK
    ) -> None:
        """Add a single shared time axis PlotItem at the bottom.

        Linked to the reference plot (star topology).

        Args:
            t_display: Full time array in display units.
            label:     Axis label string (e.g. 'Time (ms)').
        """
        row = self.ci.layout.rowCount()

        axis_plot = self.addPlot(row=row, col=0)
        self.ci.layout.setRowMaximumHeight(row, TIME_AXIS_HEIGHT)
        self.ci.layout.setRowMinimumHeight(row, TIME_AXIS_HEIGHT)

        axis_plot.hideAxis('left')
        axis_plot.getAxis('bottom').setStyle(showValues=True)
        axis_plot.getAxis('bottom').setLabel(label, color=AXIS_LABEL_COLOUR)
        axis_plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        axis_plot.getViewBox().setBorder(None)

        # Invisible flat line so X range is initialised to data extent
        if len(t_display) >= 2:
            axis_plot.plot(
                [t_display[0], t_display[-1]],
                [0, 0],
                pen=pg.mkPen(None),
            )

        if self._ref_plot is not None:
            axis_plot.setXLink(self._ref_plot)

    # ── Private — plot helpers ─────────────────────────────────────────────────

    def _configure_spacer_plot(self, plot: pg.PlotItem) -> None:
        """Configure an invisible bay-header spacer plot.

        Args:
            plot: The PlotItem to configure as a spacer.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        plot.getViewBox().setBorder(None)
        plot.setMouseEnabled(x=True, y=False)

    def _configure_analogue_plot(self, plot: pg.PlotItem) -> None:
        """Apply settings for an analogue waveform plot.

        Args:
            plot: The PlotItem to configure.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        plot.getViewBox().setBorder(None)
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        plot.enableAutoRange(axis='y', enable=True)
        plot.setMenuEnabled(False)
        plot.getViewBox().setMenuEnabled(False)

    def _configure_digital_plot(self, plot: pg.PlotItem) -> None:
        """Apply settings for a compact digital-channel plot.

        Args:
            plot: The PlotItem to configure.
        """
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.setMouseEnabled(x=True, y=False)
        plot.setYRange(DIG_Y_MIN, DIG_Y_MAX, padding=0)
        plot.getViewBox().setBackgroundColor(BACKGROUND_COLOUR)
        plot.getViewBox().setBorder(None)
        plot.setMenuEnabled(False)
        plot.getViewBox().setMenuEnabled(False)

    def _add_trigger_line(
        self,
        plot: pg.PlotItem,
        trigger_pos: float,
        show_label: bool = True,
    ) -> None:
        """Add a non-movable trigger InfiniteLine to ``plot``.

        Args:
            plot:         The PlotItem to add the line to.
            trigger_pos:  Trigger position in display units (0.0 after centring).
            show_label:   If False, omit the 'T=0' label (narrow digital rows).
        """
        line = pg.InfiniteLine(
            pos=trigger_pos,
            angle=90,
            movable=False,
            pen=pg.mkPen(
                TRIGGER_COLOUR,
                width=TRIGGER_LINE_WIDTH,
                style=Qt.PenStyle.DotLine,
            ),
            label='T=0' if show_label else '',
            labelOpts={
                'color': TRIGGER_COLOUR,
                'position': TRIGGER_LABEL_POS,
            } if show_label else {},
        )
        plot.addItem(line)
        self._trigger_lines.append(line)
