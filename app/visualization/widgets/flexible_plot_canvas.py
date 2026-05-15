from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget

from app.models import AnalogChannel, DisturbanceRecord
from app.visualization.axis.datetime_axis import (
    AXIS_MODE_ABSOLUTE,
    AXIS_MODE_RELATIVE,
    DatetimeAxisItem,
)
from app.visualization.managers.multi_axis_manager import MultiAxisManager
from app.visualization.rendering.downsampling import decimate_for_display

# Canonical phase-detection color heuristic (VIEWPORT_RENDERING_POLICY §9 + §16)
_AXIS_COLORS = ["#FF4444", "#FFCC00", "#4488FF", "#44BB44", "#AAAAAA", "#FF8800"]
_SPARSE_RATE_THRESHOLD_HZ = 2.0
_SPARSE_INTERVAL_THRESHOLD_S = 2.0


def _channel_color(ch: AnalogChannel) -> str:
    """Assign a display color based on channel name phase heuristics."""
    name = ch.name.lower()
    if any(x in name for x in ("_a", "va", "ia", "vr", "ir", "phase_a", "ph_a")):
        return "#FF4444"
    if any(x in name for x in ("_b", "vb", "ib", "vy", "iy", "phase_b", "ph_b")):
        return "#FFCC00"
    if any(x in name for x in ("_c", "vc", "ic", "vb_", "phase_c", "ph_c")):
        return "#4488FF"
    if any(x in name for x in ("earth", "zero", "neutral", "vn", "in_", "3i0", "3u0")):
        return "#44BB44"
    return _AXIS_COLORS[ch.index % len(_AXIS_COLORS)]


def _is_sparse_timeseries(record: DisturbanceRecord, time: np.ndarray) -> bool:
    """Return True for low-rate trend records that need marker/context rendering."""
    rates = [r for r in record.sampling_info.sampling_rates if r > 0]
    if rates and max(rates) <= _SPARSE_RATE_THRESHOLD_HZ:
        return True
    if len(time) < 3:
        return True
    diffs = np.diff(time)
    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
    return bool(len(valid) > 0 and np.median(valid) >= _SPARSE_INTERVAL_THRESHOLD_S)


def _sparse_display_series(
    time: np.ndarray,
    data: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return viewport samples plus nearest neighbors for sparse trend rendering."""
    _EMPTY = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
    if len(time) == 0:
        return _EMPTY
    if t_start > t_end:
        t_start, t_end = t_end, t_start

    in_view = np.flatnonzero((time >= t_start) & (time <= t_end))
    indices: list[int] = []

    left = int(np.searchsorted(time, t_start, side="left")) - 1
    right = int(np.searchsorted(time, t_end, side="right"))
    if 0 <= left < len(time):
        indices.append(left)
    indices.extend(int(i) for i in in_view)
    if 0 <= right < len(time):
        indices.append(right)

    if not indices:
        nearest = int(np.searchsorted(time, (t_start + t_end) / 2.0, side="left"))
        nearest = max(0, min(nearest, len(time) - 1))
        indices.append(nearest)

    unique = np.array(sorted(set(indices)), dtype=np.int64)
    return time[unique].astype(np.float64), data[unique].astype(np.float64)


def _finite_y_range(data: np.ndarray) -> tuple[float, float] | None:
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return None
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
    else:
        pad = max((y_max - y_min) * 0.05, 1e-9)
    return y_min - pad, y_max + pad


class FlexiblePlotCanvas(pg.GraphicsLayoutWidget):
    """N-Axis Single Canvas for analog waveform rendering (SIGRA-style).

    Renders all analog channels from a DisturbanceRecord in one canvas with
    N independent Y-axes (one ViewBox per channel) sharing a single X time axis.

    Digital channels are NOT rendered here — they belong to DigitalEventTimeline
    (Phase 3B). See VIEWPORT_RENDERING_POLICY §17 and VISUALIZATION_CONTRACT.md.

    pg.setConfigOptions(...) must be called in app/main.py before
    this widget is instantiated — NOT in this constructor.
    """

    cursor_moved = pyqtSignal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
        max_display_points: int = 4_000,
    ) -> None:
        super().__init__(parent=parent)
        self.setBackground("#1E1E1E")

        self._max_pts = max_display_points

        self._record: DisturbanceRecord | None = None
        self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
        self._data_cache: dict[str, np.ndarray] = {}
        self._sparse_mode = False

        self._resize_pending = False
        self._cursor: pg.InfiniteLine | None = None
        self._trigger_line: pg.InfiniteLine | None = None

        self._datetime_axis = DatetimeAxisItem(orientation="bottom")
        self._primary_plot: pg.PlotItem = self.addPlot(
            row=0, col=0, axisItems={"bottom": self._datetime_axis}
        )
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")

        self._axis_manager = MultiAxisManager(self._primary_plot, self)

        self._primary_plot.getViewBox().sigXRangeChanged.connect(
            self._on_x_range_changed
        )
        # Refresh viewport (data + Y ranges) once the canvas is first shown
        # and sized in a real layout.  sigResized fires after _sync_geometries
        # so secondary ViewBox geometry is already correct by the time we run.
        self._primary_plot.getViewBox().sigResized.connect(
            self._on_plot_resized
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_record(
        self,
        record: DisturbanceRecord,
        *,
        axis_mode: str = AXIS_MODE_RELATIVE,
    ) -> None:
        """Load a DisturbanceRecord. Build N-Axis layout. Zoom to trigger.

        Args:
            record:    The waveform record to render.
            axis_mode: AXIS_MODE_RELATIVE (default) — elapsed-seconds labels;
                       AXIS_MODE_ABSOLUTE — wall-clock datetime labels.
        """
        self._clear_canvas()

        self._record = record
        self._datetime_axis.set_start_time(
            record.timing_info.start_time if axis_mode == AXIS_MODE_ABSOLUTE else None
        )

        # Cache numpy arrays once — avoids per-frame DataFrame access in _update_viewport
        self._time_cache = record.waveform_data["time"].to_numpy(dtype=np.float64)
        self._sparse_mode = _is_sparse_timeseries(record, self._time_cache)
        self._data_cache = {
            ch.name: record.waveform_data[ch.name].to_numpy(dtype=np.float64)
            for ch in record.analog_channels
        }

        for ch in record.analog_channels:
            color = _channel_color(ch)
            vb = self._axis_manager.add_axis(ch.name, ch.unit, color)

            curve = self._make_curve(color)

            if vb is self._primary_plot.getViewBox():
                self._primary_plot.addItem(curve)
            else:
                vb.addItem(curve)

            self._axis_manager.register(ch.name, vb, curve, color)

        self._add_trigger_line()
        self._add_cursor()
        self.zoom_to_trigger()
        self._update_viewport()
        # For sparse records the full dataset is always in-viewport, so pinning
        # Y ranges from the raw data cache is both correct and stable.
        if self._sparse_mode:
            self._force_y_ranges()

    def add_parameter(
        self,
        name: str,
        data: np.ndarray,
        unit: str = "unknown",
        color: str | None = None,
    ) -> None:
        """Add a single analog parameter (e.g. an analytics overlay).

        The data array must have the same length as the currently cached time array
        when a record is loaded. Designed for Phase 5 analytics overlays.
        """
        if name in self._axis_manager.get_curves():
            self.remove_parameter(name)

        n_existing = len(self._axis_manager.parameter_names())
        effective_color = color if color is not None else _AXIS_COLORS[n_existing % len(_AXIS_COLORS)]

        vb = self._axis_manager.add_axis(name, unit, effective_color)

        curve = pg.PlotDataItem(
            pen=pg.mkPen(effective_color, width=1),
            skipFiniteCheck=True,
        )
        curve.setClipToView(True)

        if vb is self._primary_plot.getViewBox():
            self._primary_plot.addItem(curve)
        else:
            vb.addItem(curve)

        self._axis_manager.register(name, vb, curve, effective_color)
        self._data_cache[name] = data.astype(np.float64)
        self._update_viewport()

    def remove_parameter(self, name: str) -> None:
        """Remove a named parameter's ViewBox, axis, and curve from the canvas."""
        self._axis_manager.remove_axis(name)
        self._data_cache.pop(name, None)

    def set_visible_channels(self, names: list[str]) -> None:
        """Show or hide analog channels by name.

        Visible channels are repopulated with current viewport decimation.
        Hidden channels have their curve data cleared (zero-allocation setData).
        """
        names_set = set(names)
        t_start, t_end = self._primary_plot.getViewBox().viewRange()[0]

        for ch_name, curve in self._axis_manager.get_curves().items():
            if ch_name in names_set and ch_name in self._data_cache:
                if self._sparse_mode:
                    t_dec, d_dec = _sparse_display_series(
                        self._time_cache,
                        self._data_cache[ch_name],
                        t_start,
                        t_end,
                    )
                else:
                    t_dec, d_dec = decimate_for_display(
                        self._time_cache,
                        self._data_cache[ch_name],
                        t_start,
                        t_end,
                        self._max_pts,
                    )
                curve.setData(t_dec, d_dec)
                self._sync_curve_view(curve, d_dec, t_start, t_end)
            elif ch_name not in names_set:
                curve.setData(
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )

    def zoom_to_trigger(self, window_s: float = 0.2) -> None:
        """Centre the viewport on the trigger event ± window_s seconds."""
        if self._record is None or len(self._time_cache) == 0:
            return
        if self._sparse_mode:
            finite = self._time_cache[np.isfinite(self._time_cache)]
            if len(finite) == 0:
                return
            self._primary_plot.setXRange(
                float(np.min(finite)),
                float(np.max(finite)),
                padding=0,
            )
            return
        t_trig = (
            self._record.timing_info.trigger_time
            - self._record.timing_info.start_time
        ).total_seconds()
        t_max = float(self._time_cache[-1])
        t_start = max(0.0, t_trig - window_s)
        t_end = min(t_max, t_trig + window_s)
        self._primary_plot.setXRange(t_start, t_end, padding=0)

    def set_cursor_pos(self, t: float) -> None:
        """Move the master cursor without re-emitting cursor_moved.

        Used by VisualizationManager in Phase 3B to synchronise cursor position
        across multiple FlexiblePlotCanvas instances without signal loops.
        """
        if self._cursor is not None:
            self._cursor.blockSignals(True)
            self._cursor.setValue(t)
            self._cursor.blockSignals(False)

    def _clear_canvas(self) -> None:
        """Remove all parameters, cursor, and trigger line. Reset to blank state.

        Named _clear_canvas (not clear) to avoid shadowing: pg.GraphicsLayoutWidget
        .__init__ sets self.clear = self.ci.clear (GraphicsLayout.clear), which would
        strip the PlotItem from the layout and scene if called.
        """
        vb = self._primary_plot.getViewBox()

        # Disconnect both resize handlers before rebuilding so we can re-establish
        # the correct firing order: _sync_geometries must fire BEFORE _on_plot_resized
        # so that secondary ViewBox geometry is current when _update_viewport runs.
        try:
            vb.sigResized.disconnect(self._axis_manager._sync_geometries)
        except TypeError:
            pass
        try:
            vb.sigResized.disconnect(self._on_plot_resized)
        except TypeError:
            pass

        self._axis_manager.clear()
        self._primary_plot.clear()

        # Restore primary plot appearance (clear() strips labels and grid)
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")
        self._datetime_axis.set_start_time(None)

        # Fresh axis manager — its __init__ connects _sync_geometries FIRST
        self._axis_manager = MultiAxisManager(self._primary_plot, self)
        # Reconnect _on_plot_resized AFTER _sync_geometries
        vb.sigResized.connect(self._on_plot_resized)

        self._resize_pending = False
        self._record = None
        self._time_cache = np.empty(0, dtype=np.float64)
        self._data_cache.clear()
        self._sparse_mode = False
        self._cursor = None
        self._trigger_line = None

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_curve(self, color: str) -> pg.PlotDataItem:
        if self._sparse_mode:
            curve = pg.PlotDataItem(
                pen=pg.mkPen(color, width=1.5),
                symbol="o",
                symbolSize=7,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen("#F0F0F0", width=0.75),
                skipFiniteCheck=True,
            )
            curve.setClipToView(False)
            return curve

        curve = pg.PlotDataItem(
            pen=pg.mkPen(color, width=1),
            skipFiniteCheck=True,
        )
        curve.setClipToView(True)
        return curve

    def _on_x_range_changed(
        self,
        _viewbox: pg.ViewBox,
        x_range: tuple[float, float],
    ) -> None:
        t_start, t_end = x_range
        self._update_viewport(t_start, t_end)

    def normalize_viewport(self, t_start: float, t_end: float) -> None:
        """Force this canvas to the specified X range and re-pin Y ranges.

        Called by the main window after inter-panel X linking to override the
        pixel-width–based range that PyQtGraph's linkedViewChanged computes
        when panels have different secondary axis counts.  For sparse records,
        _force_y_ranges() is also called to prevent auto-range drift.
        """
        self._primary_plot.setXRange(t_start, t_end, padding=0)
        if self._sparse_mode:
            self._force_y_ranges()

    def _on_plot_resized(self) -> None:
        if not self._resize_pending:
            self._resize_pending = True
            QTimer.singleShot(0, self._deferred_resize_update)

    def _deferred_resize_update(self) -> None:
        """Runs after GraphicsView.resizeEvent() fully completes."""
        self._resize_pending = False
        vb = self._primary_plot.getViewBox()
        vb._matrixNeedsUpdate = True
        vb.updateMatrix()
        for secondary_vb in self._axis_manager.get_viewboxes():
            secondary_vb._matrixNeedsUpdate = True
            secondary_vb.updateMatrix()
        self._update_viewport()
        if self._sparse_mode:
            self._force_y_ranges()
        vp = self.viewport()
        if vp is not None:
            vp.update()

    def _force_y_ranges(self) -> None:
        """Pin each ViewBox's Y range to its channel's full-data extent.

        Used for sparse-mode records where the full dataset is always in the
        viewport.  Computes the range from the raw data cache rather than from
        decimated viewport data, so the range is stable regardless of when this
        is called relative to the first viewport update.
        """
        for name, entry in self._axis_manager._axes.items():
            if name not in self._data_cache:
                continue
            y_range = _finite_y_range(self._data_cache[name])
            if y_range is not None:
                entry.viewbox.setYRange(y_range[0], y_range[1], padding=0)

    def _update_viewport(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> None:
        """Decimate and push updated data to all registered curves.

        This is the rendering hot path — only array slicing and setData() here.
        No DataFrame operations, no to_numpy(), no analytics.
        """
        if len(self._time_cache) == 0:
            return

        if t_start is None or t_end is None:
            t_start, t_end = self._primary_plot.getViewBox().viewRange()[0]

        for name, curve in self._axis_manager.get_curves().items():
            if name in self._data_cache:
                if self._sparse_mode:
                    t_dec, d_dec = _sparse_display_series(
                        self._time_cache,
                        self._data_cache[name],
                        t_start,
                        t_end,
                    )
                else:
                    t_dec, d_dec = decimate_for_display(
                        self._time_cache,
                        self._data_cache[name],
                        t_start,
                        t_end,
                        self._max_pts,
                    )
                curve.setData(t_dec, d_dec)
                self._sync_curve_view(curve, d_dec, t_start, t_end)

    def _sync_curve_view(
        self,
        curve: pg.PlotDataItem,
        data: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> None:
        viewbox = curve.getViewBox()
        if viewbox is None:
            return
        if viewbox is not self._primary_plot.getViewBox():
            viewbox.setXRange(t_start, t_end, padding=0)
        y_range = _finite_y_range(data)
        if y_range is not None:
            viewbox.setYRange(y_range[0], y_range[1], padding=0)

    def _add_trigger_line(self) -> None:
        """Add a fixed red trigger marker to the primary plot."""
        if self._record is None:
            return
        trigger_time_s = (
            self._record.timing_info.trigger_time
            - self._record.timing_info.start_time
        ).total_seconds()
        self._trigger_line = pg.InfiniteLine(
            pos=trigger_time_s,
            angle=90,
            movable=False,
            pen=pg.mkPen("#FF4444", width=2, style=Qt.PenStyle.DotLine),
            label="T",
            labelOpts={"color": "#FF4444", "position": 0.95},
        )
        self._primary_plot.addItem(self._trigger_line)

    def _add_cursor(self) -> None:
        """Add a movable yellow master time cursor to the primary plot."""
        self._cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#FFFF00", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self._primary_plot.addItem(self._cursor)

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        self.cursor_moved.emit(line.value())
