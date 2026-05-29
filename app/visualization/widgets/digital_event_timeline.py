from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from app.models import DigitalChannel, DisturbanceRecord
from app.visualization.axis.datetime_axis import (
    AXIS_MODE_RELATIVE,
    DatetimeAxisItem,
    TimeDisplayMode,
)
from app.visualization.axis_management import (
    GROUPED_LEFT_AXIS_WIDTH_PX,
    reserved_right_axis_width,
)
from app.visualization.rendering.digital_transforms import (
    build_hi_lo_segments,
    clip_digital_to_viewport,
    digital_role_color,
    extract_transitions,
)
from app.visualization.signal_visibility import default_visible_digital_names

_TRACK_SPACING = 1.0   # row pitch (Y units between channel baselines)
_LINE_WIDTH_HI = 3     # pen width when signal is HIGH
_LINE_WIDTH_LO = 1     # pen width when signal is LOW


@dataclasses.dataclass
class _TrackEntry:
    name:      str
    curve_hi:  pg.PlotDataItem   # thick line — HIGH periods
    curve_lo:  pg.PlotDataItem   # thin line  — LOW periods
    label:     pg.TextItem
    bg_band:   pg.LinearRegionItem
    color:     str
    y_offset:  float             # Y position of the line in data coordinates
    t_trans:   np.ndarray        # pre-extracted transition times (float64)
    d_trans:   np.ndarray        # pre-extracted transition states (float64)


class DigitalEventTimeline(pg.PlotWidget):
    """Step-function horizontal-track display for digital channels.

    Renders all digital channels from a DisturbanceRecord in a single
    PlotWidget.  Each channel occupies one fixed-height horizontal track
    (vertical offset per channel).  Binary HIGH state is filled with the
    channel's role color; LOW state is empty.

    X-axis is linkable to FlexiblePlotCanvas via link_x_to() so that panning
    and zooming stay synchronized.  Trigger line and movable cursor are
    provided for Phase 3B integration.

    pg.setConfigOptions(...) must be called in app/main.py
    before this widget is instantiated — NOT in this constructor.
    """

    cursor_moved = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._datetime_axis = DatetimeAxisItem(orientation="bottom")
        super().__init__(parent=parent, axisItems={"bottom": self._datetime_axis})
        self.setBackground("#1E1E1E")

        self._record: DisturbanceRecord | None = None
        self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
        self._tracks: dict[str, _TrackEntry] = {}
        self._trigger_line: pg.InfiniteLine | None = None
        self._cursor: pg.InfiniteLine | None = None
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._axis_reference_time: datetime | None = None
        self._visible_channel_names: set[str] = set()
        self._reserved_right_width = 0.0

        plot = self.getPlotItem()
        plot.showGrid(x=True, y=False, alpha=0.2)
        plot.setLabel("bottom", "Time", units="s")
        plot.showAxis("left")
        plot.setMouseEnabled(x=True, y=False)

        plot.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_record(
        self,
        record: DisturbanceRecord,
        *,
        axis_mode: str = AXIS_MODE_RELATIVE,
        axis_reference_time: datetime | None = None,
    ) -> None:
        """Load digital channels from a DisturbanceRecord and build tracks."""
        self.clear()
        self._record = record
        self._axis_reference_time = axis_reference_time or record.timing_info.start_time
        self.set_time_axis_mode(
            axis_mode,
            axis_reference_time=self._axis_reference_time,
        )
        self._time_cache = record.waveform_data["time"].to_numpy(dtype=np.float64)
        self._visible_channel_names = set(default_visible_digital_names(record.digital_channels))

        for ch in record.digital_channels:
            if ch.name in self._visible_channel_names:
                self._add_track(ch)

        self._update_y_axis()
        self._add_trigger_line()
        self._add_cursor()
        self._update_viewport()
        self._update_track_labels()

    def link_x_to(self, view_or_plot: pg.ViewBox | pg.PlotItem) -> None:
        """Link X-axis to an external ViewBox or PlotItem.

        Call with FlexiblePlotCanvas._primary_plot after both widgets are shown.
        Synchronized panning and zooming will update curve data automatically
        via sigXRangeChanged.
        """
        self.getPlotItem().getViewBox().setXLink(view_or_plot)

    def set_cursor_pos(self, t: float) -> None:
        """Move the cursor without re-emitting cursor_moved (prevents sync loops)."""
        if self._cursor is not None:
            self._cursor.blockSignals(True)
            self._cursor.setValue(t)
            self._cursor.blockSignals(False)

    def set_time_axis_mode(
        self,
        mode: TimeDisplayMode | str,
        *,
        axis_reference_time: datetime | None = None,
    ) -> None:
        """Switch X-axis labels without changing digital track time arrays."""
        display_mode = TimeDisplayMode.coerce(mode)
        if axis_reference_time is not None:
            self._axis_reference_time = axis_reference_time
        self._time_axis_mode = display_mode
        self._datetime_axis.set_start_time(
            self._axis_reference_time if display_mode == TimeDisplayMode.ABSOLUTE else None
        )

    def set_visible_channels(self, names: list[str]) -> None:
        """Show only the selected digital tracks without mutating record data."""
        if self._record is None:
            return
        valid_names = {ch.name for ch in self._record.digital_channels}
        self._visible_channel_names = {name for name in names if name in valid_names}
        self._rebuild_visible_tracks()

    def normalize_viewport(self, t_start: float, t_end: float) -> None:
        """Force the digital timeline to the same numeric X range as analog panels."""
        self.getPlotItem().setXRange(float(t_start), float(t_end), padding=0)

    def reserve_grouped_axis_columns(self, right_axis_count: int) -> None:
        """Reserve the same plot chrome as analog canvases for X-pixel alignment."""
        self._reserved_right_width = reserved_right_axis_width(right_axis_count)
        self._apply_axis_geometry_reservation()

    def match_viewbox_geometry(self, reference_viewbox: pg.ViewBox) -> None:
        """Adjust fixed axis reservations to match an analog reference ViewBox."""
        reference_rect = reference_viewbox.sceneBoundingRect()
        own_rect = self.getPlotItem().getViewBox().sceneBoundingRect()
        if reference_rect.isNull() or own_rect.isNull():
            return

        plot = self.getPlotItem()
        left = plot.getAxis("left")
        right = plot.getAxis("right")

        left_delta = float(reference_rect.left() - own_rect.left())
        right_delta = float(own_rect.right() - reference_rect.right())
        if abs(left_delta) > 0.1:
            left.setWidth(max(1.0, float(left.width()) + left_delta))
        if abs(right_delta) > 0.1:
            self._reserved_right_width = max(0.0, self._reserved_right_width + right_delta)
            right.setWidth(self._reserved_right_width)

        vb = plot.getViewBox()
        vb.setGeometry(reference_rect)
        vb.linkedViewChanged(reference_viewbox, vb.XAxis)
        vb._matrixNeedsUpdate = True
        vb.updateMatrix()

    def visible_channel_names(self) -> list[str]:
        """Return visible digital channel names in record order."""
        if self._record is None:
            return []
        return [
            ch.name
            for ch in self._record.digital_channels
            if ch.name in self._visible_channel_names
        ]

    def all_channel_names(self) -> list[str]:
        """Return all digital channel names available in the current record."""
        if self._record is None:
            return []
        return [ch.name for ch in self._record.digital_channels]

    def clear(self) -> None:
        """Remove all tracks, cursor, and trigger line. Reset to blank state."""
        self.getPlotItem().clear()
        self._tracks.clear()
        self._record = None
        self._time_cache = np.empty(0, dtype=np.float64)
        self._cursor = None
        self._trigger_line = None
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._axis_reference_time = None
        self._visible_channel_names.clear()
        self._restore_plot_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _add_track(self, ch: DigitalChannel) -> None:
        track_idx = len(self._tracks)
        y_offset = track_idx * _TRACK_SPACING
        color = digital_role_color(ch.name)

        raw = self._record.waveform_data[ch.name].to_numpy(dtype=np.float64)  # type: ignore[union-attr]
        t_trans, d_trans = extract_transitions(self._time_cache, raw)

        # Alternating row background
        bg_alpha = 18 if track_idx % 2 == 0 else 0
        bg_band = pg.LinearRegionItem(
            values=[y_offset - 0.45, y_offset + 0.45],
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(255, 255, 255, bg_alpha),
            pen=pg.mkPen((0, 0, 0, 0)),
        )
        bg_band.setZValue(-10)
        self.addItem(bg_band)

        # Thick line for HIGH periods, thin dim line for LOW periods
        curve_hi = pg.PlotDataItem(
            pen=pg.mkPen(color, width=_LINE_WIDTH_HI),
            connect="finite",
        )
        curve_lo = pg.PlotDataItem(
            pen=pg.mkPen(color + "44", width=_LINE_WIDTH_LO),
            connect="finite",
        )
        self.addItem(curve_lo)   # draw LOW first (below HIGH)
        self.addItem(curve_hi)

        label = pg.TextItem(
            html=self._render_label_html(ch.name, color, 0),
            anchor=(0.0, 0.5),
            fill=pg.mkBrush("#1E1E1ECC"),
        )
        label.setZValue(20)
        self.addItem(label)

        self._tracks[ch.name] = _TrackEntry(
            name=ch.name,
            curve_hi=curve_hi,
            curve_lo=curve_lo,
            label=label,
            bg_band=bg_band,
            color=color,
            y_offset=y_offset,
            t_trans=t_trans,
            d_trans=d_trans,
        )

    def _rebuild_visible_tracks(self) -> None:
        """Rebuild digital tracks while preserving X range and cursor position."""
        if self._record is None:
            return
        x_range = self.getPlotItem().getViewBox().viewRange()[0]
        cursor_pos = float(self._cursor.value()) if self._cursor is not None else 0.0
        record = self._record
        time_cache = self._time_cache
        mode = self._time_axis_mode
        reference = self._axis_reference_time
        visible = set(self._visible_channel_names)

        self.getPlotItem().clear()
        self._tracks.clear()
        self._record = record
        self._time_cache = time_cache
        self._time_axis_mode = mode
        self._axis_reference_time = reference
        self._visible_channel_names = visible
        self._restore_plot_config()
        self._datetime_axis.set_start_time(
            reference if mode == TimeDisplayMode.ABSOLUTE else None
        )

        for ch in record.digital_channels:
            if ch.name in visible:
                self._add_track(ch)

        self._update_y_axis()
        self._add_trigger_line()
        self._add_cursor()
        self.set_cursor_pos(cursor_pos)
        self.getPlotItem().setXRange(float(x_range[0]), float(x_range[1]), padding=0)
        self._update_viewport()
        self._update_track_labels()

    def _update_y_axis(self) -> None:
        n = len(self._tracks)
        if n == 0:
            return
        plot = self.getPlotItem()
        plot.setYRange(-0.5, (n - 1) * _TRACK_SPACING + 0.5, padding=0)
        # Hide all Y-axis tick marks and values
        left = plot.getAxis("left")
        left.setTicks([[]])
        left.setStyle(showValues=False, tickLength=0)
        self._apply_axis_geometry_reservation()
        self._position_track_labels()

    def _add_trigger_line(self) -> None:
        if self._record is None:
            return
        t_trig = (
            self._record.timing_info.trigger_time
            - self._record.timing_info.start_time
        ).total_seconds()
        self._trigger_line = pg.InfiniteLine(
            pos=t_trig,
            angle=90,
            movable=False,
            pen=pg.mkPen("#FF4444", width=2, style=Qt.PenStyle.DotLine),
            label="T",
            labelOpts={"color": "#FF4444", "position": 0.95},
        )
        self.addItem(self._trigger_line)

    def _add_cursor(self) -> None:
        if self._record is not None:
            initial_pos = (
                self._record.timing_info.trigger_time
                - self._record.timing_info.start_time
            ).total_seconds()
        else:
            initial_pos = 0.0
        self._cursor = pg.InfiniteLine(
            pos=initial_pos,
            angle=90,
            movable=True,
            pen=pg.mkPen("#FFFF00", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self.addItem(self._cursor)

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        t = float(line.value())
        self._update_track_labels(t)
        self.cursor_moved.emit(t)

    def _render_label_html(self, name: str, color: str, state: int) -> str:
        """HTML for the track label: colored indicator square + state digit + name."""
        sq_color = color if state else "#3A3A3A"
        state_color = "#FFFFFF" if state else "#555555"
        return (
            f'<span style="color:{sq_color};">&#9632;</span>'
            f'&nbsp;<span style="color:{state_color};font-weight:bold;">{state}</span>'
            f'&nbsp;&nbsp;<span style="color:#CCCCCC;">{name}</span>'
        )

    def _get_state_at(self, entry: _TrackEntry, t: float) -> int:
        """Return the binary state (0 or 1) for this track at time t."""
        if len(entry.t_trans) == 0:
            return 0
        idx = int(np.searchsorted(entry.t_trans, t, side="right")) - 1
        return int(entry.d_trans[idx]) if idx >= 0 else 0

    def _update_track_labels(self, t: float | None = None) -> None:
        """Refresh all track label HTML to reflect state at cursor time t."""
        if t is None:
            t = float(self._cursor.value()) if self._cursor is not None else 0.0
        for entry in self._tracks.values():
            state = self._get_state_at(entry, t)
            entry.label.setHtml(self._render_label_html(entry.name, entry.color, state))

    def _on_x_range_changed(
        self,
        _viewbox: pg.ViewBox,
        x_range: tuple[float, float],
    ) -> None:
        t_start, t_end = x_range
        self._update_viewport(t_start, t_end)

    def _update_viewport(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> None:
        """Re-clip all digital tracks to the viewport and push to curves.

        Hot path — only clip_digital_to_viewport + build_step_series + setData().
        No DataFrame ops, no to_numpy(), no analytics.
        """
        if len(self._time_cache) == 0:
            return
        if t_start is None or t_end is None:
            t_start, t_end = self.getPlotItem().getViewBox().viewRange()[0]

        _empty = np.empty(0, dtype=np.float64)
        for entry in self._tracks.values():
            t_cl, d_cl = clip_digital_to_viewport(
                entry.t_trans, entry.d_trans, t_start, t_end
            )
            if len(t_cl) < 2:
                entry.curve_hi.setData(_empty, _empty)
                entry.curve_lo.setData(_empty, _empty)
                continue
            t_hi, y_hi = build_hi_lo_segments(t_cl, d_cl, entry.y_offset, 1)
            t_lo, y_lo = build_hi_lo_segments(t_cl, d_cl, entry.y_offset, 0)
            entry.curve_hi.setData(t_hi if len(t_hi) else _empty,
                                   y_hi if len(y_hi) else _empty)
            entry.curve_lo.setData(t_lo if len(t_lo) else _empty,
                                   y_lo if len(y_lo) else _empty)
        self._position_track_labels()

    def _position_track_labels(self) -> None:
        """Keep channel labels pinned near the left edge of the digital lanes."""
        if not self._tracks:
            return
        plot = self.getPlotItem()
        viewbox = plot.getViewBox()
        x_min, x_max = viewbox.viewRange()[0]
        width_px = max(1.0, float(viewbox.width()))
        x_label = float(x_min) + (float(x_max) - float(x_min)) * 8.0 / width_px
        for entry in self._tracks.values():
            entry.label.setPos(x_label, entry.y_offset)

    def _restore_plot_config(self) -> None:
        plot = self.getPlotItem()
        plot.showGrid(x=True, y=False, alpha=0.2)
        plot.setLabel("bottom", "Time", units="s")
        plot.showAxis("left")
        plot.getAxis("left").setTicks([[]])
        self._apply_axis_geometry_reservation()
        plot.setMouseEnabled(x=True, y=False)

    def _apply_axis_geometry_reservation(self) -> None:
        plot = self.getPlotItem()
        left = plot.getAxis("left")
        left.setWidth(GROUPED_LEFT_AXIS_WIDTH_PX)
        left.setStyle(
            autoExpandTextSpace=False,
            tickTextWidth=max(1, GROUPED_LEFT_AXIS_WIDTH_PX - 8),
            showValues=False,
        )
        left.setPen(pg.mkPen((0, 0, 0, 0)))
        left.setTextPen(pg.mkPen((0, 0, 0, 0)))

        right = plot.getAxis("right")
        plot.showAxis("right")
        right.enableAutoSIPrefix(False)
        right.setWidth(self._reserved_right_width)
        right.setStyle(showValues=False)
        right.setPen(pg.mkPen((0, 0, 0, 0)))
        right.setTextPen(pg.mkPen((0, 0, 0, 0)))

        vb = plot.getViewBox()
        vb._matrixNeedsUpdate = True
        vb.updateMatrix()
