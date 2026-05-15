from __future__ import annotations

import dataclasses

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from app.models import DigitalChannel, DisturbanceRecord
from app.visualization.rendering.digital_transforms import (
    build_step_series,
    clip_digital_to_viewport,
    digital_role_color,
    extract_transitions,
)

_TRACK_SPACING = 1.5   # vertical distance between track baselines (data coords)
_TRACK_HEIGHT  = 1.0   # height of the HIGH-state fill within each track


@dataclasses.dataclass
class _TrackEntry:
    name:     str
    curve:    pg.PlotDataItem
    color:    str
    y_offset: float        # baseline Y in data coordinates
    t_trans:  np.ndarray   # pre-extracted transition times (float64)
    d_trans:  np.ndarray   # pre-extracted transition states (float64)


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
        super().__init__(parent=parent)
        self.setBackground("#1E1E1E")

        self._record: DisturbanceRecord | None = None
        self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
        self._tracks: dict[str, _TrackEntry] = {}
        self._trigger_line: pg.InfiniteLine | None = None
        self._cursor: pg.InfiniteLine | None = None

        plot = self.getPlotItem()
        plot.showGrid(x=True, y=False, alpha=0.2)
        plot.setLabel("bottom", "Time", units="s")
        plot.showAxis("left")
        plot.setMouseEnabled(x=True, y=False)

        plot.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_record(self, record: DisturbanceRecord) -> None:
        """Load digital channels from a DisturbanceRecord and build tracks."""
        self.clear()
        self._record = record
        self._time_cache = record.waveform_data["time"].to_numpy(dtype=np.float64)

        for ch in record.digital_channels:
            self._add_track(ch)

        self._update_y_axis()
        self._add_trigger_line()
        self._add_cursor()
        self._update_viewport()

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

    def clear(self) -> None:
        """Remove all tracks, cursor, and trigger line. Reset to blank state."""
        self.getPlotItem().clear()
        self._tracks.clear()
        self._record = None
        self._time_cache = np.empty(0, dtype=np.float64)
        self._cursor = None
        self._trigger_line = None
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

        curve = pg.PlotDataItem(
            pen=pg.mkPen(color, width=1.5),
            fillLevel=y_offset,
            brush=pg.mkBrush(color + "55"),
            skipFiniteCheck=True,
        )
        self.addItem(curve)

        self._tracks[ch.name] = _TrackEntry(
            name=ch.name,
            curve=curve,
            color=color,
            y_offset=y_offset,
            t_trans=t_trans,
            d_trans=d_trans,
        )

    def _update_y_axis(self) -> None:
        n = len(self._tracks)
        if n == 0:
            return
        y_max = (n - 1) * _TRACK_SPACING + _TRACK_HEIGHT + 0.25
        plot = self.getPlotItem()
        plot.setYRange(-0.25, y_max, padding=0)

        ticks = [
            (entry.y_offset + _TRACK_HEIGHT / 2, entry.name)
            for entry in self._tracks.values()
        ]
        plot.getAxis("left").setTicks([ticks])

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
        self._cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#FFFF00", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self.addItem(self._cursor)

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        self.cursor_moved.emit(line.value())

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
                entry.curve.setData(_empty, _empty)
                continue
            t_step, y_step = build_step_series(
                t_cl, d_cl, entry.y_offset, _TRACK_HEIGHT
            )
            entry.curve.setData(t_step, y_step)

    def _restore_plot_config(self) -> None:
        plot = self.getPlotItem()
        plot.showGrid(x=True, y=False, alpha=0.2)
        plot.setLabel("bottom", "Time", units="s")
        plot.showAxis("left")
        plot.getAxis("left").setTicks(None)
        plot.setMouseEnabled(x=True, y=False)
