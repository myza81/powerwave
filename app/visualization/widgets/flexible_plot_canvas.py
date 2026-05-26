from __future__ import annotations

from datetime import datetime
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget

from app.analytics.harmonics.harmonic_models import HarmonicConfig, HarmonicDisplayMode
from app.analytics.phasors.phasor_models import PhasorConfig, PhasorDisplayMode
from app.analytics.rms.rms_models import RMSConfig, RMSDisplayMode
from app.analytics.scaling.scaling_models import EngineeringScalingMode
from app.models import AnalogChannel, DisturbanceRecord
from app.visualization.axis.datetime_axis import (
    AXIS_MODE_ABSOLUTE,
    AXIS_MODE_RELATIVE,
    DatetimeAxisItem,
    TimeDisplayMode,
)
from app.visualization.engineering_display import (
    format_rms_curve_label,
    format_rms_mode_title,
    infer_signal_type as _infer_signal_type,
)
from app.visualization.axis_management import (
    AxisDisplayMode,
    GROUPED_LEFT_AXIS_WIDTH_PX,
    GROUPED_MAX_RIGHT_AXIS_WIDTH_PX,
    GROUPED_RIGHT_AXIS_WIDTH_PX,
    axis_group_for_signal,
)
from app.visualization.managers.multi_axis_manager import MultiAxisManager
from app.visualization.overlays.overlay_colors import (
    is_waveform_phasor_candidate,
    phasor_color,
    phasor_curve_label,
    sequence_color,
)
from app.visualization.performance import timed_section
from app.visualization.rendering.downsampling import decimate_for_display
from app.visualization.signal_visibility import default_visible_analog_names

# Canonical phase-detection color heuristic (VIEWPORT_RENDERING_POLICY §9 + §16)
_AXIS_COLORS = ["#FF4444", "#FFCC00", "#4488FF", "#44BB44", "#AAAAAA", "#FF8800"]
_SPARSE_RATE_THRESHOLD_HZ = 2.0
_SPARSE_INTERVAL_THRESHOLD_S = 2.0


def _phasor_mag_pen_color(hex_color: str) -> str:
    """Return a brighter shade of *hex_color* for phasor magnitude overlay curves.

    Blends 60% toward white — brighter than the RMS overlay (40%) so the two
    overlays remain visually distinguishable when both are active.
    """
    return phasor_color("", PhasorDisplayMode.MAGNITUDE, base_color=hex_color)


def _phasor_angle_pen_color(hex_color: str) -> str:
    """Return a cyan-shifted color for phasor angle overlay curves.

    Shifts the channel color 40% toward #00FFFF (cyan) so angle traces are
    visually distinct from both raw waveforms and magnitude/RMS overlays.
    """
    return phasor_color("", PhasorDisplayMode.ANGLE, base_color=hex_color)


def _rms_pen_color(hex_color: str) -> str:
    """Return a lighter shade of *hex_color* for RMS overlay curves.

    Blends the channel color 40% toward white so the RMS envelope is clearly
    visible against the dark background while remaining visually linked to its
    raw waveform channel.
    """
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return "#FFFFFF"
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r2 = min(255, r + int((255 - r) * 0.4))
    g2 = min(255, g + int((255 - g) * 0.4))
    b2 = min(255, b + int((255 - b) * 0.4))
    return f"#{r2:02X}{g2:02X}{b2:02X}"


def _channel_color(ch: AnalogChannel) -> str:
    """Assign a display color based on channel name phase heuristics."""
    name = ch.name.lower()
    if name.endswith(("v1", "i1", "1")):
        return sequence_color("positive")
    if name.endswith(("v2", "i2", "2")):
        return sequence_color("negative")
    if name.endswith(("v0", "i0", "0")):
        return sequence_color("zero")
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


def _combined_y_range(series_list: list[np.ndarray]) -> tuple[float, float] | None:
    y_min: float | None = None
    y_max: float | None = None
    for data in series_list:
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            continue
        current_min = float(np.min(finite))
        current_max = float(np.max(finite))
        y_min = current_min if y_min is None else min(y_min, current_min)
        y_max = current_max if y_max is None else max(y_max, current_max)
    if y_min is None or y_max is None:
        return None
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
    measurement_cursors_moved = pyqtSignal(float, float)  # t_a, t_b
    # Emitted every cursor move: (t_seconds, [(name, value_or_None, unit, color), …])
    cursor_values_changed = pyqtSignal(float, list)

    def __init__(
        self,
        parent: QWidget | None = None,
        max_display_points: int = 4_000,
        axis_display_mode: AxisDisplayMode | str = AxisDisplayMode.SHARED,
    ) -> None:
        super().__init__(parent=parent)
        self.setBackground("#1E1E1E")

        self._max_pts = max_display_points

        self._record: DisturbanceRecord | None = None
        self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
        self._data_cache: dict[str, np.ndarray] = {}
        self._sparse_mode = False
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._axis_reference_time: datetime | None = None
        self._axis_display_mode = AxisDisplayMode.coerce(axis_display_mode)
        self._visible_channel_names: set[str] = set()

        self._resize_pending = False
        self._cursor: pg.InfiniteLine | None = None
        self._cursor_b: pg.InfiniteLine | None = None      # second measurement cursor
        self._measurement_mode: bool = False
        self._trigger_line: pg.InfiniteLine | None = None
        self._event_markers: list[pg.InfiniteLine] = []   # Phase 2 detection markers
        self._reserved_right_axes: list[pg.AxisItem] = []

        # Engineering scaling state (Phase 5B)
        self._scaling_mode: EngineeringScalingMode = EngineeringScalingMode.RAW
        self._scaling_registry: object = None   # ScalingRegistry — lazy import
        self._scaled_data_cache: dict[str, np.ndarray] = {}
        self._effective_units: dict[str, str] = {}

        # RMS overlay state (Phase 5A)
        self._rms_display_mode: RMSDisplayMode = RMSDisplayMode.OFF
        self._rms_config: RMSConfig = RMSConfig()
        self._rms_signal_metadata: dict = {}
        self._rms_force_channels: set[str] = set()
        self._rms_cache: object = None   # RMSCache — lazy import to avoid top-level cost
        self._rms_curves: dict[str, pg.PlotDataItem] = {}
        self._rms_time_cache: dict[str, np.ndarray] = {}
        self._rms_data_cache: dict[str, np.ndarray] = {}
        self._panel_base_title: str = ""
        self._panel_title: str = ""

        # Phasor overlay state (Phase 6B)
        self._phasor_display_mode: PhasorDisplayMode = PhasorDisplayMode.OFF
        self._phasor_config: PhasorConfig = PhasorConfig()
        self._phasor_signal_metadata: dict = {}
        self._phasor_cache: object = None   # PhasorCache — lazy
        self._phasor_curves: dict[str, pg.PlotDataItem] = {}
        self._phasor_time_cache: dict[str, np.ndarray] = {}
        self._phasor_data_cache: dict[str, np.ndarray] = {}

        # Harmonic overlay state (Phase 8 — rendering wired)
        self._harmonic_display_mode: HarmonicDisplayMode = HarmonicDisplayMode.OFF
        self._harmonic_config: HarmonicConfig = HarmonicConfig()
        self._harmonic_signal_metadata: dict = {}
        self._harmonic_cache: object = None   # HarmonicCache — lazy
        self._harmonic_curves: dict[str, dict[int, pg.PlotDataItem]] = {}
        self._harmonic_time_cache: dict[str, np.ndarray] = {}
        self._harmonic_data_cache: dict[str, dict[int, np.ndarray]] = {}
        # Default display orders: H3/H5/H7/H11/H13 (H1 omitted — too large).
        self._harmonic_display_orders: list[int] = [3, 5, 7, 11, 13]

        self._performance_timing_enabled = False
        self._performance_timing_sink = None
        self._curve_data_signatures: dict[int, tuple] = {}

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
        axis_reference_time: datetime | None = None,
    ) -> None:
        """Load a DisturbanceRecord. Build N-Axis layout. Zoom to trigger.

        Args:
            record:    The waveform record to render.
            axis_mode: AXIS_MODE_RELATIVE (default) — elapsed-seconds labels;
                       AXIS_MODE_ABSOLUTE — wall-clock datetime labels.
        """
        self._clear_canvas()

        self._record = record
        self._axis_reference_time = axis_reference_time or record.timing_info.start_time
        self.set_time_axis_mode(
            axis_mode,
            axis_reference_time=self._axis_reference_time,
        )

        # Cache numpy arrays once — avoids per-frame DataFrame access in _update_viewport
        self._time_cache = record.waveform_data["time"].to_numpy(dtype=np.float64)
        self._sparse_mode = _is_sparse_timeseries(record, self._time_cache)
        self._data_cache = {
            ch.name: record.waveform_data[ch.name].to_numpy(dtype=np.float64)
            for ch in record.analog_channels
        }

        # Apply current scaling mode before building axes so effective units
        # are available when _add_channel_axis() computes grouping keys.
        self._build_scaled_arrays()

        self._visible_channel_names = set(default_visible_analog_names(record.analog_channels))
        for ch in record.analog_channels:
            if ch.name in self._visible_channel_names:
                self._add_channel_axis(ch)

        self._add_trigger_line()
        self._add_cursor()
        self.zoom_to_trigger()
        self._update_viewport()
        # For sparse records the full dataset is always in-viewport, so pinning
        # Y ranges from the raw data cache is both correct and stable.
        if self._sparse_mode:
            self._force_y_ranges()

        # Rebuild RMS overlays for the new record if mode is active
        if self._rms_display_mode != RMSDisplayMode.OFF:
            self._build_rms_overlays(
                self._rms_config,
                self._rms_signal_metadata or None,
                self._rms_force_channels or None,
            )
            self._update_viewport()

        # Rebuild phasor overlays for the new record if mode is active
        if self._phasor_display_mode not in (
            PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS
        ):
            self._build_phasor_overlays(
                self._phasor_display_mode, self._phasor_config, None
            )
            self._update_viewport()

        # Rebuild harmonic overlays for the new record if mode is active
        if self._harmonic_display_mode == HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            self._build_harmonic_overlays(self._harmonic_config, None)
            self._update_viewport()

    def set_panel_title(self, title: str) -> None:
        """Set a concise grouped-display panel title."""
        self._panel_base_title = title.strip()
        self._refresh_panel_title()

    def set_performance_timing(self, enabled: bool, sink=None) -> None:
        """Enable optional visualization timing callbacks for overlay hot paths."""
        self._performance_timing_enabled = bool(enabled)
        self._performance_timing_sink = sink

    def set_time_axis_mode(
        self,
        mode: TimeDisplayMode | str,
        *,
        axis_reference_time: datetime | None = None,
    ) -> None:
        """Switch X-axis labels between elapsed seconds and wall-clock time."""
        display_mode = TimeDisplayMode.coerce(mode)
        if axis_reference_time is not None:
            self._axis_reference_time = axis_reference_time
        self._time_axis_mode = display_mode
        self._apply_datetime_axis_mode(
            self._axis_reference_time if display_mode == TimeDisplayMode.ABSOLUTE else None
        )

    def _apply_datetime_axis_mode(self, start_time: datetime | None) -> None:
        """Call set_start_time on the live bottom-axis object from the PlotItem.

        Retrieving the axis through getAxis() each time avoids using a stale
        Python wrapper whose underlying C++ QGraphicsItem may have been deleted
        by PyQt6's ownership transfer during PlotItem.clear() calls.
        """
        bottom = self._primary_plot.getAxis("bottom")
        if isinstance(bottom, DatetimeAxisItem):
            self._datetime_axis = bottom  # keep our reference in sync
            bottom.set_start_time(start_time)

    def set_axis_display_mode(self, mode: AxisDisplayMode | str) -> None:
        """Switch between shared and dedicated Y-axis grouping."""
        axis_mode = AxisDisplayMode.coerce(mode)
        if axis_mode == self._axis_display_mode:
            return
        self._axis_display_mode = axis_mode
        self._rebuild_visible_channel_axes()

    def axis_display_mode(self) -> AxisDisplayMode:
        """Return the active Y-axis grouping mode."""
        return self._axis_display_mode

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

        group = axis_group_for_signal(name, unit, mode=self._axis_display_mode)
        vb = self._axis_manager.add_axis(
            name,
            unit,
            effective_color,
            axis_key=group.key,
            axis_label=group.label,
        )

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

        Visibility is render-layer state only. The record and cached data arrays
        remain intact; ViewBoxes, axes, raw curves, and RMS overlay curves are
        rebuilt for the requested visible subset.
        """
        if self._record is None:
            return
        valid_names = {ch.name for ch in self._record.analog_channels}
        self._visible_channel_names = {name for name in names if name in valid_names}
        self._rebuild_visible_channel_axes()

    def visible_channel_names(self) -> list[str]:
        """Return visible analog channel names in record order."""
        if self._record is None:
            return []
        return [
            ch.name
            for ch in self._record.analog_channels
            if ch.name in self._visible_channel_names
        ]

    def all_channel_names(self) -> list[str]:
        """Return all analog channel names available in the current record."""
        if self._record is None:
            return []
        return [ch.name for ch in self._record.analog_channels]

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

    # ─────────────────────────────────────────────────────────────────────────
    # Measurement mode (Phase 1)
    # ─────────────────────────────────────────────────────────────────────────

    def set_measurement_mode(self, enabled: bool) -> None:
        """Enable or disable two-cursor measurement mode.

        When enabled, a second cyan cursor (B) appears. Both cursors emit
        measurement_cursors_moved(t_a, t_b) whenever either moves.
        When disabled, cursor B is hidden and removed.
        """
        if enabled == self._measurement_mode:
            return
        self._measurement_mode = enabled
        if enabled:
            self._add_cursor_b()
        else:
            self._remove_cursor_b()

    def measurement_mode(self) -> bool:
        return self._measurement_mode

    def cursor_positions(self) -> tuple[float, float] | None:
        """Return (t_a, t_b) or None if either cursor is missing."""
        if self._cursor is None or self._cursor_b is None:
            return None
        return float(self._cursor.value()), float(self._cursor_b.value())

    def compute_current_measurements(self) -> object | None:
        """Compute and return a MeasurementResult for the current cursor pair.

        Returns None when measurement mode is off or no record is loaded.
        """
        if not self._measurement_mode or self._record is None:
            return None
        positions = self.cursor_positions()
        if positions is None:
            return None
        from app.visualization.interaction.measurement_engine import compute_measurements
        t_a, t_b = positions
        data_by_channel: dict[str, tuple] = {}
        for ch in self._record.analog_channels:
            if ch.name not in self._visible_channel_names:
                continue
            arr = self._get_display_data(ch.name)
            if arr is not None:
                unit = self._effective_units.get(ch.name, ch.unit or "")
                data_by_channel[ch.name] = (arr, unit)
        nominal = None
        if self._record is not None:
            nominal = getattr(self._record.metadata, "nominal_frequency", None)
            try:
                nominal = float(nominal)
            except (TypeError, ValueError):
                nominal = None
        return compute_measurements(
            t_a, t_b, self._time_cache, data_by_channel, nominal_hz=nominal
        )

    def _add_cursor_b(self) -> None:
        if self._cursor_b is not None:
            return
        t_ref = float(self._cursor.value()) + 0.02 if self._cursor is not None else 0.02
        self._cursor_b = pg.InfiniteLine(
            pos=t_ref,
            angle=90,
            movable=True,
            pen=pg.mkPen("#00CCFF", width=1.5, style=Qt.PenStyle.DashLine),
            label="B",
            labelOpts={"color": "#00CCFF", "position": 0.85},
        )
        self._cursor_b.sigPositionChanged.connect(self._on_cursor_b_moved)
        self._primary_plot.addItem(self._cursor_b)
        self._emit_measurement()

    def _remove_cursor_b(self) -> None:
        if self._cursor_b is not None:
            try:
                self._primary_plot.removeItem(self._cursor_b)
            except Exception:
                pass
            self._cursor_b = None

    def _on_cursor_b_moved(self, _line: pg.InfiniteLine) -> None:
        self._emit_measurement()

    def _emit_measurement(self) -> None:
        if self._cursor is None or self._cursor_b is None:
            return
        self.measurement_cursors_moved.emit(
            float(self._cursor.value()),
            float(self._cursor_b.value()),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Event markers (Phase 2)
    # ─────────────────────────────────────────────────────────────────────────

    def add_event_markers(self, events: list) -> None:
        """Place InfiniteLine markers on the canvas for each detected event.

        Each event gets one marker at t_start colored by event type. An end
        marker is also added when duration > 0.5 cycle (20ms) to bracket the
        event region.
        """
        self.clear_event_markers()
        for ev in events:
            color = ev.event_type.color
            label = ev.event_type.label
            start_line = pg.InfiniteLine(
                pos=ev.t_start,
                angle=90,
                movable=False,
                pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashDotLine),
                label=label,
                labelOpts={"color": color, "position": 0.75},
            )
            self._primary_plot.addItem(start_line)
            self._event_markers.append(start_line)

            if ev.duration_ms > 20.0:
                end_line = pg.InfiniteLine(
                    pos=ev.t_end,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(color, width=1.0, style=Qt.PenStyle.DotLine),
                )
                self._primary_plot.addItem(end_line)
                self._event_markers.append(end_line)

    def clear_event_markers(self) -> None:
        """Remove all event marker lines from the canvas."""
        for line in self._event_markers:
            try:
                self._primary_plot.removeItem(line)
            except Exception:
                pass
        self._event_markers.clear()

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

        self._remove_reserved_right_axes()
        self._axis_manager.clear()
        self._primary_plot.clear()

        # Restore primary plot appearance (clear() strips labels and grid)
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")
        self._apply_datetime_axis_mode(None)
        self._refresh_panel_title()

        # Fresh axis manager — its __init__ connects _sync_geometries FIRST
        self._axis_manager = MultiAxisManager(self._primary_plot, self)
        # Reconnect _on_plot_resized AFTER _sync_geometries
        vb.sigResized.connect(self._on_plot_resized)

        self._resize_pending = False
        self._record = None
        self._time_cache = np.empty(0, dtype=np.float64)
        self._data_cache.clear()
        self._scaled_data_cache.clear()
        self._effective_units.clear()
        self._sparse_mode = False
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._axis_reference_time = None
        self._visible_channel_names.clear()
        self._cursor = None
        self._cursor_b = None  # removed by PlotItem.clear() above
        self._trigger_line = None
        self._event_markers.clear()  # items removed by PlotItem.clear() above
        self._curve_data_signatures.clear()

        # RMS overlay — curves removed by PlotItem.clear() / scene removeItem;
        # only Python-side dicts need explicit reset here.
        self._rms_curves.clear()
        self._rms_time_cache.clear()
        self._rms_data_cache.clear()
        if self._rms_cache is not None:
            self._rms_cache.clear()  # type: ignore[union-attr]

        # Phasor overlay — same pattern: scene handles item removal on clear().
        self._phasor_curves.clear()
        self._phasor_time_cache.clear()
        self._phasor_data_cache.clear()
        if self._phasor_cache is not None:
            self._phasor_cache.clear()  # type: ignore[union-attr]

        # Harmonic overlay — same pattern.
        self._harmonic_curves.clear()
        self._harmonic_time_cache.clear()
        self._harmonic_data_cache.clear()
        if self._harmonic_cache is not None:
            self._harmonic_cache.clear()  # type: ignore[union-attr]

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

    def _add_channel_axis(self, ch: AnalogChannel) -> None:
        """Create a visible axis and raw curve for one analog channel."""
        color = _channel_color(ch)
        effective_unit = self._effective_units.get(ch.name, ch.unit or "")
        # When scaling changes the unit (e.g. kV → pu) we still want voltage
        # channels to share a "Voltage (pu)" axis rather than merging with the
        # generic "per_unit" role.  Passing the original signal type as a hint
        # keeps the grouping key role-correct.
        signal_hint = _infer_signal_type(ch.name, ch.unit or "")
        group = axis_group_for_signal(
            ch.name,
            effective_unit,
            mode=self._axis_display_mode,
            signal_type_hint=signal_hint,
        )
        vb = self._axis_manager.add_axis(
            ch.name,
            effective_unit,
            color,
            axis_key=group.key,
            axis_label=group.label,
        )
        curve = self._make_curve(color)
        if vb is self._primary_plot.getViewBox():
            self._primary_plot.addItem(curve)
        else:
            vb.addItem(curve)
        self._axis_manager.register(ch.name, vb, curve, color)

    def _reset_plot_for_axis_rebuild(self) -> None:
        """Clear plot items and axes while preserving record/data caches."""
        vb = self._primary_plot.getViewBox()
        try:
            vb.sigResized.disconnect(self._axis_manager._sync_geometries)
        except TypeError:
            pass
        try:
            vb.sigResized.disconnect(self._on_plot_resized)
        except TypeError:
            pass

        self._remove_rms_curves()
        self._remove_harmonic_curves()
        self._remove_reserved_right_axes()
        self._axis_manager.clear()
        self._primary_plot.clear()
        self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
        self._primary_plot.setLabel("bottom", "Time")
        self._apply_datetime_axis_mode(
            self._axis_reference_time
            if self._time_axis_mode == TimeDisplayMode.ABSOLUTE
            else None
        )
        self._refresh_panel_title()

        self._axis_manager = MultiAxisManager(self._primary_plot, self)
        vb.sigResized.connect(self._on_plot_resized)
        self._cursor = None
        self._cursor_b = None  # removed by PlotItem.clear() above
        self._trigger_line = None
        self._event_markers.clear()  # removed by PlotItem.clear() above

    def _rebuild_visible_channel_axes(self) -> None:
        """Rebuild visible axes without reloading or mutating waveform data."""
        if self._record is None:
            return
        x_range = self._primary_plot.getViewBox().viewRange()[0]
        cursor_pos = float(self._cursor.value()) if self._cursor is not None else 0.0
        cursor_b_pos = float(self._cursor_b.value()) if self._cursor_b is not None else None

        self._reset_plot_for_axis_rebuild()
        for ch in self._record.analog_channels:
            if ch.name in self._visible_channel_names:
                self._add_channel_axis(ch)

        self._add_trigger_line()
        self._add_cursor()
        self.set_cursor_pos(cursor_pos)
        if self._measurement_mode:
            self._add_cursor_b()
            if cursor_b_pos is not None and self._cursor_b is not None:
                self._cursor_b.blockSignals(True)
                self._cursor_b.setValue(cursor_b_pos)
                self._cursor_b.blockSignals(False)
        self._primary_plot.setXRange(float(x_range[0]), float(x_range[1]), padding=0)

        if self._rms_display_mode != RMSDisplayMode.OFF:
            self._build_rms_overlays(
                self._rms_config,
                self._rms_signal_metadata or None,
                self._rms_force_channels or None,
            )

        if self._phasor_display_mode not in (
            PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS
        ):
            self._build_phasor_overlays(
                self._phasor_display_mode, self._phasor_config, None
            )

        if self._harmonic_display_mode == HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            self._build_harmonic_overlays(self._harmonic_config, None)

        self._update_viewport()
        if self._sparse_mode:
            self._force_y_ranges()

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

    def right_axis_count(self) -> int:
        """Return the number of real secondary right-side Y axes."""
        return max(0, self._axis_manager.axis_count() - 1)

    def reserve_grouped_axis_columns(self, right_axis_count: int) -> None:
        """Reserve matching axis columns for stacked grouped panels.

        PyQtGraph sizes each PlotItem from the axes present in that widget. In a
        grouped display, a dual-axis panel therefore gets a narrower ViewBox
        than a single-axis panel, which makes identical X values map to different
        pixels even when their numeric X ranges match. Reserving the same axis
        columns across the stack keeps the canonical X domain visually aligned.
        """
        self._remove_reserved_right_axes()

        left_axis = self._primary_plot.getAxis("left")
        left_axis.setWidth(GROUPED_LEFT_AXIS_WIDTH_PX)

        real_right_count = self.right_axis_count()
        target_right_count = max(real_right_count, int(right_axis_count))
        target_right_width = min(
            target_right_count * GROUPED_RIGHT_AXIS_WIDTH_PX,
            GROUPED_MAX_RIGHT_AXIS_WIDTH_PX,
        )
        if target_right_count == 0 or target_right_width <= 0:
            primary_vb = self._primary_plot.getViewBox()
            primary_vb._matrixNeedsUpdate = True
            primary_vb.updateMatrix()
            return
        target_slots = min(
            target_right_count,
            max(1, int(np.ceil(target_right_width / GROUPED_RIGHT_AXIS_WIDTH_PX))),
        )
        if real_right_count <= target_slots:
            real_right_width = target_right_width / target_slots
            placeholder_count = target_slots - real_right_count
            placeholder_width = real_right_width
        else:
            real_right_width = target_right_width / real_right_count
            placeholder_count = 0
            placeholder_width = 0.0

        primary_vb = self._primary_plot.getViewBox()
        seen_axes: set[int] = set()
        for entry in self._axis_manager._axes.values():
            if entry.viewbox is not primary_vb:
                axis_id = id(entry.axis_item)
                if axis_id in seen_axes:
                    continue
                seen_axes.add(axis_id)
                entry.axis_item.setWidth(real_right_width)

        for placeholder_index in range(placeholder_count):
            axis = pg.AxisItem(orientation="right")
            axis.enableAutoSIPrefix(False)
            axis.setWidth(placeholder_width)
            axis.setStyle(showValues=False)
            axis.setPen(pg.mkPen((0, 0, 0, 0)))
            axis.setTextPen(pg.mkPen((0, 0, 0, 0)))
            self.addItem(axis, row=0, col=1 + real_right_count + placeholder_index)
            self._reserved_right_axes.append(axis)

        primary_vb._matrixNeedsUpdate = True
        primary_vb.updateMatrix()

    def _remove_reserved_right_axes(self) -> None:
        for axis in self._reserved_right_axes:
            try:
                self.removeItem(axis)
            except Exception:
                if axis.scene():
                    axis.scene().removeItem(axis)
        self._reserved_right_axes.clear()

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

    def _get_display_data(self, name: str) -> np.ndarray | None:
        """Return the current display array for a channel.

        Returns the scaled view when engineering scaling is active, else the
        raw data.  Returns None if the channel has no data at all.
        """
        if name in self._scaled_data_cache:
            return self._scaled_data_cache[name]
        return self._data_cache.get(name)

    def _force_y_ranges(self) -> None:
        """Pin each ViewBox's Y range to its channel's full-data extent.

        Used for sparse-mode records where the full dataset is always in the
        viewport.  Computes the range from the display data cache (scaled when
        active, else raw) so Y ranges are stable and engineering-correct.
        """
        data_by_viewbox: dict[pg.ViewBox, list[np.ndarray]] = {}
        for name, entry in self._axis_manager._axes.items():
            display_data = self._get_display_data(name)
            if display_data is not None:
                data_by_viewbox.setdefault(entry.viewbox, []).append(display_data)
            if name in self._rms_data_cache:
                data_by_viewbox.setdefault(entry.viewbox, []).append(self._rms_data_cache[name])
        self._set_grouped_y_ranges(data_by_viewbox)

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

        rms_only = self._rms_display_mode == RMSDisplayMode.RMS_ONLY
        rms_active = self._rms_display_mode in (
            RMSDisplayMode.OVERLAY,
            RMSDisplayMode.RMS_ONLY,
        )

        data_by_viewbox: dict[pg.ViewBox, list[np.ndarray]] = {}

        for name, curve in self._axis_manager.get_curves().items():
            display_data = self._get_display_data(name)
            if display_data is not None:
                # RMS_ONLY: suppress raw/scaled data for channels with an RMS overlay
                if rms_only and name in self._rms_data_cache:
                    self._set_curve_data_if_changed(
                        curve,
                        np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64),
                    )
                else:
                    if self._sparse_mode:
                        t_dec, d_dec = _sparse_display_series(
                            self._time_cache,
                            display_data,
                            t_start,
                            t_end,
                        )
                    else:
                        t_dec, d_dec = decimate_for_display(
                            self._time_cache,
                            display_data,
                            t_start,
                            t_end,
                            self._max_pts,
                        )
                    self._set_curve_data_if_changed(curve, t_dec, d_dec)
                    viewbox = curve.getViewBox()
                    if viewbox is not None:
                        data_by_viewbox.setdefault(viewbox, []).append(d_dec)

        # RMS overlay rendering — no analytics here, only array slicing + setData
        if rms_active and self._rms_curves:
            for name, rms_curve in self._rms_curves.items():
                rms_t = self._rms_time_cache.get(name)
                rms_d = self._rms_data_cache.get(name)
                if rms_t is not None and rms_d is not None and len(rms_t) > 0:
                    t_rms_dec, d_rms_dec = decimate_for_display(
                        rms_t, rms_d, t_start, t_end, self._max_pts
                    )
                    self._set_curve_data_if_changed(rms_curve, t_rms_dec, d_rms_dec)
                    viewbox = rms_curve.getViewBox()
                    if viewbox is not None:
                        data_by_viewbox.setdefault(viewbox, []).append(d_rms_dec)

        # Phasor overlay rendering — only array slicing + setData here, no analytics
        _phasor_active = self._phasor_display_mode in (
            PhasorDisplayMode.MAGNITUDE, PhasorDisplayMode.ANGLE
        )
        if _phasor_active and self._phasor_curves:
            for name, phasor_curve in self._phasor_curves.items():
                ph_t = self._phasor_time_cache.get(name)
                ph_d = self._phasor_data_cache.get(name)
                if ph_t is not None and ph_d is not None and len(ph_t) > 0:
                    t_ph_dec, d_ph_dec = decimate_for_display(
                        ph_t, ph_d, t_start, t_end, self._max_pts
                    )
                    self._set_curve_data_if_changed(phasor_curve, t_ph_dec, d_ph_dec)
                    # Only magnitude contributes to ViewBox Y range.
                    # Angle is degrees [-180, 180] — including it would corrupt
                    # the voltage/current Y axis scale.
                    if self._phasor_display_mode == PhasorDisplayMode.MAGNITUDE:
                        viewbox = phasor_curve.getViewBox()
                        if viewbox is not None:
                            data_by_viewbox.setdefault(viewbox, []).append(d_ph_dec)

        # Harmonic overlay rendering — HARMONIC_MAGNITUDE mode only
        _harmonic_active = (
            self._harmonic_display_mode == HarmonicDisplayMode.HARMONIC_MAGNITUDE
        )
        if _harmonic_active and self._harmonic_curves:
            for name, order_curves in self._harmonic_curves.items():
                h_t = self._harmonic_time_cache.get(name)
                if h_t is None or len(h_t) == 0:
                    continue
                order_data = self._harmonic_data_cache.get(name, {})
                for order, h_curve in order_curves.items():
                    h_d = order_data.get(order)
                    if h_d is not None and len(h_d) > 0:
                        t_h_dec, d_h_dec = decimate_for_display(
                            h_t, h_d, t_start, t_end, self._max_pts
                        )
                        self._set_curve_data_if_changed(h_curve, t_h_dec, d_h_dec)
                        viewbox = h_curve.getViewBox()
                        if viewbox is not None:
                            data_by_viewbox.setdefault(viewbox, []).append(d_h_dec)

        self._set_grouped_y_ranges(data_by_viewbox)

    def _set_curve_data_if_changed(
        self,
        curve: pg.PlotDataItem,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        """Avoid redundant setData calls for repeated synchronized viewport echoes."""
        signature = self._curve_data_signature(x_data, y_data)
        key = id(curve)
        if self._curve_data_signatures.get(key) == signature:
            return
        curve.setData(x_data, y_data)
        self._curve_data_signatures[key] = signature

    @staticmethod
    def _curve_data_signature(
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> tuple:
        x_len = int(len(x_data))
        y_len = int(len(y_data))
        if x_len == 0 or y_len == 0:
            return (x_len, y_len)
        return (
            x_len,
            y_len,
            float(x_data[0]),
            float(x_data[-1]),
            float(y_data[0]),
            float(y_data[-1]),
        )

    def _set_grouped_y_ranges(
        self,
        data_by_viewbox: dict[pg.ViewBox, list[np.ndarray]],
    ) -> None:
        for viewbox, series_list in data_by_viewbox.items():
            y_range = _combined_y_range(series_list)
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
        self._primary_plot.addItem(self._cursor)

    def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
        t = float(line.value())
        self.cursor_moved.emit(t)
        if self._measurement_mode:
            self._emit_measurement()
        values = self._compute_cursor_values(t)
        if values:
            self.cursor_values_changed.emit(t, values)

    def _compute_cursor_values(
        self, t: float
    ) -> list[tuple[str, float | None, str, str]]:
        """Return interpolated channel values at time t.

        Returns a list of (channel_name, value_or_None, unit, hex_color) for
        every currently visible channel. Uses linear interpolation between the
        two samples bracketing t.
        """
        if len(self._time_cache) < 2:
            return []
        idx = int(np.searchsorted(self._time_cache, t, side="left"))
        results: list[tuple[str, float | None, str, str]] = []
        for name, entry in self._axis_manager._axes.items():
            data = self._get_display_data(name)
            if data is None or len(data) < 1:
                continue
            if idx <= 0:
                raw = data[0]
                val = float(raw) if np.isfinite(raw) else None
            elif idx >= len(data):
                raw = data[-1]
                val = float(raw) if np.isfinite(raw) else None
            else:
                t0 = float(self._time_cache[idx - 1])
                t1 = float(self._time_cache[idx])
                y0, y1 = float(data[idx - 1]), float(data[idx])
                if np.isfinite(y0) and np.isfinite(y1) and abs(t1 - t0) > 1e-15:
                    frac = (t - t0) / (t1 - t0)
                    val = y0 + frac * (y1 - y0)
                elif np.isfinite(y0):
                    val = y0
                else:
                    val = None
            unit = self._effective_units.get(name, "") or ""
            results.append((name, val, unit, entry.color))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Engineering scaling (Phase 5B)
    # ─────────────────────────────────────────────────────────────────────────

    def set_scaling_mode(
        self,
        mode: EngineeringScalingMode,
        *,
        registry: object = None,
    ) -> None:
        """Apply an engineering scaling mode to all visible waveforms.

        This is a visualization-layer operation only — raw waveform arrays in
        _data_cache are never mutated.  _scaled_data_cache holds the current
        display view (raw * factor) and is rebuilt on every mode change.

        Args:
            mode:     New scaling mode (RAW / PRIMARY / SECONDARY / PER_UNIT).
            registry: Optional ScalingRegistry to replace the current one.
        """
        if registry is not None:
            self._scaling_registry = registry
        if mode == self._scaling_mode and registry is None:
            return

        self._scaling_mode = mode
        self._build_scaled_arrays()

        # RMS cache holds values computed from the previous scale — clear it so
        # overlays are recomputed on the new scaled arrays when rebuilt.
        if self._rms_cache is not None:
            self._rms_cache.clear()  # type: ignore[union-attr]
        self._rms_curves.clear()
        self._rms_time_cache.clear()
        self._rms_data_cache.clear()
        curve_signatures = getattr(self, "_curve_data_signatures", None)
        if curve_signatures is not None:
            curve_signatures.clear()

        # Rebuild axes: effective unit may have changed (e.g. kV → pu), which
        # changes axis labels and shared-axis grouping keys.
        if self._record is not None:
            self._rebuild_visible_channel_axes()
            if self._rms_display_mode != RMSDisplayMode.OFF:
                self._build_rms_overlays(
                    self._rms_config,
                    self._rms_signal_metadata or None,
                    self._rms_force_channels or None,
                )

        self._update_viewport()

    def set_scaling_registry(self, registry: object) -> None:
        """Update scaling configuration without changing the scaling mode."""
        self.set_scaling_mode(self._scaling_mode, registry=registry)

    def _build_scaled_arrays(self) -> None:
        """Pre-compute scaled display arrays for the current scaling mode.

        If mode is RAW, _scaled_data_cache is left empty and _update_viewport
        falls back to _data_cache directly — no extra allocation.
        """
        self._scaled_data_cache.clear()
        self._effective_units.clear()

        if self._scaling_mode == EngineeringScalingMode.RAW or self._record is None:
            return

        from app.analytics.scaling.scaling_registry import ScalingRegistry

        if self._scaling_registry is None:
            self._scaling_registry = ScalingRegistry()

        for ch in self._record.analog_channels:
            if ch.name not in self._data_cache:
                continue
            result = self._scaling_registry.compute_scaling_result(  # type: ignore[union-attr]
                ch.name, ch.unit, self._scaling_mode
            )
            self._effective_units[ch.name] = result.display_unit
            if result.configured and result.factor != 1.0:
                self._scaled_data_cache[ch.name] = (
                    self._data_cache[ch.name] * result.factor
                )
            # unconfigured (e.g. PER_UNIT without a voltage base) → fall back
            # to raw so we never display a silently wrong per-unit value.

    # ─────────────────────────────────────────────────────────────────────────
    # RMS overlay (Phase 5A)
    # ─────────────────────────────────────────────────────────────────────────

    def set_rms_display_mode(
        self,
        mode: RMSDisplayMode,
        *,
        config: RMSConfig | None = None,
        signal_metadata: dict | None = None,
        force_channels: set[str] | None = None,
    ) -> None:
        """Set the global RMS overlay display mode for this canvas.

        OFF       — raw waveform only (default).
        OVERLAY   — raw waveform + RMS envelope together.
        RMS_ONLY  — RMS envelope only; raw waveform hidden for eligible channels.

        Switching to OVERLAY/RMS_ONLY triggers RMS computation for eligible
        channels on the first call; subsequent mode switches reuse the cache.
        Switching back to OFF removes overlay curves and restores raw rendering.

        Args:
            mode:            New display mode.
            config:          RMS configuration (nominal frequency, cycles/window).
            signal_metadata: Optional dict[channel_name → SignalMetadata] for
                             improved eligibility classification.
            force_channels:  Channel names to force RMS on regardless of the
                             automatic eligibility check (operator override).
        """
        if config is not None:
            self._rms_config = config
        if signal_metadata is not None:
            self._rms_signal_metadata = signal_metadata
        if force_channels is not None:
            self._rms_force_channels = set(force_channels)

        self._rms_display_mode = mode
        self._refresh_panel_title()

        if mode == RMSDisplayMode.OFF:
            self._remove_rms_curves()
            self._update_viewport()
            return

        # Build overlays when none exist yet (first activation or after OFF)
        if not self._rms_curves:
            self._build_rms_overlays(
                self._rms_config,
                self._rms_signal_metadata or None,
                self._rms_force_channels or None,
            )

        self._update_viewport()

    def _build_rms_overlays(
        self,
        config: RMSConfig,
        signal_metadata: dict | None,
        force_channels: set[str] | None,
    ) -> None:
        """Compute RMS for eligible channels and add overlay curves to ViewBoxes.

        Hot-path guard: all heavy computation (sliding_rms) happens here once
        per record load.  Subsequent viewport updates only call setData().

        The RMSCache is queried first so that switching modes OFF→ON reuses
        previously computed results without re-running the O(N) calculation.
        """
        from app.analytics.rms.rms_cache import RMSCache
        from app.analytics.rms.rms_overlay import classify_rms_eligibility
        from app.analytics.rms.sliding_rms import compute_rms_overlay, compute_rms_window_samples

        if self._record is None or len(self._time_cache) == 0:
            return

        config = self._resolve_rms_config(config)

        # Lazily create the per-canvas cache
        if self._rms_cache is None:
            self._rms_cache = RMSCache()

        # Estimate sample rate — prefer record metadata, fall back to time array
        rates = [r for r in self._record.sampling_info.sampling_rates if r > 0]
        if rates:
            sample_rate_hz = float(rates[0])
        elif len(self._time_cache) >= 2:
            diffs = np.diff(self._time_cache[: min(100, len(self._time_cache))])
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            sample_rate_hz = 1.0 / float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            sample_rate_hz = 0.0

        if sample_rate_hz <= 0:
            return

        window = compute_rms_window_samples(sample_rate_hz, config)
        forced: set[str] = force_channels or set()

        for ch in self._record.analog_channels:
            name = ch.name
            display_data = self._get_display_data(name)
            if display_data is None or name not in self._axis_manager._axes:
                continue

            sig_meta = None if signal_metadata is None else signal_metadata.get(name)
            eligibility = classify_rms_eligibility(name, sig_meta, force=name in forced)

            if not eligibility.eligible:
                continue

            if not eligibility.auto_classified:
                import sys
                print(f"[RMS] Manual override active for '{name}'", file=sys.stderr)

            # Cache lookup before compute.  Cache is cleared when scaling mode
            # changes so stale entries from a different scale are never reused.
            cached = self._rms_cache.get(name, window, sample_rate_hz)  # type: ignore[union-attr]
            if cached is not None:
                rms_t, rms_d = cached
            else:
                try:
                    rms_t, rms_d = compute_rms_overlay(
                        self._time_cache,
                        display_data,   # scaled when active, raw otherwise
                        sample_rate_hz,
                        config=config,
                    )
                except ValueError:
                    continue
                self._rms_cache.put(name, window, sample_rate_hz, rms_t, rms_d)  # type: ignore[union-attr]

            self._rms_time_cache[name] = rms_t
            self._rms_data_cache[name] = rms_d

            # Create RMS curve — lighter shade of the channel color, width=2
            channel_color = self._axis_manager._axes[name].color
            rms_color = _rms_pen_color(channel_color)
            rms_curve = pg.PlotDataItem(
                pen=pg.mkPen(rms_color, width=2, style=Qt.PenStyle.DashLine),
                name=format_rms_curve_label(name, ch.unit),
                skipFiniteCheck=True,
            )
            rms_curve.setClipToView(True)

            # Add to the same ViewBox as the raw channel
            vb = self._axis_manager._axes[name].viewbox
            if vb is self._primary_plot.getViewBox():
                self._primary_plot.addItem(rms_curve)
            else:
                vb.addItem(rms_curve)

            self._rms_curves[name] = rms_curve

    def _resolve_rms_config(self, config: RMSConfig) -> RMSConfig:
        """Use record nominal frequency when the global config is still default."""
        if self._record is None:
            return config
        record_nominal = getattr(self._record.metadata, "nominal_frequency", None)
        try:
            nominal = float(record_nominal)
        except (TypeError, ValueError):
            return config
        if nominal <= 0:
            return config
        if config.nominal_frequency_hz == RMSConfig().nominal_frequency_hz:
            return replace(config, nominal_frequency_hz=nominal)
        return config

    def _remove_rms_curves(self) -> None:
        """Remove all RMS overlay curves from their ViewBoxes."""
        for rms_curve in self._rms_curves.values():
            try:
                vb = rms_curve.getViewBox()
                if vb is not None:
                    vb.removeItem(rms_curve)
            except Exception:  # noqa: BLE001
                pass
        self._rms_curves.clear()
        self._rms_time_cache.clear()
        self._rms_data_cache.clear()
        # _rms_cache intentionally preserved — reused when mode is re-enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Phasor overlay (Phase 6B)
    # ─────────────────────────────────────────────────────────────────────────

    def set_phasor_display_mode(
        self,
        mode: PhasorDisplayMode,
        *,
        config: PhasorConfig | None = None,
        signal_metadata: dict | None = None,
        phasor_cache: object = None,
    ) -> None:
        """Set the phasor overlay display mode for this canvas.

        OFF               — raw waveform only; all phasor curves removed.
        MAGNITUDE         — phasor RMS magnitude envelope overlaid on raw waveforms.
        ANGLE             — phasor angle trace overlaid on raw waveforms.
        SEQUENCE_COMPONENTS — no overlay drawn here; handled by dedicated panels.

        Switching between MAGNITUDE and ANGLE discards and rebuilds curves;
        switching back to the same active mode is a no-op (no redundant compute).
        The underlying PhasorCache is preserved across mode switches so that
        re-enabling a mode reuses previously computed results.

        Args:
            mode:           New display mode.
            config:         Optional PhasorConfig override (nominal Hz, window).
            signal_metadata: Optional per-channel SignalMetadata mapping.
            phasor_cache:   Optional shared PhasorCache for cross-canvas reuse.
        """
        prev_mode = self._phasor_display_mode
        if config is not None:
            self._phasor_config = config
        if signal_metadata is not None:
            self._phasor_signal_metadata = signal_metadata
        self._phasor_display_mode = mode

        if mode in (PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS):
            self._remove_phasor_curves()
            self._update_viewport()
            return

        # Rebuild when mode switches (MAGNITUDE ↔ ANGLE) or no curves exist yet
        if prev_mode != mode or not self._phasor_curves:
            self._remove_phasor_curves()
            with timed_section(
                "phasor_overlay_rebuild",
                enabled=self._performance_timing_enabled,
                sink=self._performance_timing_sink,
            ):
                self._build_phasor_overlays(mode, self._phasor_config, phasor_cache)

        self._update_viewport()

    def _build_phasor_overlays(
        self,
        mode: PhasorDisplayMode,
        config: PhasorConfig,
        external_cache: object,
    ) -> None:
        """Compute phasor overlays for eligible channels and add curves.

        Hot-path guard: all heavy computation (extract_phasor) happens once per
        record load. Subsequent viewport updates only call setData().

        The PhasorCache is queried before computing so that mode switches reuse
        previously extracted results. Cache is keyed by (channel, window, Hz).
        """
        from app.analytics.phasors.phasor_cache import PhasorCache
        from app.analytics.phasors.phasor_extraction import (
            compute_phasor_window_samples,
            extract_phasor,
        )
        from app.analytics.phasors.phasor_models import PhasorChannelRole
        from app.analytics.phasors.phasor_overlay import classify_phasor_role

        if self._record is None or len(self._time_cache) == 0:
            return

        phasor_cache = external_cache
        if phasor_cache is None:
            if self._phasor_cache is None:
                self._phasor_cache = PhasorCache()
            phasor_cache = self._phasor_cache

        # Estimate sample rate — same logic as _build_rms_overlays
        rates = [r for r in self._record.sampling_info.sampling_rates if r > 0]
        if rates:
            sample_rate_hz = float(rates[0])
        elif len(self._time_cache) >= 2:
            diffs = np.diff(self._time_cache[: min(100, len(self._time_cache))])
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            sample_rate_hz = 1.0 / float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            sample_rate_hz = 0.0

        if sample_rate_hz <= 0:
            return

        window = compute_phasor_window_samples(sample_rate_hz, config)

        for ch in self._record.analog_channels:
            name = ch.name
            if name not in self._axis_manager._axes:
                continue

            sig_meta = self._phasor_signal_metadata.get(name)
            if not is_waveform_phasor_candidate(sig_meta):
                continue

            result = classify_phasor_role(name, ch.unit, sig_meta)
            if result.role == PhasorChannelRole.UNKNOWN:
                continue

            display_data = self._get_display_data(name)
            if display_data is None:
                continue

            # Cache lookup before compute
            cached = phasor_cache.get_phasor(  # type: ignore[union-attr]
                name, window, config.nominal_hz
            )
            if cached is not None:
                phasor_t, mag_rms, angle_deg, _ = cached
            else:
                try:
                    phasor_t, mag_rms, angle_deg, cpx = extract_phasor(
                        self._time_cache, display_data, sample_rate_hz, config
                    )
                except ValueError:
                    continue
                phasor_cache.put_phasor(  # type: ignore[union-attr]
                    name, window, config.nominal_hz,
                    (phasor_t, mag_rms, angle_deg, cpx),
                )

            if mode == PhasorDisplayMode.MAGNITUDE:
                display_arr = mag_rms
                label_suffix = " |mag|"
                pen_color = _phasor_mag_pen_color(self._axis_manager._axes[name].color)
                pen_style = Qt.PenStyle.DotLine
            else:  # ANGLE
                display_arr = angle_deg
                label_suffix = " angle°"
                pen_color = _phasor_angle_pen_color(self._axis_manager._axes[name].color)
                pen_style = Qt.PenStyle.DashDotLine

            self._phasor_time_cache[name] = phasor_t
            self._phasor_data_cache[name] = display_arr

            phasor_curve = pg.PlotDataItem(
                pen=pg.mkPen(pen_color, width=1.5, style=pen_style),
                name=phasor_curve_label(name, mode),
                skipFiniteCheck=True,
            )
            phasor_curve.setClipToView(True)

            vb = self._axis_manager._axes[name].viewbox
            if vb is self._primary_plot.getViewBox():
                self._primary_plot.addItem(phasor_curve)
            else:
                vb.addItem(phasor_curve)

            self._phasor_curves[name] = phasor_curve

    def _remove_phasor_curves(self) -> None:
        """Remove all phasor overlay curves from their ViewBoxes."""
        for phasor_curve in self._phasor_curves.values():
            try:
                vb = phasor_curve.getViewBox()
                if vb is not None:
                    vb.removeItem(phasor_curve)
            except Exception:  # noqa: BLE001
                pass
        self._phasor_curves.clear()
        self._phasor_time_cache.clear()
        self._phasor_data_cache.clear()
        self._curve_data_signatures.clear()
        # _phasor_cache preserved — reused when mode is re-enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Harmonic overlay (Phase 8)
    # ─────────────────────────────────────────────────────────────────────────

    def set_harmonic_display_mode(
        self,
        mode: HarmonicDisplayMode,
        config: HarmonicConfig | None = None,
        signal_metadata: dict | None = None,
        harmonic_cache: object = None,
    ) -> None:
        """Set harmonic analysis display mode for this canvas.

        OFF               — raw waveform only; all harmonic curves removed.
        HARMONIC_MAGNITUDE — per-order RMS magnitude envelopes overlaid on
                            raw waveforms (H3/H5/H7/H11/H13 by default).
        THD / SPECTRUM    — handled by dedicated panels in main_window; this
                            canvas removes any magnitude overlays and shows
                            raw waveforms only.

        Args:
            mode:            New harmonic display mode.
            config:          Optional HarmonicConfig override.
            signal_metadata: Optional per-channel SignalMetadata mapping.
            harmonic_cache:  Optional shared HarmonicCache for cross-canvas reuse.
        """
        prev_mode = self._harmonic_display_mode
        if config is not None:
            self._harmonic_config = config
        if signal_metadata is not None:
            self._harmonic_signal_metadata = signal_metadata
        self._harmonic_display_mode = mode

        if mode != HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            self._remove_harmonic_curves()
            self._update_viewport()
            return

        # HARMONIC_MAGNITUDE: rebuild when mode switches or no curves yet
        if prev_mode != mode or not self._harmonic_curves:
            self._remove_harmonic_curves()
            with timed_section(
                "harmonic_overlay_rebuild",
                enabled=self._performance_timing_enabled,
                sink=self._performance_timing_sink,
            ):
                self._build_harmonic_overlays(self._harmonic_config, harmonic_cache)

        self._update_viewport()

    def _build_harmonic_overlays(
        self,
        config: HarmonicConfig,
        external_cache: object,
    ) -> None:
        """Compute harmonic overlays for eligible channels and add curves.

        Hot-path guard: all heavy FFT computation happens once per record load.
        Subsequent viewport updates only call setData().  HarmonicCache is
        queried first so that mode switches reuse previously extracted results.
        """
        from app.analytics.harmonics.harmonic_cache import HarmonicCache
        from app.analytics.harmonics.harmonic_extraction import (
            compute_harmonic_window_samples,
            extract_harmonics,
        )
        from app.analytics.harmonics.harmonic_models import HarmonicChannelRole
        from app.analytics.harmonics.harmonic_overlay import classify_harmonic_role
        from app.visualization.overlays.overlay_colors import harmonic_order_pen

        if self._record is None or len(self._time_cache) == 0:
            return

        harmonic_cache = external_cache
        if harmonic_cache is None:
            if self._harmonic_cache is None:
                self._harmonic_cache = HarmonicCache()
            harmonic_cache = self._harmonic_cache

        # Estimate sample rate — same logic as _build_rms_overlays / _build_phasor_overlays
        rates = [r for r in self._record.sampling_info.sampling_rates if r > 0]
        if rates:
            sample_rate_hz = float(rates[0])
        elif len(self._time_cache) >= 2:
            diffs = np.diff(self._time_cache[: min(100, len(self._time_cache))])
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            sample_rate_hz = 1.0 / float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            sample_rate_hz = 0.0

        if sample_rate_hz <= 0:
            return

        window = compute_harmonic_window_samples(sample_rate_hz, config)
        overlap_clamped = max(0.0, min(config.overlap, 0.999))
        hop = max(1, int(round(window * (1.0 - overlap_clamped))))

        for ch in self._record.analog_channels:
            name = ch.name
            if name not in self._axis_manager._axes:
                continue

            sig_meta = self._harmonic_signal_metadata.get(name)
            role = classify_harmonic_role(name, ch.unit, sig_meta).role
            if role == HarmonicChannelRole.UNKNOWN:
                continue

            display_data = self._get_display_data(name)
            if display_data is None:
                continue

            # Cache lookup before compute
            cached = harmonic_cache.get(  # type: ignore[union-attr]
                name, window, hop, config.nominal_hz, config.max_order
            )
            if cached is not None:
                h_result = cached
            else:
                try:
                    h_result = extract_harmonics(
                        display_data, sample_rate_hz, config, time=self._time_cache
                    )
                except Exception:  # noqa: BLE001
                    continue
                harmonic_cache.put(  # type: ignore[union-attr]
                    name, window, hop, config.nominal_hz, config.max_order, h_result
                )

            if h_result.n_windows == 0:
                continue

            self._harmonic_time_cache[name] = h_result.harmonic_time
            self._harmonic_data_cache[name] = {}
            self._harmonic_curves[name] = {}

            vb = self._axis_manager._axes[name].viewbox

            for order in self._harmonic_display_orders:
                mag_arr = h_result.get_magnitude(order)
                if mag_arr is None:
                    continue

                self._harmonic_data_cache[name][order] = mag_arr

                harmonic_curve = pg.PlotDataItem(
                    pen=harmonic_order_pen(order),
                    name=f"{name} H{order}",
                    skipFiniteCheck=True,
                )
                harmonic_curve.setClipToView(True)

                if vb is self._primary_plot.getViewBox():
                    self._primary_plot.addItem(harmonic_curve)
                else:
                    vb.addItem(harmonic_curve)

                self._harmonic_curves[name][order] = harmonic_curve

    def _remove_harmonic_curves(self) -> None:
        """Remove all harmonic overlay curves from their ViewBoxes."""
        for order_curves in self._harmonic_curves.values():
            for curve in order_curves.values():
                try:
                    vb = curve.getViewBox()
                    if vb is not None:
                        vb.removeItem(curve)
                except Exception:  # noqa: BLE001
                    pass
        self._harmonic_curves.clear()
        self._harmonic_time_cache.clear()
        self._harmonic_data_cache.clear()
        self._curve_data_signatures.clear()
        # _harmonic_cache intentionally preserved — reused when mode is re-enabled

    def _refresh_panel_title(self) -> None:
        """Apply base panel title plus a lightweight RMS mode suffix."""
        self._panel_title = format_rms_mode_title(
            self._panel_base_title,
            self._rms_display_mode.value,
        )
        self._primary_plot.setTitle(self._panel_title)
