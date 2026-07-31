"""Visualization coordination layer — wires FlexiblePlotCanvas + DigitalEventTimeline.

Pure coordinator: owns neither widget, adds no analytics, no application state.
See VISUALIZATION_CONTRACT.md and VIEWPORT_RENDERING_POLICY §6.3, §8, §12.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from PyQt6 import sip

from app.models import AnalogChannel, DisturbanceRecord
from app.visualization.axis.datetime_axis import (
    AXIS_MODE_ABSOLUTE,
    AXIS_MODE_RELATIVE,
    TimeDisplayMode,
)
from app.visualization.managers.synchronization_manager import SynchronizationManager
from app.visualization.axis_management import AxisDisplayMode
from app.visualization.engineering_display import format_panel_title
from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline
from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

if TYPE_CHECKING:
    from app.data.multi_source_session import MultiSourceSession


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _DisplayRecord:
    record: DisturbanceRecord
    source_id: str
    offset_seconds: float
    max_sampling_rate: float


def _make_filtered_record(
    record: DisturbanceRecord,
    channel_names: list[str],
) -> DisturbanceRecord:
    """Return a DisturbanceRecord containing only the specified analog channels.

    The waveform_data DataFrame is sliced to [time, *channel_names].
    Channel indices are recontiguized starting from 0.
    All other record fields (metadata, timing, sampling) are shared by reference.
    """
    names_set = set(channel_names)
    raw_channels = [ch for ch in record.analog_channels if ch.name in names_set]
    # Keep only names that actually exist in the DataFrame
    valid_names = [
        ch.name for ch in raw_channels if ch.name in record.waveform_data.columns
    ]
    reindexed = [
        AnalogChannel(
            name=ch.name,
            unit=ch.unit,
            index=i,
            phase=ch.phase,
            description=ch.description,
            scale=ch.scale,
            offset=ch.offset,
            primary_ratio=ch.primary_ratio,
            secondary_ratio=ch.secondary_ratio,
        )
        for i, ch in enumerate(raw_channels)
        if ch.name in valid_names
    ]
    cols = ["time"] + valid_names
    cols_present = [c for c in cols if c in record.waveform_data.columns]
    filtered_df = record.waveform_data.loc[:, cols_present].copy()

    return DisturbanceRecord(
        metadata=record.metadata,
        waveform_data=filtered_df,
        analog_channels=reindexed,
        digital_channels=[],
        sampling_info=record.sampling_info,
        timing_info=record.timing_info,
        disturbance_info=record.disturbance_info,
    )


def _widget_alive(widget: object) -> bool:
    """Return False only for Qt wrappers whose underlying C++ object is gone."""
    if widget is None:
        return False
    try:
        return not sip.isdeleted(widget)
    except TypeError:
        return True


def _apply_time_offset(
    record: DisturbanceRecord,
    offset_seconds: float,
) -> DisturbanceRecord:
    """Return a display copy of record with the time column shifted by offset_seconds.

    The original record is never mutated. If offset is 0.0 the original is
    returned unchanged (no copy needed for a no-op).
    """
    if offset_seconds == 0.0:
        return record
    new_df = record.waveform_data.copy()
    if "time" in new_df.columns:
        new_df["time"] = new_df["time"] + offset_seconds
    return DisturbanceRecord(
        metadata=record.metadata,
        waveform_data=new_df,
        analog_channels=record.analog_channels,
        digital_channels=record.digital_channels,
        sampling_info=record.sampling_info,
        timing_info=record.timing_info,
        disturbance_info=record.disturbance_info,
    )


def _time_extent(record: DisturbanceRecord) -> tuple[float, float] | None:
    if "time" not in record.waveform_data.columns or record.waveform_data.empty:
        return None
    t = record.waveform_data["time"].to_numpy(dtype=np.float64)
    if len(t) == 0:
        return None
    return float(np.nanmin(t)), float(np.nanmax(t))


def _trigger_display_time(
    record: DisturbanceRecord,
    offset_seconds: float,
) -> float | None:
    try:
        local_trigger_s = (
            record.timing_info.trigger_time - record.timing_info.start_time
        ).total_seconds()
    except Exception:  # noqa: BLE001
        return None
    return float(offset_seconds) + float(local_trigger_s)


def _nearest_sample_bounds(
    record: DisturbanceRecord,
    center_s: float,
) -> tuple[float, float] | None:
    if "time" not in record.waveform_data.columns or record.waveform_data.empty:
        return None
    t = record.waveform_data["time"].to_numpy(dtype=np.float64)
    if len(t) == 0:
        return None
    idx = int(np.searchsorted(t, center_s, side="left"))
    lo_idx = max(0, min(idx - 1, len(t) - 1))
    hi_idx = max(0, min(idx, len(t) - 1))
    return float(t[lo_idx]), float(t[hi_idx])


def _padded_range(t_start: float, t_end: float) -> tuple[float, float]:
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    width = t_end - t_start
    if width <= 0:
        width = max(abs(t_start) * 0.01, 1.0)
    pad = max(width * 0.05, 0.001)
    return t_start - pad, t_end + pad


def _select_initial_multi_source_viewport(
    display_records: list[_DisplayRecord],
) -> tuple[float, float] | None:
    """Choose an initial linked X range for aligned multi-source panels."""
    extents: list[tuple[float, float]] = []
    event_candidates: list[tuple[float, float, tuple[float, float]]] = []

    for item in display_records:
        extent = _time_extent(item.record)
        if extent is None:
            continue
        extents.append(extent)
        trigger_s = _trigger_display_time(item.record, item.offset_seconds)
        if trigger_s is not None and extent[0] <= trigger_s <= extent[1]:
            event_candidates.append((item.max_sampling_rate, trigger_s, extent))

    if event_candidates:
        _, event_s, event_extent = max(event_candidates, key=lambda x: x[0])
        t_start, t_end = event_extent
        for item in display_records:
            extent = _time_extent(item.record)
            if extent is None or not (extent[0] <= event_s <= extent[1]):
                continue
            bounds = _nearest_sample_bounds(item.record, event_s)
            if bounds is None:
                continue
            t_start = min(t_start, bounds[0])
            t_end = max(t_end, bounds[1])
        return _padded_range(t_start, t_end)

    if not extents:
        return None

    return _padded_range(
        min(start for start, _ in extents),
        max(end for _, end in extents),
    )


class VisualizationManager:
    """DEPRECATED — retained for legacy tests only.

    The active display path is now EventAnalysisSession + SessionCanvasController.
    This class is no longer wired into PowerwaveMainWindow.

    Coordinates FlexiblePlotCanvas (analog) and DigitalEventTimeline (digital).

    Responsibilities:
      - Coordinated set_record() / clear() / zoom_to_trigger() / reset_viewport()
      - Shared X-axis linking (DigitalEventTimeline.link_x_to FlexiblePlotCanvas)
      - Bidirectional master cursor synchronization (loop-free via blockSignals)

    NOT responsible for:
      - File loading or provider orchestration
      - Analytics computation or overlay management
      - Application-wide state, singletons, or event buses
      - UI docking, layout, or measurement panels

    Lifetime contract:
      This instance MUST be kept alive by the caller while cursor sync is needed.
      pyqtSignal stores a weak reference to bound methods; garbage-collecting this
      object silently drops cursor_moved connections.
    """

    def __init__(
        self,
        canvas: FlexiblePlotCanvas,
        timeline: DigitalEventTimeline,
    ) -> None:
        self._canvas = canvas
        self._timeline = timeline
        self._record: DisturbanceRecord | None = None
        self._x_linked: bool = False
        self._panel_canvases: dict = {}
        self._sequence_panels_visible = False
        self._sequence_panel_data: dict = {}
        self._harmonic_panels_visible = False
        self._harmonic_panel_data: dict = {}
        self._sync_manager = SynchronizationManager()
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._axis_reference_time: datetime | None = None
        self._axis_display_mode = AxisDisplayMode.SHARED
        self._crosshair_snap_enabled = False
        self._info_box_visible = False
        self._canvas_theme = "dark"

        # Bidirectional cursor sync — receivers use blockSignals to prevent loops
        canvas.cursor_moved.connect(self._on_canvas_cursor_moved)
        timeline.cursor_moved.connect(self._on_timeline_cursor_moved)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def canvas(self) -> FlexiblePlotCanvas:
        return self._canvas

    @property
    def timeline(self) -> DigitalEventTimeline:
        return self._timeline

    @property
    def record(self) -> DisturbanceRecord | None:
        return self._record

    @property
    def is_x_linked(self) -> bool:
        return self._x_linked

    @property
    def panel_canvases(self) -> dict:
        """Shallow copy of the display-group → canvas mapping from the last
        display_grouped_record() call. Empty until that method is called."""
        return dict(self._panel_canvases)

    @property
    def synchronization_manager(self) -> SynchronizationManager:
        return self._sync_manager

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def link_x_axis(self) -> None:
        """Link DigitalEventTimeline X-axis to FlexiblePlotCanvas.

        MUST be called after both widgets are added to a visible Qt scene.
        setXLink() on unparented items is undefined behavior in PyQtGraph
        (VIEWPORT_RENDERING_POLICY §8).

        After linking, canvas zoom/pan propagates to the timeline automatically —
        zoom_to_trigger() and reset_viewport() need only act on the canvas.
        """
        # _primary_plot access is by design — see DigitalEventTimeline.link_x_to() docstring
        self.register_synchronized_panels([self._canvas], timeline=self._timeline)
        self._x_linked = True

    def set_record(
        self,
        record: DisturbanceRecord,
        *,
        axis_mode: str = "relative_seconds",
    ) -> None:
        """Load a DisturbanceRecord into both canvas and timeline."""
        self._record = record
        self._time_axis_mode = TimeDisplayMode.coerce(axis_mode)
        self._axis_reference_time = record.timing_info.start_time
        self._canvas.set_record(record, axis_mode=axis_mode)
        if self._time_axis_mode == TimeDisplayMode.ABSOLUTE:
            self._timeline.set_record(
                record,
                axis_mode=axis_mode,
                axis_reference_time=self._axis_reference_time,
            )
        else:
            self._timeline.set_record(record)
        if self._x_linked:
            self.register_synchronized_panels([self._canvas], timeline=self._timeline)

    def set_time_axis_mode(
        self,
        mode: TimeDisplayMode | str,
        *,
        axis_reference_time: datetime | None = None,
    ) -> None:
        """Switch all currently managed axes between relative and absolute labels."""
        display_mode = TimeDisplayMode.coerce(mode)
        if axis_reference_time is not None:
            self._axis_reference_time = axis_reference_time
        elif self._axis_reference_time is None and self._record is not None:
            self._axis_reference_time = self._record.timing_info.start_time
        self._time_axis_mode = display_mode

        reference = self._axis_reference_time
        targets = list(self._panel_canvases.values()) or [self._canvas]
        for canvas in targets:
            if _widget_alive(canvas) and hasattr(canvas, "set_time_axis_mode"):
                canvas.set_time_axis_mode(display_mode, axis_reference_time=reference)
        if _widget_alive(self._timeline) and hasattr(self._timeline, "set_time_axis_mode"):
            self._timeline.set_time_axis_mode(display_mode, axis_reference_time=reference)

    def set_axis_display_mode(self, mode: AxisDisplayMode | str) -> None:
        """Switch all managed analog canvases between shared/dedicated axes."""
        axis_mode = AxisDisplayMode.coerce(mode)
        self._axis_display_mode = axis_mode
        targets = list(self._panel_canvases.values()) or [self._canvas]
        for canvas in targets:
            if _widget_alive(canvas) and hasattr(canvas, "set_axis_display_mode"):
                canvas.set_axis_display_mode(axis_mode)

    def set_crosshair_snap_enabled(self, enabled: bool) -> None:
        """Switch managed analog canvases between free and waveform-snapped crosshair."""
        self._crosshair_snap_enabled = bool(enabled)
        targets = list(self._panel_canvases.values()) or [self._canvas]
        for canvas in targets:
            if _widget_alive(canvas) and hasattr(canvas, "set_crosshair_snap_enabled"):
                canvas.set_crosshair_snap_enabled(self._crosshair_snap_enabled)

    def set_info_box_visible(self, visible: bool) -> None:
        """Show or hide floating waveform information boxes on managed canvases."""
        self._info_box_visible = bool(visible)
        targets = list(self._panel_canvases.values()) or [self._canvas]
        for canvas in targets:
            if _widget_alive(canvas) and hasattr(canvas, "set_info_box_visible"):
                canvas.set_info_box_visible(self._info_box_visible)

    def set_canvas_theme(self, theme: str) -> None:
        """Apply the canvas theme to all managed waveform canvases."""
        self._canvas_theme = "light" if str(theme).lower() == "light" else "dark"
        targets = list(self._panel_canvases.values()) or [self._canvas]
        for canvas in targets:
            if _widget_alive(canvas) and hasattr(canvas, "set_canvas_theme"):
                canvas.set_canvas_theme(self._canvas_theme)
        if _widget_alive(self._timeline) and hasattr(self._timeline, "set_canvas_theme"):
            self._timeline.set_canvas_theme(self._canvas_theme)

    def clear(self) -> None:
        """Clear both widgets and discard the record reference."""
        self._record = None
        self._axis_reference_time = None
        self._time_axis_mode = TimeDisplayMode.RELATIVE
        self._harmonic_panels_visible = False
        self._harmonic_panel_data = {}
        self._sync_manager.clear()
        self._canvas._clear_canvas()
        self._timeline.clear()

    def zoom_to_trigger(self, window_s: float = 0.2) -> None:
        """Zoom both widgets to centre on the trigger event ± window_s seconds.

        When X-linked, canvas zoom propagates to the timeline for free.
        When not linked, the timeline is zoomed independently.
        """
        self._canvas.zoom_to_trigger(window_s)
        if not self._x_linked:
            self._zoom_timeline_to_trigger(window_s)

    def reset_viewport(self) -> None:
        """Reset the viewport to show the full time range of the loaded record.

        When X-linked, canvas reset propagates to the timeline for free.
        When not linked, both widgets are reset independently.
        """
        if self._record is None:
            return
        time_col = self._record.waveform_data["time"]
        t_max = float(time_col.iloc[-1]) if len(time_col) > 0 else 1.0
        self._canvas._primary_plot.setXRange(0.0, t_max, padding=0)
        if not self._x_linked:
            self._timeline.getPlotItem().setXRange(0.0, t_max, padding=0)

    def set_cursor_pos(self, t: float) -> None:
        """Move master cursor on both widgets without emitting cursor_moved."""
        self._canvas.set_cursor_pos(t)
        self._timeline.set_cursor_pos(t)

    def show_sequence_panels(self, *args, **kwargs) -> None:
        """Phase 6B hook: mark sequence component panels as requested.

        This method intentionally does not create panels or render sequence
        data yet. It provides a stable routing point for the future phasor UI.
        """
        self._sequence_panels_visible = True

    def hide_sequence_panels(self) -> None:
        """Phase 6B hook: mark sequence component panels as hidden."""
        self._sequence_panels_visible = False

    def set_sequence_panel_data(self, *args, **kwargs) -> None:
        """Phase 6B hook: accept future sequence panel payloads without rendering."""
        self._sequence_panel_data = {
            "args": args,
            "kwargs": kwargs,
        }

    def show_harmonic_panels(self, *args, **kwargs) -> None:
        """Phase 8 hook: mark harmonic panels as requested without rendering."""
        self._harmonic_panels_visible = True
        self._harmonic_panel_data = {
            "args": args,
            "kwargs": kwargs,
        }

    def hide_harmonic_panels(self) -> None:
        """Phase 8 hook: mark harmonic panels hidden without altering waveforms."""
        self._harmonic_panels_visible = False

    def set_harmonic_panel_data(self, *args, **kwargs) -> None:
        """Phase 8 hook: accept future harmonic panel payloads without rendering."""
        self._harmonic_panel_data = {
            "args": args,
            "kwargs": kwargs,
        }

    def register_synchronized_panels(
        self,
        canvases,
        *,
        timeline: DigitalEventTimeline | None = None,
    ) -> None:
        """Register analog panel canvases plus optional digital timeline for sync."""
        canvases = list(canvases)
        self._sync_manager.clear()
        if not canvases:
            return
        master = canvases[0]
        self._sync_manager.register_many(canvases, master_canvas=master)
        if timeline is not None:
            self._sync_manager.register_canvas(timeline)
        x_range = master._primary_plot.getViewBox().viewRange()[0]
        if len(x_range) == 2:
            t_start, t_end = x_range
            self._sync_manager.synchronize_x_range(master, (t_start, t_end))
        cursor = getattr(master, "_cursor", None)
        if cursor is not None:
            self._sync_manager.synchronize_cursor(master, float(cursor.value()))

    def display_grouped_record(
        self,
        record: DisturbanceRecord,
        signal_metadata: dict | None = None,
        canvas_factory=None,
        *,
        axis_mode: str = AXIS_MODE_RELATIVE,
    ) -> dict:
        """Group channels by display group and create one canvas per non-empty group.

        Each panel canvas receives a filtered DisturbanceRecord containing only
        its group's analog channels, so viewport updates never bleed across panels.

        Digital channels are routed to the existing timeline widget, not a canvas.

        Args:
            record:          The DisturbanceRecord to display.
            signal_metadata: Optional dict of channel_name → SignalMetadata for
                             metadata-driven group assignment.
            canvas_factory:  Callable() → FlexiblePlotCanvas.  Defaults to
                             FlexiblePlotCanvas (lazy import).  Pass a mock in
                             tests to avoid needing a QApplication.

        Returns:
            dict mapping display_group_name → FlexiblePlotCanvas.
            Caller is responsible for adding these widgets to a layout.
        """
        from app.visualization.channel_grouper import (
            DISPLAY_GROUP_DIGITAL,
            group_channels_for_display,
        )

        if canvas_factory is None:
            canvas_factory = FlexiblePlotCanvas

        groups = group_channels_for_display(record, signal_metadata)
        panel_canvases: dict = {}
        display_mode = TimeDisplayMode.coerce(axis_mode)
        self._time_axis_mode = display_mode
        self._axis_reference_time = record.timing_info.start_time

        for group_name, channel_names in groups.items():
            if group_name == DISPLAY_GROUP_DIGITAL:
                if channel_names:
                    if display_mode == TimeDisplayMode.ABSOLUTE:
                        self._timeline.set_record(
                            record,
                            axis_mode=axis_mode,
                            axis_reference_time=self._axis_reference_time,
                        )
                    else:
                        self._timeline.set_record(record)
                continue
            if not channel_names:
                continue
            filtered = _make_filtered_record(record, channel_names)
            canvas = canvas_factory()
            if display_mode == TimeDisplayMode.ABSOLUTE:
                canvas.set_record(
                    filtered,
                    axis_mode=axis_mode,
                    axis_reference_time=self._axis_reference_time,
                )
            else:
                canvas.set_record(filtered, axis_mode=axis_mode)
            if hasattr(canvas, "set_axis_display_mode"):
                canvas.set_axis_display_mode(self._axis_display_mode)
            if hasattr(canvas, "set_crosshair_snap_enabled"):
                canvas.set_crosshair_snap_enabled(self._crosshair_snap_enabled)
            if hasattr(canvas, "set_info_box_visible"):
                canvas.set_info_box_visible(self._info_box_visible)
            if hasattr(canvas, "set_canvas_theme"):
                canvas.set_canvas_theme(self._canvas_theme)
            if hasattr(canvas, "set_panel_title"):
                canvas.set_panel_title(format_panel_title(group_name, len(channel_names)))
            panel_canvases[group_name] = canvas

        self._panel_canvases = panel_canvases
        self._record = record
        return panel_canvases

    def display_multi_source_session(
        self,
        session: MultiSourceSession,
        canvas_factory=None,
        *,
        axis_mode: str = AXIS_MODE_ABSOLUTE,
    ) -> dict:
        """Display a MultiSourceSession as time-aligned stacked panels.

        Each source contributes one canvas per non-empty display group.
        Panel keys are ``"{source_id}/{group_name}"`` to prevent collision
        when multiple sources share a group name.

        Time axes are aligned to the earliest source start time; the original
        records inside the session are never modified.

        Args:
            session:        MultiSourceSession with one or more SourceRecords.
            canvas_factory: Callable() → FlexiblePlotCanvas. Defaults to
                            FlexiblePlotCanvas (lazy import). Pass a mock in
                            tests to avoid needing QApplication.

        Returns:
            dict mapping ``"{source_id}/{group_name}"`` → canvas instance.
            Caller is responsible for adding these widgets to a layout.
        """
        from app.data.display_alignment import (
            compute_relative_offsets,
            determine_reference_start,
        )
        from app.visualization.channel_grouper import (
            DISPLAY_GROUP_DIGITAL,
            group_channels_for_display,
        )

        if canvas_factory is None:
            canvas_factory = FlexiblePlotCanvas

        reference_start = determine_reference_start(session.sources)
        offsets = compute_relative_offsets(session.sources, reference_start)
        display_mode = TimeDisplayMode.coerce(axis_mode)
        self._time_axis_mode = display_mode
        self._axis_reference_time = reference_start

        panel_canvases: dict = {}
        display_records: list[_DisplayRecord] = []

        for source in session.sources:
            offset = offsets.get(source.source_id, 0.0)
            max_sampling_rate = max(source.sampling_rates or [], default=0.0)
            groups = group_channels_for_display(source.record, source.signal_metadata)

            for group_name, channel_names in groups.items():
                if group_name == DISPLAY_GROUP_DIGITAL:
                    if channel_names:
                        if display_mode == TimeDisplayMode.ABSOLUTE:
                            self._timeline.set_record(
                                source.record,
                                axis_mode=axis_mode,
                                axis_reference_time=reference_start,
                            )
                        else:
                            self._timeline.set_record(source.record)
                    continue
                if not channel_names:
                    continue
                filtered = _make_filtered_record(source.record, channel_names)
                display_record = _apply_time_offset(filtered, offset)
                canvas = canvas_factory()
                if display_mode == TimeDisplayMode.ABSOLUTE:
                    canvas.set_record(
                        display_record,
                        axis_mode=axis_mode,
                        axis_reference_time=reference_start,
                    )
                else:
                    canvas.set_record(display_record, axis_mode=axis_mode)
                if hasattr(canvas, "set_axis_display_mode"):
                    canvas.set_axis_display_mode(self._axis_display_mode)
                if hasattr(canvas, "set_crosshair_snap_enabled"):
                    canvas.set_crosshair_snap_enabled(self._crosshair_snap_enabled)
                if hasattr(canvas, "set_info_box_visible"):
                    canvas.set_info_box_visible(self._info_box_visible)
                if hasattr(canvas, "set_canvas_theme"):
                    canvas.set_canvas_theme(self._canvas_theme)
                panel_key = f"{source.source_id}/{group_name}"
                if hasattr(canvas, "set_panel_title"):
                    canvas.set_panel_title(
                        format_panel_title(group_name, len(channel_names), source.source_id)
                    )
                panel_canvases[panel_key] = canvas
                display_records.append(
                    _DisplayRecord(
                        record=display_record,
                        source_id=source.source_id,
                        offset_seconds=offset,
                        max_sampling_rate=max_sampling_rate,
                    )
                )

        self._panel_canvases = panel_canvases
        if session.sources:
            self._record = session.sources[0].record
        initial_viewport = _select_initial_multi_source_viewport(display_records)
        if initial_viewport is not None:
            t_start, t_end = initial_viewport
            for canvas in panel_canvases.values():
                canvas._primary_plot.setXRange(t_start, t_end, padding=0)
        return panel_canvases

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_canvas_cursor_moved(self, t: float) -> None:
        self._timeline.set_cursor_pos(t)

    def _on_timeline_cursor_moved(self, t: float) -> None:
        self._canvas.set_cursor_pos(t)

    def _zoom_timeline_to_trigger(self, window_s: float) -> None:
        """Independently zoom the timeline when not X-linked."""
        if self._record is None:
            return
        t_trig = (
            self._record.timing_info.trigger_time
            - self._record.timing_info.start_time
        ).total_seconds()
        time_col = self._record.waveform_data["time"]
        t_max = float(time_col.iloc[-1]) if len(time_col) > 0 else 0.0
        t_start = max(0.0, t_trig - window_s)
        t_end = min(t_max, t_trig + window_s)
        self._timeline.getPlotItem().setXRange(t_start, t_end, padding=0)
