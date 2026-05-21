"""SessionCanvasController — layout manager for the Phase 9D/9E session canvas.

Responsibilities
----------------
- Own one SessionCanvasWidget per visible session panel.
- Build/rebuild the QSplitter central widget from session.list_panels().
- Refresh curve data via EventAnalysisSession.build_aligned_data().
- React incrementally to offset changes, visibility changes, source removal.
- Drive per-panel ChannelLegendWidget (Phase 9E).
- Handle panel merge/split operations via canvas header context menu (Phase 9E).

Colour strategy (Phase 9E)
--------------------------
- Phase-detected channels (VA/VB/VC, IA/IB/IC, …) receive a canonical phase
  base colour (A=blue, B=yellow, C=red) modified by a per-source saturation/hue
  variant via generate_source_variant_colour().
- Non-phase channels receive a flat source-palette colour.
- Zero-line markers always use the flat source-palette colour.
- User overrides (SessionChannel.color_hex) always win.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QWidget

from app.visualization.widgets.session_canvas import SessionCanvasWidget
from app.ui.session.legend_widget import generate_source_variant_colour

# Flat source palette (matplotlib default) — used for zero lines and non-phase channels
_SOURCE_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]

# Canonical base colours for A/B/C phases
_PHASE_BASE_COLOURS: dict[str, str] = {
    "A": "#1f77b4",   # blue
    "B": "#e6c030",   # yellow
    "C": "#d62728",   # red
}


def _session_window(session) -> tuple[float, float]:
    """Compute the global display window for all active, offset-shifted sources."""
    t_min = float("inf")
    t_max = float("-inf")
    for source in session.list_sources():
        if not source.is_active:
            continue
        time_col = source.record.waveform_data.get("time")
        if time_col is None or len(time_col) == 0:
            continue
        try:
            arr = np.asarray(time_col, dtype=np.float64)
            t_min = min(t_min, float(arr[0]) + source.time_offset_s)
            t_max = max(t_max, float(arr[-1]) + source.time_offset_s)
        except (IndexError, ValueError):
            continue
    if not np.isfinite(t_min) or t_min >= t_max:
        return -1.0, 1.0
    span = t_max - t_min
    margin = max(span * 0.02, 0.001)
    return t_min - margin, t_max + margin


_RIGHT_AXIS_TYPES = frozenset({"current", "mw", "mvar"})
_RIGHT_AXIS_KEYWORDS = frozenset({"_i", "ia", "ib", "ic", "in_", "3i0", "current", "mw", "mvar", "kw", "kvar"})


def _auto_y_axis_side(channel_name: str, ch) -> str:
    """Return 'left' or 'right' for dedicated-axis auto-assignment.

    Channels identified as current or power go to the right axis; voltage,
    frequency, and other types stay on the left.  Matches the grouping logic
    used by FlexiblePlotCanvas when AxisDisplayMode.DEDICATED is active.
    """
    param_type = getattr(ch, "parameter_type", None) or ""
    if param_type in _RIGHT_AXIS_TYPES:
        return "right"
    name_lower = channel_name.lower()
    if any(kw in name_lower for kw in _RIGHT_AXIS_KEYWORDS):
        return "right"
    return "left"


class SessionCanvasController:
    """Manage SessionCanvasWidgets for one EventAnalysisSession."""

    def __init__(self) -> None:
        self._canvases: dict[str, SessionCanvasWidget] = {}
        self._source_color_idx: dict[str, int] = {}
        self._next_color: int = 0
        self._current_panel_ids: list[str] = []
        self._splitter: QSplitter | None = None
        self._session_ref: object = None
        self._legend_visible: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def rebuild_layout(self, session) -> QWidget:
        """Return a QSplitter containing one SessionCanvasWidget per panel."""
        self._session_ref = session
        visible_panels = [p for p in session.list_panels() if p.is_visible]
        new_ids = [p.panel_id for p in visible_panels]

        dead_ids = [pid for pid, c in self._canvases.items() if sip.isdeleted(c)]
        for pid in dead_ids:
            del self._canvases[pid]

        stale_ids = [pid for pid in list(self._canvases) if pid not in set(new_ids)]
        for pid in stale_ids:
            canvas = self._canvases.pop(pid)
            if not sip.isdeleted(canvas):
                canvas.clear_all()
                canvas.deleteLater()

        for canvas in self._canvases.values():
            if not sip.isdeleted(canvas):
                canvas.setParent(None)

        splitter = QSplitter(Qt.Orientation.Vertical)
        sources_by_id = {s.source_id: s for s in session.list_sources()}

        for i, panel in enumerate(visible_panels):
            if panel.panel_id not in self._canvases:
                canvas = SessionCanvasWidget(panel.panel_id, panel.title)
                self._canvases[panel.panel_id] = canvas
                self._wire_canvas(canvas, sources_by_id)
            else:
                canvas = self._canvases[panel.panel_id]
                canvas.set_panel_title(panel.title)
            canvas.setMinimumHeight(160)
            canvas.set_legend_visible(self._legend_visible)
            splitter.addWidget(canvas)
            splitter.setStretchFactor(i, 1)

        self._current_panel_ids = new_ids
        self._splitter = splitter
        self._update_mergeable_panels()
        self._apply_time_reference(session)
        return splitter

    def active_canvases(self) -> list[SessionCanvasWidget]:
        return [self._canvases[pid] for pid in self._current_panel_ids if pid in self._canvases]

    # ─────────────────────────────────────────────────────────────────────────
    # Full refresh
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_all(self, session) -> None:
        self._session_ref = session
        t_start, t_end = _session_window(session)
        panels = {p.panel_id: p for p in session.list_panels()}
        all_channels = self._channel_lookup(session)
        active_sids = {s.source_id for s in session.list_sources() if s.is_active}
        sources_by_id = {s.source_id: s for s in session.list_sources()}

        for panel_id, canvas in self._canvases.items():
            panel = panels.get(panel_id)
            if panel is None:
                continue
            self._paint_panel(
                canvas, panel, session, all_channels, active_sids,
                sources_by_id, t_start, t_end,
            )

        self._apply_time_reference(session)
        self._refresh_zero_lines(session)

    def refresh_panel(self, panel_id: str, session) -> None:
        self._session_ref = session
        canvas = self._canvases.get(panel_id)
        panels = {p.panel_id: p for p in session.list_panels()}
        panel = panels.get(panel_id)
        if canvas is None or panel is None:
            return
        t_start, t_end = _session_window(session)
        all_channels = self._channel_lookup(session)
        active_sids = {s.source_id for s in session.list_sources() if s.is_active}
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        self._paint_panel(
            canvas, panel, session, all_channels, active_sids,
            sources_by_id, t_start, t_end,
        )
        self._refresh_zero_lines(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental updates
    # ─────────────────────────────────────────────────────────────────────────

    def on_offset_changed(self, source_id: str, offset_s: float, session) -> None:
        self._session_ref = session
        t_start, t_end = _session_window(session)
        panels_by_id = {p.panel_id: p for p in session.list_panels()}
        all_channels = self._channel_lookup(session)
        sources_by_id = {s.source_id: s for s in session.list_sources()}

        for panel_id, canvas in self._canvases.items():
            panel = panels_by_id.get(panel_id)
            if panel is None:
                continue
            for sid, channel_name in panel.channel_refs:
                if sid != source_id:
                    continue
                ch = all_channels.get((sid, channel_name))
                if ch is not None and not ch.is_visible:
                    continue
                try:
                    aligned = session.build_aligned_data(sid, channel_name, t_start, t_end)
                    color = (ch.color_hex if (ch and ch.color_hex) else None) or self._auto_colour(sid, channel_name)
                    source = sources_by_id.get(sid)
                    badge = source.display_name if source else sid
                    canvas.update_curve(
                        sid, channel_name, aligned.time, aligned.values,
                        color=color, visible=True,
                        display_name=ch.display_name if ch else channel_name,
                        source_badge=badge,
                        unit=aligned.unit,
                    )
                except Exception:  # noqa: BLE001
                    pass

        self._refresh_zero_lines(session)

    def on_channel_visibility_changed(
        self,
        source_id: str,
        channel_name: str,
        visible: bool,
        session=None,  # noqa: ARG002
    ) -> None:
        for canvas in self._canvases.values():
            canvas.set_curve_visible(source_id, channel_name, visible)

    def on_source_removed(self, source_id: str) -> None:
        for canvas in self._canvases.values():
            canvas.remove_source(source_id)
        self._source_color_idx.pop(source_id, None)

    def on_panel_merged(self, panel_id_a: str, panel_id_b: str, session) -> None:
        self.rebuild_layout(session)
        self.refresh_all(session)

    def on_colour_changed(
        self, source_id: str, channel_name: str, color_hex: str
    ) -> None:
        effective = color_hex if color_hex else self._auto_colour(source_id, channel_name)
        for canvas in self._canvases.values():
            canvas.update_curve_pen(source_id, channel_name, effective)
            canvas.legend.update_row_colour(source_id, channel_name, effective)

    def on_display_name_changed(
        self, source_id: str, channel_name: str, name: str
    ) -> None:
        for canvas in self._canvases.values():
            canvas.legend.update_row_display_name(source_id, channel_name, name)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel merge / split  (Phase 9E — wired via canvas header context menu)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_merge_panels(self, panel_id_a: str, panel_id_b: str) -> None:
        if self._session_ref is None:
            return
        try:
            self._session_ref.merge_panels(panel_id_a, panel_id_b)
        except KeyError:
            return
        self.rebuild_layout(self._session_ref)
        self.refresh_all(self._session_ref)

    def _handle_split_by_source(self, panel_id: str) -> None:
        if self._session_ref is None:
            return
        session = self._session_ref
        panel = next((p for p in session.list_panels() if p.panel_id == panel_id), None)
        if panel is None or len(panel.channel_refs) == 0:
            return

        by_source: dict[str, list] = {}
        for ref in panel.channel_refs:
            by_source.setdefault(ref[0], []).append(ref)
        if len(by_source) <= 1:
            return

        sources_list = list(by_source.items())
        for source_id, refs in sources_list[1:]:
            source = session.get_source(source_id)
            new_title = source.display_name if source else source_id
            new_pid = session.add_panel(new_title)
            for sid, cname in refs:
                session.set_channel_panel(sid, cname, new_pid)

        self.rebuild_layout(session)
        self.refresh_all(session)

    def _handle_split_by_type(self, panel_id: str) -> None:
        if self._session_ref is None:
            return
        session = self._session_ref
        panel = next((p for p in session.list_panels() if p.panel_id == panel_id), None)
        if panel is None or len(panel.channel_refs) == 0:
            return

        from app.sessions.event_session import _infer_panel_for_channel
        _PANEL_TITLES = {
            "voltage": "Voltage", "current": "Current", "power": "Power",
            "frequency": "Frequency", "digital": "Digital", "other": "Other Analog",
        }

        by_type: dict[str, list] = {}
        for ref in panel.channel_refs:
            type_key, _ = _infer_panel_for_channel(ref[1])
            by_type.setdefault(type_key, []).append(ref)
        if len(by_type) <= 1:
            return

        type_list = list(by_type.items())
        for type_key, refs in type_list[1:]:
            new_title = _PANEL_TITLES.get(type_key, type_key.title())
            new_pid = session.add_panel(new_title)
            for sid, cname in refs:
                session.set_channel_panel(sid, cname, new_pid)

        self.rebuild_layout(session)
        self.refresh_all(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Legend
    # ─────────────────────────────────────────────────────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = visible
        for canvas in self._canvases.values():
            canvas.set_legend_visible(visible)

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronization
    # ─────────────────────────────────────────────────────────────────────────

    def register_with_sync(self, sync_manager) -> None:
        canvases = self.active_canvases()
        if not canvases:
            return
        sync_manager.clear()
        sync_manager.register_many(canvases, master_canvas=canvases[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Time axis mode (mirrors FlexiblePlotCanvas.set_time_axis_mode)
    # ─────────────────────────────────────────────────────────────────────────

    def set_time_axis_mode(self, mode, session) -> None:
        """Switch all session canvases between relative and absolute time labels.

        Matches the View → Time Axis Mode menu that controls FlexiblePlotCanvas.
        ABSOLUTE re-derives the session origin from loaded sources; RELATIVE
        clears the reference so raw elapsed-second values are shown.
        """
        from app.visualization.axis.datetime_axis import TimeDisplayMode
        m = TimeDisplayMode.coerce(mode)
        if m == TimeDisplayMode.ABSOLUTE:
            self._apply_time_reference(session)
        else:
            for canvas in self._canvases.values():
                canvas.clear_time_reference()

    # ─────────────────────────────────────────────────────────────────────────
    # Y-axis display mode (mirrors FlexiblePlotCanvas.set_axis_display_mode)
    # ─────────────────────────────────────────────────────────────────────────

    def set_axis_display_mode(self, mode, session) -> None:
        """Apply Shared or Dedicated Y-axis assignment to all session channels.

        SHARED — every channel is placed on the left (primary) axis, resetting
                 any per-channel overrides.
        DEDICATED — channels are auto-assigned left/right by signal type:
                    voltage and frequency → left; current and power → right.
                    Users can still override individual channels via the legend.

        Mirrors the View → Axis Mode menu that controls FlexiblePlotCanvas.
        """
        from app.visualization.axis_management import AxisDisplayMode
        m = AxisDisplayMode.coerce(mode)
        all_channels = self._channel_lookup(session)

        for (source_id, channel_name), ch in all_channels.items():
            if m == AxisDisplayMode.SHARED:
                side = "left"
            else:
                side = _auto_y_axis_side(channel_name, ch)

            try:
                session.set_channel_y_axis_side(source_id, channel_name, side)
            except Exception:  # noqa: BLE001
                pass
            for canvas in self._canvases.values():
                canvas.set_curve_y_axis(source_id, channel_name, side)
                canvas.legend.update_row_y_axis_side(source_id, channel_name, side)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_source_index(self, source_id: str) -> int:
        """Get or assign a stable 0-based ordinal index for this source."""
        if source_id not in self._source_color_idx:
            self._source_color_idx[source_id] = self._next_color
            self._next_color += 1
        return self._source_color_idx[source_id]

    def _source_color(self, source_id: str) -> str:
        """Flat palette colour for one source (used for zero lines)."""
        idx = self._ensure_source_index(source_id)
        return _SOURCE_COLORS[idx % len(_SOURCE_COLORS)]

    def _auto_colour(self, source_id: str, channel_name: str) -> str:
        """Phase-aware + source-variant colour for waveform curves.

        Channels with a recognised phase suffix (A/B/C) share a canonical hue
        family across all sources; different sources get progressive saturation
        and hue shift via generate_source_variant_colour().

        Non-phase channels fall back to the flat source palette colour.
        """
        from app.analytics.phasors.phasor_overlay import identify_phase

        idx = self._ensure_source_index(source_id)
        phase = identify_phase(channel_name)
        # phase.value is 'A', 'B', 'C', or 'unknown'
        base = _PHASE_BASE_COLOURS.get(phase.value)  # None for 'unknown'
        if base is not None:
            return generate_source_variant_colour(base, idx)
        return _SOURCE_COLORS[idx % len(_SOURCE_COLORS)]

    @staticmethod
    def _channel_lookup(session) -> dict[tuple[str, str], object]:
        return {
            (ch.source_id, ch.channel_name): ch
            for ch in session.list_analog_channels(active_only=False)
        }

    def _paint_panel(
        self,
        canvas: SessionCanvasWidget,
        panel,
        session,
        all_channels: dict,
        active_sids: set[str],
        sources_by_id: dict,
        t_start: float,
        t_end: float,
    ) -> None:
        for source_id, channel_name in panel.channel_refs:
            if source_id not in active_sids:
                canvas.set_curve_visible(source_id, channel_name, False)
                continue
            ch = all_channels.get((source_id, channel_name))
            visible = ch.is_visible if ch is not None else True
            color = (ch.color_hex if (ch and ch.color_hex) else None) or self._auto_colour(source_id, channel_name)
            if not visible:
                canvas.set_curve_visible(source_id, channel_name, False)
                continue
            try:
                aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                source = sources_by_id.get(source_id)
                badge = source.display_name if source else source_id
                canvas.update_curve(
                    source_id, channel_name, aligned.time, aligned.values,
                    color=color, visible=True,
                    display_name=ch.display_name if ch else channel_name,
                    source_badge=badge,
                    unit=aligned.unit,
                    line_style=ch.line_style if ch else "solid",
                    line_width=ch.line_width if ch else 1.0,
                    y_axis_side=ch.y_axis_side if ch else "left",
                )
            except Exception:  # noqa: BLE001
                pass

    def _refresh_zero_lines(self, session) -> None:
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        for canvas in self._canvases.values():
            for sid in list(canvas._zero_lines.keys()):
                source = sources_by_id.get(sid)
                if source is None or not source.is_active:
                    canvas.remove_zero_line(sid)

            for source in sources_by_id.values():
                if not source.is_active:
                    continue
                color = self._source_color(source.source_id)
                canvas.update_zero_line(
                    source.source_id,
                    source.display_name,
                    source.time_offset_s,
                    color,
                )

    def _compute_session_reference_time(self, session) -> datetime | None:
        """Return the wall-clock time that corresponds to session t=0.

        For each active source: session_origin = start_time − time_offset_s.
        All properly aligned sources converge to the same origin; we return the
        earliest one so the axis is never in the future relative to any source.
        """
        origin: datetime | None = None
        for source in session.list_sources():
            if not source.is_active:
                continue
            try:
                start = source.record.timing_info.start_time
                if start is None:
                    continue
                candidate = start - timedelta(seconds=source.time_offset_s)
                if origin is None or candidate < origin:
                    origin = candidate
            except Exception:  # noqa: BLE001
                continue
        return origin

    def _apply_time_reference(self, session) -> None:
        """Push the session wall-clock origin to all canvas axes."""
        ref = self._compute_session_reference_time(session)
        for canvas in self._canvases.values():
            if ref is not None:
                canvas.set_time_reference(ref)
            else:
                canvas.clear_time_reference()

    def _update_mergeable_panels(self) -> None:
        """After a rebuild, tell each canvas what other panels exist for the merge menu."""
        panel_list = [
            (pid, self._canvases[pid]._panel_title)
            for pid in self._current_panel_ids
            if pid in self._canvases
        ]
        for pid, canvas in self._canvases.items():
            other_panels = [(p, t) for p, t in panel_list if p != pid]
            canvas.set_mergeable_panels(other_panels)

    def _wire_canvas(self, canvas: SessionCanvasWidget, sources_by_id: dict) -> None:
        """Connect legend and panel-header signals for a newly created canvas."""
        legend = canvas.legend

        legend.colour_changed.connect(
            lambda sid, cname, color: self._handle_legend_colour(sid, cname, color)
        )
        legend.display_name_changed.connect(
            lambda sid, cname, name: self._handle_legend_name(sid, cname, name)
        )
        legend.visibility_changed.connect(
            lambda sid, cname, vis: self._handle_legend_visibility(sid, cname, vis)
        )
        legend.line_style_changed.connect(
            lambda sid, cname, style: self._handle_legend_line_style(sid, cname, style)
        )
        legend.line_width_changed.connect(
            lambda sid, cname, width: self._handle_legend_line_width(sid, cname, width)
        )
        legend.y_axis_changed.connect(
            lambda sid, cname, side: self._handle_legend_y_axis(sid, cname, side)
        )

        canvas.merge_with_requested.connect(self._handle_merge_panels)
        canvas.split_by_source_requested.connect(self._handle_split_by_source)
        canvas.split_by_type_requested.connect(self._handle_split_by_type)

    def _handle_legend_colour(
        self, source_id: str, channel_name: str, color_hex: str
    ) -> None:
        if self._session_ref is not None:
            effective = color_hex if color_hex else None
            self._session_ref.set_channel_colour(source_id, channel_name, effective)
            if not color_hex:
                color_hex = self._auto_colour(source_id, channel_name)
        for canvas in self._canvases.values():
            canvas.update_curve_pen(source_id, channel_name, color_hex)
            canvas.legend.update_row_colour(source_id, channel_name, color_hex)

    def _handle_legend_name(
        self, source_id: str, channel_name: str, name: str
    ) -> None:
        if self._session_ref is not None:
            effective_name = name if name else channel_name
            self._session_ref.set_channel_display_name(
                source_id, channel_name, effective_name
            )
        for canvas in self._canvases.values():
            canvas.legend.update_row_display_name(source_id, channel_name, name or channel_name)

    def _handle_legend_visibility(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        if self._session_ref is not None:
            self._session_ref.set_channel_visibility(source_id, channel_name, visible)
        for canvas in self._canvases.values():
            canvas.set_curve_visible(source_id, channel_name, visible)

    def _handle_legend_line_style(
        self, source_id: str, channel_name: str, style: str
    ) -> None:
        if self._session_ref is not None:
            self._session_ref.set_channel_line_style(source_id, channel_name, style)
        for canvas in self._canvases.values():
            meta = canvas._metadata.get((source_id, channel_name))
            width = meta.line_width if meta else 1.0
            canvas.update_curve_style(source_id, channel_name, style, width)

    def _handle_legend_line_width(
        self, source_id: str, channel_name: str, width: float
    ) -> None:
        if self._session_ref is not None:
            self._session_ref.set_channel_line_width(source_id, channel_name, width)
        for canvas in self._canvases.values():
            meta = canvas._metadata.get((source_id, channel_name))
            style = meta.line_style if meta else "solid"
            canvas.update_curve_style(source_id, channel_name, style, width)

    def _handle_legend_y_axis(
        self, source_id: str, channel_name: str, side: str
    ) -> None:
        if self._session_ref is not None:
            self._session_ref.set_channel_y_axis_side(source_id, channel_name, side)
        for canvas in self._canvases.values():
            canvas.set_curve_y_axis(source_id, channel_name, side)
