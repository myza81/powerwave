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

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QScrollArea, QSplitter, QWidget

from app.analytics.harmonics.harmonic_models import HarmonicConfig, HarmonicDisplayMode
from app.analytics.phasors.phasor_models import PhasorConfig, PhasorDisplayMode
from app.analytics.rms.rms_models import RMSConfig, RMSDisplayMode
from app.analytics.scaling.scaling_models import EngineeringScalingMode
from app.analytics.scaling.scaling_registry import ScalingRegistry
from app.calculated_signals.models import CalculationStatus
from app.visualization.widgets.session_canvas import SessionCanvasWidget
from app.ui.session.legend_widget import generate_source_variant_colour
from app.visualization.axis_management import AxisDisplayMode, axis_group_for_signal
from app.visualization.axis.datetime_axis import TimeDisplayMode
from app.visualization.viewport_policy import (
    MIN_SOURCES_FOR_EVENT_FOCUS,
    select_initial_viewport,
)

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


def _session_time_display_mode(session) -> TimeDisplayMode:
    """Infer the visualization x-axis semantics from active source records."""
    references: set[str] = set()
    for source in session.list_sources():
        if not source.is_active:
            continue
        try:
            references.add(str(getattr(source.record.timing_info, "timing_reference", "absolute")))
        except Exception:  # noqa: BLE001
            continue
    if references and references <= {"sample_index"}:
        return TimeDisplayMode.SAMPLE_INDEX
    if references and references <= {"relative_elapsed", "synthetic_elapsed"}:
        return TimeDisplayMode.RELATIVE
    return TimeDisplayMode.ABSOLUTE


# ─────────────────────────────────────────────────────────────────────────
# Calculated signal canvas identity (Phase 3B)
# ─────────────────────────────────────────────────────────────────────────
#
# SessionCanvasWidget keys every curve by a (source_id, channel_name) tuple.
# A calculated signal is never a SessionSource, so it has no real source_id
# -- but it still needs a curve key. Real source_ids are always bare
# str(uuid.uuid4()) values (EventAnalysisSession.add_source), which can
# never contain a ':' character, so prefixing calc_id with "calc:" is
# structurally guaranteed to never collide with a real source_id, for any
# calc_id. calc_id (never the user-editable display name) is the
# permanent, stable component of this key; the display name is only the
# tuple's second element, exactly as a real channel_name is -- so a future
# rename only needs to remove the old curve and repaint under the new key,
# the same way any channel display-name-driven curve identity would.

_CALC_SOURCE_PREFIX = "calc:"


def _calc_curve_key(calc_id: str, definition_name: str) -> tuple[str, str]:
    """Return the SessionCanvasWidget curve key for one calculated signal."""
    return (_CALC_SOURCE_PREFIX + calc_id, definition_name)


def _is_calc_curve_key(key: tuple[str, str]) -> bool:
    return key[0].startswith(_CALC_SOURCE_PREFIX)


_RIGHT_AXIS_TYPES = frozenset({"current", "mw", "mvar", "frequency"})
_RIGHT_AXIS_KEYWORDS = frozenset({"_i", "ia", "ib", "ic", "in_", "3i0", "current", "mw", "mvar", "kw", "kvar"})
_FREQ_KEYWORDS = frozenset({"freq", "hz", "rocof", "df_dt"})


def _auto_y_axis_side(channel_name: str, ch) -> str:
    """Return 'left' or 'right' for dedicated-axis auto-assignment.

    Channels identified as current, power, or frequency go to the right axis
    (each unit group gets its own independent ViewBox via _RightAxisManager).
    Voltage and other types stay on the left primary axis.
    """
    param_type = getattr(ch, "parameter_type", None) or ""
    if param_type in _RIGHT_AXIS_TYPES:
        return "right"
    name_lower = channel_name.lower()
    if any(kw in name_lower for kw in _RIGHT_AXIS_KEYWORDS):
        return "right"
    if any(kw in name_lower for kw in _FREQ_KEYWORDS):
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
        self._rms_mode: RMSDisplayMode = RMSDisplayMode.OFF  # S2
        self._rms_config: RMSConfig = RMSConfig()            # S2
        self._phasor_mode: PhasorDisplayMode = PhasorDisplayMode.OFF  # S3
        self._phasor_config: PhasorConfig = PhasorConfig()            # S3
        self._harmonic_mode: HarmonicDisplayMode = HarmonicDisplayMode.OFF  # S4
        self._harmonic_config: HarmonicConfig = HarmonicConfig()            # S4
        self._harmonic_orders: list[int] = [3, 5, 7, 11, 13]               # S4
        self._scaling_mode: EngineeringScalingMode = EngineeringScalingMode.RAW  # S5
        self._scaling_registry: ScalingRegistry = ScalingRegistry()              # S5
        self._measurement_mode: bool = False      # S6
        self._measurement_callback: object = None # S6
        self._cursor_sync: bool = True            # sync cursors across all panels
        self._measurement_mode_changed_cb: Callable[[bool], None] | None = None
        self._digital_panel_rows: dict[str, int] = {}   # panel_id → number of digital rows
        self._needs_digital_resize: bool = False         # trigger one resize after rebuild
        self._scroll_area: QScrollArea | None = None
        self._crosshair_readout_cb: object = None        # callable(t, values) for live readout
        self._crosshair_snap_enabled: bool = False
        self._canvas_theme: str = "dark"
        self._navigator = None                           # WaveformNavigatorStrip | None
        self._channel_panel_changed_cb: object = None   # callable(source_id, ch_name, panel_id)
        self._layout_rebuilt_callback: object = None
        # Phase 3B: calculated-signal curves, tracked separately from real
        # (source_id, channel_name) curves -- see _calc_curve_key().
        self._calc_curve_keys: dict[str, tuple[str, str]] = {}   # calc_id -> curve key
        self._calc_panel_by_id: dict[str, str] = {}              # calc_id -> panel_id
        # Stage 2 initial-viewport lifecycle. _viewport_initialized_session
        # holds the session object (strong reference, compared with `is`) whose
        # initial visible range has already been chosen, so a later source add
        # never re-snaps a range the analyst may have zoomed or panned.
        # _x_range_before_rebuild is the shared X range captured at the top of
        # rebuild_layout(), used to pull freshly created canvases into the
        # range the session is already showing.
        self._viewport_initialized_session: object | None = None
        self._x_range_before_rebuild: tuple[float, float] | None = None

    def set_layout_rebuilt_callback(self, callback) -> None:
        """Notify the owner when panel merge/split produces a new splitter."""
        self._layout_rebuilt_callback = callback

    def _notify_layout_rebuilt(self, splitter: QSplitter) -> None:
        if self._layout_rebuilt_callback is not None:
            self._layout_rebuilt_callback(splitter)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def rebuild_layout(self, session) -> QWidget:
        """Return a QScrollArea containing a QSplitter with one SessionCanvasWidget per panel."""
        self._session_ref = session
        # Stage 2: remember what the session is currently showing BEFORE any
        # canvas is added or removed, so canvases created by this rebuild can
        # be brought into that same range instead of auto-ranging to their own
        # data and dragging every other panel with them through
        # SynchronizationManager.
        self._x_range_before_rebuild = self._current_shared_x_range()
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
            canvas.setMinimumHeight(200)
            canvas.set_legend_visible(self._legend_visible)
            canvas.set_crosshair_snap_enabled(self._crosshair_snap_enabled)
            canvas.set_canvas_theme(self._canvas_theme)
            if self._measurement_mode:  # S6: propagate measurement state to (re-)created canvases
                canvas.set_measurement_mode(True)
            splitter.addWidget(canvas)
            splitter.setStretchFactor(i, 1)

        self._current_panel_ids = new_ids
        self._splitter = splitter
        self._digital_panel_rows.clear()
        self._needs_digital_resize = True
        self._update_mergeable_panels()
        self._apply_time_reference(session)
        self._refresh_zero_lines(session)
        self._refresh_trigger_markers(session)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(splitter)
        self._scroll_area = scroll
        return scroll

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
            # Remove curves that are no longer assigned to this panel.
            # Calculated-signal curves are never in panel.channel_refs (see
            # module docstring on _calc_curve_key) -- refresh_calculated_signals()
            # owns their lifecycle, so this sweep must not remove them.
            expected = {(sid, cname) for sid, cname in panel.channel_refs}
            for key in list(canvas._curves.keys()):
                if key not in expected and not _is_calc_curve_key(key):
                    canvas.remove_curve(key[0], key[1])
            self._paint_panel(
                canvas, panel, session, all_channels, active_sids,
                sources_by_id, t_start, t_end,
            )
            canvas.set_canvas_theme(self._canvas_theme)

        self._apply_time_reference(session)
        self._refresh_zero_lines(session)
        self._refresh_trigger_markers(session)
        self._populate_navigator(session)

        if self._needs_digital_resize:
            self._needs_digital_resize = False
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self._resize_digital_panels)

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
        canvas.set_canvas_theme(self._canvas_theme)
        self._refresh_zero_lines(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Calculated signals (Phase 3B)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # A small parallel path to refresh_all()/_paint_panel(): calculated
    # signals are session-owned derived curves, never a SessionSource or
    # DisturbanceRecord, so they never go through
    # EventAnalysisSession.build_aligned_data() (which clips/decimates for
    # display and requires a real source). Each CalculatedSignalResult is
    # already the correct, complete, aligned array at its own reference time
    # base -- result.time/result.values are painted directly.

    def refresh_calculated_signals(self, session, calc_ids: "Iterable[str] | None" = None) -> None:
        """Paint/update/remove calculated-signal curves.

        calc_ids=None (default) syncs every calculated signal in the
        session -- new ones are painted, existing ones are updated in
        place (no duplicate curve creation, see SessionCanvasWidget.
        update_curve()'s key-reuse), and ones that were deleted or no
        longer have a usable result are removed. Passing an explicit
        calc_ids restricts the sync to just those signals (e.g. after
        resolve_for_source() recalculates only the signals depending on one
        source) -- other calculated signals' curves are left untouched, so
        this never rebuilds every original source curve or every other
        calculated signal.

        OK and STALE results both render (a stale result's last-known-good
        arrays remain visible, labelled "(stale)"); ERROR results and
        never-calculated entries (result is None) are removed rather than
        fabricating placeholder data.
        """
        self._session_ref = session
        entries = session.list_calculated_signals()
        if calc_ids is not None:
            wanted = set(calc_ids)
            entries = [e for e in entries if e.definition.calc_id in wanted]

        live_ids: set[str] = set()
        for entry in entries:
            calc_id = entry.definition.calc_id
            result = entry.result
            if result is None or result.status == CalculationStatus.ERROR:
                self.remove_calculated_signal_curve(calc_id)
                continue

            placement = session.ensure_calculated_signal_panel(calc_id)
            if placement is None:
                continue
            panel_id, _created = placement
            canvas = self._canvases.get(panel_id)
            if canvas is None:
                # Panel exists in the session but no SessionCanvasWidget has
                # been built for it yet -- the caller must call
                # rebuild_layout() first when a brand-new panel is needed
                # (see EventAnalysisSession.ensure_calculated_signal_panel's
                # `created` flag).
                continue

            live_ids.add(calc_id)
            key = _calc_curve_key(calc_id, entry.definition.name)
            old_key = self._calc_curve_keys.get(calc_id)
            old_panel_id = self._calc_panel_by_id.get(calc_id)
            if old_key is not None and (old_key != key or old_panel_id != panel_id):
                old_canvas = self._canvases.get(old_panel_id) if old_panel_id else None
                if old_canvas is not None:
                    old_canvas.remove_curve(old_key[0], old_key[1])

            display_name = f"ƒ {entry.definition.name}"
            if result.status == CalculationStatus.STALE:
                display_name += " (stale)"

            canvas.update_curve(
                key[0], key[1], result.time, result.values,
                color=self._auto_colour(key[0], key[1]),
                visible=getattr(entry, "is_visible", True),
                display_name=display_name,
                source_badge="calc",
                unit=result.unit,
                line_style="solid",
                line_width=1.5,
                y_axis_side="left",
                signal_type=None,
            )
            self._calc_curve_keys[calc_id] = key
            self._calc_panel_by_id[calc_id] = panel_id

        if calc_ids is None:
            # Full sync: anything no longer present/usable gets removed.
            for calc_id in set(self._calc_curve_keys) - live_ids:
                self.remove_calculated_signal_curve(calc_id)

    def remove_calculated_signal_curve(self, calc_id: str) -> None:
        """Remove one calculated signal's curve from canvas, if present.

        Does not touch the session -- callers that also want to delete the
        definition/result call EventAnalysisSession.remove_calculated_signal()
        separately (see Phase 3B deletion flow).
        """
        key = self._calc_curve_keys.pop(calc_id, None)
        panel_id = self._calc_panel_by_id.pop(calc_id, None)
        if key is None or panel_id is None:
            return
        canvas = self._canvases.get(panel_id)
        if canvas is not None:
            canvas.remove_curve(key[0], key[1])

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
                    scaled_values, display_unit = self._scale_aligned(
                        channel_name, aligned.values, aligned.unit
                    )
                    color = (ch.color_hex if (ch and ch.color_hex) else None) or self._auto_colour(sid, channel_name)
                    source = sources_by_id.get(sid)
                    badge = source.display_name if source else sid
                    canvas.update_curve(
                        sid, channel_name, aligned.time, scaled_values,
                        color=color, visible=True,
                        display_name=ch.display_name if ch else channel_name,
                        source_badge=badge,
                        unit=display_unit,
                    )
                except Exception:  # noqa: BLE001
                    pass

        self._refresh_zero_lines(session)
        self._refresh_trigger_markers(session)
        if self._rms_mode != RMSDisplayMode.OFF:
            self._refresh_rms_overlays(session)
        if self._phasor_mode not in (PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS):
            self._refresh_phasor_overlays(session)
        if self._harmonic_mode == HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            self._refresh_harmonic_overlays(session)

    def on_channel_visibility_changed(
        self,
        source_id: str,
        channel_name: str,
        visible: bool,
        session=None,
    ) -> None:
        for canvas in self._canvases.values():
            canvas.set_curve_visible(source_id, channel_name, visible)
        sess = session or self._session_ref
        if sess is not None:
            self._repack_digital_panel_for_channel(source_id, channel_name, sess)

    def _repack_digital_panel_for_channel(
        self, source_id: str, channel_name: str, session
    ) -> None:
        """If the toggled channel is digital, repaint its panel to repack Y offsets."""
        digital_keys = {
            (ch.source_id, ch.channel_name)
            for ch in session.list_digital_channels(active_only=False)
        }
        if (source_id, channel_name) not in digital_keys:
            return
        all_channels = self._channel_lookup(session)
        active_sids = {s.source_id for s in session.list_sources() if s.is_active}
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        t_start, t_end = _session_window(session)
        for panel in session.list_panels():
            if (source_id, channel_name) in panel.channel_refs:
                canvas = self._canvases.get(panel.panel_id)
                if canvas is not None:
                    self._digital_panel_rows.pop(panel.panel_id, None)
                    self._paint_panel(
                        canvas, panel, session,
                        all_channels, active_sids, sources_by_id,
                        t_start, t_end,
                    )

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

    def _handle_channel_drop(
        self, source_id: str, channel_name: str, from_panel_id: str, to_panel_id: str
    ) -> None:
        """Move a channel from one panel to another via drag-and-drop."""
        if from_panel_id == to_panel_id or self._session_ref is None:
            return
        try:
            self._session_ref.set_channel_panel(source_id, channel_name, to_panel_id)
        except Exception:  # noqa: BLE001
            return
        # Notify main window to do a full rebuild (same path as "Move to panel" in legend)
        if self._channel_panel_changed_cb is not None:
            self._channel_panel_changed_cb(source_id, channel_name, to_panel_id)

    def _handle_panel_rename(self, panel_id: str, new_title: str) -> None:
        if self._session_ref is not None:
            self._session_ref.rename_panel(panel_id, new_title)

    def _handle_merge_panels(self, panel_id_a: str, panel_id_b: str) -> None:
        if self._session_ref is None:
            return
        try:
            self._session_ref.merge_panels(panel_id_a, panel_id_b)
        except KeyError:
            return
        splitter = self.rebuild_layout(self._session_ref)
        self._notify_layout_rebuilt(splitter)
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

        splitter = self.rebuild_layout(session)
        self._notify_layout_rebuilt(splitter)
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

        splitter = self.rebuild_layout(session)
        self._notify_layout_rebuilt(splitter)
        self.refresh_all(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Legend
    # ─────────────────────────────────────────────────────────────────────────

    def set_legend_visible(self, visible: bool) -> None:
        self._legend_visible = visible
        for canvas in self._canvases.values():
            canvas.set_legend_visible(visible)

    def set_crosshair_snap_enabled(self, enabled: bool) -> None:
        self._crosshair_snap_enabled = bool(enabled)
        for canvas in self._canvases.values():
            if not sip.isdeleted(canvas):
                canvas.set_crosshair_snap_enabled(self._crosshair_snap_enabled)

    def set_canvas_theme(self, theme: str) -> None:
        self._canvas_theme = "light" if str(theme).lower() == "light" else "dark"
        for canvas in self._canvases.values():
            if not sip.isdeleted(canvas):
                canvas.set_canvas_theme(self._canvas_theme)

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronization
    # ─────────────────────────────────────────────────────────────────────────

    def register_with_sync(self, sync_manager) -> None:
        canvases = self.active_canvases()
        if not canvases:
            return
        sync_manager.clear()
        sync_manager.register_many(canvases, master_canvas=canvases[0])

    def normalize_all_to_session_window(self, session) -> None:
        """Set all canvas X ranges to the global session window ("Fit All").

        Call this after refresh_all() via QTimer.singleShot(0) so pyqtgraph's
        auto-range events have settled before we force a common viewport.

        Stage 2 leaves this untouched: it remains the one operation that fits
        the complete session DATA DOMAIN, and is what restores the full range
        after an event-focused initial view.
        """
        t_start, t_end = _session_window(session)
        for canvas in self._canvases.values():
            canvas.normalize_viewport(t_start, t_end)

    # ─────────────────────────────────────────────────────────────────────────
    # Initial viewport (Stage 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _live_canvases(self) -> list[SessionCanvasWidget]:
        return [c for c in self._canvases.values() if not sip.isdeleted(c)]

    def _current_shared_x_range(self) -> tuple[float, float] | None:
        """X range the session is currently showing, read from any live canvas.

        All panels are held to one range by SynchronizationManager and per-panel
        setXLink, so any live canvas reports it.
        """
        for canvas in self._live_canvases():
            try:
                x_range = canvas.getViewBox().viewRange()[0]
                return float(x_range[0]), float(x_range[1])
            except (RuntimeError, IndexError, TypeError, ValueError):  # noqa: PERF203
                continue
        return None

    def _set_all_x_range(self, t_start: float, t_end: float) -> None:
        for canvas in self._live_canvases():
            canvas.normalize_viewport(t_start, t_end)

    def apply_initial_viewport(self, session) -> tuple[float, float] | None:
        """Choose and apply this session's INITIAL visible X range, once.

        Replaces the unconditional normalize_all_to_session_window() that used
        to run on every session-canvas activation. Semantics:

        - First activation with at least two active sources: ask
          viewport_policy.select_initial_viewport(). If it returns a window,
          every panel is set to it; if it returns None, the previous behaviour
          (fit the full session domain) is used unchanged. Either way the
          session is then marked initialised.
        - A session that has never had two active sources (an ordinary
          single-source session) is never marked initialised, so it keeps
          fitting the full domain exactly as before — and a second source
          arriving later still gets a proper initial decision.
        - Every later activation: the visible range is left alone, because by
          then it may be a range the analyst deliberately zoomed or panned to.
          Only canvases created by the latest rebuild are pulled into the
          range already on screen, so a new panel cannot auto-range to its own
          data and drag the others with it.

        Returns the event-focused window when one was applied, else None.
        Never mutates the session, its offsets, or any record.
        """
        self._session_ref = session
        if self._viewport_initialized_session is session:
            if self._x_range_before_rebuild is not None:
                self._set_all_x_range(*self._x_range_before_rebuild)
            return None

        window = select_initial_viewport(session)
        if window is None:
            self.normalize_all_to_session_window(session)
        else:
            self._set_all_x_range(*window)

        active = sum(1 for s in session.list_sources() if s.is_active)
        if active >= MIN_SOURCES_FOR_EVENT_FOCUS:
            self._viewport_initialized_session = session
        return window

    # ─────────────────────────────────────────────────────────────────────────
    # Time axis mode (mirrors FlexiblePlotCanvas.set_time_axis_mode)
    # ─────────────────────────────────────────────────────────────────────────

    def set_time_axis_mode(self, mode, session) -> None:
        """Switch all session canvases between relative and absolute time labels.

        Matches the View → Time Axis Mode menu that controls FlexiblePlotCanvas.
        ABSOLUTE re-derives the session origin from loaded sources; RELATIVE
        clears the reference so raw elapsed-second values are shown.
        """
        m = TimeDisplayMode.coerce(mode)
        if _session_time_display_mode(session) == TimeDisplayMode.SAMPLE_INDEX:
            for canvas in self._canvases.values():
                canvas.set_sample_index_axis()
            return
        if m == TimeDisplayMode.ABSOLUTE:
            self._apply_time_reference(session)
        else:
            for canvas in self._canvases.values():
                canvas.clear_time_reference()

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

    @staticmethod
    def _analog_channel_lookup(session) -> dict[tuple[str, str], object]:
        lookup: dict[tuple[str, str], object] = {}
        for source in session.list_sources():
            for ch in source.record.analog_channels:
                lookup[(source.source_id, ch.name)] = ch
        return lookup

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
        digital_keys = {
            (ch.source_id, ch.channel_name)
            for ch in session.list_digital_channels(active_only=False)
        }
        digital_row = 0
        digital_tick_labels: list[tuple[float, str]] = []
        analog_channels = self._analog_channel_lookup(session)
        panel_axis_sides: dict[str, str] = {}
        for source_id, channel_name in panel.channel_refs:
            if source_id not in active_sids:
                canvas.set_curve_visible(source_id, channel_name, False)
                continue
            ch = all_channels.get((source_id, channel_name))
            is_digital = (source_id, channel_name) in digital_keys
            source_channel = analog_channels.get((source_id, channel_name))
            visible = ch.is_visible if ch is not None else True
            color = (ch.color_hex if (ch and ch.color_hex) else None) or self._auto_colour(source_id, channel_name)
            if not visible:
                canvas.set_curve_visible(source_id, channel_name, False)
                continue
            try:
                aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                source = sources_by_id.get(source_id)
                badge = source.display_name if source else source_id
                if is_digital:
                    label = (ch.display_name if ch and ch.display_name else channel_name)
                    canvas.update_digital_curve(
                        source_id, channel_name,
                        aligned.time, aligned.values,
                        color=color, visible=True,
                        y_offset=float(digital_row),
                        display_name=label,
                        source_badge=badge,
                        unit=ch.unit if ch else None,
                    )
                    digital_tick_labels.append((float(digital_row), label))
                    digital_row += 1
                    continue

                scaled_values, display_unit = self._scale_aligned(
                    channel_name, aligned.values, aligned.unit
                )
                signal_type = str(getattr(source_channel, "parameter_type", "") or "") or None
                axis_group = axis_group_for_signal(
                    channel_name,
                    display_unit,
                    mode=AxisDisplayMode.SHARED,
                    signal_type_hint=signal_type,
                )
                if ch is not None and ch.y_axis_side == "right":
                    y_axis_side = "right"
                else:
                    y_axis_side = panel_axis_sides.setdefault(
                        axis_group.key,
                        "left" if not panel_axis_sides else "right",
                    )
                canvas.update_curve(
                    source_id, channel_name, aligned.time, scaled_values,
                    color=color, visible=True,
                    display_name=ch.display_name if ch else channel_name,
                    source_badge=badge,
                    unit=display_unit,
                    line_style=ch.line_style if ch else "solid",
                    line_width=ch.line_width if ch else 1.0,
                    y_axis_side=y_axis_side,
                    signal_type=signal_type,
                )
            except Exception:  # noqa: BLE001
                pass

        # For digital-only panels: show channel names as Y-axis tick labels
        is_digital_only = digital_row > 0 and not any(
            (sid, cname) not in digital_keys
            for sid, cname in panel.channel_refs
        )
        if is_digital_only:
            plot = canvas._primary_plot
            vb = plot.getViewBox()
            left = plot.getAxis("left")
            left.setTicks([digital_tick_labels])
            tick_font = QFont()
            tick_font.setPointSize(8)
            left.setStyle(showValues=True, tickLength=0, autoExpandTextSpace=False, tickFont=tick_font)
            left.setWidth(160)
            plot.setMouseEnabled(x=True, y=True)
            vb.enableAutoRange(axis="y", enable=False)
            # Only reset Y range when row count changes (preserves scroll position)
            if self._digital_panel_rows.get(panel.panel_id, -1) != digital_row:
                plot.setYRange(-0.5, float(digital_row) - 0.5, padding=0)
                self._digital_panel_rows[panel.panel_id] = digital_row
            canvas.setup_digital_scroll(digital_row)

        # S2: compute RMS overlays for this panel after all raw curves are updated
        if self._rms_mode != RMSDisplayMode.OFF:
            self._compute_rms_for_panel(canvas, panel, session, all_channels, sources_by_id, t_start, t_end)

        # S3: compute phasor overlays for this panel after all raw curves are updated
        if self._phasor_mode not in (PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS):
            self._compute_phasor_for_panel(canvas, panel, session, all_channels, sources_by_id, t_start, t_end)

        # S4: compute harmonic magnitude overlays for this panel
        if self._harmonic_mode == HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            self._compute_harmonic_for_panel(canvas, panel, session, all_channels, sources_by_id, t_start, t_end)

        canvas.refresh_render_state()

    def _resize_digital_panels(self) -> None:
        """Size panels after rebuild: digital panels get a capped height, analog panels ~250px.

        The splitter's minimum height is set to the sum of all panel heights so the
        outer QScrollArea scrollbar activates when total content exceeds the viewport.
        The user can still drag splitter handles freely after this initial sizing.
        """
        _ROW_PX = 30       # pixels per digital channel row
        _MARGIN_PX = 65    # header strip + legend widget + padding
        _MAX_DIGITAL_PX = 220  # cap — beyond this the per-panel Y scrollbar takes over
        _ANALOG_PX = 250   # default height for analog panels

        if self._splitter is None or sip.isdeleted(self._splitter):
            return

        n_panels = len(self._current_panel_ids)
        if n_panels == 0:
            return

        current = list(self._splitter.sizes())
        if len(current) != n_panels:
            return

        new_sizes: list[int] = []
        for pid in self._current_panel_ids:
            n_rows = self._digital_panel_rows.get(pid, 0)
            if n_rows > 0:
                natural = n_rows * _ROW_PX + _MARGIN_PX
                new_sizes.append(min(natural, _MAX_DIGITAL_PX))
            else:
                new_sizes.append(_ANALOG_PX)

        self._splitter.setMinimumHeight(sum(new_sizes))
        self._splitter.setSizes(new_sizes)

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

    def _source_trigger_t(self, source) -> float | None:
        """Return the trigger time in session-relative seconds for a source.

        Computes (trigger_time − start_time) from the record timing info, then
        adds the source's current time offset so the marker stays aligned after
        auto-alignment adjustments.  Returns None when timing info is absent.
        """
        try:
            timing = source.record.timing_info
            if timing.trigger_time is None or timing.start_time is None:
                return None
            trigger_offset = (timing.trigger_time - timing.start_time).total_seconds()
            return trigger_offset + source.time_offset_s
        except Exception:  # noqa: BLE001
            return None

    def _refresh_trigger_markers(self, session) -> None:
        """Add/update per-source trigger markers on all active canvases (S1)."""
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        for canvas in self._canvases.values():
            # Remove markers for deactivated or removed sources
            for sid in list(canvas._trigger_markers.keys()):
                source = sources_by_id.get(sid)
                if source is None or not source.is_active:
                    canvas.remove_trigger_marker(sid)

            # Add/refresh markers for active sources
            for source in sources_by_id.values():
                if not source.is_active:
                    continue
                t_trigger = self._source_trigger_t(source)
                if t_trigger is None:
                    continue
                color = self._source_color(source.source_id)
                label = f"{source.display_name} ▲"
                canvas.set_trigger_marker(source.source_id, t_trigger, color, label)

    # ─────────────────────────────────────────────────────────────────────────
    # RMS overlay (S2)
    # ─────────────────────────────────────────────────────────────────────────

    def set_rms_mode(
        self,
        mode: RMSDisplayMode,
        session,
        *,
        config: RMSConfig | None = None,
    ) -> None:
        """Enable or disable RMS overlay on all session canvas panels.

        OFF     — raw waveforms only; all RMS curves removed.
        OVERLAY — raw waveform + dashed RMS envelope.
        RMS_ONLY — RMS envelope only; raw waveform hidden for eligible channels.
        """
        self._rms_mode = mode
        if config is not None:
            self._rms_config = config

        if mode == RMSDisplayMode.OFF:
            for canvas in self._canvases.values():
                canvas.clear_rms_curves()
            return

        self._refresh_rms_overlays(session)

    def _refresh_rms_overlays(self, session) -> None:
        """Recompute RMS overlays for all panels using the current mode/config."""
        if self._rms_mode == RMSDisplayMode.OFF:
            return
        t_start, t_end = _session_window(session)
        all_channels = self._channel_lookup(session)
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        panels = {p.panel_id: p for p in session.list_panels()}
        for panel_id, canvas in self._canvases.items():
            panel = panels.get(panel_id)
            if panel is None:
                continue
            self._compute_rms_for_panel(
                canvas, panel, session, all_channels, sources_by_id, t_start, t_end
            )

    @staticmethod
    def _estimate_sample_rate(source) -> float:
        """Estimate sampling rate (Hz) from a source's waveform time column."""
        try:
            time_col = source.record.waveform_data.get("time")
            if time_col is None or len(time_col) < 2:
                return 0.0
            arr = np.asarray(time_col[: min(200, len(time_col))], dtype=np.float64)
            diffs = np.diff(arr)
            valid = diffs[np.isfinite(diffs) & (diffs > 0)]
            return float(1.0 / np.median(valid)) if len(valid) > 0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _compute_rms_for_panel(
        self,
        canvas: SessionCanvasWidget,
        panel,
        session,
        all_channels: dict,
        sources_by_id: dict,
        t_start: float,
        t_end: float,
    ) -> None:
        """Compute sliding-RMS for eligible channels in one panel and push to canvas."""
        from app.analytics.rms.rms_overlay import classify_rms_eligibility
        from app.analytics.rms.sliding_rms import compute_rms_overlay

        rms_only = (self._rms_mode == RMSDisplayMode.RMS_ONLY)

        for source_id, channel_name in panel.channel_refs:
            source = sources_by_id.get(source_id)
            if source is None or not source.is_active:
                continue
            ch = all_channels.get((source_id, channel_name))
            if ch is not None and not ch.is_visible:
                continue

            if not classify_rms_eligibility(channel_name).eligible:
                continue

            try:
                sample_rate = self._estimate_sample_rate(source)
                if sample_rate <= 0:
                    continue
                aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                scaled_values, _ = self._scale_aligned(
                    channel_name, aligned.values, aligned.unit
                )
                rms_t, rms_d = compute_rms_overlay(
                    aligned.time, scaled_values, sample_rate, config=self._rms_config
                )
                color = (
                    (ch.color_hex if (ch and ch.color_hex) else None)
                    or self._auto_colour(source_id, channel_name)
                )
                canvas.update_rms_curve(
                    source_id, channel_name, rms_t, rms_d, color,
                    unit=aligned.unit, rms_only=rms_only,
                )
            except Exception:  # noqa: BLE001
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Phasor overlay (S3)
    # ─────────────────────────────────────────────────────────────────────────

    def set_phasor_mode(
        self,
        mode: PhasorDisplayMode,
        session,
        *,
        config: PhasorConfig | None = None,
    ) -> None:
        """Enable or disable phasor magnitude/angle overlay on all session canvas panels.

        OFF / SEQUENCE_COMPONENTS — in-panel overlays cleared (sequence panels
                                    are not managed by the session canvas controller).
        MAGNITUDE — RMS magnitude envelope drawn as a dotted brighter curve.
        ANGLE     — Phase angle trace drawn as a dash-dot cyan-shifted curve.
        """
        self._phasor_mode = mode
        if config is not None:
            self._phasor_config = config

        if mode in (PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS):
            for canvas in self._canvases.values():
                canvas.clear_phasor_curves()
            return

        self._refresh_phasor_overlays(session)

    def _refresh_phasor_overlays(self, session) -> None:
        """Recompute phasor overlays for all panels using the current mode/config."""
        if self._phasor_mode in (PhasorDisplayMode.OFF, PhasorDisplayMode.SEQUENCE_COMPONENTS):
            return
        t_start, t_end = _session_window(session)
        all_channels = self._channel_lookup(session)
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        panels = {p.panel_id: p for p in session.list_panels()}
        for panel_id, canvas in self._canvases.items():
            panel = panels.get(panel_id)
            if panel is None:
                continue
            self._compute_phasor_for_panel(
                canvas, panel, session, all_channels, sources_by_id, t_start, t_end
            )

    def _compute_phasor_for_panel(
        self,
        canvas: SessionCanvasWidget,
        panel,
        session,
        all_channels: dict,
        sources_by_id: dict,
        t_start: float,
        t_end: float,
    ) -> None:
        """Compute phasor magnitude/angle overlay for eligible channels in one panel."""
        from app.analytics.phasors.phasor_extraction import (
            compute_phasor_window_samples,  # noqa: F401 — ensures import chain is valid
            extract_phasor,
        )
        from app.analytics.phasors.phasor_models import PhasorChannelRole
        from app.analytics.phasors.phasor_overlay import classify_phasor_role

        mode = self._phasor_mode

        for source_id, channel_name in panel.channel_refs:
            source = sources_by_id.get(source_id)
            if source is None or not source.is_active:
                continue
            ch = all_channels.get((source_id, channel_name))
            if ch is not None and not ch.is_visible:
                continue

            try:
                sample_rate = self._estimate_sample_rate(source)
                if sample_rate <= 0:
                    continue
                aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                result = classify_phasor_role(channel_name, aligned.unit)
                if result.role == PhasorChannelRole.UNKNOWN:
                    continue
                scaled_values, _ = self._scale_aligned(
                    channel_name, aligned.values, aligned.unit
                )
                phasor_t, mag_rms, angle_deg, _ = extract_phasor(
                    aligned.time, scaled_values, sample_rate, self._phasor_config
                )
                phasor_vals = (
                    mag_rms if mode == PhasorDisplayMode.MAGNITUDE else angle_deg
                )
                color = (
                    (ch.color_hex if (ch and ch.color_hex) else None)
                    or self._auto_colour(source_id, channel_name)
                )
                canvas.update_phasor_curve(
                    source_id, channel_name, phasor_t, phasor_vals, color, mode
                )
            except Exception:  # noqa: BLE001
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Engineering scaling (S5)
    # ─────────────────────────────────────────────────────────────────────────

    def set_scaling_mode(
        self,
        mode: EngineeringScalingMode,
        session,
        *,
        registry: ScalingRegistry | None = None,
    ) -> None:
        """Apply an engineering scaling mode to all session canvas panels.

        Triggers a full repaint so all raw curves, RMS/phasor/harmonic overlays,
        and axis unit labels reflect the new scaling.  When *registry* is supplied
        it replaces the current ScalingRegistry (PT/CT ratios, per-unit bases).
        """
        if registry is not None:
            self._scaling_registry = registry
        self._scaling_mode = mode
        self.refresh_all(session)

    def _scale_aligned(
        self,
        channel_name: str,
        values: np.ndarray,
        unit: str | None,
    ) -> tuple[np.ndarray, str | None]:
        """Return (scaled_values, display_unit) for the current scaling mode.

        RAW mode and unconfigured channels (e.g. PER_UNIT without a voltage base)
        are returned unchanged so we never display silently wrong values.
        """
        if self._scaling_mode == EngineeringScalingMode.RAW:
            return values, unit
        result = self._scaling_registry.compute_scaling_result(
            channel_name, unit, self._scaling_mode
        )
        display_unit: str | None = result.display_unit or unit
        if result.configured and result.factor != 1.0:
            return values * result.factor, display_unit
        return values, display_unit

    # ─────────────────────────────────────────────────────────────────────────
    # Two-cursor measurement (S6)
    # ─────────────────────────────────────────────────────────────────────────

    def get_channel_colour(self, source_id: str, channel_name: str) -> str:
        """Return the effective display colour for a channel.

        Reads from any canvas that already has the curve rendered (post refresh_all),
        falling back to the auto-colour computation when not yet rendered.
        """
        key = (source_id, channel_name)
        for canvas in self._canvases.values():
            meta = canvas._metadata.get(key)
            if meta is not None:
                return meta.colour
        return self._auto_colour(source_id, channel_name)

    def set_measurement_callback(self, cb) -> None:
        """Register a callable(MeasurementResult|None) invoked when cursors move."""
        self._measurement_callback = cb

    def set_measurement_mode_changed_callback(self, cb) -> None:
        """Register a callback(enabled: bool) to notify the main window when
        measurement mode is toggled from the right-click canvas menu."""
        self._measurement_mode_changed_cb = cb

    def set_cursor_sync(self, enabled: bool) -> None:
        """Enable or disable cross-panel cursor synchronisation."""
        self._cursor_sync = enabled
        for canvas in self._canvases.values():
            canvas.set_cursor_sync_state(enabled)

    def set_measurement_mode(self, enabled: bool, session) -> None:
        """Enable or disable two-cursor measurement mode on all session panels.

        Each panel gets its own independent A/B cursor pair. When either cursor
        moves the registered measurement callback receives a MeasurementResult
        computed from all visible channels in that panel.
        """
        self._measurement_mode = enabled
        self._session_ref = session
        for canvas in self._canvases.values():
            canvas.set_measurement_mode(enabled)
            canvas.set_cursor_sync_state(self._cursor_sync)

    def compute_current_measurements(
        self, canvas: SessionCanvasWidget, session
    ) -> object | None:
        """Compute cursor measurements for one panel canvas.

        Sprint 1A (per-curve measurement architecture): every VISIBLE curve
        already painted on *canvas* -- real channel or calculated signal,
        it makes no difference -- contributes its own already-scaled,
        already-materialized (time, values) arrays directly from the
        PlotDataItem the user is actually looking at. No curve is required
        to share a length, sample rate, or time base with any other; no
        curve is dropped because its length differs from another's. This
        also means measurement no longer re-fetches or re-scales data via
        session.build_aligned_data() -- what is measured is guaranteed
        identical to what is plotted.

        Calculated signals receive no special-case code here: they are
        already painted via the same SessionCanvasWidget.update_curve()
        path as real channels (see refresh_calculated_signals()), and this
        method never distinguishes a curve by type or origin -- it iterates
        canvas._curves uniformly. The one label-disambiguation step below
        is driven purely by an actual label collision (two curves that
        would otherwise render the identical name), not by curve type; a
        calculated signal's name is already unique within the session, so
        it is never affected in practice.

        Returns a ``MeasurementResult`` or ``None`` when measurement mode
        is off, no cursors exist, or no curve is currently visible.
        """
        from app.visualization.interaction.measurement_engine import (
            CurveSample,
            compute_measurements,
        )
        if not self._measurement_mode:
            return None
        positions = canvas.cursor_positions()
        if positions is None:
            return None
        t_a, t_b = positions

        visible = [
            (item, canvas._metadata[key])
            for key, item in canvas._curves.items()
            if item.isVisible() and key in canvas._metadata
        ]
        if not visible:
            return None

        # Disambiguate only labels that actually collide (e.g. two sources
        # both have a channel named "Va") -- everything else keeps its
        # plain display name, regardless of how many curves/sources
        # contribute to the panel.
        base_labels = [meta.display_name or meta.channel_name for _, meta in visible]
        label_counts: dict[str, int] = {}
        for label in base_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        curves: list[CurveSample] = []
        for (item, meta), base_label in zip(visible, base_labels):
            time_arr, values_arr = item.getOriginalDataset()
            if time_arr is None:
                time_arr = np.array([], dtype=np.float64)
            if values_arr is None:
                values_arr = np.array([], dtype=np.float64)
            label = base_label
            if label_counts[base_label] > 1 and meta.source_badge:
                label = f"{meta.source_badge}/{base_label}"
            curves.append(CurveSample(
                name=label,
                unit=meta.unit or "",
                time=np.asarray(time_arr, dtype=np.float64),
                values=np.asarray(values_arr, dtype=np.float64),
            ))

        nominal_hz = None
        for source in session.list_sources():
            if not source.is_active:
                continue
            try:
                raw = getattr(source.record.metadata, "nominal_frequency", None)
                nominal_hz = float(raw) if raw else None
            except Exception:  # noqa: BLE001
                pass
            if nominal_hz:
                break

        return compute_measurements(t_a, t_b, curves, nominal_hz=nominal_hz)

    def _on_canvas_measurement_moved(
        self, canvas: SessionCanvasWidget, t_a: float, t_b: float
    ) -> None:
        """Invoked when any panel canvas emits measurement_cursors_moved."""
        if self._measurement_callback is None or self._session_ref is None:
            return
        result = self.compute_current_measurements(canvas, self._session_ref)
        self._measurement_callback(result)

    # ─────────────────────────────────────────────────────────────────────────
    # Harmonic magnitude overlay (S4)
    # ─────────────────────────────────────────────────────────────────────────

    def set_harmonic_mode(
        self,
        mode: HarmonicDisplayMode,
        session,
        *,
        config: HarmonicConfig | None = None,
    ) -> None:
        """Enable or disable harmonic magnitude overlay on all session canvas panels.

        Only HARMONIC_MAGNITUDE produces in-panel curves.  THD, SPECTRUM and OFF
        all clear any existing harmonic overlays — the session canvas controller
        does not manage separate THD/spectrum panels.
        """
        self._harmonic_mode = mode
        if config is not None:
            self._harmonic_config = config

        if mode != HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            for canvas in self._canvases.values():
                canvas.clear_harmonic_curves()
            return

        self._refresh_harmonic_overlays(session)

    def _refresh_harmonic_overlays(self, session) -> None:
        """Recompute harmonic overlays for all panels using the current mode/config."""
        if self._harmonic_mode != HarmonicDisplayMode.HARMONIC_MAGNITUDE:
            return
        t_start, t_end = _session_window(session)
        all_channels = self._channel_lookup(session)
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        panels = {p.panel_id: p for p in session.list_panels()}
        for panel_id, canvas in self._canvases.items():
            panel = panels.get(panel_id)
            if panel is None:
                continue
            self._compute_harmonic_for_panel(
                canvas, panel, session, all_channels, sources_by_id, t_start, t_end
            )

    def _compute_harmonic_for_panel(
        self,
        canvas: SessionCanvasWidget,
        panel,
        session,
        all_channels: dict,
        sources_by_id: dict,
        t_start: float,
        t_end: float,
    ) -> None:
        """Compute H3/H5/H7/H11/H13 magnitude overlays for eligible channels in one panel."""
        from app.analytics.harmonics.harmonic_extraction import (
            compute_harmonic_window_samples,
            extract_harmonics,
        )
        from app.analytics.harmonics.harmonic_models import HarmonicChannelRole
        from app.analytics.harmonics.harmonic_overlay import classify_harmonic_role

        for source_id, channel_name in panel.channel_refs:
            source = sources_by_id.get(source_id)
            if source is None or not source.is_active:
                continue
            ch = all_channels.get((source_id, channel_name))
            if ch is not None and not ch.is_visible:
                continue

            try:
                sample_rate = self._estimate_sample_rate(source)
                if sample_rate <= 0:
                    continue
                aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                role = classify_harmonic_role(channel_name, aligned.unit).role
                if role == HarmonicChannelRole.UNKNOWN:
                    continue
                scaled_values, _ = self._scale_aligned(
                    channel_name, aligned.values, aligned.unit
                )
                h_result = extract_harmonics(
                    scaled_values, sample_rate, self._harmonic_config,
                    time=aligned.time,
                )
                if h_result.n_windows == 0:
                    continue
                magnitudes_by_order: dict[int, np.ndarray] = {}
                for order in self._harmonic_orders:
                    mag = h_result.get_magnitude(order)
                    if mag is not None:
                        magnitudes_by_order[order] = mag
                if magnitudes_by_order:
                    canvas.update_harmonic_curves(
                        source_id, channel_name,
                        h_result.harmonic_time,
                        magnitudes_by_order,
                    )
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _derive_reference_time_from_sources(session) -> datetime | None:
        """Legacy axis origin: earliest ``start_time − time_offset_s``.

        Retained as the fallback for sessions that never establish an explicit
        absolute origin — single-source sessions, relative/synthetic/
        sample-index sources, and legacy manifests whose saved offsets have no
        recorded origin (Stage 3 treats those as opaque geometry and
        deliberately invents no origin for them).

        Returning None also answers a second question the caller depends on:
        whether an absolute axis is meaningful at all. None means no active
        source carries a real wall-clock anchor, so the axis stays in elapsed
        seconds.
        """
        origin: datetime | None = None
        for source in session.list_sources():
            if not source.is_active:
                continue
            try:
                if getattr(source.record.timing_info, "timing_reference", "absolute") != "absolute":
                    continue
                start = source.record.timing_info.start_time
                if start is None:
                    continue
                candidate = start - timedelta(seconds=source.time_offset_s)
                if origin is None or candidate < origin:
                    origin = candidate
            except Exception:  # noqa: BLE001
                continue
        return origin

    def _compute_session_reference_time(self, session) -> datetime | None:
        """Return the wall-clock time that corresponds to session x = 0.

        Stage 3.1: when the session has an explicit ``absolute_time_origin``,
        that IS the answer — it is the Stage 1 definition of what x = 0 means,
        and it is stable under alignment edits.

        The previous implementation always re-derived the origin as
        ``min(start_time − time_offset_s)``. That is equal to the session
        origin only while every offset is exactly timestamp-derived, and it
        breaks the moment an analyst adjusts one source: because it takes the
        MINIMUM, moving a single source 250 ms later dragged the whole axis
        250 ms earlier, which both relabelled every unrelated source and
        cancelled out the adjustment so the analyst could not see it. An
        alignment edit must move that source against a stable session clock,
        not redefine the clock.

        The legacy derivation is still used when no explicit origin exists,
        and is still what decides whether an absolute axis applies at all: if
        it finds no wall-clock-anchored active source, None is returned and the
        axis stays in elapsed seconds even though a stale origin may linger
        from sources that have since been removed or deactivated.

        Affects X-axis tick labels only — the sole consumer is
        SessionCanvasWidget.set_time_reference() → DatetimeAxisItem.
        """
        derived = self._derive_reference_time_from_sources(session)
        if derived is None:
            return None
        origin = getattr(session, "absolute_time_origin", None)
        return origin if origin is not None else derived

    def _apply_time_reference(self, session) -> None:
        """Push the session wall-clock origin to all canvas axes."""
        if _session_time_display_mode(session) == TimeDisplayMode.SAMPLE_INDEX:
            for canvas in self._canvases.values():
                canvas.set_sample_index_axis()
            return
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

    def _on_canvas_cursor_a_moved(self, source: SessionCanvasWidget, t: float) -> None:
        """Propagate cursor A position to all other canvases when sync is on."""
        if not self._cursor_sync:
            return
        for canvas in self._canvases.values():
            if canvas is not source:
                canvas.set_cursor_a_pos(t)

    def _on_canvas_cursor_b_moved(self, source: SessionCanvasWidget, t: float) -> None:
        """Propagate cursor B position to all other canvases when sync is on."""
        if not self._cursor_sync:
            return
        for canvas in self._canvases.values():
            if canvas is not source:
                canvas.set_cursor_b_pos(t)

    def _on_measurement_toggle_from_canvas(self, enabled: bool) -> None:
        """Handle measurement mode toggle originating from a canvas right-click menu."""
        if self._session_ref is not None:
            self.set_measurement_mode(enabled, self._session_ref)
        if self._measurement_mode_changed_cb is not None:
            self._measurement_mode_changed_cb(enabled)

    def _on_cursor_sync_toggled(self, enabled: bool) -> None:
        self.set_cursor_sync(enabled)

    def set_crosshair_readout_callback(self, cb) -> None:
        """Register callable(t, values) called when the hover crosshair moves over any panel."""
        self._crosshair_readout_cb = cb

    def set_channel_panel_changed_callback(self, cb) -> None:
        """Register callable(source_id, ch_name, panel_id) — fired when a channel is drag-dropped."""
        self._channel_panel_changed_cb = cb

    def set_navigator(self, navigator) -> None:
        """Wire the WaveformNavigatorStrip to the session: populate it and sync viewport."""
        self._navigator = navigator
        # User dragging the navigator region → pan all canvases
        navigator.viewport_changed.connect(self._on_navigator_viewport_changed)
        # Wire master canvas X range changes → update navigator region
        if self._current_panel_ids and self._canvases:
            master_pid = self._current_panel_ids[0]
            master_canvas = self._canvases.get(master_pid)
            if master_canvas is not None:
                master_canvas.getViewBox().sigXRangeChanged.connect(
                    self._on_canvas_x_range_changed_for_nav
                )

    def _on_navigator_viewport_changed(self, t_start: float, t_end: float) -> None:
        """Navigator region dragged → set X range on all panels."""
        for canvas in self._canvases.values():
            canvas.normalize_viewport(t_start, t_end)

    def _on_canvas_x_range_changed_for_nav(self, _vb, x_range) -> None:
        """Master canvas X range changed → update navigator region."""
        if self._navigator is not None:
            t_start, t_end = float(x_range[0]), float(x_range[1])
            self._navigator.set_viewport(t_start, t_end)

    def _populate_navigator(self, session) -> None:
        """Fill the navigator strip with downsampled channel data."""
        if self._navigator is None:
            return
        self._navigator.clear_channels()
        t_start, t_end = _session_window(session)
        self._navigator.set_time_range(t_start, t_end)
        all_channels = self._channel_lookup(session)
        sources_by_id = {s.source_id: s for s in session.list_sources()}
        digital_keys = {
            (ch.source_id, ch.channel_name)
            for ch in session.list_digital_channels(active_only=False)
        }

        for panel_id, canvas in self._canvases.items():
            panels = {p.panel_id: p for p in session.list_panels()}
            panel = panels.get(panel_id)
            if panel is None:
                continue
            for source_id, channel_name in panel.channel_refs:
                if (source_id, channel_name) in digital_keys:
                    continue  # skip digital channels in navigator
                source = sources_by_id.get(source_id)
                if source is None or not source.is_active:
                    continue
                ch = all_channels.get((source_id, channel_name))
                if ch is not None and not ch.is_visible:
                    continue
                try:
                    aligned = session.build_aligned_data(source_id, channel_name, t_start, t_end)
                    color = (ch.color_hex if (ch and ch.color_hex) else None) or self._auto_colour(source_id, channel_name)
                    nav_key = f"{source_id}:{channel_name}"
                    self._navigator.update_channel(nav_key, aligned.time, aligned.values, color)
                except Exception:  # noqa: BLE001
                    pass

    def _on_crosshair_moved(self, source_canvas: SessionCanvasWidget, t: float) -> None:
        """Propagate crosshair position to all other panels without feedback loops."""
        for canvas in self._canvases.values():
            if canvas is not source_canvas:
                canvas.set_crosshair_pos(t)

    def _on_crosshair_values_changed(self, t: float, values: list) -> None:
        """Forward interpolated channel values at the crosshair to the readout bar."""
        if self._crosshair_readout_cb is not None:
            self._crosshair_readout_cb(t, values)

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
        legend.batch_visibility_changed.connect(
            lambda keys, vis: self._handle_legend_batch_visibility(keys, vis)
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
        canvas.channel_drop_requested.connect(self._handle_channel_drop)
        canvas.panel_title_changed.connect(self._handle_panel_rename)
        # S6: route cursor movement through the controller to compute measurements
        canvas.measurement_cursors_moved.connect(
            lambda t_a, t_b, _c=canvas: self._on_canvas_measurement_moved(_c, t_a, t_b)
        )
        # Cursor sync: propagate A/B positions to all other canvases
        canvas.cursor_a_moved.connect(
            lambda t, _c=canvas: self._on_canvas_cursor_a_moved(_c, t)
        )
        canvas.cursor_b_moved.connect(
            lambda t, _c=canvas: self._on_canvas_cursor_b_moved(_c, t)
        )
        # Right-click menu signals
        canvas.measurement_mode_toggle_requested.connect(
            self._on_measurement_toggle_from_canvas
        )
        canvas.cursor_sync_toggle_requested.connect(self._on_cursor_sync_toggled)
        # Hover crosshair: propagate to all other panels + live readout bar
        canvas.crosshair_moved.connect(
            lambda t, _c=canvas: self._on_crosshair_moved(_c, t)
        )
        canvas.crosshair_values_changed.connect(self._on_crosshair_values_changed)

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

    def _handle_legend_batch_visibility(
        self, keys: list, visible: bool
    ) -> None:
        for source_id, channel_name in keys:
            if self._session_ref is not None:
                self._session_ref.set_channel_visibility(source_id, channel_name, visible)
            for canvas in self._canvases.values():
                canvas.set_curve_visible(source_id, channel_name, visible)
                canvas.legend.update_row_visible(source_id, channel_name, visible)

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
