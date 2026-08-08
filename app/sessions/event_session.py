"""EventAnalysisSession — multi-source analyst workspace for Phase 9A.

Key invariants:
- DisturbanceRecord.waveform_data is NEVER mutated.
- Time offsets are stored in SessionSource and applied lazily at render time.
- session_version and created_at are set once at construction and never changed.
- Qt-free: no PyQt6 imports anywhere in this module.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from app.sessions.session_models import (
    ALIGNMENT_METHODS,
    AlignedChannelData,
    CalculatedSignalEntry,
    ChannelEligibility,
    ChannelEligibilityResult,
    DependencyStatus,
    PanelConfig,
    SessionChannel,
    SessionSource,
    SourceQualityMetrics,
)
from app.sessions import alignment_engine
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)

if TYPE_CHECKING:
    from app.models.disturbance_record import DisturbanceRecord

# Panel ID constants for default_layout()
_PANEL_VOLTAGE = "voltage"
_PANEL_CURRENT = "current"
_PANEL_POWER = "power"
_PANEL_FREQUENCY = "frequency"
_PANEL_DIGITAL = "digital"
_PANEL_OTHER = "other"

# Channel-name heuristics for default panel assignment (mirrors channel_grouper logic).
#
# Split into exact-token and prefix-root sets so matching is boundary-aware:
# a candidate must appear as its own delimiter-separated token (see _tokenize)
# to satisfy an "exact" entry, or as the leading characters of a token to
# satisfy a "prefix" root. This prevents short/ambiguous fragments (e.g. "in",
# "va") from matching inside unrelated words (e.g. "Input", "Interval") the
# way raw substring containment did.
_VOLTAGE_EXACT_TOKENS = {"va", "vb", "vc", "vn", "vab", "vbc", "vca", "kv"}
_VOLTAGE_PREFIX_ROOTS = ("volt",)
_CURRENT_EXACT_TOKENS = {"ia", "ib", "ic", "in", "current"}
_CURRENT_PREFIX_ROOTS = ("amp",)
_POWER_EXACT_TOKENS = {"mw", "mvar", "power", "apparent", "active", "reactive", "p", "q"}
_POWER_PREFIX_ROOTS: tuple[str, ...] = ()
_FREQ_EXACT_TOKENS = {"hz", "rocof", "df"}
_FREQ_PREFIX_ROOTS = ("freq",)

# Unit-based heuristics — exact match after lower+strip (more authoritative than name keywords)
_VOLTAGE_UNITS = {"v", "kv", "mv", "pu", "p.u.", "volt", "volts"}
_CURRENT_UNITS = {"a", "ka", "ma", "amp", "amps"}
_POWER_UNITS   = {"w", "kw", "mw", "gw", "var", "kvar", "mvar", "gvar", "va", "kva", "mva"}
_FREQ_UNITS    = {"hz", "rad/s", "hz/s", "rad/s2", "rad/s²"}

# ParameterType.value → panel_id (most authoritative — explicit user/import classification).
# Recognizes both the Import Wizard's ParameterType vocabulary (mw, mvar,
# voltage, current, frequency, rocof) and the shared classifier vocabulary
# used by the direct CSV/Excel providers (active_power, reactive_power,
# voltage_rms, current_rms, frequency, rocof) — the two taxonomies previously
# only overlapped by coincidence on "frequency"/"rocof", so direct-provider
# channels fell through to the unit/name fallback even when confidently typed.
_TYPE_TO_PANEL: dict[str, str] = {
    "voltage":        _PANEL_VOLTAGE,
    "voltage_rms":    _PANEL_VOLTAGE,
    "current":        _PANEL_CURRENT,
    "current_rms":    _PANEL_CURRENT,
    "mw":             _PANEL_POWER,
    "active_power":   _PANEL_POWER,
    "mvar":           _PANEL_POWER,
    "reactive_power": _PANEL_POWER,
    "frequency":      _PANEL_FREQUENCY,
    "rocof":          _PANEL_FREQUENCY,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(name: str) -> list[str]:
    """Split a channel name into lowercase alphanumeric tokens.

    Delimiters (space, underscore, hyphen, slash, brackets, punctuation) all
    act as boundaries, so "P_Total", "P Total", and "P-Total" tokenize
    identically. Tokens with no delimiter between them (e.g. "IndexValue")
    remain a single token — this is what keeps a short fragment like "va"
    from matching inside "IndexValue" or "Interval".
    """
    return _TOKEN_RE.findall(name.lower())


def _matches_keywords(
    tokens: list[str],
    exact: set[str],
    prefixes: tuple[str, ...] = (),
) -> bool:
    for tok in tokens:
        if tok in exact:
            return True
        if prefixes and tok.startswith(prefixes):
            return True
    return False


def _infer_panel_for_type(param_type: str | None) -> str | None:
    """Return panel_id from a parameter-type string, or None if not mapped."""
    if not param_type:
        return None
    return _TYPE_TO_PANEL.get(param_type.lower())


def _infer_panel_for_unit(unit: str | None) -> str | None:
    """Return panel_id from engineering unit, or None if not deterministic."""
    if not unit:
        return None
    u = unit.lower().strip()
    if u in _VOLTAGE_UNITS:
        return _PANEL_VOLTAGE
    if u in _CURRENT_UNITS:
        return _PANEL_CURRENT
    if u in _POWER_UNITS:
        return _PANEL_POWER
    if u in _FREQ_UNITS:
        return _PANEL_FREQUENCY
    return None


def _infer_panel_for_channel(
    channel_name: str,
    unit: str | None = None,
    param_type: str | None = None,
) -> tuple[str, str]:
    """Return (panel_id, panel_type) using type → unit → name-keyword priority."""
    pid = _infer_panel_for_type(param_type)
    if pid is not None:
        return pid, "analog"
    pid = _infer_panel_for_unit(unit)
    if pid is not None:
        return pid, "analog"
    tokens = _tokenize(channel_name)
    if _matches_keywords(tokens, _VOLTAGE_EXACT_TOKENS, _VOLTAGE_PREFIX_ROOTS):
        return _PANEL_VOLTAGE, "analog"
    if _matches_keywords(tokens, _CURRENT_EXACT_TOKENS, _CURRENT_PREFIX_ROOTS):
        return _PANEL_CURRENT, "analog"
    if _matches_keywords(tokens, _POWER_EXACT_TOKENS, _POWER_PREFIX_ROOTS):
        return _PANEL_POWER, "analog"
    if _matches_keywords(tokens, _FREQ_EXACT_TOKENS, _FREQ_PREFIX_ROOTS):
        return _PANEL_FREQUENCY, "analog"
    return _PANEL_OTHER, "analog"


_DEFAULT_PANEL_TITLES = {
    _PANEL_VOLTAGE: "Voltage",
    _PANEL_CURRENT: "Current",
    _PANEL_POWER: "Power",
    _PANEL_FREQUENCY: "Frequency",
    _PANEL_DIGITAL: "Digital",
    _PANEL_OTHER: "Other Analog",
}


class EventAnalysisSession:
    """Container for one analyst workspace: N sources + channel registry + panel layout.

    All time offsets are applied lazily at render time — DisturbanceRecord is never mutated.

    Persistence hooks (Enhancement 2):
    - session_version: int  — increment when serialisation format changes
    - created_at: datetime  — UTC; set once at construction, treated as immutable
    """

    SESSION_VERSION = 1

    def __init__(self) -> None:
        self.session_version: int = self.SESSION_VERSION
        self.created_at: datetime = datetime.now(tz=timezone.utc)

        self._sources: dict[str, SessionSource] = {}      # source_id → SessionSource
        self._channels: dict[tuple[str, str], SessionChannel] = {}
        # key: (source_id, channel_name)
        self._panels: dict[str, PanelConfig] = {}         # panel_id → PanelConfig
        self._quality_cache: dict[str, SourceQualityMetrics] = {}
        # Phase 9C: human-readable alignment notes from the last auto-align result
        self._alignment_notes: dict[str, str] = {}        # source_id → notes

        # Calculated Signals (Phase 2C-1): session-owned definitions/results,
        # plus forward and reverse dependency indexes derived from each
        # definition's variable_bindings (never a second, competing source
        # of truth -- see CalculatedSignalEntry's docstring).
        self._calc_signals: dict[str, CalculatedSignalEntry] = {}
        self._calc_dependencies: dict[str, frozenset[ChannelRef]] = {}
        self._calc_dependents_by_source: dict[str, set[str]] = {}
        self._calc_dependents_by_channel: dict[ChannelRef, set[str]] = {}

    # -------------------------------------------------------------------------
    # Source management
    # -------------------------------------------------------------------------

    def add_source(
        self,
        record: "DisturbanceRecord",
        display_name: str,
        provider_type: str,
        origin_path: str | None = None,
    ) -> str:
        """Add a DisturbanceRecord as a new source. Returns the new source_id."""
        source_id = str(uuid.uuid4())
        source = SessionSource(
            source_id=source_id,
            display_name=display_name,
            record=record,
            provider_type=provider_type,
            origin_path=origin_path,
            time_offset_s=0.0,
            is_active=True,
            alignment_method="none",
            alignment_confidence=None,
        )
        self._sources[source_id] = source
        self._quality_cache.pop(source_id, None)
        self._register_channels(source_id, record)
        return source_id

    def remove_source(self, source_id: str) -> None:
        """Remove a source and all its channels from the session.

        Calculated Signals (Phase 2C-2): any calculated signal depending on
        this source has its latest result marked stale BEFORE the source is
        removed -- its definition, dependency metadata, and (now-stale)
        result all remain; nothing calculated-signal-related is deleted
        here. get_calculated_dependents_for_source(source_id) continues to
        answer "what depended on this source" afterward (Phase 2C-1's
        existing discoverability-over-deletion design for the reverse
        index), and get_dependency_status() will report MISSING_SOURCE for
        those signals from this point on.
        """
        if source_id not in self._sources:
            return
        self._mark_calculated_dependents_stale_for_source(source_id, reason="Source removed.")
        del self._sources[source_id]
        self._quality_cache.pop(source_id, None)
        self._alignment_notes.pop(source_id, None)
        # Remove all channels belonging to this source
        keys_to_remove = [k for k in self._channels if k[0] == source_id]
        for k in keys_to_remove:
            del self._channels[k]
        # Remove channel_refs from all panels
        for panel in self._panels.values():
            panel.channel_refs = [
                ref for ref in panel.channel_refs if ref[0] != source_id
            ]

    def get_source(self, source_id: str) -> SessionSource | None:
        return self._sources.get(source_id)

    def list_sources(self) -> list[SessionSource]:
        return list(self._sources.values())

    def has_meaningful_work(self) -> bool:
        """True if this session has anything a user would not want to lose
        silently -- at least one loaded source or one calculated signal
        (Sprint 1D: the minimum generic rule for whether a destructive
        session action, e.g. Clear Session or starting a new session,
        needs a confirmation prompt).
        """
        return bool(self._sources or self._calc_signals)

    def set_source_active(self, source_id: str, active: bool) -> None:
        """Set a source's active/inactive state.

        Calculated Signals (Phase 2C-2): any dependent calculated signal's
        latest result is marked stale on EITHER transition direction
        (active→inactive or inactive→active) -- reactivating a source does
        not make a previously-computed result fresh again, because that
        result was computed under an earlier source state; only an actual
        recalculation (a later phase) can restore OK. No-op (no
        invalidation) when the active state does not actually change.
        """
        source = self._sources.get(source_id)
        if source is None:
            return
        old_active = source.is_active
        source.is_active = active
        if old_active == active:
            return
        reason = "Source reactivated." if active else "Source deactivated."
        self._mark_calculated_dependents_stale_for_source(source_id, reason=reason)

    # -------------------------------------------------------------------------
    # Time alignment
    # -------------------------------------------------------------------------

    def set_time_offset(
        self,
        source_id: str,
        offset_s: float,
        method: str = "manual",
        confidence: float | None = None,
    ) -> None:
        """Set the view-only time offset for a source.

        method must be one of ALIGNMENT_METHODS.
        confidence is [0.0, 1.0] for auto methods; None for manual/imported.

        Calculated Signals (Phase 2C-2): this is the single canonical
        mutator for both the offset and the alignment method (there is no
        separate method-only setter), so both are compared against their
        prior values and any dependent calculated signal is marked stale
        AT MOST ONCE per call, with a reason reflecting whichever changed
        (or both). No invalidation when neither the offset nor the method
        actually changes (exact float comparison -- no tolerance is
        introduced here, since none exists elsewhere in this module for
        offset comparisons). confidence is purely informational/display
        metadata and is not itself compared for staleness.
        """
        if method not in ALIGNMENT_METHODS:
            raise ValueError(f"Unknown alignment_method {method!r}. "
                             f"Must be one of {sorted(ALIGNMENT_METHODS)}")
        source = self._sources.get(source_id)
        if source is None:
            return
        old_offset = source.time_offset_s
        old_method = source.alignment_method
        source.time_offset_s = float(offset_s)
        source.alignment_method = method
        source.alignment_confidence = confidence

        offset_changed = old_offset != source.time_offset_s
        method_changed = old_method != method
        if offset_changed or method_changed:
            reason = self._offset_or_method_change_reason(offset_changed, method_changed)
            self._mark_calculated_dependents_stale_for_source(source_id, reason=reason)

    def get_time_offset(self, source_id: str) -> float:
        source = self._sources.get(source_id)
        return source.time_offset_s if source is not None else 0.0

    @staticmethod
    def _offset_or_method_change_reason(offset_changed: bool, method_changed: bool) -> str:
        if offset_changed and method_changed:
            return "Source alignment offset and method changed."
        if offset_changed:
            return "Source alignment offset changed."
        return "Source alignment method changed."

    def reset_all_offsets(self) -> None:
        """Reset all source offsets to 0.0 and clear alignment metadata.

        Calculated Signals (Phase 2C-2): for each source whose effective
        offset or method actually changes as a result of the reset,
        dependent calculated signals are marked stale exactly as
        set_time_offset() would -- this method changes the same fields via
        a different call path, so it must trigger the same invalidation.
        """
        for source in self._sources.values():
            old_offset = source.time_offset_s
            old_method = source.alignment_method
            source.time_offset_s = 0.0
            source.alignment_method = "none"
            source.alignment_confidence = None
            offset_changed = old_offset != 0.0
            method_changed = old_method != "none"
            if offset_changed or method_changed:
                reason = self._offset_or_method_change_reason(offset_changed, method_changed)
                self._mark_calculated_dependents_stale_for_source(source.source_id, reason=reason)
        self._alignment_notes.clear()

    def set_alignment_notes(self, source_id: str, notes: str) -> None:
        """Store human-readable notes from the latest alignment operation."""
        if source_id in self._sources:
            self._alignment_notes[source_id] = notes

    def get_alignment_notes(self, source_id: str) -> str:
        """Return the latest alignment notes for a source, or empty string."""
        return self._alignment_notes.get(source_id, "")

    def get_global_time_range(self) -> tuple[float, float]:
        """Return the intersection of all active source time ranges after offsets.

        Returns (0.0, 1.0) as a safe fallback when no active sources exist or
        when there is no overlap between active source ranges.
        """
        active = [s for s in self._sources.values() if s.is_active]
        if not active:
            return (0.0, 1.0)

        range_start = -float("inf")
        range_end = float("inf")

        for source in active:
            t = source.record.waveform_data.get("time")
            if t is None or len(t) == 0:
                continue
            t_shifted_start = float(np.min(t)) + source.time_offset_s
            t_shifted_end = float(np.max(t)) + source.time_offset_s
            range_start = max(range_start, t_shifted_start)
            range_end = min(range_end, t_shifted_end)

        if range_start >= range_end or range_start == -float("inf"):
            # No overlap — return union start/end as a graceful fallback
            all_starts = []
            all_ends = []
            for source in active:
                t = source.record.waveform_data.get("time")
                if t is None or len(t) == 0:
                    continue
                all_starts.append(float(np.min(t)) + source.time_offset_s)
                all_ends.append(float(np.max(t)) + source.time_offset_s)
            if all_starts:
                return (min(all_starts), max(all_ends))
            return (0.0, 1.0)

        return (range_start, range_end)

    # -------------------------------------------------------------------------
    # Source quality metrics (Enhancement 4)
    # -------------------------------------------------------------------------

    def get_source_quality_metrics(self, source_id: str) -> SourceQualityMetrics:
        """Return SourceQualityMetrics for a source. Computed lazily; cached."""
        if source_id in self._quality_cache:
            return self._quality_cache[source_id]

        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"Unknown source_id: {source_id!r}")

        record = source.record
        time_arr = record.waveform_data.get("time")
        if time_arr is None or len(time_arr) == 0:
            metrics = SourceQualityMetrics(
                source_id=source_id,
                sample_count=0,
                inferred_sample_rate_hz=0.0,
                sample_rate_stability=0.0,
                missing_data_pct=100.0,
                duplicate_timestamp_pct=0.0,
                interpolated_pct=0.0,
                resampling_ratio=1.0,
                time_is_uniform=False,
            )
        else:
            time_arr = np.asarray(time_arr, dtype=np.float64)
            # Pick the first analog channel's values for NaN analysis
            values_arr = None
            for ch in record.analog_channels:
                v = record.waveform_data.get(ch.name)
                if v is not None:
                    values_arr = np.asarray(v, dtype=np.float64)
                    break
            if values_arr is None:
                values_arr = np.zeros(len(time_arr), dtype=np.float64)

            metrics = alignment_engine.compute_source_quality(
                source_id, time_arr, values_arr
            )

        self._quality_cache[source_id] = metrics
        return metrics

    # -------------------------------------------------------------------------
    # Channel registry
    # -------------------------------------------------------------------------

    def _register_channels(
        self, source_id: str, record: "DisturbanceRecord"
    ) -> None:
        """Populate the channel registry from a newly added DisturbanceRecord."""
        for ch in record.analog_channels:
            panel_id, _ = _infer_panel_for_channel(
                ch.name,
                getattr(ch, "unit", None),
                getattr(ch, "parameter_type", None),
            )
            key = (source_id, ch.name)
            self._channels[key] = SessionChannel(
                source_id=source_id,
                channel_name=ch.name,
                channel_type="analog",
                display_name=ch.name,
                color_hex=None,
                line_style="solid",
                line_width=1.0,
                is_visible=True,
                panel_id=panel_id,
            )

        for ch in record.digital_channels:
            key = (source_id, ch.name)
            self._channels[key] = SessionChannel(
                source_id=source_id,
                channel_name=ch.name,
                channel_type="digital",
                display_name=ch.name,
                color_hex=None,
                line_style="solid",
                line_width=1.0,
                is_visible=True,
                panel_id=_PANEL_DIGITAL,
            )

    def get_channel(self, source_id: str, channel_name: str) -> SessionChannel | None:
        return self._channels.get((source_id, channel_name))

    def list_analog_channels(self, active_only: bool = True) -> list[SessionChannel]:
        active_ids = (
            {s.source_id for s in self._sources.values() if s.is_active}
            if active_only
            else set(self._sources.keys())
        )
        return [
            ch for ch in self._channels.values()
            if ch.channel_type == "analog" and ch.source_id in active_ids
        ]

    def list_digital_channels(self, active_only: bool = True) -> list[SessionChannel]:
        active_ids = (
            {s.source_id for s in self._sources.values() if s.is_active}
            if active_only
            else set(self._sources.keys())
        )
        return [
            ch for ch in self._channels.values()
            if ch.channel_type == "digital" and ch.source_id in active_ids
        ]

    def set_channel_display_name(
        self, source_id: str, channel_name: str, name: str
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.display_name = name

    def set_channel_colour(
        self, source_id: str, channel_name: str, color_hex: str | None
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.color_hex = color_hex

    def set_channel_visibility(
        self, source_id: str, channel_name: str, visible: bool
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.is_visible = visible

    def set_channel_line_style(
        self, source_id: str, channel_name: str, style: str
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.line_style = style

    def set_channel_line_width(
        self, source_id: str, channel_name: str, width: float
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.line_width = width

    def set_channel_y_axis_side(
        self, source_id: str, channel_name: str, side: str
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is not None:
            ch.y_axis_side = side

    def set_channel_panel(
        self, source_id: str, channel_name: str, panel_id: str
    ) -> None:
        ch = self._channels.get((source_id, channel_name))
        if ch is None:
            return
        old_panel_id = ch.panel_id
        ch.panel_id = panel_id
        # Update channel_refs in panels
        old_panel = self._panels.get(old_panel_id)
        if old_panel is not None:
            old_panel.channel_refs = [
                r for r in old_panel.channel_refs
                if r != (source_id, channel_name)
            ]
        new_panel = self._panels.get(panel_id)
        if new_panel is not None:
            ref = (source_id, channel_name)
            if ref not in new_panel.channel_refs:
                new_panel.channel_refs.append(ref)

    # -------------------------------------------------------------------------
    # Panel layout
    # -------------------------------------------------------------------------

    def list_panels(self) -> list[PanelConfig]:
        return list(self._panels.values())

    def add_panel(self, title: str, panel_type: str = "analog") -> str:
        panel_id = str(uuid.uuid4())
        self._panels[panel_id] = PanelConfig(
            panel_id=panel_id,
            title=title,
            channel_refs=[],
            panel_type=panel_type,
            is_visible=True,
        )
        return panel_id

    def remove_panel(self, panel_id: str) -> None:
        self._panels.pop(panel_id, None)

    def set_panel_visible(self, panel_id: str, visible: bool) -> None:
        panel = self._panels.get(panel_id)
        if panel is not None:
            panel.is_visible = visible

    def rename_panel(self, panel_id: str, title: str) -> None:
        panel = self._panels.get(panel_id)
        if panel is not None:
            panel.title = title

    def merge_panels(self, panel_id_a: str, panel_id_b: str) -> str:
        """Merge panel B into panel A. Returns panel_id_a.

        All channel_refs from B are appended to A (deduped).
        Panel B is removed. Channels previously in B are reassigned to A.
        """
        panel_a = self._panels.get(panel_id_a)
        panel_b = self._panels.get(panel_id_b)
        if panel_a is None or panel_b is None:
            raise KeyError(
                f"Cannot merge: panel_id_a={panel_id_a!r}, panel_id_b={panel_id_b!r}"
            )
        existing_refs = set(panel_a.channel_refs)
        for ref in panel_b.channel_refs:
            if ref not in existing_refs:
                panel_a.channel_refs.append(ref)
                existing_refs.add(ref)
            # Update channel registry panel assignment
            ch = self._channels.get(ref)
            if ch is not None:
                ch.panel_id = panel_id_a

        del self._panels[panel_id_b]
        return panel_id_a

    def default_layout(self) -> None:
        """Rebuild panel layout by grouping channels by type across all sources.

        Clears existing panels and reassigns all channels to canonical panels.
        """
        self._panels.clear()

        # Build unit + type lookup from source records
        unit_lookup: dict[tuple[str, str], str | None] = {}
        type_lookup: dict[tuple[str, str], str | None] = {}
        for src_id, source in self._sources.items():
            for ach in source.record.analog_channels:
                key = (src_id, ach.name)
                unit_lookup[key] = getattr(ach, "unit", None)
                type_lookup[key] = getattr(ach, "parameter_type", None)

        # Determine which canonical panels are needed
        needed_panels: dict[str, str] = {}  # panel_id → panel_type
        for ch in self._channels.values():
            if ch.channel_type == "digital":
                needed_panels[_PANEL_DIGITAL] = "digital"
            else:
                key = (ch.source_id, ch.channel_name)
                pid, ptype = _infer_panel_for_channel(
                    ch.channel_name, unit_lookup.get(key), type_lookup.get(key)
                )
                needed_panels[pid] = ptype

        # Create panels in a fixed order
        panel_order = [
            _PANEL_VOLTAGE, _PANEL_CURRENT, _PANEL_POWER,
            _PANEL_FREQUENCY, _PANEL_DIGITAL, _PANEL_OTHER
        ]
        for pid in panel_order:
            if pid in needed_panels:
                self._panels[pid] = PanelConfig(
                    panel_id=pid,
                    title=_DEFAULT_PANEL_TITLES[pid],
                    channel_refs=[],
                    panel_type=needed_panels[pid],
                    is_visible=True,
                )

        # Assign channels to panels
        for ch in self._channels.values():
            if ch.channel_type == "digital":
                ch.panel_id = _PANEL_DIGITAL
            else:
                key = (ch.source_id, ch.channel_name)
                pid, _ = _infer_panel_for_channel(
                    ch.channel_name, unit_lookup.get(key), type_lookup.get(key)
                )
                ch.panel_id = pid
            panel = self._panels.get(ch.panel_id)
            if panel is not None:
                ref = (ch.source_id, ch.channel_name)
                if ref not in panel.channel_refs:
                    panel.channel_refs.append(ref)

    # -------------------------------------------------------------------------
    # Data access for rendering
    # -------------------------------------------------------------------------

    def build_aligned_data(
        self,
        source_id: str,
        channel_name: str,
        t_start: float,
        t_end: float,
        max_points: int = 4000,
    ) -> AlignedChannelData:
        """Produce a render-ready AlignedChannelData for one channel.

        Steps:
        1. Retrieve raw time and values from DisturbanceRecord (no mutation).
        2. Apply time_offset_s to the time array (new array, original unchanged).
        3. Clip to [t_start, t_end].
        4. If points > max_points: decimate using decimate_for_display().
           If within a max_points grid: pass through as-is.
        5. Populate time_is_uniform from assess_time_uniformity on the raw time.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"Unknown source_id: {source_id!r}")

        record = source.record
        raw_time = record.waveform_data.get("time")
        raw_values = record.waveform_data.get(channel_name)

        # Determine unit from channel metadata
        unit: str | None = None
        ch_meta = self.get_channel(source_id, channel_name)
        channel_type = "analog"
        if ch_meta is not None:
            channel_type = ch_meta.channel_type

        for ch in record.analog_channels:
            if ch.name == channel_name:
                unit = ch.unit or None
                break

        if raw_time is None or raw_values is None or len(raw_time) == 0:
            return AlignedChannelData(
                source_id=source_id,
                channel_name=channel_name,
                time=np.array([], dtype=np.float64),
                values=np.array([], dtype=np.float64),
                original_sample_rate_hz=0.0,
                time_offset_s=source.time_offset_s,
                unit=unit,
                time_is_uniform=True,
            )

        # Convert DataFrame columns (Series) → ndarray after the None guard
        raw_time = np.asarray(raw_time, dtype=np.float64)
        raw_values = np.asarray(raw_values, dtype=np.float64)

        # Assess uniformity on the raw (pre-offset) time array
        is_uniform, _ = alignment_engine.assess_time_uniformity(raw_time)

        # Infer original sample rate from raw time (median interval)
        diffs = np.diff(raw_time.astype(np.float64))
        pos_diffs = diffs[diffs > 0]
        original_rate = (
            float(1.0 / np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0
        )

        # Step 1+2: apply offset — returns new array, raw_time unchanged
        shifted_time = alignment_engine.apply_time_offset(
            raw_time.astype(np.float64), source.time_offset_s
        )

        # Step 3: clip to [t_start, t_end]
        mask = (shifted_time >= t_start) & (shifted_time <= t_end)
        clipped_time = shifted_time[mask]
        clipped_values = raw_values.astype(np.float64)[mask]

        if len(clipped_time) == 0:
            return AlignedChannelData(
                source_id=source_id,
                channel_name=channel_name,
                time=np.array([], dtype=np.float64),
                values=np.array([], dtype=np.float64),
                original_sample_rate_hz=original_rate,
                time_offset_s=source.time_offset_s,
                unit=unit,
                time_is_uniform=is_uniform,
            )

        # Step 4: decimate if needed
        if len(clipped_time) > max_points:
            from app.visualization.rendering.downsampling import decimate_for_display
            out_time, out_values = decimate_for_display(
                clipped_time, clipped_values, t_start, t_end, max_points
            )
        else:
            out_time = clipped_time
            out_values = clipped_values

        return AlignedChannelData(
            source_id=source_id,
            channel_name=channel_name,
            time=out_time,
            values=out_values,
            original_sample_rate_hz=original_rate,
            time_offset_s=source.time_offset_s,
            unit=unit,
            time_is_uniform=is_uniform,
        )

    # -------------------------------------------------------------------------
    # Calculated Signals — analog-input eligibility (Phase 2C-1)
    # -------------------------------------------------------------------------
    #
    # This is the ONLY place Calculated Signals eligibility is decided. It
    # never resolves waveform arrays (that is Phase 2C-2's job, producing
    # ResolvedAnalogInput) -- it only answers "can this ChannelRef be used
    # as a Calculated Signals input right now", using analog/digital LIST
    # MEMBERSHIP as the primary type contract (not parameter_type, not
    # unit, not DataFrame dtype, not channel name), matching the Phase 1
    # architecture assessment's finding.

    def check_calculated_input_eligibility(self, ref: ChannelRef) -> ChannelEligibilityResult:
        """Return whether *ref* is currently usable as a Calculated Signals
        input, and if not, exactly why.

        A channel declared in BOTH analog_channels and digital_channels
        (an inconsistent record) is treated as DIGITAL_CHANNEL, not VALID
        -- the digital check is deliberately evaluated first, so an
        analog/digital name collision can never be silently accepted as an
        analog input.
        """
        source = self._sources.get(ref.source_id)
        if source is None:
            return ChannelEligibilityResult(ref, ChannelEligibility.MISSING_SOURCE)
        if not source.is_active:
            return ChannelEligibilityResult(ref, ChannelEligibility.INACTIVE_SOURCE)

        record = source.record
        is_digital = any(ch.name == ref.channel_name for ch in record.digital_channels)
        if is_digital:
            return ChannelEligibilityResult(ref, ChannelEligibility.DIGITAL_CHANNEL)

        is_analog = any(ch.name == ref.channel_name for ch in record.analog_channels)
        if not is_analog:
            return ChannelEligibilityResult(ref, ChannelEligibility.MISSING_CHANNEL)

        column = record.waveform_data.get(ref.channel_name)
        if column is None:
            return ChannelEligibilityResult(ref, ChannelEligibility.DATA_COLUMN_MISSING)
        if not pd.api.types.is_numeric_dtype(column):
            return ChannelEligibilityResult(ref, ChannelEligibility.NON_NUMERIC_CHANNEL)

        return ChannelEligibilityResult(ref, ChannelEligibility.VALID)

    def is_valid_calculated_input(self, ref: ChannelRef) -> bool:
        """Convenience wrapper around check_calculated_input_eligibility()."""
        return self.check_calculated_input_eligibility(ref).is_valid

    def get_dependency_status(self, calc_id: str) -> DependencyStatus:
        """Return the current, freshly-computed resolvability of every
        dependency of *calc_id* -- never cached, so a source becoming
        inactive or being removed is reflected immediately on the next call.

        Returns an all-True/empty DependencyStatus for an unknown calc_id
        (no dependencies to report), matching this module's existing
        "unknown id → empty/no-op" convention (see get_time_offset).
        """
        deps = self.get_calculated_dependencies(calc_id)
        missing_sources: list[str] = []
        inactive_sources: list[str] = []
        missing_channels: list[ChannelRef] = []
        digital_channels: list[ChannelRef] = []

        for ref in deps:
            result = self.check_calculated_input_eligibility(ref)
            if result.eligibility == ChannelEligibility.MISSING_SOURCE:
                missing_sources.append(ref.source_id)
            elif result.eligibility == ChannelEligibility.INACTIVE_SOURCE:
                inactive_sources.append(ref.source_id)
            elif result.eligibility == ChannelEligibility.DIGITAL_CHANNEL:
                digital_channels.append(ref)
            elif result.eligibility in (
                ChannelEligibility.MISSING_CHANNEL,
                ChannelEligibility.DATA_COLUMN_MISSING,
                ChannelEligibility.NON_NUMERIC_CHANNEL,
            ):
                missing_channels.append(ref)

        is_resolvable = not (missing_sources or inactive_sources or missing_channels or digital_channels)
        return DependencyStatus(
            is_resolvable=is_resolvable,
            missing_sources=tuple(dict.fromkeys(missing_sources)),
            inactive_sources=tuple(dict.fromkeys(inactive_sources)),
            missing_channels=tuple(missing_channels),
            digital_channels=tuple(digital_channels),
        )

    def _validate_calculated_dependencies(self, deps: Iterable[ChannelRef]) -> None:
        """Raise ValueError on the first invalid dependency found.

        Checked separately and first: a ChannelRef whose source_id names a
        calculated signal (not a real SessionSource) always fails with a
        specific message -- Calculated Signals V1 does not support
        calculated-signal-to-calculated-signal dependencies, and this is a
        clearer diagnostic than letting it fall through to the generic
        MISSING_SOURCE case (which would also technically reject it, since
        calculated signals are never stored in self._sources).
        """
        for ref in deps:
            if ref.source_id in self._calc_signals:
                raise ValueError(
                    f"calculated signal dependency {ref!r} refers to another "
                    "calculated signal; Calculated Signals V1 does not support "
                    "calculated-signal-to-calculated-signal dependencies"
                )
            result = self.check_calculated_input_eligibility(ref)
            if not result.is_valid:
                raise ValueError(
                    f"calculated signal dependency {ref!r} is not a valid "
                    f"analog input: {result.eligibility.value}"
                )

    def _register_calculated_dependencies(self, calc_id: str, deps: frozenset[ChannelRef]) -> None:
        self._calc_dependencies[calc_id] = deps
        for ref in deps:
            self._calc_dependents_by_source.setdefault(ref.source_id, set()).add(calc_id)
            self._calc_dependents_by_channel.setdefault(ref, set()).add(calc_id)

    def _unregister_calculated_dependencies(self, calc_id: str, deps: Iterable[ChannelRef]) -> None:
        for ref in deps:
            by_source = self._calc_dependents_by_source.get(ref.source_id)
            if by_source is not None:
                by_source.discard(calc_id)
                if not by_source:
                    del self._calc_dependents_by_source[ref.source_id]
            by_channel = self._calc_dependents_by_channel.get(ref)
            if by_channel is not None:
                by_channel.discard(calc_id)
                if not by_channel:
                    del self._calc_dependents_by_channel[ref]

    # -------------------------------------------------------------------------
    # Calculated Signals — CRUD (Phase 2C-1)
    # -------------------------------------------------------------------------
    #
    # No UI, no Session Canvas rendering, no persistence, no automatic or
    # lazy recalculation, and no calculated-signal-to-calculated-signal
    # dependencies exist anywhere in this section — all of that is deferred
    # to later phases. A calculated signal here is purely session-owned
    # lifecycle state: it is never a SessionSource and never touches (or is
    # merged into) any DisturbanceRecord.

    # Fields whose change makes an existing result numerically stale --
    # name/output metadata that doesn't affect the computation itself
    # (currently just `name`) is deliberately excluded.
    _MEANINGFUL_DEFINITION_FIELDS = (
        "expression", "variable_bindings", "reference_variable",
        "output_unit", "interpolation",
    )

    def add_calculated_signal(
        self,
        definition: CalculatedSignalDefinition,
        result: CalculatedSignalResult | None = None,
    ) -> str:
        """Register a new calculated signal, owned by this session.

        Validates every dependency (existence, active source, analog-only)
        before committing anything -- on any validation failure, session
        state (including the dependency indexes) is left completely
        unchanged; nothing is registered.

        The definition supplies its own calc_id (Phase 2A's contract: a
        CalculatedSignalDefinition always already has one) -- this method
        never generates one, unlike add_source()'s source_id.
        """
        calc_id = definition.calc_id
        if calc_id in self._calc_signals:
            raise ValueError(f"calc_id {calc_id!r} already exists in this session")
        if any(entry.definition.name == definition.name for entry in self._calc_signals.values()):
            raise ValueError(
                f"a calculated signal named {definition.name!r} already exists in this session"
            )
        if result is not None and result.calc_id != calc_id:
            raise ValueError(
                f"result.calc_id {result.calc_id!r} does not match definition.calc_id {calc_id!r}"
            )

        deps = frozenset(definition.variable_bindings.values())
        self._validate_calculated_dependencies(deps)

        # All validation passed -- commit atomically.
        self._calc_signals[calc_id] = CalculatedSignalEntry(definition=definition, result=result)
        self._register_calculated_dependencies(calc_id, deps)
        return calc_id

    def get_calculated_signal(self, calc_id: str) -> CalculatedSignalEntry | None:
        return self._calc_signals.get(calc_id)

    def list_calculated_signals(self) -> list[CalculatedSignalEntry]:
        return list(self._calc_signals.values())

    def get_calculated_signal_definition(self, calc_id: str) -> CalculatedSignalDefinition | None:
        entry = self._calc_signals.get(calc_id)
        return entry.definition if entry is not None else None

    def get_calculated_signal_result(self, calc_id: str) -> CalculatedSignalResult | None:
        entry = self._calc_signals.get(calc_id)
        return entry.result if entry is not None else None

    def update_calculated_signal_definition(
        self,
        calc_id: str,
        new_definition: CalculatedSignalDefinition,
    ) -> None:
        """Replace calc_id's definition, validating first.

        calc_id itself cannot change (new_definition.calc_id must match).
        The new name must remain unique among calculated signals. New
        dependencies are validated exactly as in add_calculated_signal().
        On any failure, the existing definition/result/dependency indexes
        are left completely unchanged.

        If the change is "meaningful" (expression, variable_bindings,
        reference_variable, output_unit, or interpolation differs from the
        current definition) and a result currently exists, that result is
        marked STALE (its arrays are preserved, not discarded or
        recomputed) -- see _MEANINGFUL_DEFINITION_FIELDS and
        mark_calculated_signal_stale(). A name-only edit never staled an
        existing result.
        """
        old_entry = self._calc_signals.get(calc_id)
        if old_entry is None:
            raise KeyError(f"Unknown calculated signal calc_id: {calc_id!r}")
        if new_definition.calc_id != calc_id:
            raise ValueError(
                f"new_definition.calc_id {new_definition.calc_id!r} does not match "
                f"the calc_id being updated {calc_id!r}; calc_id cannot change"
            )

        old_definition = old_entry.definition
        for other_id, entry in self._calc_signals.items():
            if other_id != calc_id and entry.definition.name == new_definition.name:
                raise ValueError(
                    f"a calculated signal named {new_definition.name!r} already exists in this session"
                )

        new_deps = frozenset(new_definition.variable_bindings.values())
        self._validate_calculated_dependencies(new_deps)

        # All validation passed -- commit atomically.
        meaningfully_changed = any(
            getattr(old_definition, field_name) != getattr(new_definition, field_name)
            for field_name in self._MEANINGFUL_DEFINITION_FIELDS
        )
        new_result = old_entry.result
        if meaningfully_changed and new_result is not None:
            new_result = self._stale_copy(new_result, reason="Definition changed")

        old_deps = self._calc_dependencies.get(calc_id, frozenset())
        self._unregister_calculated_dependencies(calc_id, old_deps)
        self._register_calculated_dependencies(calc_id, new_deps)
        self._calc_signals[calc_id] = CalculatedSignalEntry(definition=new_definition, result=new_result)

    def set_calculated_signal_result(self, calc_id: str, result: CalculatedSignalResult) -> None:
        """Store a freshly computed result for calc_id (Phase 2C-2's entry
        point). No dependency changes, no UI action, no recalculation --
        purely replaces the stored result.
        """
        entry = self._calc_signals.get(calc_id)
        if entry is None:
            raise KeyError(f"Unknown calculated signal calc_id: {calc_id!r}")
        if result.calc_id != calc_id:
            raise ValueError(f"result.calc_id {result.calc_id!r} does not match {calc_id!r}")
        entry.result = result

    def set_calculated_signal_visible(self, calc_id: str, visible: bool) -> None:
        """Set a calculated signal's session/UI display visibility (Phase 3B).

        Pure display state, mirroring set_channel_visibility() for real
        channels: never touches the definition, the result, or dependency
        indexes. Hiding does not delete anything; showing again restores
        the existing result without recalculation. A no-op for an unknown
        calc_id, matching this module's existing "unknown id -> no-op"
        convention (see set_channel_visibility).
        """
        entry = self._calc_signals.get(calc_id)
        if entry is not None:
            entry.is_visible = visible

    def mark_calculated_signal_stale(self, calc_id: str, reason: str | None = None) -> None:
        """Mark calc_id's latest result STALE without discarding its arrays.

        State transitions (Phase 2C-2 correction -- see below):
          OK    -> STALE
          STALE -> STALE (reason appended if not already present; see _stale_copy)
          ERROR -> ERROR, unchanged (a no-op; see note)
          None  -> None  (a no-op; "not yet calculated" is already honest)

        Raises for an unknown calc_id, matching set_calculated_signal_result().

        Behaviour correction from the initial Phase 2C-1 implementation:
        that version unconditionally converted ANY existing result
        (including ERROR) to STALE. This method now leaves an ERROR result
        untouched instead -- an error result is not a valid numerical
        result whose "freshness" can meaningfully become stale; it already
        communicates "the last calculation attempt failed", which is a
        strictly more specific and more useful fact than "outdated". This
        correction is covered by dedicated tests (test_calculated_signal_
        session.py's TestLifecycleInvalidation.test_error_result_stays_
        error_under_source_mutation and the direct mark_calculated_signal_
        stale ERROR test) rather than changed silently.
        """
        entry = self._calc_signals.get(calc_id)
        if entry is None:
            raise KeyError(f"Unknown calculated signal calc_id: {calc_id!r}")
        if entry.result is None:
            return
        if entry.result.status == CalculationStatus.ERROR:
            return
        entry.result = self._stale_copy(entry.result, reason)

    def _stale_copy(
        self, result: CalculatedSignalResult, reason: str | None
    ) -> CalculatedSignalResult:
        """Return a new CalculatedSignalResult with status=STALE, reusing
        (not copying-before-passing) the existing time/values/validity_mask
        arrays -- CalculatedSignalResult's own constructor already performs
        exactly one defensive copy of each array, so passing them directly
        here avoids a redundant second copy.

        error_message is intentionally not carried over: STALE means "these
        arrays are still meaningful but may no longer reflect current
        inputs", which is a different claim from ERROR's "the last
        calculation attempt failed" -- carrying a leftover error_message
        into a STALE result would conflate the two.

        Warning deduplication (Phase 2C-2): the exact reason string is
        appended only if it is not already the last-known occurrence in
        `warnings` -- this keeps e.g. 20 identical repeated offset-drag
        events from appending 20 copies of "Marked stale: Source alignment
        offset changed." while still allowing genuinely different reasons
        (e.g. a later "Source removed.") to coexist. Pre-existing numerical
        warnings from the calculation engine are never touched or removed.
        """
        warnings = list(result.warnings)
        if reason:
            entry_text = f"Marked stale: {reason}"
            if entry_text not in warnings:
                warnings.append(entry_text)
        return CalculatedSignalResult(
            calc_id=result.calc_id,
            time=result.time,
            values=result.values,
            validity_mask=result.validity_mask,
            unit=result.unit,
            status=CalculationStatus.STALE,
            error_message=None,
            computed_at=result.computed_at,
            warnings=warnings,
        )

    def remove_calculated_signal(self, calc_id: str) -> None:
        """Remove calc_id's definition/result and all dependency-index
        references. A no-op for an unknown calc_id, matching
        remove_source()'s style. Never touches any SessionSource/
        DisturbanceRecord.
        """
        if calc_id not in self._calc_signals:
            return
        deps = self._calc_dependencies.pop(calc_id, frozenset())
        self._unregister_calculated_dependencies(calc_id, deps)
        del self._calc_signals[calc_id]

    # -------------------------------------------------------------------------
    # Calculated Signals — dependency queries (Phase 2C-1)
    # -------------------------------------------------------------------------

    def get_calculated_dependencies(self, calc_id: str) -> tuple[ChannelRef, ...]:
        """Return calc_id's dependencies in deterministic (source_id,
        channel_name) order. Empty tuple for an unknown calc_id.
        """
        deps = self._calc_dependencies.get(calc_id)
        if not deps:
            return ()
        return tuple(sorted(deps, key=lambda r: (r.source_id, r.channel_name)))

    def get_calculated_dependents_for_source(self, source_id: str) -> tuple[str, ...]:
        """Return, in sorted order, the calc_ids of every calculated signal
        that depends on any channel of *source_id* -- e.g. to discover the
        impact of removing or deactivating that source.
        """
        return tuple(sorted(self._calc_dependents_by_source.get(source_id, ())))

    def get_calculated_dependents_for_channel(self, ref: ChannelRef) -> tuple[str, ...]:
        """Return, in sorted order, the calc_ids of every calculated signal
        that depends on this exact (source_id, channel_name).
        """
        return tuple(sorted(self._calc_dependents_by_channel.get(ref, ())))

    def get_stale_calculated_signal_ids(self) -> tuple[str, ...]:
        """Return, in deterministic (insertion) order, the calc_ids whose
        latest result currently has status=STALE.

        Convenience for a future recalculation controller so it does not
        need to iterate list_calculated_signals() and inspect
        entry.result.status itself. Deliberately narrow: result=None
        ("never calculated") is a distinct state from STALE and is not
        included here -- a controller that also wants to compute
        never-calculated signals should query list_calculated_signals()
        directly.
        """
        return tuple(
            calc_id for calc_id, entry in self._calc_signals.items()
            if entry.result is not None and entry.result.status == CalculationStatus.STALE
        )

    # -------------------------------------------------------------------------
    # Calculated Signals — canvas panel routing (Phase 3B)
    # -------------------------------------------------------------------------
    #
    # A calculated signal is never a SessionChannel and is never added to any
    # PanelConfig.channel_refs -- that list is real-channel identity that
    # other panel operations (default_layout, merge/split, drag-and-drop
    # channel moves) assume resolves through session._channels. Instead, a
    # calculated signal's panel is looked up on demand from its own result,
    # reusing the exact same type/unit/name taxonomy _infer_panel_for_channel
    # already applies to real channels -- no second taxonomy is introduced.

    def ensure_calculated_signal_panel(self, calc_id: str) -> tuple[str, bool] | None:
        """Return (panel_id, created) for calc_id's current result, creating
        the canonical panel (Voltage/Current/Power/Frequency/Other Analog)
        if it does not already exist.

        Returns None when calc_id is unknown or has never been calculated
        (result is None) -- there is nothing to place yet. Uses only the
        calculated signal's display name and result.unit, exactly as
        directed for calculated-signal panel inference -- parameter_type is
        left unknown, and no engineering meaning is inferred from the
        expression text.
        """
        entry = self._calc_signals.get(calc_id)
        if entry is None or entry.result is None:
            return None
        panel_id, panel_type = _infer_panel_for_channel(
            entry.definition.name, entry.result.unit, None
        )
        if panel_id in self._panels:
            return panel_id, False
        self._panels[panel_id] = PanelConfig(
            panel_id=panel_id,
            title=_DEFAULT_PANEL_TITLES.get(panel_id, panel_id.title()),
            channel_refs=[],
            panel_type=panel_type,
            is_visible=True,
        )
        return panel_id, True

    # -------------------------------------------------------------------------
    # Calculated Signals — lifecycle invalidation (Phase 2C-2)
    # -------------------------------------------------------------------------
    #
    # EventAnalysisSession's responsibility ends at "know that a result is
    # no longer fresh" -- it never recomputes. Nothing in this section (or
    # anywhere in this file) calls calculate_signal(), resolves a ChannelRef
    # into a ResolvedAnalogInput, or touches waveform arrays; the numerical
    # engine remains fully independent, to be invoked by a separate
    # controller/service in a later phase.

    def _mark_calculated_dependents_stale_for_source(
        self,
        source_id: str,
        *,
        reason: str,
    ) -> tuple[str, ...]:
        """Mark every calculated signal that depends on any channel of
        *source_id* STALE, and return the affected calc_ids (deterministic
        order, from get_calculated_dependents_for_source).

        Reuses the existing reverse-dependency index and
        mark_calculated_signal_stale()'s own status-transition/dedup rules
        verbatim -- this helper does not construct or duplicate a
        CalculatedSignalResult itself, and performs no recalculation.
        """
        affected = self.get_calculated_dependents_for_source(source_id)
        for calc_id in affected:
            self.mark_calculated_signal_stale(calc_id, reason=reason)
        return affected
