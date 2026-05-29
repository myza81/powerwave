"""EventAnalysisSession — multi-source analyst workspace for Phase 9A.

Key invariants:
- DisturbanceRecord.waveform_data is NEVER mutated.
- Time offsets are stored in SessionSource and applied lazily at render time.
- session_version and created_at are set once at construction and never changed.
- Qt-free: no PyQt6 imports anywhere in this module.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from app.sessions.session_models import (
    ALIGNMENT_METHODS,
    AlignedChannelData,
    PanelConfig,
    SessionChannel,
    SessionSource,
    SourceQualityMetrics,
)
from app.sessions import alignment_engine

if TYPE_CHECKING:
    from app.models.disturbance_record import DisturbanceRecord

# Panel ID constants for default_layout()
_PANEL_VOLTAGE = "voltage"
_PANEL_CURRENT = "current"
_PANEL_POWER = "power"
_PANEL_FREQUENCY = "frequency"
_PANEL_DIGITAL = "digital"
_PANEL_OTHER = "other"

# Channel-name heuristics for default panel assignment (mirrors channel_grouper logic)
_VOLTAGE_KEYWORDS = {"va", "vb", "vc", "vn", "vab", "vbc", "vca", "voltage", "volt", "kv"}
_CURRENT_KEYWORDS = {"ia", "ib", "ic", "in", "current", "amp"}
_POWER_KEYWORDS = {"mw", "mvar", "power", "p_", "q_", "apparent", "active", "reactive"}
_FREQ_KEYWORDS = {"freq", "hz", "rocof", "df"}

# Unit-based heuristics — exact match after lower+strip (more authoritative than name keywords)
_VOLTAGE_UNITS = {"v", "kv", "mv", "pu", "p.u.", "volt", "volts"}
_CURRENT_UNITS = {"a", "ka", "ma", "amp", "amps"}
_POWER_UNITS   = {"w", "kw", "mw", "gw", "var", "kvar", "mvar", "gvar", "va", "kva", "mva"}
_FREQ_UNITS    = {"hz", "rad/s", "hz/s", "rad/s2", "rad/s²"}

# ParameterType.value → panel_id (most authoritative — explicit user/import classification)
_TYPE_TO_PANEL: dict[str, str] = {
    "voltage":   _PANEL_VOLTAGE,
    "current":   _PANEL_CURRENT,
    "mw":        _PANEL_POWER,
    "mvar":      _PANEL_POWER,
    "frequency": _PANEL_FREQUENCY,
    "rocof":     _PANEL_FREQUENCY,
}


def _infer_panel_for_type(param_type: str | None) -> str | None:
    """Return panel_id from ParameterType string value, or None if not mapped."""
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
    lower = channel_name.lower()
    for kw in _VOLTAGE_KEYWORDS:
        if kw in lower:
            return _PANEL_VOLTAGE, "analog"
    for kw in _CURRENT_KEYWORDS:
        if kw in lower:
            return _PANEL_CURRENT, "analog"
    for kw in _POWER_KEYWORDS:
        if kw in lower:
            return _PANEL_POWER, "analog"
    for kw in _FREQ_KEYWORDS:
        if kw in lower:
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
        """Remove a source and all its channels from the session."""
        if source_id not in self._sources:
            return
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

    def set_source_active(self, source_id: str, active: bool) -> None:
        source = self._sources.get(source_id)
        if source is not None:
            source.is_active = active

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
        """
        if method not in ALIGNMENT_METHODS:
            raise ValueError(f"Unknown alignment_method {method!r}. "
                             f"Must be one of {sorted(ALIGNMENT_METHODS)}")
        source = self._sources.get(source_id)
        if source is not None:
            source.time_offset_s = float(offset_s)
            source.alignment_method = method
            source.alignment_confidence = confidence

    def get_time_offset(self, source_id: str) -> float:
        source = self._sources.get(source_id)
        return source.time_offset_s if source is not None else 0.0

    def reset_all_offsets(self) -> None:
        """Reset all source offsets to 0.0 and clear alignment metadata."""
        for source in self._sources.values():
            source.time_offset_s = 0.0
            source.alignment_method = "none"
            source.alignment_confidence = None
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
