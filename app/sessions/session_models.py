"""Session data models for Phase 9A — Event Analysis Session.

All models are pure Python dataclasses with no Qt dependency.
DisturbanceRecord objects are held by reference; waveform_data is never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.models.disturbance_record import DisturbanceRecord
    from app.calculated_signals.models import (
        CalculatedSignalDefinition,
        CalculatedSignalResult,
        ChannelRef,
    )

# ---------------------------------------------------------------------------
# Alignment method vocabulary
# ---------------------------------------------------------------------------
# 'none'         — no offset applied (default, time_offset_s == 0.0)
# 'manual'       — analyst typed or dragged the offset
# 'auto_trigger' — detect_trigger_time() heuristic
# 'correlation'  — cross-correlation against reference (future)
# 'imported'     — offset loaded from a saved session or manifest
ALIGNMENT_METHODS = frozenset(
    {"none", "manual", "auto_trigger", "correlation", "imported"}
)


@dataclass
class SessionSource:
    """One loaded recording within an EventAnalysisSession."""

    source_id: str                   # UUID, unique within session
    display_name: str                # user-editable; defaults to filename
    record: "DisturbanceRecord"      # held by reference, never mutated
    provider_type: str               # 'comtrade' | 'csv' | 'excel'
    origin_path: str | None          # original file path, informational only
    time_offset_s: float             # view-only shift; positive = shift right
    is_active: bool                  # False = excluded from canvas and analytics

    # Alignment provenance (Enhancement 1)
    alignment_method: str = "none"
    # [0.0, 1.0] for auto methods; None for manual/imported (not applicable)
    alignment_confidence: float | None = None


@dataclass
class SessionChannel:
    """One channel within a session, with user-editable display properties."""

    source_id: str
    channel_name: str        # canonical name from DisturbanceRecord
    channel_type: str        # 'analog' | 'digital'
    display_name: str        # user override; defaults to channel_name
    color_hex: str | None    # user override hex colour; None = auto-assign
    line_style: str          # 'solid' | 'dashed' | 'dotted'
    is_visible: bool
    panel_id: str            # which PanelConfig this channel is assigned to
    line_width: float = 1.0  # pen width in pixels (0.5 / 1.0 / 1.5 / 2.0 / 3.0)
    y_axis_side: str = "left"  # 'left' | 'right'


@dataclass
class PanelConfig:
    """Layout configuration for one display panel in the session canvas."""

    panel_id: str
    title: str
    channel_refs: list[tuple[str, str]] = field(default_factory=list)
    # list of (source_id, channel_name) — ordered display list
    panel_type: str = "analog"       # 'analog' | 'digital'
    is_visible: bool = True


@dataclass
class AlignedChannelData:
    """Output of EventAnalysisSession.build_aligned_data().

    time and values are already:
      - offset-shifted (time_offset_s applied)
      - clipped to [t_start, t_end]
      - resampled/decimated to ≤ max_points

    DisturbanceRecord.waveform_data is never mutated to produce this.
    """

    source_id: str
    channel_name: str
    time: np.ndarray            # float64 seconds, offset applied
    values: np.ndarray          # float64, resampled/decimated
    original_sample_rate_hz: float
    time_offset_s: float        # the offset that was applied (informational)
    unit: str | None

    # PMU / non-uniform sampling flag (Enhancement 3)
    time_is_uniform: bool = True
    # False when the source time array has jitter, dropouts, or missing frames.
    # Analytics layers that require uniform spacing (e.g. FFT) should check this
    # flag and issue a warning rather than silently producing wrong results.


@dataclass
class AlignmentResult:
    """Structured result from one auto-align operation (Enhancement 1)."""

    source_id: str
    suggested_offset_s: float
    alignment_method: str           # matches ALIGNMENT_METHODS vocabulary
    alignment_confidence: float | None
    reference_time: float | None    # trigger time detected in source's own axis
    notes: str                      # human-readable explanation for UI tooltip


@dataclass
class SourceQualityMetrics:
    """Per-source data quality summary (Enhancement 4).

    Computed lazily by EventAnalysisSession.get_source_quality_metrics().
    Exposed in the Session Panel as a quality bar + tooltip.
    """

    source_id: str
    sample_count: int
    inferred_sample_rate_hz: float
    sample_rate_stability: float    # [0.0, 1.0]; 1.0 = perfectly uniform
    missing_data_pct: float         # % of expected samples absent (NaN or gap)
    duplicate_timestamp_pct: float  # % of timestamps that are duplicated
    interpolated_pct: float         # % of output samples that were interpolated
    resampling_ratio: float         # output_rate / input_rate
    time_is_uniform: bool           # mirrors AlignedChannelData.time_is_uniform


# ---------------------------------------------------------------------------
# Calculated Signals — session ownership (Phase 2C-1)
# ---------------------------------------------------------------------------
#
# These models describe session-level lifecycle/dependency state only. The
# durable definition and the numeric result themselves are Phase 2A/2B
# models (app.calculated_signals.models.CalculatedSignalDefinition /
# CalculatedSignalResult) and are held here by reference, never duplicated
# or re-implemented.


class ChannelEligibility(str, Enum):
    """Why a ChannelRef is, or is not, usable as a Calculated Signals input.

    VALID is the only eligible outcome. Every other value names a specific,
    distinguishable reason a future UI or Phase 2C-2 recalculation step can
    surface directly to the user, per the Phase 1 architecture assessment's
    finding that analog/digital list membership -- not parameter_type, unit,
    dtype, or channel name -- is the primary type contract for a
    DisturbanceRecord's channels.
    """

    VALID = "valid"
    MISSING_SOURCE = "missing_source"
    INACTIVE_SOURCE = "inactive_source"
    MISSING_CHANNEL = "missing_channel"           # not declared analog or digital at all
    DIGITAL_CHANNEL = "digital_channel"            # declared digital (or both analog and digital)
    DATA_COLUMN_MISSING = "data_column_missing"    # declared analog, but waveform_data lacks the column
    NON_NUMERIC_CHANNEL = "non_numeric_channel"    # column present but not a numeric dtype


@dataclass(frozen=True, slots=True)
class ChannelEligibilityResult:
    """The result of checking one ChannelRef against the live session."""

    ref: "ChannelRef"
    eligibility: ChannelEligibility

    @property
    def is_valid(self) -> bool:
        return self.eligibility == ChannelEligibility.VALID


@dataclass(slots=True)
class DependencyStatus:
    """A calculated signal's current dependency resolvability, computed on
    demand by EventAnalysisSession.get_dependency_status() -- never cached,
    so it always reflects the session's current state (active/inactive
    sources, source removal, etc.), not the state at creation or last edit.
    """

    is_resolvable: bool
    missing_sources: tuple[str, ...] = ()
    inactive_sources: tuple[str, ...] = ()
    missing_channels: tuple["ChannelRef", ...] = ()
    digital_channels: tuple["ChannelRef", ...] = ()


@dataclass(slots=True)
class CalculatedSignalEntry:
    """Session-lifecycle state for one calculated signal.

    Wraps, without duplicating, the Phase 2A/2B models: `definition` is the
    durable, immutable CalculatedSignalDefinition; `result` is the latest
    CalculatedSignalResult, or None when the signal has never been
    calculated yet (a legitimate, honest state -- see
    EventAnalysisSession.add_calculated_signal). Dependency state is
    deliberately NOT stored on this entry (it would be a second, cache-able
    source of truth that could silently go stale relative to the session's
    actual current state, e.g. a source becoming inactive) -- see
    DependencyStatus and EventAnalysisSession.get_dependency_status(),
    which are always computed fresh from the session's live source/channel
    state.

    is_visible (Phase 3B) is pure session/UI display state, mirroring
    SessionChannel.is_visible for real channels -- hiding a calculated
    signal never touches its definition or result; it only controls whether
    a later render step (SessionCanvasController) paints its curve.
    """

    definition: "CalculatedSignalDefinition"
    result: "CalculatedSignalResult | None" = None
    is_visible: bool = True
