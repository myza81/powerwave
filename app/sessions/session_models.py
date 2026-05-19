"""Session data models for Phase 9A — Event Analysis Session.

All models are pure Python dataclasses with no Qt dependency.
DisturbanceRecord objects are held by reference; waveform_data is never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.models.disturbance_record import DisturbanceRecord

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
