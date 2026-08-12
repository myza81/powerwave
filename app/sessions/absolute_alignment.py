"""Absolute-time alignment policy (Stage 1).

Derives ``SessionSource.time_offset_s`` from the absolute ``start_time`` each
provider already parses, so that records captured by different devices are
placed on one physical time coordinate.

What this module does NOT do
----------------------------
- It never mutates a ``DisturbanceRecord``, and in particular never touches
  ``waveform_data["time"]`` — that array stays in its own elapsed-seconds
  representation forever. Alignment is a scalar per source, applied lazily by
  ``EventAnalysisSession.build_aligned_data()``.
- It never resamples, interpolates, or builds a common grid.
- It never claims that two device clocks were *verified* to agree. Deriving an
  offset from recorded timestamps is "timestamp alignment"; proving the clocks
  matched is "clock synchronization", and no metadata any current provider
  produces can establish the latter. ``assess_session_timing_compatibility()``
  keeps reporting its own verdict independently — a pair of timezone-naive
  records stays ``requires_review`` even after being aligned here.

Stable session origin
---------------------
The session's coordinate origin (the wall-clock instant that session x = 0
refers to) is established ONCE, from the earliest eligible source present when
absolute alignment first becomes possible, and then held stable. A source
loaded later that starts *earlier* than the origin receives a NEGATIVE offset
rather than moving the origin.

That stability is the point: re-deriving the origin on every source add would
silently re-place every already-positioned waveform, invalidating manual,
trigger-based and correlation offsets the analyst had already committed, and
shifting the coordinate system underneath existing calculated signals. Only an
explicit analyst action (Set as Reference) may move the origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Sequence

from app.sessions.timing_compatibility import (
    TimingReferenceClass,
    classify_timing_reference,
)

if TYPE_CHECKING:
    from app.sessions.session_models import SessionSource

# The alignment_method recorded on a source whose offset this module derived.
# Names the EVIDENCE used (the recorded start_time), matching the existing
# vocabulary's convention where 'auto_trigger' names the trigger heuristic and
# 'correlation' names cross-correlation. Deliberately not "wall_clock" or
# "synchronized", which would overclaim — see the module docstring.
ABSOLUTE_TIMESTAMP = "absolute_timestamp"

# Only these two classes carry a start_time that is a genuine recording
# timestamp. classify_timing_reference() already downgrades a record whose
# timing_reference says "absolute" but whose anchor is the known
# datetime(2000, 1, 1) no-real-timestamp sentinel to ELAPSED_UNANCHORED, so
# that check is inherited here rather than duplicated.
_ELIGIBLE_CLASSES = frozenset(
    {TimingReferenceClass.ABSOLUTE_AWARE, TimingReferenceClass.ABSOLUTE_NAIVE}
)

# Methods whose offset this module is allowed to write. Anything else is
# analyst-owned ('manual' typed/dragged, 'auto_trigger' heuristic,
# 'correlation' cross-correlation, 'imported' from a manifest) and is left
# exactly as the analyst left it. ABSOLUTE_TIMESTAMP is derivable so that
# re-running the policy is idempotent.
DERIVABLE_METHODS = frozenset({"none", ABSOLUTE_TIMESTAMP})

# Absolute alignment only becomes meaningful once two sources must share one
# coordinate system. With fewer than this, a session keeps today's behaviour
# exactly: offset 0.0, method 'none', no origin.
MIN_ELIGIBLE_SOURCES = 2


@dataclass(frozen=True, slots=True)
class AbsoluteAlignmentPlan:
    """What one alignment pass decided. Purely descriptive: applying it is
    ``EventAnalysisSession.apply_absolute_alignment()``'s job.
    """

    origin: datetime | None
    """Wall-clock instant that session x = 0 denotes, or None when no origin
    has been (or could be) established."""

    origin_established: bool
    """True only on the pass that first created the origin — useful for UI
    messaging; False on every later idempotent pass."""

    offsets: dict[str, float] = field(default_factory=dict)
    """source_id → derived offset in seconds. Only derivable eligible sources
    appear here."""

    skipped_ineligible: tuple[str, ...] = ()
    """source_ids with no trustworthy absolute anchor (elapsed, synthetic,
    sample-index, epoch sentinel, or missing start_time)."""

    skipped_analyst_owned: tuple[str, ...] = ()
    """source_ids that are eligible but whose alignment the analyst owns."""

    clock_verified: bool = False
    """Always False. Present so callers state the assumption explicitly rather
    than implying verified clock synchronization. See the module docstring."""

    @property
    def derived_any(self) -> bool:
        """True when this pass derived an offset for at least one source.

        Deliberately not named "changed": the policy is idempotent, so a
        re-run over an unchanged source set derives the same values again and
        this stays True even though nothing moved. Callers wanting "did
        anything actually move" must compare offsets themselves.
        """
        return bool(self.offsets)


def is_eligible(source: "SessionSource") -> bool:
    """True when *source* carries a trustworthy absolute recording anchor.

    Inactive sources are still eligible: activity controls what is drawn and
    analysed, not whether a record's timestamps are meaningful, and a source
    must not silently change coordinates when it is toggled back on.
    """
    timing = getattr(source.record, "timing_info", None)
    if timing is None or timing.start_time is None:
        return False
    return classify_timing_reference(timing) in _ELIGIBLE_CLASSES


def eligible_sources(sources: Iterable["SessionSource"]) -> list["SessionSource"]:
    return [s for s in sources if is_eligible(s)]


def derive_offset(start_time: datetime, origin: datetime) -> float:
    """Seconds to add to a source's own elapsed axis to place it in session time."""
    return (start_time - origin).total_seconds()


def plan_absolute_alignment(
    sources: Sequence["SessionSource"],
    current_origin: datetime | None,
) -> AbsoluteAlignmentPlan:
    """Decide the origin and per-source offsets for the current source set.

    When *current_origin* is None, an origin is established only once at least
    MIN_ELIGIBLE_SOURCES eligible sources exist. When it is not None it is
    reused verbatim — a later, earlier-starting source yields a negative offset
    instead of moving the origin (see the module docstring).
    """
    all_sources = list(sources)
    eligible = eligible_sources(all_sources)
    eligible_ids = {s.source_id for s in eligible}
    ineligible = tuple(s.source_id for s in all_sources if s.source_id not in eligible_ids)

    origin = current_origin
    established = False
    if origin is None:
        if len(eligible) < MIN_ELIGIBLE_SOURCES:
            # Nothing was skipped for analyst-ownership reasons here -- there
            # simply are not yet two anchored sources to share a coordinate
            # system. origin=None / origin_established=False already says that,
            # so reporting these sources as analyst-owned would be false.
            return AbsoluteAlignmentPlan(
                origin=None,
                origin_established=False,
                skipped_ineligible=ineligible,
            )
        origin = min(s.record.timing_info.start_time for s in eligible)
        established = True

    offsets: dict[str, float] = {}
    analyst_owned: list[str] = []
    for source in eligible:
        if source.alignment_method not in DERIVABLE_METHODS:
            analyst_owned.append(source.source_id)
            continue
        offsets[source.source_id] = derive_offset(
            source.record.timing_info.start_time, origin
        )

    return AbsoluteAlignmentPlan(
        origin=origin,
        origin_established=established,
        offsets=offsets,
        skipped_ineligible=ineligible,
        skipped_analyst_owned=tuple(analyst_owned),
    )
