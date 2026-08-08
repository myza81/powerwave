"""Session timing-reference compatibility detection (Sprint 1B).

Detection and transparency only -- this module never repairs, shifts,
reinterprets, or synchronizes anything. It classifies each active source's
existing ``TimingInformation`` (already produced by the provider/import
pipeline, never modified here) and reports whether the active sources'
displayed time axes can be safely compared, so the engineer can decide
what corrective action, if any, is appropriate.

Evidence used (verified against actual production assignments -- see the
Sprint 1B audit, not assumed from field names):

- ``TimingInformation.timing_reference`` is a plain string. Only the
  Import Wizard path (app.import_wizard.disturbance_record_bridge) ever
  sets it to a value other than the dataclass default "absolute":
  "relative_elapsed", "synthetic_elapsed", or "sample_index" (the exact
  vocabulary produced by app.import_wizard.timestamp_normalizer). COMTRADE
  and the direct CSV/Excel providers never set this field at all, so it
  is always the "absolute" default for those paths.
- "absolute" does NOT always mean "a genuine recording timestamp is
  available". app.providers.csv.csv_provider._build_time_array and
  app.providers.excel.excel_provider._build_time_array both fall back to
  a fixed sentinel start_time (2000-01-01 00:00:00) when no time column
  exists or none of it can be parsed as a datetime -- while leaving
  timing_reference at its "absolute" default. app.import_wizard.
  disturbance_record_bridge uses the identical sentinel for its own
  missing-timestamp-column fallback. COMTRADE's start_time/trigger_time,
  by contrast, always come from a successfully parsed CFG date/time line
  (app.providers.comtrade.comtrade_provider._parse_timestamp raises
  rather than falling back), so a COMTRADE record's "absolute" default is
  genuinely trustworthy -- this module must not mistake COMTRADE's
  elapsed-seconds waveform_data["time"] axis for an unanchored source
  merely because the data column itself holds small numbers; the anchor
  is what matters, not the column's literal values.
- ``TimingInformation.timezone`` is never set to a non-None value by any
  current provider (verified by repository-wide search), so every
  presently-producible record is timezone-naive when it does carry a
  real anchor. The AWARE branches below are implemented correctly for
  when this field is populated by some future path, not fabricated.

Nothing in this module mutates a DisturbanceRecord, a TimingInformation, a
SessionSource, or session offsets/panels/calculated signals. It only reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.timing import TimingInformation
    from app.sessions.event_session import EventAnalysisSession
    from app.sessions.session_models import SessionSource

# The exact sentinel value app.providers.csv.csv_provider,
# app.providers.excel.excel_provider, and
# app.import_wizard.disturbance_record_bridge all use (independently, but
# with the identical literal) when no genuine timestamp could be
# determined. Recognizing it is evidence, not a guess: it is the specific,
# deliberate value those three modules already emit for exactly this
# situation -- see this module's docstring.
_KNOWN_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)

# The explicit, self-describing timing_reference strings actually produced
# by app.import_wizard.timestamp_normalizer.normalize_timestamps(). Any
# other value (including the dataclass default "absolute") is evaluated
# using the anchor evidence in _has_real_anchor() rather than trusted at
# face value -- see module docstring.
_RELATIVE_ELAPSED = "relative_elapsed"
_SYNTHETIC_ELAPSED = "synthetic_elapsed"
_SAMPLE_INDEX = "sample_index"
_ABSOLUTE = "absolute"


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────


class TimingReferenceClass(str, Enum):
    """What one source's own timing metadata actually supports -- answers
    "what does this source's displayed time axis mean, and can it be
    compared with another source's?" independent of any session offset.
    """

    ABSOLUTE_AWARE = "absolute_aware"        # real anchor, explicit timezone
    ABSOLUTE_NAIVE = "absolute_naive"        # real anchor, no timezone declared
    ELAPSED_UNANCHORED = "elapsed_unanchored"  # elapsed axis, no real-world anchor
    RECONSTRUCTED = "reconstructed"          # anchor AND sample spacing both assumed
    SAMPLE_INDEX = "sample_index"            # pure row count, no time unit at all
    UNKNOWN = "unknown"                      # insufficient/unrecognized evidence


def _has_real_anchor(timing: "TimingInformation") -> bool:
    """True when start_time is a genuine recording timestamp rather than
    the known epoch-fallback sentinel emitted when no real timestamp was
    determined (see module docstring)."""
    start = timing.start_time
    if start is None:
        return False
    naive_start = start.replace(tzinfo=None) if start.tzinfo is not None else start
    return naive_start != _KNOWN_EPOCH_FALLBACK


def classify_timing_reference(timing: "TimingInformation") -> TimingReferenceClass:
    """Classify one TimingInformation's reference class from its existing
    fields alone. Read-only: never mutates *timing*.
    """
    raw = timing.timing_reference or _ABSOLUTE

    if raw == _SAMPLE_INDEX:
        return TimingReferenceClass.SAMPLE_INDEX
    if raw == _SYNTHETIC_ELAPSED:
        return TimingReferenceClass.RECONSTRUCTED
    if raw == _RELATIVE_ELAPSED:
        return TimingReferenceClass.ELAPSED_UNANCHORED
    if raw != _ABSOLUTE:
        # An unrecognized timing_reference string -- conservative fallback
        # rather than guessing what it might mean.
        return TimingReferenceClass.UNKNOWN

    if not _has_real_anchor(timing):
        # Declared "absolute" (the field's own default) but the anchor is
        # the known no-real-timestamp sentinel -- do not trust the label.
        return TimingReferenceClass.ELAPSED_UNANCHORED
    return (
        TimingReferenceClass.ABSOLUTE_AWARE
        if timing.timezone
        else TimingReferenceClass.ABSOLUTE_NAIVE
    )


_ANCHORED = frozenset({TimingReferenceClass.ABSOLUTE_AWARE, TimingReferenceClass.ABSOLUTE_NAIVE})


@dataclass(frozen=True, slots=True)
class SourceTimingProfile:
    """One source's classified timing reference, plus the raw evidence."""

    source_id: str
    display_name: str
    reference_class: TimingReferenceClass
    timing_reference_raw: str
    timezone_label: str | None
    has_real_anchor: bool
    # Manual session alignment context, included for transparency -- this
    # is SESSION ALIGNMENT, not raw timing compatibility (see module
    # docstring's "Raw Timing vs Session Alignment" distinction); it is
    # never used to upgrade or downgrade reference_class itself.
    time_offset_s: float
    alignment_method: str


def classify_source_timing(source: "SessionSource") -> SourceTimingProfile:
    """Classify one SessionSource's timing reference from its record's
    existing TimingInformation. Read-only: never mutates *source* or its
    record.
    """
    timing = source.record.timing_info
    reference_class = classify_timing_reference(timing)
    return SourceTimingProfile(
        source_id=source.source_id,
        display_name=source.display_name,
        reference_class=reference_class,
        timing_reference_raw=timing.timing_reference or _ABSOLUTE,
        timezone_label=timing.timezone,
        has_real_anchor=reference_class in _ANCHORED,
        time_offset_s=source.time_offset_s,
        alignment_method=source.alignment_method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility levels and pairwise assessment
# ─────────────────────────────────────────────────────────────────────────────


class TimingCompatibilityLevel(str, Enum):
    """Factual, not judgmental: what the available metadata supports --
    never an assertion that a comparison IS correct, only whether it CAN
    be trusted from source metadata alone.
    """

    COMPATIBLE = "compatible"
    REQUIRES_REVIEW = "requires_review"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


# Severity order for combining multiple pairwise results into one overall
# session-level verdict (highest severity wins). INCOMPATIBLE is defined
# for API completeness and is intentionally never asserted by the rules
# below from the evidence currently available anywhere in this
# application -- every conservative rule below resolves an uncertain pair
# to REQUIRES_REVIEW or UNKNOWN rather than the stronger, unproven claim
# that two sources definitely cannot be compared. See this module's
# docstring and the Sprint 1B report for why.
_LEVEL_SEVERITY: dict[TimingCompatibilityLevel, int] = {
    TimingCompatibilityLevel.COMPATIBLE: 0,
    TimingCompatibilityLevel.REQUIRES_REVIEW: 1,
    TimingCompatibilityLevel.UNKNOWN: 2,
    TimingCompatibilityLevel.INCOMPATIBLE: 3,
}


@dataclass(frozen=True, slots=True)
class TimingCompatibilityResult:
    """Factual result for one source pair (or the whole session)."""

    level: TimingCompatibilityLevel
    message: str
    affected_source_ids: tuple[str, ...]
    details: tuple[str, ...]


def assess_pair_compatibility(
    a: SourceTimingProfile, b: SourceTimingProfile
) -> TimingCompatibilityResult:
    """Assess whether two sources' RAW timing metadata (never their
    session offsets) supports direct comparison. See module docstring's
    Case 1-6 policy; the rules below implement exactly that policy.
    """
    details = (
        f"{a.display_name}: {a.reference_class.value}",
        f"{b.display_name}: {b.reference_class.value}",
    )
    ids = (a.source_id, b.source_id)

    if TimingReferenceClass.UNKNOWN in (a.reference_class, b.reference_class):
        return TimingCompatibilityResult(
            level=TimingCompatibilityLevel.UNKNOWN,
            message=(
                f"{a.display_name} and {b.display_name} include an unrecognized "
                "timing reference; compatibility cannot be assessed from "
                "available metadata."
            ),
            affected_source_ids=ids, details=details,
        )

    if a.reference_class is TimingReferenceClass.ABSOLUTE_AWARE and b.reference_class is TimingReferenceClass.ABSOLUTE_AWARE:
        return TimingCompatibilityResult(
            level=TimingCompatibilityLevel.COMPATIBLE,
            message=(
                f"{a.display_name} and {b.display_name} both report "
                "timezone-aware absolute timestamps and can be compared directly."
            ),
            affected_source_ids=ids, details=details,
        )

    if a.reference_class in _ANCHORED and b.reference_class in _ANCHORED:
        # naive+naive, or aware+naive -- Case 1: timezone equivalence would
        # have to be assumed; the app cannot prove it.
        return TimingCompatibilityResult(
            level=TimingCompatibilityLevel.REQUIRES_REVIEW,
            message=(
                f"{a.display_name} and {b.display_name} both report absolute "
                "timestamps, but timezone information is missing for at least "
                "one; equivalence is assumed, not verified."
            ),
            affected_source_ids=ids, details=details,
        )

    if a.reference_class in _ANCHORED or b.reference_class in _ANCHORED:
        # Case 3/4: one side has a real anchor, the other does not.
        anchored, other = (a, b) if a.reference_class in _ANCHORED else (b, a)
        return TimingCompatibilityResult(
            level=TimingCompatibilityLevel.REQUIRES_REVIEW,
            message=(
                f"{other.display_name} has no reliable absolute timing anchor "
                f"({other.reference_class.value}); its relationship to "
                f"{anchored.display_name} depends on manual session alignment."
            ),
            affected_source_ids=ids, details=details,
        )

    # Case 5/6: neither side is anchored (elapsed_unanchored / reconstructed
    # / sample_index in any combination). Matching t=0 or index 0 does not
    # prove a common real-world instant.
    return TimingCompatibilityResult(
        level=TimingCompatibilityLevel.REQUIRES_REVIEW,
        message=(
            f"{a.display_name} and {b.display_name} both lack a reliable "
            "absolute timing anchor; their reference points are not proven "
            "to correspond to the same real-world instant."
        ),
        affected_source_ids=ids, details=details,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-level assessment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionTimingAssessment:
    """Session-wide timing compatibility assessment -- read-only."""

    level: TimingCompatibilityLevel
    summary: str
    source_profiles: tuple[SourceTimingProfile, ...]
    pair_results: tuple[TimingCompatibilityResult, ...]
    manual_alignment_present: bool

    @property
    def has_warning(self) -> bool:
        """True when the session-level result is anything other than a
        clean COMPATIBLE (or the trivial 0/1-active-source case)."""
        return self.level is not TimingCompatibilityLevel.COMPATIBLE


def assess_session_timing_compatibility(session: "EventAnalysisSession") -> SessionTimingAssessment:
    """Assess timing-reference compatibility across the session's ACTIVE
    sources. Read-only: never modifies the session, any source, any
    record, any offset, or any panel/calculated-signal state.
    """
    active_sources = [s for s in session.list_sources() if s.is_active]
    profiles = tuple(classify_source_timing(s) for s in active_sources)

    if len(profiles) <= 1:
        summary = (
            "No multi-source timing comparison required."
            if not profiles
            else "Only one active source; no cross-source timing comparison applies."
        )
        return SessionTimingAssessment(
            level=TimingCompatibilityLevel.COMPATIBLE,
            summary=summary,
            source_profiles=profiles,
            pair_results=(),
            manual_alignment_present=False,
        )

    pair_results = tuple(
        assess_pair_compatibility(a, b) for a, b in combinations(profiles, 2)
    )
    overall_level = max(
        (r.level for r in pair_results),
        key=lambda lvl: _LEVEL_SEVERITY[lvl],
    )
    manual_alignment_present = any(
        p.alignment_method != "none" or p.time_offset_s != 0.0 for p in profiles
    )

    summary = _build_session_summary(overall_level, pair_results, manual_alignment_present)

    return SessionTimingAssessment(
        level=overall_level,
        summary=summary,
        source_profiles=profiles,
        pair_results=pair_results,
        manual_alignment_present=manual_alignment_present,
    )


def _build_session_summary(
    level: TimingCompatibilityLevel,
    pair_results: tuple[TimingCompatibilityResult, ...],
    manual_alignment_present: bool,
) -> str:
    if level is TimingCompatibilityLevel.COMPATIBLE:
        return "Timing references appear compatible based on available metadata."

    problem_count = sum(1 for r in pair_results if r.level is not TimingCompatibilityLevel.COMPATIBLE)
    if level is TimingCompatibilityLevel.UNKNOWN:
        base = (
            f"{problem_count} source pair(s) have an unrecognized timing "
            "reference; compatibility cannot be confirmed."
        )
    elif level is TimingCompatibilityLevel.INCOMPATIBLE:
        base = f"{problem_count} source pair(s) have incompatible timing references."
    else:
        base = f"{problem_count} source pair(s) use timing references that require review."

    if manual_alignment_present:
        base += " Current comparison relies on manual session alignment."
    return base
