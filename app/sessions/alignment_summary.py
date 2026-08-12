"""Human-readable alignment and clock-verification summary (Stage 3).

Pure and Qt-free: turns the session's existing alignment state into short,
honest strings for the Session Panel and source rows. Introduces no new
clock-quality engine — every judgement about what a source's timestamps mean
comes from ``classify_timing_reference()``, and the session-wide compatibility
verdict remains ``assess_session_timing_compatibility()``'s job and is not
altered, suppressed, or second-guessed here.

The distinction this module exists to keep visible:

    ALIGNMENT            Powerwave positioned these records using the
                         timestamps stored in the files.
    CLOCK VERIFICATION   Whether the device clocks were known to agree.
                         Powerwave cannot establish this from any metadata a
                         current provider produces, so it never claims it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from app.sessions.absolute_alignment import ABSOLUTE_TIMESTAMP, is_eligible
from app.sessions.timing_compatibility import (
    TimingReferenceClass,
    classify_timing_reference,
)

if TYPE_CHECKING:
    from app.sessions.session_models import SessionSource

# Display labels for the ALIGNMENT_METHODS vocabulary. The internal identifier
# is never shown to the engineer when a clearer label exists.
ALIGNMENT_METHOD_LABELS: dict[str, str] = {
    "none": "Not aligned",
    ABSOLUTE_TIMESTAMP: "Absolute timestamp",
    "manual": "Manual",
    "auto_trigger": "Trigger detection",
    "correlation": "Cross-correlation",
    "imported": "Imported",
}

# Short forms used inside a mixed-session one-liner.
_SHORT_LABELS: dict[str, str] = {
    "none": "not aligned",
    ABSOLUTE_TIMESTAMP: "absolute timestamp",
    "manual": "manual",
    "auto_trigger": "trigger",
    "correlation": "correlation",
    "imported": "imported",
}

# Order used when rendering mixed counts, so the summary is deterministic.
_METHOD_ORDER: tuple[str, ...] = (
    ABSOLUTE_TIMESTAMP, "manual", "auto_trigger", "correlation", "imported", "none",
)

# Why a source could not take part in absolute-timestamp alignment. Phrased as
# a fact about the source, never as an error -- an analyst may have overridden
# a source deliberately.
_INELIGIBLE_REASONS: dict[TimingReferenceClass, str] = {
    TimingReferenceClass.ELAPSED_UNANCHORED: "elapsed time axis, no absolute anchor",
    TimingReferenceClass.RECONSTRUCTED: "reconstructed time axis, anchor assumed",
    TimingReferenceClass.SAMPLE_INDEX: "sample-index axis, not a time axis",
    TimingReferenceClass.UNKNOWN: "unrecognised timing reference",
}


def method_label(method: str) -> str:
    """Human-readable label for an alignment method identifier."""
    return ALIGNMENT_METHOD_LABELS.get(method, method)


class ClockVerification(str, Enum):
    """What can honestly be said about device-clock agreement."""

    NOT_APPLICABLE = "not_applicable"
    """No active source carries a real wall-clock anchor, so clock agreement
    is not a meaningful question -- these axes are relative."""

    NOT_VERIFIED = "not_verified"
    """At least one source is wall-clock anchored, but nothing in the
    available metadata proves the device clocks agreed. This is the only
    other value: Powerwave has no evidence that could produce a VERIFIED."""


@dataclass(frozen=True, slots=True)
class AlignmentSummary:
    """One session's alignment state, ready to render."""

    headline: str
    """e.g. "Absolute timestamp", "Mixed (2 absolute timestamp, 1 manual)"."""

    clock_verification: ClockVerification
    clock_line: str
    """e.g. "Not verified — timezone/clock equivalence is not confirmed"."""

    method_counts: dict[str, int] = field(default_factory=dict)
    per_source: tuple[tuple[str, str], ...] = ()
    """(display_name, method label) for every active source."""

    not_timestamp_aligned: tuple[tuple[str, str], ...] = ()
    """(display_name, reason) for active sources not absolute-timestamp aligned."""

    @property
    def is_mixed(self) -> bool:
        return len(self.method_counts) > 1

    def detail_text(self) -> str:
        """Multi-line tooltip body. Empty when there is nothing extra to say."""
        lines: list[str] = [f"{name}: {label}" for name, label in self.per_source]
        if self.not_timestamp_aligned:
            lines.append("")
            lines.append("Not timestamp-aligned:")
            lines.extend(f"  {name} — {reason}" for name, reason in self.not_timestamp_aligned)
        return "\n".join(lines)


def _format_mixed(counts: dict[str, int]) -> str:
    parts = [
        f"{counts[m]} {_SHORT_LABELS.get(m, m)}"
        for m in _METHOD_ORDER
        if counts.get(m)
    ]
    for method, n in counts.items():          # any method outside _METHOD_ORDER
        if method not in _METHOD_ORDER:
            parts.append(f"{n} {method}")
    return f"Mixed ({', '.join(parts)})"


def summarize_alignment(session) -> AlignmentSummary:
    """Summarise how *session*'s active sources are aligned. Never mutates."""
    sources: list["SessionSource"] = [s for s in session.list_sources() if s.is_active]

    if not sources:
        return AlignmentSummary(
            headline="No sources",
            clock_verification=ClockVerification.NOT_APPLICABLE,
            clock_line="",
        )

    counts: dict[str, int] = {}
    per_source: list[tuple[str, str]] = []
    not_aligned: list[tuple[str, str]] = []
    anchored = 0
    missing_timezone = False

    for source in sources:
        method = source.alignment_method
        counts[method] = counts.get(method, 0) + 1
        per_source.append((source.display_name, method_label(method)))

        reference_class = classify_timing_reference(source.record.timing_info)
        if reference_class in (
            TimingReferenceClass.ABSOLUTE_AWARE,
            TimingReferenceClass.ABSOLUTE_NAIVE,
        ):
            anchored += 1
            if reference_class is TimingReferenceClass.ABSOLUTE_NAIVE:
                missing_timezone = True

        if method != ABSOLUTE_TIMESTAMP:
            if is_eligible(source):
                # Eligible but not timestamp-aligned: the analyst owns it.
                # Deliberately not phrased as a problem.
                not_aligned.append(
                    (source.display_name, f"{method_label(method).lower()} alignment")
                )
            else:
                not_aligned.append(
                    (
                        source.display_name,
                        _INELIGIBLE_REASONS.get(reference_class, "no absolute anchor"),
                    )
                )

    headline = (
        method_label(next(iter(counts))) if len(counts) == 1 else _format_mixed(counts)
    )

    if anchored == 0:
        verification = ClockVerification.NOT_APPLICABLE
        clock_line = "Timing reference: Relative — sources are not wall-clock anchored"
    else:
        verification = ClockVerification.NOT_VERIFIED
        if missing_timezone:
            clock_line = (
                "Not verified — timezone/clock equivalence is not confirmed"
            )
        else:
            clock_line = (
                "Not verified — device clock agreement is not confirmed"
            )
        if anchored < len(sources):
            clock_line += f" ({len(sources) - anchored} source(s) not wall-clock anchored)"

    return AlignmentSummary(
        headline=headline,
        clock_verification=verification,
        clock_line=clock_line,
        method_counts=counts,
        per_source=tuple(per_source),
        not_timestamp_aligned=tuple(not_aligned),
    )
