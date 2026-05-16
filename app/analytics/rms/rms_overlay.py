"""RMS eligibility classification for analog waveform channels.

Automatic eligibility rules identify channels likely to contain instantaneous
AC waveforms (voltage / current) for which a sliding RMS envelope is meaningful.

Priority chain (highest → lowest):
  1. force=True              → always eligible (operator override)
  2. measurement_kind field  → deterministic when set
  3. electrical_type field   → deterministic when set
  4. channel name heuristics → best-effort pattern matching

CRITICAL: operator decisions MUST override automatic classification.
The analytics system is assistive intelligence only — it MUST NOT block
an operator from applying RMS to any channel they choose.

Default for unrecognised channels: ineligible (conservative).
CSV/Excel SCADA trend channels are typically ineligible unless explicitly
overridden because they carry averaged/telemetry values, not instantaneous AC.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.rms.rms_models import RMSEligibilityResult

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Classification tables
# ─────────────────────────────────────────────────────────────────────────────

# measurement_kind values that make a channel deterministically eligible
_ELIGIBLE_KINDS: frozenset[str] = frozenset(["instantaneous"])

# measurement_kind values that make a channel deterministically ineligible
_INELIGIBLE_KINDS: frozenset[str] = frozenset(["rms", "average", "calculated", "telemetry"])

# electrical_type values that indicate raw waveform (eligible for RMS)
_ELIGIBLE_ELECTRICAL: frozenset[str] = frozenset(["voltage", "current"])

# electrical_type values that indicate derived/processed signal (ineligible)
_INELIGIBLE_ELECTRICAL: frozenset[str] = frozenset(["power", "frequency", "rocof"])

# Name substrings that indicate already-processed or engineering-quantity channels
_INELIGIBLE_NAME_FRAGMENTS: frozenset[str] = frozenset(
    [
        "rms",
        "vrms",
        "irms",
        "vpu",
        "_pu",
        " pu",
        "mw",
        "mvar",
        "hz",
        "freq",
        "rocof",
    ]
)

# Name substrings suggesting instantaneous AC voltage or current
_ELIGIBLE_NAME_FRAGMENTS: frozenset[str] = frozenset(
    [
        "va",
        "vb",
        "vc",
        "vr",
        "vy",
        "vn",
        "ia",
        "ib",
        "ic",
        "ir",
        "iy",
        "in_",
        "3i0",
        "3u0",
        "_v",
        "_i",
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def classify_rms_eligibility(
    channel_name: str,
    signal_meta: "SignalMetadata | None" = None,
    *,
    force: bool = False,
) -> RMSEligibilityResult:
    """Determine whether a channel is eligible for RMS overlay.

    Args:
        channel_name: Column / channel name to check.
        signal_meta:  Optional SignalMetadata for additional context.
        force:        If True, always return eligible regardless of heuristics.
                      Use this to honour explicit operator override requests.

    Returns:
        RMSEligibilityResult with eligible flag, reason string, and
        auto_classified flag (False when force is used).
    """
    if force:
        return RMSEligibilityResult(
            eligible=True,
            reason="operator_override",
            auto_classified=False,
        )

    if signal_meta is not None:
        # measurement_kind is the most authoritative field when present
        mk = getattr(signal_meta, "measurement_kind", None)
        if mk in _ELIGIBLE_KINDS:
            return RMSEligibilityResult(
                eligible=True,
                reason=f"measurement_kind:{mk}",
                auto_classified=True,
            )
        if mk in _INELIGIBLE_KINDS:
            return RMSEligibilityResult(
                eligible=False,
                reason=f"measurement_kind:{mk}",
                auto_classified=True,
            )

        # electrical_type is reliable when set by providers or manifest loader
        et = getattr(signal_meta, "electrical_type", None)
        if et in _ELIGIBLE_ELECTRICAL:
            return RMSEligibilityResult(
                eligible=True,
                reason=f"electrical_type:{et}",
                auto_classified=True,
            )
        if et in _INELIGIBLE_ELECTRICAL:
            return RMSEligibilityResult(
                eligible=False,
                reason=f"electrical_type:{et}",
                auto_classified=True,
            )

    # Fall back to channel name heuristics
    name_lower = channel_name.lower()

    for fragment in _INELIGIBLE_NAME_FRAGMENTS:
        if fragment in name_lower:
            return RMSEligibilityResult(
                eligible=False,
                reason=f"name_fragment:{fragment}",
                auto_classified=True,
            )

    for fragment in _ELIGIBLE_NAME_FRAGMENTS:
        if fragment in name_lower:
            return RMSEligibilityResult(
                eligible=True,
                reason=f"name_fragment:{fragment}",
                auto_classified=True,
            )

    # Conservative default: unknown channels are not eligible
    return RMSEligibilityResult(
        eligible=False,
        reason="default_ineligible",
        auto_classified=True,
    )
