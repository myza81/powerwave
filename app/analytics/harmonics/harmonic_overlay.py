"""Harmonic channel classification and eligibility filtering — Phase 7.

Classification priority chain (mirrors phasor_overlay.py and rms_overlay.py):
  1. operator force_role override
  2. SignalMetadata.measurement_kind  — deterministic when set
  3. SignalMetadata.electrical_type   — deterministic when set
  4. Unit heuristics                  (kV/V → voltage; A/kA → current)
  5. Name heuristics                  (V-prefix → voltage; I-prefix → current)
  6. Default: UNKNOWN (conservative)

Eligibility vs classification:
  classify_harmonic_role() returns VOLTAGE_HARMONIC, CURRENT_HARMONIC, or UNKNOWN.
  A result of UNKNOWN means the channel is ineligible for harmonic extraction.
  Channels classified as UNKNOWN are excluded from harmonic analysis panels.

Exclusion rules (ineligible channels):
  - measurement_kind: rms, average, calculated, telemetry
  - electrical_type:  power, frequency, rocof
  - Name fragments:   rms, hz, freq, mw, mvar, rocof, _pu, vpu, irms, vrms

These rules are intentionally conservative: power, telemetry and RMS channels
should never enter the FFT pipeline. Operator force_role overrides this.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelResult,
    HarmonicChannelRole,
)

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata

# ─────────────────────────────────────────────────────────────────────────────
# Classification tables
# ─────────────────────────────────────────────────────────────────────────────

# measurement_kind values → HarmonicChannelRole
_MEASUREMENT_KIND_ROLE: dict[str, HarmonicChannelRole | None] = {
    "voltage": HarmonicChannelRole.VOLTAGE_HARMONIC,
    "current": HarmonicChannelRole.CURRENT_HARMONIC,
    "instantaneous": None,  # eligible but type determined by other fields
}

# measurement_kind values that unconditionally exclude a channel
_INELIGIBLE_KINDS: frozenset[str] = frozenset(
    [
        "rms",
        "average",
        "calculated",
        "telemetry",
        "frequency",
        "rocof",
        "phasor",
        "voltage_phasor",
        "current_phasor",
        "sequence",
        "sequence_component",
    ]
)

# electrical_type values → HarmonicChannelRole
_ELECTRICAL_TYPE_ROLE: dict[str, HarmonicChannelRole] = {
    "voltage": HarmonicChannelRole.VOLTAGE_HARMONIC,
    "current": HarmonicChannelRole.CURRENT_HARMONIC,
}

# electrical_type values that unconditionally exclude a channel
_INELIGIBLE_ELECTRICAL: frozenset[str] = frozenset(["power", "frequency", "rocof"])

# Unit strings that indicate voltage (lower-cased)
_VOLTAGE_UNITS: frozenset[str] = frozenset(["kv", "v", "volts", "volt"])

# Unit strings that indicate current (lower-cased)
_CURRENT_UNITS: frozenset[str] = frozenset(["a", "ka", "amps", "amp"])

# Name fragments that mark a channel as ineligible for harmonic analysis
_INELIGIBLE_NAME_FRAGMENTS: frozenset[str] = frozenset([
    "rms", "vrms", "irms", "vpu", "_pu", " pu", "mw", "mvar",
    "hz", "freq", "rocof", "power", "telemetry", "calculated",
    "phasor", "sequence", "magnitude", "angle", "positive sequence",
    "negative sequence", "zero sequence",
])

# Display units by role
_DISPLAY_UNITS: dict[HarmonicChannelRole, str] = {
    HarmonicChannelRole.VOLTAGE_HARMONIC: "kV",
    HarmonicChannelRole.CURRENT_HARMONIC: "A",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public classification API
# ─────────────────────────────────────────────────────────────────────────────


def classify_harmonic_role(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
    *,
    force_role: HarmonicChannelRole | None = None,
) -> HarmonicChannelResult:
    """Classify one channel as VOLTAGE_HARMONIC, CURRENT_HARMONIC, or UNKNOWN.

    UNKNOWN means the channel is ineligible for harmonic extraction.

    Args:
        channel_name: Raw channel name string.
        unit:         Engineering unit, or None.
        signal_meta:  Optional SignalMetadata for authoritative overrides.
        force_role:   Operator override; always wins when not None.

    Returns:
        Immutable HarmonicChannelResult.
    """
    # 1. Operator override
    if force_role is not None:
        return HarmonicChannelResult(
            role=force_role,
            reason="operator_override",
            auto_classified=False,
            display_unit=_DISPLAY_UNITS.get(force_role),
        )

    # 2. measurement_kind
    if signal_meta is not None:
        mk = (getattr(signal_meta, "measurement_kind", None) or "").strip().lower()
        if mk:
            if mk in _INELIGIBLE_KINDS:
                return HarmonicChannelResult(
                    role=HarmonicChannelRole.UNKNOWN,
                    reason=f"measurement_kind:{mk}",
                    auto_classified=True,
                )
            role_from_mk = _MEASUREMENT_KIND_ROLE.get(mk)
            if role_from_mk is not None:
                return HarmonicChannelResult(
                    role=role_from_mk,
                    reason=f"measurement_kind:{mk}",
                    auto_classified=False,
                    display_unit=_DISPLAY_UNITS.get(role_from_mk),
                )
            # mk == "instantaneous": fall through to electrical_type / unit / name

    # 3. electrical_type
    if signal_meta is not None:
        et = (getattr(signal_meta, "electrical_type", None) or "").strip().lower()
        if et:
            if et in _INELIGIBLE_ELECTRICAL:
                return HarmonicChannelResult(
                    role=HarmonicChannelRole.UNKNOWN,
                    reason=f"electrical_type:{et}",
                    auto_classified=True,
                )
            role_from_et = _ELECTRICAL_TYPE_ROLE.get(et)
            if role_from_et is not None:
                return HarmonicChannelResult(
                    role=role_from_et,
                    reason=f"electrical_type:{et}",
                    auto_classified=True,
                    display_unit=_DISPLAY_UNITS.get(role_from_et),
                )

    # 4. Unit heuristics
    if unit:
        u = unit.strip().lower()
        if u in _VOLTAGE_UNITS:
            return HarmonicChannelResult(
                role=HarmonicChannelRole.VOLTAGE_HARMONIC,
                reason=f"unit:{unit}",
                auto_classified=True,
                display_unit="kV",
            )
        if u in _CURRENT_UNITS:
            return HarmonicChannelResult(
                role=HarmonicChannelRole.CURRENT_HARMONIC,
                reason=f"unit:{unit}",
                auto_classified=True,
                display_unit="A",
            )

    # 5. Name heuristics
    name_lower = channel_name.lower()

    # Ineligible name fragments first (conservative exclusion)
    for frag in _INELIGIBLE_NAME_FRAGMENTS:
        if frag in name_lower:
            return HarmonicChannelResult(
                role=HarmonicChannelRole.UNKNOWN,
                reason=f"name_fragment:{frag}",
                auto_classified=True,
            )

    role_from_name = _classify_by_name(channel_name)
    if role_from_name is not None:
        return HarmonicChannelResult(
            role=role_from_name,
            reason=f"name:{channel_name}",
            auto_classified=True,
            display_unit=_DISPLAY_UNITS.get(role_from_name),
        )

    # 6. Default: UNKNOWN (conservative — do not extract harmonics from unknown channels)
    return HarmonicChannelResult(
        role=HarmonicChannelRole.UNKNOWN,
        reason="no_match",
        auto_classified=True,
    )


def is_harmonic_eligible(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
) -> bool:
    """Return True if the channel is eligible for harmonic extraction.

    A channel is eligible when classified as VOLTAGE_HARMONIC or CURRENT_HARMONIC.
    """
    role = classify_harmonic_role(channel_name, unit, signal_meta).role
    return role in (
        HarmonicChannelRole.VOLTAGE_HARMONIC,
        HarmonicChannelRole.CURRENT_HARMONIC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _classify_by_name(name: str) -> HarmonicChannelRole | None:
    """Name-based heuristic. Returns None if no confident match."""
    lower = name.lower().strip()

    # Explicit fragment matches
    if "current" in lower:
        return HarmonicChannelRole.CURRENT_HARMONIC
    if "voltage" in lower or "volt" in lower:
        return HarmonicChannelRole.VOLTAGE_HARMONIC

    # Short prefix: "I..." → current, "V..." → voltage (min 2 chars)
    if len(name) >= 2:
        first = name[0].lower()
        if first == "i":
            return HarmonicChannelRole.CURRENT_HARMONIC
        if first == "v":
            return HarmonicChannelRole.VOLTAGE_HARMONIC

    return None
