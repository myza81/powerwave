"""Frequency and ROCOF channel role classification — Phase 5C.

Classifies a channel as FREQUENCY, ROCOF, or UNKNOWN using a deterministic
priority chain.  Classification is metadata-only: no waveform values are read
and no computation is performed.

Priority chain (highest → lowest):
  1. force_role keyword argument   → operator override, always wins
  2. measurement_kind metadata     → deterministic when set
  3. electrical_type metadata      → deterministic when set
  4. Unit heuristics               → reliable when a unit is provided
  5. Name heuristics               → best-effort pattern matching

Default for unrecognised channels: UNKNOWN (conservative).

Engineering unit policy:
  FREQUENCY channels: fixed at "Hz" — never auto-scaled to kHz, MHz, etc.
  ROCOF channels:     fixed at "Hz/s" — sign (+/-) is preserved in data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.frequency.frequency_models import (
    FrequencyChannelResult,
    FrequencyChannelRole,
)

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Classification tables
# ─────────────────────────────────────────────────────────────────────────────

# measurement_kind values that map to a frequency role
_KIND_FREQUENCY: frozenset[str] = frozenset(["frequency"])
_KIND_ROCOF: frozenset[str] = frozenset(["rocof"])

# electrical_type values that map to a frequency role
_ELEC_FREQUENCY: frozenset[str] = frozenset(["frequency"])
_ELEC_ROCOF: frozenset[str] = frozenset(["rocof"])

# Normalised unit strings that conclusively identify the role
_UNIT_FREQUENCY: frozenset[str] = frozenset(["hz", "hertz"])
_UNIT_ROCOF: frozenset[str] = frozenset(
    ["hz/s", "hz/sec", "hzpersecond", "hz_s", "df/dt", "dfdt", "df_dt"]
)

# Name substrings that suggest a ROCOF channel (checked before frequency
# fragments because "rocof" and "df/dt" substrings are unambiguous)
_ROCOF_NAME_FRAGMENTS: tuple[str, ...] = (
    "rocof",
    "df/dt",
    "dfdt",
    "df_dt",
    "hz/s",
    "hzpersec",
    "rateofchange",
    "rate_of_change",
)

# Name substrings that suggest a frequency channel
_FREQUENCY_NAME_FRAGMENTS: tuple[str, ...] = (
    "frequency",
    "freq",
    "system frequency",
    "sysfreq",
    "busfreq",
    "pmufreq",
    "gridfreq",
)

# Single short names that are commonly used for frequency channels
_FREQUENCY_SHORT_NAMES: frozenset[str] = frozenset(["f", "hz", "freq"])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_DISPLAY_UNITS: dict[FrequencyChannelRole, str | None] = {
    FrequencyChannelRole.FREQUENCY: "Hz",
    FrequencyChannelRole.ROCOF: "Hz/s",
    FrequencyChannelRole.UNKNOWN: None,
}


def classify_frequency_role(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
    *,
    force_role: FrequencyChannelRole | None = None,
) -> FrequencyChannelResult:
    """Classify a channel as FREQUENCY, ROCOF, or UNKNOWN.

    Args:
        channel_name: Channel / column name to classify.
        unit:         Optional unit string from provider metadata (e.g. "Hz",
                      "Hz/s", "kV").  Improves accuracy when available.
        signal_meta:  Optional SignalMetadata for authoritative classification.
        force_role:   If not None, returns this role unconditionally.
                      Use this to honour an explicit operator selection.

    Returns:
        FrequencyChannelResult with role, reason, auto_classified, display_unit.
    """
    if force_role is not None:
        return FrequencyChannelResult(
            role=force_role,
            reason="operator_override",
            auto_classified=False,
            display_unit=_DISPLAY_UNITS[force_role],
        )

    if signal_meta is not None:
        mk = getattr(signal_meta, "measurement_kind", None) or ""
        if mk in _KIND_FREQUENCY:
            return _result(FrequencyChannelRole.FREQUENCY, f"measurement_kind:{mk}")
        if mk in _KIND_ROCOF:
            return _result(FrequencyChannelRole.ROCOF, f"measurement_kind:{mk}")

        et = getattr(signal_meta, "electrical_type", None) or ""
        if et in _ELEC_FREQUENCY:
            return _result(FrequencyChannelRole.FREQUENCY, f"electrical_type:{et}")
        if et in _ELEC_ROCOF:
            return _result(FrequencyChannelRole.ROCOF, f"electrical_type:{et}")

    # Unit heuristics — reliable when unit is explicitly set by the provider
    if unit:
        unit_norm = unit.strip().lower().replace(" ", "")
        if unit_norm in _UNIT_ROCOF:
            return _result(FrequencyChannelRole.ROCOF, f"unit:{unit_norm}")
        if unit_norm in _UNIT_FREQUENCY:
            return _result(FrequencyChannelRole.FREQUENCY, f"unit:{unit_norm}")

    # Name heuristics — ROCOF fragments checked first (more specific)
    name_lower = channel_name.strip().lower()
    for fragment in _ROCOF_NAME_FRAGMENTS:
        if fragment in name_lower:
            return _result(FrequencyChannelRole.ROCOF, f"name_fragment:{fragment}")

    for fragment in _FREQUENCY_NAME_FRAGMENTS:
        if fragment in name_lower:
            return _result(FrequencyChannelRole.FREQUENCY, f"name_fragment:{fragment}")

    if name_lower in _FREQUENCY_SHORT_NAMES:
        return _result(FrequencyChannelRole.FREQUENCY, f"name_exact:{name_lower}")

    return FrequencyChannelResult(
        role=FrequencyChannelRole.UNKNOWN,
        reason="default_unknown",
        auto_classified=True,
        display_unit=None,
    )


def is_frequency_channel(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
) -> bool:
    """Return True if the channel is classified as FREQUENCY."""
    return (
        classify_frequency_role(channel_name, unit, signal_meta).role
        == FrequencyChannelRole.FREQUENCY
    )


def is_rocof_channel(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
) -> bool:
    """Return True if the channel is classified as ROCOF."""
    return (
        classify_frequency_role(channel_name, unit, signal_meta).role
        == FrequencyChannelRole.ROCOF
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _result(role: FrequencyChannelRole, reason: str) -> FrequencyChannelResult:
    return FrequencyChannelResult(
        role=role,
        reason=reason,
        auto_classified=True,
        display_unit=_DISPLAY_UNITS[role],
    )
