"""ROCOF-specific display helpers for Powerwave — Phase 5C.

Thin wrapper over the shared classify_frequency_role() classifier that provides
ROCOF-focused utilities:

  - rocof_display_label()   — canonical curve/axis label
  - rocof_axis_label()      — short axis label (for shared-axis title)
  - is_rocof_channel()      — convenience bool check (re-exported)

ROCOF engineering policy:
  - Unit is always Hz/s — never relabelled or rescaled
  - Sign is preserved (+ = rising frequency, - = falling frequency)
  - ROCOF channels NEVER share axes with frequency channels
  - ROCOF channels NEVER share axes with voltage, current, or power channels
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.frequency.frequency_models import (
    FrequencyChannelResult,
    FrequencyChannelRole,
)
from app.analytics.frequency.frequency_overlay import (
    classify_frequency_role,
    is_rocof_channel as _is_rocof_channel,
)

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata


# Fixed operational ROCOF display unit — never changed
ROCOF_DISPLAY_UNIT = "Hz/s"

# Fixed operational frequency display unit — never changed
FREQUENCY_DISPLAY_UNIT = "Hz"


def is_rocof_channel(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
) -> bool:
    """Return True if the channel carries ROCOF measurements.

    Delegates to the shared classification function so heuristic tables are
    maintained in one place.
    """
    return _is_rocof_channel(channel_name, unit, signal_meta)


def rocof_display_label(channel_name: str) -> str:
    """Return a human-readable curve label for a ROCOF channel.

    Examples:
        "ROCOF_PULU"  → "ROCOF_PULU (Hz/s)"
        "dfdt"        → "dfdt (Hz/s)"
        "df/dt"       → "df/dt (Hz/s)"
    """
    name = channel_name.strip() or "ROCOF"
    return f"{name} ({ROCOF_DISPLAY_UNIT})"


def rocof_axis_label() -> str:
    """Return the standard shared-axis label for a ROCOF axis."""
    return f"ROCOF ({ROCOF_DISPLAY_UNIT})"


def frequency_display_label(channel_name: str) -> str:
    """Return a human-readable curve label for a frequency channel.

    Examples:
        "SysFreq"    → "SysFreq (Hz)"
        "Frequency"  → "Frequency (Hz)"
    """
    name = channel_name.strip() or "Frequency"
    return f"{name} ({FREQUENCY_DISPLAY_UNIT})"


def frequency_axis_label() -> str:
    """Return the standard shared-axis label for a frequency axis."""
    return f"Frequency ({FREQUENCY_DISPLAY_UNIT})"


def classify_rocof(
    channel_name: str,
    unit: str | None = None,
    signal_meta: "SignalMetadata | None" = None,
    *,
    force: bool = False,
) -> FrequencyChannelResult:
    """Classify a channel specifically for ROCOF eligibility.

    Args:
        channel_name: Channel name.
        unit:         Optional unit string.
        signal_meta:  Optional SignalMetadata.
        force:        If True, treats channel as ROCOF regardless of heuristics.

    Returns:
        FrequencyChannelResult with ROCOF role when eligible, UNKNOWN otherwise.
    """
    if force:
        from app.analytics.frequency.frequency_models import FrequencyChannelResult
        return FrequencyChannelResult(
            role=FrequencyChannelRole.ROCOF,
            reason="operator_override",
            auto_classified=False,
            display_unit=ROCOF_DISPLAY_UNIT,
        )
    result = classify_frequency_role(channel_name, unit, signal_meta)
    if result.role == FrequencyChannelRole.ROCOF:
        return result
    # Return UNKNOWN — not forcing ROCOF on non-ROCOF channels
    from app.analytics.frequency.frequency_models import FrequencyChannelResult
    return FrequencyChannelResult(
        role=FrequencyChannelRole.UNKNOWN,
        reason=f"not_rocof:{result.reason}",
        auto_classified=True,
        display_unit=None,
    )
