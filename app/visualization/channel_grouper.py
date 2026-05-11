"""Channel display grouping for mixed-source disturbance records.

Assigns each channel to a display group for stacked-pane visualization.
Priority: SignalMetadata.display_group > name heuristics.
See docs/CHANNEL_MAPPING_POLICY.md for the authoritative role taxonomy.
"""
from __future__ import annotations

from app.models import DisturbanceRecord

DISPLAY_GROUP_VOLTAGE_RAW = "voltage_raw"
DISPLAY_GROUP_CURRENT_RAW = "current_raw"
DISPLAY_GROUP_POWER = "power"
DISPLAY_GROUP_FREQUENCY = "frequency"
DISPLAY_GROUP_ROCOF = "rocof"
DISPLAY_GROUP_DIGITAL = "digital"
DISPLAY_GROUP_OTHER = "other"


def group_channels_for_display(
    record: DisturbanceRecord,
    signal_metadata: dict | None = None,
) -> dict[str, list[str]]:
    """Group channel names by display group.

    Args:
        record:          DisturbanceRecord whose channels will be grouped.
        signal_metadata: Optional dict mapping channel name → SignalMetadata.
                         When provided, SignalMetadata.display_group overrides
                         the name-heuristic for that channel.

    Returns:
        Dict mapping display_group_name → list of channel names in declaration order.
        Digital channels always go to DISPLAY_GROUP_DIGITAL.
    """
    groups: dict[str, list[str]] = {}

    for ch in record.analog_channels:
        if signal_metadata and ch.name in signal_metadata:
            grp = signal_metadata[ch.name].display_group or DISPLAY_GROUP_OTHER
        else:
            grp = _infer_display_group(ch.name)
        groups.setdefault(grp, []).append(ch.name)

    for ch in record.digital_channels:
        groups.setdefault(DISPLAY_GROUP_DIGITAL, []).append(ch.name)

    return groups


def _infer_display_group(name: str) -> str:
    """Infer display group from channel name using role-heuristic rules."""
    lower = name.lower()

    # Voltage: starts with 'v' + recognisable phase/raw/voltage keyword
    if lower.startswith("v") and (
        "_raw" in lower
        or "voltage" in lower
        or (len(lower) > 1 and lower[1] in "abcn_")
    ):
        return DISPLAY_GROUP_VOLTAGE_RAW

    # Current: starts with 'i' + recognisable phase/raw/current keyword
    if lower.startswith("i") and (
        "_raw" in lower
        or "current" in lower
        or (len(lower) > 1 and lower[1] in "abcn_")
    ):
        return DISPLAY_GROUP_CURRENT_RAW

    # Power: MW, MVar, P_*, Q_*, 'power'
    if "mw" in lower or "mvar" in lower or lower.startswith(("p_", "q_")) or "power" in lower:
        return DISPLAY_GROUP_POWER

    # Frequency
    if "freq" in lower or lower == "f" or lower == "hz":
        return DISPLAY_GROUP_FREQUENCY

    # ROCOF
    if "rocof" in lower or lower.startswith("df"):
        return DISPLAY_GROUP_ROCOF

    return DISPLAY_GROUP_OTHER
