"""Provider-neutral axis grouping and geometry policy for visualization.

This module is runtime display policy only. It does not scale waveform values
or mutate records; it decides when compatible signals can share a Y axis and
defines the geometry reservations used to keep stacked panels pixel-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.visualization.engineering_display import (
    EngineeringAxisLabel,
    format_axis_label,
    infer_signal_type,
    normalize_engineering_unit,
)


class AxisDisplayMode(str, Enum):
    """Global Y-axis display policy."""

    SHARED = "shared"
    DEDICATED = "dedicated"

    @classmethod
    def coerce(cls, value: "AxisDisplayMode | str") -> "AxisDisplayMode":
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for mode in cls:
            if text in {mode.value, mode.name.lower()}:
                return mode
        raise ValueError(f"Unknown axis display mode: {value!r}")


@dataclass(frozen=True, slots=True)
class AxisGroup:
    """Resolved axis group for one signal."""

    key: str
    label: EngineeringAxisLabel


GROUPED_LEFT_AXIS_WIDTH_PX = 80
GROUPED_RIGHT_AXIS_WIDTH_PX = 64
GROUPED_MAX_RIGHT_AXIS_WIDTH_PX = 192

_SHARED_ROLE_TITLES = {
    "active_power": "Power",
    "reactive_power": "Reactive Power",
    "frequency": "Frequency",
    "rocof": "ROCOF",
    "voltage": "Voltage",
    "voltage_rms": "Voltage RMS",
    "current": "Current",
    "current_rms": "Current RMS",
    "per_unit": "Per Unit",
}


def axis_group_for_signal(
    channel_name: str,
    unit: str | None,
    *,
    mode: AxisDisplayMode | str = AxisDisplayMode.SHARED,
    signal_type_hint: str | None = None,
) -> AxisGroup:
    """Return the display-axis group for a signal.

    SHARED groups only known engineering-compatible roles by role and fixed
    operational unit. Unknown signals remain dedicated so unrelated quantities
    are never merged accidentally.

    Args:
        signal_type_hint: Override automatic role detection (e.g., pass the
            original signal type when unit has been changed to "pu" by the
            engineering scaling layer so that voltage-pu and current-pu are
            kept on separate shared axes rather than merged as "per_unit").
    """
    axis_mode = AxisDisplayMode.coerce(mode)
    label = format_axis_label(channel_name, unit)
    if axis_mode == AxisDisplayMode.DEDICATED:
        return AxisGroup(key=f"dedicated:{channel_name}", label=label)

    role = signal_type_hint or infer_signal_type(channel_name, unit)
    if role not in _SHARED_ROLE_TITLES:
        return AxisGroup(key=f"signal:{channel_name}", label=label)

    display_unit = normalize_engineering_unit(
        unit,
        signal_type=role,
        channel_name=channel_name,
    )
    shared_label = EngineeringAxisLabel(
        text=_SHARED_ROLE_TITLES[role],
        unit=display_unit,
    )
    return AxisGroup(
        key=f"{role}:{display_unit.lower()}",
        label=shared_label,
    )


def reserved_right_axis_width(right_axis_count: int) -> float:
    """Return the total right-side reservation for a stack of panels."""
    count = max(0, int(right_axis_count))
    return float(min(
        count * GROUPED_RIGHT_AXIS_WIDTH_PX,
        GROUPED_MAX_RIGHT_AXIS_WIDTH_PX,
    ))
