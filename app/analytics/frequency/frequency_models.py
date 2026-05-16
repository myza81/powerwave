"""Data models for frequency and ROCOF analytics — Phase 5C.

These models are display-layer only. They control how existing frequency and
ROCOF channels (imported from CSV, COMTRADE, PMU, or SCADA telemetry) are
classified, routed, and rendered inside Powerwave's synchronized visualization
environment.

No waveform-derived frequency estimation is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FrequencyDisplayMode(str, Enum):
    """Global frequency/ROCOF channel visibility mode.

    OFF        — frequency and ROCOF channels hidden from all panels.
    OVERLAY    — frequency/ROCOF visible in the shared engineering context.
                 Currently behaves identically to PANEL_ONLY; cross-panel
                 secondary-axis overlay is reserved for a later phase.
    PANEL_ONLY — dedicated panel per frequency/ROCOF group (default).
                 Frequency channels appear in the "frequency" panel;
                 ROCOF channels appear in the "rocof" panel.
    """

    OFF = "off"
    OVERLAY = "overlay"
    PANEL_ONLY = "panel_only"


class FrequencyChannelRole(str, Enum):
    """Classification role of a channel for frequency/ROCOF display routing.

    FREQUENCY — channel carries Hz measurements (system frequency, bus
                frequency, PMU frequency, SCADA frequency telemetry, etc.)
    ROCOF     — channel carries Hz/s measurements (rate of change of
                frequency, df/dt, ROCOF protection relay output, etc.)
    UNKNOWN   — channel does not appear to carry frequency or ROCOF data.
    """

    FREQUENCY = "frequency"
    ROCOF = "rocof"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FrequencyChannelResult:
    """Classification result for a single channel's frequency/ROCOF role.

    Attributes:
        role:            Assigned FrequencyChannelRole.
        reason:          Machine-readable reason string (e.g. "unit:hz",
                         "name_fragment:rocof", "electrical_type:frequency").
        auto_classified: True = heuristic / automatic; False = operator override.
        display_unit:    Canonical fixed operational display unit.
                         FREQUENCY → "Hz"; ROCOF → "Hz/s"; UNKNOWN → None.
    """

    role: FrequencyChannelRole
    reason: str
    auto_classified: bool
    display_unit: str | None = None


@dataclass(frozen=True)
class FrequencyConfig:
    """Configuration for frequency/ROCOF channel display.

    Attributes:
        nominal_hz: Power system nominal frequency in Hz (50 or 60).
                    Carried here for future deviation annotation overlays
                    and threshold display. Does not affect classification.
    """

    nominal_hz: float = 50.0
