from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnalogChannel:
    """Descriptor for one analog waveform channel (voltage, current, frequency, etc.)."""

    name: str
    unit: str
    index: int

    phase: str | None = None
    description: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    primary_ratio: float | None = None
    secondary_ratio: float | None = None
    parameter_type: str | None = None   # ParameterType.value string; None for COMTRADE/unknown


@dataclass(slots=True)
class DigitalChannel:
    """Descriptor for one digital/binary channel (breaker status, relay trip, etc.)."""

    name: str
    index: int

    normal_state: int = 0
    description: str | None = None
