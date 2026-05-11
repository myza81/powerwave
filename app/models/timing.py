from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SamplingInformation:
    """Describes the sampling structure of a disturbance recording.

    Supports COMTRADE multi-rate recordings via parallel lists:
    sampling_rates[i] is the rate (Hz) for the section with samples_per_rate[i] samples.
    """

    sampling_rates: list[float]
    samples_per_rate: list[int]

    samples_per_cycle: float | None = None
    nominal_frequency: float | None = None


@dataclass(slots=True)
class TimingInformation:
    """Absolute time references for a disturbance recording."""

    start_time: datetime
    trigger_time: datetime

    time_multiplier: float = 1.0
    timezone: str | None = None


@dataclass(slots=True)
class DisturbanceInformation:
    """Optional engineering context and classification for a disturbance event."""

    event_type: str | None = None
    notes: str | None = None
    tags: list[str] = field(default_factory=list)
