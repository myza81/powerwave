from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RecordingMetadata:
    """Recording-level identity and configuration for a disturbance record."""

    station_name: str
    recorder_name: str
    source_file: str
    provider_type: str
    nominal_frequency: float

    device_id: str | None = None
    location: str | None = None
    timezone: str | None = None
    comments: str | None = None
