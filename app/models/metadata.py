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

    # Sprint 1E: set by CsvProvider/ExcelProvider to the raw date string that
    # triggered Powerwave's DD/MM/YYYY ambiguous-date default (e.g. "3/6/2026"),
    # or None when the source's time column had no genuinely ambiguous date
    # order. Diagnostic only -- never used to alter parsing.
    timestamp_ambiguity_sample: str | None = None
