"""Multi-source session container for co-loaded disturbance records.

Preserves original DisturbanceRecords independently; display alignment
is computed on demand via display_alignment.py (non-destructive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.data.signal_metadata import SignalMetadata
from app.models.disturbance_record import DisturbanceRecord


@dataclass(slots=True)
class SourceRecord:
    """A single source within a MultiSourceSession.

    Carries the original DisturbanceRecord and its temporal context so
    multi-source alignment can be computed without destructive resampling.
    """

    source_id: str
    provider_type: str
    record: DisturbanceRecord
    signal_metadata: dict[str, SignalMetadata]
    original_start_time: datetime | None
    sampling_rates: list[float]


@dataclass(slots=True)
class MultiSourceSession:
    """Container for multiple co-loaded DisturbanceRecords.

    Sources are preserved independently; display alignment is computed
    on demand (non-destructive). All original records remain immutable
    from this container's perspective.
    """

    sources: list[SourceRecord] = field(default_factory=list)

    def add_source(self, source: SourceRecord) -> None:
        self.sources.append(source)

    def source_count(self) -> int:
        return len(self.sources)

    def is_empty(self) -> bool:
        return len(self.sources) == 0

    def source_ids(self) -> list[str]:
        return [s.source_id for s in self.sources]

    def get_source(self, source_id: str) -> SourceRecord | None:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        return None
