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
class SessionAlignmentState:
    """Alignment geometry restored from a manifest's ``alignment`` block.

    Keyed throughout by the MANIFEST source_id (``SourceRecord.source_id``),
    which is the stable identity across save/reload — a live
    ``SessionSource.source_id`` is a fresh uuid4 minted on every load and can
    never be used to look up persisted state.

    Every field is optional so a manifest written before Stage 3 (or by hand)
    still loads. ``absolute_time_origin is None`` means the manifest did not
    record a session origin, which is the signal that restored offsets have no
    provable wall-clock meaning and must be treated as opaque — see
    ``has_trustworthy_origin``.
    """

    absolute_time_origin: datetime | None = None
    reference_source: str | None = None
    offsets_seconds: dict[str, float] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def has_offsets(self) -> bool:
        return bool(self.offsets_seconds)

    @property
    def has_trustworthy_origin(self) -> bool:
        """True when the manifest recorded the session coordinate origin.

        When False, restored offsets are still applied verbatim but their
        provenance cannot be reconstructed, so the loader downgrades every
        restored method to 'imported' rather than letting a claimed
        'absolute_timestamp' be re-derived against an origin nobody saved.
        """
        return self.absolute_time_origin is not None


@dataclass(slots=True)
class MultiSourceSession:
    """Container for multiple co-loaded DisturbanceRecords.

    Sources are preserved independently; display alignment is computed
    on demand (non-destructive). All original records remain immutable
    from this container's perspective.
    """

    sources: list[SourceRecord] = field(default_factory=list)
    alignment: SessionAlignmentState = field(default_factory=lambda: SessionAlignmentState())
    """Stage 3: alignment geometry parsed from the manifest, empty when the
    manifest carried no alignment block."""

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
