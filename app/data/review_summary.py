"""app/data/review_summary.py

Review summary data model for the Data Mapping Review Dialog (Phase D4.2).

Converts a MultiSourceSession plus optional raw manifest dict into UI-friendly
review structures without any Qt dependency. This is a pure data layer.

Classes:
    ColumnReviewRow       — classification metadata for one data column
    TimestampReviewSummary — timestamp interpretation summary per source
    SourceReviewSummary   — per-source review aggregate
    EventReviewSummary    — top-level review across all sources

Entry point:
    build_event_review_summary(session, manifest_data=None) -> EventReviewSummary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.data.display_alignment import compute_relative_offsets, determine_reference_start
from app.data.multi_source_session import MultiSourceSession
from app.data.signal_metadata import SignalMetadata

# ISO 8601 / space-separated timestamp formats (same set as manifest_loader)
_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ColumnReviewRow:
    """Classification metadata for one data column — the smallest unit of review."""

    column_name: str
    signal_type: str | None      # "frequency", "active_power", "voltage", "current", …
    unit: str | None             # "Hz", "MW", "MVAr", "kV", "A", …
    display_group: str           # "frequency", "power", "voltage_raw", "other", …
    confidence: float            # 0.0 – 1.0 (1.0 = provider-inferred / confirmed)
    inferred_from: str           # "name_exact", "name_keyword", "provider_inferred", …
    requires_user_confirmation: bool
    notes: str | None = None


@dataclass(slots=True)
class TimestampReviewSummary:
    """Timestamp interpretation summary for one source."""

    source_id: str
    raw_format: str | None       # e.g. "ambiguous", "ISO8601 (CFG header)", "detected"
    confirmed_format: str | None # e.g. "M/D/YYYY", "ISO8601" — None if unresolved
    confidence: float
    inferred_from: str           # "provider_parsed", "manifest_notes", "detected", …
    warnings: list[str]          # actionable messages for the operator


@dataclass(slots=True)
class SourceReviewSummary:
    """Per-source review data: metadata + timestamp + column classifications."""

    source_id: str
    provider_type: str           # "comtrade", "csv", "excel"
    file_path: str
    start_time: datetime | None
    trigger_time: datetime | None
    sample_count: int
    sampling_rates: list[float]
    analog_channel_count: int
    digital_channel_count: int
    display_offset_seconds: float
    timestamp_summary: TimestampReviewSummary | None
    column_rows: list[ColumnReviewRow]

    def unconfirmed_columns(self) -> list[ColumnReviewRow]:
        """Return column rows that require operator confirmation."""
        return [r for r in self.column_rows if r.requires_user_confirmation]


@dataclass(slots=True)
class EventReviewSummary:
    """Top-level review aggregator for a loaded event."""

    event_id: str
    reference_start: datetime | None
    sources: list[SourceReviewSummary]
    manifest_notes: list[str]

    def has_unconfirmed_columns(self) -> bool:
        return any(bool(src.unconfirmed_columns()) for src in self.sources)

    def unconfirmed_count(self) -> int:
        return sum(len(src.unconfirmed_columns()) for src in self.sources)

    def all_sources_have_timestamps(self) -> bool:
        return all(src.start_time is not None for src in self.sources)

    def has_timestamp_warnings(self) -> bool:
        return any(
            src.timestamp_summary is not None and bool(src.timestamp_summary.warnings)
            for src in self.sources
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_timestamp_summary(
    source_id: str,
    provider_type: str,
    notes: list[str],
) -> TimestampReviewSummary:
    if provider_type == "comtrade":
        return TimestampReviewSummary(
            source_id=source_id,
            raw_format="ISO8601 (CFG header)",
            confirmed_format="ISO8601",
            confidence=1.0,
            inferred_from="provider_parsed",
            warnings=[],
        )

    # CSV/Excel — check notes for ambiguity signals
    warnings = [n for n in notes if n]
    is_ambiguous = any(
        "ambiguous" in n.lower() or "warning" in n.lower()
        for n in warnings
    )

    if is_ambiguous:
        return TimestampReviewSummary(
            source_id=source_id,
            raw_format="ambiguous",
            confirmed_format=None,
            confidence=0.5,
            inferred_from="manifest_notes",
            warnings=warnings,
        )

    return TimestampReviewSummary(
        source_id=source_id,
        raw_format="detected",
        confirmed_format=None,
        confidence=0.9,
        inferred_from="detected",
        warnings=warnings,
    )


def _infer_display_group(electrical_type: str | None) -> str:
    if electrical_type == "voltage":
        return "voltage_raw"
    if electrical_type == "current":
        return "current_raw"
    return "other"


def _build_column_rows(
    signal_metadata: dict[str, SignalMetadata],
    provider_type: str,
) -> list[ColumnReviewRow]:
    """Build ColumnReviewRow list from SignalMetadata entries.

    COMTRADE channels without explicit classification (confidence=None,
    requires_user_confirmation=False) are suppressed — they are
    provider-inferred with high confidence and need no operator review.
    CSV/Excel channels are always included.
    """
    rows: list[ColumnReviewRow] = []
    for name, meta in signal_metadata.items():
        # Skip COMTRADE auto-inferred channels that need no review
        if (
            provider_type == "comtrade"
            and meta.confidence is None
            and not meta.requires_user_confirmation
        ):
            continue

        signal_type = meta.signal_type or meta.electrical_type
        display_group = meta.display_group or _infer_display_group(meta.electrical_type)
        confidence = meta.confidence if meta.confidence is not None else 1.0
        inferred_from = meta.inferred_from or (
            "provider_inferred" if provider_type == "comtrade" else "unknown"
        )

        rows.append(ColumnReviewRow(
            column_name=name,
            signal_type=signal_type,
            unit=meta.unit,
            display_group=display_group,
            confidence=confidence,
            inferred_from=inferred_from,
            requires_user_confirmation=meta.requires_user_confirmation,
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_event_review_summary(
    session: MultiSourceSession,
    manifest_data: dict | None = None,
) -> EventReviewSummary:
    """Convert a MultiSourceSession into an EventReviewSummary for the review dialog.

    Args:
        session:       Loaded MultiSourceSession (from build_session_from_manifest or manual).
        manifest_data: Raw manifest dict from load_manifest(), used to extract event_id,
                       alignment offsets, and timestamp notes. Optional — defaults to
                       best-effort derivation from session data alone.

    Returns:
        EventReviewSummary with source summaries, column rows, and timestamp info.
    """
    data = manifest_data or {}

    event_id: str = str(data.get("event_id", "unknown"))
    manifest_notes: list[str] = list(data.get("notes", []) or [])

    # Reference start: prefer manifest alignment block, fall back to session
    alignment: dict = data.get("alignment", {}) or {}
    ref_start_str: str | None = alignment.get("reference_start")
    manifest_offsets: dict = alignment.get("offsets_seconds", {}) or {}

    reference_start: datetime | None
    if ref_start_str:
        reference_start = _parse_dt(str(ref_start_str))
    else:
        candidate = determine_reference_start(session.sources)
        reference_start = candidate if isinstance(candidate, datetime) else None

    # Fallback per-source offsets computed from session if not in manifest
    computed_offsets = compute_relative_offsets(session.sources, reference_start)

    # Index manifest source definitions by source_id for fast lookup
    src_defs: dict[str, dict] = {}
    for src_def in data.get("sources", []):
        sid = str(src_def.get("source_id", ""))
        src_defs[sid] = src_def

    source_summaries: list[SourceReviewSummary] = []
    for src in session.sources:
        src_def: dict = src_defs.get(src.source_id, {})

        # Display offset: manifest takes precedence over session computation
        offset: float
        if src.source_id in manifest_offsets:
            offset = float(manifest_offsets[src.source_id])
        else:
            offset = computed_offsets.get(src.source_id, 0.0)

        # Absolute timing from record
        try:
            start_time: datetime | None = src.record.timing_info.start_time
            trigger_time: datetime | None = src.record.timing_info.trigger_time
        except Exception:  # noqa: BLE001
            start_time = None
            trigger_time = None

        # Manifest start_time overrides record timing (e.g. CSV date-corrected)
        manifest_start = _parse_dt(src_def.get("start_time"))
        if manifest_start is not None:
            start_time = manifest_start

        source_notes: list[str] = list(src_def.get("notes", []) or [])
        ts_summary = _build_timestamp_summary(src.source_id, src.provider_type, source_notes)
        column_rows = _build_column_rows(src.signal_metadata, src.provider_type)

        source_summaries.append(SourceReviewSummary(
            source_id=src.source_id,
            provider_type=src.provider_type,
            file_path=src.record.metadata.source_file,
            start_time=start_time,
            trigger_time=trigger_time,
            sample_count=src.record.sample_count(),
            sampling_rates=list(src.sampling_rates),
            analog_channel_count=len(src.record.analog_channels),
            digital_channel_count=len(src.record.digital_channels),
            display_offset_seconds=offset,
            timestamp_summary=ts_summary,
            column_rows=column_rows,
        ))

    return EventReviewSummary(
        event_id=event_id,
        reference_start=reference_start,
        sources=source_summaries,
        manifest_notes=manifest_notes,
    )
