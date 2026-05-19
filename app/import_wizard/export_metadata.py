"""Export metadata sidecar model and JSON serialization.

ExportMetadata captures the complete audit trail for a normalized export:
source traceability, per-column mapping decisions, timestamp repair strategy,
and accumulated validation history.

The sidecar is written as a JSON file alongside the exported data file so that
any future re-import, review, or regulatory audit can reconstruct exactly how
the normalized file was produced.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.import_wizard.normalized_dataset import NormalizedDataset

EXPORT_SCHEMA_VERSION = "1.0"


@dataclass
class ExportParameterRecord:
    """Serializable record for one normalized data column."""

    canonical_name: str
    parameter_type: str
    unit: str | None
    source_name: str
    source_index: int
    user_overridden: bool
    confidence: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ExportValidationRecord:
    """Serializable record for one validation message."""

    severity: str
    code: str
    message: str
    affected_column: str | None


@dataclass
class ExportMetadata:
    """Complete audit metadata for one normalized export.

    Fields
    ------
    schema_version          Powerwave export schema version string.
    exported_at             ISO-8601 UTC timestamp of when the file was written.
    source_file_name        Basename of the original source file.
    source_file_path        Absolute path to the original source file.
    row_count               Number of rows in the exported dataset.
    column_count            Number of data columns (excludes the timestamp column).
    timestamp_column        Name of the datetime column in the exported file.
    timestamp_repair_strategy  Strategy label used to produce the time axis.
    parameters              Per-column mapping and classification metadata.
    excluded_columns        Source column names excluded from the export.
    validation_messages     Warnings and errors from the import/assembly pipeline.
    export_format           Format used: 'csv', 'parquet', or 'feather'.
    output_path             Absolute path of the exported data file.
    """

    schema_version: str
    exported_at: str
    source_file_name: str | None
    source_file_path: str | None
    row_count: int
    column_count: int
    timestamp_column: str
    timestamp_repair_strategy: str
    parameters: list[ExportParameterRecord]
    excluded_columns: list[str]
    validation_messages: list[ExportValidationRecord]
    export_format: str
    output_path: str


def build_export_metadata(
    dataset: NormalizedDataset,
    *,
    export_format: str,
    output_path: str,
    exported_at: str,
) -> ExportMetadata:
    """Build an ExportMetadata instance from a NormalizedDataset.

    Does not write any files.
    """
    params = [
        ExportParameterRecord(
            canonical_name=p.canonical_name,
            parameter_type=p.parameter_type.value,
            unit=p.unit,
            source_name=p.source_name,
            source_index=p.source_index,
            user_overridden=p.user_overridden,
            confidence=p.confidence,
            notes=list(p.notes),
        )
        for p in dataset.parameters
    ]
    msgs = [
        ExportValidationRecord(
            severity=m.severity.value,
            code=m.code,
            message=m.message,
            affected_column=m.affected_column,
        )
        for m in dataset.validation_messages
    ]
    return ExportMetadata(
        schema_version=EXPORT_SCHEMA_VERSION,
        exported_at=exported_at,
        source_file_name=dataset.source_file_name,
        source_file_path=dataset.source_path,
        row_count=len(dataset.data),
        column_count=len(dataset.parameters),
        timestamp_column=dataset.timestamp_column,
        timestamp_repair_strategy=dataset.timestamp_repair_strategy,
        parameters=params,
        excluded_columns=list(dataset.excluded_columns),
        validation_messages=msgs,
        export_format=export_format,
        output_path=output_path,
    )


def metadata_sidecar_path(output_path: str | Path) -> Path:
    """Return the canonical sidecar path for a given export file.

    ``/some/dir/sample.csv`` → ``/some/dir/sample.normalized.json``
    """
    p = Path(output_path)
    return p.parent / f"{p.stem}.normalized.json"


def write_metadata_sidecar(metadata: ExportMetadata, sidecar_path: Path) -> None:
    """Serialize *metadata* as indented JSON to *sidecar_path*.

    Raises OSError on write failure (caller decides how to handle it).
    """
    data = asdict(metadata)
    sidecar_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_metadata_sidecar(sidecar_path: str | Path) -> dict:
    """Load and parse a JSON metadata sidecar.  Returns the raw dict."""
    return json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
