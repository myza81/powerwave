"""Top-level file profiling entry point.

profile_import_file()  — auto-detect CSV vs Excel and build a FileProfileResult.
populate_session()     — write profile results into an ImportWizardSession.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.import_wizard.column_detector import detect_column_mappings
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.models import (
    ColumnMappingCandidate,
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.timestamp_detector import detect_timestamp_candidates


@dataclass(slots=True)
class FileProfileResult:
    """Result of profiling one CSV or Excel file.

    Fields
    ------
    raw_preview         Lightweight snapshot of the file structure.
    timestamp_candidates  Candidates ranked by confidence (highest first).
    column_mappings     Per-column classification results.
    validation_messages  All warnings/errors from profiling.
    provider_type       "csv" or "excel".
    sheet_name          Active sheet for Excel; None for CSV.
    delimiter           Detected or user-supplied CSV delimiter; None for Excel.
    """

    raw_preview: RawPreviewModel
    timestamp_candidates: list[TimestampCandidate] = field(default_factory=list)
    column_mappings: list[ColumnMappingCandidate] = field(default_factory=list)
    validation_messages: list[ValidationMessage] = field(default_factory=list)
    provider_type: str = "csv"
    sheet_name: str | None = None
    delimiter: str | None = None

    def has_errors(self) -> bool:
        return any(m.severity == ValidationSeverity.ERROR for m in self.validation_messages)


def _is_excel(path: Path) -> bool:
    return path.suffix.lower() in (".xlsx", ".xls", ".xlsm", ".xlsb")


def profile_import_file(
    path: str,
    *,
    sheet_name: str | None = None,
    delimiter: str | None = None,
    max_preview_rows: int = 50,
    max_scan_rows: int = 200,
) -> FileProfileResult:
    """Profile a CSV or Excel file and return a self-contained FileProfileResult.

    Parameters
    ----------
    path:               Absolute path to the file to profile.
    sheet_name:         Excel sheet to use; ignored for CSV.
    delimiter:          Force CSV delimiter; auto-detect when None.
    max_preview_rows:   Max data rows in the preview snapshot.
    max_scan_rows:      Max rows to read when searching for the header.

    Notes
    -----
    Never raises.  File-level errors are returned as ERROR-severity
    ValidationMessage entries in the result.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        empty = RawPreviewModel(column_names=[], preview_rows=[])
        return FileProfileResult(
            raw_preview=empty,
            validation_messages=[
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="FILE_NOT_FOUND",
                    message=f"File not found: {path}",
                )
            ],
        )

    if _is_excel(path_obj):
        from app.import_wizard.excel_profiler import profile_excel

        raw_preview, warnings = profile_excel(
            str(path_obj),
            sheet_name=sheet_name,
            max_preview_rows=max_preview_rows,
            max_scan_rows=max_scan_rows,
        )
        provider_type = "excel"
        effective_sheet = raw_preview.sheet_name
        effective_delimiter = None
    else:
        from app.import_wizard.csv_profiler import profile_csv

        raw_preview, warnings = profile_csv(
            str(path_obj),
            delimiter=delimiter,
            max_preview_rows=max_preview_rows,
            max_scan_rows=max_scan_rows,
        )
        provider_type = "csv"
        effective_sheet = None
        from app.import_wizard.csv_profiler import detect_delimiter as _dd

        effective_delimiter = delimiter or _dd(str(path_obj))

    # Abort early if profiling produced errors
    errors = [w for w in warnings if w.severity == ValidationSeverity.ERROR]
    if errors:
        return FileProfileResult(
            raw_preview=raw_preview,
            validation_messages=warnings,
            provider_type=provider_type,
            sheet_name=effective_sheet,
            delimiter=effective_delimiter,
        )

    # Timestamp detection
    ts_candidates = detect_timestamp_candidates(
        raw_preview.column_names,
        raw_preview.preview_rows,
    )

    ts_names = {c.column_name for c in ts_candidates if c.confidence >= 0.5}

    # Column classification (exclude timestamp columns)
    col_mappings = detect_column_mappings(
        raw_preview.column_names,
        raw_preview.preview_rows,
        timestamp_column_names=ts_names,
    )

    return FileProfileResult(
        raw_preview=raw_preview,
        timestamp_candidates=ts_candidates,
        column_mappings=col_mappings,
        validation_messages=warnings,
        provider_type=provider_type,
        sheet_name=effective_sheet,
        delimiter=effective_delimiter,
    )


def populate_session(session: ImportWizardSession, result: FileProfileResult) -> None:
    """Write all profiling results into *session* in-place.

    Called after a successful profile_import_file() call.  Any prior preview,
    timestamp candidates, and column mappings are replaced.
    """
    session.raw_preview = result.raw_preview
    session.timestamp_candidates = list(result.timestamp_candidates)
    session.column_mappings = list(result.column_mappings)

    for msg in result.validation_messages:
        session.add_message(msg)

    if result.delimiter is not None:
        session.delimiter = result.delimiter
    if result.sheet_name is not None:
        session.sheet_name = result.sheet_name

    # Auto-select the highest-confidence timestamp candidate
    if session.timestamp_candidates:
        best = max(session.timestamp_candidates, key=lambda c: c.confidence)
        session.selected_timestamp_column = best.column_name
