"""Normalized export writer for the Import Wizard pipeline.

write_normalized_export() serializes a NormalizedDataset to CSV, Parquet, or
Feather and optionally writes an audit JSON metadata sidecar alongside.

Design constraints
------------------
- Never mutates NormalizedDataset, its DataFrame, or source files.
- Uses native pandas writers; no row-by-row loops.
- Returns ExportWriteResult; never raises on recoverable errors.
- Parquet/Feather availability is detected at import time; missing dependencies
  produce clear validation messages rather than ImportError.
- write_from_export_plan() bridges the ExportPlan planning layer to the writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.export_metadata import (
    build_export_metadata,
    metadata_sidecar_path,
    write_metadata_sidecar,
)
from app.import_wizard.export_planner import ExportPlan
from app.import_wizard.normalized_dataset import NormalizedDataset

# ─────────────────────────────────────────────────────────────────────────────
# Format availability probes  (run once at import time)
# ─────────────────────────────────────────────────────────────────────────────


def _probe_parquet() -> bool:
    try:
        import pyarrow  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


def _probe_feather() -> bool:
    try:
        import pyarrow.feather  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


_PARQUET_AVAILABLE: bool = _probe_parquet()
_FEATHER_AVAILABLE: bool = _probe_feather()

SUPPORTED_FORMATS: frozenset[str] = frozenset({"csv", "parquet", "feather"})
_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".ftr": "feather",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public option / result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ExportWriteOptions:
    """Fine-grained options for write_normalized_export().

    Fields
    ------
    include_metadata_sidecar  Write a JSON audit sidecar next to the data file.
    overwrite                 Replace existing files when True; fail when False.
    include_index             Include the DataFrame integer index as a column.
    timestamp_format          strftime format for the timestamp column (CSV only).
                              When None the default ISO-8601 format is used.
    float_precision           Decimal places for floating-point values (CSV only).
                              When None pandas uses its default representation.
    """

    include_metadata_sidecar: bool = True
    overwrite: bool = False
    include_index: bool = False
    timestamp_format: str | None = None
    float_precision: int | None = None


@dataclass
class ExportWriteResult:
    """Result of a write_normalized_export() call.

    Fields
    ------
    success              True when the data file was written without errors.
    output_path          Absolute path of the written file, or None on failure.
    format_used          Format string: 'csv', 'parquet', 'feather', or None.
    rows_written         Number of rows written to the data file.
    columns_written      Total columns written (including timestamp column).
    metadata_path        Path of the written sidecar, or None if not written.
    validation_messages  Warnings and errors accumulated during export.
    diagnostics_summary  One-line human-readable export summary.
    """

    success: bool
    output_path: str | None
    format_used: str | None
    rows_written: int
    columns_written: int
    metadata_path: str | None
    validation_messages: list[ValidationMessage]
    diagnostics_summary: str

    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.WARNING]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _err(code: str, message: str, column: str | None = None) -> ValidationMessage:
    return ValidationMessage(ValidationSeverity.ERROR, code, message, column)


def _warn(code: str, message: str, column: str | None = None) -> ValidationMessage:
    return ValidationMessage(ValidationSeverity.WARNING, code, message, column)


def _fail(
    messages: list[ValidationMessage],
    *extra: ValidationMessage,
) -> ExportWriteResult:
    return ExportWriteResult(
        success=False,
        output_path=None,
        format_used=None,
        rows_written=0,
        columns_written=0,
        metadata_path=None,
        validation_messages=list(messages) + list(extra),
        diagnostics_summary="Export failed during validation.",
    )


def _resolve_format(
    output_path: Path,
    explicit_format: str | None,
) -> tuple[str | None, ValidationMessage | None]:
    if explicit_format is not None:
        fmt = explicit_format.lower().lstrip(".")
        if fmt not in SUPPORTED_FORMATS:
            return None, _err(
                "EXPORT_UNSUPPORTED_FORMAT",
                f"Unsupported export format: '{explicit_format}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}.",
            )
        return fmt, None
    ext = output_path.suffix.lower()
    fmt = _EXT_TO_FORMAT.get(ext)
    if fmt is None:
        supported_exts = ", ".join(sorted(_EXT_TO_FORMAT))
        return None, _err(
            "EXPORT_UNSUPPORTED_FORMAT",
            f"Cannot determine format from extension '{ext}'. "
            f"Supported extensions: {supported_exts}.",
        )
    return fmt, None


def _check_dependency(fmt: str) -> ValidationMessage | None:
    if fmt == "parquet" and not _PARQUET_AVAILABLE:
        return _err(
            "EXPORT_DEPENDENCY_MISSING",
            "Parquet export requires 'pyarrow' or 'fastparquet'. "
            "Install one to enable Parquet export.",
        )
    if fmt == "feather" and not _FEATHER_AVAILABLE:
        return _err(
            "EXPORT_DEPENDENCY_MISSING",
            "Feather export requires 'pyarrow'. Install pyarrow to enable Feather export.",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def write_normalized_export(
    dataset: NormalizedDataset,
    output_path: str | Path,
    format: str | None = None,
    options: ExportWriteOptions | None = None,
) -> ExportWriteResult:
    """Serialize *dataset* to a file and optionally write a JSON metadata sidecar.

    Parameters
    ----------
    dataset       Normalized dataset to export.  Never mutated.
    output_path   Target file path.  Format is inferred from the extension when
                  *format* is None.
    format        Override format string: 'csv', 'parquet', or 'feather'.
    options       Write options.  Defaults are used when None.

    Returns
    -------
    ExportWriteResult — always returned, never raises on recoverable errors.
    """
    opts = options if options is not None else ExportWriteOptions()
    out = Path(output_path).resolve()
    messages: list[ValidationMessage] = []

    # ── 1. Resolve and validate format ───────────────────────────────────────
    _fmt, fmt_err = _resolve_format(out, format)
    if fmt_err is not None or _fmt is None:
        return _fail(messages, *([] if fmt_err is None else [fmt_err]))
    fmt: str = _fmt

    dep_err = _check_dependency(fmt)
    if dep_err is not None:
        return _fail(messages, dep_err)

    # ── 2. Validate dataset ──────────────────────────────────────────────────
    if dataset.data.empty:
        return _fail(
            messages,
            _err("EXPORT_EMPTY_DATASET", "Dataset is empty; nothing to export."),
        )
    if dataset.timestamp_column not in dataset.data.columns:
        return _fail(
            messages,
            _err(
                "EXPORT_NO_TIMESTAMP",
                f"Timestamp column '{dataset.timestamp_column}' is missing from dataset.",
            ),
        )
    if not dataset.parameters:
        return _fail(
            messages,
            _err("EXPORT_NO_DATA_COLUMNS", "Dataset has no data columns to export."),
        )

    # ── 3. Validate output path ──────────────────────────────────────────────
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _fail(
            messages,
            _err(
                "EXPORT_DIRECTORY_ERROR",
                f"Cannot create output directory '{out.parent}': {exc}",
            ),
        )

    if out.exists() and not opts.overwrite:
        return _fail(
            messages,
            _err(
                "EXPORT_FILE_EXISTS",
                f"Output file already exists: '{out}'. Set overwrite=True to replace.",
            ),
        )

    # ── 4. Prepare data (copy — never mutate the original) ───────────────────
    df = dataset.data.copy()

    ts_col = dataset.timestamp_column
    if fmt == "csv" and ts_col in df.columns:
        fmt_str = opts.timestamp_format or "%Y-%m-%d %H:%M:%S.%f"
        df[ts_col] = df[ts_col].dt.strftime(fmt_str)  # type: ignore[union-attr]

    # ── 5. Write data file ───────────────────────────────────────────────────
    try:
        if fmt == "csv":
            csv_kw: dict = {}
            if opts.float_precision is not None:
                csv_kw["float_format"] = f"%.{opts.float_precision}f"
            df.to_csv(out, index=opts.include_index, **csv_kw)
        elif fmt == "parquet":
            df.to_parquet(out, index=opts.include_index)
        else:  # feather
            df.to_feather(out)
    except OSError as exc:
        return _fail(
            messages,
            _err("EXPORT_WRITE_ERROR", f"Failed to write '{out}': {exc}"),
        )
    except Exception as exc:
        return _fail(
            messages,
            _err("EXPORT_WRITE_ERROR", f"Unexpected error writing '{out}': {exc}"),
        )

    written_rows = len(df)
    # include_index=True adds the index as an extra column in CSV/Parquet output
    written_cols = len(df.columns) + (1 if opts.include_index else 0)

    # ── 6. Metadata sidecar ──────────────────────────────────────────────────
    metadata_out: str | None = None
    if opts.include_metadata_sidecar:
        sidecar = metadata_sidecar_path(out)
        if sidecar.exists() and not opts.overwrite:
            messages.append(
                _warn(
                    "EXPORT_SIDECAR_EXISTS",
                    f"Metadata sidecar already exists: '{sidecar}'. "
                    "Set overwrite=True to replace.",
                )
            )
        else:
            try:
                exported_at = datetime.now(tz=timezone.utc).isoformat()
                meta = build_export_metadata(
                    dataset,
                    export_format=fmt,
                    output_path=str(out),
                    exported_at=exported_at,
                )
                write_metadata_sidecar(meta, sidecar)
                metadata_out = str(sidecar)
            except OSError as exc:
                messages.append(
                    _err(
                        "EXPORT_SIDECAR_WRITE_ERROR",
                        f"Failed to write metadata sidecar '{sidecar}': {exc}",
                    )
                )
            except Exception as exc:
                messages.append(
                    _err(
                        "EXPORT_SIDECAR_WRITE_ERROR",
                        f"Unexpected error writing metadata sidecar: {exc}",
                    )
                )

    # ── 7. Build result ──────────────────────────────────────────────────────
    summary = (
        f"Exported {written_rows:,} rows × {written_cols} column(s) "
        f"to {fmt.upper()} ({out.name})"
    )
    return ExportWriteResult(
        success=True,
        output_path=str(out),
        format_used=fmt,
        rows_written=written_rows,
        columns_written=written_cols,
        metadata_path=metadata_out,
        validation_messages=messages,
        diagnostics_summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExportPlan integration
# ─────────────────────────────────────────────────────────────────────────────


def write_from_export_plan(
    dataset: NormalizedDataset,
    plan: ExportPlan,
    *,
    format: str = "csv",
    options: ExportWriteOptions | None = None,
) -> ExportWriteResult:
    """Write *dataset* using the output path suggested by *plan*.

    Parameters
    ----------
    dataset  Normalized dataset to export.
    plan     ExportPlan produced by plan_export().  The suggested path for
             *format* is used as the output destination.
    format   Target format: 'csv' (default), 'parquet', or 'feather'.
    options  Write options.  Defaults are used when None.

    Returns
    -------
    ExportWriteResult — always returned, never raises on recoverable errors.
    """
    if not plan.is_export_ready:
        return _fail(
            [],
            _err(
                "EXPORT_NOT_READY",
                f"Dataset is not export-ready: {'; '.join(plan.readiness_issues)}",
            ),
        )
    path_map: dict[str, str | None] = {
        "csv": plan.suggested_csv_path,
        "parquet": plan.suggested_parquet_path,
        "feather": plan.suggested_feather_path,
    }
    output_path = path_map.get(format.lower())
    if output_path is None:
        return _fail(
            [],
            _err(
                "EXPORT_UNSUPPORTED_FORMAT",
                f"ExportPlan has no suggested path for format '{format}'.",
            ),
        )
    return write_normalized_export(dataset, output_path, format=format, options=options)
