"""Export planning for normalized datasets.

plan_export() generates suggested output filenames and an export readiness
assessment without writing any files.  Actual file writers are a future phase.

Supported future formats: CSV, Parquet, Feather.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.import_wizard.normalized_dataset import NormalizedDataset


@dataclass(slots=True)
class ExportPlan:
    """Export metadata for a NormalizedDataset.

    Fields
    ------
    suggested_csv_path      Suggested .csv output path.
    suggested_parquet_path  Suggested .parquet output path.
    suggested_feather_path  Suggested .feather output path.
    row_count               Rows in the normalized dataset.
    column_count            Data columns (excludes the timestamp column).
    is_export_ready         True when the dataset is valid and non-empty.
    readiness_issues        Human-readable issues blocking export (empty if ready).
    export_summary          One-line summary string for logging or UI display.
    """

    suggested_csv_path: str | None
    suggested_parquet_path: str | None
    suggested_feather_path: str | None
    row_count: int
    column_count: int
    is_export_ready: bool
    readiness_issues: list[str] = field(default_factory=list)
    export_summary: str = ""


def _stem_from_source(source_file_name: str | None) -> str:
    """Extract a clean filename stem from a source file name.

    Strips the extension, sanitises non-alphanumeric characters, and falls
    back to "dataset" if the result is empty.
    """
    if not source_file_name:
        return "dataset"
    base = os.path.basename(source_file_name)
    stem, _ = os.path.splitext(base)
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).strip("_")
    return clean or "dataset"


def plan_export(
    dataset: NormalizedDataset,
    *,
    output_dir: str | None = None,
) -> ExportPlan:
    """Generate an export plan for a NormalizedDataset.

    Does not write any files.

    Parameters
    ----------
    dataset     The normalized dataset to plan export for.
    output_dir  Directory for suggested output paths.  When None, uses the
                source file's directory if known, otherwise ".".

    Returns
    -------
    ExportPlan with suggested filenames, readiness assessment, and summary.
    """
    stem = _stem_from_source(dataset.source_file_name)
    normalized_stem = f"{stem}_normalized"

    if output_dir is not None:
        base_dir = output_dir
    elif dataset.source_path is not None:
        base_dir = os.path.dirname(os.path.abspath(dataset.source_path))
    else:
        base_dir = "."

    def _path(ext: str) -> str:
        return os.path.join(base_dir, f"{normalized_stem}{ext}")

    # ── Readiness checks ───────────────────────────────────────────────────────
    issues: list[str] = []
    if not dataset.is_valid:
        issues.append("Dataset is not valid (check validation_messages).")
    if dataset.data.empty:
        issues.append("Dataset is empty.")
    if not dataset.parameters:
        issues.append("No data columns included.")
    if dataset.diagnostics.duplicate_timestamp_count > 0:
        issues.append(
            f"{dataset.diagnostics.duplicate_timestamp_count} "
            "duplicate timestamp(s) present."
        )

    is_ready = dataset.is_export_ready()

    # ── Summary ────────────────────────────────────────────────────────────────
    row_count = len(dataset.data)
    col_count = len(dataset.parameters)
    ts_strategy = dataset.timestamp_repair_strategy
    summary = (
        f"{row_count:,} rows × {col_count} channel(s); "
        f"timestamp strategy: {ts_strategy}"
    )
    if dataset.source_file_name:
        summary = f"{dataset.source_file_name} → {summary}"

    return ExportPlan(
        suggested_csv_path=_path(".csv"),
        suggested_parquet_path=_path(".parquet"),
        suggested_feather_path=_path(".feather"),
        row_count=row_count,
        column_count=col_count,
        is_export_ready=is_ready,
        readiness_issues=issues,
        export_summary=summary,
    )
