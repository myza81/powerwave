"""Import diagnostics summary — lightweight aggregation for engineering UI.

build_import_diagnostics() aggregates existing backend results (ImportPipelineResult
and optional ExportWriteResult) into an ImportDiagnosticsSummary.  No raw data
is re-read or recomputed.

The summary drives DiagnosticsPanel in the Import Wizard complete page and
provides a structured view of what happened to the imported data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity

# Forward-declared to avoid heavy imports at module level; resolved at call time.
# (Both modules are already imported transitively when the wizard runs.)

_LARGE_FILE_ROW_THRESHOLD = 100_000

_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "no_repair": "No repair needed — timestamps already valid",
    "parse_detected_format": "Parsed using auto-detected format",
    "parse_user_format": "Parsed using user-supplied format",
    "interpolate_missing": "Missing timestamps interpolated",
    "reconstruct_from_interval": "Timestamps reconstructed from sampling interval",
    "combine_date_time_columns": "Date and time columns merged",
    "excel_serial_conversion": "Excel date serial numbers converted",
    "timezone_alignment": "Timestamps converted to target timezone",
}


def _confidence_label(confidence: float | None) -> str:
    """Return a human-readable confidence tier label."""
    if confidence is None:
        return "N/A"
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.60:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# Summary dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ImportDiagnosticsSummary:
    """Engineering-readable aggregated diagnostics for one completed import.

    All fields are derived from ImportPipelineResult — nothing is recomputed
    from raw data.  Designed to drive DiagnosticsPanel and any future
    diagnostics surfaces.

    Fields
    ------
    source_file_name        Basename of the imported source file.
    provider_type           'csv' or 'excel'.
    success                 Whether the pipeline completed successfully.

    -- Timestamp --
    timestamp_column        Name of the selected time-axis column.
    timestamp_format        Format string used (user override or auto-detected).
    timestamp_format_source 'auto-detected', 'user override', or 'none'.
    timestamp_strategy      Human-readable repair strategy description.
    timestamp_confidence    Detection confidence in [0.0, 1.0], or None.
    timestamp_confidence_label  'High', 'Medium', 'Low', or 'N/A'.
    repair_actions          Repair action descriptions in engineering language.

    -- Data quality --
    total_rows              Rows in the source before filtering.
    normalized_rows         Rows in the assembled output.
    dropped_rows            Rows removed due to unrecoverable timestamps.
    duplicate_timestamps    Duplicate timestamp count retained in output.
    invalid_value_count     Total NaN count across data columns.

    -- Channels --
    analog_channels         Analog channel count in DisturbanceRecord.
    digital_channels        Digital channel count in DisturbanceRecord.
    excluded_column_count   Source columns explicitly excluded.
    user_overridden_count   Columns where user changed name, type, or unit.
    low_confidence_columns  Canonical names with classification confidence < 0.60.
    classification_confidence_label  Overall classification confidence tier.

    -- Validation --
    errors / warnings / infos  Messages grouped by severity.

    -- Guidance --
    large_file_guidance     Operational tips for large imports (empty list if small).
    export_guidance         Tips about the normalized export.

    -- Timing --
    import_duration_s       Total import duration in seconds (None if unknown).
    """

    source_file_name: str | None
    provider_type: str
    success: bool

    # Timestamp
    timestamp_column: str | None
    timestamp_format: str | None
    timestamp_format_source: str
    timestamp_strategy: str
    timestamp_confidence: float | None
    timestamp_confidence_label: str
    repair_actions: list[str]

    # Data quality
    total_rows: int
    normalized_rows: int
    dropped_rows: int
    duplicate_timestamps: int
    invalid_value_count: int

    # Channels
    analog_channels: int
    digital_channels: int
    excluded_column_count: int
    user_overridden_count: int
    low_confidence_columns: list[str]
    classification_confidence_label: str

    # Validation
    errors: list[ValidationMessage]
    warnings: list[ValidationMessage]
    infos: list[ValidationMessage]

    # Guidance
    large_file_guidance: list[str]
    export_guidance: list[str]

    # Timing
    import_duration_s: float | None

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def data_completeness_pct(self) -> float | None:
        """Percentage of source rows retained (0–100), or None if unknown."""
        if self.total_rows <= 0:
            return None
        return round(100.0 * self.normalized_rows / self.total_rows, 1)

    @property
    def has_data_loss(self) -> bool:
        """True when any rows were dropped during import."""
        return self.dropped_rows > 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_repair_actions(result) -> list[str]:  # type: ignore[type-arg]
    """Build human-readable repair action strings from pipeline result."""
    actions: list[str] = []
    dataset = result.dataset

    if dataset is not None:
        asm = dataset.diagnostics
        if asm.dropped_rows > 0:
            s = "s" if asm.dropped_rows != 1 else ""
            actions.append(
                f"{asm.dropped_rows:,} row{s} removed due to unrecoverable timestamps"
            )
        if asm.duplicate_timestamp_count > 0:
            s = "s" if asm.duplicate_timestamp_count != 1 else ""
            actions.append(
                f"{asm.duplicate_timestamp_count:,} duplicate timestamp{s} detected"
            )

    repair_plan = result.repair_plan
    if repair_plan is not None:
        strategy_val = (
            repair_plan.strategy.value
            if hasattr(repair_plan.strategy, "value")
            else str(repair_plan.strategy)
        )
        if strategy_val == "interpolate_missing":
            actions.append("Missing timestamps were interpolated")
        elif strategy_val == "reconstruct_from_interval":
            interval = getattr(repair_plan, "sampling_interval_seconds", None)
            if interval is not None:
                actions.append(
                    f"Timestamps reconstructed from {interval:.4g} s sampling interval"
                )
            else:
                actions.append("Timestamps reconstructed from detected sampling interval")
        elif strategy_val == "combine_date_time_columns":
            actions.append("Date and time columns merged into a single timestamp")
        elif strategy_val == "excel_serial_conversion":
            actions.append("Excel date serial numbers converted to datetime")

        if repair_plan.user_format:
            actions.append(f"User-specified format applied: {repair_plan.user_format}")

    return actions


def _build_large_file_guidance(result) -> list[str]:  # type: ignore[type-arg]
    guidance: list[str] = []
    row_estimate = 0
    profile = result.profile
    if profile is not None and profile.raw_preview is not None:
        row_estimate = profile.raw_preview.row_count_estimate
    if row_estimate <= 0:
        row_estimate = result.diagnostics.normalized_row_count

    if row_estimate >= _LARGE_FILE_ROW_THRESHOLD:
        guidance.append(f"Large dataset detected (~{row_estimate:,} rows)")
        guidance.append(
            "Consider exporting to Parquet or Feather for significantly faster reload"
        )
        guidance.append(
            "The waveform viewer uses viewport rendering — only visible rows are rendered at a time"
        )
    return guidance


def _build_export_guidance(result, export_result) -> list[str]:  # type: ignore[type-arg]
    guidance: list[str] = []

    if result.dataset is not None and result.dataset.is_export_ready():
        guidance.append(
            "Metadata sidecar preserves repair strategy, canonical names, and source traceability"
        )
        guidance.append(
            "Exported file uses canonical channel names for consistent downstream processing"
        )
        guidance.append(
            "Original source column names are retained in the metadata sidecar"
        )

    if export_result is not None:
        if export_result.success:
            guidance.append(
                f"Normalized export written: {export_result.rows_written:,} rows"
                + (f" to {export_result.format_used.upper()}" if export_result.format_used else "")
            )
            if export_result.metadata_path:
                guidance.append("Audit metadata sidecar written alongside the data file")
        else:
            for m in export_result.errors():
                guidance.append(f"Export issue: {m.message}")

    return guidance


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_import_diagnostics(result, export_result=None) -> ImportDiagnosticsSummary:  # type: ignore[type-arg]
    """Build an engineering-readable diagnostics summary from a pipeline result.

    Does not re-read the source file or execute any backend computation.

    Parameters
    ----------
    result         Completed ImportPipelineResult.
    export_result  Optional ExportWriteResult to include in export guidance.

    Returns
    -------
    ImportDiagnosticsSummary with all sections populated.
    """
    # Import here to avoid module-level circular dependency scans
    from app.import_wizard.import_pipeline import ImportPipelineResult

    diag = result.diagnostics
    dataset = result.dataset

    # ── Source file name ─────────────────────────────────────────────────────
    source_file_name: str | None = None
    if dataset is not None and dataset.source_file_name:
        source_file_name = dataset.source_file_name
    elif result.session is not None:
        source_file_name = os.path.basename(result.session.source_path)

    # ── Timestamp ────────────────────────────────────────────────────────────
    candidate = result.selected_candidate
    repair_plan = result.repair_plan

    ts_column = candidate.column_name if candidate is not None else diag.selected_timestamp_column
    ts_confidence = candidate.confidence if candidate is not None else None
    ts_confidence_label = _confidence_label(ts_confidence)

    # Prefer diagnostics.timestamp_format (actual format used by the pipeline)
    # over candidate.detected_format (initial detection).
    if repair_plan is not None:
        strategy_val = (
            repair_plan.strategy.value
            if hasattr(repair_plan.strategy, "value")
            else str(repair_plan.strategy)
        )
        ts_strategy = _STRATEGY_DESCRIPTIONS.get(strategy_val, strategy_val)
        if repair_plan.user_format:
            ts_format: str | None = repair_plan.user_format
            ts_format_source = "user override"
        elif repair_plan.detected_format:
            ts_format = repair_plan.detected_format
            ts_format_source = "auto-detected"
        else:
            ts_format = diag.timestamp_format or None
            ts_format_source = "auto-detected" if ts_format else "none"
    elif candidate is not None:
        ts_strategy = _STRATEGY_DESCRIPTIONS.get("parse_detected_format", "Parsed using auto-detected format")
        ts_format = diag.timestamp_format or candidate.detected_format
        ts_format_source = "auto-detected" if ts_format else "none"
    else:
        ts_strategy = "Unknown"
        ts_format = diag.timestamp_format or None
        ts_format_source = "auto-detected" if ts_format else "none"

    repair_actions = _build_repair_actions(result)

    # ── Data quality ─────────────────────────────────────────────────────────
    if dataset is not None:
        asm = dataset.diagnostics
        total_rows = asm.total_rows
        normalized_rows = asm.normalized_rows
        dropped_rows = asm.dropped_rows
        duplicate_timestamps = asm.duplicate_timestamp_count
        invalid_value_count = asm.invalid_value_count
    else:
        total_rows = diag.normalized_row_count
        normalized_rows = diag.normalized_row_count
        dropped_rows = 0
        duplicate_timestamps = 0
        invalid_value_count = 0

    # ── Channels ─────────────────────────────────────────────────────────────
    excluded_count = 0
    user_overridden_count = 0
    low_confidence_columns: list[str] = []

    if dataset is not None:
        excluded_count = len(dataset.excluded_columns)
        for p in dataset.parameters:
            if p.user_overridden:
                user_overridden_count += 1
            if p.confidence < 0.60:
                low_confidence_columns.append(p.canonical_name)
        if low_confidence_columns and dataset.parameters:
            low_ratio = len(low_confidence_columns) / len(dataset.parameters)
            classification_confidence_label = "Low" if low_ratio > 0.5 else "Medium"
        elif dataset.parameters:
            avg_conf = sum(p.confidence for p in dataset.parameters) / len(dataset.parameters)
            classification_confidence_label = _confidence_label(avg_conf)
        else:
            classification_confidence_label = "N/A"
    else:
        classification_confidence_label = "N/A"

    # ── Validation ───────────────────────────────────────────────────────────
    all_msgs = list(result.validation_messages)
    errors = [m for m in all_msgs if m.severity == ValidationSeverity.ERROR]
    warnings_ = [m for m in all_msgs if m.severity == ValidationSeverity.WARNING]
    infos = [m for m in all_msgs if m.severity == ValidationSeverity.INFO]

    # ── Guidance ─────────────────────────────────────────────────────────────
    large_file_guidance = _build_large_file_guidance(result)
    export_guidance = _build_export_guidance(result, export_result)

    return ImportDiagnosticsSummary(
        source_file_name=source_file_name,
        provider_type=diag.provider_type,
        success=result.success,
        timestamp_column=ts_column,
        timestamp_format=ts_format,
        timestamp_format_source=ts_format_source,
        timestamp_strategy=ts_strategy,
        timestamp_confidence=ts_confidence,
        timestamp_confidence_label=ts_confidence_label,
        repair_actions=repair_actions,
        total_rows=total_rows,
        normalized_rows=normalized_rows,
        dropped_rows=dropped_rows,
        duplicate_timestamps=duplicate_timestamps,
        invalid_value_count=invalid_value_count,
        analog_channels=diag.analog_channel_count,
        digital_channels=diag.digital_channel_count,
        excluded_column_count=excluded_count,
        user_overridden_count=user_overridden_count,
        low_confidence_columns=low_confidence_columns,
        classification_confidence_label=classification_confidence_label,
        errors=errors,
        warnings=warnings_,
        infos=infos,
        large_file_guidance=large_file_guidance,
        export_guidance=export_guidance,
        import_duration_s=getattr(diag, "elapsed_seconds", None),
    )
