"""End-to-end Import Wizard backend pipeline.

Entry point: run_import_pipeline()

Pipeline stages
---------------
1.  Validate file extension (early rejection of clearly unsupported types).
2.  Profile the source file (profile_import_file).
3.  Select the top-ranked timestamp candidate.
4.  Build a TimestampRepairPlan from the candidate's detected format.
5.  Load the full raw DataFrame.
6.  Normalize timestamps (normalize_timestamps).
7.  Build a NormalizationPlan from the profile's column mappings.
8.  Assemble a NormalizedDataset (assemble_normalized_dataset).
9.  Convert to a DisturbanceRecord via the conversion bridge.
10. Return an ImportPipelineResult with full diagnostics.

Design principles
-----------------
- Backend-only: no Qt, no GUI.
- Never mutates source file, raw DataFrame, profile objects, or NormalizedDataset.
- Always returns an ImportPipelineResult; never raises.
- Thin orchestration layer: delegates to existing phase modules.
- Full traceability: every intermediate result is preserved in the output.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.data_assembler import assemble_normalized_dataset
from app.import_wizard.disturbance_record_bridge import (
    BridgeResult,
    ConversionOptions,
    build_disturbance_record,
)
from app.import_wizard.file_profiler import (
    FileProfileResult,
    populate_session,
    profile_import_file,
)
from app.import_wizard.models import (
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.normalized_dataset import NormalizedDataset
from app.import_wizard.timestamp_contracts import (
    TimestampRepairPlan,
    TimestampRepairStrategy,
)
from app.import_wizard.timestamp_normalizer import (
    TimestampNormalizationResult,
    normalize_timestamps,
)
from app.models import DisturbanceRecord

_SUPPORTED_CSV_EXTENSIONS: frozenset[str] = frozenset({".csv", ".txt", ".tsv", ".dat"})
_SUPPORTED_EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm", ".xlsb"})
_SUPPORTED_EXTENSIONS: frozenset[str] = _SUPPORTED_CSV_EXTENSIONS | _SUPPORTED_EXCEL_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ImportPipelineOptions:
    """Configuration for run_import_pipeline().

    Fields
    ------
    nominal_frequency         Power system frequency in Hz (50 or 60).
    station_name              Override the station name in DisturbanceRecord metadata.
                              When None, the file stem is used.
    min_timestamp_confidence  Timestamp candidates with confidence below this value
                              are rejected.  0.0 accepts any candidate.
    drop_nat_rows             Drop rows whose timestamp is NaT after normalization.
    max_preview_rows          Rows included in the profiling preview snapshot.
    max_scan_rows             Rows scanned when searching for the CSV/Excel header.
    """

    nominal_frequency: float = 50.0
    station_name: str | None = None
    min_timestamp_confidence: float = 0.0
    drop_nat_rows: bool = True
    max_preview_rows: int = 50
    max_scan_rows: int = 200


@dataclass
class PipelineDiagnostics:
    """Lightweight, serializable summary of one pipeline run.

    All numeric fields default to zero and string fields to None when the
    pipeline fails before the relevant stage completes.
    """

    source_file_path: str
    provider_type: str
    selected_sheet: str | None = None
    detected_delimiter: str | None = None
    selected_timestamp_column: str | None = None
    timestamp_format: str | None = None
    normalized_row_count: int = 0
    analog_channel_count: int = 0
    digital_channel_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    elapsed_seconds: float | None = None


@dataclass
class ImportPipelineResult:
    """Complete result of a run_import_pipeline() call.

    All intermediate fields are included for traceability.  Fields that
    could not be produced (because the pipeline failed earlier) are None.

    Fields
    ------
    session             ImportWizardSession built from the pipeline run.
    profile             Raw file profiling result (always present).
    selected_candidate  Timestamp candidate chosen by auto-selection.
    repair_plan         TimestampRepairPlan built from the candidate.
    normalization_result  Timestamp normalization output (Series + diagnostics).
    dataset             Assembled NormalizedDataset.
    bridge_result       Full BridgeResult including warnings and validation errors.
    record              Final DisturbanceRecord (None when conversion failed).
    diagnostics         Serializable pipeline summary.
    success             True when a valid DisturbanceRecord was produced.
    validation_messages All warnings/errors accumulated across all pipeline stages.
    """

    session: ImportWizardSession
    profile: FileProfileResult
    selected_candidate: TimestampCandidate | None
    repair_plan: TimestampRepairPlan | None
    normalization_result: TimestampNormalizationResult | None
    dataset: NormalizedDataset | None
    bridge_result: BridgeResult | None
    record: DisturbanceRecord | None
    diagnostics: PipelineDiagnostics
    success: bool
    validation_messages: list[ValidationMessage] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_import_pipeline(
    path: str,
    provider_type: str | None = None,
    sheet_name: str | None = None,
    options: ImportPipelineOptions | None = None,
) -> ImportPipelineResult:
    """Run the complete Import Wizard backend pipeline for one CSV or Excel file.

    Parameters
    ----------
    path            Absolute path to the source file.
    provider_type   "csv" or "excel".  When None, inferred from the file extension.
    sheet_name      Excel sheet name to load.  Ignored for CSV files.
    options         Pipeline configuration.  Defaults are used when None.

    Returns
    -------
    ImportPipelineResult — always returns, never raises.  Check ``success`` and
    ``validation_messages`` for any issues.  Intermediate results are preserved
    in the result for full traceability.
    """
    start_t = time.monotonic()
    opts = options or ImportPipelineOptions()
    messages: list[ValidationMessage] = []

    # ── Stage 1: Extension validation ─────────────────────────────────────────
    ext_err = _check_supported_extension(path)
    if ext_err:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PIPELINE_UNSUPPORTED_TYPE",
            message=ext_err,
        ))
        eff_provider = provider_type or "csv"
        stub_profile = FileProfileResult(
            raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
        )
        stub_session = ImportWizardSession(source_path=path, provider_type=eff_provider)
        elapsed = time.monotonic() - start_t
        diag = PipelineDiagnostics(
            source_file_path=path,
            provider_type=eff_provider,
            error_count=1,
            elapsed_seconds=elapsed,
        )
        return _make_result(
            stub_session, stub_profile, None, None, None, None, None, None,
            diag, False, messages,
        )

    # ── Stage 2: Profile the file ─────────────────────────────────────────────
    profile = profile_import_file(
        path,
        sheet_name=sheet_name,
        max_preview_rows=opts.max_preview_rows,
        max_scan_rows=opts.max_scan_rows,
    )
    messages.extend(profile.validation_messages)

    effective_provider = provider_type or profile.provider_type

    session = ImportWizardSession(
        source_path=path,
        provider_type=effective_provider,
        sheet_name=profile.sheet_name,
        delimiter=profile.delimiter,
    )
    populate_session(session, profile)

    if profile.has_errors():
        diag = _build_diagnostics(
            path, effective_provider, profile, None, None, None, messages,
            time.monotonic() - start_t,
        )
        return _make_result(
            session, profile, None, None, None, None, None, None,
            diag, False, messages,
        )

    # ── Stage 3: Timestamp candidate selection ────────────────────────────────
    if not profile.timestamp_candidates:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PIPELINE_NO_TIMESTAMP",
            message="No timestamp candidate was detected in the file.",
        ))
        diag = _build_diagnostics(
            path, effective_provider, profile, None, None, None, messages,
            time.monotonic() - start_t,
        )
        return _make_result(
            session, profile, None, None, None, None, None, None,
            diag, False, messages,
        )

    candidate = profile.timestamp_candidates[0]

    if candidate.confidence < opts.min_timestamp_confidence:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PIPELINE_LOW_TS_CONFIDENCE",
            message=(
                f"Best timestamp candidate '{candidate.column_name}' has confidence "
                f"{candidate.confidence:.2f}, below minimum {opts.min_timestamp_confidence:.2f}."
            ),
        ))
        diag = _build_diagnostics(
            path, effective_provider, profile, candidate, None, None, messages,
            time.monotonic() - start_t,
        )
        return _make_result(
            session, profile, candidate, None, None, None, None, None,
            diag, False, messages,
        )

    session.selected_timestamp_column = candidate.column_name

    # ── Stage 4: Build timestamp repair plan ──────────────────────────────────
    repair_plan = _build_repair_plan(candidate)
    session.timestamp_repair_plan = repair_plan

    # ── Stage 5: Load full raw DataFrame ──────────────────────────────────────
    try:
        raw_df = _load_full_dataframe(path, effective_provider, profile)
    except Exception as exc:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PIPELINE_LOAD_FAILED",
            message=f"Failed to load full dataset: {exc}",
        ))
        diag = _build_diagnostics(
            path, effective_provider, profile, candidate, None, None, messages,
            time.monotonic() - start_t,
        )
        return _make_result(
            session, profile, candidate, repair_plan, None, None, None, None,
            diag, False, messages,
        )

    if candidate.column_name not in raw_df.columns:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PIPELINE_TS_COLUMN_MISSING",
            message=(
                f"Timestamp column '{candidate.column_name}' was detected in the "
                "preview but is absent from the loaded data."
            ),
        ))
        diag = _build_diagnostics(
            path, effective_provider, profile, candidate, None, None, messages,
            time.monotonic() - start_t,
        )
        return _make_result(
            session, profile, candidate, repair_plan, None, None, None, None,
            diag, False, messages,
        )

    # ── Stage 6: Normalize timestamps ─────────────────────────────────────────
    norm_result = normalize_timestamps(raw_df[candidate.column_name], repair_plan)
    messages.extend(norm_result.validation_messages)

    # ── Stage 7: Build normalization plan ─────────────────────────────────────
    norm_plan = _build_normalization_plan(profile, candidate, repair_plan, messages)
    session.normalization_plan = norm_plan

    if not norm_plan.selected_columns:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="PIPELINE_NO_DATA_COLUMNS",
            message=(
                "No data columns were selected after excluding the timestamp column. "
                "The resulting DisturbanceRecord will have no analog or digital channels."
            ),
        ))

    # ── Stage 8: Assemble NormalizedDataset ───────────────────────────────────
    dataset = assemble_normalized_dataset(
        raw_df,
        norm_plan,
        norm_result,
        column_mappings=profile.column_mappings,
        source_path=path,
        drop_nat_rows=opts.drop_nat_rows,
    )
    messages.extend(dataset.validation_messages)

    # ── Stage 9: Convert to DisturbanceRecord ─────────────────────────────────
    bridge_opts = ConversionOptions(
        nominal_frequency=opts.nominal_frequency,
        station_name=opts.station_name,
    )
    bridge_result = build_disturbance_record(dataset, options=bridge_opts)

    for w in bridge_result.warnings:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code="BRIDGE_WARNING",
            message=w,
        ))
    for e in bridge_result.validation_errors:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="BRIDGE_VALIDATION_ERROR",
            message=e,
        ))

    record = bridge_result.record if bridge_result.success else None
    success = bridge_result.success and dataset.is_valid

    # ── Stage 10: Diagnostics ─────────────────────────────────────────────────
    elapsed = time.monotonic() - start_t
    diag = _build_diagnostics(
        path, effective_provider, profile, candidate, dataset, bridge_result, messages, elapsed,
    )

    return _make_result(
        session, profile, candidate, repair_plan, norm_result,
        dataset, bridge_result, record, diag, success, messages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_supported_extension(path: str) -> str | None:
    """Return an error string for unsupported file extensions, or None."""
    ext = os.path.splitext(path)[1].lower()
    if ext and ext not in _SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file extension '{ext}'. "
            "Supported: .csv .txt .tsv .dat .xlsx .xls .xlsm .xlsb"
        )
    return None


def _build_repair_plan(candidate: TimestampCandidate) -> TimestampRepairPlan:
    """Construct a TimestampRepairPlan from a detected timestamp candidate.

    Strategy selection:
    - excel_serial          → EXCEL_SERIAL_CONVERSION
    - epoch_seconds / _ms   → PARSE_DETECTED_FORMAT (executor handles unit conversion)
    - strptime format       → PARSE_DETECTED_FORMAT
    - None (no format)      → NO_REPAIR (pandas auto-detection)
    """
    fmt = candidate.detected_format

    if fmt == "excel_serial":
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION,
            repair_validated=True,
            repair_notes="Auto-selected: Excel serial date column detected.",
        )

    if fmt in ("epoch_seconds", "epoch_milliseconds"):
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format=fmt,
            repair_validated=True,
            repair_notes=f"Auto-selected: {fmt} column detected.",
        )

    if fmt is not None:
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format=fmt,
            repair_validated=True,
            repair_notes=f"Auto-selected: strptime format '{fmt}' detected.",
        )

    return TimestampRepairPlan(
        strategy=TimestampRepairStrategy.NO_REPAIR,
        repair_validated=True,
        repair_notes="No specific format detected; using pandas auto-detection.",
    )


def _build_normalization_plan(
    profile: FileProfileResult,
    candidate: TimestampCandidate,
    repair_plan: TimestampRepairPlan,
    messages: list[ValidationMessage],
) -> NormalizationPlan:
    """Build a NormalizationPlan from auto-detected column mappings.

    Excludes the timestamp column and any column already marked excluded or
    classified as TIMESTAMP type.  Emits a WARNING for UNKNOWN-type columns
    so callers know they were preserved as analog.
    """
    ts_col = candidate.column_name
    selected: list[str] = []
    excluded: list[str] = []
    renames: dict[str, str] = {}
    units: dict[str, str] = {}
    types: dict[str, ParameterType] = {}

    for mapping in profile.column_mappings:
        src = mapping.source_name

        if src == ts_col or mapping.effective_type == ParameterType.TIMESTAMP:
            excluded.append(src)
            continue

        if mapping.excluded:
            excluded.append(src)
            continue

        selected.append(src)
        effective = mapping.effective_name

        if effective != src:
            renames[src] = effective
        if mapping.effective_unit:
            units[effective] = mapping.effective_unit
        types[effective] = mapping.effective_type

        if mapping.effective_type == ParameterType.UNKNOWN:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="PIPELINE_UNKNOWN_COLUMN",
                message=(
                    f"Column '{src}' has unknown type and will be treated "
                    "as an analog channel."
                ),
                affected_column=src,
            ))

    return NormalizationPlan(
        timestamp_plan=repair_plan,
        selected_columns=selected,
        excluded_columns=excluded,
        column_renames=renames,
        column_units=units,
        column_types=types,
    )


def _load_full_dataframe(
    path: str,
    provider_type: str,
    profile: FileProfileResult,
) -> pd.DataFrame:
    """Load the full raw DataFrame, respecting the profiler's header row detection."""
    header_row = profile.raw_preview.header_row_index or 0

    if provider_type == "excel":
        sheet: str | int = profile.sheet_name if profile.sheet_name is not None else 0
        return pd.read_excel(path, sheet_name=sheet, header=header_row)

    delimiter = profile.delimiter or ","
    return pd.read_csv(path, sep=delimiter, header=header_row)


def _build_diagnostics(
    path: str,
    provider_type: str,
    profile: FileProfileResult,
    candidate: TimestampCandidate | None,
    dataset: NormalizedDataset | None,
    bridge_result: BridgeResult | None,
    messages: list[ValidationMessage],
    elapsed: float,
) -> PipelineDiagnostics:
    analog_count = 0
    digital_count = 0
    if bridge_result is not None:
        analog_count = len(bridge_result.record.analog_channels)
        digital_count = len(bridge_result.record.digital_channels)

    return PipelineDiagnostics(
        source_file_path=path,
        provider_type=provider_type,
        selected_sheet=profile.sheet_name,
        detected_delimiter=profile.delimiter,
        selected_timestamp_column=candidate.column_name if candidate else None,
        timestamp_format=candidate.detected_format if candidate else None,
        normalized_row_count=dataset.diagnostics.normalized_rows if dataset else 0,
        analog_channel_count=analog_count,
        digital_channel_count=digital_count,
        warning_count=sum(1 for m in messages if m.severity == ValidationSeverity.WARNING),
        error_count=sum(1 for m in messages if m.severity == ValidationSeverity.ERROR),
        elapsed_seconds=elapsed,
    )


def _make_result(
    session: ImportWizardSession,
    profile: FileProfileResult,
    candidate: TimestampCandidate | None,
    repair_plan: TimestampRepairPlan | None,
    norm_result: TimestampNormalizationResult | None,
    dataset: NormalizedDataset | None,
    bridge_result: BridgeResult | None,
    record: DisturbanceRecord | None,
    diag: PipelineDiagnostics,
    success: bool,
    messages: list[ValidationMessage],
) -> ImportPipelineResult:
    return ImportPipelineResult(
        session=session,
        profile=profile,
        selected_candidate=candidate,
        repair_plan=repair_plan,
        normalization_result=norm_result,
        dataset=dataset,
        bridge_result=bridge_result,
        record=record,
        diagnostics=diag,
        success=success,
        validation_messages=messages,
    )
