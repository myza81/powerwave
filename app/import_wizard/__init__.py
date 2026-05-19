"""Import Wizard — architecture, data contracts, file profiling, and data assembly.

Public surface
--------------
Models:
    ImportWizardSession      Complete state for one import workflow.
    RawPreviewModel          Raw file structure preview.
    TimestampCandidate       One candidate time-axis column with confidence score.
    ColumnMappingCandidate   One data column with classification + user overrides.
    NormalizationPlan        Final executable normalization contract.
    TimestampRepairPlan      Timestamp repair parameters (strategy + config).
    FileProfileResult        Result of profiling one CSV or Excel file.
    NormalizedDataset        Intermediate engineering dataset after assembly.
    ParameterMetadata        Per-column metadata in a NormalizedDataset.
    AssemblyDiagnostics      Quantitative statistics from data assembly.
    ExportPlan               Export metadata and readiness assessment.

Enumerations:
    WizardStep               Ordered wizard stages.
    TimestampRepairStrategy  How the repair engine should handle the time column.
    ParameterType            Semantic classification of a data column.
    ValidationSeverity       info / warning / error.

Validation:
    ValidationMessage        Structured feedback with severity, code, column, action.

State helpers:
    can_transition(from, to) Whether a step transition is permitted.
    next_step(step)          The immediately following step (or None).
    steps_before(step)       All steps preceding the given step.
    step_index(step)         Zero-based ordinal of a step.

Profiling:
    profile_import_file(path, ...)    Auto-detect CSV/Excel and build a FileProfileResult.
    populate_session(session, result) Write profile results into an ImportWizardSession.

Assembly (Phase 8.55D):
    assemble_normalized_dataset(...)  Produce a NormalizedDataset from raw data.
    validate_normalized_dataset(...)  Re-validate a NormalizedDataset independently.
    plan_export(dataset, ...)         Generate export filenames and readiness info.

Conversion bridge (Phase 8.55E):
    BridgeResult                      Result of NormalizedDataset → DisturbanceRecord conversion.
    ConversionOptions                 Conversion settings (frequency, station name, …).
    build_disturbance_record(...)     Full conversion with BridgeResult (warnings + errors).
    convert_normalized_dataset_to_record(...)  High-level wrapper; raises on failure.

End-to-end pipeline (Phase 8.55F):
    ImportPipelineOptions             Pipeline configuration.
    ImportPipelineResult              Complete pipeline result with all intermediate outputs.
    PipelineDiagnostics               Serializable pipeline summary.
    run_import_pipeline(...)          Full CSV/Excel → DisturbanceRecord pipeline.

Plan-aware pipeline (Phase 8.55H):
    PlanBuildResult                   Result of build_execution_plan().
    build_execution_plan(...)         GUI/session state → validated PlanBuildResult.
    run_import_pipeline_with_plan(...)  Authoritative plan → DisturbanceRecord pipeline.

Normalized export writer (Phase 8.55K):
    ExportWriteOptions                Fine-grained write options (format, overwrite, sidecar).
    ExportWriteResult                 Result of write_normalized_export().
    ExportMetadata                    Audit sidecar model with full traceability.
    write_normalized_export(...)      Serialize NormalizedDataset → CSV/Parquet/Feather + sidecar.
    write_from_export_plan(...)       Write using paths suggested by an ExportPlan.
    build_export_metadata(...)        Build audit metadata without writing.
    metadata_sidecar_path(...)        Derive sidecar path from a data file path.
"""
from __future__ import annotations

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.data_assembler import assemble_normalized_dataset
from app.import_wizard.dataset_validation import validate_normalized_dataset
from app.import_wizard.disturbance_record_bridge import (
    BridgeResult,
    ConversionOptions,
    build_disturbance_record,
    convert_normalized_dataset_to_record,
)
from app.import_wizard.export_metadata import (
    ExportMetadata,
    build_export_metadata,
    metadata_sidecar_path,
)
from app.import_wizard.diagnostics_summary import (
    ImportDiagnosticsSummary,
    build_import_diagnostics,
)
from app.import_wizard.export_planner import ExportPlan, plan_export
from app.import_wizard.export_writer import (
    ExportWriteOptions,
    ExportWriteResult,
    write_from_export_plan,
    write_normalized_export,
)
from app.import_wizard.import_pipeline import (
    ImportPipelineOptions,
    ImportPipelineResult,
    PipelineDiagnostics,
    run_import_pipeline,
    run_import_pipeline_with_plan,
)
from app.import_wizard.pipeline_plan_builder import (
    PlanBuildResult,
    build_execution_plan,
)
from app.import_wizard.file_profiler import FileProfileResult, populate_session, profile_import_file
from app.import_wizard.models import (
    ColumnMappingCandidate,
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.import_wizard.timestamp_format_validator import (
    TimestampFormatValidationResult,
    validate_user_timestamp_format,
)
from app.import_wizard.wizard_state import (
    WizardStep,
    can_transition,
    next_step,
    step_index,
    steps_before,
)

__all__ = [
    # Models
    "AssemblyDiagnostics",
    "BridgeResult",
    "ColumnMappingCandidate",
    "ConversionOptions",
    "ExportPlan",
    "ImportPipelineOptions",
    "ImportPipelineResult",
    "PipelineDiagnostics",
    "FileProfileResult",
    "ImportWizardSession",
    "NormalizationPlan",
    "NormalizedDataset",
    "ParameterMetadata",
    "ParameterType",
    "RawPreviewModel",
    "TimestampCandidate",
    "TimestampRepairPlan",
    "TimestampRepairStrategy",
    "TimestampFormatValidationResult",
    "ValidationMessage",
    "ValidationSeverity",
    "WizardStep",
    # Functions
    "assemble_normalized_dataset",
    "build_disturbance_record",
    "build_execution_plan",
    "can_transition",
    "convert_normalized_dataset_to_record",
    "run_import_pipeline",
    "run_import_pipeline_with_plan",
    "next_step",
    "plan_export",
    "populate_session",
    "profile_import_file",
    "step_index",
    "steps_before",
    "validate_normalized_dataset",
    "validate_user_timestamp_format",
    # Phase 8.55H
    "PlanBuildResult",
    # Phase 8.55N — diagnostics
    "ImportDiagnosticsSummary",
    "build_import_diagnostics",
    # Phase 8.55K — export writer
    "ExportMetadata",
    "ExportWriteOptions",
    "ExportWriteResult",
    "build_export_metadata",
    "metadata_sidecar_path",
    "write_from_export_plan",
    "write_normalized_export",
]
