"""Core import wizard models.

ImportWizardSession  — one end-to-end import workflow.
RawPreviewModel      — raw file preview metadata (no full load required).
TimestampCandidate   — one candidate timestamp column with confidence score.
ColumnMappingCandidate — one data column with classification + user overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.timestamp_contracts import TimestampRepairPlan
from app.import_wizard.wizard_state import WizardStep, can_transition


# ─────────────────────────────────────────────────────────────────────────────
# RawPreviewModel
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RawPreviewModel:
    """Lightweight snapshot of the raw file structure.

    Never requires loading the entire file — callers should limit preview_rows
    to a small head slice (e.g. 20–100 rows).

    Fields
    ------
    column_names        Header names as read from the file.
    preview_rows        List of rows, each row is a list of raw cell values.
    header_row_index    Zero-based index of the row that contains column names.
    skipped_row_count   Number of metadata/blank rows before the header.
    row_count_estimate  Total data row estimate (may be approximate for large files).
    sheet_name          Active sheet name for Excel sources; None for CSV.
    parse_warnings      Non-fatal issues encountered during preview extraction.
    """

    column_names: list[str]
    preview_rows: list[list[Any]]
    header_row_index: int = 0
    skipped_row_count: int = 0
    row_count_estimate: int = 0
    sheet_name: str | None = None
    parse_warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# TimestampCandidate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TimestampCandidate:
    """One candidate column for the import time axis.

    Candidates are ranked by confidence (1.0 = certain, 0.0 = unlikely).
    The highest-confidence candidate is pre-selected by default; the user may
    override the selection.

    Fields
    ------
    column_name         Raw column name from the file.
    column_index        Zero-based column position.
    confidence          Detection confidence in [0.0, 1.0].
    detected_format     strptime format string inferred from example values.
    example_values      Small sample of raw cell values used for detection.
    invalid_sample_count  Number of cells that failed to parse with detected_format.
    duplicate_sample_count  Number of duplicate parsed timestamp values in the
                        preview sample. The first occurrence is not counted.
    timezone_detected   IANA timezone string if one was embedded in the values.
    user_selected       True when the user explicitly chose this column.
    """

    column_name: str
    column_index: int
    confidence: float
    detected_format: str | None = None
    example_values: list[str] = field(default_factory=list)
    invalid_sample_count: int = 0
    duplicate_sample_count: int = 0
    timezone_detected: str | None = None
    user_selected: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ColumnMappingCandidate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ColumnMappingCandidate:
    """Classification and mapping for one data column.

    Auto-classification is stored in (suggested_name, parameter_type, unit).
    User corrections are stored in the *_override fields and take precedence
    via the effective_* properties.

    Fields
    ------
    source_name         Column name as it appears in the file.
    source_index        Zero-based column position.
    suggested_name      Auto-suggested canonical output name.
    parameter_type      Auto-classified semantic type.
    unit                Auto-inferred engineering unit string.
    confidence          Classification confidence in [0.0, 1.0].
    classification_reason  Short human-readable explanation of how the type was determined.
    user_name_override  User-supplied output name (overrides suggested_name).
    user_unit_override  User-supplied unit string (overrides unit).
    user_type_override  User-supplied type (overrides parameter_type).
    excluded            When True this column is omitted from the output.
    """

    source_name: str
    source_index: int
    suggested_name: str
    parameter_type: ParameterType
    unit: str | None = None
    confidence: float = 0.0
    classification_reason: str = ""
    user_name_override: str | None = None
    user_unit_override: str | None = None
    user_type_override: ParameterType | None = None
    excluded: bool = False

    # ------------------------------------------------------------------
    # Effective-value properties — always prefer user overrides
    # ------------------------------------------------------------------

    @property
    def effective_name(self) -> str:
        """Output column name after applying any user override."""
        return self.user_name_override if self.user_name_override is not None else self.suggested_name

    @property
    def effective_unit(self) -> str | None:
        """Output unit after applying any user override."""
        return self.user_unit_override if self.user_unit_override is not None else self.unit

    @property
    def effective_type(self) -> ParameterType:
        """Output parameter type after applying any user override."""
        return self.user_type_override if self.user_type_override is not None else self.parameter_type

    @property
    def has_user_override(self) -> bool:
        """True when the user has changed any classification field."""
        return (
            self.user_name_override is not None
            or self.user_unit_override is not None
            or self.user_type_override is not None
        )


# ─────────────────────────────────────────────────────────────────────────────
# ImportWizardSession
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ImportWizardSession:
    """Complete state for one CSV/Excel import workflow.

    The session advances through WizardStep stages.  Each stage populates its
    own fields; earlier stages are never cleared when advancing (the user may
    go back and change earlier choices, which invalidates later stages).

    Fields
    ------
    source_path             Absolute path to the source file.
    provider_type           "csv" or "excel".
    sheet_name              Active Excel sheet; None for CSV.
    delimiter               Column delimiter for CSV; None for Excel or auto-detect.
    raw_preview             Result of the RAW_PREVIEW step.
    timestamp_candidates    Candidates ranked by confidence (highest first).
    selected_timestamp_column  Column name chosen as the time axis.
    timestamp_repair_plan   How to produce a clean time axis from that column.
    column_mappings         Per-column classification and override state.
    normalization_plan      Final executable plan built at NORMALIZATION_REVIEW.
    validation_messages     All messages accumulated across all steps.
    current_step            Where the wizard currently is.
    """

    source_path: str
    provider_type: str
    sheet_name: str | None = None
    delimiter: str | None = None
    raw_preview: RawPreviewModel | None = None
    timestamp_candidates: list[TimestampCandidate] = field(default_factory=list)
    selected_timestamp_column: str | None = None
    time_axis_mode: str = "auto"
    elapsed_time_unit: str | None = None
    sample_rate_hz: float | None = None
    sample_interval_seconds: float | None = None
    timestamp_repair_plan: TimestampRepairPlan | None = None
    column_mappings: list[ColumnMappingCandidate] = field(default_factory=list)
    normalization_plan: NormalizationPlan | None = None
    validation_messages: list[ValidationMessage] = field(default_factory=list)
    current_step: WizardStep = WizardStep.LOAD_FILE

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def advance_to(self, step: WizardStep, *, allow_skip: bool = False) -> bool:
        """Attempt to move to *step*.

        Returns True and updates current_step on success.
        Returns False without modifying state when the transition is not
        permitted (e.g. forward-jumping more than one step without allow_skip).
        """
        if can_transition(self.current_step, step, allow_skip=allow_skip):
            self.current_step = step
            return True
        return False

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_message(self, message: ValidationMessage) -> None:
        """Append a validation message to the session log."""
        self.validation_messages.append(message)

    def clear_messages(self) -> None:
        """Remove all accumulated validation messages."""
        self.validation_messages.clear()

    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.WARNING]

    def infos(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.INFO]

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def is_ready_to_normalize(self) -> bool:
        """True when a fully validated normalization plan is present."""
        return (
            self.normalization_plan is not None
            and self.normalization_plan.is_executable
        )

    def has_errors(self) -> bool:
        return bool(self.errors())

    # ------------------------------------------------------------------
    # Convenience — best timestamp candidate
    # ------------------------------------------------------------------

    def best_timestamp_candidate(self) -> TimestampCandidate | None:
        """Return the highest-confidence candidate, or the user-selected one."""
        selected = [c for c in self.timestamp_candidates if c.user_selected]
        if selected:
            return selected[0]
        if self.timestamp_candidates:
            return max(self.timestamp_candidates, key=lambda c: c.confidence)
        return None
