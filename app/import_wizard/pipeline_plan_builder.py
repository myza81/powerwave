"""Plan builder: converts GUI/session state into validated executable plans.

This module has NO Qt dependency — it operates only on pure data models.
It is the single authoritative layer that turns wizard review/edit state
into the concrete plans consumed by run_import_pipeline_with_plan().

Public surface
--------------
PlanBuildResult          Result of build_execution_plan().
build_execution_plan()   Session + column mappings → PlanBuildResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.models import ColumnMappingCandidate, ImportWizardSession, TimestampCandidate
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.import_wizard.timestamp_format_validator import validate_user_timestamp_format


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PlanBuildResult:
    """Result of build_execution_plan().

    Fields
    ------
    normalization_plan    Executable plan built from the GUI state, or None when
                          the plan has blocking errors.
    timestamp_candidate   The candidate the plan was built from (may be None).
    validation_messages   All messages from plan construction (errors + warnings).
    """

    normalization_plan: NormalizationPlan | None
    timestamp_candidate: TimestampCandidate | None
    validation_messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def is_executable(self) -> bool:
        """True when the plan has no blocking errors and can be dispatched."""
        if self.normalization_plan is None:
            return False
        if not self.normalization_plan.is_executable:
            return False
        return not any(
            m.severity == ValidationSeverity.ERROR for m in self.validation_messages
        )

    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.WARNING]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_execution_plan(
    session: ImportWizardSession,
    mappings: list[ColumnMappingCandidate],
) -> PlanBuildResult:
    """Convert GUI/session state into a validated, executable NormalizationPlan.

    This function is the authoritative translation from wizard review decisions
    into the concrete plan that the backend pipeline will execute.  It must be
    called with the *current* column mapping list from the GUI model (not the
    session snapshot, which may lag behind live edits).

    Validation rules (blocking errors)
    -----------------------------------
    PLAN_NO_TIMESTAMP       No timestamp candidate has been selected.
    PLAN_NO_DATA_COLUMNS    All data columns are excluded — nothing to import.
    PLAN_DUPLICATE_NAME     Two included columns share the same effective output name.

    Warnings (non-blocking)
    -----------------------
    PLAN_UNKNOWN_COLUMN     Column type is UNKNOWN; treated as analog.

    Parameters
    ----------
    session     The active ImportWizardSession (carries selected_timestamp_column,
                timestamp_repair_plan, and timestamp_candidates).
    mappings    The current ColumnMappingCandidate list from the GUI table model.
                This is the authoritative source of include/exclude state, output
                names, types, and units — not the session snapshot.

    Returns
    -------
    PlanBuildResult — always returns, never raises.
    """
    messages: list[ValidationMessage] = []

    # ── Step 1: Resolve the authoritative timestamp candidate ─────────────────
    candidate = session.best_timestamp_candidate()
    if candidate is None:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PLAN_NO_TIMESTAMP",
            message="No timestamp column has been selected. Choose a timestamp column before importing.",
        ))
        return PlanBuildResult(None, None, messages)

    # ── Step 2: Resolve the repair plan ───────────────────────────────────────
    # Use the session's repair plan if already constructed; else derive from candidate.
    repair_plan: TimestampRepairPlan
    if session.timestamp_repair_plan is not None:
        repair_plan = session.timestamp_repair_plan
    else:
        repair_plan = _derive_repair_plan(candidate)
    repair_plan, repair_messages = _validate_repair_plan(repair_plan, candidate)
    messages.extend(repair_messages)

    # ── Step 3: Build column mapping from GUI state ───────────────────────────
    ts_col = candidate.column_name
    selected: list[str] = []
    excluded: list[str] = []
    renames: dict[str, str] = {}
    units: dict[str, str] = {}
    types: dict[str, ParameterType] = {}

    # Track effective output names to catch duplicates.
    canonical_to_source: dict[str, str] = {}

    for mapping in mappings:
        src = mapping.source_name

        # Timestamp column and any TIMESTAMP-typed column are always excluded.
        if src == ts_col or mapping.effective_type == ParameterType.TIMESTAMP:
            excluded.append(src)
            continue

        if mapping.excluded:
            excluded.append(src)
            continue

        effective = mapping.effective_name

        # Duplicate canonical name check (blocking error — prevents channel collisions).
        if effective in canonical_to_source:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="PLAN_DUPLICATE_NAME",
                message=(
                    f"Duplicate output name '{effective}' for columns "
                    f"'{canonical_to_source[effective]}' and '{src}'. "
                    "Rename one of them before importing."
                ),
                affected_column=src,
            ))
            # Do not include this column — skip it to avoid downstream confusion.
            continue

        canonical_to_source[effective] = src
        selected.append(src)

        if effective != src:
            renames[src] = effective
        if mapping.effective_unit:
            units[effective] = mapping.effective_unit
        types[effective] = mapping.effective_type

        if mapping.effective_type == ParameterType.UNKNOWN:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="PLAN_UNKNOWN_COLUMN",
                message=(
                    f"Column '{src}' has unknown type and will be treated "
                    "as an analog channel."
                ),
                affected_column=src,
            ))

    # ── Step 4: Require at least one data column ──────────────────────────────
    if not selected:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="PLAN_NO_DATA_COLUMNS",
            message=(
                "No data columns are included. Include at least one column before importing."
            ),
        ))

    plan = NormalizationPlan(
        timestamp_plan=repair_plan,
        selected_columns=selected,
        excluded_columns=excluded,
        column_renames=renames,
        column_units=units,
        column_types=types,
        validation_messages=messages,
    )

    return PlanBuildResult(plan, candidate, messages)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _derive_repair_plan(candidate: TimestampCandidate) -> TimestampRepairPlan:
    """Build a TimestampRepairPlan from a detected timestamp candidate.

    Used only when the session does not already carry a repair plan (i.e. the
    user skipped the timestamp review step).  Mirrors the logic in
    import_pipeline._build_repair_plan() to keep the two paths consistent.
    """
    fmt = candidate.detected_format

    if fmt == "excel_serial":
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION,
            repair_validated=True,
            repair_notes="Derived from Excel serial date column.",
        )

    if fmt in ("epoch_seconds", "epoch_milliseconds"):
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format=fmt,
            repair_validated=True,
            repair_notes=f"Derived from {fmt} column.",
        )

    if fmt is not None:
        return TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format=fmt,
            repair_validated=True,
            repair_notes=f"Derived from detected format '{fmt}'.",
        )

    return TimestampRepairPlan(
        strategy=TimestampRepairStrategy.NO_REPAIR,
        repair_validated=True,
        repair_notes="No specific format; using pandas auto-detection.",
    )


def _validate_repair_plan(
    repair_plan: TimestampRepairPlan,
    candidate: TimestampCandidate,
) -> tuple[TimestampRepairPlan, list[ValidationMessage]]:
    """Validate timestamp repair settings before execution.

    User format plans are authoritative. They must parse at least one sampled
    timestamp; complete failure blocks execution instead of falling back to the
    detected format.
    """
    if repair_plan.strategy != TimestampRepairStrategy.PARSE_USER_FORMAT:
        return repair_plan, []

    fmt = (repair_plan.user_format or "").strip()
    if not fmt:
        detected_plan = _derive_repair_plan(candidate)
        return detected_plan, [
            ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="PLAN_USING_DETECTED_FORMAT",
                message="Manual timestamp format is empty; using detected format.",
                affected_column=candidate.column_name,
            )
        ]

    result = validate_user_timestamp_format(
        fmt,
        candidate.example_values,
        affected_column=candidate.column_name,
    )
    validated_plan = replace(
        repair_plan,
        user_format=fmt,
        repair_validated=result.is_valid,
        repair_notes=f"User format override: {fmt}",
    )
    return validated_plan, result.validation_messages
