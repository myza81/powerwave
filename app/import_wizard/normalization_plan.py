"""Normalization plan — the final executable contract.

A NormalizationPlan captures every decision made across the wizard steps in a
single, self-contained object.  When is_executable is True, a (future) normalization
engine can consume it to produce a DisturbanceRecord-compatible output file without
any further user interaction.

Dependency order (no cycles):
    contracts → column_mapping → timestamp_contracts → normalization_plan
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.timestamp_contracts import TimestampRepairPlan


@dataclass(slots=True)
class NormalizationPlan:
    """Complete specification for transforming a raw CSV/Excel file into a
    DisturbanceRecord-compatible normalized output.

    Fields
    ------
    timestamp_plan          How to produce a valid time axis.
    selected_columns        Column names to include in the output (after renames).
    excluded_columns        Column names explicitly excluded by the user.
    column_renames          {source_name: output_name} mapping.
    column_units            {output_name: unit_string} mapping.
    column_types            {output_name: ParameterType} mapping.
    output_path_suggestion  Default save path for the normalized file.
    validation_messages     Accumulated warnings/errors from wizard validation passes.
    """

    timestamp_plan: TimestampRepairPlan | None = None
    selected_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    column_renames: dict[str, str] = field(default_factory=dict)
    column_units: dict[str, str] = field(default_factory=dict)
    column_types: dict[str, ParameterType] = field(default_factory=dict)
    output_path_suggestion: str | None = None
    validation_messages: list[ValidationMessage] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────────
    # Readiness
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_executable(self) -> bool:
        """True when the plan has enough information to run without user input.

        Requirements:
        - A timestamp plan must be present and validated.
        - At least one output column must be selected.
        - No ERROR-severity validation messages must be present.
        """
        if self.timestamp_plan is None or not self.timestamp_plan.repair_validated:
            return False
        if not self.selected_columns:
            return False
        return not any(
            m.severity == ValidationSeverity.ERROR
            for m in self.validation_messages
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.validation_messages if m.severity == ValidationSeverity.WARNING]

    def effective_name(self, source_name: str) -> str:
        """Return the output name for *source_name* after applying renames."""
        return self.column_renames.get(source_name, source_name)
