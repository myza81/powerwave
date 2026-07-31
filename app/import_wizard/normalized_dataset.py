"""Normalized dataset model — intermediate engineering output of the Import Wizard.

NormalizedDataset is the clean, typed, renamed representation produced by the
data assembly pipeline after timestamp repair, column mapping, and validation.
It is auditable (full source traceability via ParameterMetadata), exportable
(CSV / Parquet / Feather), and designed for future DisturbanceRecord conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity


@dataclass(slots=True)
class ParameterMetadata:
    """Metadata for one normalized data column.

    Fields
    ------
    canonical_name  Name of the column in the normalized DataFrame.
    parameter_type  Semantic classification (VOLTAGE, CURRENT, MW, ...).
    unit            Engineering unit string, e.g. "kV", "A", "MW".
    source_name     Original column name in the source file.
    source_index    Zero-based column position in the source file.
    user_overridden True when any user correction was applied to this column.
    confidence      Auto-classification confidence in [0.0, 1.0].
    notes           Non-fatal warnings or informational notes about this column.
    """

    canonical_name: str
    parameter_type: ParameterType
    unit: str | None
    source_name: str
    source_index: int
    user_overridden: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AssemblyDiagnostics:
    """Quantitative statistics from the data assembly pipeline.

    Fields
    ------
    total_rows                 Rows in the source DataFrame before filtering.
    normalized_rows            Rows retained in the assembled output.
    dropped_rows               Rows dropped due to NaT timestamps.
    included_columns           Canonical names of all output columns (incl. timestamp).
    excluded_columns           Source names of excluded columns.
    duplicate_timestamp_count  Duplicate timestamps retained in the output.
    invalid_value_count        Total NaN count across data columns in the output.
    parameter_counts           {ParameterType.value: count} — columns per type.
    """

    total_rows: int = 0
    normalized_rows: int = 0
    dropped_rows: int = 0
    included_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    duplicate_timestamp_count: int = 0
    invalid_value_count: int = 0
    parameter_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class NormalizedDataset:
    """Intermediate normalized engineering dataset produced by the Import Wizard.

    Not yet a DisturbanceRecord — an auditable, traceable intermediate suitable
    for export preview, validation, and future DisturbanceRecord conversion.

    The ``data`` DataFrame has:
    - Integer RangeIndex (not a timestamp index).
    - A datetime column named by ``timestamp_column`` (always inserted first).
    - One column per included data parameter, named by canonical names.

    Fields
    ------
    data                      Clean DataFrame with integer RangeIndex.
    timestamp_column          Name of the datetime column within data.
    parameters                ParameterMetadata for each included data column
                              (excludes the timestamp column itself).
    excluded_columns          Source column names excluded from the output.
    validation_messages       Warnings and errors accumulated during assembly.
    diagnostics               Quantitative assembly statistics.
    source_path               Absolute path to the original source file.
    source_file_name          Basename of the source file.
    timestamp_repair_strategy Strategy label used to produce the time axis.
    time_axis_mode           "absolute" or "relative_elapsed".
    time_axis_unit           Source unit/format label used for the time axis.
    time_axis_seconds        Authoritative elapsed seconds for relative elapsed
                             records. None for absolute timestamp records.
    is_valid                  True when no ERROR messages, non-empty data, valid
                              timestamp column, and at least one data column.
    """

    data: pd.DataFrame
    timestamp_column: str
    parameters: list[ParameterMetadata]
    excluded_columns: list[str]
    validation_messages: list[ValidationMessage]
    diagnostics: AssemblyDiagnostics
    source_path: str | None = None
    source_file_name: str | None = None
    timestamp_repair_strategy: str = "unknown"
    time_axis_mode: str = "absolute"
    time_axis_unit: str | None = None
    time_axis_seconds: pd.Series | None = None
    is_valid: bool = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_export_ready(self) -> bool:
        """True when the dataset has data and can be safely written to disk."""
        return self.is_valid and len(self.data) > 0 and bool(self.parameters)

    def parameter_by_canonical(self, name: str) -> ParameterMetadata | None:
        """Return ParameterMetadata for the given canonical column name, or None."""
        for p in self.parameters:
            if p.canonical_name == name:
                return p
        return None

    def parameters_by_type(self, ptype: ParameterType) -> list[ParameterMetadata]:
        """Return all parameters of the given ParameterType."""
        return [p for p in self.parameters if p.parameter_type == ptype]

    def has_errors(self) -> bool:
        """True when any ERROR-severity validation message is present."""
        return any(
            m.severity == ValidationSeverity.ERROR for m in self.validation_messages
        )

    def valid_data(self) -> pd.DataFrame:
        """Return rows where the timestamp column is not NaT, with reset index."""
        ts = self.data[self.timestamp_column]
        return self.data[ts.notna()].reset_index(drop=True)
