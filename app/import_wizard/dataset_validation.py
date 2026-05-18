"""Standalone validation of a NormalizedDataset.

Call validate_normalized_dataset() to obtain a fresh list of ValidationMessages
without mutating the dataset.  Useful for re-validating after user corrections
or for independent quality checks before export.
"""
from __future__ import annotations

import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.normalized_dataset import NormalizedDataset


def validate_normalized_dataset(dataset: NormalizedDataset) -> list[ValidationMessage]:
    """Validate a NormalizedDataset and return any issues found.

    Checks performed
    ----------------
    - Timestamp column exists in the DataFrame.
    - Timestamp column has datetime64 dtype.
    - All timestamps are not NaT (ERROR if all-NaT).
    - Timestamps are monotonically increasing (WARNING if not).
    - Duplicate timestamps present (WARNING if any).
    - Dataset is non-empty (ERROR if empty).
    - At least one data column is present (ERROR if none).
    - All parameter types are UNKNOWN (WARNING).

    Parameters
    ----------
    dataset   The NormalizedDataset to validate.

    Returns
    -------
    list[ValidationMessage] — empty when no issues are found.
    """
    msgs: list[ValidationMessage] = []
    df = dataset.data
    ts_col = dataset.timestamp_column

    # ── Timestamp column existence ─────────────────────────────────────────────
    if ts_col not in df.columns:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="NV_NO_TIMESTAMP_COL",
            message=f"Timestamp column '{ts_col}' not found in the normalized DataFrame.",
        ))
        return msgs

    ts = df[ts_col]

    # ── Empty dataset ──────────────────────────────────────────────────────────
    if df.empty:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="NV_EMPTY_DATASET",
            message="Normalized dataset is empty.",
        ))
        return msgs

    # ── Timestamp dtype ────────────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(ts):
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="NV_TS_NOT_DATETIME",
            message=(
                f"Timestamp column '{ts_col}' has dtype {ts.dtype!r}, "
                "expected datetime64."
            ),
        ))

    # ── All NaT ───────────────────────────────────────────────────────────────
    if ts.isna().all():
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="NV_ALL_NAT",
            message="All timestamps are NaT in the normalized dataset.",
        ))
        return msgs

    # ── Monotonic timestamps ───────────────────────────────────────────────────
    valid_ts = ts.dropna()
    if not valid_ts.is_monotonic_increasing:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="NV_NON_MONOTONIC",
            message="Timestamps are not strictly monotonically increasing.",
        ))

    # ── Duplicate timestamps ───────────────────────────────────────────────────
    dup_count = int(ts.duplicated().sum())
    if dup_count > 0:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="NV_DUPLICATE_TIMESTAMPS",
            message=f"{dup_count} duplicate timestamp(s) found in the normalized dataset.",
        ))

    # ── No data columns ────────────────────────────────────────────────────────
    data_cols = [c for c in df.columns if c != ts_col]
    if not data_cols:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="NV_NO_DATA_COLUMNS",
            message="Normalized dataset has no data columns (only the timestamp column).",
        ))

    # ── No parameter metadata ──────────────────────────────────────────────────
    if not dataset.parameters:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="NV_NO_PARAMETERS",
            message="No parameter metadata found; all columns may have been excluded.",
        ))

    # ── All UNKNOWN types ──────────────────────────────────────────────────────
    if dataset.parameters and all(
        p.parameter_type == ParameterType.UNKNOWN for p in dataset.parameters
    ):
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="NV_ALL_UNKNOWN_TYPES",
            message="All parameters have UNKNOWN type; column classification may have failed.",
        ))

    return msgs
