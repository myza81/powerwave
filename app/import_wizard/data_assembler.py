"""Data assembly pipeline for the Import Wizard.

Converts a raw DataFrame + normalization decisions into a NormalizedDataset.

Entry point: assemble_normalized_dataset()

Design principles
-----------------
- Never mutate the caller's raw_dataframe.
- Prefer vectorized pandas operations.
- Every transformation is traceable via ParameterMetadata.
- Always return a NormalizedDataset; never raise.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.models import ColumnMappingCandidate
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.import_wizard.timestamp_normalizer import TimestampNormalizationResult

_TIMESTAMP_COLUMN = "timestamp"

# Canonical unit casing for common power system units.
_UNIT_CANONICAL: dict[str, str] = {
    "kv": "kV",
    "v": "V",
    "pu_v": "pu(V)",
    "pu": "pu",
    "a": "A",
    "ka": "kA",
    "mw": "MW",
    "kw": "kW",
    "mvar": "Mvar",
    "kvar": "kVar",
    "hz": "Hz",
    "hz/s": "Hz/s",
    "s": "s",
    "ms": "ms",
}


def _sanitize_name(name: str) -> str:
    """Convert a raw column name to a safe, lowercase identifier.

    Replaces non-alphanumeric runs with underscores and strips leading/trailing
    underscores.  Never returns an empty string.
    """
    s = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return s.strip("_") or "column"


def _disambiguate_names(names: list[str]) -> list[str]:
    """Ensure all names in the list are unique.

    Unique names are left unchanged.  Duplicate names all receive _1, _2, ...
    suffixes starting at 1.
    """
    if not names:
        return []
    counts = Counter(names)
    if len(counts) == len(names):
        return list(names)
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if counts[name] == 1:
            result.append(name)
        else:
            n = seen.get(name, 0) + 1
            seen[name] = n
            result.append(f"{name}_{n}")
    return result


def _normalize_unit(unit: str | None) -> str | None:
    """Normalize engineering unit formatting.

    Strips whitespace and applies canonical casing for common power system units.
    Unrecognised units are returned unchanged (stripped).
    """
    if unit is None:
        return None
    stripped = unit.strip()
    if not stripped:
        return None
    return _UNIT_CANONICAL.get(stripped.lower(), stripped)


def _resolve_canonical_name(
    source_name: str,
    plan: NormalizationPlan,
    candidate: ColumnMappingCandidate | None,
) -> str:
    """Determine the canonical output name for one source column.

    Resolution priority (highest wins):
    1. NormalizationPlan.column_renames[source_name] — explicit wizard decision.
    2. ColumnMappingCandidate.user_name_override — user typed a name in the UI.
    3. ColumnMappingCandidate.suggested_name — auto-detected canonical name.
    4. _sanitize_name(source_name) — lowercase sanitization fallback.
    """
    if source_name in plan.column_renames:
        return plan.column_renames[source_name]
    if candidate is not None:
        if candidate.user_name_override is not None:
            return candidate.user_name_override
        if candidate.suggested_name:
            return candidate.suggested_name
    return _sanitize_name(source_name)


def assemble_normalized_dataset(
    raw_dataframe: pd.DataFrame,
    normalization_plan: NormalizationPlan,
    timestamp_result: TimestampNormalizationResult,
    *,
    column_mappings: list[ColumnMappingCandidate] | None = None,
    source_path: str | None = None,
    drop_nat_rows: bool = True,
) -> NormalizedDataset:
    """Assemble a NormalizedDataset from raw data and normalization decisions.

    Parameters
    ----------
    raw_dataframe       Full raw source DataFrame.  Never mutated.
    normalization_plan  Final plan capturing column selections, renames, types,
                        units, and the timestamp repair plan.
    timestamp_result    Result of normalize_timestamps() — provides the clean
                        datetime Series aligned to raw_dataframe.
    column_mappings     Optional per-column classification and user overrides.
                        When provided, enriches ParameterMetadata with source
                        traceability and confidence scores.
    source_path         Absolute path to the original file (for traceability and
                        export filename suggestion).
    drop_nat_rows       When True (default) rows where the timestamp is NaT are
                        dropped from the output.

    Returns
    -------
    NormalizedDataset — always returns, never raises.  Check ``is_valid`` and
    ``validation_messages`` for any issues.
    """
    messages: list[ValidationMessage] = []
    diagnostics = AssemblyDiagnostics(total_rows=len(raw_dataframe))

    # ── Non-destructive copy ───────────────────────────────────────────────────
    df = raw_dataframe.copy()

    # ── Timestamp integration ──────────────────────────────────────────────────
    ts_series = timestamp_result.normalized.copy()

    if not timestamp_result.success:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="ND_TS_REPAIR_FAILED",
            message="Timestamp repair was not fully successful; some rows may have NaT timestamps.",
        ))

    # Align Series length to DataFrame
    if len(ts_series) != len(df):
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="ND_TS_LENGTH_MISMATCH",
            message=(
                f"Timestamp Series length ({len(ts_series)}) does not match "
                f"DataFrame row count ({len(df)})."
            ),
        ))
        if len(ts_series) > len(df):
            ts_series = ts_series.iloc[: len(df)]
        else:
            pad = pd.Series(
                [pd.NaT] * (len(df) - len(ts_series)),
                dtype="datetime64[ns]",
            )
            ts_series = pd.concat([ts_series, pad], ignore_index=True)

    ts_series = ts_series.reset_index(drop=True)

    # ── Resolve selected/excluded columns ──────────────────────────────────────
    selected_sources = list(normalization_plan.selected_columns)
    excluded_sources = set(normalization_plan.excluded_columns)
    available_cols = set(df.columns)

    valid_selected = [c for c in selected_sources if c in available_cols]
    missing_selected = [c for c in selected_sources if c not in available_cols]

    if missing_selected:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="ND_MISSING_COLUMNS",
            message=f"Selected columns not found in DataFrame: {missing_selected}",
        ))

    if not valid_selected:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="ND_NO_COLUMNS",
            message="No valid columns selected for normalization.",
        ))

    # Columns not explicitly selected are auto-excluded (for diagnostics)
    auto_excluded = [
        c for c in df.columns
        if c not in set(valid_selected) and c not in excluded_sources
    ]
    all_excluded = sorted(set(list(excluded_sources) + auto_excluded))

    # ── Build candidate lookup ─────────────────────────────────────────────────
    candidate_index: dict[str, ColumnMappingCandidate] = (
        {c.source_name: c for c in column_mappings} if column_mappings else {}
    )

    # ── Resolve canonical names ────────────────────────────────────────────────
    raw_canonical: list[str] = [
        _resolve_canonical_name(src, normalization_plan, candidate_index.get(src))
        for src in valid_selected
    ]

    # Guard: canonical name "timestamp" would collide with the time-axis column
    guarded: list[str] = []
    had_ts_conflict = False
    for name in raw_canonical:
        if name == _TIMESTAMP_COLUMN:
            guarded.append(f"data_{_TIMESTAMP_COLUMN}")
            had_ts_conflict = True
        else:
            guarded.append(name)
    if had_ts_conflict:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="ND_TIMESTAMP_NAME_CONFLICT",
            message=(
                "A data column was renamed from 'timestamp' to 'data_timestamp' "
                "to avoid collision with the time axis column."
            ),
        ))

    canonical_names = _disambiguate_names(guarded)

    # ── Build ParameterMetadata ────────────────────────────────────────────────
    raw_col_list = list(df.columns)
    parameters: list[ParameterMetadata] = []

    for src, canonical in zip(valid_selected, canonical_names):
        cand = candidate_index.get(src)
        src_index = cand.source_index if cand is not None else (
            raw_col_list.index(src) if src in raw_col_list else -1
        )

        # Type: plan (by output name) → plan (by source name) → candidate → UNKNOWN
        if canonical in normalization_plan.column_types:
            ptype = normalization_plan.column_types[canonical]
        elif src in normalization_plan.column_types:
            ptype = normalization_plan.column_types[src]
        elif cand is not None:
            ptype = cand.effective_type
        else:
            ptype = ParameterType.UNKNOWN

        # Unit: plan (by output name) → plan (by source name) → candidate → None
        raw_unit: str | None = None
        if canonical in normalization_plan.column_units:
            raw_unit = normalization_plan.column_units[canonical]
        elif src in normalization_plan.column_units:
            raw_unit = normalization_plan.column_units[src]
        elif cand is not None:
            raw_unit = cand.effective_unit

        unit = _normalize_unit(raw_unit)

        user_overridden = (
            src in normalization_plan.column_renames
            or (cand is not None and cand.has_user_override)
        )
        confidence = cand.confidence if cand is not None else 0.0

        parameters.append(ParameterMetadata(
            canonical_name=canonical,
            parameter_type=ptype,
            unit=unit,
            source_name=src,
            source_index=src_index,
            user_overridden=user_overridden,
            confidence=confidence,
        ))

    # ── Assemble output DataFrame ──────────────────────────────────────────────
    if valid_selected:
        data_df = df[valid_selected].copy()
        rename_map = dict(zip(valid_selected, canonical_names))
        data_df = data_df.rename(columns=rename_map)
    else:
        data_df = pd.DataFrame(index=range(len(df)))

    data_df.insert(0, _TIMESTAMP_COLUMN, ts_series.values)
    data_df = data_df.reset_index(drop=True)

    # ── Drop NaT timestamp rows ────────────────────────────────────────────────
    pre_drop = len(data_df)
    if drop_nat_rows:
        data_df = data_df[data_df[_TIMESTAMP_COLUMN].notna()].reset_index(drop=True)
    dropped_rows = pre_drop - len(data_df)

    # ── Post-assembly statistics ───────────────────────────────────────────────
    dup_count = int(data_df[_TIMESTAMP_COLUMN].duplicated().sum()) if not data_df.empty else 0
    data_cols = [c for c in data_df.columns if c != _TIMESTAMP_COLUMN]
    invalid_count = int(data_df[data_cols].isna().sum().sum()) if data_cols else 0
    type_counts: dict[str, int] = {}
    for pm in parameters:
        key = pm.parameter_type.value
        type_counts[key] = type_counts.get(key, 0) + 1

    # ── Finalise diagnostics ───────────────────────────────────────────────────
    diagnostics.normalized_rows = len(data_df)
    diagnostics.dropped_rows = dropped_rows
    diagnostics.included_columns = [_TIMESTAMP_COLUMN] + list(canonical_names)
    diagnostics.excluded_columns = all_excluded
    diagnostics.duplicate_timestamp_count = dup_count
    diagnostics.invalid_value_count = invalid_count
    diagnostics.parameter_counts = type_counts

    # ── is_valid ──────────────────────────────────────────────────────────────
    has_errors = any(m.severity == ValidationSeverity.ERROR for m in messages)
    is_valid = (
        not has_errors
        and not data_df.empty
        and bool(parameters)
        and data_df[_TIMESTAMP_COLUMN].notna().any()
    )

    # ── Source file metadata ───────────────────────────────────────────────────
    source_file_name = os.path.basename(source_path) if source_path else None
    repair_strategy = (
        normalization_plan.timestamp_plan.strategy.value
        if normalization_plan.timestamp_plan is not None
        else timestamp_result.diagnostics.strategy_used
    )

    return NormalizedDataset(
        data=data_df,
        timestamp_column=_TIMESTAMP_COLUMN,
        parameters=parameters,
        excluded_columns=all_excluded,
        validation_messages=messages,
        diagnostics=diagnostics,
        source_path=source_path,
        source_file_name=source_file_name,
        timestamp_repair_strategy=repair_strategy,
        is_valid=is_valid,
    )
