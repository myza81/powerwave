"""High-level timestamp normalization pipeline.

Entry point: normalize_timestamps()

The pipeline:
1. Dispatches to the correct strategy executor.
2. Runs post-repair quality checks (monotonic, duplicate, jitter).
3. Emits structured ValidationMessages.
4. Returns a self-contained TimestampNormalizationResult.

Raw source data is NEVER mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.repair_diagnostics import RepairDiagnostics
from app.import_wizard.timestamp_contracts import TimestampRepairPlan
from app.import_wizard.timestamp_contracts import TimestampRepairStrategy
from app.import_wizard.timestamp_repair_executor import coerce_elapsed_seconds, dispatch


@dataclass
class TimestampNormalizationResult:
    """Complete output of one normalize_timestamps() call.

    Fields
    ------
    normalized          pd.Series of datetime64 (timezone-aware when a timezone
                        conversion was applied; timezone-naive otherwise).
                        NaT entries indicate rows that could not be recovered.
    diagnostics         Quantitative quality metrics.
    validation_messages All warnings and errors generated during normalization.
    success             True when at least one non-NaT timestamp was produced
                        and no ERROR-severity messages are present.
    time_axis_mode      "absolute", "relative_elapsed", "synthetic_elapsed",
                        or "sample_index".
    time_axis_unit      Source unit/format label used for the time axis.
    elapsed_seconds     For relative elapsed sources, the authoritative time
                        axis values converted to seconds.
    """

    normalized: pd.Series
    diagnostics: RepairDiagnostics
    validation_messages: list[ValidationMessage] = field(default_factory=list)
    success: bool = False
    time_axis_mode: str = "absolute"
    time_axis_unit: str | None = None
    elapsed_seconds: pd.Series | None = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_errors(self) -> bool:
        return any(m.severity == ValidationSeverity.ERROR for m in self.validation_messages)

    def valid_timestamps(self) -> pd.Series:
        """Return only the non-NaT normalized timestamps."""
        return self.normalized.dropna()

    def nat_mask(self) -> pd.Series:
        """Boolean mask: True where timestamp is NaT."""
        return self.normalized.isna()


def normalize_timestamps(
    raw_series: pd.Series,
    repair_plan: TimestampRepairPlan,
    *,
    aux_series: dict[str, pd.Series] | None = None,
) -> TimestampNormalizationResult:
    """Normalize a raw timestamp column using the given TimestampRepairPlan.

    Parameters
    ----------
    raw_series:     Raw timestamp values (strings, floats, or datetime64).
                    This Series is never mutated.
    repair_plan:    Immutable plan from the Import Wizard — specifies strategy
                    and all required parameters.
    aux_series:     Additional columns needed by multi-column strategies such as
                    COMBINE_DATE_TIME_COLUMNS.  Keys are column names.

    Returns
    -------
    TimestampNormalizationResult with normalized Series, diagnostics, and messages.

    Notes
    -----
    - Fails gracefully: always returns a result, even on completely unparseable input.
    - Raw source values are preserved in the caller's DataFrame; this function
      operates on a copy of the input.
    """
    # Defensive copy — never mutate caller's data
    raw_copy = raw_series.copy()

    extra_msgs: list[ValidationMessage] = []

    # Dispatch
    elapsed_seconds: pd.Series | None = None
    time_axis_mode = "absolute"
    time_axis_unit = repair_plan.detected_format or repair_plan.user_format
    try:
        normalized, diagnostics, messages = dispatch(raw_copy, repair_plan, aux_series)
        if repair_plan.strategy == TimestampRepairStrategy.PARSE_ELAPSED_TIME:
            time_axis_mode = "relative_elapsed"
            time_axis_unit = repair_plan.elapsed_time_unit or repair_plan.detected_format
            elapsed_seconds = coerce_elapsed_seconds(raw_copy, time_axis_unit)
        elif repair_plan.strategy == TimestampRepairStrategy.GENERATE_SYNTHETIC_ELAPSED:
            time_axis_mode = "synthetic_elapsed"
            interval = repair_plan.sampling_interval_seconds
            if interval is None and repair_plan.sample_rate_hz and repair_plan.sample_rate_hz > 0:
                interval = 1.0 / repair_plan.sample_rate_hz
            interval = interval if interval is not None and interval > 0 else 1.0
            time_axis_unit = "s"
            elapsed_seconds = pd.Series(range(len(raw_copy)), index=raw_copy.index, dtype="float64") * float(interval)
        elif repair_plan.strategy == TimestampRepairStrategy.GENERATE_SAMPLE_INDEX:
            time_axis_mode = "sample_index"
            time_axis_unit = "sample"
            elapsed_seconds = pd.Series(range(len(raw_copy)), index=raw_copy.index, dtype="float64")
    except Exception as exc:
        diagnostics = RepairDiagnostics(
            strategy_used=repair_plan.strategy.value,
            total_rows=len(raw_series),
            dropped_rows=len(raw_series),
        )
        diagnostics.add_warning(f"Unexpected error during repair: {exc}")
        normalized = pd.Series(pd.NaT, index=raw_series.index, dtype="datetime64[ns]")
        messages = [
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="TS_REPAIR_EXCEPTION",
                message=f"Timestamp repair raised an unexpected error: {exc}",
            )
        ]

    # Post-repair quality validation
    extra_msgs.extend(_post_repair_checks(normalized, diagnostics))

    all_messages = messages + extra_msgs

    success = (
        normalized.notna().any()
        and not any(m.severity == ValidationSeverity.ERROR for m in all_messages)
    )

    return TimestampNormalizationResult(
        normalized=normalized,
        diagnostics=diagnostics,
        validation_messages=all_messages,
        success=success,
        time_axis_mode=time_axis_mode,
        time_axis_unit=time_axis_unit,
        elapsed_seconds=elapsed_seconds,
    )


def _post_repair_checks(
    normalized: pd.Series,
    diagnostics: RepairDiagnostics,
) -> list[ValidationMessage]:
    """Generate additional validation messages based on final normalized state."""
    msgs: list[ValidationMessage] = []

    if normalized.isna().all():
        msgs.append(
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="TS_ALL_NAT",
                message="All timestamps are NaT after repair. No valid time axis produced.",
            )
        )
        return msgs

    valid_frac = normalized.notna().sum() / max(len(normalized), 1)
    if valid_frac < 0.5:
        msgs.append(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TS_LOW_COVERAGE",
                message=(
                    f"Only {valid_frac:.0%} of timestamps were successfully parsed. "
                    "Consider reviewing the repair strategy."
                ),
            )
        )

    return msgs
