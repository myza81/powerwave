"""Strategy-specific timestamp repair executors.

Each public function accepts:
    raw_series     — pd.Series of raw timestamp values (strings, floats, or
                     already-parsed datetimes) — NEVER mutated.
    plan           — TimestampRepairPlan describing what to do.
    aux_series     — optional {col_name: pd.Series} for multi-column strategies.

Each returns:
    (normalized: pd.Series[datetime64], diagnostics: RepairDiagnostics,
     messages: list[ValidationMessage])

All functions fail gracefully — they never raise on malformed data.  Rows that
cannot be recovered are set to NaT and counted in diagnostics.dropped_rows.
"""
from __future__ import annotations

import pandas as pd

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.interval_inference import infer_interval
from app.import_wizard.repair_diagnostics import RepairDiagnostics
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy

# Excel epoch: 1899-12-30 (openpyxl / Lotus 1-2-3 convention)
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")

_EPOCH_SECONDS_LABEL = "epoch_seconds"
_EPOCH_MS_LABEL = "epoch_milliseconds"
_EXCEL_SERIAL_LABEL = "excel_serial"

_EMPTY_DATETIME = pd.Series([], dtype="datetime64[ns]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_to_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse *series* to datetime64.  Returns NaT for unparseable cells.

    Uses format="mixed" when no explicit format is given so that pandas 3.x
    still parses common non-ISO strings (e.g. "2024-01-01 00:00:00").
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.copy()

    if fmt in (_EPOCH_SECONDS_LABEL, _EPOCH_MS_LABEL, _EXCEL_SERIAL_LABEL):
        return _coerce_special(series, fmt)  # type: ignore[arg-type]

    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")

    # No format: "mixed" lets pandas try each value independently (pandas ≥ 2.0)
    return pd.to_datetime(series, format="mixed", errors="coerce")


def _coerce_special(series: pd.Series, fmt: str) -> pd.Series:
    """Handle epoch_seconds / epoch_milliseconds / excel_serial."""
    numeric: pd.Series = pd.to_numeric(series, errors="coerce")  # type: ignore[assignment]
    if fmt == _EPOCH_SECONDS_LABEL:
        return pd.to_datetime(numeric, unit="s", errors="coerce")  # type: ignore[return-value]
    if fmt == _EPOCH_MS_LABEL:
        return pd.to_datetime(numeric, unit="ms", errors="coerce")  # type: ignore[return-value]
    # Excel serial: days (+ fractional days) since 1899-12-30
    # Use timedelta arithmetic — avoids nanosecond int64 precision issues
    valid_mask: pd.Series = numeric.notna()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if valid_mask.any():
        valid_days: pd.Series = numeric[valid_mask]
        converted = _EXCEL_EPOCH + pd.to_timedelta(valid_days, unit="D")
        result = result.copy()
        result[valid_mask] = converted.values
    return result


def _build_diagnostics(
    strategy: TimestampRepairStrategy,
    raw: pd.Series,
    normalized: pd.Series,
    repaired_rows: int = 0,
) -> RepairDiagnostics:
    diag = RepairDiagnostics(strategy_used=strategy.value)
    diag.total_rows = len(raw)

    nat_after = normalized.isna()
    diag.nat_rows = int(nat_after.sum())
    diag.dropped_rows = diag.nat_rows
    diag.valid_rows = diag.total_rows - diag.nat_rows
    diag.repaired_rows = repaired_rows

    analysis = infer_interval(normalized)
    diag.duplicate_rows = analysis.duplicate_count
    diag.non_monotonic_rows = analysis.non_monotonic_count
    diag.missing_rows = analysis.missing_count
    diag.is_monotonic = analysis.is_monotonic
    diag.inferred_interval_seconds = analysis.dominant_interval_seconds
    diag.interval_jitter_seconds = analysis.jitter_seconds
    diag.interval_jitter_fraction = analysis.jitter_fraction
    diag.is_irregular = analysis.is_irregular
    return diag


def _messages_from_diagnostics(diag: RepairDiagnostics) -> list[ValidationMessage]:
    msgs: list[ValidationMessage] = []
    if diag.duplicate_rows:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_DUPLICATES",
            message=f"{diag.duplicate_rows} duplicate timestamp(s) detected.",
        ))
    if diag.non_monotonic_rows:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_NON_MONOTONIC",
            message=f"{diag.non_monotonic_rows} non-monotonic timestamp step(s) detected.",
        ))
    if diag.missing_rows:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_MISSING_GAPS",
            message=f"~{diag.missing_rows} estimated missing sample(s) (gap analysis).",
        ))
    if diag.dropped_rows:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_DROPPED_ROWS",
            message=f"{diag.dropped_rows} timestamp row(s) could not be parsed and were dropped (NaT).",
        ))
    if diag.is_irregular:
        pct = (diag.interval_jitter_fraction or 0) * 100
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code="TS_JITTER",
            message=f"Irregular sampling detected: jitter {pct:.1f}% of dominant interval.",
        ))
    return msgs


# ---------------------------------------------------------------------------
# Strategy executors
# ---------------------------------------------------------------------------


def execute_no_repair(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Use timestamps as-is (attempt generic parse)."""
    normalized = _coerce_to_datetime(raw_series, fmt=None)
    diag = _build_diagnostics(plan.strategy, raw_series, normalized)
    return normalized, diag, _messages_from_diagnostics(diag)


def execute_parse_detected_format(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Parse using the auto-detected format from the profiler."""
    msgs: list[ValidationMessage] = []
    fmt = plan.detected_format

    if fmt is None:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_NO_FORMAT",
            message="No detected format available; falling back to generic parse.",
        ))

    normalized = _coerce_to_datetime(raw_series, fmt=fmt)
    was_string = not pd.api.types.is_datetime64_any_dtype(raw_series)
    repaired = int((~normalized.isna() & raw_series.astype(str).ne("")).sum()) if was_string else 0

    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=repaired)
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


def execute_parse_user_format(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Parse using the explicit user-supplied format string."""
    msgs: list[ValidationMessage] = []
    fmt = plan.user_format

    if not fmt:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_NO_USER_FORMAT",
            message="No user format string provided; falling back to generic parse.",
        ))
        normalized = _coerce_to_datetime(raw_series, fmt=None)
    else:
        normalized = _coerce_to_datetime(raw_series, fmt=fmt)
        if normalized.isna().all():
            msgs.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="TS_FORMAT_MISMATCH",
                message=f"User format '{fmt}' matched 0 rows. All timestamps are NaT.",
            ))

    diag = _build_diagnostics(plan.strategy, raw_series, normalized)
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


def execute_interpolate_missing(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Parse then linearly interpolate NaT values.

    Uses relative-timedelta arithmetic (not raw int64 nanoseconds) so that
    results are correct regardless of the pandas datetime precision (ns vs us).
    """
    msgs: list[ValidationMessage] = []
    fmt = plan.detected_format or plan.user_format

    parsed = _coerce_to_datetime(raw_series, fmt=fmt)
    nat_before = int(parsed.isna().sum())

    if nat_before > 0:
        valid = parsed.dropna()
        if len(valid) < 2:
            # Cannot interpolate without at least two anchor points
            normalized = parsed
            repaired = 0
        else:
            origin = valid.iloc[0]
            # Express everything as float seconds relative to origin;
            # NaT - origin gives NaT timedelta → .total_seconds() → NaN
            seconds = (parsed - origin).dt.total_seconds()
            seconds_interp = seconds.interpolate(method="linear", limit_direction="both")
            offsets = pd.to_timedelta(seconds_interp, unit="s")
            normalized = origin + offsets
            repaired = nat_before - int(normalized.isna().sum())
            if repaired > 0:
                msgs.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="TS_INTERPOLATED",
                    message=f"{repaired} missing timestamp(s) filled via linear interpolation.",
                ))
    else:
        normalized = parsed
        repaired = 0

    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=repaired)
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


def execute_reconstruct_from_interval(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Rebuild timestamps from a fixed sampling interval."""
    msgs: list[ValidationMessage] = []
    n = len(raw_series)

    interval_s = plan.sampling_interval_seconds
    if not interval_s or interval_s <= 0:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="TS_NO_INTERVAL",
            message="sampling_interval_seconds is missing or ≤ 0; cannot reconstruct timestamps.",
        ))
        normalized = pd.Series(pd.NaT, index=raw_series.index, dtype="datetime64[ns]")
        diag = _build_diagnostics(plan.strategy, raw_series, normalized)
        msgs += _messages_from_diagnostics(diag)
        return normalized, diag, msgs

    # Use first parseable timestamp as origin; fall back to Unix epoch
    fmt = plan.detected_format or plan.user_format
    parsed_raw = _coerce_to_datetime(raw_series, fmt=fmt)
    first_valid = parsed_raw.dropna()
    if len(first_valid) > 0:
        start = first_valid.iloc[0]
    else:
        start = pd.Timestamp("1970-01-01")
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_EPOCH_START",
            message="No parseable start timestamp found; using 1970-01-01 as origin.",
        ))

    # Use pd.date_range to avoid unit/precision issues entirely
    freq = pd.Timedelta(seconds=interval_s)
    normalized = pd.Series(
        pd.date_range(start=start, periods=n, freq=freq),
        index=raw_series.index,
    )

    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=n)
    diag.repaired_rows = n
    msgs += _messages_from_diagnostics(diag)
    msgs.append(ValidationMessage(
        severity=ValidationSeverity.INFO,
        code="TS_RECONSTRUCTED",
        message=f"Timestamps reconstructed from fixed interval {interval_s} s ({n} rows).",
    ))
    return normalized, diag, msgs


def execute_combine_date_time_columns(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Combine separate date and time columns into a single datetime Series."""
    msgs: list[ValidationMessage] = []
    aux = aux_series or {}

    date_col = plan.date_column
    time_col = plan.time_column

    date_s = aux.get(date_col, raw_series) if date_col else raw_series
    time_s = aux.get(time_col) if time_col else None

    if time_s is None:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_NO_TIME_COL",
            message=f"Time column '{time_col}' not found in aux_series; using date column only.",
        ))
        combined_str = date_s.astype(str)
    else:
        combined_str = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()

    normalized = pd.to_datetime(combined_str, format="mixed", errors="coerce")
    repaired = int(normalized.notna().sum())

    if normalized.isna().all():
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="TS_COMBINE_FAILED",
            message="Combined date+time column could not be parsed at all.",
        ))

    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=repaired)  # type: ignore[arg-type]
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs  # type: ignore[return-value]


def execute_excel_serial_conversion(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Convert Excel serial date numbers (days since 1899-12-30) to datetime."""
    msgs: list[ValidationMessage] = []

    normalized = _coerce_special(raw_series, _EXCEL_SERIAL_LABEL)

    if normalized.isna().all():
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="TS_EXCEL_SERIAL_FAILED",
            message="Excel serial conversion produced no valid timestamps.",
        ))
    else:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code="TS_EXCEL_SERIAL_CONVERTED",
            message=f"{normalized.notna().sum()} Excel serial date(s) converted.",
        ))

    repaired = int(normalized.notna().sum())
    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=repaired)
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


def execute_reconstruct_hybrid(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Hybrid anchor + sub-interval reconstruction for low-resolution timestamps.

    Each distinct consecutive anchor value is treated as the floor of a time
    window.  Within that window, samples are assigned:

        t[i_in_window] = anchor + i_in_window × dt

    where dt = 1 / sample_rate_hz.  When date and time live in separate columns
    (plan.date_column + plan.time_column) they are combined before parsing.

    The user may override:
        plan.override_start_datetime  — forces the first anchor to a specific time.
        plan.override_sample_rate_hz  — uses an explicit Fs instead of the inferred one.
        plan.sampling_interval_seconds — alternative to override_sample_rate_hz.
    """
    msgs: list[ValidationMessage] = []
    aux = aux_series or {}
    n = len(raw_series)

    # --- Step 1: Parse anchor timestamps -----------------------------------
    fmt = plan.detected_format or plan.user_format
    date_col = plan.date_column
    time_col = plan.time_column

    if date_col and time_col:
        date_s = aux.get(date_col, raw_series)
        time_s = aux.get(time_col)
        if time_s is None:
            msgs.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TS_HYBRID_NO_TIME_COL",
                message=f"Time column '{time_col}' not in aux_series; using date column only.",
            ))
            combined = date_s.astype(str)
        else:
            combined = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()
        anchors = pd.to_datetime(combined, format="mixed", errors="coerce")
    else:
        anchors = _coerce_to_datetime(raw_series, fmt=fmt)

    # --- Step 2: Determine dt ----------------------------------------------
    hz: float | None = plan.override_sample_rate_hz
    if hz is None and plan.sampling_interval_seconds and plan.sampling_interval_seconds > 0:
        hz = 1.0 / plan.sampling_interval_seconds
    if hz is None or hz <= 0:
        msgs.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TS_HYBRID_NO_RATE",
            message="Sample rate not specified and could not be inferred; defaulting to 50 Hz.",
        ))
        hz = 50.0
    dt_s = 1.0 / hz

    # --- Step 3: Override first anchor if user specified start datetime -----
    override_ts: pd.Timestamp | None = None
    if plan.override_start_datetime:
        try:
            override_ts = pd.Timestamp(plan.override_start_datetime)
        except Exception:  # noqa: BLE001
            msgs.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TS_HYBRID_BAD_START",
                message=f"Could not parse override start datetime '{plan.override_start_datetime}'; "
                        "using first anchor from file.",
            ))

    # --- Step 4: Walk consecutive runs and assign times --------------------
    result_vals: list[pd.Timestamp | type(pd.NaT)] = [pd.NaT] * n
    i = 0
    first_group = True
    while i < n:
        anchor_raw = anchors.iloc[i]
        if pd.isna(anchor_raw):
            i += 1
            continue
        # Collect the full consecutive run with this anchor value
        j = i + 1
        while j < n and anchors.iloc[j] == anchor_raw:
            j += 1
        # Determine the effective anchor for this group
        if first_group and override_ts is not None:
            effective_anchor = override_ts
            first_group = False
        else:
            effective_anchor = anchor_raw
            first_group = False
        # Assign sub-interval times: anchor is the floor of the window
        dt_td = pd.Timedelta(seconds=dt_s)
        for k in range(j - i):
            result_vals[i + k] = effective_anchor + k * dt_td
        i = j

    normalized = pd.Series(result_vals, index=raw_series.index, dtype="datetime64[ns]")
    repaired = int(normalized.notna().sum())

    msgs.append(ValidationMessage(
        severity=ValidationSeverity.INFO,
        code="TS_HYBRID_RECONSTRUCTED",
        message=(
            f"Hybrid reconstruction applied: {repaired} timestamps rebuilt "
            f"at {hz:.2f} Hz ({dt_s * 1000:.3f} ms/sample) from anchor windows."
        ),
    ))

    diag = _build_diagnostics(plan.strategy, raw_series, normalized, repaired_rows=repaired)
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


def execute_timezone_alignment(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Localize to source_timezone then convert to target_timezone (default UTC)."""
    msgs: list[ValidationMessage] = []

    fmt = plan.detected_format or plan.user_format
    parsed = _coerce_to_datetime(raw_series, fmt=fmt)

    src_tz = plan.source_timezone
    tgt_tz = plan.target_timezone or "UTC"

    # Check if already timezone-aware by inspecting the dtype
    is_tz_aware = isinstance(parsed.dtype, pd.DatetimeTZDtype)

    if is_tz_aware:
        try:
            normalized = parsed.dt.tz_convert(tgt_tz)
        except Exception as exc:
            msgs.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TS_TZ_CONVERT_FAILED",
                message=f"Timezone conversion to '{tgt_tz}' failed: {exc}",
            ))
            normalized = parsed
    else:
        try:
            if src_tz:
                localized = parsed.dt.tz_localize(src_tz, ambiguous="NaT", nonexistent="NaT")
            else:
                msgs.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="TS_TZ_ASSUMED_UTC",
                    message="No source timezone specified; timestamps assumed UTC.",
                ))
                localized = parsed.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            normalized = localized.dt.tz_convert(tgt_tz)
        except Exception as exc:
            msgs.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TS_TZ_LOCALIZE_FAILED",
                message=f"Timezone localize to '{src_tz}' failed: {exc}. Returning naive timestamps.",
            ))
            normalized = parsed

    diag = _build_diagnostics(plan.strategy, raw_series, normalized)
    diag.timezone_detected = src_tz
    diag.target_timezone = tgt_tz
    msgs += _messages_from_diagnostics(diag)
    return normalized, diag, msgs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXECUTORS = {
    TimestampRepairStrategy.NO_REPAIR:                 execute_no_repair,
    TimestampRepairStrategy.PARSE_DETECTED_FORMAT:     execute_parse_detected_format,
    TimestampRepairStrategy.PARSE_USER_FORMAT:         execute_parse_user_format,
    TimestampRepairStrategy.INTERPOLATE_MISSING:       execute_interpolate_missing,
    TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL: execute_reconstruct_from_interval,
    TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS: execute_combine_date_time_columns,
    TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION:   execute_excel_serial_conversion,
    TimestampRepairStrategy.TIMEZONE_ALIGNMENT:        execute_timezone_alignment,
    TimestampRepairStrategy.RECONSTRUCT_HYBRID:        execute_reconstruct_hybrid,
}


def dispatch(
    raw_series: pd.Series,
    plan: TimestampRepairPlan,
    aux_series: dict[str, pd.Series] | None = None,
) -> tuple[pd.Series, RepairDiagnostics, list[ValidationMessage]]:
    """Dispatch to the correct executor for plan.strategy."""
    executor = _EXECUTORS.get(plan.strategy, execute_no_repair)
    return executor(raw_series, plan, aux_series)
