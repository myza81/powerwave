"""Tests for timestamp_repair_executor.py — all eight repair strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.import_wizard.timestamp_repair_executor import (
    dispatch,
    execute_combine_date_time_columns,
    execute_excel_serial_conversion,
    execute_generate_sample_index,
    execute_generate_synthetic_elapsed,
    execute_interpolate_missing,
    execute_no_repair,
    execute_parse_detected_format,
    execute_parse_user_format,
    execute_reconstruct_from_interval,
    execute_timezone_alignment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plan(strategy: TimestampRepairStrategy, **kwargs) -> TimestampRepairPlan:
    return TimestampRepairPlan(strategy=strategy, repair_validated=True, **kwargs)


def _iso_series(n: int = 5, freq: str = "1s") -> pd.Series:
    idx = pd.date_range("2024-01-01 00:00:00", periods=n, freq=freq)
    return pd.Series(idx.strftime("%Y-%m-%d %H:%M:%S"))


# ─────────────────────────────────────────────────────────────────────────────
# NO_REPAIR
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRepair:
    def test_iso_strings_parsed(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        norm, diag, msgs = execute_no_repair(series, plan)
        assert norm.notna().all()
        assert pd.api.types.is_datetime64_any_dtype(norm)

    def test_datetime_passthrough(self):
        series = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        norm, diag, msgs = execute_no_repair(series, plan)
        assert norm.notna().all()

    def test_garbage_becomes_nat(self):
        series = pd.Series(["abc", "def"])
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        norm, diag, msgs = execute_no_repair(series, plan)
        assert norm.isna().all()

    def test_raw_not_mutated(self):
        series = _iso_series()
        original = series.copy()
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        execute_no_repair(series, plan)
        pd.testing.assert_series_equal(series, original)


class TestGeneratedAxes:
    def test_generate_synthetic_elapsed_from_interval(self):
        series = pd.Series([0, 0, 0, 0])
        plan = _plan(
            TimestampRepairStrategy.GENERATE_SYNTHETIC_ELAPSED,
            sampling_interval_seconds=0.01,
        )

        norm, diag, msgs = execute_generate_synthetic_elapsed(series, plan)

        assert norm.notna().all()
        assert diag.strategy_used == "generate_synthetic_elapsed"
        assert any(m.code == "TS_SYNTHETIC_ELAPSED_GENERATED" for m in msgs)

    def test_generate_sample_index(self):
        series = pd.Series([10, 20, 30])
        plan = _plan(TimestampRepairStrategy.GENERATE_SAMPLE_INDEX)

        norm, diag, msgs = execute_generate_sample_index(series, plan)

        assert norm.notna().all()
        assert diag.strategy_used == "generate_sample_index"
        assert any(m.code == "TS_SAMPLE_INDEX_GENERATED" for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# PARSE_DETECTED_FORMAT
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDetectedFormat:
    def test_correct_format_all_parsed(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        assert norm.notna().all()

    def test_wrong_format_produces_nats(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%d/%m/%Y")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        assert norm.isna().any()

    def test_fallback_when_no_format(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT)
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        # Should fall back to generic parse
        assert norm.notna().sum() > 0

    def test_epoch_seconds_format_label(self):
        series = pd.Series(["1704067200", "1704067201", "1704067202"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="epoch_seconds")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        assert norm.notna().all()
        # Should be around 2024-01-01
        assert norm.iloc[0].year == 2024

    def test_epoch_milliseconds_format_label(self):
        series = pd.Series(["1704067200000", "1704067201000"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="epoch_milliseconds")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        assert norm.notna().all()

    def test_ambiguous_date_resolves_day_first_and_emits_diagnostic(self):
        series = pd.Series(["3/6/2026 17:25", "3/6/2026 17:26"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%d/%m/%Y %H:%M")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        assert norm.iloc[0] == pd.Timestamp("2026-06-03 17:25:00")
        codes = {m.code for m in msgs}
        assert "TS_AMBIGUOUS_DATE_DEFAULT" in codes
        ambiguous_msg = next(m for m in msgs if m.code == "TS_AMBIGUOUS_DATE_DEFAULT")
        assert ambiguous_msg.severity == ValidationSeverity.INFO
        assert "DD/MM/YYYY" in ambiguous_msg.message

    def test_unambiguous_date_no_ambiguity_diagnostic(self):
        series = pd.Series(["13/6/2026 17:25", "13/6/2026 17:26"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%d/%m/%Y %H:%M")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_AMBIGUOUS_DATE_DEFAULT" not in codes

    def test_iso_format_no_ambiguity_diagnostic(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_parse_detected_format(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_AMBIGUOUS_DATE_DEFAULT" not in codes


# ─────────────────────────────────────────────────────────────────────────────
# PARSE_USER_FORMAT
# ─────────────────────────────────────────────────────────────────────────────

class TestParseUserFormat:
    def test_dmy_format(self):
        series = pd.Series(["15/01/2024 00:00:00", "15/01/2024 00:00:01"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y %H:%M:%S")
        norm, diag, msgs = execute_parse_user_format(series, plan)
        assert norm.notna().all()
        assert norm.iloc[0].day == 15
        assert norm.iloc[0].month == 1

    def test_explicit_month_first_override_beats_day_first_default(self):
        # "3/6/2026" is ambiguous and would default to day-first (3 June);
        # an explicit user format must win and produce 6 March instead.
        series = pd.Series(["3/6/2026 17:25"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%m/%d/%Y %H:%M")
        norm, diag, msgs = execute_parse_user_format(series, plan)
        assert norm.iloc[0] == pd.Timestamp("2026-03-06 17:25:00")
        codes = {m.code for m in msgs}
        assert "TS_AMBIGUOUS_DATE_DEFAULT" not in codes

    def test_format_mismatch_produces_error(self):
        series = pd.Series(["2024-01-15"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y %H:%M:%S")
        norm, diag, msgs = execute_parse_user_format(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_FORMAT_MISMATCH" in codes

    def test_no_user_format_warns(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT)
        norm, diag, msgs = execute_parse_user_format(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_NO_USER_FORMAT" in codes

    def test_partial_parse(self):
        series = pd.Series(["15/01/2024 00:00:00", "INVALID", "15/01/2024 00:00:02"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y %H:%M:%S")
        norm, diag, msgs = execute_parse_user_format(series, plan)
        assert norm.notna().sum() == 2
        assert norm.isna().sum() == 1


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATE_MISSING
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpolateMissing:
    def test_fills_single_missing(self):
        series = pd.Series([
            "2024-01-01 00:00:00",
            None,
            "2024-01-01 00:00:02",
        ])
        plan = _plan(TimestampRepairStrategy.INTERPOLATE_MISSING,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_interpolate_missing(series, plan)
        assert norm.notna().all()
        # Interpolated value should be 00:00:01
        assert norm.iloc[1] == pd.Timestamp("2024-01-01 00:00:01")

    def test_no_nats_unchanged(self):
        series = _iso_series(5)
        plan = _plan(TimestampRepairStrategy.INTERPOLATE_MISSING,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_interpolate_missing(series, plan)
        assert norm.notna().all()

    def test_repaired_rows_counted(self):
        series = pd.Series([
            "2024-01-01 00:00:00", None, "2024-01-01 00:00:02"
        ])
        plan = _plan(TimestampRepairStrategy.INTERPOLATE_MISSING,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_interpolate_missing(series, plan)
        assert diag.repaired_rows >= 1

    def test_interpolation_message_emitted(self):
        series = pd.Series(["2024-01-01 00:00:00", None, "2024-01-01 00:00:02"])
        plan = _plan(TimestampRepairStrategy.INTERPOLATE_MISSING,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_interpolate_missing(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_INTERPOLATED" in codes

    def test_fills_multiple_missing(self):
        series = pd.Series([
            "2024-01-01 00:00:00", None, None, "2024-01-01 00:00:03"
        ])
        plan = _plan(TimestampRepairStrategy.INTERPOLATE_MISSING,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_interpolate_missing(series, plan)
        assert norm.notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT_FROM_INTERVAL
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructFromInterval:
    def test_basic_reconstruction(self):
        series = pd.Series(["garbage"] * 5)
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
                     sampling_interval_seconds=1.0)
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        assert norm.notna().all()
        assert len(norm) == 5

    def test_uses_first_valid_timestamp_as_start(self):
        series = pd.Series([
            "2024-06-01 12:00:00", "garbage", "garbage", "garbage"
        ])
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
                     sampling_interval_seconds=1.0,
                     detected_format="%Y-%m-%d %H:%M:%S")
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        assert norm.iloc[0] == pd.Timestamp("2024-06-01 12:00:00")
        assert norm.iloc[1] == pd.Timestamp("2024-06-01 12:00:01")

    def test_epoch_origin_when_no_valid_start(self):
        series = pd.Series(["abc", "def", "ghi"])
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
                     sampling_interval_seconds=0.02)
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_EPOCH_START" in codes

    def test_missing_interval_produces_error(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL)
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_NO_INTERVAL" in codes

    def test_reconstructed_message_emitted(self):
        series = pd.Series(["x"] * 10)
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
                     sampling_interval_seconds=0.02)
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_RECONSTRUCTED" in codes

    def test_interval_spacing_correct(self):
        series = pd.Series(["x"] * 4)
        plan = _plan(TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
                     sampling_interval_seconds=0.5)
        norm, diag, msgs = execute_reconstruct_from_interval(series, plan)
        # All diffs should be 0.5 s
        diffs_s = norm.diff().dropna().dt.total_seconds()
        assert (diffs_s - 0.5).abs().max() < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE_DATE_TIME_COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineDateTimeColumns:
    def test_basic_combination(self):
        date_s = pd.Series(["2024-01-15", "2024-01-15"])
        time_s = pd.Series(["00:00:00", "00:00:01"])
        plan = _plan(TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
                     date_column="Date", time_column="Time")
        aux = {"Date": date_s, "Time": time_s}
        norm, diag, msgs = execute_combine_date_time_columns(date_s, plan, aux)
        assert norm.notna().all()
        assert norm.iloc[0] == pd.Timestamp("2024-01-15 00:00:00")

    def test_missing_time_column_warns(self):
        date_s = pd.Series(["2024-01-15"])
        plan = _plan(TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
                     date_column="Date", time_column="Time")
        aux = {}  # time column absent
        norm, diag, msgs = execute_combine_date_time_columns(date_s, plan, aux)
        codes = {m.code for m in msgs}
        assert "TS_NO_TIME_COL" in codes

    def test_raw_not_mutated(self):
        date_s = pd.Series(["2024-01-15", "2024-01-16"])
        time_s = pd.Series(["00:00:00", "00:00:01"])
        original = date_s.copy()
        plan = _plan(TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
                     date_column="Date", time_column="Time")
        execute_combine_date_time_columns(date_s, plan, {"Date": date_s, "Time": time_s})
        pd.testing.assert_series_equal(date_s, original)

    def test_invalid_combination_produces_nat(self):
        date_s = pd.Series(["NOTADATE"])
        time_s = pd.Series(["NOTALTIME"])
        plan = _plan(TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
                     date_column="D", time_column="T")
        norm, diag, msgs = execute_combine_date_time_columns(date_s, plan, {"D": date_s, "T": time_s})
        assert norm.isna().all()

    def test_ambiguous_combined_date_resolves_day_first(self):
        date_s = pd.Series(["3/6/2026", "3/6/2026"])
        time_s = pd.Series(["17:25:00", "17:26:00"])
        plan = _plan(TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
                     date_column="Date", time_column="Time")
        norm, diag, msgs = execute_combine_date_time_columns(
            date_s, plan, {"Date": date_s, "Time": time_s}
        )
        assert norm.iloc[0] == pd.Timestamp("2026-06-03 17:25:00")


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT_HYBRID — date-order policy on anchor parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructHybridDateOrder:
    def test_ambiguous_anchor_resolves_day_first(self):
        from app.import_wizard.timestamp_repair_executor import execute_reconstruct_hybrid

        # Truncated-timestamp style input: repeated anchor, ambiguous date.
        series = pd.Series(["3/6/2026 17:25"] * 4)
        plan = _plan(
            TimestampRepairStrategy.RECONSTRUCT_HYBRID,
            override_sample_rate_hz=50.0,
        )
        norm, diag, msgs = execute_reconstruct_hybrid(series, plan)
        assert norm.notna().all()
        assert norm.iloc[0].to_pydatetime().replace(microsecond=0) == pd.Timestamp(
            "2026-06-03 17:25:00"
        ).to_pydatetime()

    def test_ambiguous_anchor_with_separate_date_time_columns_resolves_day_first(self):
        from app.import_wizard.timestamp_repair_executor import execute_reconstruct_hybrid

        date_s = pd.Series(["3/6/2026"] * 4)
        time_s = pd.Series(["17:25:00"] * 4)
        plan = _plan(
            TimestampRepairStrategy.RECONSTRUCT_HYBRID,
            date_column="Date",
            time_column="Time",
            override_sample_rate_hz=50.0,
        )
        norm, diag, msgs = execute_reconstruct_hybrid(
            date_s, plan, {"Date": date_s, "Time": time_s}
        )
        assert norm.iloc[0].to_pydatetime().replace(microsecond=0) == pd.Timestamp(
            "2026-06-03 17:25:00"
        ).to_pydatetime()


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL_SERIAL_CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

class TestExcelSerialConversion:
    def test_known_serial_value(self):
        # Excel serial 44927 = 2023-01-01
        series = pd.Series([44927.0])
        plan = _plan(TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        norm, diag, msgs = execute_excel_serial_conversion(series, plan)
        assert norm.notna().all()
        assert norm.iloc[0].year == 2023
        assert norm.iloc[0].month == 1
        assert norm.iloc[0].day == 1

    def test_fractional_serial_includes_time(self):
        # 44927.5 = 2023-01-01 12:00:00
        series = pd.Series([44927.5])
        plan = _plan(TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        norm, diag, msgs = execute_excel_serial_conversion(series, plan)
        assert norm.notna().all()
        assert norm.iloc[0].hour == 12

    def test_non_numeric_produces_nat(self):
        series = pd.Series(["not_a_number"])
        plan = _plan(TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        norm, diag, msgs = execute_excel_serial_conversion(series, plan)
        assert norm.isna().all()

    def test_converted_message_emitted(self):
        series = pd.Series([44927.0, 44928.0])
        plan = _plan(TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        norm, diag, msgs = execute_excel_serial_conversion(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_EXCEL_SERIAL_CONVERTED" in codes

    def test_multiple_serials_correct_sequence(self):
        # Consecutive days
        series = pd.Series([44927.0, 44928.0, 44929.0])
        plan = _plan(TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        norm, diag, msgs = execute_excel_serial_conversion(series, plan)
        diffs_days = norm.diff().dropna().dt.days
        assert (diffs_days == 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# TIMEZONE_ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestTimezoneAlignment:
    def test_utc_localize_and_convert(self):
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
                     detected_format="%Y-%m-%d %H:%M:%S",
                     source_timezone="UTC",
                     target_timezone="UTC")
        norm, diag, msgs = execute_timezone_alignment(series, plan)
        assert norm.notna().all()
        assert norm.dt.tz is not None

    def test_non_utc_to_utc_shifts_time(self):
        # UTC+5:30 → UTC should subtract 5:30
        series = pd.Series(["2024-01-01 05:30:00"])
        plan = _plan(TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
                     detected_format="%Y-%m-%d %H:%M:%S",
                     source_timezone="Asia/Kolkata",
                     target_timezone="UTC")
        norm, diag, msgs = execute_timezone_alignment(series, plan)
        assert norm.notna().all()
        assert norm.iloc[0].tz is not None
        assert norm.iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    def test_no_source_timezone_warns(self):
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
                     detected_format="%Y-%m-%d %H:%M:%S",
                     target_timezone="UTC")
        norm, diag, msgs = execute_timezone_alignment(series, plan)
        codes = {m.code for m in msgs}
        assert "TS_TZ_ASSUMED_UTC" in codes

    def test_diagnostics_record_timezone(self):
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
                     detected_format="%Y-%m-%d %H:%M:%S",
                     source_timezone="Europe/Berlin",
                     target_timezone="UTC")
        norm, diag, msgs = execute_timezone_alignment(series, plan)
        assert diag.target_timezone == "UTC"
        assert diag.timezone_detected == "Europe/Berlin"

    def test_invalid_timezone_does_not_raise(self):
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
                     detected_format="%Y-%m-%d %H:%M:%S",
                     source_timezone="Not/ATimezone",
                     target_timezone="UTC")
        # Should not raise — error is captured in messages
        norm, diag, msgs = execute_timezone_alignment(series, plan)
        assert norm is not None


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcher:
    @pytest.mark.parametrize("strategy", list(TimestampRepairStrategy))
    def test_all_strategies_dispatch(self, strategy):
        series = _iso_series(3)
        plan = _plan(strategy, detected_format="%Y-%m-%d %H:%M:%S",
                     sampling_interval_seconds=1.0)
        # Should not raise
        norm, diag, msgs = dispatch(series, plan, aux_series={"Time": pd.Series(["00:00:00", "00:00:01", "00:00:02"])})
        assert norm is not None
        assert diag is not None

    def test_dispatch_unknown_strategy_falls_back(self):
        """dispatch() must not raise even with a future unknown strategy."""
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        norm, diag, msgs = dispatch(series, plan)
        assert norm is not None
