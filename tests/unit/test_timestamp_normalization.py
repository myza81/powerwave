"""Tests for timestamp_normalizer.py — high-level normalization pipeline."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.import_wizard.timestamp_normalizer import (
    TimestampNormalizationResult,
    normalize_timestamps,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso_series(n: int = 5, start: str = "2024-01-01 00:00:00", freq: str = "1s") -> pd.Series:
    idx = pd.date_range(start=start, periods=n, freq=freq)
    return pd.Series(idx.strftime("%Y-%m-%d %H:%M:%S"))


def _plan(strategy: TimestampRepairStrategy, **kwargs) -> TimestampRepairPlan:
    return TimestampRepairPlan(strategy=strategy, repair_validated=True, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TimestampNormalizationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTimestampNormalizationResult:
    def test_success_flag_set_correctly(self):
        series = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert result.success is True

    def test_has_errors_false_when_clean(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert not result.has_errors()

    def test_valid_timestamps_excludes_nat(self):
        series = pd.Series(["2024-01-01 00:00:00", None, "2024-01-01 00:00:02"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert len(result.valid_timestamps()) == 2

    def test_nat_mask_correct(self):
        series = pd.Series(["2024-01-01 00:00:00", None, "2024-01-01 00:00:02"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.nat_mask().sum() == 1

    def test_result_contains_diagnostics(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert result.diagnostics is not None
        assert result.diagnostics.total_rows == 5

    def test_synthetic_elapsed_result_carries_seconds_axis(self):
        series = pd.Series([0, 0, 0])
        plan = _plan(
            TimestampRepairStrategy.GENERATE_SYNTHETIC_ELAPSED,
            sampling_interval_seconds=0.02,
        )

        result = normalize_timestamps(series, plan)

        assert result.success
        assert result.time_axis_mode == "synthetic_elapsed"
        assert result.time_axis_unit == "s"
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds.tolist() == pytest.approx([0.0, 0.02, 0.04])

    def test_sample_index_result_carries_sample_axis(self):
        series = pd.Series([0, 0, 0])
        plan = _plan(TimestampRepairStrategy.GENERATE_SAMPLE_INDEX)

        result = normalize_timestamps(series, plan)

        assert result.success
        assert result.time_axis_mode == "sample_index"
        assert result.time_axis_unit == "sample"
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds.tolist() == pytest.approx([0.0, 1.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# ISO timestamp normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestIsoNormalization:
    def test_iso_parses_correctly(self):
        series = _iso_series(5)
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.success
        assert result.normalized.notna().all()

    def test_normalized_is_datetime64(self):
        series = _iso_series(5)
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert pd.api.types.is_datetime64_any_dtype(result.normalized)

    def test_no_repair_generic_parse(self):
        series = _iso_series(5)
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert result.normalized.notna().sum() == 5

    def test_raw_series_not_mutated(self):
        series = _iso_series(5)
        original = series.copy()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        normalize_timestamps(series, plan)
        pd.testing.assert_series_equal(series, original)


# ─────────────────────────────────────────────────────────────────────────────
# User-specified format parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestUserFormatParsing:
    def test_dmy_format(self):
        series = pd.Series(["15/01/2024 00:00:00", "15/01/2024 00:00:01"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.normalized.notna().all()

    def test_wrong_format_produces_warning(self):
        series = pd.Series(["2024-01-15 00:00:00"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y")
        result = normalize_timestamps(series, plan)
        codes = {m.code for m in result.validation_messages}
        assert "TS_FORMAT_MISMATCH" in codes or result.normalized.isna().any()

    def test_missing_user_format_falls_back(self):
        series = _iso_series(3)
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT)
        result = normalize_timestamps(series, plan)
        msgs = result.validation_messages
        assert any(m.code == "TS_NO_USER_FORMAT" for m in msgs)

    def test_partial_parse_counts_nats(self):
        series = pd.Series(["15/01/2024 00:00:00", "INVALID", "15/01/2024 00:00:02"])
        plan = _plan(TimestampRepairStrategy.PARSE_USER_FORMAT,
                     user_format="%d/%m/%Y %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.normalized.isna().sum() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Malformed timestamp handling
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedHandling:
    def test_all_garbage_produces_error(self):
        series = pd.Series(["abc", "def", "ghi"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.has_errors()
        assert not result.success

    def test_mixed_valid_invalid(self):
        series = pd.Series([
            "2024-01-01 00:00:00", "GARBAGE", "2024-01-01 00:00:02"
        ])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.normalized.notna().sum() == 2
        assert result.normalized.isna().sum() == 1

    def test_empty_series(self):
        series = pd.Series([], dtype=object)
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert not result.success

    def test_none_values_become_nat(self):
        series = pd.Series([None, None, None])
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert result.normalized.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics integration
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticsIntegration:
    def test_total_rows_correct(self):
        series = _iso_series(10)
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        assert result.diagnostics.total_rows == 10

    def test_valid_rows_correct(self):
        series = pd.Series(["2024-01-01 00:00:00", None, "2024-01-01 00:00:02"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.diagnostics.valid_rows == 2

    def test_duplicate_rows_detected(self):
        series = pd.Series([
            "2024-01-01 00:00:00", "2024-01-01 00:00:00",
            "2024-01-01 00:00:01", "2024-01-01 00:00:02"
        ])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.diagnostics.duplicate_rows >= 1

    def test_non_monotonic_detected(self):
        series = pd.Series([
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:02",
            "2024-01-01 00:00:01",  # backward
            "2024-01-01 00:00:03"
        ])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.diagnostics.non_monotonic_rows >= 1

    def test_strategy_label_in_diagnostics(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        assert result.diagnostics.strategy_used == "parse_detected_format"


# ─────────────────────────────────────────────────────────────────────────────
# Validation message generation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationMessages:
    def test_duplicate_timestamp_warning(self):
        series = pd.Series(["2024-01-01 00:00:00", "2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        codes = {m.code for m in result.validation_messages}
        assert "TS_DUPLICATES" in codes

    def test_non_monotonic_warning(self):
        series = pd.Series([
            "2024-01-01 00:00:02", "2024-01-01 00:00:01"
        ])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        codes = {m.code for m in result.validation_messages}
        assert "TS_NON_MONOTONIC" in codes

    def test_dropped_rows_warning(self):
        series = pd.Series(["GARBAGE", "GARBAGE", "GARBAGE"])
        plan = _plan(TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
                     detected_format="%Y-%m-%d %H:%M:%S")
        result = normalize_timestamps(series, plan)
        codes = {m.code for m in result.validation_messages}
        assert "TS_ALL_NAT" in codes or "TS_DROPPED_ROWS" in codes

    def test_messages_have_correct_severity_types(self):
        series = _iso_series()
        plan = _plan(TimestampRepairStrategy.NO_REPAIR)
        result = normalize_timestamps(series, plan)
        for msg in result.validation_messages:
            assert isinstance(msg.severity, ValidationSeverity)
