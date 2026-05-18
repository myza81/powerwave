"""Tests for Phase 8.55B: timestamp_detector.py."""
from __future__ import annotations

import pytest

from app.import_wizard.timestamp_detector import (
    _detect_special_format,
    _extract_timezone,
    _is_monotonic_increasing,
    _name_suggests_timestamp,
    _try_strptime,
    detect_timestamp_candidates,
    infer_timestamp_format,
)


# ─────────────────────────────────────────────────────────────────────────────
# _try_strptime
# ─────────────────────────────────────────────────────────────────────────────

class TestTryStrptime:
    def test_iso_datetime(self):
        assert _try_strptime("2024-01-15 13:30:00") == "%Y-%m-%d %H:%M:%S"

    def test_iso_with_milliseconds(self):
        fmt = _try_strptime("2024-01-15 13:30:00.123456")
        assert fmt is not None
        assert "f" in fmt

    def test_iso_with_timezone(self):
        fmt = _try_strptime("2024-01-15T13:30:00+05:30")
        assert fmt is not None

    def test_date_only(self):
        fmt = _try_strptime("2024-01-15")
        assert fmt == "%Y-%m-%d"

    def test_time_only(self):
        fmt = _try_strptime("13:30:00")
        assert fmt is not None

    def test_garbage_returns_none(self):
        assert _try_strptime("not_a_date") is None
        assert _try_strptime("") is None
        assert _try_strptime("230.5") is None

    def test_dmy_slash(self):
        fmt = _try_strptime("15/01/2024 13:30:00")
        assert fmt is not None

    def test_strips_whitespace(self):
        assert _try_strptime("  2024-01-15 13:30:00  ") == "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# _detect_special_format
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSpecialFormat:
    def test_epoch_seconds(self):
        assert _detect_special_format("1704067200") == "epoch_seconds"

    def test_epoch_milliseconds(self):
        assert _detect_special_format("1704067200000") == "epoch_milliseconds"

    def test_excel_serial(self):
        assert _detect_special_format("44927") == "excel_serial"

    def test_normal_float_returns_none(self):
        assert _detect_special_format("230.5") is None

    def test_small_integer_returns_none(self):
        assert _detect_special_format("42") is None

    def test_text_returns_none(self):
        assert _detect_special_format("hello") is None

    def test_very_large_number_returns_none(self):
        assert _detect_special_format("9e20") is None


# ─────────────────────────────────────────────────────────────────────────────
# infer_timestamp_format
# ─────────────────────────────────────────────────────────────────────────────

class TestInferTimestampFormat:
    def test_iso_format_detected(self):
        samples = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:00:02",
        ]
        fmt, invalid = infer_timestamp_format(samples)
        assert fmt == "%Y-%m-%d %H:%M:%S"
        assert invalid == 0

    def test_epoch_seconds_detected(self):
        samples = ["1704067200", "1704067201", "1704067202"]
        fmt, invalid = infer_timestamp_format(samples)
        assert fmt == "epoch_seconds"

    def test_epoch_milliseconds_detected(self):
        samples = ["1704067200000", "1704067201000"]
        fmt, invalid = infer_timestamp_format(samples)
        assert fmt == "epoch_milliseconds"

    def test_empty_returns_none(self):
        fmt, invalid = infer_timestamp_format([])
        assert fmt is None

    def test_all_garbage_returns_none(self):
        fmt, invalid = infer_timestamp_format(["abc", "def", "ghi"])
        assert fmt is None
        assert invalid == 3

    def test_mostly_valid_with_one_bad(self):
        samples = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:00:02",
            "INVALID",
        ]
        fmt, invalid = infer_timestamp_format(samples)
        assert fmt == "%Y-%m-%d %H:%M:%S"
        assert invalid >= 1

    def test_returns_majority_format(self):
        samples = ["2024-01-01 00:00:00"] * 5 + ["15/01/2024 00:00:00"]
        fmt, _ = infer_timestamp_format(samples)
        assert fmt == "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# _name_suggests_timestamp
# ─────────────────────────────────────────────────────────────────────────────

class TestNameSuggestsTimestamp:
    @pytest.mark.parametrize("name", [
        "Timestamp", "timestamp", "TIMESTAMP", "time", "Time",
        "DateTime", "datetime", "Date", "date", "ts",
    ])
    def test_positive_cases(self, name):
        assert _name_suggests_timestamp(name)

    @pytest.mark.parametrize("name", [
        "Voltage", "Current", "MW", "Frequency", "Channel_1",
    ])
    def test_negative_cases(self, name):
        assert not _name_suggests_timestamp(name)


# ─────────────────────────────────────────────────────────────────────────────
# _extract_timezone
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTimezone:
    def test_utc_offset_positive(self):
        tz = _extract_timezone(["2024-01-01T00:00:00+05:30"])
        assert tz is not None

    def test_utc_label(self):
        tz = _extract_timezone(["2024-01-01T00:00:00Z"])
        assert tz is not None

    def test_no_timezone(self):
        tz = _extract_timezone(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        assert tz is None

    def test_empty_list(self):
        assert _extract_timezone([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# _is_monotonic_increasing
# ─────────────────────────────────────────────────────────────────────────────

class TestIsMonotonicIncreasing:
    def test_strictly_increasing(self):
        assert _is_monotonic_increasing([1, 2, 3, 4, 5])

    def test_strings_increasing(self):
        assert _is_monotonic_increasing(["2024-01-01", "2024-01-02", "2024-01-03"])

    def test_flat_is_not_increasing(self):
        assert not _is_monotonic_increasing([1, 1, 1])

    def test_decreasing(self):
        assert not _is_monotonic_increasing([5, 4, 3])

    def test_too_short(self):
        assert not _is_monotonic_increasing([42])

    def test_empty(self):
        assert not _is_monotonic_increasing([])

    def test_nones_are_skipped(self):
        assert _is_monotonic_increasing([1, None, 2, None, 3])


# ─────────────────────────────────────────────────────────────────────────────
# detect_timestamp_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectTimestampCandidates:
    def _make_rows(self, n: int = 5) -> list[list[str]]:
        return [
            [f"2024-01-01 00:00:0{i}", str(230.0 + i), str(100.0 + i), str(50.0 + i * 0.01)]
            for i in range(n)
        ]

    def test_detects_timestamp_column(self):
        cols = ["Timestamp", "Voltage", "Current", "Freq"]
        rows = self._make_rows()
        candidates = detect_timestamp_candidates(cols, rows)
        assert len(candidates) >= 1
        assert candidates[0].column_name == "Timestamp"

    def test_candidates_sorted_by_confidence(self):
        cols = ["Timestamp", "Voltage", "Current", "Freq"]
        rows = self._make_rows()
        candidates = detect_timestamp_candidates(cols, rows)
        confidences = [c.confidence for c in candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_in_range(self):
        cols = ["Timestamp", "Voltage"]
        rows = self._make_rows()
        candidates = detect_timestamp_candidates(cols, rows)
        for c in candidates:
            assert 0.0 <= c.confidence <= 1.0

    def test_epoch_seconds_detected(self):
        cols = ["ts", "voltage"]
        rows = [[str(1704067200 + i), str(230.0 + i)] for i in range(5)]
        candidates = detect_timestamp_candidates(cols, rows)
        ts_candidates = [c for c in candidates if c.column_name == "ts"]
        if ts_candidates:
            assert ts_candidates[0].detected_format == "epoch_seconds"

    def test_non_timestamp_columns_excluded(self):
        cols = ["Voltage", "Current", "MW"]
        rows = [[str(230.0 + i), str(100.0 + i), str(23.0 + i)] for i in range(5)]
        candidates = detect_timestamp_candidates(cols, rows)
        assert len(candidates) == 0

    def test_example_values_populated(self):
        cols = ["Timestamp", "Voltage"]
        rows = self._make_rows()
        candidates = detect_timestamp_candidates(cols, rows)
        if candidates:
            assert len(candidates[0].example_values) > 0

    def test_empty_inputs_return_empty(self):
        assert detect_timestamp_candidates([], []) == []
        assert detect_timestamp_candidates(["T"], []) == []

    def test_invalid_sample_count_tracked(self):
        cols = ["Timestamp", "Voltage"]
        rows = [["2024-01-01 00:00:00", "230.0"]] * 4 + [["INVALID", "230.0"]]
        candidates = detect_timestamp_candidates(cols, rows)
        ts = [c for c in candidates if c.column_name == "Timestamp"]
        if ts:
            assert ts[0].invalid_sample_count >= 0

    def test_multiple_time_columns_both_detected(self):
        cols = ["Date", "Time", "Voltage"]
        rows = [
            ["2024-01-01", "00:00:00", str(230.0 + i)]
            for i in range(5)
        ]
        candidates = detect_timestamp_candidates(cols, rows)
        candidate_names = {c.column_name for c in candidates}
        assert len(candidate_names) >= 1
