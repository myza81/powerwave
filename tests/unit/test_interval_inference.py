"""Tests for interval_inference.py — interval inference and regularity analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.import_wizard.interval_inference import (
    IntervalAnalysis,
    detect_duplicates,
    detect_non_monotonic,
    infer_interval,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(freq: str = "1s", n: int = 10, start: str = "2024-01-01") -> pd.Series:
    return pd.Series(pd.date_range(start=start, periods=n, freq=freq))


def _ts_str(freq: str = "1s", n: int = 10) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.Series(idx.strftime("%Y-%m-%d %H:%M:%S.%f"))


# ─────────────────────────────────────────────────────────────────────────────
# infer_interval — basic properties
# ─────────────────────────────────────────────────────────────────────────────

class TestInferIntervalBasic:
    def test_returns_interval_analysis(self):
        result = infer_interval(_ts())
        assert isinstance(result, IntervalAnalysis)

    def test_total_samples_correct(self):
        result = infer_interval(_ts(n=20))
        assert result.total_samples == 20

    def test_valid_samples_correct(self):
        result = infer_interval(_ts(n=10))
        assert result.valid_samples == 10

    def test_nat_count_zero_for_clean_series(self):
        result = infer_interval(_ts())
        assert result.nat_count == 0

    def test_dominant_interval_1s(self):
        result = infer_interval(_ts(freq="1s", n=100))
        assert result.dominant_interval_seconds == pytest.approx(1.0, abs=0.001)

    def test_dominant_interval_20ms(self):
        result = infer_interval(_ts(freq="20ms", n=200))
        assert result.dominant_interval_seconds == pytest.approx(0.02, abs=0.001)

    def test_dominant_interval_100ms(self):
        result = infer_interval(_ts(freq="100ms", n=100))
        assert result.dominant_interval_seconds == pytest.approx(0.1, abs=0.001)

    def test_median_interval_close_to_dominant(self):
        result = infer_interval(_ts(freq="1s", n=100))
        assert result.median_interval_seconds == pytest.approx(1.0, abs=0.01)

    def test_mean_interval_close_to_dominant(self):
        result = infer_interval(_ts(freq="1s", n=100))
        assert result.mean_interval_seconds == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# infer_interval — jitter detection
# ─────────────────────────────────────────────────────────────────────────────

class TestJitterDetection:
    def test_jitter_near_zero_for_perfect_series(self):
        result = infer_interval(_ts(freq="1s", n=100))
        assert result.jitter_seconds == pytest.approx(0.0, abs=0.001)
        assert not result.is_irregular

    def test_jitter_detected_for_noisy_series(self):
        rng = np.random.default_rng(42)
        base = pd.date_range("2024-01-01", periods=100, freq="1s")
        noise_ns = (rng.normal(0, 0.1, 100) * 1e9).astype(int)
        noisy = pd.to_datetime(base.asi8 + noise_ns, unit="ns")
        result = infer_interval(pd.Series(noisy))
        # 100ms jitter on 1s interval = 10% → is_irregular
        assert result.jitter_seconds is not None
        assert result.is_irregular

    def test_jitter_fraction_computed(self):
        result = infer_interval(_ts(freq="1s", n=50))
        assert result.jitter_fraction is not None
        assert result.jitter_fraction >= 0.0

    def test_regular_5pct_threshold(self):
        # Exactly at threshold: jitter = 0.05 * dominant
        base = pd.date_range("2024-01-01", periods=50, freq="1s")
        dither = [0] * 50
        dither[25] = int(0.04 * 1e9)  # 40 ms jitter on one sample
        shifted = pd.to_datetime(base.asi8 + np.array(dither), unit="ns")
        result = infer_interval(pd.Series(shifted))
        # 40ms on 1s = 4% → not irregular
        assert not result.is_irregular


# ─────────────────────────────────────────────────────────────────────────────
# infer_interval — missing sample detection
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingSampleDetection:
    def test_no_missing_in_complete_series(self):
        result = infer_interval(_ts(freq="1s", n=50))
        assert result.missing_count == 0

    def test_detects_one_missing_sample(self):
        # Drop row 5 → gap of 2 s in 1 s series
        ts = pd.Series(pd.date_range("2024-01-01", periods=20, freq="1s"))
        ts = ts.drop(5).reset_index(drop=True)
        result = infer_interval(ts)
        assert result.missing_count >= 1

    def test_detects_multiple_missing_samples(self):
        # Drop 3 consecutive rows → gap of 4 s in 1 s series
        ts = pd.Series(pd.date_range("2024-01-01", periods=20, freq="1s"))
        ts = ts.drop([5, 6, 7]).reset_index(drop=True)
        result = infer_interval(ts)
        assert result.missing_count >= 3


# ─────────────────────────────────────────────────────────────────────────────
# infer_interval — duplicate and non-monotonic
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateNonMonotonic:
    def test_no_duplicates_in_clean_series(self):
        result = infer_interval(_ts())
        assert result.duplicate_count == 0

    def test_counts_duplicates(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        ts = pd.concat([ts, pd.Series([ts.iloc[2]])]).reset_index(drop=True)
        result = infer_interval(ts)
        assert result.duplicate_count >= 1

    def test_monotonic_for_clean_series(self):
        result = infer_interval(_ts())
        assert result.is_monotonic

    def test_non_monotonic_detected(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1s"))
        # Introduce backward step
        vals = ts.values.copy()
        vals[3] = vals[1]  # backward
        result = infer_interval(pd.Series(vals))
        assert not result.is_monotonic
        assert result.non_monotonic_count >= 1

    def test_non_monotonic_count_correct(self):
        ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1s"))
        vals = ts.values.copy()
        vals[3] = vals[1]
        vals[7] = vals[5]
        result = infer_interval(pd.Series(vals))
        assert result.non_monotonic_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# infer_interval — NaT handling
# ─────────────────────────────────────────────────────────────────────────────

class TestNatHandling:
    def test_counts_nats(self):
        ts = _ts(n=5).copy()
        ts.iloc[2] = pd.NaT
        result = infer_interval(ts)
        assert result.nat_count == 1
        assert result.valid_samples == 4

    def test_all_nat_returns_empty_analysis(self):
        ts = pd.Series([pd.NaT] * 5)
        result = infer_interval(ts)
        assert result.dominant_interval_seconds is None
        assert result.valid_samples == 0

    def test_single_valid_returns_empty_analysis(self):
        ts = pd.Series([pd.NaT, pd.Timestamp("2024-01-01"), pd.NaT])
        result = infer_interval(ts)
        assert result.dominant_interval_seconds is None

    def test_string_series_auto_coerced(self):
        ts = _ts_str(n=10)
        result = infer_interval(ts)
        assert result.dominant_interval_seconds == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# detect_duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDuplicates:
    def test_no_duplicates_returns_false_mask(self):
        ts = _ts(n=5)
        mask = detect_duplicates(ts)
        assert not mask.any()

    def test_duplicate_flagged(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 00:00:01"),
            pd.Timestamp("2024-01-01 00:00:01"),  # duplicate
            pd.Timestamp("2024-01-01 00:00:02"),
        ])
        mask = detect_duplicates(ts)
        assert mask.iloc[2]
        assert not mask.iloc[1]  # first occurrence not flagged

    def test_triple_duplicate_flags_second_and_third(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-01"),
        ])
        mask = detect_duplicates(ts)
        assert not mask.iloc[0]
        assert mask.iloc[1]
        assert mask.iloc[2]


# ─────────────────────────────────────────────────────────────────────────────
# detect_non_monotonic
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectNonMonotonic:
    def test_clean_series_returns_false_mask(self):
        ts = _ts(n=5)
        mask = detect_non_monotonic(ts)
        assert not mask.any()

    def test_backward_step_flagged(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 00:00:02"),
            pd.Timestamp("2024-01-01 00:00:01"),  # backward
            pd.Timestamp("2024-01-01 00:00:03"),
        ])
        mask = detect_non_monotonic(ts)
        assert mask.iloc[2]
        assert not mask.iloc[1]

    def test_flat_step_is_non_monotonic(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),  # flat = not strictly increasing
        ])
        mask = detect_non_monotonic(ts)
        assert mask.iloc[1]

    def test_nat_rows_never_flagged(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.NaT,
            pd.Timestamp("2024-01-01 00:00:02"),
        ])
        mask = detect_non_monotonic(ts)
        assert not mask.iloc[1]  # NaT should not be flagged


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_series(self):
        result = infer_interval(pd.Series([], dtype="datetime64[ns]"))
        assert result.total_samples == 0
        assert result.dominant_interval_seconds is None

    def test_single_element_series(self):
        result = infer_interval(pd.Series([pd.Timestamp("2024-01-01")]))
        assert result.total_samples == 1
        assert result.dominant_interval_seconds is None

    def test_two_element_series(self):
        ts = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        result = infer_interval(ts)
        assert result.dominant_interval_seconds == pytest.approx(86400.0, abs=1.0)

    def test_high_frequency_microsecond(self):
        # 20 kHz → 50 µs intervals
        ts = pd.Series(pd.date_range("2024-01-01", periods=200, freq="50us"))
        result = infer_interval(ts)
        assert result.dominant_interval_seconds == pytest.approx(50e-6, rel=0.01)
