"""Unit tests for the per-curve measurement engine (Sprint 1A) in
app.visualization.interaction.measurement_engine.

Pure computation tests -- no Qt, no session, no canvas. Each curve supplies
its own (time, values) array; compute_measurements() must never require
curves to share a length or time base, and must never silently omit a
curve's row from the result.
"""
from __future__ import annotations

import time as time_module

import numpy as np
import pytest

from app.visualization.interaction.measurement_engine import (
    ChannelMeasurement,
    CurveSample,
    MeasurementResult,
    compute_measurements,
)


def _curve(name: str, unit: str, time: np.ndarray, values: np.ndarray) -> CurveSample:
    return CurveSample(name=name, unit=unit, time=np.asarray(time, dtype=np.float64), values=np.asarray(values, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Same sample rate (regression baseline against the previous shared-array behaviour)
# ─────────────────────────────────────────────────────────────────────────────


class TestSameSampleRateRegression:
    """These values were computed by hand against the exact arrays below and
    must match what the old shared-time-array implementation produced for
    same-rate channels -- this is the "no regression" contract."""

    def _two_matched_curves(self) -> tuple[CurveSample, CurveSample]:
        t = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
        a = _curve("A", "kV", t, np.arange(11, dtype=float))          # 0..10
        b = _curve("B", "kV", t, np.arange(11, dtype=float) * 2.0)    # 0..20
        return a, b

    def test_cursor_values_and_delta(self) -> None:
        a, b = self._two_matched_curves()
        result = compute_measurements(0.2, 0.5, [a, b])
        ch_a = result.channels[0]
        assert ch_a.available
        assert ch_a.y_at_a == pytest.approx(2.0)
        assert ch_a.y_at_b == pytest.approx(5.0)
        assert ch_a.delta_y == pytest.approx(3.0)

    def test_segment_statistics(self) -> None:
        a, b = self._two_matched_curves()
        result = compute_measurements(0.2, 0.5, [a, b])
        ch_a = result.channels[0]
        # samples at t=0.2..0.5 inclusive -> values [2,3,4,5]
        assert ch_a.sample_count == 4
        assert ch_a.mean == pytest.approx(3.5)
        assert ch_a.peak == pytest.approx(5.0)
        assert ch_a.peak_to_peak == pytest.approx(3.0)
        assert ch_a.rms == pytest.approx(np.sqrt(np.mean(np.array([2.0, 3.0, 4.0, 5.0]) ** 2)))

    def test_time_summary(self) -> None:
        a, b = self._two_matched_curves()
        result = compute_measurements(0.2, 0.5, [a, b], nominal_hz=50.0)
        assert result.delta_t_s == pytest.approx(0.3)
        assert result.delta_t_ms == pytest.approx(300.0)
        assert result.delta_t_cycles == pytest.approx(15.0)
        assert result.frequency_hz == pytest.approx(1.0 / 0.3)

    def test_reversed_cursor_order_gives_same_segment(self) -> None:
        a, b = self._two_matched_curves()
        forward = compute_measurements(0.2, 0.5, [a])
        backward = compute_measurements(0.5, 0.2, [a])
        assert forward.channels[0].mean == pytest.approx(backward.channels[0].mean)
        assert forward.delta_t_s == pytest.approx(backward.delta_t_s)

    def test_energy_computed_for_matching_time_base_vi_pair(self) -> None:
        t = np.linspace(0.0, 1.0, 11)
        v = _curve("Va", "kV", t, np.full(11, 2.0))
        i = _curve("Ia", "A", t, np.full(11, 3.0))
        result = compute_measurements(0.0, 1.0, [v, i])
        assert result.energy_ws is not None
        assert result.energy_pair == ("Va", "Ia")
        # constant 2kV * 3A over 1s = 6 (kJ-ish in these synthetic units)
        assert result.energy_ws == pytest.approx(6.0, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed sample rates -- the core Sprint 1A fix
# ─────────────────────────────────────────────────────────────────────────────


class TestMixedSampleRates:
    def test_two_curves_different_lengths_both_available(self) -> None:
        """This is the exact bug being fixed: previously any curve whose
        length differed from the shared time array was silently dropped."""
        fast = _curve("Relay", "A", np.linspace(0, 1, 6400), np.sin(np.linspace(0, 1, 6400) * 50))
        slow = _curve("PMU", "kV", np.linspace(0, 1, 50), np.cos(np.linspace(0, 1, 50) * 50))
        result = compute_measurements(0.3, 0.6, [fast, slow])
        names = {ch.name: ch for ch in result.channels}
        assert set(names) == {"Relay", "PMU"}
        assert names["Relay"].available
        assert names["PMU"].available
        assert names["Relay"].y_at_a is not None
        assert names["PMU"].y_at_a is not None

    def test_five_different_rates_all_return_a_value(self) -> None:
        """6400 sps, 3200 sps, 1600 sps, PMU 50 sps, and a full-resolution
        calculated-signal-like curve, all visible together in one panel."""
        span = (0.0, 2.0)
        curves = [
            _curve("Relay_6400", "A", np.linspace(*span, 12800), np.sin(np.linspace(*span, 12800) * 100)),
            _curve("Relay_3200", "A", np.linspace(*span, 6400), np.sin(np.linspace(*span, 6400) * 100)),
            _curve("Relay_1600", "A", np.linspace(*span, 3200), np.sin(np.linspace(*span, 3200) * 100)),
            _curve("PMU_50", "kV", np.linspace(*span, 100), np.cos(np.linspace(*span, 100) * 5)),
            _curve("ƒ CalcSignal", "MW", np.linspace(*span, 250_000), np.sin(np.linspace(*span, 250_000) * 3)),
        ]
        result = compute_measurements(0.5, 1.5, curves)
        assert len(result.channels) == 5
        for ch in result.channels:
            assert ch.available, f"{ch.name} should be available"
            assert ch.y_at_a is not None
            assert ch.y_at_b is not None

    def test_lengths_never_required_to_match(self) -> None:
        a = _curve("A", "", np.linspace(0, 1, 3), [1.0, 2.0, 3.0])
        b = _curve("B", "", np.linspace(0, 1, 1000), np.linspace(10, 20, 1000))
        result = compute_measurements(0.0, 1.0, [a, b])
        assert all(ch.available for ch in result.channels)


# ─────────────────────────────────────────────────────────────────────────────
# Visible-but-unavailable curves
# ─────────────────────────────────────────────────────────────────────────────


class TestUnavailableCurvesNeverOmitted:
    def test_empty_curve_reported_as_unavailable(self) -> None:
        empty = _curve("Empty", "V", [], [])
        result = compute_measurements(0.0, 1.0, [empty])
        assert len(result.channels) == 1
        ch = result.channels[0]
        assert ch.name == "Empty"
        assert not ch.available
        assert ch.unavailable_reason is not None

    def test_malformed_curve_length_mismatch_reported_as_unavailable(self) -> None:
        bad = CurveSample(name="Bad", unit="", time=np.array([0.0, 1.0, 2.0]), values=np.array([1.0, 2.0]))
        result = compute_measurements(0.0, 1.0, [bad])
        assert not result.channels[0].available
        assert "mismatch" in result.channels[0].unavailable_reason.lower()

    def test_no_overlap_with_cursor_range_reported_as_unavailable(self) -> None:
        curve = _curve("Early", "V", np.linspace(0.0, 1.0, 100), np.ones(100))
        result = compute_measurements(5.0, 6.0, [curve])
        ch = result.channels[0]
        assert not ch.available
        assert "outside" in ch.unavailable_reason.lower() or "overlap" in ch.unavailable_reason.lower()

    def test_all_nan_in_overlap_reported_as_unavailable(self) -> None:
        t = np.linspace(0.0, 1.0, 50)
        v = np.full(50, np.nan)
        curve = _curve("AllNaN", "V", t, v)
        result = compute_measurements(0.2, 0.6, [curve])
        ch = result.channels[0]
        assert not ch.available
        assert ch.unavailable_reason is not None

    def test_unavailable_curve_does_not_remove_available_curves(self) -> None:
        good = _curve("Good", "V", np.linspace(0, 1, 100), np.ones(100))
        empty = _curve("Empty", "V", [], [])
        result = compute_measurements(0.0, 1.0, [good, empty])
        assert len(result.channels) == 2
        by_name = {ch.name: ch for ch in result.channels}
        assert by_name["Good"].available
        assert not by_name["Empty"].available

    def test_curve_order_preserved(self) -> None:
        curves = [_curve(f"C{i}", "", np.linspace(0, 1, 10), np.arange(10, dtype=float)) for i in range(5)]
        result = compute_measurements(0.0, 1.0, curves)
        assert [ch.name for ch in result.channels] == ["C0", "C1", "C2", "C3", "C4"]


# ─────────────────────────────────────────────────────────────────────────────
# Cursor outside range / edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestCursorEdgeCases:
    def test_one_cursor_outside_curve_range_still_available_if_curve_overlaps(self) -> None:
        """Curve spans [0, 1]; cursor B sits past the curve's end, but the
        curve still overlaps the cursor segment [0.5, 5.0] -- available,
        using the existing clamp-to-boundary nearest-sample policy."""
        curve = _curve("C", "V", np.linspace(0.0, 1.0, 11), np.arange(11, dtype=float))
        result = compute_measurements(0.5, 5.0, [curve])
        ch = result.channels[0]
        assert ch.available
        assert ch.y_at_b == pytest.approx(10.0)  # clamped to the last sample

    def test_sparse_curve_between_close_cursors_has_no_segment_stats(self) -> None:
        """A low-rate curve may have zero samples strictly between two
        closely-spaced cursors -- still available (nearest values exist),
        just no interior statistics."""
        curve = _curve("PMU", "kV", np.linspace(0.0, 10.0, 20), np.arange(20, dtype=float))  # ~0.53s spacing
        result = compute_measurements(1.0, 1.01, [curve])
        ch = result.channels[0]
        assert ch.available
        assert ch.y_at_a is not None
        assert ch.y_at_b is not None
        assert ch.sample_count == 1
        assert ch.rms is None
        assert ch.mean is None

    def test_coincident_cursors(self) -> None:
        curve = _curve("C", "V", np.linspace(0.0, 1.0, 11), np.arange(11, dtype=float))
        result = compute_measurements(0.5, 0.5, [curve])
        assert result.delta_t_s == 0.0
        assert result.frequency_hz is None
        assert not result.valid  # MeasurementResult.valid requires delta_t_s > 1e-12

    def test_frequency_none_but_channel_still_available_when_coincident(self) -> None:
        curve = _curve("C", "V", np.linspace(0.0, 1.0, 11), np.arange(11, dtype=float))
        result = compute_measurements(0.5, 0.5, [curve])
        assert result.channels[0].available


# ─────────────────────────────────────────────────────────────────────────────
# Delta / statistics sanity (sign convention preserved from prior implementation)
# ─────────────────────────────────────────────────────────────────────────────


class TestDeltaAndStatistics:
    def test_delta_y_is_signed_b_minus_a(self) -> None:
        t = np.linspace(0, 1, 11)
        curve = _curve("C", "V", t, np.arange(11, dtype=float))
        # t_a=0.8 -> y_at_a=8; t_b=0.2 -> y_at_b=2; delta_y = y_at_b - y_at_a = -6 (signed, can be negative)
        result = compute_measurements(0.8, 0.2, [curve])
        ch = result.channels[0]
        assert ch.y_at_a == pytest.approx(8.0)
        assert ch.y_at_b == pytest.approx(2.0)
        assert ch.delta_y == pytest.approx(-6.0)

    def test_peak_is_max_absolute_value(self) -> None:
        t = np.linspace(0, 1, 11)
        curve = _curve("C", "V", t, np.array([-9.0, -5, -1, 0, 1, 2, 3, 4, 5, 6, 7]))
        result = compute_measurements(0.0, 1.0, [curve])
        assert result.channels[0].peak == pytest.approx(9.0)


# ─────────────────────────────────────────────────────────────────────────────
# Energy pairing at mixed rates -- must not fabricate a resampled result
# ─────────────────────────────────────────────────────────────────────────────


class TestEnergyMixedRates:
    def test_energy_not_computed_for_mismatched_time_bases(self) -> None:
        v = _curve("Va", "kV", np.linspace(0, 1, 100), np.full(100, 2.0))
        i = _curve("Ia", "A", np.linspace(0, 1, 55), np.full(55, 3.0))
        result = compute_measurements(0.0, 1.0, [v, i])
        assert result.energy_ws is None
        assert result.energy_pair is None
        # both channels are still individually available/measurable
        assert all(ch.available for ch in result.channels)

    def test_energy_skipped_when_either_pair_channel_unavailable(self) -> None:
        v = _curve("Va", "kV", [], [])
        i = _curve("Ia", "A", np.linspace(0, 1, 10), np.full(10, 3.0))
        result = compute_measurements(0.0, 1.0, [v, i])
        assert result.energy_ws is None


# ─────────────────────────────────────────────────────────────────────────────
# Performance -- one search per curve, no per-sample Python loop
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_large_curves_no_per_sample_python_loop(self) -> None:
        n = 2_000_000
        t = np.linspace(0.0, 100.0, n)
        curves = [
            _curve("Big1", "MW", t, np.sin(t)),
            _curve("Big2", "MW", t, np.cos(t)),
        ]
        started = time_module.perf_counter()
        result = compute_measurements(10.0, 20.0, curves)
        elapsed = time_module.perf_counter() - started
        assert all(ch.available for ch in result.channels)
        assert elapsed < 5.0, f"compute_measurements took {elapsed:.2f}s for {n}-sample curves"

    def test_many_curves_each_large(self) -> None:
        n = 200_000
        t = np.linspace(0.0, 10.0, n)
        curves = [_curve(f"C{i}", "", t, np.sin(t + i)) for i in range(10)]
        started = time_module.perf_counter()
        result = compute_measurements(2.0, 8.0, curves)
        elapsed = time_module.perf_counter() - started
        assert len(result.channels) == 10
        assert elapsed < 5.0, f"took {elapsed:.2f}s for 10x{n}-sample curves"
