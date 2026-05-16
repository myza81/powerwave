"""Unit tests for app/analytics/rms/sliding_rms.py

Tests cover:
  - compute_window_samples: engineering-parameter → integer
  - compute_rms_overlay: sinewave correctness, NaN handling, time alignment
  - Boundary / edge cases
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.analytics.rms.rms_models import RMSConfig, RMSWindowMode
from app.analytics.rms.sliding_rms import (
    compute_rms_overlay,
    compute_rms_window_samples,
    compute_window_samples,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeWindowSamples
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeWindowSamples:
    def test_5khz_50hz_one_cycle(self) -> None:
        assert compute_window_samples(5000.0, 50.0, 1) == 100

    def test_5khz_50hz_two_cycles(self) -> None:
        assert compute_window_samples(5000.0, 50.0, 2) == 200

    def test_3200hz_50hz(self) -> None:
        assert compute_window_samples(3200.0, 50.0, 1) == 64

    def test_6400hz_60hz(self) -> None:
        # 6400 / 60 = 106.67 → rounds to 107
        assert compute_window_samples(6400.0, 60.0, 1) == 107

    def test_zero_sample_rate_returns_one(self) -> None:
        assert compute_window_samples(0.0, 50.0, 1) == 1

    def test_zero_frequency_returns_one(self) -> None:
        assert compute_window_samples(5000.0, 0.0, 1) == 1

    def test_negative_cycles_returns_one(self) -> None:
        assert compute_window_samples(5000.0, 50.0, 0) == 1

    def test_result_always_at_least_one(self) -> None:
        assert compute_window_samples(0.001, 50.0, 1) >= 1

    def test_100hz_50hz_one_cycle(self) -> None:
        assert compute_window_samples(100.0, 50.0, 1) == 2

    def test_50hz_50hz_one_cycle(self) -> None:
        # Nyquist limit — 1 sample per half-cycle → rounds to 1
        assert compute_window_samples(50.0, 50.0, 1) == 1

    def test_explicit_half_cycle_mode_5khz_50hz(self) -> None:
        cfg = RMSConfig(window_mode=RMSWindowMode.HALF_CYCLE)
        assert compute_rms_window_samples(5000.0, cfg) == 50

    def test_explicit_one_cycle_mode_5khz_50hz(self) -> None:
        cfg = RMSConfig(window_mode=RMSWindowMode.ONE_CYCLE)
        assert compute_rms_window_samples(5000.0, cfg) == 100

    def test_explicit_two_cycle_mode_5khz_50hz(self) -> None:
        cfg = RMSConfig(window_mode=RMSWindowMode.TWO_CYCLE)
        assert compute_rms_window_samples(5000.0, cfg) == 200

    def test_custom_samples_mode_uses_operator_sample_count(self) -> None:
        cfg = RMSConfig(
            window_mode=RMSWindowMode.CUSTOM_SAMPLES,
            custom_window_samples=123,
        )
        assert compute_rms_window_samples(5000.0, cfg) == 123


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeRmsOverlay
# ─────────────────────────────────────────────────────────────────────────────


def _sine_wave(
    amplitude: float,
    frequency_hz: float,
    sample_rate_hz: float,
    n_cycles: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, values) for a pure sinewave."""
    n = int(n_cycles * sample_rate_hz / frequency_hz)
    t = np.linspace(0.0, n / sample_rate_hz, n, endpoint=False)
    v = amplitude * np.sin(2.0 * math.pi * frequency_hz * t)
    return t, v


class TestComputeRmsOverlayCorrectness:
    def test_rms_of_unity_sinewave_is_one_over_sqrt2(self) -> None:
        # V_rms = A / sqrt(2) for a pure sinewave
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        # Skip first few windows (edge transient) — after one full cycle is stable
        stable = rms_v[100:]
        expected = 1.0 / math.sqrt(2.0)
        np.testing.assert_allclose(stable, expected, rtol=1e-3)

    def test_325v_peak_50hz_one_cycle_is_230vrms_and_smooth(self) -> None:
        amplitude = 325.27
        t, v = _sine_wave(
            amplitude=amplitude,
            frequency_hz=50.0,
            sample_rate_hz=5000.0,
            n_cycles=8.0,
        )
        rms_t, rms_v = compute_rms_overlay(
            t,
            v,
            sample_rate_hz=5000.0,
            config=RMSConfig(window_mode=RMSWindowMode.ONE_CYCLE),
        )
        stable = rms_v[100:]
        expected = amplitude / math.sqrt(2.0)
        np.testing.assert_allclose(stable, expected, rtol=1e-3)
        assert float(np.ptp(stable)) < expected * 0.002
        assert rms_t[0] == pytest.approx(t[99])

    def test_rms_of_10kv_sinewave(self) -> None:
        amplitude = 10_000.0
        t, v = _sine_wave(amplitude=amplitude, frequency_hz=50.0, sample_rate_hz=5000.0)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        stable = rms_v[100:]
        expected = amplitude / math.sqrt(2.0)
        np.testing.assert_allclose(stable, expected, rtol=1e-3)

    def test_rms_different_samples_per_cycle(self) -> None:
        # 3200 Hz at 50 Hz → 64 samples/cycle
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=3200.0)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=3200.0)
        stable = rms_v[64:]
        np.testing.assert_allclose(stable, 1.0 / math.sqrt(2.0), rtol=2e-3)

    def test_rms_60hz_system(self) -> None:
        t, v = _sine_wave(
            amplitude=1.0, frequency_hz=60.0, sample_rate_hz=6000.0, n_cycles=6.0
        )
        rms_t, rms_v = compute_rms_overlay(
            t, v, sample_rate_hz=6000.0, nominal_frequency_hz=60.0
        )
        stable = rms_v[100:]
        np.testing.assert_allclose(stable, 1.0 / math.sqrt(2.0), rtol=2e-3)

    def test_60hz_one_cycle_mode_is_smooth(self) -> None:
        t, v = _sine_wave(
            amplitude=169.7,
            frequency_hz=60.0,
            sample_rate_hz=6000.0,
            n_cycles=8.0,
        )
        _, rms_v = compute_rms_overlay(
            t,
            v,
            sample_rate_hz=6000.0,
            config=RMSConfig(
                nominal_frequency_hz=60.0,
                window_mode=RMSWindowMode.ONE_CYCLE,
            ),
        )
        stable = rms_v[100:]
        expected = 169.7 / math.sqrt(2.0)
        np.testing.assert_allclose(stable, expected, rtol=1e-3)
        assert float(np.ptp(stable)) < expected * 0.002

    def test_half_and_two_cycle_modes_remain_engineering_envelopes(self) -> None:
        t, v = _sine_wave(
            amplitude=10.0,
            frequency_hz=50.0,
            sample_rate_hz=5000.0,
            n_cycles=8.0,
        )
        expected = 10.0 / math.sqrt(2.0)
        for mode, skip in [
            (RMSWindowMode.HALF_CYCLE, 50),
            (RMSWindowMode.TWO_CYCLE, 200),
        ]:
            _, rms_v = compute_rms_overlay(
                t,
                v,
                sample_rate_hz=5000.0,
                config=RMSConfig(window_mode=mode),
            )
            stable = rms_v[skip:]
            np.testing.assert_allclose(stable, expected, rtol=1e-3)
            assert float(np.ptp(stable)) < expected * 0.002

    def test_rms_dc_signal(self) -> None:
        # RMS of a DC signal equals the DC value
        t = np.linspace(0.0, 0.1, 500)
        v = np.full(500, 5.0)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        np.testing.assert_allclose(rms_v, 5.0, rtol=1e-10)


class TestComputeRmsOverlayTimeAlignment:
    def test_output_length_is_n_minus_window_plus_one(self) -> None:
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        window = 100  # 5000 / 50 * 1 = 100
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        assert len(rms_v) == len(v) - window + 1
        assert len(rms_t) == len(rms_v)

    def test_rms_time_is_right_aligned(self) -> None:
        # output[0] corresponds to time[window-1]
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        window = 100
        rms_t, _ = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        assert rms_t[0] == pytest.approx(t[window - 1])

    def test_rms_time_last_element_matches_raw_last(self) -> None:
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        rms_t, _ = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        assert rms_t[-1] == pytest.approx(t[-1])


class TestComputeRmsOverlayNanHandling:
    def test_nan_replaced_by_zero_energy(self) -> None:
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        # Inject NaN in a region far from the start
        v_with_nan = v.copy()
        v_with_nan[400:410] = np.nan
        # Should not raise
        rms_t, rms_v = compute_rms_overlay(t, v_with_nan, sample_rate_hz=5000.0)
        assert np.all(np.isfinite(rms_v)), "NaN should not propagate into RMS output"

    def test_all_nan_input_gives_zero_rms(self) -> None:
        t = np.linspace(0.0, 0.1, 500)
        v = np.full(500, np.nan)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        np.testing.assert_allclose(rms_v, 0.0)

    def test_inf_replaced_by_zero_energy(self) -> None:
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        v_with_inf = v.copy()
        v_with_inf[300] = np.inf
        rms_t, rms_v = compute_rms_overlay(t, v_with_inf, sample_rate_hz=5000.0)
        assert np.all(np.isfinite(rms_v))


class TestComputeRmsOverlayEdgeCases:
    def test_raises_when_window_exceeds_length(self) -> None:
        t = np.linspace(0.0, 0.001, 10)
        v = np.ones(10)
        # 5000 Hz at 50 Hz → window=100 > len=10 → ValueError
        with pytest.raises(ValueError):
            compute_rms_overlay(t, v, sample_rate_hz=5000.0)

    def test_single_sample_window_returns_abs_value(self) -> None:
        # Very low sample rate forces window=1; RMS of scalar x is |x|
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([-3.0, 4.0, -5.0])
        rms_t, rms_v = compute_rms_overlay(
            t, v, sample_rate_hz=1.0, nominal_frequency_hz=50.0
        )
        # window=1 → rms[i] = sqrt(v[i]²) = |v[i]|
        np.testing.assert_allclose(rms_v, np.abs(v))

    def test_returns_float64(self) -> None:
        t, v = _sine_wave(amplitude=1.0, frequency_hz=50.0, sample_rate_hz=5000.0)
        rms_t, rms_v = compute_rms_overlay(t, v, sample_rate_hz=5000.0)
        assert rms_t.dtype == np.float64
        assert rms_v.dtype == np.float64
