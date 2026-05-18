"""Unit tests for harmonic extraction engine — Phase 7.

Tests cover:
  - compute_harmonic_window_samples: cycle-aligned sizes at common sample rates
  - extract_harmonics: magnitude correctness on pure sinusoids
  - Harmonic injection detection: 5th and 7th harmonic
  - Output shape: (n_windows,) consistent with config
  - Right-aligned time convention
  - 50 Hz and 60 Hz nominal frequency support
  - NaN/Inf handling (zero-energy replacement)
  - Signal shorter than one window → empty result
  - Overlap controls window count
  - rectangular window still extracts correct magnitude
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.harmonics.harmonic_extraction import (
    compute_harmonic_window_samples,
    extract_harmonics,
)
from app.analytics.harmonics.harmonic_models import (
    HarmonicConfig,
    HarmonicWindowMode,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

FS = 5000.0   # sample rate Hz
F0 = 50.0     # nominal Hz


def _make_sine(
    amplitude: float,
    freq_hz: float,
    sample_rate: float = FS,
    n_cycles: float = 6.0,
    phase_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, signal) for a pure sinusoid."""
    n = int(round(n_cycles * sample_rate / freq_hz))
    t = np.arange(n) / sample_rate
    sig = amplitude * np.sin(2.0 * np.pi * freq_hz * t + np.radians(phase_deg))
    return t, sig


def _two_cycle_cfg(nominal_hz: float = F0, max_order: int = 13) -> HarmonicConfig:
    return HarmonicConfig(
        nominal_hz=nominal_hz,
        max_order=max_order,
        window_mode=HarmonicWindowMode.TWO_CYCLE,
        overlap=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestWindowSizeCalculation
# ─────────────────────────────────────────────────────────────────────────────


class TestWindowSizeCalculation:
    def test_two_cycle_5000hz_50hz(self) -> None:
        cfg = HarmonicConfig(nominal_hz=50.0, window_mode=HarmonicWindowMode.TWO_CYCLE)
        assert compute_harmonic_window_samples(5000.0, cfg) == 200

    def test_one_cycle_5000hz_50hz(self) -> None:
        cfg = HarmonicConfig(nominal_hz=50.0, window_mode=HarmonicWindowMode.ONE_CYCLE)
        assert compute_harmonic_window_samples(5000.0, cfg) == 100

    def test_four_cycle_5000hz_50hz(self) -> None:
        cfg = HarmonicConfig(nominal_hz=50.0, window_mode=HarmonicWindowMode.FOUR_CYCLE)
        assert compute_harmonic_window_samples(5000.0, cfg) == 400

    def test_two_cycle_60hz(self) -> None:
        cfg = HarmonicConfig(nominal_hz=60.0, window_mode=HarmonicWindowMode.TWO_CYCLE)
        # round(2 * 5000/60) = round(166.67) = 167
        assert compute_harmonic_window_samples(5000.0, cfg) == 167

    def test_two_cycle_6000hz_60hz(self) -> None:
        cfg = HarmonicConfig(nominal_hz=60.0, window_mode=HarmonicWindowMode.TWO_CYCLE)
        # 2 * round(6000/60) = 2 * 100 = 200
        assert compute_harmonic_window_samples(6000.0, cfg) == 200

    def test_zero_sample_rate_returns_one(self) -> None:
        assert compute_harmonic_window_samples(0.0) == 1

    def test_negative_sample_rate_returns_one(self) -> None:
        assert compute_harmonic_window_samples(-100.0) == 1

    def test_default_config(self) -> None:
        # Default: TWO_CYCLE, 50 Hz, 5000 Hz → 200
        assert compute_harmonic_window_samples(5000.0) == 200


# ─────────────────────────────────────────────────────────────────────────────
# TestFundamentalExtraction
# ─────────────────────────────────────────────────────────────────────────────


class TestFundamentalExtraction:
    def test_pure_sine_h1_magnitude_rms(self) -> None:
        """Pure 50 Hz sine at amplitude A → H1 ≈ A/sqrt(2) in every window."""
        A = 100.0
        cfg = _two_cycle_cfg()
        _, sig = _make_sine(A, F0)
        result = extract_harmonics(sig, FS, cfg)

        assert result.n_windows > 0
        h1 = result.magnitudes[1]
        expected_rms = A / np.sqrt(2.0)
        # Allow 2% tolerance for window-edge effects
        assert np.all(np.abs(h1 - expected_rms) / expected_rms < 0.02)

    def test_different_amplitude_scales_linearly(self) -> None:
        """Doubling amplitude doubles extracted H1."""
        cfg = _two_cycle_cfg()
        _, sig_1 = _make_sine(1.0, F0)
        _, sig_2 = _make_sine(2.0, F0)
        r1 = extract_harmonics(sig_1, FS, cfg)
        r2 = extract_harmonics(sig_2, FS, cfg)
        ratio = r2.magnitudes[1] / (r1.magnitudes[1] + 1e-12)
        assert np.allclose(ratio, 2.0, rtol=0.02)

    def test_60hz_nominal(self) -> None:
        """Extraction works correctly at 60 Hz nominal frequency."""
        A = 200.0
        cfg = HarmonicConfig(
            nominal_hz=60.0,
            max_order=7,
            window_mode=HarmonicWindowMode.TWO_CYCLE,
            overlap=0.5,
        )
        _, sig = _make_sine(A, 60.0, sample_rate=6000.0)
        result = extract_harmonics(sig, 6000.0, cfg)

        assert result.n_windows > 0
        h1 = result.magnitudes[1]
        expected_rms = A / np.sqrt(2.0)
        assert np.all(np.abs(h1 - expected_rms) / expected_rms < 0.02)

    def test_phase_does_not_affect_magnitude(self) -> None:
        """RMS magnitude is phase-independent."""
        cfg = _two_cycle_cfg()
        _, sig_0 = _make_sine(100.0, F0, phase_deg=0.0)
        _, sig_90 = _make_sine(100.0, F0, phase_deg=90.0)
        r0 = extract_harmonics(sig_0, FS, cfg)
        r90 = extract_harmonics(sig_90, FS, cfg)
        assert np.allclose(r0.magnitudes[1], r90.magnitudes[1], rtol=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TestHarmonicInjectionDetection
# ─────────────────────────────────────────────────────────────────────────────


class TestHarmonicInjectionDetection:
    def test_5th_harmonic_detected(self) -> None:
        """Injected 5th harmonic (250 Hz) is detected above noise floor."""
        cfg = _two_cycle_cfg()
        _, fund = _make_sine(100.0, F0)
        _, h5 = _make_sine(5.0, 5 * F0, n_cycles=6.0 * 5)
        # Align lengths
        n = min(len(fund), len(h5))
        sig = fund[:n] + h5[:n]
        result = extract_harmonics(sig, FS, cfg)

        h1 = result.magnitudes[1].mean()
        h5_mag = result.magnitudes[5].mean()
        # H5 should be present and significantly smaller than H1
        assert h5_mag > 0.1 * (5.0 / np.sqrt(2.0))
        assert h1 > h5_mag

    def test_7th_harmonic_detected(self) -> None:
        """Injected 7th harmonic (350 Hz) is detected above noise floor."""
        cfg = _two_cycle_cfg()
        _, fund = _make_sine(100.0, F0)
        _, h7 = _make_sine(3.0, 7 * F0, n_cycles=6.0 * 7)
        n = min(len(fund), len(h7))
        sig = fund[:n] + h7[:n]
        result = extract_harmonics(sig, FS, cfg)

        h7_mag = result.magnitudes[7].mean()
        assert h7_mag > 0.1 * (3.0 / np.sqrt(2.0))

    def test_clean_sine_higher_harmonics_near_zero(self) -> None:
        """Pure fundamental → H3, H5, H7 near zero (spectral leakage tolerance)."""
        cfg = _two_cycle_cfg()
        _, sig = _make_sine(100.0, F0)
        result = extract_harmonics(sig, FS, cfg)

        h1_mean = result.magnitudes[1].mean()
        for order in (3, 5, 7):
            hn_mean = result.magnitudes[order].mean()
            # Higher harmonics must be < 2% of fundamental
            assert hn_mean < 0.02 * h1_mean, (
                f"H{order} = {hn_mean:.4f} is not near-zero (H1 = {h1_mean:.4f})"
            )

    def test_max_order_limits_output(self) -> None:
        """Only orders 1..max_order are present in result."""
        cfg = HarmonicConfig(nominal_hz=F0, max_order=7, window_mode=HarmonicWindowMode.TWO_CYCLE)
        _, sig = _make_sine(100.0, F0)
        result = extract_harmonics(sig, FS, cfg)

        assert set(result.orders) == set(range(1, 8))
        assert 8 not in result.magnitudes
        assert 13 not in result.magnitudes


# ─────────────────────────────────────────────────────────────────────────────
# TestOutputShape
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputShape:
    def test_n_windows_formula(self) -> None:
        """n_windows == (N - W) // H + 1."""
        cfg = HarmonicConfig(
            nominal_hz=F0,
            max_order=5,
            window_mode=HarmonicWindowMode.TWO_CYCLE,
            overlap=0.5,
        )
        W = compute_harmonic_window_samples(FS, cfg)
        H = max(1, int(round(W * 0.5)))
        N = 2000
        expected_windows = (N - W) // H + 1

        sig = np.random.default_rng(0).standard_normal(N)
        result = extract_harmonics(sig, FS, cfg)

        assert result.n_windows == expected_windows
        assert len(result.harmonic_time) == expected_windows
        for arr in result.magnitudes.values():
            assert len(arr) == expected_windows

    def test_all_magnitude_arrays_same_length(self) -> None:
        """All per-order magnitude arrays have the same length."""
        cfg = _two_cycle_cfg(max_order=13)
        _, sig = _make_sine(100.0, F0)
        result = extract_harmonics(sig, FS, cfg)

        lengths = {len(arr) for arr in result.magnitudes.values()}
        assert len(lengths) == 1  # all identical

    def test_time_axis_right_aligned(self) -> None:
        """harmonic_time[0] >= window_duration (right-aligned convention)."""
        cfg = _two_cycle_cfg()
        _, sig = _make_sine(100.0, F0)
        t = np.arange(len(sig)) / FS
        result = extract_harmonics(sig, FS, cfg, time=t)

        W = compute_harmonic_window_samples(FS, cfg)
        min_first_time = (W - 1) / FS
        assert result.harmonic_time[0] >= min_first_time - 1e-9

    def test_signal_shorter_than_window_returns_empty(self) -> None:
        """Signal shorter than one window → HarmonicResult.empty()."""
        cfg = _two_cycle_cfg()
        short_sig = np.ones(10)
        result = extract_harmonics(short_sig, FS, cfg)

        assert result.n_windows == 0
        assert len(result.harmonic_time) == 0
        assert result.magnitudes == {}

    def test_no_overlap_produces_fewer_windows(self) -> None:
        """overlap=0.0 produces fewer windows than overlap=0.5."""
        n = 5000
        sig = np.random.default_rng(1).standard_normal(n)
        cfg_half = HarmonicConfig(nominal_hz=F0, max_order=5,
                                   window_mode=HarmonicWindowMode.TWO_CYCLE, overlap=0.5)
        cfg_none = HarmonicConfig(nominal_hz=F0, max_order=5,
                                   window_mode=HarmonicWindowMode.TWO_CYCLE, overlap=0.0)
        r_half = extract_harmonics(sig, FS, cfg_half)
        r_none = extract_harmonics(sig, FS, cfg_none)

        assert r_none.n_windows < r_half.n_windows


# ─────────────────────────────────────────────────────────────────────────────
# TestRobustness
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_nan_samples_replaced_by_zero(self) -> None:
        """NaN samples do not propagate; result has finite values."""
        cfg = _two_cycle_cfg()
        _, sig = _make_sine(100.0, F0)
        sig = sig.copy()
        sig[100:110] = np.nan
        result = extract_harmonics(sig, FS, cfg)

        assert result.n_windows > 0
        for arr in result.magnitudes.values():
            assert np.all(np.isfinite(arr))

    def test_inf_samples_replaced_by_zero(self) -> None:
        """Inf samples do not propagate."""
        cfg = _two_cycle_cfg()
        _, sig = _make_sine(100.0, F0)
        sig = sig.copy()
        sig[50] = np.inf
        result = extract_harmonics(sig, FS, cfg)

        for arr in result.magnitudes.values():
            assert np.all(np.isfinite(arr))

    def test_rectangular_window_extracts_correct_rms(self) -> None:
        """Rectangular window also gives correct RMS (within 2%)."""
        A = 100.0
        cfg = HarmonicConfig(
            nominal_hz=F0,
            max_order=5,
            window_mode=HarmonicWindowMode.TWO_CYCLE,
            window_function="rectangular",
            overlap=0.0,
        )
        _, sig = _make_sine(A, F0)
        result = extract_harmonics(sig, FS, cfg)

        h1 = result.magnitudes[1]
        expected_rms = A / np.sqrt(2.0)
        assert np.all(np.abs(h1 - expected_rms) / expected_rms < 0.02)

    def test_result_metadata_matches_config(self) -> None:
        """HarmonicResult carries correct metadata from config."""
        cfg = HarmonicConfig(
            nominal_hz=60.0,
            max_order=11,
            window_mode=HarmonicWindowMode.FOUR_CYCLE,
            overlap=0.25,
        )
        _, sig = _make_sine(50.0, 60.0, sample_rate=6000.0, n_cycles=10.0)
        result = extract_harmonics(sig, 6000.0, cfg)

        assert result.nominal_hz == 60.0
        assert result.sample_rate_hz == 6000.0
        W = compute_harmonic_window_samples(6000.0, cfg)
        assert result.window_samples == W
