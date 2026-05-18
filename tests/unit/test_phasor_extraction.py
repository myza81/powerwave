"""Unit tests for phasor extraction (Phase 6A).

Tests cover:
  - Magnitude correctness: pure sine at known amplitude → known RMS magnitude
  - Angle correctness: sine at known phase offset → known angle
  - Window-size calculation for half/one/two-cycle modes
  - Output array length matches N - window + 1
  - Right-aligned time convention
  - NaN/Inf sample handling (zero-energy contribution)
  - Degenerate input detection (too short, non-1D)
  - 50 Hz and 60 Hz nominal frequency support
  - Multiple sample rates
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.phasors.phasor_extraction import (
    compute_phasor_window_samples,
    extract_phasor,
)
from app.analytics.phasors.phasor_models import PhasorConfig, PhasorWindowMode


# ─────────────────────────────────────────────────────────────────────────────
# TestWindowSizeCalculation
# ─────────────────────────────────────────────────────────────────────────────


class TestWindowSizeCalculation:
    def test_one_cycle_5000hz_50hz(self) -> None:
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        assert compute_phasor_window_samples(5000.0, cfg) == 100

    def test_half_cycle_5000hz_50hz(self) -> None:
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.HALF_CYCLE)
        assert compute_phasor_window_samples(5000.0, cfg) == 50

    def test_two_cycle_5000hz_50hz(self) -> None:
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.TWO_CYCLE)
        assert compute_phasor_window_samples(5000.0, cfg) == 200

    def test_one_cycle_60hz(self) -> None:
        cfg = PhasorConfig(nominal_hz=60.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        # 5000 / 60 = 83.33 → rounds to 83
        assert compute_phasor_window_samples(5000.0, cfg) == 83

    def test_one_cycle_3200hz_50hz(self) -> None:
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        assert compute_phasor_window_samples(3200.0, cfg) == 64

    def test_default_config(self) -> None:
        # Default: ONE_CYCLE, 50 Hz
        assert compute_phasor_window_samples(5000.0) == 100

    def test_zero_sample_rate_returns_one(self) -> None:
        assert compute_phasor_window_samples(0.0) == 1

    def test_negative_sample_rate_returns_one(self) -> None:
        assert compute_phasor_window_samples(-100.0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorExtractionMagnitude
# ─────────────────────────────────────────────────────────────────────────────


def _make_sine(
    amplitude: float,
    freq_hz: float,
    phase_deg: float,
    sample_rate: float,
    n_cycles: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pure sinusoid: values = amplitude * sin(2πft + φ)."""
    n = int(round(n_cycles * sample_rate / freq_hz))
    t = np.arange(n) / sample_rate
    values = amplitude * np.sin(2.0 * np.pi * freq_hz * t + np.radians(phase_deg))
    return t, values


class TestPhasorExtractionMagnitude:
    def test_magnitude_230v_rms(self) -> None:
        """230 V RMS → extract_phasor should give magnitude ≈ 230 V."""
        v_rms = 230.0
        v_peak = v_rms * np.sqrt(2.0)
        t, vals = _make_sine(v_peak, 50.0, 0.0, 5000.0, n_cycles=6.0)
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        _, mag, _, _ = extract_phasor(t, vals, 5000.0, cfg)
        # After first transient cycles, magnitude should stabilise at v_rms
        steady = mag[100:]  # skip first 100 output samples
        assert np.allclose(steady, v_rms, rtol=1e-3), (
            f"Expected {v_rms:.1f} V RMS, got {steady.mean():.3f}"
        )

    def test_magnitude_zero_signal(self) -> None:
        t, vals = _make_sine(0.0, 50.0, 0.0, 5000.0)
        _, mag, _, _ = extract_phasor(t, vals, 5000.0)
        assert np.allclose(mag, 0.0, atol=1e-10)

    def test_magnitude_scales_linearly(self) -> None:
        t1, v1 = _make_sine(100.0, 50.0, 0.0, 5000.0)
        t2, v2 = _make_sine(200.0, 50.0, 0.0, 5000.0)
        cfg = PhasorConfig()
        _, mag1, _, _ = extract_phasor(t1, v1, 5000.0, cfg)
        _, mag2, _, _ = extract_phasor(t2, v2, 5000.0, cfg)
        ratio = mag2.mean() / mag1.mean()
        assert abs(ratio - 2.0) < 0.01

    def test_magnitude_60hz_nominal(self) -> None:
        v_rms = 115.0
        v_peak = v_rms * np.sqrt(2.0)
        t, vals = _make_sine(v_peak, 60.0, 0.0, 6000.0, n_cycles=6.0)
        cfg = PhasorConfig(nominal_hz=60.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        _, mag, _, _ = extract_phasor(t, vals, 6000.0, cfg)
        steady = mag[100:]
        assert np.allclose(steady, v_rms, rtol=1e-3)

    def test_magnitude_half_cycle_window(self) -> None:
        v_rms = 100.0
        v_peak = v_rms * np.sqrt(2.0)
        t, vals = _make_sine(v_peak, 50.0, 0.0, 5000.0, n_cycles=6.0)
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.HALF_CYCLE)
        _, mag, _, _ = extract_phasor(t, vals, 5000.0, cfg)
        steady = mag[50:]
        assert np.allclose(steady, v_rms, rtol=2e-3)


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorExtractionAngle
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorExtractionAngle:
    def test_angle_rotates_at_carrier_frequency(self) -> None:
        """DFT phasor angle for sin(ωt) rotates at 360°/cycle = 3.6°/sample at 5 kHz/50 Hz.

        This is the physically correct behavior: the phasor angle is not
        constant — it rotates with the carrier frequency.  Steady phase
        information is read from the *relative* angle between two channels.
        """
        t, vals = _make_sine(100.0, 50.0, 0.0, 5000.0, n_cycles=6.0)
        _, _, angle, _ = extract_phasor(t, vals, 5000.0)
        # Angle should advance by 3.6° per sample (360° / 100 samples/cycle)
        # Check: unwrapped angle progression is linear
        unwrapped = np.unwrap(np.radians(angle))
        rate_rad_per_sample = np.diff(unwrapped).mean()
        expected_rad = 2.0 * np.pi * 50.0 / 5000.0  # 2π f / fs
        assert abs(rate_rad_per_sample - expected_rad) < 1e-6, (
            f"Angle rate {np.degrees(rate_rad_per_sample):.4f}°/sample, "
            f"expected {np.degrees(expected_rad):.4f}°/sample"
        )

    def test_angle_90_degree_offset(self) -> None:
        """Two signals with 90° offset should have angles differing by ~90°."""
        t, v0 = _make_sine(100.0, 50.0, 0.0, 5000.0, n_cycles=8.0)
        _, v90 = _make_sine(100.0, 50.0, 90.0, 5000.0, n_cycles=8.0)
        cfg = PhasorConfig()
        _, _, ang0, _ = extract_phasor(t, v0, 5000.0, cfg)
        _, _, ang90, _ = extract_phasor(t, v90, 5000.0, cfg)
        # After settling, angle difference should be ≈ 90°
        diff = ang90[100:] - ang0[100:]
        # Normalize to [-180, 180]
        diff = (diff + 180) % 360 - 180
        assert np.allclose(diff, 90.0, atol=2.0), (
            f"Expected 90° offset, got mean diff {diff.mean():.2f}°"
        )

    def test_angle_120_degree_offset_abc(self) -> None:
        """Three-phase ABC channels should have angles ≈ -120° apart."""
        t, va = _make_sine(100.0, 50.0, 0.0,   5000.0, n_cycles=8.0)
        _, vb = _make_sine(100.0, 50.0, -120.0, 5000.0, n_cycles=8.0)
        _, vc = _make_sine(100.0, 50.0, -240.0, 5000.0, n_cycles=8.0)
        cfg = PhasorConfig()
        _, _, ang_a, _ = extract_phasor(t, va, 5000.0, cfg)
        _, _, ang_b, _ = extract_phasor(t, vb, 5000.0, cfg)
        _, _, ang_c, _ = extract_phasor(t, vc, 5000.0, cfg)
        skip = 150
        diff_ab = ((ang_b[skip:] - ang_a[skip:] + 180) % 360 - 180)
        diff_ac = ((ang_c[skip:] - ang_a[skip:] + 180) % 360 - 180)
        assert np.allclose(diff_ab, -120.0, atol=2.0), (
            f"A-B angle diff: expected -120°, got {diff_ab.mean():.2f}°"
        )
        assert np.allclose(diff_ac, -240.0 % 360 - 180, atol=2.0) or \
               np.allclose(diff_ac, 120.0, atol=2.0), (
            f"A-C angle diff: expected ±120°, got {diff_ac.mean():.2f}°"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorOutputShape
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorOutputShape:
    def test_output_length_one_cycle(self) -> None:
        n = 500
        t = np.linspace(0, 1.0, n)
        vals = np.sin(2 * np.pi * 50 * t)
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        window = compute_phasor_window_samples(500.0, cfg)  # 10 samples
        pt, mag, ang, cpx = extract_phasor(t, vals, 500.0, cfg)
        expected_len = n - window + 1
        assert len(pt) == expected_len
        assert len(mag) == expected_len
        assert len(ang) == expected_len
        assert len(cpx) == expected_len

    def test_right_aligned_time(self) -> None:
        n = 200
        t = np.linspace(0.0, 1.0, n)
        vals = np.sin(2 * np.pi * 50 * t)
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        window = compute_phasor_window_samples(200.0, cfg)
        pt, _, _, _ = extract_phasor(t, vals, 200.0, cfg)
        # Right-aligned: first output time should be time[window-1]
        assert abs(pt[0] - t[window - 1]) < 1e-12

    def test_complex_phasor_dtype(self) -> None:
        t = np.linspace(0, 0.1, 500)
        vals = np.sin(2 * np.pi * 50 * t)
        _, _, _, cpx = extract_phasor(t, vals, 5000.0)
        assert cpx.dtype == np.complex128


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorNaNHandling
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorNaNHandling:
    def test_nan_samples_do_not_propagate(self) -> None:
        t = np.linspace(0, 0.2, 1000)
        vals = np.sin(2 * np.pi * 50 * t)
        vals[50] = np.nan
        vals[51] = np.nan
        _, mag, _, _ = extract_phasor(t, vals, 5000.0)
        # Should not produce NaN in output
        assert not np.any(np.isnan(mag))

    def test_inf_samples_do_not_propagate(self) -> None:
        t = np.linspace(0, 0.2, 1000)
        vals = np.sin(2 * np.pi * 50 * t)
        vals[100] = np.inf
        _, mag, _, _ = extract_phasor(t, vals, 5000.0)
        assert not np.any(np.isnan(mag))
        assert not np.any(np.isinf(mag))


# ─────────────────────────────────────────────────────────────────────────────
# TestPhasorDegenerateInputs
# ─────────────────────────────────────────────────────────────────────────────


class TestPhasorDegenerateInputs:
    def test_signal_shorter_than_window_raises(self) -> None:
        t = np.array([0.0, 0.01])
        vals = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shorter than phasor window"):
            extract_phasor(t, vals, 5000.0)

    def test_2d_input_raises(self) -> None:
        t = np.zeros((10, 2))
        vals = np.zeros((10, 2))
        with pytest.raises(ValueError):
            extract_phasor(t, vals, 5000.0)

    def test_mismatched_lengths_raises(self) -> None:
        t = np.linspace(0, 1, 200)
        vals = np.linspace(0, 1, 100)
        with pytest.raises(ValueError):
            extract_phasor(t, vals, 5000.0)
