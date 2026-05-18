"""Unit tests for symmetrical component computation (Phase 6A).

Tests cover:
  - Balanced three-phase: V1 ≈ Vnom, V2 ≈ 0, V0 ≈ 0
  - Single-line-to-ground fault: elevated V0 and V2
  - Phase reversal (ACB sequence): elevated V2, V1 ≈ Vnom, V0 ≈ 0
  - Pure zero sequence (all three phases equal): V0 ≈ V, V1 ≈ 0, V2 ≈ 0
  - compute_sequence_from_phasor_arrays end-to-end helper
  - sequence_magnitudes returns float64 arrays
  - unbalance_factor: balanced → 0%, unbalanced → elevated
  - Length preservation: output == input length
  - Zero-division safety in unbalance_factor
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.phasors.symmetrical_components import (
    compute_sequence_components,
    compute_sequence_from_phasor_arrays,
    sequence_magnitudes,
    unbalance_factor,
)
from app.analytics.phasors.phasor_extraction import extract_phasor
from app.analytics.phasors.phasor_models import PhasorConfig, PhasorWindowMode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SQRT2 = np.sqrt(2.0)
_A = np.exp(1j * 2.0 * np.pi / 3.0)


def _balanced_phasors(v_rms: float = 1.0) -> tuple[complex, complex, complex]:
    """Return balanced three-phase peak phasors (positive-sequence ABC)."""
    vpk = v_rms * _SQRT2
    va = vpk * np.exp(1j * 0.0)
    vb = vpk * np.exp(-1j * 2.0 * np.pi / 3.0)
    vc = vpk * np.exp(-1j * 4.0 * np.pi / 3.0)
    return va, vb, vc


def _make_3phase_sine(
    v_rms: float,
    angle_a_deg: float,
    angle_b_deg: float,
    angle_c_deg: float,
    sample_rate: float = 5000.0,
    n_cycles: float = 6.0,
    nominal_hz: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build three-phase sinusoids and return (time, va, vb, vc)."""
    n = int(round(n_cycles * sample_rate / nominal_hz))
    t = np.arange(n) / sample_rate
    vp = v_rms * _SQRT2
    va = vp * np.sin(2 * np.pi * nominal_hz * t + np.radians(angle_a_deg))
    vb = vp * np.sin(2 * np.pi * nominal_hz * t + np.radians(angle_b_deg))
    vc = vp * np.sin(2 * np.pi * nominal_hz * t + np.radians(angle_c_deg))
    return t, va, vb, vc


# ─────────────────────────────────────────────────────────────────────────────
# TestBalancedThreePhase
# ─────────────────────────────────────────────────────────────────────────────


class TestBalancedThreePhase:
    def test_v1_equals_vnom_for_balanced(self) -> None:
        v_rms = 100.0
        va, vb, vc = _balanced_phasors(v_rms)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        mag_v1, mag_v2, mag_v0 = sequence_magnitudes(v1, v2, v0)
        # Peak magnitudes: |V1_peak| = v_rms * sqrt(2); after /3: still proportional
        # The transform preserves: |V1| = v_rms * sqrt(2) for balanced input
        assert abs(mag_v1[0] - v_rms * _SQRT2) < 0.001

    def test_v2_near_zero_for_balanced(self) -> None:
        va, vb, vc = _balanced_phasors(1.0)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        _, mag_v2, _ = sequence_magnitudes(v1, v2, v0)
        assert mag_v2[0] < 1e-10

    def test_v0_near_zero_for_balanced(self) -> None:
        va, vb, vc = _balanced_phasors(1.0)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        _, _, mag_v0 = sequence_magnitudes(v1, v2, v0)
        assert mag_v0[0] < 1e-10

    def test_balanced_via_phasor_extraction(self) -> None:
        """End-to-end: balanced waveforms → V1 dominant, V2 ≈ 0, V0 ≈ 0."""
        v_rms = 230.0
        t, va, vb, vc = _make_3phase_sine(v_rms, 0.0, -120.0, -240.0)
        cfg = PhasorConfig(nominal_hz=50.0, window_mode=PhasorWindowMode.ONE_CYCLE)
        sr = 5000.0

        ph_a = extract_phasor(t, va, sr, cfg)
        ph_b = extract_phasor(t, vb, sr, cfg)
        ph_c = extract_phasor(t, vc, sr, cfg)

        seq = compute_sequence_from_phasor_arrays(ph_a, ph_b, ph_c)
        skip = 200
        mag_v1 = seq["mag_v1"][skip:]
        mag_v2 = seq["mag_v2"][skip:]
        mag_v0 = seq["mag_v0"][skip:]

        assert np.allclose(mag_v1, v_rms, rtol=5e-3), (
            f"V1 expected {v_rms:.1f}, got {mag_v1.mean():.3f}"
        )
        assert np.allclose(mag_v2, 0.0, atol=0.5), (
            f"V2 expected ≈0, got {mag_v2.mean():.3f}"
        )
        assert np.allclose(mag_v0, 0.0, atol=0.5), (
            f"V0 expected ≈0, got {mag_v0.mean():.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestSLGFaultCondition
# ─────────────────────────────────────────────────────────────────────────────


class TestSLGFaultCondition:
    def test_slg_fault_elevates_v0_and_v2(self) -> None:
        """SLG fault: Va depressed to 50%, Vb and Vc remain nominal."""
        v_nom = 1.0 * _SQRT2
        va = complex(v_nom * 0.5, 0.0)   # depressed phase A
        vb = v_nom * np.exp(-1j * 2 * np.pi / 3)
        vc = v_nom * np.exp(-1j * 4 * np.pi / 3)

        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        mag_v1, mag_v2, mag_v0 = sequence_magnitudes(v1, v2, v0)

        # V0 and V2 must be non-zero
        assert mag_v0[0] > 0.05, f"Expected elevated V0, got {mag_v0[0]:.4f}"
        assert mag_v2[0] > 0.05, f"Expected elevated V2, got {mag_v2[0]:.4f}"
        # V1 should still be close to nominal but reduced
        assert mag_v1[0] > 0.5, f"V1 dropped too low: {mag_v1[0]:.4f}"

    def test_slg_fault_via_waveforms(self) -> None:
        """Waveform-level SLG: Va amplitude halved."""
        v_rms_nom = 100.0
        v_rms_fault = 50.0  # Va depressed
        t, va_n, vb, vc = _make_3phase_sine(v_rms_nom, 0.0, -120.0, -240.0)
        # Replace Va with a depressed version
        n = int(round(6.0 * 5000.0 / 50.0))
        va = v_rms_fault * _SQRT2 * np.sin(2 * np.pi * 50.0 * np.arange(n) / 5000.0)

        cfg = PhasorConfig()
        sr = 5000.0
        ph_a = extract_phasor(t, va, sr, cfg)
        ph_b = extract_phasor(t, vb, sr, cfg)
        ph_c = extract_phasor(t, vc, sr, cfg)

        seq = compute_sequence_from_phasor_arrays(ph_a, ph_b, ph_c)
        skip = 200
        # V0 and V2 should be significantly elevated vs balanced case
        assert seq["mag_v0"][skip:].mean() > 10.0
        assert seq["mag_v2"][skip:].mean() > 10.0


# ─────────────────────────────────────────────────────────────────────────────
# TestPhaseReversal
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseReversal:
    def test_phase_reversal_acb_elevates_v2(self) -> None:
        """ACB (negative) sequence: V2 dominant, V1 ≈ 0."""
        v_nom = 1.0 * _SQRT2
        va = v_nom * np.exp(1j * 0.0)
        # ACB: C leads A by 120°, B lags A by 120°
        vb = v_nom * np.exp(1j * 2 * np.pi / 3)   # +120°
        vc = v_nom * np.exp(-1j * 2 * np.pi / 3)  # -120°

        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        mag_v1, mag_v2, mag_v0 = sequence_magnitudes(v1, v2, v0)

        assert mag_v2[0] > 0.99 * v_nom, (
            f"Expected V2 ≈ {v_nom:.3f}, got {mag_v2[0]:.3f}"
        )
        assert mag_v1[0] < 0.01, f"Expected V1 ≈ 0, got {mag_v1[0]:.3f}"
        assert mag_v0[0] < 0.01, f"Expected V0 ≈ 0, got {mag_v0[0]:.3f}"

    def test_phase_reversal_via_waveforms(self) -> None:
        """Reversed phase waveforms (ACB) → V2 dominant after phasor extraction."""
        v_rms = 100.0
        # ACB: B leads A, C lags A
        t, va, vb, vc = _make_3phase_sine(v_rms, 0.0, 120.0, -120.0)
        cfg = PhasorConfig()
        sr = 5000.0
        ph_a = extract_phasor(t, va, sr, cfg)
        ph_b = extract_phasor(t, vb, sr, cfg)
        ph_c = extract_phasor(t, vc, sr, cfg)
        seq = compute_sequence_from_phasor_arrays(ph_a, ph_b, ph_c)
        skip = 200
        assert seq["mag_v2"][skip:].mean() > 80.0
        assert seq["mag_v1"][skip:].mean() < 5.0


# ─────────────────────────────────────────────────────────────────────────────
# TestPureZeroSequence
# ─────────────────────────────────────────────────────────────────────────────


class TestPureZeroSequence:
    def test_in_phase_gives_pure_v0(self) -> None:
        """All three phases equal and in-phase → pure V0."""
        v = complex(1.0, 0.0)
        v1, v2, v0 = compute_sequence_components(
            np.array([v]), np.array([v]), np.array([v])
        )
        mag_v1, mag_v2, mag_v0 = sequence_magnitudes(v1, v2, v0)
        assert abs(mag_v0[0] - 1.0) < 1e-10
        assert mag_v1[0] < 1e-10
        assert mag_v2[0] < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# TestSequenceComponentHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestSequenceComponentHelpers:
    def test_sequence_magnitudes_returns_float64(self) -> None:
        va, vb, vc = _balanced_phasors(1.0)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        m1, m2, m0 = sequence_magnitudes(v1, v2, v0)
        assert m1.dtype == np.float64
        assert m2.dtype == np.float64
        assert m0.dtype == np.float64

    def test_output_length_preserved(self) -> None:
        n = 500
        va = np.ones(n, dtype=complex)
        vb = va * np.exp(-1j * 2 * np.pi / 3)
        vc = va * np.exp(-1j * 4 * np.pi / 3)
        v1, v2, v0 = compute_sequence_components(va, vb, vc)
        assert len(v1) == n
        assert len(v2) == n
        assert len(v0) == n

    def test_unbalance_factor_zero_for_balanced(self) -> None:
        va, vb, vc = _balanced_phasors(1.0)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        uf = unbalance_factor(v1, v2)
        assert uf[0] < 1e-8

    def test_unbalance_factor_elevated_for_unbalanced(self) -> None:
        # SLG: Va = 0 (faulted), Vb, Vc nominal
        v_nom = 1.0 * _SQRT2
        va = complex(0.0, 0.0)
        vb = v_nom * np.exp(-1j * 2 * np.pi / 3)
        vc = v_nom * np.exp(-1j * 4 * np.pi / 3)
        v1, v2, v0 = compute_sequence_components(
            np.array([va]), np.array([vb]), np.array([vc])
        )
        uf = unbalance_factor(v1, v2)
        assert uf[0] > 20.0  # significantly unbalanced

    def test_unbalance_factor_zero_v1_returns_zero(self) -> None:
        v1 = np.array([0.0 + 0j])
        v2 = np.array([1.0 + 0j])
        uf = unbalance_factor(v1, v2)
        assert uf[0] == 0.0

    def test_mismatched_phasor_lengths_raises(self) -> None:
        ph_a = (
            np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0 + 0j])
        )
        ph_b = (
            np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0 + 0j])
        )
        ph_c_short = (
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0 + 0j, 1.0 + 0j]),
        )
        with pytest.raises(ValueError, match="equal length"):
            compute_sequence_from_phasor_arrays(ph_a, ph_b, ph_c_short)
