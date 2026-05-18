"""Unit tests for harmonic distortion metrics — Phase 7.

Tests cover:
  - compute_thd: scalar THD from {order: magnitude} dict
  - compute_thd_array: vectorized time-varying THD
  - compute_thd_from_result: convenience wrapper
  - individual_harmonic_distortion: H_n / H_1 ratio
  - Divide-by-zero safety (near-zero fundamental)
  - Known mixture produces analytically expected THD
  - Output shape consistency
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.harmonics.harmonic_metrics import (
    compute_thd,
    compute_thd_array,
    compute_thd_from_result,
    individual_harmonic_distortion,
)
from app.analytics.harmonics.harmonic_models import HarmonicConfig, HarmonicResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _scalar_mags(**kwargs: float) -> dict[int, float]:
    """Build a {order: magnitude} dict from keyword args like h1=100, h3=5."""
    return {int(k[1:]): v for k, v in kwargs.items()}


def _array_mags(n: int, **kwargs: float) -> dict[int, np.ndarray]:
    """Build constant-value magnitude arrays of length n."""
    return {int(k[1:]): np.full(n, v, dtype=np.float64) for k, v in kwargs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeTHD (scalar)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTHD:
    def test_pure_fundamental_thd_zero(self) -> None:
        """No higher harmonics → THD = 0.0."""
        mags = _scalar_mags(h1=100.0)
        assert compute_thd(mags) == pytest.approx(0.0, abs=1e-10)

    def test_known_mixture_thd(self) -> None:
        """H1=100, H3=5, H5=3 → THD = sqrt(25+9)/100 = sqrt(34)/100."""
        mags = _scalar_mags(h1=100.0, h3=5.0, h5=3.0)
        expected = np.sqrt(5.0**2 + 3.0**2) / 100.0
        assert compute_thd(mags) == pytest.approx(expected, rel=1e-6)

    def test_single_harmonic_only(self) -> None:
        """H1=100, H3=10 → THD = 0.10."""
        mags = _scalar_mags(h1=100.0, h3=10.0)
        assert compute_thd(mags) == pytest.approx(0.10, rel=1e-6)

    def test_zero_fundamental_returns_zero(self) -> None:
        """Fundamental = 0 → THD = 0.0 (no divide-by-zero)."""
        mags = _scalar_mags(h1=0.0, h3=5.0)
        assert compute_thd(mags) == 0.0

    def test_below_threshold_fundamental_returns_zero(self) -> None:
        """Fundamental below safe_threshold → THD = 0.0."""
        mags = _scalar_mags(h1=1e-12, h3=5.0)
        assert compute_thd(mags) == 0.0

    def test_max_order_limits_sum(self) -> None:
        """max_order=3 excludes H5 from the distortion sum."""
        mags = _scalar_mags(h1=100.0, h3=10.0, h5=20.0)
        thd_full = compute_thd(mags)
        thd_limited = compute_thd(mags, max_order=3)
        # thd_limited should only include H3
        assert thd_limited == pytest.approx(10.0 / 100.0, rel=1e-6)
        assert thd_full > thd_limited

    def test_non_finite_harmonics_excluded(self) -> None:
        """NaN magnitude in higher harmonic is treated as zero."""
        mags: dict[int, float] = {1: 100.0, 3: float("nan"), 5: 5.0}
        thd = compute_thd(mags)
        # Only H5 contributes
        assert thd == pytest.approx(5.0 / 100.0, rel=1e-6)

    def test_negative_harmonic_magnitude_excluded(self) -> None:
        """Negative magnitudes are excluded from the sum (treated as 0)."""
        mags = _scalar_mags(h1=100.0, h3=-5.0)
        assert compute_thd(mags) == pytest.approx(0.0, abs=1e-10)

    def test_fundamental_not_included_in_distortion(self) -> None:
        """H1 is the reference, not part of the distortion sum."""
        mags = _scalar_mags(h1=100.0)
        assert compute_thd(mags) == pytest.approx(0.0, abs=1e-10)

    def test_empty_dict_returns_zero(self) -> None:
        """Empty harmonics dict → THD = 0.0 (no fundamental present)."""
        assert compute_thd({}) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeTHDArray (vectorized)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTHDArray:
    def test_output_shape_matches_input(self) -> None:
        """Output length equals length of the magnitude arrays."""
        n = 50
        mags = _array_mags(n, h1=100.0, h3=5.0)
        thd = compute_thd_array(mags)
        assert thd.shape == (n,)

    def test_pure_fundamental_array_thd_zero(self) -> None:
        """Only H1 present → all THD values = 0."""
        n = 20
        mags = _array_mags(n, h1=100.0)
        thd = compute_thd_array(mags)
        assert np.all(thd == pytest.approx(0.0, abs=1e-10))

    def test_known_mixture_array_thd(self) -> None:
        """Constant H1=100, H3=5, H5=3 → THD = sqrt(34)/100 at every window."""
        n = 30
        mags = _array_mags(n, h1=100.0, h3=5.0, h5=3.0)
        thd = compute_thd_array(mags)
        expected = np.sqrt(5.0**2 + 3.0**2) / 100.0
        assert np.allclose(thd, expected, rtol=1e-6)

    def test_zero_fundamental_windows_return_zero(self) -> None:
        """Windows with H1 < threshold → THD = 0, no division error."""
        n = 10
        h1 = np.ones(n) * 100.0
        h1[3] = 0.0  # one dead window
        h3 = np.ones(n) * 5.0
        thd = compute_thd_array({1: h1, 3: h3})
        assert thd[3] == pytest.approx(0.0, abs=1e-10)
        assert np.all(thd >= 0.0)

    def test_missing_fundamental_returns_empty(self) -> None:
        """No H1 key → returns empty array."""
        mags = _array_mags(20, h3=5.0)
        thd = compute_thd_array(mags)
        assert len(thd) == 0

    def test_max_order_excludes_higher_orders(self) -> None:
        """max_order=3 excludes H5 from vectorized sum."""
        n = 15
        mags = _array_mags(n, h1=100.0, h3=10.0, h5=20.0)
        thd_full = compute_thd_array(mags)
        thd_limited = compute_thd_array(mags, max_order=3)
        assert np.all(thd_limited < thd_full)
        assert np.allclose(thd_limited, 10.0 / 100.0, rtol=1e-6)

    def test_mismatched_array_length_skipped(self) -> None:
        """Arrays with wrong length are skipped (no crash)."""
        mags = {1: np.ones(20) * 100.0, 3: np.ones(10) * 5.0}  # H3 wrong length
        thd = compute_thd_array(mags)
        # H3 skipped → THD = 0
        assert np.allclose(thd, 0.0, atol=1e-10)

    def test_output_dtype_is_float64(self) -> None:
        thd = compute_thd_array(_array_mags(5, h1=100.0, h3=5.0))
        assert thd.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeTHDFromResult
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTHDFromResult:
    def _make_result(self, n: int = 20, **kwargs: float) -> HarmonicResult:
        t = np.linspace(0.0, 1.0, n)
        mags = _array_mags(n, **kwargs)
        return HarmonicResult(
            harmonic_time=t,
            magnitudes=mags,
            sample_rate_hz=5000.0,
            nominal_hz=50.0,
            window_samples=200,
            hop_samples=100,
        )

    def test_delegates_to_compute_thd_array(self) -> None:
        """compute_thd_from_result matches compute_thd_array on same data."""
        result = self._make_result(h1=100.0, h3=5.0, h5=3.0)
        thd_from_result = compute_thd_from_result(result)
        thd_direct = compute_thd_array(result.magnitudes)
        assert np.allclose(thd_from_result, thd_direct)

    def test_output_length_matches_n_windows(self) -> None:
        n = 25
        result = self._make_result(n, h1=100.0, h3=10.0)
        thd = compute_thd_from_result(result)
        assert len(thd) == n


# ─────────────────────────────────────────────────────────────────────────────
# TestIndividualHarmonicDistortion
# ─────────────────────────────────────────────────────────────────────────────


class TestIndividualHarmonicDistortion:
    def test_known_ratio(self) -> None:
        """H3/H1 = 5/100 = 0.05."""
        mags = _scalar_mags(h1=100.0, h3=5.0)
        assert individual_harmonic_distortion(mags, order=3) == pytest.approx(0.05, rel=1e-6)

    def test_absent_order_returns_zero(self) -> None:
        mags = _scalar_mags(h1=100.0)
        assert individual_harmonic_distortion(mags, order=5) == 0.0

    def test_zero_fundamental_returns_zero(self) -> None:
        mags = _scalar_mags(h1=0.0, h3=5.0)
        assert individual_harmonic_distortion(mags, order=3) == 0.0

    def test_fundamental_order_1_returns_one(self) -> None:
        mags = _scalar_mags(h1=100.0)
        assert individual_harmonic_distortion(mags, order=1) == pytest.approx(1.0, rel=1e-6)
