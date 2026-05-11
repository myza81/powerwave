"""Unit tests for app.visualization.rendering.downsampling.decimate_for_display.

All tests are pure NumPy — no PyQt6 or display required.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.visualization.rendering.downsampling import decimate_for_display

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _uniform(n: int, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, data) arrays of length n with uniform dt spacing."""
    t = np.arange(n, dtype=np.float64) * dt
    d = np.sin(2 * np.pi * 50 * t)
    return t, d


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateBelowMaxPoints
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateBelowMaxPoints:
    def test_fewer_than_max_returns_all(self) -> None:
        t, d = _uniform(100)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=1_000)
        assert len(t_out) == 100
        assert len(d_out) == 100

    def test_exactly_max_returns_unchanged(self) -> None:
        t, d = _uniform(500)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=500)
        assert len(t_out) == 500

    def test_single_sample_in_viewport(self) -> None:
        t = np.array([0.5], dtype=np.float64)
        d = np.array([1.0], dtype=np.float64)
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0, max_points=1_000)
        assert len(t_out) == 1
        assert t_out[0] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateAboveMaxPoints
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateAboveMaxPoints:
    def test_output_within_max_points(self) -> None:
        t, d = _uniform(50_000)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=4_000)
        assert len(t_out) <= 4_000
        assert len(d_out) <= 4_000

    def test_output_values_are_subset_of_input(self) -> None:
        t, d = _uniform(20_000)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=100)
        input_set = set(d.tolist())
        for v in d_out.tolist():
            assert v in input_set

    def test_output_time_values_are_subset_of_input(self) -> None:
        t, d = _uniform(10_000)
        t_out, _ = decimate_for_display(t, d, 0.0, t[-1], max_points=100)
        input_set = set(np.round(t, 10).tolist())
        for v in np.round(t_out, 10).tolist():
            assert v in input_set

    def test_first_sample_preserved(self) -> None:
        t, d = _uniform(10_000)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=100)
        assert t_out[0] == pytest.approx(t[0])
        assert d_out[0] == pytest.approx(d[0])

    def test_non_uniform_spacing_handled(self) -> None:
        # Non-uniform spacing: still decimates without interpolation
        t = np.concatenate([np.linspace(0, 0.1, 500), np.linspace(0.1, 1.0, 9_500)])
        d = np.random.default_rng(42).standard_normal(len(t))
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0, max_points=200)
        assert len(t_out) <= 200
        assert len(d_out) == len(t_out)


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateClipping
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateClipping:
    def test_clips_to_viewport(self) -> None:
        t, d = _uniform(1_000, dt=0.001)   # 0 … 0.999 s
        t_out, _ = decimate_for_display(t, d, 0.2, 0.5, max_points=4_000)
        assert float(t_out[0]) >= 0.2
        assert float(t_out[-1]) <= 0.5

    def test_samples_after_t_end_excluded(self) -> None:
        t, d = _uniform(1_000, dt=0.001)
        t_out, _ = decimate_for_display(t, d, 0.0, 0.3, max_points=4_000)
        assert float(t_out[-1]) <= 0.3

    def test_samples_before_t_start_excluded(self) -> None:
        t, d = _uniform(1_000, dt=0.001)
        t_out, _ = decimate_for_display(t, d, 0.5, 0.9, max_points=4_000)
        assert float(t_out[0]) >= 0.5

    def test_out_of_range_returns_empty(self) -> None:
        t, d = _uniform(500, dt=0.001)   # 0 … 0.499 s
        t_out, d_out = decimate_for_display(t, d, 10.0, 20.0, max_points=4_000)
        assert len(t_out) == 0
        assert len(d_out) == 0

    def test_t_start_equals_t_end_returns_at_most_one(self) -> None:
        t = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        t_out, d_out = decimate_for_display(t, d, 0.1, 0.1, max_points=4_000)
        assert len(t_out) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateEdgeCases
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateEdgeCases:
    def test_empty_time_array_returns_empty(self) -> None:
        t = np.empty(0, dtype=np.float64)
        d = np.empty(0, dtype=np.float64)
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0)
        assert len(t_out) == 0
        assert len(d_out) == 0

    def test_empty_data_array_returns_empty(self) -> None:
        # Same as above — empty arrays are always equal length
        t = np.empty(0, dtype=np.float64)
        d = np.empty(0, dtype=np.float64)
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0)
        assert len(t_out) == 0
        assert len(d_out) == 0

    def test_single_in_viewport_returned(self) -> None:
        t = np.array([0.5], dtype=np.float64)
        d = np.array([42.0], dtype=np.float64)
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0)
        assert len(t_out) == 1
        assert d_out[0] == pytest.approx(42.0)

    def test_t_start_greater_than_t_end_swapped_silently(self) -> None:
        t, d = _uniform(200, dt=0.001)   # 0 … 0.199 s
        # Swap t_start and t_end — should still return data in 0.05–0.15 range
        t_out, d_out = decimate_for_display(t, d, 0.15, 0.05)
        assert len(t_out) > 0
        assert float(t_out[0]) >= 0.05
        assert float(t_out[-1]) <= 0.15

    def test_large_dataset_within_max_points(self) -> None:
        t = np.arange(1_000_000, dtype=np.float64) * 1e-4
        d = np.random.default_rng(7).standard_normal(1_000_000)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=4_000)
        assert len(t_out) <= 4_000
        assert len(d_out) <= 4_000


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateInputValidation
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateInputValidation:
    def test_mismatched_lengths_raises(self) -> None:
        t = np.arange(10, dtype=np.float64)
        d = np.arange(5, dtype=np.float64)
        with pytest.raises(ValueError, match="length mismatch"):
            decimate_for_display(t, d, 0.0, 1.0)

    def test_2d_time_raises(self) -> None:
        t = np.ones((5, 2), dtype=np.float64)
        d = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            decimate_for_display(t, d, 0.0, 1.0)

    def test_2d_data_raises(self) -> None:
        t = np.arange(10, dtype=np.float64)
        d = np.ones((5, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            decimate_for_display(t, d, 0.0, 1.0)

    def test_0d_input_raises(self) -> None:
        t = np.array(5.0)   # 0-D scalar
        d = np.array(1.0)
        with pytest.raises(ValueError, match="1-D"):
            decimate_for_display(t, d, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestDecimateReturnTypes
# ─────────────────────────────────────────────────────────────────────────────


class TestDecimateReturnTypes:
    def test_output_time_is_float64(self) -> None:
        t, d = _uniform(100)
        t_out, _ = decimate_for_display(t, d, 0.0, t[-1])
        assert t_out.dtype == np.float64

    def test_output_data_is_float64(self) -> None:
        t, d = _uniform(100)
        _, d_out = decimate_for_display(t, d, 0.0, t[-1])
        assert d_out.dtype == np.float64

    def test_float32_input_produces_float64_output(self) -> None:
        t = np.linspace(0, 1, 100, dtype=np.float32)
        d = np.ones(100, dtype=np.float32)
        t_out, d_out = decimate_for_display(t, d, 0.0, 1.0)
        assert t_out.dtype == np.float64
        assert d_out.dtype == np.float64

    def test_integer_input_produces_float64_output(self) -> None:
        t = np.arange(100, dtype=np.int32)
        d = np.ones(100, dtype=np.int64)
        t_out, d_out = decimate_for_display(t, d, 0, 99)
        assert t_out.dtype == np.float64
        assert d_out.dtype == np.float64

    def test_input_arrays_not_modified(self) -> None:
        t = np.linspace(0, 1, 10_000, dtype=np.float64)
        d = np.sin(t)
        t_copy = t.copy()
        d_copy = d.copy()
        decimate_for_display(t, d, 0.2, 0.8, max_points=100)
        np.testing.assert_array_equal(t, t_copy)
        np.testing.assert_array_equal(d, d_copy)

    def test_both_outputs_are_1d(self) -> None:
        t, d = _uniform(500)
        t_out, d_out = decimate_for_display(t, d, 0.0, t[-1], max_points=50)
        assert t_out.ndim == 1
        assert d_out.ndim == 1
