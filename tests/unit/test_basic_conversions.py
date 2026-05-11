"""Unit tests for app.analytics.basic_conversions."""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.basic_conversions import sliding_rms, to_per_unit


class TestSlidingRms:
    def test_sine_wave_rms_value(self) -> None:
        fs = 6400.0
        f0 = 50.0
        cycles = 10
        t = np.arange(0, cycles / f0, 1.0 / fs)
        values = np.cos(2.0 * np.pi * f0 * t)
        window = int(fs / f0)  # one cycle = 128 samples
        result = sliding_rms(values, window)
        # RMS of unit cosine = 1/√2 ≈ 0.7071
        assert np.allclose(result, 1.0 / np.sqrt(2.0), atol=1e-3)

    def test_output_length(self) -> None:
        values = np.ones(100, dtype=np.float64)
        result = sliding_rms(values, 10)
        assert len(result) == 91  # N - W + 1 = 100 - 10 + 1

    def test_window_1_equals_abs(self) -> None:
        values = np.array([-3.0, 1.0, -2.0, 4.0])
        result = sliding_rms(values, 1)
        np.testing.assert_array_almost_equal(result, np.abs(values))

    def test_constant_input(self) -> None:
        values = np.full(50, 3.0)
        result = sliding_rms(values, 10)
        np.testing.assert_array_almost_equal(result, np.full(41, 3.0))

    def test_returns_float64(self) -> None:
        values = np.array([1, 2, 3, 4, 5])
        result = sliding_rms(values, 3)
        assert result.dtype == np.float64

    def test_zero_input_gives_zero_rms(self) -> None:
        values = np.zeros(50)
        result = sliding_rms(values, 10)
        np.testing.assert_array_almost_equal(result, np.zeros(41))

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window_samples"):
            sliding_rms(np.ones(10), 0)

    def test_window_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="window_samples"):
            sliding_rms(np.ones(10), -5)

    def test_window_exceeds_length_raises(self) -> None:
        with pytest.raises(ValueError):
            sliding_rms(np.ones(5), 10)

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            sliding_rms(np.ones((10, 2)), 5)


class TestToPerUnit:
    def test_correct_division(self) -> None:
        values = np.array([100.0, 200.0, 300.0])
        result = to_per_unit(values, 100.0)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_preserves_shape(self) -> None:
        values = np.random.rand(50)
        result = to_per_unit(values, 10.0)
        assert result.shape == values.shape

    def test_float64_output(self) -> None:
        values = np.array([1, 2, 3])  # int input
        result = to_per_unit(values, 2.0)
        assert result.dtype == np.float64

    def test_zero_base_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            to_per_unit(np.ones(5), 0.0)

    def test_negative_base(self) -> None:
        values = np.array([1.0, -1.0])
        result = to_per_unit(values, -1.0)
        np.testing.assert_array_almost_equal(result, [-1.0, 1.0])

    def test_scalar_array(self) -> None:
        result = to_per_unit(np.array([10.0]), 5.0)
        np.testing.assert_almost_equal(result[0], 2.0)
