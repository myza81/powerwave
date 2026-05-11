"""Unit tests for app.data.time_alignment.build_display_time_seconds."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from app.data.time_alignment import build_display_time_seconds


class TestNumericInput:
    def test_numeric_passthrough(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        result = build_display_time_seconds(arr)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_offset_applied(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        result = build_display_time_seconds(arr, offset_seconds=0.5)
        np.testing.assert_array_almost_equal(result, [0.5, 1.5, 2.5])

    def test_reference_start_numeric(self) -> None:
        arr = np.array([10.0, 11.0, 12.0])
        result = build_display_time_seconds(arr, reference_start=10.0)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_reference_start_and_offset_combined(self) -> None:
        arr = np.array([10.0, 11.0, 12.0])
        result = build_display_time_seconds(arr, reference_start=10.0, offset_seconds=5.0)
        np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0])

    def test_returns_float64(self) -> None:
        arr = np.array([1, 2, 3])  # int input
        result = build_display_time_seconds(arr)
        assert result.dtype == np.float64

    def test_empty_input_returns_empty(self) -> None:
        result = build_display_time_seconds(np.array([]))
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_single_element(self) -> None:
        result = build_display_time_seconds(np.array([7.0]))
        np.testing.assert_array_almost_equal(result, [7.0])

    def test_does_not_mutate_input(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        original = arr.copy()
        build_display_time_seconds(arr, reference_start=0.5)
        np.testing.assert_array_equal(arr, original)


class TestDatetimeInput:
    def test_datetime_list_to_seconds(self) -> None:
        t0 = datetime(2024, 1, 1, 0, 0, 0)
        t1 = datetime(2024, 1, 1, 0, 0, 1)
        t2 = datetime(2024, 1, 1, 0, 0, 2)
        result = build_display_time_seconds([t0, t1, t2])
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_datetime_with_explicit_reference_start(self) -> None:
        ref = datetime(2024, 1, 1, 0, 0, 0)
        t1 = datetime(2024, 1, 1, 0, 0, 5)
        result = build_display_time_seconds([t1], reference_start=ref)
        np.testing.assert_almost_equal(result[0], 5.0)

    def test_datetime_with_offset(self) -> None:
        t0 = datetime(2024, 1, 1, 0, 0, 0)
        t1 = datetime(2024, 1, 1, 0, 0, 1)
        result = build_display_time_seconds([t0, t1], offset_seconds=10.0)
        np.testing.assert_array_almost_equal(result, [10.0, 11.0])

    def test_datetime_returns_float64(self) -> None:
        t0 = datetime(2024, 1, 1)
        t1 = datetime(2024, 1, 1, 0, 0, 1)
        result = build_display_time_seconds([t0, t1])
        assert result.dtype == np.float64
