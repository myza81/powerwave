"""Tests for DatetimeAxisItem (app/visualization/axis/datetime_axis.py).

Pure-Python tests cover _choose_format and tickStrings logic without Qt.
Qt-dependent instantiation tests use the qapp fixture.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ─────────────────────────────────────────────────────────────────────────────
# _choose_format — pure Python, no Qt needed
# ─────────────────────────────────────────────────────────────────────────────

from app.visualization.axis.datetime_axis import _choose_format
from app.visualization.axis.datetime_axis import (
    AXIS_MODE_ABSOLUTE,
    AXIS_MODE_RELATIVE,
    TimeDisplayMode,
)


class TestTimeDisplayMode:
    def test_relative_value_matches_axis_mode_constant(self) -> None:
        assert TimeDisplayMode.RELATIVE.value == AXIS_MODE_RELATIVE

    def test_absolute_value_matches_axis_mode_constant(self) -> None:
        assert TimeDisplayMode.ABSOLUTE.value == AXIS_MODE_ABSOLUTE

    def test_coerce_accepts_legacy_axis_mode_string(self) -> None:
        assert TimeDisplayMode.coerce("absolute_datetime") is TimeDisplayMode.ABSOLUTE


class TestChooseFormat:
    def test_sub_second_returns_ms_sentinel(self) -> None:
        assert _choose_format(0.5) == "ms"

    def test_zero_spacing_returns_ms_sentinel(self) -> None:
        assert _choose_format(0.0) == "ms"

    def test_one_second_spacing(self) -> None:
        assert _choose_format(1.0) == "%H:%M:%S"

    def test_thirty_second_spacing(self) -> None:
        assert _choose_format(30.0) == "%H:%M:%S"

    def test_minute_spacing(self) -> None:
        assert _choose_format(60.0) == "%H:%M:%S"

    def test_one_hour_spacing(self) -> None:
        assert _choose_format(3600.0) == "%m-%d %H:%M"

    def test_six_hour_spacing(self) -> None:
        assert _choose_format(6 * 3600.0) == "%m-%d %H:%M"

    def test_one_day_spacing(self) -> None:
        assert _choose_format(86400.0) == "%Y-%m-%d"

    def test_multi_day_spacing(self) -> None:
        assert _choose_format(7 * 86400.0) == "%Y-%m-%d"

    def test_just_below_hour_threshold(self) -> None:
        assert _choose_format(3599.9) == "%H:%M:%S"

    def test_just_below_day_threshold(self) -> None:
        assert _choose_format(86399.9) == "%m-%d %H:%M"


# ─────────────────────────────────────────────────────────────────────────────
# DatetimeAxisItem — requires Qt
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    import pyqtgraph as pg
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    pg.setConfigOptions(useOpenGL=False, antialias=False)
    return app


@pytest.fixture
def axis(qapp):
    from app.visualization.axis.datetime_axis import DatetimeAxisItem
    return DatetimeAxisItem(orientation="bottom")


class TestDatetimeAxisItemNoStartTime:
    def test_empty_values_returns_empty(self, axis) -> None:
        result = axis.tickStrings([], 1.0, 1.0)
        assert result == []

    def test_no_start_time_returns_seconds_labels(self, axis) -> None:
        labels = axis.tickStrings([0.0, 1.0, 2.0], 1.0, 1.0)
        assert len(labels) == 3
        assert all("s" in lbl for lbl in labels)

    def test_no_start_time_float_format(self, axis) -> None:
        labels = axis.tickStrings([3.5], 1.0, 1.0)
        assert "3.500 s" in labels[0]


class TestDatetimeAxisItemWithStartTime:
    _START = datetime(2026, 3, 6, 17, 25, 0)

    def test_set_start_time_accepted(self, axis) -> None:
        axis.set_start_time(self._START)
        assert axis._start_time == self._START

    def test_set_start_time_none_resets(self, axis) -> None:
        axis.set_start_time(self._START)
        axis.set_start_time(None)
        labels = axis.tickStrings([0.0], 1.0, 1.0)
        assert "s" in labels[0]

    def test_second_spacing_shows_hms(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([0.0, 1.0, 2.0], 1.0, 1.0)
        # Each label should contain "17:25"
        for lbl in labels:
            assert ":" in lbl

    def test_zero_elapsed_matches_start_time(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([0.0], 1.0, 1.0)
        assert "17:25:00" in labels[0]

    def test_sixty_seconds_elapsed(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([60.0], 1.0, 1.0)
        # 17:25:00 + 60s = 17:26:00
        assert "17:26:00" in labels[0]

    def test_sub_second_spacing_shows_milliseconds(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([0.0, 0.1], 1.0, 0.1)
        # sub-second spacing → "ms" format → HH:MM:SS.mmm
        for lbl in labels:
            assert "." in lbl

    def test_hour_spacing_shows_date_and_hour(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([0.0], 1.0, 3600.0)
        # hour spacing → "%m-%d %H:%M"
        assert "-" in labels[0]
        assert ":" in labels[0]

    def test_day_spacing_shows_date_only(self, axis) -> None:
        axis.set_start_time(self._START)
        labels = axis.tickStrings([0.0], 1.0, 86400.0)
        # day spacing → "%Y-%m-%d"
        assert "2026" in labels[0]
        assert ":" not in labels[0]

    def test_sparse_data_long_span(self, axis) -> None:
        """Low-rate trend data: start at epoch, ticks spaced 600 s (10 min)."""
        axis.set_start_time(datetime(2026, 3, 6, 0, 0, 0))
        labels = axis.tickStrings([0.0, 600.0, 1200.0], 1.0, 600.0)
        assert len(labels) == 3
        # minute-level spacing → HH:MM:SS
        assert "00:00:00" in labels[0]
        assert "00:10:00" in labels[1]
        assert "00:20:00" in labels[2]
