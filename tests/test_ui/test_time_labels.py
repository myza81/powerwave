"""
tests/test_ui/test_time_labels.py

Focused tests for actual-time labels in the legacy unified canvas.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ui.unified_canvas import UnifiedCanvasWidget, _TimeAxis


def test_time_axis_formats_actual_timestamp_ticks() -> None:
    epoch = 1_704_067_200.123

    assert _TimeAxis._format_epoch_tick(epoch, 0.1) == "00:00:00.123"
    assert _TimeAxis._format_epoch_tick(epoch, 1.0) == "00:00:00"


def test_time_axis_label_uses_utc_when_reference_epoch_is_valid() -> None:
    canvas = UnifiedCanvasWidget.__new__(UnifiedCanvasWidget)
    canvas._xaxis_cycles = False

    assert canvas._time_axis_label(1_764_565_698.0) == "Time (UTC)"
    assert canvas._time_axis_label(0.0) == "Time (s)"


def test_time_axis_label_keeps_cycles_when_enabled() -> None:
    canvas = UnifiedCanvasWidget.__new__(UnifiedCanvasWidget)
    canvas._xaxis_cycles = True

    assert canvas._time_axis_label(1_764_565_698.0) == "Time (cycles)"


def test_cursor_display_time_adds_group_reference_epoch() -> None:
    canvas = UnifiedCanvasWidget.__new__(UnifiedCanvasWidget)
    line = object()
    canvas._cursor1_lines = {(0, 0): line}
    canvas._cursor2_lines = {}
    canvas._groups = {0: SimpleNamespace(ref_epoch=1_764_565_000.0)}

    assert canvas._cursor_display_time(1, 12.5) == pytest.approx(1_764_565_012.5)
