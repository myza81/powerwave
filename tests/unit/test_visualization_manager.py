"""Unit tests for VisualizationManager coordination logic.

Uses MagicMock stubs for FlexiblePlotCanvas and DigitalEventTimeline so that
no Qt display or QApplication is required. Only coordination behavior is tested.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.visualization.managers.visualization_manager import VisualizationManager


def _make_manager() -> tuple[VisualizationManager, MagicMock, MagicMock]:
    """Build a VisualizationManager with MagicMock canvas and timeline."""
    canvas = MagicMock()
    timeline = MagicMock()
    mgr = VisualizationManager(canvas, timeline)
    return mgr, canvas, timeline


# ─────────────────────────────────────────────────────────────────────────────
# TestConstruction
# ─────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_canvas_property_returns_canvas(self) -> None:
        mgr, canvas, _ = _make_manager()
        assert mgr.canvas is canvas

    def test_timeline_property_returns_timeline(self) -> None:
        mgr, _, timeline = _make_manager()
        assert mgr.timeline is timeline

    def test_record_is_none_initially(self) -> None:
        mgr, _, _ = _make_manager()
        assert mgr.record is None

    def test_is_x_linked_false_initially(self) -> None:
        mgr, _, _ = _make_manager()
        assert mgr.is_x_linked is False

    def test_canvas_cursor_signal_connected(self) -> None:
        mgr, canvas, _ = _make_manager()
        canvas.cursor_moved.connect.assert_called_once_with(mgr._on_canvas_cursor_moved)

    def test_timeline_cursor_signal_connected(self) -> None:
        mgr, _, timeline = _make_manager()
        timeline.cursor_moved.connect.assert_called_once_with(mgr._on_timeline_cursor_moved)


# ─────────────────────────────────────────────────────────────────────────────
# TestSetRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestSetRecord:
    def test_canvas_set_record_called(self) -> None:
        mgr, canvas, _ = _make_manager()
        record = MagicMock()
        mgr.set_record(record)
        canvas.set_record.assert_called_once_with(record, axis_mode="relative_seconds")

    def test_timeline_set_record_called(self) -> None:
        mgr, _, timeline = _make_manager()
        record = MagicMock()
        mgr.set_record(record)
        timeline.set_record.assert_called_once_with(record)

    def test_record_property_updated(self) -> None:
        mgr, _, _ = _make_manager()
        record = MagicMock()
        mgr.set_record(record)
        assert mgr.record is record

    def test_second_set_record_replaces_first(self) -> None:
        mgr, canvas, timeline = _make_manager()
        r1, r2 = MagicMock(), MagicMock()
        mgr.set_record(r1)
        mgr.set_record(r2)
        assert mgr.record is r2
        assert canvas.set_record.call_count == 2
        assert timeline.set_record.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestClear
# ─────────────────────────────────────────────────────────────────────────────


class TestClear:
    def test_canvas_clear_called(self) -> None:
        # VisualizationManager.clear() calls _clear_canvas() (not clear) to avoid
        # shadowing by pg.GraphicsLayoutWidget.__init__ (see flexible_plot_canvas.py)
        mgr, canvas, _ = _make_manager()
        mgr.clear()
        canvas._clear_canvas.assert_called_once()

    def test_timeline_clear_called(self) -> None:
        mgr, _, timeline = _make_manager()
        mgr.clear()
        timeline.clear.assert_called_once()

    def test_record_is_none_after_clear(self) -> None:
        mgr, _, _ = _make_manager()
        mgr.set_record(MagicMock())
        mgr.clear()
        assert mgr.record is None

    def test_clear_without_record_does_not_raise(self) -> None:
        mgr, canvas, timeline = _make_manager()
        mgr.clear()  # no record loaded
        canvas._clear_canvas.assert_called_once()
        timeline.clear.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestCursorSync
# ─────────────────────────────────────────────────────────────────────────────


class TestCursorSync:
    def test_canvas_cursor_forwarded_to_timeline(self) -> None:
        mgr, _, timeline = _make_manager()
        mgr._on_canvas_cursor_moved(2.5)
        timeline.set_cursor_pos.assert_called_once_with(2.5)

    def test_timeline_cursor_forwarded_to_canvas(self) -> None:
        mgr, canvas, _ = _make_manager()
        mgr._on_timeline_cursor_moved(3.7)
        canvas.set_cursor_pos.assert_called_once_with(3.7)

    def test_set_cursor_pos_calls_canvas(self) -> None:
        mgr, canvas, _ = _make_manager()
        mgr.set_cursor_pos(1.0)
        canvas.set_cursor_pos.assert_called_once_with(1.0)

    def test_set_cursor_pos_calls_timeline(self) -> None:
        mgr, _, timeline = _make_manager()
        mgr.set_cursor_pos(1.0)
        timeline.set_cursor_pos.assert_called_once_with(1.0)

    def test_forwarded_value_is_exact(self) -> None:
        mgr, canvas, timeline = _make_manager()
        mgr._on_canvas_cursor_moved(0.12345)
        timeline.set_cursor_pos.assert_called_once_with(0.12345)
        mgr._on_timeline_cursor_moved(9.99876)
        canvas.set_cursor_pos.assert_called_once_with(9.99876)


# ─────────────────────────────────────────────────────────────────────────────
# TestLinkXAxis
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkXAxis:
    def test_link_x_axis_registers_canvas_and_timeline_for_synchronization(self) -> None:
        mgr, canvas, timeline = _make_manager()
        mgr.link_x_axis()
        assert mgr.synchronization_manager.registered_count == 2

    def test_is_x_linked_true_after_link(self) -> None:
        mgr, _, _ = _make_manager()
        mgr.link_x_axis()
        assert mgr.is_x_linked is True

    def test_is_x_linked_false_before_link(self) -> None:
        mgr, _, _ = _make_manager()
        assert mgr.is_x_linked is False


# ─────────────────────────────────────────────────────────────────────────────
# TestZoomToTrigger
# ─────────────────────────────────────────────────────────────────────────────


class TestZoomToTrigger:
    def test_canvas_zoom_to_trigger_called_with_window_s(self) -> None:
        mgr, canvas, _ = _make_manager()
        mgr.zoom_to_trigger(0.3)
        canvas.zoom_to_trigger.assert_called_once_with(0.3)

    def test_timeline_not_zoomed_when_x_linked(self) -> None:
        mgr, canvas, timeline = _make_manager()
        mgr._x_linked = True
        with patch.object(mgr, "_zoom_timeline_to_trigger") as mock_inner:
            mgr.zoom_to_trigger(0.2)
        mock_inner.assert_not_called()
        canvas.zoom_to_trigger.assert_called_once_with(0.2)

    def test_timeline_zoomed_when_not_x_linked(self) -> None:
        mgr, _, _ = _make_manager()
        with patch.object(mgr, "_zoom_timeline_to_trigger") as mock_inner:
            mgr.zoom_to_trigger(0.15)
        mock_inner.assert_called_once_with(0.15)

    def test_default_window_s_is_0_2(self) -> None:
        mgr, canvas, _ = _make_manager()
        mgr._x_linked = True
        mgr.zoom_to_trigger()
        canvas.zoom_to_trigger.assert_called_once_with(0.2)


# ─────────────────────────────────────────────────────────────────────────────
# TestResetViewport
# ─────────────────────────────────────────────────────────────────────────────


class TestResetViewport:
    def test_no_op_without_record(self) -> None:
        mgr, canvas, _ = _make_manager()
        mgr.reset_viewport()
        canvas._primary_plot.setXRange.assert_not_called()

    def test_canvas_set_x_range_called_when_x_linked(self) -> None:
        # MagicMock: len(time_col) == 0, so t_max falls back to 1.0
        mgr, canvas, timeline = _make_manager()
        mgr._record = MagicMock()
        mgr._x_linked = True
        mgr.reset_viewport()
        canvas._primary_plot.setXRange.assert_called_once_with(0.0, 1.0, padding=0)

    def test_timeline_not_reset_when_x_linked(self) -> None:
        mgr, _, timeline = _make_manager()
        mgr._record = MagicMock()
        mgr._x_linked = True
        mgr.reset_viewport()
        timeline.getPlotItem.assert_not_called()

    def test_timeline_reset_when_not_x_linked(self) -> None:
        # MagicMock: len(time_col) == 0, so t_max falls back to 1.0
        mgr, canvas, timeline = _make_manager()
        mgr._record = MagicMock()
        mgr._x_linked = False
        mgr.reset_viewport()
        canvas._primary_plot.setXRange.assert_called_once_with(0.0, 1.0, padding=0)
        timeline.getPlotItem.return_value.setXRange.assert_called_once_with(
            0.0, 1.0, padding=0
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestZoomTimelineToTrigger
# ─────────────────────────────────────────────────────────────────────────────


class TestZoomTimelineToTrigger:
    def test_no_op_when_no_record(self) -> None:
        mgr, _, timeline = _make_manager()
        mgr._zoom_timeline_to_trigger(0.2)  # record is None
        timeline.getPlotItem.assert_not_called()

    def test_calls_timeline_set_x_range(self) -> None:
        mgr, _, timeline = _make_manager()
        record = MagicMock()
        # Configure mock so total_seconds() returns a float (avoids TypeError in max/min)
        diff_mock = MagicMock()
        diff_mock.total_seconds.return_value = 1.0
        record.timing_info.trigger_time.__sub__ = MagicMock(return_value=diff_mock)
        mgr._record = record
        mgr._zoom_timeline_to_trigger(0.2)
        timeline.getPlotItem.return_value.setXRange.assert_called()
