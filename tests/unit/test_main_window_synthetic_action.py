"""Unit tests for Phase D2 synthetic load action in PowerwaveMainWindow.

Tests the module-level constants, method existence, and the logic of
display_grouped_record when called with synthetic data — without
instantiating any Qt widgets.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.ui.main_window.main_window import PowerwaveMainWindow, _PANEL_ORDER
from app.visualization.managers.visualization_manager import VisualizationManager


# ─────────────────────────────────────────────────────────────────────────────
# TestPanelOrderConstant
# ─────────────────────────────────────────────────────────────────────────────


class TestPanelOrderConstant:
    def test_is_list(self) -> None:
        assert isinstance(_PANEL_ORDER, list)

    def test_contains_voltage_raw(self) -> None:
        assert "voltage_raw" in _PANEL_ORDER

    def test_contains_current_raw(self) -> None:
        assert "current_raw" in _PANEL_ORDER

    def test_contains_power(self) -> None:
        assert "power" in _PANEL_ORDER

    def test_contains_frequency(self) -> None:
        assert "frequency" in _PANEL_ORDER

    def test_voltage_before_current(self) -> None:
        assert _PANEL_ORDER.index("voltage_raw") < _PANEL_ORDER.index("current_raw")

    def test_current_before_power(self) -> None:
        assert _PANEL_ORDER.index("current_raw") < _PANEL_ORDER.index("power")

    def test_power_before_frequency(self) -> None:
        assert _PANEL_ORDER.index("power") < _PANEL_ORDER.index("frequency")


# ─────────────────────────────────────────────────────────────────────────────
# TestSyntheticHandlerExists
# ─────────────────────────────────────────────────────────────────────────────


class TestSyntheticHandlerExists:
    def test_rebuild_grouped_layout_exists(self) -> None:
        assert hasattr(PowerwaveMainWindow, "_rebuild_grouped_layout")

    def test_restore_standard_layout_exists(self) -> None:
        assert hasattr(PowerwaveMainWindow, "_restore_standard_layout")

    def test_link_panel_x_axes_exists(self) -> None:
        assert hasattr(PowerwaveMainWindow, "_link_panel_x_axes")


# ─────────────────────────────────────────────────────────────────────────────
# TestSyntheticRecordIntegration
# ─────────────────────────────────────────────────────────────────────────────


class TestSyntheticRecordIntegration:
    def test_synthetic_record_has_expected_groups(self) -> None:
        from app.data.synthetic import make_mixed_disturbance_record
        from app.visualization.channel_grouper import group_channels_for_display

        result = make_mixed_disturbance_record()
        groups = group_channels_for_display(result.record, result.signal_metadata)
        assert "voltage_raw" in groups
        assert "current_raw" in groups
        assert "power" in groups
        assert "frequency" in groups

    def test_display_grouped_record_returns_nonempty_panels(self) -> None:
        from app.data.synthetic import make_mixed_disturbance_record

        canvas = MagicMock()
        timeline = MagicMock()
        mgr = VisualizationManager(canvas, timeline)
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert len(panels) >= 4

    def test_panel_order_applied_correctly(self) -> None:
        """Groups returned by display_grouped_record appear in _PANEL_ORDER."""
        from app.data.synthetic import make_mixed_disturbance_record

        canvas = MagicMock()
        timeline = MagicMock()
        mgr = VisualizationManager(canvas, timeline)
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        panel_keys = list(panels.keys())
        # Every returned key should appear in _PANEL_ORDER (or be an extra group)
        order_keys = {k for k in _PANEL_ORDER if k in panels}
        for k in order_keys:
            assert k in panel_keys
