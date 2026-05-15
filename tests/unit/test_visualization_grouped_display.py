"""Unit tests for VisualizationManager grouped display (Phase D2).

Uses MagicMock canvas/timeline factories — no QApplication required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.data.synthetic import make_high_rate_record, make_mixed_disturbance_record
from app.visualization.channel_grouper import (
    DISPLAY_GROUP_CURRENT_RAW,
    DISPLAY_GROUP_DIGITAL,
    DISPLAY_GROUP_FREQUENCY,
    DISPLAY_GROUP_POWER,
    DISPLAY_GROUP_VOLTAGE_RAW,
)
from app.visualization.managers.visualization_manager import (
    VisualizationManager,
    _make_filtered_record,
)


def _make_manager() -> tuple[VisualizationManager, MagicMock, MagicMock]:
    canvas = MagicMock()
    timeline = MagicMock()
    return VisualizationManager(canvas, timeline), canvas, timeline


# ─────────────────────────────────────────────────────────────────────────────
# TestMakeFilteredRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeFilteredRecord:
    def test_returns_only_specified_channels(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw", "VB_raw"])
        names = {ch.name for ch in filtered.analog_channels}
        assert names == {"VA_raw", "VB_raw"}

    def test_excluded_channel_absent_from_dataframe(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw"])
        assert "IA_raw" not in filtered.waveform_data.columns

    def test_time_column_preserved(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw"])
        assert "time" in filtered.waveform_data.columns

    def test_channel_indices_recontiguized(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VB_raw", "VC_raw"])
        indices = [ch.index for ch in filtered.analog_channels]
        assert indices == list(range(len(indices)))

    def test_digital_channels_empty(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw"])
        assert filtered.digital_channels == []

    def test_metadata_shared_by_reference(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw"])
        assert filtered.metadata is result.record.metadata

    def test_timing_shared_by_reference(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw"])
        assert filtered.timing_info is result.record.timing_info

    def test_validate_passes_after_filtering(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["VA_raw", "VB_raw", "VC_raw"])
        assert filtered.validate() == []

    def test_all_group_channels_form_valid_record(self) -> None:
        result = make_mixed_disturbance_record()
        filtered = _make_filtered_record(result.record, ["MW", "MVar"])
        errors = filtered.validate()
        assert errors == []
        assert {"MW", "MVar"} == {ch.name for ch in filtered.analog_channels}


# ─────────────────────────────────────────────────────────────────────────────
# TestDisplayGroupedRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayGroupedRecord:
    def test_returns_dict(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert isinstance(panels, dict)

    def test_voltage_group_in_panels(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_VOLTAGE_RAW in panels

    def test_current_group_in_panels(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_CURRENT_RAW in panels

    def test_power_group_in_panels(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_POWER in panels

    def test_frequency_group_in_panels(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_FREQUENCY in panels

    def test_digital_group_not_in_panels(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_DIGITAL not in panels

    def test_empty_groups_skipped_for_high_rate_record(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_high_rate_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert DISPLAY_GROUP_VOLTAGE_RAW in panels
        assert DISPLAY_GROUP_CURRENT_RAW in panels
        assert DISPLAY_GROUP_POWER not in panels
        assert DISPLAY_GROUP_FREQUENCY not in panels

    def test_set_record_called_on_each_canvas(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        created: list[MagicMock] = []

        def factory() -> MagicMock:
            m = MagicMock()
            created.append(m)
            return m

        mgr.display_grouped_record(result.record, result.signal_metadata, canvas_factory=factory)
        for canvas in created:
            canvas.set_record.assert_called_once()

    def test_each_canvas_receives_filtered_record(self) -> None:
        """Each panel canvas should get a record with only its group's channels."""
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        received_records: list = []

        def factory() -> MagicMock:
            m = MagicMock()
            m.set_record.side_effect = lambda r, **kw: received_records.append(r)
            return m

        mgr.display_grouped_record(result.record, result.signal_metadata, canvas_factory=factory)
        # Every filtered record should have fewer channels than the full record
        full_count = len(result.record.analog_channels)
        for rec in received_records:
            assert len(rec.analog_channels) < full_count

    def test_panel_canvases_property_matches_return_value(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        panels = mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert mgr.panel_canvases == panels

    def test_record_property_updated(self) -> None:
        mgr, _, _ = _make_manager()
        result = make_mixed_disturbance_record()
        mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        assert mgr.record is result.record

    def test_timeline_not_called_for_record_without_digital_channels(self) -> None:
        mgr, _, timeline = _make_manager()
        result = make_mixed_disturbance_record()  # no digital channels
        mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        timeline.set_record.assert_not_called()

    def test_existing_set_record_path_unaffected(self) -> None:
        """Calling set_record() after display_grouped_record() still routes to
        the original canvas and timeline."""
        mgr, canvas, timeline = _make_manager()
        result = make_mixed_disturbance_record()
        mgr.display_grouped_record(
            result.record, result.signal_metadata, canvas_factory=MagicMock
        )
        record2 = MagicMock()
        mgr.set_record(record2)
        canvas.set_record.assert_called_once_with(record2, axis_mode="relative_seconds")
        timeline.set_record.assert_called_once_with(record2)
