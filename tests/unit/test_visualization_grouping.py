"""Unit tests for app.visualization.channel_grouper."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.data.signal_metadata import SignalMetadata
from app.models.channels import AnalogChannel, DigitalChannel
from app.visualization.channel_grouper import (
    DISPLAY_GROUP_CURRENT_RAW,
    DISPLAY_GROUP_DIGITAL,
    DISPLAY_GROUP_FREQUENCY,
    DISPLAY_GROUP_OTHER,
    DISPLAY_GROUP_POWER,
    DISPLAY_GROUP_ROCOF,
    DISPLAY_GROUP_VOLTAGE_RAW,
    group_channels_for_display,
)


def _make_record(
    analog_specs: list[tuple[str, str]],
    digital_names: list[str] | None = None,
) -> MagicMock:
    record = MagicMock()
    record.analog_channels = [
        AnalogChannel(name=n, unit=u, index=i)
        for i, (n, u) in enumerate(analog_specs)
    ]
    record.digital_channels = [
        DigitalChannel(name=n, index=i) for i, n in enumerate(digital_names or [])
    ]
    return record


class TestNameHeuristicGrouping:
    def test_voltage_channels_grouped(self) -> None:
        record = _make_record([("VA_raw", "pu"), ("VB_raw", "pu"), ("VC_raw", "pu")])
        groups = group_channels_for_display(record)
        assert set(groups[DISPLAY_GROUP_VOLTAGE_RAW]) == {"VA_raw", "VB_raw", "VC_raw"}

    def test_current_channels_grouped(self) -> None:
        record = _make_record([("IA_raw", "pu"), ("IB_raw", "pu"), ("IC_raw", "pu")])
        groups = group_channels_for_display(record)
        assert set(groups[DISPLAY_GROUP_CURRENT_RAW]) == {"IA_raw", "IB_raw", "IC_raw"}

    def test_power_channels_grouped(self) -> None:
        record = _make_record([("MW", "MW"), ("MVar", "MVar")])
        groups = group_channels_for_display(record)
        assert set(groups[DISPLAY_GROUP_POWER]) == {"MW", "MVar"}

    def test_frequency_channel_grouped(self) -> None:
        record = _make_record([("Frequency", "Hz")])
        groups = group_channels_for_display(record)
        assert "Frequency" in groups[DISPLAY_GROUP_FREQUENCY]

    def test_rocof_channel_grouped(self) -> None:
        record = _make_record([("ROCOF", "Hz/s")])
        groups = group_channels_for_display(record)
        assert "ROCOF" in groups[DISPLAY_GROUP_ROCOF]

    def test_digital_channels_grouped(self) -> None:
        record = _make_record([], digital_names=["CB_Trip", "Relay_52"])
        groups = group_channels_for_display(record)
        assert set(groups[DISPLAY_GROUP_DIGITAL]) == {"CB_Trip", "Relay_52"}

    def test_unknown_channel_goes_to_other(self) -> None:
        record = _make_record([("XYZ_custom", "units")])
        groups = group_channels_for_display(record)
        assert "XYZ_custom" in groups[DISPLAY_GROUP_OTHER]

    def test_empty_record_returns_empty_dict(self) -> None:
        record = _make_record([])
        groups = group_channels_for_display(record)
        assert groups == {}


class TestMetadataDrivenGrouping:
    def test_metadata_overrides_heuristic(self) -> None:
        record = _make_record([("MW", "MW")])  # MW → power by heuristic
        metadata = {
            "MW": SignalMetadata(name="MW", display_group=DISPLAY_GROUP_FREQUENCY)
        }
        groups = group_channels_for_display(record, signal_metadata=metadata)
        assert "MW" in groups[DISPLAY_GROUP_FREQUENCY]
        assert DISPLAY_GROUP_POWER not in groups

    def test_none_display_group_in_metadata_falls_back_to_heuristic(self) -> None:
        record = _make_record([("VA_raw", "pu")])
        metadata = {
            "VA_raw": SignalMetadata(name="VA_raw", display_group=None)
        }
        groups = group_channels_for_display(record, signal_metadata=metadata)
        # display_group=None → falls back to _infer → voltage_raw
        assert "VA_raw" in groups[DISPLAY_GROUP_OTHER]

    def test_mixed_source_record_groups_correctly(self) -> None:
        from app.data.synthetic import make_mixed_disturbance_record
        result = make_mixed_disturbance_record()
        groups = group_channels_for_display(
            result.record, signal_metadata=result.signal_metadata
        )
        assert DISPLAY_GROUP_VOLTAGE_RAW in groups
        assert DISPLAY_GROUP_CURRENT_RAW in groups
        assert DISPLAY_GROUP_POWER in groups
        assert DISPLAY_GROUP_FREQUENCY in groups
