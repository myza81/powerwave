"""Unit tests for frequency/ROCOF visualization integration.

Tests cover:
  - Channel grouper routes frequency/ROCOF to correct display groups
  - RMS eligibility: frequency/ROCOF channels are always ineligible
  - FrequencyRegistry bulk helpers with DisturbanceRecord analog channels
  - Timestamp mode independence: frequency channels work with any time mode
  - Signal visibility: default visibility policy includes frequency channels
  - Multi-source frequency/ROCOF panel key detection
  - FrequencyConfig nominal Hz validation
  - classify_rocof from rocof_overlay (force/non-force)
  - Synchronization: frequency channels preserved after mode switches
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analytics.frequency.frequency_models import (
    FrequencyChannelRole,
    FrequencyConfig,
    FrequencyDisplayMode,
)
from app.analytics.frequency.frequency_registry import FrequencyRegistry
from app.analytics.frequency.rocof_overlay import classify_rocof
from app.analytics.rms.rms_overlay import classify_rms_eligibility
from app.data.signal_metadata import SignalMetadata
from app.visualization.channel_grouper import (
    DISPLAY_GROUP_FREQUENCY,
    DISPLAY_GROUP_ROCOF,
    group_channels_for_display,
)
from app.visualization.signal_visibility import default_visible_analog_names


# ─────────────────────────────────────────────────────────────────────────────
# Minimal test record builder
# ─────────────────────────────────────────────────────────────────────────────


def _make_minimal_record(channel_names: list[str]):
    """Build a minimal DisturbanceRecord with the specified analog channel names."""
    from datetime import datetime
    import pandas as pd
    from app.models import AnalogChannel, DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    n = 100
    t = np.linspace(0.0, 1.0, n)
    data = {"time": t}
    channels = []
    for i, name in enumerate(channel_names):
        data[name] = np.zeros(n, dtype=np.float64)
        channels.append(AnalogChannel(name=name, index=i, unit=""))

    _t = datetime(2026, 1, 1)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST", recorder_name="TEST",
            source_file="test.csv", provider_type="csv",
            nominal_frequency=50.0,
        ),
        waveform_data=pd.DataFrame(data),
        analog_channels=channels,
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[100.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=_t, trigger_time=_t),
        disturbance_info=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestChannelGrouperFrequency
# ─────────────────────────────────────────────────────────────────────────────


class TestChannelGrouperFrequency:
    def test_frequency_name_goes_to_frequency_group(self) -> None:
        record = _make_minimal_record(["frequency"])
        groups = group_channels_for_display(record)
        assert "frequency" in groups.get(DISPLAY_GROUP_FREQUENCY, [])

    def test_freq_name_goes_to_frequency_group(self) -> None:
        record = _make_minimal_record(["freq"])
        groups = group_channels_for_display(record)
        assert "freq" in groups.get(DISPLAY_GROUP_FREQUENCY, [])

    def test_f_lowercase_goes_to_frequency_group(self) -> None:
        record = _make_minimal_record(["f"])
        groups = group_channels_for_display(record)
        assert "f" in groups.get(DISPLAY_GROUP_FREQUENCY, [])

    def test_rocof_name_goes_to_rocof_group(self) -> None:
        record = _make_minimal_record(["rocof"])
        groups = group_channels_for_display(record)
        assert "rocof" in groups.get(DISPLAY_GROUP_ROCOF, [])

    def test_dfdt_name_goes_to_rocof_group(self) -> None:
        record = _make_minimal_record(["dfdt"])
        groups = group_channels_for_display(record)
        assert "dfdt" in groups.get(DISPLAY_GROUP_ROCOF, [])

    def test_frequency_and_rocof_in_separate_groups(self) -> None:
        record = _make_minimal_record(["frequency", "rocof"])
        groups = group_channels_for_display(record)
        freq_channels = groups.get(DISPLAY_GROUP_FREQUENCY, [])
        rocof_channels = groups.get(DISPLAY_GROUP_ROCOF, [])
        assert "frequency" in freq_channels
        assert "rocof" in rocof_channels
        # Must be in separate groups
        assert not (set(freq_channels) & set(rocof_channels))

    def test_signal_metadata_display_group_overrides_heuristic(self) -> None:
        record = _make_minimal_record(["MyChannel"])
        meta = {"MyChannel": SignalMetadata(name="MyChannel", display_group="frequency")}
        groups = group_channels_for_display(record, signal_metadata=meta)
        assert "MyChannel" in groups.get(DISPLAY_GROUP_FREQUENCY, [])

    def test_waveform_channels_not_in_frequency_group(self) -> None:
        record = _make_minimal_record(["VA", "VB", "Ia"])
        groups = group_channels_for_display(record)
        freq_channels = groups.get(DISPLAY_GROUP_FREQUENCY, [])
        for name in ["VA", "VB", "Ia"]:
            assert name not in freq_channels

    def test_power_channels_not_in_frequency_group(self) -> None:
        record = _make_minimal_record(["MW", "MVar"])
        groups = group_channels_for_display(record)
        freq_channels = groups.get(DISPLAY_GROUP_FREQUENCY, [])
        assert "MW" not in freq_channels
        assert "MVar" not in freq_channels


# ─────────────────────────────────────────────────────────────────────────────
# TestRMSIneligibility
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSIneligibility:
    @pytest.mark.parametrize(
        "name",
        [
            "frequency",
            "Frequency",
            "SysFreq",
            "freq",
            "FREQ",
            "Hz",
        ],
    )
    def test_frequency_channels_rms_ineligible(self, name: str) -> None:
        result = classify_rms_eligibility(name)
        assert not result.eligible, (
            f"Expected frequency channel {name!r} to be RMS-ineligible"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "rocof",
            "ROCOF",
            "dfdt",
            "df/dt",
            "df_dt",
        ],
    )
    def test_rocof_channels_rms_ineligible(self, name: str) -> None:
        result = classify_rms_eligibility(name)
        assert not result.eligible, (
            f"Expected ROCOF channel {name!r} to be RMS-ineligible"
        )

    def test_frequency_electrical_type_rms_ineligible(self) -> None:
        meta = SignalMetadata(name="ch", electrical_type="frequency")
        result = classify_rms_eligibility("ch", signal_meta=meta)
        assert not result.eligible

    def test_rocof_electrical_type_rms_ineligible(self) -> None:
        meta = SignalMetadata(name="ch", electrical_type="rocof")
        result = classify_rms_eligibility("ch", signal_meta=meta)
        assert not result.eligible


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyRegistryWithRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyRegistryWithRecord:
    def test_frequency_channels_extracted_from_record(self) -> None:
        record = _make_minimal_record(["VA", "frequency", "MW", "ROCOF"])
        r = FrequencyRegistry()
        names = [ch.name for ch in record.analog_channels]
        freq = r.frequency_channels(names)
        assert freq == ["frequency"]

    def test_rocof_channels_extracted_from_record(self) -> None:
        record = _make_minimal_record(["VA", "frequency", "MW", "ROCOF"])
        r = FrequencyRegistry()
        names = [ch.name for ch in record.analog_channels]
        rocof = r.rocof_channels(names)
        assert rocof == ["ROCOF"]

    def test_no_frequency_channels_in_waveform_record(self) -> None:
        record = _make_minimal_record(["VA", "VB", "VC", "Ia", "Ib", "Ic"])
        r = FrequencyRegistry()
        names = [ch.name for ch in record.analog_channels]
        assert r.frequency_channels(names) == []
        assert r.rocof_channels(names) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyConfig:
    def test_default_nominal_hz(self) -> None:
        cfg = FrequencyConfig()
        assert cfg.nominal_hz == 50.0

    def test_60hz_system(self) -> None:
        cfg = FrequencyConfig(nominal_hz=60.0)
        assert cfg.nominal_hz == 60.0

    def test_registry_uses_config(self) -> None:
        cfg = FrequencyConfig(nominal_hz=60.0)
        r = FrequencyRegistry(config=cfg)
        assert r.config.nominal_hz == 60.0

    def test_registry_set_config(self) -> None:
        r = FrequencyRegistry()
        r.set_config(FrequencyConfig(nominal_hz=60.0))
        assert r.config.nominal_hz == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyRocofHelper
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyRocofHelper:
    def test_rocof_channel_returns_rocof_role(self) -> None:
        result = classify_rocof("ROCOF")
        assert result.role == FrequencyChannelRole.ROCOF

    def test_frequency_channel_returns_unknown(self) -> None:
        result = classify_rocof("SysFreq")
        assert result.role == FrequencyChannelRole.UNKNOWN

    def test_force_rocof_overrides_frequency(self) -> None:
        result = classify_rocof("SysFreq", force=True)
        assert result.role == FrequencyChannelRole.ROCOF
        assert not result.auto_classified

    def test_waveform_channel_returns_unknown(self) -> None:
        result = classify_rocof("VA")
        assert result.role == FrequencyChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestSignalVisibilityPreservation
# ─────────────────────────────────────────────────────────────────────────────


class TestSignalVisibilityPreservation:
    def test_frequency_channels_included_in_default_visible(self) -> None:
        record = _make_minimal_record(["VA", "VB", "frequency", "rocof"])
        channels = record.analog_channels
        visible = default_visible_analog_names(channels)
        # All 4 channels fit within the default limit (8) — all visible
        assert "frequency" in visible
        assert "rocof" in visible

    def test_frequency_channels_visible_in_small_record(self) -> None:
        record = _make_minimal_record(["frequency", "ROCOF"])
        visible = default_visible_analog_names(record.analog_channels)
        assert "frequency" in visible
        assert "ROCOF" in visible

    def test_frequency_registry_display_mode_off_does_not_affect_classification(
        self,
    ) -> None:
        r = FrequencyRegistry()
        r.set_display_mode(FrequencyDisplayMode.OFF)
        # Display mode OFF means panels are hidden — it does not change
        # channel role classification
        assert r.is_frequency("SysFreq") is True
        assert r.is_rocof("ROCOF") is True

    def test_display_mode_can_be_cycled(self) -> None:
        r = FrequencyRegistry()
        for mode in FrequencyDisplayMode:
            r.set_display_mode(mode)
            assert r.display_mode == mode


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiSourcePanelKeyDetection
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSourcePanelKeyDetection:
    def test_mixed_panel_keys_frequency_only(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys([
            "COMTRADE/voltage_raw",
            "COMTRADE/current_raw",
            "CSV/frequency",
            "CSV/power",
        ])
        assert keys == ["CSV/frequency"]

    def test_mixed_panel_keys_rocof_only(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys([
            "COMTRADE/voltage_raw",
            "CSV/rocof",
            "CSV/power",
        ])
        assert keys == ["CSV/rocof"]

    def test_direct_and_multi_source_combined(self) -> None:
        r = FrequencyRegistry()
        keys = r.frequency_panel_keys([
            "voltage_raw",
            "frequency",
            "CSV/frequency",
            "CSV/rocof",
            "rocof",
            "power",
        ])
        assert set(keys) == {"frequency", "rocof", "CSV/frequency", "CSV/rocof"}

    def test_empty_panel_keys(self) -> None:
        r = FrequencyRegistry()
        assert r.frequency_panel_keys([]) == []
