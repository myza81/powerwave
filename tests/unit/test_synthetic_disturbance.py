"""Unit tests for app.data.synthetic disturbance generators."""
from __future__ import annotations

import numpy as np
import pytest

from app.data.synthetic import (
    make_high_rate_record,
    make_low_rate_record,
    make_mixed_disturbance_record,
)
from app.models import DisturbanceRecord


class TestHighRateRecord:
    def test_returns_disturbance_record(self) -> None:
        result = make_high_rate_record()
        assert isinstance(result.record, DisturbanceRecord)

    def test_has_voltage_channels(self) -> None:
        result = make_high_rate_record()
        names = {ch.name for ch in result.record.analog_channels}
        assert {"VA_raw", "VB_raw", "VC_raw"}.issubset(names)

    def test_has_current_channels(self) -> None:
        result = make_high_rate_record(include_current=True)
        names = {ch.name for ch in result.record.analog_channels}
        assert {"IA_raw", "IB_raw", "IC_raw"}.issubset(names)

    def test_no_current_channels_when_excluded(self) -> None:
        result = make_high_rate_record(include_current=False)
        names = {ch.name for ch in result.record.analog_channels}
        assert "IA_raw" not in names

    def test_sampling_rate_recorded(self) -> None:
        result = make_high_rate_record(sampling_rate_hz=6400.0)
        assert result.record.sampling_info.sampling_rates[0] == 6400.0

    def test_fault_creates_voltage_sag(self) -> None:
        result = make_high_rate_record(
            fault_start_s=1.0, fault_end_s=1.5, fault_voltage_pu=0.3
        )
        df = result.record.waveform_data
        t = df["time"].to_numpy()
        va = df["VA_raw"].to_numpy()
        pre_std = np.std(va[t < 1.0])
        fault_std = np.std(va[(t >= 1.0) & (t < 1.5)])
        assert fault_std < pre_std * 0.5

    def test_signal_metadata_voltage_display_group(self) -> None:
        result = make_high_rate_record()
        assert result.signal_metadata["VA_raw"].display_group == "voltage_raw"
        assert result.signal_metadata["VB_raw"].display_group == "voltage_raw"

    def test_signal_metadata_current_display_group(self) -> None:
        result = make_high_rate_record()
        assert result.signal_metadata["IA_raw"].display_group == "current_raw"

    def test_validate_returns_no_errors(self) -> None:
        result = make_high_rate_record()
        errors = result.record.validate()
        assert errors == []

    def test_time_column_monotonic(self) -> None:
        result = make_high_rate_record()
        t = result.record.waveform_data["time"].to_numpy()
        assert np.all(np.diff(t) > 0)


class TestLowRateRecord:
    def test_has_mw_mvar_frequency(self) -> None:
        result = make_low_rate_record()
        names = {ch.name for ch in result.record.analog_channels}
        assert {"MW", "MVar", "Frequency"}.issubset(names)

    def test_sampling_rate_recorded(self) -> None:
        result = make_low_rate_record(sampling_rate_hz=100.0)
        assert result.record.sampling_info.sampling_rates[0] == 100.0

    def test_time_offset_applied(self) -> None:
        result = make_low_rate_record(time_offset_s=0.01)
        t0 = result.record.waveform_data["time"].iloc[0]
        assert t0 > 0.0

    def test_frequency_dip_during_fault(self) -> None:
        result = make_low_rate_record(fault_start_s=1.0, fault_end_s=1.5)
        df = result.record.waveform_data
        t = df["time"].to_numpy()
        freq = df["Frequency"].to_numpy()
        pre_freq = np.mean(freq[t < 1.0])
        fault_freq = np.mean(freq[(t >= 1.0) & (t < 1.5)])
        assert fault_freq < pre_freq

    def test_rocof_included_when_requested(self) -> None:
        result = make_low_rate_record(include_rocof=True)
        names = {ch.name for ch in result.record.analog_channels}
        assert "ROCOF" in names
        assert result.signal_metadata["ROCOF"].display_group == "rocof"

    def test_validate_returns_no_errors(self) -> None:
        result = make_low_rate_record()
        errors = result.record.validate()
        assert errors == []


class TestMixedDisturbanceRecord:
    def test_has_voltage_and_system_channels(self) -> None:
        result = make_mixed_disturbance_record()
        names = {ch.name for ch in result.record.analog_channels}
        assert "VA_raw" in names
        assert "MW" in names
        assert "Frequency" in names

    def test_time_axis_monotonic(self) -> None:
        result = make_mixed_disturbance_record()
        t = result.record.waveform_data["time"].to_numpy()
        assert np.all(np.diff(t) > 0)

    def test_both_sampling_rates_recorded(self) -> None:
        result = make_mixed_disturbance_record(high_rate_hz=6400.0, low_rate_hz=100.0)
        rates = result.record.sampling_info.sampling_rates
        assert 6400.0 in rates
        assert 100.0 in rates

    def test_metadata_covers_all_channels(self) -> None:
        result = make_mixed_disturbance_record()
        for ch in result.record.analog_channels:
            assert ch.name in result.signal_metadata

    def test_validate_returns_no_errors(self) -> None:
        result = make_mixed_disturbance_record()
        errors = result.record.validate()
        assert errors == []

    def test_event_duration_valid(self) -> None:
        result = make_mixed_disturbance_record(duration_s=5.0)
        assert result.record.duration_seconds() > 0.0

    def test_channel_indices_contiguous(self) -> None:
        result = make_mixed_disturbance_record()
        indices = [ch.index for ch in result.record.analog_channels]
        assert indices == list(range(len(indices)))


class TestElectricalReferenceMetadata:
    """Verify electrical_type / phase_reference annotations on synthetic generators."""

    def test_voltage_channels_have_electrical_type(self) -> None:
        result = make_high_rate_record()
        for name in ("VA_raw", "VB_raw", "VC_raw"):
            assert result.signal_metadata[name].electrical_type == "voltage", name

    def test_voltage_channels_have_phase_ground_reference(self) -> None:
        result = make_high_rate_record()
        for name in ("VA_raw", "VB_raw", "VC_raw"):
            assert result.signal_metadata[name].phase_reference == "phase_ground", name

    def test_current_channels_have_electrical_type(self) -> None:
        result = make_high_rate_record(include_current=True)
        for name in ("IA_raw", "IB_raw", "IC_raw"):
            assert result.signal_metadata[name].electrical_type == "current", name

    def test_current_channels_have_no_phase_reference(self) -> None:
        result = make_high_rate_record(include_current=True)
        for name in ("IA_raw", "IB_raw", "IC_raw"):
            assert result.signal_metadata[name].phase_reference is None, name

    def test_mw_mvar_have_power_electrical_type(self) -> None:
        result = make_low_rate_record()
        assert result.signal_metadata["MW"].electrical_type == "power"
        assert result.signal_metadata["MVar"].electrical_type == "power"

    def test_frequency_has_frequency_electrical_type(self) -> None:
        result = make_low_rate_record()
        assert result.signal_metadata["Frequency"].electrical_type == "frequency"

    def test_rocof_has_rocof_electrical_type(self) -> None:
        result = make_low_rate_record(include_rocof=True)
        assert result.signal_metadata["ROCOF"].electrical_type == "rocof"

    def test_nominal_voltage_defaults_none(self) -> None:
        result = make_high_rate_record()
        for name in ("VA_raw", "VB_raw", "VC_raw"):
            assert result.signal_metadata[name].nominal_voltage is None, name

    def test_mixed_record_preserves_electrical_type(self) -> None:
        result = make_mixed_disturbance_record()
        assert result.signal_metadata["VA_raw"].electrical_type == "voltage"
        assert result.signal_metadata["VA_raw"].phase_reference == "phase_ground"
        assert result.signal_metadata["MW"].electrical_type == "power"
        assert result.signal_metadata["Frequency"].electrical_type == "frequency"
