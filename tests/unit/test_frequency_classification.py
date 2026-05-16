"""Unit tests for app/analytics/frequency/frequency_overlay.py

Tests cover:
  - Frequency channel name heuristics → FREQUENCY role
  - ROCOF channel name heuristics → ROCOF role
  - Waveform / power channel names → UNKNOWN role
  - Unit-based classification: Hz → FREQUENCY, Hz/s → ROCOF
  - SignalMetadata.electrical_type override
  - SignalMetadata.measurement_kind override
  - force_role operator override always wins
  - Display units: FREQUENCY → "Hz", ROCOF → "Hz/s", UNKNOWN → None
  - is_frequency_channel / is_rocof_channel convenience helpers
"""
from __future__ import annotations

import pytest

from app.analytics.frequency.frequency_models import FrequencyChannelRole
from app.analytics.frequency.frequency_overlay import (
    classify_frequency_role,
    is_frequency_channel,
    is_rocof_channel,
)
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _meta(**kwargs) -> SignalMetadata:
    return SignalMetadata(name="ch", **kwargs)


def _freq(name: str, **kw) -> FrequencyChannelRole:
    return classify_frequency_role(name, **kw).role


# ─────────────────────────────────────────────────────────────────────────────
# TestFrequencyNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestFrequencyNameHeuristics:
    @pytest.mark.parametrize(
        "name",
        [
            "Frequency",
            "frequency",
            "FREQUENCY",
            "SysFreq",
            "sysfreq",
            "system frequency",
            "BusFreq",
            "PMUFreq",
            "gridfreq",
            "GridFrequency",
            "f",
            "F",
            "Hz",
            "hz",
            "freq",
            "FREQ",
        ],
    )
    def test_frequency_names_classified_as_frequency(self, name: str) -> None:
        result = classify_frequency_role(name)
        assert result.role == FrequencyChannelRole.FREQUENCY, (
            f"Expected FREQUENCY for {name!r}, got {result.role} ({result.reason})"
        )
        assert result.auto_classified

    @pytest.mark.parametrize(
        "name",
        [
            "ROCOF",
            "rocof",
            "Rocof",
            "df/dt",
            "dfdt",
            "DFDT",
            "df_dt",
            "DF_DT",
            "Hz/s",
            "hz/s",
            "hzpersec",
        ],
    )
    def test_rocof_names_not_classified_as_frequency(self, name: str) -> None:
        result = classify_frequency_role(name)
        assert result.role != FrequencyChannelRole.FREQUENCY, (
            f"Expected non-FREQUENCY for {name!r}, got {result.role}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestROCOFNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestROCOFNameHeuristics:
    @pytest.mark.parametrize(
        "name",
        [
            "ROCOF",
            "rocof",
            "Rocof",
            "df/dt",
            "dfdt",
            "DFDT",
            "df_dt",
            "DF_DT",
            "Hz/s",
            "hz/s",
            "hzpersec",
            "RateOfChange",
            "rate_of_change",
        ],
    )
    def test_rocof_names_classified_as_rocof(self, name: str) -> None:
        result = classify_frequency_role(name)
        assert result.role == FrequencyChannelRole.ROCOF, (
            f"Expected ROCOF for {name!r}, got {result.role} ({result.reason})"
        )
        assert result.auto_classified

    @pytest.mark.parametrize(
        "name",
        [
            "Frequency",
            "freq",
            "SysFreq",
            "f",
        ],
    )
    def test_frequency_names_not_classified_as_rocof(self, name: str) -> None:
        result = classify_frequency_role(name)
        assert result.role != FrequencyChannelRole.ROCOF, (
            f"Expected non-ROCOF for {name!r}, got {result.role}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestUnknownChannels
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownChannels:
    @pytest.mark.parametrize(
        "name",
        [
            "VA",
            "VB",
            "VC",
            "Ia",
            "Ib",
            "Ic",
            "MW",
            "MVar",
            "KPDN1_VA",
            "SGT1_IB_HV",
            "ActivePower",
            "ReactivePower",
            "Voltage",
            "Current",
            "Unknown_Channel",
            "3I0",
            "3U0",
        ],
    )
    def test_non_frequency_channels_classified_as_unknown(self, name: str) -> None:
        result = classify_frequency_role(name)
        assert result.role == FrequencyChannelRole.UNKNOWN, (
            f"Expected UNKNOWN for {name!r}, got {result.role} ({result.reason})"
        )
        assert result.auto_classified


# ─────────────────────────────────────────────────────────────────────────────
# TestUnitBasedClassification
# ─────────────────────────────────────────────────────────────────────────────


class TestUnitBasedClassification:
    @pytest.mark.parametrize("unit", ["Hz", "hz", "hertz", "Hertz"])
    def test_hz_unit_gives_frequency_role(self, unit: str) -> None:
        result = classify_frequency_role("Channel", unit=unit)
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert "unit:" in result.reason

    @pytest.mark.parametrize(
        "unit",
        ["Hz/s", "hz/s", "Hz/sec", "HzPerSecond", "hz_s", "df/dt", "dfdt", "df_dt"],
    )
    def test_rocof_unit_gives_rocof_role(self, unit: str) -> None:
        result = classify_frequency_role("Channel", unit=unit)
        assert result.role == FrequencyChannelRole.ROCOF
        assert "unit:" in result.reason

    def test_unit_takes_priority_over_ambiguous_name(self) -> None:
        # Name is generic "Channel" → would be UNKNOWN without unit
        result_no_unit = classify_frequency_role("Channel")
        assert result_no_unit.role == FrequencyChannelRole.UNKNOWN

        result_with_hz = classify_frequency_role("Channel", unit="Hz")
        assert result_with_hz.role == FrequencyChannelRole.FREQUENCY

    def test_kv_unit_gives_unknown(self) -> None:
        result = classify_frequency_role("Channel", unit="kV")
        assert result.role == FrequencyChannelRole.UNKNOWN

    def test_mw_unit_gives_unknown(self) -> None:
        result = classify_frequency_role("Channel", unit="MW")
        assert result.role == FrequencyChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestMetadataOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestMetadataOverride:
    def test_electrical_type_frequency_gives_frequency_role(self) -> None:
        meta = _meta(electrical_type="frequency")
        result = classify_frequency_role("VA", signal_meta=meta)
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert "electrical_type" in result.reason

    def test_electrical_type_rocof_gives_rocof_role(self) -> None:
        meta = _meta(electrical_type="rocof")
        result = classify_frequency_role("VA", signal_meta=meta)
        assert result.role == FrequencyChannelRole.ROCOF
        assert "electrical_type" in result.reason

    def test_measurement_kind_frequency_gives_frequency_role(self) -> None:
        meta = _meta(measurement_kind="frequency")
        result = classify_frequency_role("VA", signal_meta=meta)
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert "measurement_kind" in result.reason

    def test_measurement_kind_rocof_gives_rocof_role(self) -> None:
        meta = _meta(measurement_kind="rocof")
        result = classify_frequency_role("VA", signal_meta=meta)
        assert result.role == FrequencyChannelRole.ROCOF
        assert "measurement_kind" in result.reason

    def test_measurement_kind_takes_priority_over_electrical_type(self) -> None:
        # measurement_kind is checked before electrical_type in the priority chain
        meta = _meta(measurement_kind="frequency", electrical_type="rocof")
        result = classify_frequency_role("VA", signal_meta=meta)
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert "measurement_kind" in result.reason

    def test_electrical_type_voltage_gives_unknown(self) -> None:
        meta = _meta(electrical_type="voltage")
        result = classify_frequency_role("VA", signal_meta=meta)
        # No match in frequency/rocof tables → falls through to name heuristics
        # "VA" doesn't match frequency fragments → UNKNOWN
        assert result.role == FrequencyChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestForceOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestForceOverride:
    def test_force_frequency_overrides_rocof_name(self) -> None:
        result = classify_frequency_role(
            "ROCOF", force_role=FrequencyChannelRole.FREQUENCY
        )
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert result.reason == "operator_override"
        assert not result.auto_classified

    def test_force_rocof_overrides_frequency_name(self) -> None:
        result = classify_frequency_role(
            "Frequency", force_role=FrequencyChannelRole.ROCOF
        )
        assert result.role == FrequencyChannelRole.ROCOF
        assert not result.auto_classified

    def test_force_unknown_overrides_frequency_name(self) -> None:
        result = classify_frequency_role(
            "Frequency", force_role=FrequencyChannelRole.UNKNOWN
        )
        assert result.role == FrequencyChannelRole.UNKNOWN
        assert not result.auto_classified

    def test_force_overrides_metadata(self) -> None:
        meta = _meta(electrical_type="frequency", measurement_kind="frequency")
        result = classify_frequency_role(
            "Frequency", signal_meta=meta, force_role=FrequencyChannelRole.ROCOF
        )
        assert result.role == FrequencyChannelRole.ROCOF
        assert not result.auto_classified


# ─────────────────────────────────────────────────────────────────────────────
# TestDisplayUnits
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayUnits:
    def test_frequency_role_display_unit_is_hz(self) -> None:
        result = classify_frequency_role("Frequency")
        assert result.role == FrequencyChannelRole.FREQUENCY
        assert result.display_unit == "Hz"

    def test_rocof_role_display_unit_is_hz_per_s(self) -> None:
        result = classify_frequency_role("ROCOF")
        assert result.role == FrequencyChannelRole.ROCOF
        assert result.display_unit == "Hz/s"

    def test_unknown_role_display_unit_is_none(self) -> None:
        result = classify_frequency_role("VA")
        assert result.role == FrequencyChannelRole.UNKNOWN
        assert result.display_unit is None

    def test_frequency_display_unit_never_khz(self) -> None:
        result = classify_frequency_role("Frequency")
        assert result.display_unit != "kHz"
        assert result.display_unit != "MHz"


# ─────────────────────────────────────────────────────────────────────────────
# TestConvenienceHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestConvenienceHelpers:
    def test_is_frequency_channel_true(self) -> None:
        assert is_frequency_channel("SysFreq") is True

    def test_is_frequency_channel_false_for_rocof(self) -> None:
        assert is_frequency_channel("ROCOF") is False

    def test_is_frequency_channel_false_for_waveform(self) -> None:
        assert is_frequency_channel("VA") is False

    def test_is_rocof_channel_true(self) -> None:
        assert is_rocof_channel("ROCOF") is True

    def test_is_rocof_channel_true_for_dfdt(self) -> None:
        assert is_rocof_channel("dfdt") is True

    def test_is_rocof_channel_false_for_frequency(self) -> None:
        assert is_rocof_channel("Frequency") is False

    def test_is_rocof_channel_false_for_waveform(self) -> None:
        assert is_rocof_channel("VA") is False

    def test_unit_hz_triggers_is_frequency(self) -> None:
        assert is_frequency_channel("Channel", unit="Hz") is True

    def test_unit_hzs_triggers_is_rocof(self) -> None:
        assert is_rocof_channel("Channel", unit="Hz/s") is True
