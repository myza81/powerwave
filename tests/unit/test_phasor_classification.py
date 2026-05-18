"""Unit tests for phasor channel classification and phase identification (Phase 6A).

Tests cover:
  - Voltage channel name heuristics → VOLTAGE_PHASOR
  - Current channel name heuristics → CURRENT_PHASOR
  - Unknown channels → UNKNOWN
  - Unit-based classification: kV/V → voltage, A/kA → current
  - SignalMetadata.electrical_type override
  - SignalMetadata.measurement_kind override
  - force_role operator override always wins
  - Phase identification: A/B/C from name suffix
  - Phase identification: R/Y → A/B mapping
  - Phase identification: operator override (force_phase)
  - Phase identification: AnalogChannel.phase field priority
  - Three-phase group detection: complete groups returned
  - Three-phase group detection: incomplete groups excluded
  - detect_three_phase_groups with mixed voltage + current channels
  - is_voltage_channel / is_current_channel convenience helpers
"""
from __future__ import annotations

import pytest

from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorChannelRole,
)
from app.analytics.phasors.phasor_overlay import (
    classify_phasor_role,
    detect_three_phase_groups,
    identify_phase,
    is_current_channel,
    is_voltage_channel,
)
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _meta(**kwargs) -> SignalMetadata:
    return SignalMetadata(name="ch", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TestVoltageNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestVoltageNameHeuristics:
    @pytest.mark.parametrize(
        "name",
        [
            "VA", "VB", "VC",
            "Va", "Vb", "Vc",
            "VAN", "VBN", "VCN",
            "VR", "VY",
            "Voltage",
            "voltage",
            "VOLTAGE",
            "VoltageA",
            "Volt_A",
        ],
    )
    def test_voltage_names_classified_as_voltage(self, name: str) -> None:
        result = classify_phasor_role(name)
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR, (
            f"Expected VOLTAGE_PHASOR for {name!r}, got {result.role}"
        )
        assert result.auto_classified

    @pytest.mark.parametrize(
        "name",
        ["IA", "IB", "IC", "rocof", "MW", "frequency", "ROCOF"],
    )
    def test_non_voltage_names_not_classified_as_voltage(self, name: str) -> None:
        result = classify_phasor_role(name)
        assert result.role != PhasorChannelRole.VOLTAGE_PHASOR, (
            f"Expected non-VOLTAGE_PHASOR for {name!r}, got {result.role}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestCurrentNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestCurrentNameHeuristics:
    @pytest.mark.parametrize(
        "name",
        [
            "IA", "IB", "IC",
            "Ia", "Ib", "Ic",
            "IR", "IY",
            "Current",
            "current",
            "CURRENT",
            "CurrentA",
            "I_A",
        ],
    )
    def test_current_names_classified_as_current(self, name: str) -> None:
        result = classify_phasor_role(name)
        assert result.role == PhasorChannelRole.CURRENT_PHASOR, (
            f"Expected CURRENT_PHASOR for {name!r}, got {result.role}"
        )

    @pytest.mark.parametrize(
        "name",
        ["VA", "VB", "VC", "rocof", "MW", "frequency"],
    )
    def test_non_current_names_not_classified_as_current(self, name: str) -> None:
        result = classify_phasor_role(name)
        assert result.role != PhasorChannelRole.CURRENT_PHASOR


# ─────────────────────────────────────────────────────────────────────────────
# TestUnknownChannels
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownChannels:
    @pytest.mark.parametrize(
        "name",
        [
            "MW", "MVar",
            "frequency", "freq", "ROCOF",
            "Power", "ActivePower",
            "Unknown_Channel",
            "3I0", "3U0",
        ],
    )
    def test_unknown_channels_classified_as_unknown(self, name: str) -> None:
        result = classify_phasor_role(name)
        assert result.role == PhasorChannelRole.UNKNOWN, (
            f"Expected UNKNOWN for {name!r}, got {result.role}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestUnitBasedClassification
# ─────────────────────────────────────────────────────────────────────────────


class TestUnitBasedClassification:
    @pytest.mark.parametrize("unit", ["kV", "kv", "V", "v", "Volts"])
    def test_voltage_unit_gives_voltage_role(self, unit: str) -> None:
        result = classify_phasor_role("Channel", unit=unit)
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR
        assert "unit:" in result.reason

    @pytest.mark.parametrize("unit", ["A", "a", "kA", "ka", "Amps"])
    def test_current_unit_gives_current_role(self, unit: str) -> None:
        result = classify_phasor_role("Channel", unit=unit)
        assert result.role == PhasorChannelRole.CURRENT_PHASOR
        assert "unit:" in result.reason

    def test_unknown_unit_falls_back_to_name(self) -> None:
        result = classify_phasor_role("MW", unit="MW")
        assert result.role == PhasorChannelRole.UNKNOWN

    def test_unit_kv_gives_voltage_display_unit_kv(self) -> None:
        result = classify_phasor_role("Channel", unit="kV")
        assert result.display_unit == "kV"


# ─────────────────────────────────────────────────────────────────────────────
# TestMetadataOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestMetadataOverride:
    def test_electrical_type_voltage_gives_voltage_role(self) -> None:
        meta = _meta(electrical_type="voltage")
        result = classify_phasor_role("IA", signal_meta=meta)
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR
        assert "electrical_type" in result.reason

    def test_electrical_type_current_gives_current_role(self) -> None:
        meta = _meta(electrical_type="current")
        result = classify_phasor_role("VA", signal_meta=meta)
        assert result.role == PhasorChannelRole.CURRENT_PHASOR
        assert "electrical_type" in result.reason

    def test_measurement_kind_voltage_phasor(self) -> None:
        meta = _meta(measurement_kind="voltage_phasor")
        result = classify_phasor_role("Channel", signal_meta=meta)
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR
        assert "measurement_kind" in result.reason

    def test_measurement_kind_current_phasor(self) -> None:
        meta = _meta(measurement_kind="current_phasor")
        result = classify_phasor_role("Channel", signal_meta=meta)
        assert result.role == PhasorChannelRole.CURRENT_PHASOR

    def test_measurement_kind_takes_priority_over_electrical_type(self) -> None:
        meta = _meta(measurement_kind="voltage_phasor", electrical_type="current")
        result = classify_phasor_role("Channel", signal_meta=meta)
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR
        assert "measurement_kind" in result.reason

    def test_electrical_type_frequency_gives_unknown(self) -> None:
        meta = _meta(electrical_type="frequency")
        result = classify_phasor_role("Channel", signal_meta=meta)
        assert result.role == PhasorChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestForceOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestForceOverride:
    def test_force_voltage_overrides_current_name(self) -> None:
        result = classify_phasor_role(
            "IA", force_role=PhasorChannelRole.VOLTAGE_PHASOR
        )
        assert result.role == PhasorChannelRole.VOLTAGE_PHASOR
        assert result.reason == "operator_override"
        assert not result.auto_classified

    def test_force_current_overrides_voltage_name(self) -> None:
        result = classify_phasor_role(
            "VA", force_role=PhasorChannelRole.CURRENT_PHASOR
        )
        assert result.role == PhasorChannelRole.CURRENT_PHASOR
        assert not result.auto_classified

    def test_force_unknown_overrides_voltage_name(self) -> None:
        result = classify_phasor_role(
            "VA", force_role=PhasorChannelRole.UNKNOWN
        )
        assert result.role == PhasorChannelRole.UNKNOWN
        assert not result.auto_classified

    def test_force_overrides_metadata(self) -> None:
        meta = _meta(electrical_type="voltage", measurement_kind="voltage_phasor")
        result = classify_phasor_role(
            "VA", signal_meta=meta, force_role=PhasorChannelRole.CURRENT_PHASOR
        )
        assert result.role == PhasorChannelRole.CURRENT_PHASOR


# ─────────────────────────────────────────────────────────────────────────────
# TestPhaseIdentification
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseIdentification:
    @pytest.mark.parametrize(
        "name",
        ["VA", "Ia", "VAN", "VoltageA", "V_A", "IR", "VR"],
    )
    def test_phase_a_from_suffix(self, name: str) -> None:
        phase = identify_phase(name)
        assert phase == PhaseLabel.A, (
            f"Expected Phase A for {name!r}, got {phase}"
        )

    @pytest.mark.parametrize(
        "name",
        ["VB", "Ib", "VBN", "VoltageB", "V_B", "IY", "VY"],
    )
    def test_phase_b_from_suffix(self, name: str) -> None:
        phase = identify_phase(name)
        assert phase == PhaseLabel.B, (
            f"Expected Phase B for {name!r}, got {phase}"
        )

    @pytest.mark.parametrize(
        "name",
        ["VC", "Ic", "VCN", "VoltageC", "V_C"],
    )
    def test_phase_c_from_suffix(self, name: str) -> None:
        phase = identify_phase(name)
        assert phase == PhaseLabel.C, (
            f"Expected Phase C for {name!r}, got {phase}"
        )

    def test_unknown_channel_returns_unknown_phase(self) -> None:
        phase = identify_phase("Channel_XY")
        assert phase == PhaseLabel.UNKNOWN

    def test_force_phase_a_overrides_name(self) -> None:
        phase = identify_phase("VC", force_phase=PhaseLabel.A)
        assert phase == PhaseLabel.A

    def test_channel_phase_field_takes_priority(self) -> None:
        # AnalogChannel.phase='C' but name suggests A
        phase = identify_phase("VA", channel_phase="C")
        assert phase == PhaseLabel.C

    def test_ryb_mapping_r_is_a(self) -> None:
        phase = identify_phase("VR")
        assert phase == PhaseLabel.A

    def test_ryb_mapping_y_is_b(self) -> None:
        phase = identify_phase("VY")
        assert phase == PhaseLabel.B


# ─────────────────────────────────────────────────────────────────────────────
# TestThreePhaseGroupDetection
# ─────────────────────────────────────────────────────────────────────────────


class TestThreePhaseGroupDetection:
    def test_abc_voltage_group_detected(self) -> None:
        names = ["VA", "VB", "VC"]
        groups = detect_three_phase_groups(names)
        assert len(groups) == 1
        g = groups[0]
        assert g["a"] == "VA"
        assert g["b"] == "VB"
        assert g["c"] == "VC"
        assert g["signal_type"] == "voltage"

    def test_abc_current_group_detected(self) -> None:
        names = ["IA", "IB", "IC"]
        groups = detect_three_phase_groups(names)
        assert len(groups) == 1
        assert groups[0]["signal_type"] == "current"

    def test_incomplete_group_not_returned(self) -> None:
        # Only VA and VB, missing VC
        names = ["VA", "VB"]
        groups = detect_three_phase_groups(names)
        assert groups == []

    def test_mixed_voltage_and_current_two_groups(self) -> None:
        names = ["VA", "VB", "VC", "IA", "IB", "IC"]
        groups = detect_three_phase_groups(names)
        types = {g["signal_type"] for g in groups}
        assert "voltage" in types
        assert "current" in types

    def test_extra_channels_ignored(self) -> None:
        names = ["VA", "VB", "VC", "MW", "MVar", "frequency"]
        groups = detect_three_phase_groups(names)
        assert len(groups) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert detect_three_phase_groups([]) == []

    def test_waveform_only_no_groups(self) -> None:
        # Channels that don't have clear phase labels
        names = ["Channel1", "Channel2", "Channel3"]
        groups = detect_three_phase_groups(names)
        assert groups == []


# ─────────────────────────────────────────────────────────────────────────────
# TestConvenienceHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestConvenienceHelpers:
    def test_is_voltage_channel_true(self) -> None:
        assert is_voltage_channel("VA") is True

    def test_is_voltage_channel_false_for_current(self) -> None:
        assert is_voltage_channel("IA") is False

    def test_is_voltage_channel_false_for_unknown(self) -> None:
        assert is_voltage_channel("MW") is False

    def test_is_current_channel_true(self) -> None:
        assert is_current_channel("IA") is True

    def test_is_current_channel_false_for_voltage(self) -> None:
        assert is_current_channel("VA") is False

    def test_unit_kv_triggers_is_voltage(self) -> None:
        assert is_voltage_channel("Channel", unit="kV") is True

    def test_unit_a_triggers_is_current(self) -> None:
        assert is_current_channel("Channel", unit="A") is True
