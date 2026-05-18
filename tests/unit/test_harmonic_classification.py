"""Unit tests for harmonic channel classification — Phase 7.

Tests cover:
  - classify_harmonic_role: all priority chain levels
  - is_harmonic_eligible: convenience bool
  - Exclusion rules: RMS, frequency, ROCOF, power, telemetry channels
  - Inclusion rules: voltage and current waveform channels
  - Operator force_role override
  - Name heuristics: V-prefix → voltage, I-prefix → current
  - Unit heuristics: kV/V → voltage, A/kA → current
  - measurement_kind priority over electrical_type over unit over name
  - auto_classified flag correctness
"""
from __future__ import annotations

import pytest

from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelResult,
    HarmonicChannelRole,
)
from app.analytics.harmonics.harmonic_overlay import (
    classify_harmonic_role,
    is_harmonic_eligible,
)
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _meta(**kwargs) -> SignalMetadata:
    return SignalMetadata(name="test", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TestForceRoleOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestForceRoleOverride:
    def test_force_voltage_overrides_everything(self) -> None:
        result = classify_harmonic_role(
            "Freq_01",
            force_role=HarmonicChannelRole.VOLTAGE_HARMONIC,
        )
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC
        assert not result.auto_classified
        assert result.reason == "operator_override"

    def test_force_current_overrides_ineligible_name(self) -> None:
        result = classify_harmonic_role(
            "RMS_voltage",
            force_role=HarmonicChannelRole.CURRENT_HARMONIC,
        )
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC
        assert not result.auto_classified

    def test_force_unknown_results_in_unknown(self) -> None:
        result = classify_harmonic_role(
            "VA",
            force_role=HarmonicChannelRole.UNKNOWN,
        )
        assert result.role == HarmonicChannelRole.UNKNOWN
        assert not result.auto_classified


# ─────────────────────────────────────────────────────────────────────────────
# TestMeasurementKindPriority
# ─────────────────────────────────────────────────────────────────────────────


class TestMeasurementKindPriority:
    def test_rms_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("VA", signal_meta=_meta(measurement_kind="rms"))
        assert result.role == HarmonicChannelRole.UNKNOWN
        assert "measurement_kind:rms" in result.reason

    def test_average_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("VA", signal_meta=_meta(measurement_kind="average"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_telemetry_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("VA", signal_meta=_meta(measurement_kind="telemetry"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_calculated_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("VA", signal_meta=_meta(measurement_kind="calculated"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_frequency_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("Freq", signal_meta=_meta(measurement_kind="frequency"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_voltage_kind_gives_voltage_harmonic(self) -> None:
        result = classify_harmonic_role("X", signal_meta=_meta(measurement_kind="voltage"))
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC
        assert not result.auto_classified

    def test_current_kind_gives_current_harmonic(self) -> None:
        result = classify_harmonic_role("X", signal_meta=_meta(measurement_kind="current"))
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC
        assert not result.auto_classified

    def test_voltage_phasor_kind_is_ineligible(self) -> None:
        result = classify_harmonic_role("X", signal_meta=_meta(measurement_kind="voltage_phasor"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_instantaneous_kind_falls_through_to_next_level(self) -> None:
        # "instantaneous" alone doesn't determine voltage/current — falls through to electrical_type
        result = classify_harmonic_role(
            "VA",
            signal_meta=_meta(measurement_kind="instantaneous", electrical_type="voltage"),
        )
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC


# ─────────────────────────────────────────────────────────────────────────────
# TestElectricalTypePriority
# ─────────────────────────────────────────────────────────────────────────────


class TestElectricalTypePriority:
    def test_voltage_type_gives_voltage_harmonic(self) -> None:
        result = classify_harmonic_role("X", signal_meta=_meta(electrical_type="voltage"))
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC
        assert result.auto_classified
        assert "electrical_type:voltage" in result.reason

    def test_current_type_gives_current_harmonic(self) -> None:
        result = classify_harmonic_role("X", signal_meta=_meta(electrical_type="current"))
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC
        assert "electrical_type:current" in result.reason

    def test_power_type_is_ineligible(self) -> None:
        result = classify_harmonic_role("MW", signal_meta=_meta(electrical_type="power"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_frequency_type_is_ineligible(self) -> None:
        result = classify_harmonic_role("F", signal_meta=_meta(electrical_type="frequency"))
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_rocof_type_is_ineligible(self) -> None:
        result = classify_harmonic_role("ROCOF", signal_meta=_meta(electrical_type="rocof"))
        assert result.role == HarmonicChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestUnitHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestUnitHeuristics:
    @pytest.mark.parametrize("unit", ["kV", "KV", "V", "Volts", "volt"])
    def test_voltage_units(self, unit: str) -> None:
        result = classify_harmonic_role("Ch1", unit=unit)
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC
        assert "unit:" in result.reason

    @pytest.mark.parametrize("unit", ["A", "kA", "Amps", "amp"])
    def test_current_units(self, unit: str) -> None:
        result = classify_harmonic_role("Ch1", unit=unit)
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC
        assert "unit:" in result.reason

    def test_unknown_unit_falls_through_to_name(self) -> None:
        result = classify_harmonic_role("VA", unit="pu")
        # "VA" starts with V → voltage via name heuristic
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC


# ─────────────────────────────────────────────────────────────────────────────
# TestNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestNameHeuristics:
    @pytest.mark.parametrize("name", ["VA", "VB", "VC", "Voltage_A", "v_phase_a"])
    def test_voltage_names(self, name: str) -> None:
        result = classify_harmonic_role(name)
        assert result.role == HarmonicChannelRole.VOLTAGE_HARMONIC, f"Failed for {name!r}"

    @pytest.mark.parametrize("name", ["IA", "IB", "IC", "current_a", "I_phase_b"])
    def test_current_names(self, name: str) -> None:
        result = classify_harmonic_role(name)
        assert result.role == HarmonicChannelRole.CURRENT_HARMONIC, f"Failed for {name!r}"

    def test_rms_name_fragment_is_ineligible(self) -> None:
        result = classify_harmonic_role("VA_RMS")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_freq_name_fragment_is_ineligible(self) -> None:
        result = classify_harmonic_role("Freq_50Hz")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_mw_name_fragment_is_ineligible(self) -> None:
        result = classify_harmonic_role("ActivePower_MW")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_rocof_name_fragment_is_ineligible(self) -> None:
        result = classify_harmonic_role("ROCOF_signal")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_unknown_name_returns_unknown(self) -> None:
        result = classify_harmonic_role("Analog_01")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_single_char_name_returns_unknown(self) -> None:
        # Single-char name → can't apply prefix heuristic (min 2 chars required)
        result = classify_harmonic_role("V")
        assert result.role == HarmonicChannelRole.UNKNOWN

    def test_name_reason_contains_channel_name(self) -> None:
        result = classify_harmonic_role("VA")
        assert "VA" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# TestIsHarmonicEligible
# ─────────────────────────────────────────────────────────────────────────────


class TestIsHarmonicEligible:
    def test_voltage_channel_eligible(self) -> None:
        assert is_harmonic_eligible("VA") is True

    def test_current_channel_eligible(self) -> None:
        assert is_harmonic_eligible("IA") is True

    def test_rms_channel_ineligible(self) -> None:
        assert is_harmonic_eligible("VA_RMS") is False

    def test_frequency_channel_ineligible(self) -> None:
        assert is_harmonic_eligible("Freq") is False

    def test_unknown_channel_ineligible(self) -> None:
        assert is_harmonic_eligible("Analog_01") is False

    def test_force_override_makes_eligible(self) -> None:
        # is_harmonic_eligible doesn't support force_role; use classify directly
        result = classify_harmonic_role(
            "RMS_VA",
            force_role=HarmonicChannelRole.VOLTAGE_HARMONIC,
        )
        assert result.role != HarmonicChannelRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# TestDisplayUnit
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayUnit:
    def test_voltage_harmonic_display_unit_kv(self) -> None:
        result = classify_harmonic_role("VA")
        assert result.display_unit == "kV"

    def test_current_harmonic_display_unit_a(self) -> None:
        result = classify_harmonic_role("IA")
        assert result.display_unit == "A"

    def test_unknown_display_unit_none(self) -> None:
        result = classify_harmonic_role("Analog_01")
        assert result.display_unit is None


# ─────────────────────────────────────────────────────────────────────────────
# TestResultImmutability
# ─────────────────────────────────────────────────────────────────────────────


class TestResultImmutability:
    def test_result_is_frozen(self) -> None:
        result = classify_harmonic_role("VA")
        with pytest.raises((AttributeError, TypeError)):
            result.role = HarmonicChannelRole.UNKNOWN  # type: ignore[misc]
