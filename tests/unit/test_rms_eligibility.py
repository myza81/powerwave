"""Unit tests for app/analytics/rms/rms_overlay.py — eligibility classification.

Tests cover:
  - COMTRADE V/I channel names → eligible
  - MW / frequency / ROCOF → ineligible
  - Vrms / Irms name fragments → ineligible
  - electrical_type metadata override
  - measurement_kind metadata override
  - force=True operator override always wins
  - Default unknown channels → ineligible (conservative)
"""
from __future__ import annotations

import pytest

from app.analytics.rms.rms_models import RMSEligibilityResult
from app.analytics.rms.rms_overlay import classify_rms_eligibility
from app.data.signal_metadata import SignalMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _meta(**kwargs) -> SignalMetadata:
    return SignalMetadata(name="ch", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TestNameHeuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestNameHeuristics:
    @pytest.mark.parametrize(
        "name",
        [
            "VA",
            "VB",
            "VC",
            "Vr",
            "Vy",
            "Ia",
            "Ib",
            "Ic",
            "KPDN1_VA",
            "SGT1_IB_HV",
            "KPDN1 VR",   # COMTRADE-style with space
            "3I0",
            "3U0",
        ],
    )
    def test_voltage_current_names_eligible(self, name: str) -> None:
        result = classify_rms_eligibility(name)
        assert result.eligible, f"Expected {name!r} to be eligible, got {result.reason}"
        assert result.auto_classified

    @pytest.mark.parametrize(
        "name",
        [
            "MW",
            "MVar",
            "System Demand",
            "Frequency",
            "FREQ",
            "SysFreq",
            "ROCOF",
            "Vrms",
            "Irms",
            "V_RMS",
            "I_RMS",
            "Vpu",
            "V_pu",
        ],
    )
    def test_processed_signal_names_ineligible(self, name: str) -> None:
        result = classify_rms_eligibility(name)
        assert not result.eligible, f"Expected {name!r} to be ineligible, got {result.reason}"
        assert result.auto_classified

    def test_unknown_channel_defaults_to_ineligible(self) -> None:
        result = classify_rms_eligibility("XyZUnknownChannel123")
        assert not result.eligible
        assert result.reason == "default_ineligible"
        assert result.auto_classified

    def test_reason_contains_matched_fragment(self) -> None:
        result = classify_rms_eligibility("SystemFrequency")
        assert "freq" in result.reason or "hz" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# TestElectricalTypeMetadata
# ─────────────────────────────────────────────────────────────────────────────


class TestElectricalTypeMetadata:
    def test_electrical_type_voltage_eligible(self) -> None:
        result = classify_rms_eligibility("SomeCol", _meta(electrical_type="voltage"))
        assert result.eligible
        assert "electrical_type:voltage" in result.reason

    def test_electrical_type_current_eligible(self) -> None:
        result = classify_rms_eligibility("SomeCol", _meta(electrical_type="current"))
        assert result.eligible
        assert "electrical_type:current" in result.reason

    def test_electrical_type_power_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCol", _meta(electrical_type="power"))
        assert not result.eligible

    def test_electrical_type_frequency_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCol", _meta(electrical_type="frequency"))
        assert not result.eligible

    def test_electrical_type_rocof_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCol", _meta(electrical_type="rocof"))
        assert not result.eligible

    def test_electrical_type_none_falls_back_to_name(self) -> None:
        # electrical_type=None → fall back to name heuristic
        result = classify_rms_eligibility("VA", _meta(electrical_type=None))
        assert result.eligible


# ─────────────────────────────────────────────────────────────────────────────
# TestMeasurementKindMetadata
# ─────────────────────────────────────────────────────────────────────────────


class TestMeasurementKindMetadata:
    def test_instantaneous_eligible(self) -> None:
        result = classify_rms_eligibility("SomeCh", _meta(measurement_kind="instantaneous"))
        assert result.eligible
        assert "measurement_kind:instantaneous" in result.reason

    def test_rms_kind_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCh", _meta(measurement_kind="rms"))
        assert not result.eligible

    def test_average_kind_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCh", _meta(measurement_kind="average"))
        assert not result.eligible

    def test_calculated_kind_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCh", _meta(measurement_kind="calculated"))
        assert not result.eligible

    def test_telemetry_kind_ineligible(self) -> None:
        result = classify_rms_eligibility("SomeCh", _meta(measurement_kind="telemetry"))
        assert not result.eligible

    def test_measurement_kind_takes_priority_over_name(self) -> None:
        # measurement_kind="instantaneous" overrides ineligible-looking name
        result = classify_rms_eligibility("MW_RMS", _meta(measurement_kind="instantaneous"))
        assert result.eligible

    def test_measurement_kind_none_falls_back_to_electrical_type(self) -> None:
        meta = _meta(measurement_kind=None, electrical_type="voltage")
        result = classify_rms_eligibility("AnyName", meta)
        assert result.eligible


# ─────────────────────────────────────────────────────────────────────────────
# TestOperatorOverride
# ─────────────────────────────────────────────────────────────────────────────


class TestOperatorOverride:
    def test_force_true_makes_any_channel_eligible(self) -> None:
        result = classify_rms_eligibility("MW", force=True)
        assert result.eligible
        assert result.reason == "operator_override"
        assert not result.auto_classified

    def test_force_true_overrides_ineligible_electrical_type(self) -> None:
        result = classify_rms_eligibility(
            "Freq", _meta(electrical_type="frequency"), force=True
        )
        assert result.eligible
        assert not result.auto_classified

    def test_force_true_overrides_telemetry_measurement_kind(self) -> None:
        result = classify_rms_eligibility(
            "SysDemand", _meta(measurement_kind="telemetry"), force=True
        )
        assert result.eligible

    def test_force_false_still_applies_heuristics(self) -> None:
        result = classify_rms_eligibility("MW", force=False)
        assert not result.eligible
        assert result.auto_classified

    def test_return_type_is_rms_eligibility_result(self) -> None:
        result = classify_rms_eligibility("VA")
        assert isinstance(result, RMSEligibilityResult)


# ─────────────────────────────────────────────────────────────────────────────
# TestCaseSensitivity
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseSensitivity:
    def test_uppercase_va_eligible(self) -> None:
        assert classify_rms_eligibility("VA").eligible

    def test_lowercase_va_eligible(self) -> None:
        assert classify_rms_eligibility("va").eligible

    def test_mixed_case_ia_eligible(self) -> None:
        assert classify_rms_eligibility("Ia_HV").eligible

    def test_uppercase_mw_ineligible(self) -> None:
        assert not classify_rms_eligibility("MW").eligible

    def test_lowercase_mw_ineligible(self) -> None:
        assert not classify_rms_eligibility("mw_import").eligible
