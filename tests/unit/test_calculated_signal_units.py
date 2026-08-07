"""Unit tests for app.calculated_signals.units."""
from __future__ import annotations

import numpy as np
import pytest

from app.calculated_signals.units import (
    DIMENSIONLESS_UNIT,
    NormalizedUnit,
    UnitFamily,
    are_compatible_units,
    convert_values,
    normalize_unit,
)


class TestNormalizeUnit:
    def test_none_is_unresolved(self) -> None:
        result = normalize_unit(None)
        assert result.raw is None
        assert result.canonical is None
        assert result.family is None
        assert not result.is_resolved

    def test_unrecognized_string_is_unresolved_not_an_error(self) -> None:
        result = normalize_unit("bananas")
        assert result.raw == "bananas"
        assert not result.is_resolved

    @pytest.mark.parametrize("unit,expected_canonical", [
        ("V", "V"), ("v", "V"), ("volt", "V"), ("volts", "V"),
        ("kV", "kV"), ("kv", "kV"),
        ("mV", "mV"), ("mv", "mV"),
    ])
    def test_voltage_family(self, unit: str, expected_canonical: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.VOLTAGE
        assert result.canonical == expected_canonical

    @pytest.mark.parametrize("unit,expected_canonical", [
        ("A", "A"), ("a", "A"), ("amp", "A"), ("amps", "A"),
        ("ampere", "A"), ("amperes", "A"),
        ("kA", "kA"), ("mA", "mA"),
    ])
    def test_current_family(self, unit: str, expected_canonical: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.CURRENT
        assert result.canonical == expected_canonical

    @pytest.mark.parametrize("unit,expected_canonical", [
        ("W", "W"), ("watt", "W"), ("watts", "W"),
        ("kW", "kW"), ("MW", "MW"), ("mw", "MW"), ("GW", "GW"),
    ])
    def test_active_power_family(self, unit: str, expected_canonical: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.ACTIVE_POWER
        assert result.canonical == expected_canonical

    @pytest.mark.parametrize("unit,expected_canonical", [
        ("var", "var"),
        ("kVAr", "kVAr"), ("kvar", "kVAr"),
        ("MVAr", "MVAr"), ("MVar", "MVAr"), ("mvarr", "MVAr"),
        ("GVAr", "GVAr"),
    ])
    def test_reactive_power_family(self, unit: str, expected_canonical: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.REACTIVE_POWER
        assert result.canonical == expected_canonical

    def test_frequency_family(self) -> None:
        result = normalize_unit("Hz")
        assert result.family == UnitFamily.FREQUENCY
        assert result.canonical == "Hz"

    @pytest.mark.parametrize("unit", ["Hz/s", "hz/s", "Hz/sec", "HzPerSecond"])
    def test_rocof_family(self, unit: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.ROCOF
        assert result.canonical == "Hz/s"

    @pytest.mark.parametrize("unit", ["pu", "p.u.", "per_unit", "dimensionless", "1"])
    def test_dimensionless_family(self, unit: str) -> None:
        result = normalize_unit(unit)
        assert result.family == UnitFamily.DIMENSIONLESS
        assert result.canonical == DIMENSIONLESS_UNIT

    def test_case_insensitive(self) -> None:
        assert normalize_unit("MW").family == normalize_unit("mw").family == normalize_unit("Mw").family

    def test_whitespace_stripped(self) -> None:
        assert normalize_unit("  MW  ").family == UnitFamily.ACTIVE_POWER


class TestAreCompatibleUnits:
    def test_same_family_compatible(self) -> None:
        assert are_compatible_units(normalize_unit("MW"), normalize_unit("kW"))

    def test_different_family_incompatible(self) -> None:
        assert not are_compatible_units(normalize_unit("MW"), normalize_unit("kV"))

    def test_two_unresolved_units_not_compatible(self) -> None:
        assert not are_compatible_units(normalize_unit("bananas"), normalize_unit("oranges"))

    def test_resolved_and_unresolved_not_compatible(self) -> None:
        assert not are_compatible_units(normalize_unit("MW"), normalize_unit(None))


class TestConvertValues:
    def test_kw_to_mw(self) -> None:
        result = convert_values(1.0, normalize_unit("kW"), normalize_unit("MW"))
        assert result == pytest.approx(0.001)

    def test_mw_to_kw(self) -> None:
        result = convert_values(1.0, normalize_unit("MW"), normalize_unit("kW"))
        assert result == pytest.approx(1000.0)

    def test_kv_to_v(self) -> None:
        result = convert_values(2.0, normalize_unit("kV"), normalize_unit("V"))
        assert result == pytest.approx(2000.0)

    def test_identity_conversion(self) -> None:
        result = convert_values(5.0, normalize_unit("MW"), normalize_unit("MW"))
        assert result == pytest.approx(5.0)

    def test_array_conversion(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        result = convert_values(values, normalize_unit("kA"), normalize_unit("A"))
        np.testing.assert_array_almost_equal(result, np.array([1000.0, 2000.0, 3000.0]))

    def test_incompatible_families_raise(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            convert_values(1.0, normalize_unit("MW"), normalize_unit("kV"))

    def test_unresolved_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            convert_values(1.0, normalize_unit("bananas"), normalize_unit("MW"))

    def test_input_array_not_mutated(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        before = values.copy()
        convert_values(values, normalize_unit("kW"), normalize_unit("MW"))
        np.testing.assert_array_equal(values, before)

    def test_returns_new_array(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        result = convert_values(values, normalize_unit("MW"), normalize_unit("MW"))
        result[0] = 999.0
        assert values[0] == 1.0
