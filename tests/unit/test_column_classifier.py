"""Unit tests for app.data.column_classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.column_classifier import (
    CONFIRMATION_THRESHOLD,
    ColumnClassification,
    classify_csv_column,
    classify_csv_columns,
)


# ─────────────────────────────────────────────────────────────────────────────
# Exact-name classification
# ─────────────────────────────────────────────────────────────────────────────


class TestExactNameClassification:
    def test_frequency_exact(self) -> None:
        cl = classify_csv_column("frequency")
        assert cl.signal_type == "frequency"
        assert cl.unit == "Hz"
        assert cl.display_group == "frequency"
        assert cl.confidence >= 0.90
        assert cl.inferred_from == "name_exact"

    def test_mw_exact(self) -> None:
        cl = classify_csv_column("MW")
        assert cl.signal_type == "active_power"
        assert cl.unit == "MW"
        assert cl.confidence >= 0.90

    def test_mvar_exact(self) -> None:
        cl = classify_csv_column("MVar")
        assert cl.signal_type == "reactive_power"
        assert cl.unit == "MVAr"
        assert cl.confidence >= 0.90

    def test_rocof_exact(self) -> None:
        cl = classify_csv_column("rocof")
        assert cl.signal_type == "rocof"
        assert cl.unit == "Hz/s"
        assert cl.confidence >= 0.90

    def test_kv_exact(self) -> None:
        cl = classify_csv_column("kV")
        assert cl.signal_type == "voltage_rms"
        assert cl.confidence >= 0.85

    def test_freq_alias(self) -> None:
        cl = classify_csv_column("freq")
        assert cl.signal_type == "frequency"

    def test_case_insensitive(self) -> None:
        cl = classify_csv_column("FREQUENCY")
        assert cl.signal_type == "frequency"

    def test_leading_trailing_whitespace(self) -> None:
        cl = classify_csv_column("  MW  ")
        assert cl.signal_type == "active_power"


# ─────────────────────────────────────────────────────────────────────────────
# Single-letter names — always below threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleLetterLowConfidence:
    def test_p_requires_confirmation(self) -> None:
        cl = classify_csv_column("P")
        assert cl.signal_type == "active_power"
        assert cl.confidence < CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is True

    def test_q_requires_confirmation(self) -> None:
        cl = classify_csv_column("Q")
        assert cl.signal_type == "reactive_power"
        assert cl.requires_user_confirmation is True

    def test_f_requires_confirmation(self) -> None:
        cl = classify_csv_column("F")
        assert cl.signal_type == "frequency"
        assert cl.requires_user_confirmation is True


# ─────────────────────────────────────────────────────────────────────────────
# Keyword classification
# ─────────────────────────────────────────────────────────────────────────────


class TestKeywordClassification:
    def test_frequency_in_name(self) -> None:
        cl = classify_csv_column("Bus Frequency")
        assert cl.signal_type == "frequency"
        assert cl.inferred_from == "name_keyword"

    def test_system_frequency_high_confidence(self) -> None:
        cl = classify_csv_column("System Frequency")
        assert cl.signal_type == "frequency"
        assert cl.confidence >= 0.90

    def test_system_demand_is_active_power(self) -> None:
        cl = classify_csv_column("System Demand")
        assert cl.signal_type == "active_power"
        assert cl.unit == "MW"
        assert cl.display_group == "power"
        assert cl.confidence >= 0.80
        assert cl.requires_user_confirmation is False

    def test_total_demand_is_active_power(self) -> None:
        cl = classify_csv_column("Total Demand")
        assert cl.signal_type == "active_power"
        assert cl.confidence >= 0.80

    def test_active_power_phrase(self) -> None:
        cl = classify_csv_column("Active Power")
        assert cl.signal_type == "active_power"
        assert cl.confidence >= 0.88

    def test_reactive_power_phrase(self) -> None:
        cl = classify_csv_column("Reactive Power")
        assert cl.signal_type == "reactive_power"
        assert cl.confidence >= 0.88

    def test_mvar_in_name(self) -> None:
        cl = classify_csv_column("Site MVar")
        assert cl.signal_type == "reactive_power"

    def test_tie_line_requires_confirmation(self) -> None:
        cl = classify_csv_column("Tie-Line")
        assert cl.signal_type == "active_power"
        assert cl.confidence < CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is True

    def test_interchange_requires_confirmation(self) -> None:
        cl = classify_csv_column("Net Interchange")
        assert cl.requires_user_confirmation is True

    def test_demand_alone_requires_confirmation(self) -> None:
        cl = classify_csv_column("Demand")
        assert cl.signal_type == "active_power"
        assert cl.requires_user_confirmation is True

    def test_voltage_keyword(self) -> None:
        cl = classify_csv_column("Bus Voltage")
        assert cl.signal_type == "voltage_rms"
        assert cl.display_group == "voltage_rms"

    def test_current_keyword(self) -> None:
        cl = classify_csv_column("Line Current")
        assert cl.signal_type == "current_rms"

    def test_rocof_keyword(self) -> None:
        cl = classify_csv_column("ROCOF_HV")
        assert cl.signal_type == "rocof"

    def test_dfdt_keyword(self) -> None:
        cl = classify_csv_column("df/dt")
        assert cl.signal_type == "rocof"


# ─────────────────────────────────────────────────────────────────────────────
# Additional active/reactive-power terminology (generic, not PULU-specific —
# these headers do not appear in any repository fixture and must resolve
# purely from the name rules below).
# ─────────────────────────────────────────────────────────────────────────────


class TestActivePowerTerminology:
    def test_grid_demand_is_active_power_and_flagged(self) -> None:
        cl = classify_csv_column("Grid Demand")
        assert cl.signal_type == "active_power"
        assert cl.unit == "MW"
        assert cl.requires_user_confirmation is True

    def test_net_generation_is_active_power(self) -> None:
        cl = classify_csv_column("Net Generation")
        assert cl.signal_type == "active_power"

    def test_plant_output_is_active_power(self) -> None:
        cl = classify_csv_column("Plant Output")
        assert cl.signal_type == "active_power"

    def test_import_power_is_active_power_and_flagged(self) -> None:
        cl = classify_csv_column("Import Power")
        assert cl.signal_type == "active_power"
        assert cl.requires_user_confirmation is True

    def test_export_power_is_active_power_and_flagged(self) -> None:
        cl = classify_csv_column("Export Power")
        assert cl.signal_type == "active_power"
        assert cl.requires_user_confirmation is True

    def test_real_power_is_active_power(self) -> None:
        cl = classify_csv_column("Real Power")
        assert cl.signal_type == "active_power"
        assert cl.confidence >= 0.88

    def test_p_total_is_active_power_and_conservative(self) -> None:
        cl = classify_csv_column("P Total")
        assert cl.signal_type == "active_power"
        assert cl.unit == "MW"
        assert cl.requires_user_confirmation is True

    def test_p_total_underscore_variant(self) -> None:
        cl = classify_csv_column("P_TOTAL")
        assert cl.signal_type == "active_power"

    def test_q_total_is_reactive_power_and_conservative(self) -> None:
        cl = classify_csv_column("Q Total")
        assert cl.signal_type == "reactive_power"
        assert cl.unit == "MVAr"
        assert cl.requires_user_confirmation is True

    def test_mvar_capital_r_variant(self) -> None:
        cl = classify_csv_column("MVAr")
        assert cl.signal_type == "reactive_power"


# ─────────────────────────────────────────────────────────────────────────────
# Relay-style phase voltage/current names (name-based only — never inferred
# from magnitude).
# ─────────────────────────────────────────────────────────────────────────────


class TestRelayStyleVoltageCurrent:
    @pytest.mark.parametrize("name", ["Va", "Vb", "Vc", "Vab", "Vbc", "Vca"])
    def test_phase_voltage_names(self, name: str) -> None:
        cl = classify_csv_column(name)
        assert cl.signal_type == "voltage_rms"
        assert cl.unit == "V"
        assert cl.requires_user_confirmation is False

    @pytest.mark.parametrize("name", ["Ia", "Ib", "Ic"])
    def test_phase_current_names(self, name: str) -> None:
        cl = classify_csv_column(name)
        assert cl.signal_type == "current_rms"
        assert cl.unit == "A"
        assert cl.requires_user_confirmation is False

    def test_neutral_current(self) -> None:
        cl = classify_csv_column("In")
        assert cl.signal_type == "current_rms"

    @pytest.mark.parametrize("name,expected", [
        ("V0", "voltage_rms"), ("V1", "voltage_rms"), ("V2", "voltage_rms"),
        ("I0", "current_rms"), ("I1", "current_rms"), ("I2", "current_rms"),
    ])
    def test_sequence_component_names(self, name: str, expected: str) -> None:
        cl = classify_csv_column(name)
        assert cl.signal_type == expected
        assert cl.requires_user_confirmation is False

    def test_relay_names_not_influenced_by_magnitude(self) -> None:
        # Values are power-magnitude-like (tens of thousands); the relay name
        # must still win on its own, and the result must not depend on values.
        cl_no_values = classify_csv_column("Va")
        cl_with_values = classify_csv_column("Va", [18738.85, 18751.21, 18739.43])
        assert cl_no_values.signal_type == cl_with_values.signal_type == "voltage_rms"


# ─────────────────────────────────────────────────────────────────────────────
# Value-profile inference
# ─────────────────────────────────────────────────────────────────────────────


class TestValueProfileInference:
    def _near_50_values(self) -> list[float]:
        rng = np.random.default_rng(42)
        return (50.0 + rng.normal(0, 0.05, 100)).tolist()

    def _near_60_values(self) -> list[float]:
        rng = np.random.default_rng(42)
        return (60.0 + rng.normal(0, 0.05, 100)).tolist()

    def test_values_near_50hz_suggest_frequency(self) -> None:
        cl = classify_csv_column("UnknownSignal", self._near_50_values())
        assert cl.signal_type == "frequency"
        assert cl.inferred_from == "value_profile"

    def test_values_near_60hz_suggest_frequency(self) -> None:
        cl = classify_csv_column("Col1", self._near_60_values())
        assert cl.signal_type == "frequency"

    def test_value_profile_always_below_threshold(self) -> None:
        cl = classify_csv_column("UnknownSignal", self._near_50_values())
        assert cl.confidence < CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is True

    def test_per_unit_values_suggest_voltage(self) -> None:
        rng = np.random.default_rng(42)
        vals = (1.0 + rng.normal(0, 0.02, 100)).tolist()
        cl = classify_csv_column("VoltageChannel", vals)
        # name "voltage" keyword should win over value profile
        assert cl.signal_type == "voltage_rms"

    def test_per_unit_values_alone_low_confidence(self) -> None:
        rng = np.random.default_rng(42)
        vals = (1.0 + rng.normal(0, 0.02, 100)).tolist()
        cl = classify_csv_column("Signal_ABC", vals)
        assert cl.requires_user_confirmation is True

    def test_name_takes_precedence_over_values(self) -> None:
        # Column named "Frequency" should get name_exact, not value_profile
        rng = np.random.default_rng(42)
        vals = (1.0 + rng.normal(0, 0.02, 100)).tolist()
        cl = classify_csv_column("Frequency", vals)
        assert cl.inferred_from == "name_exact"

    def test_too_few_values_no_value_classification(self) -> None:
        cl = classify_csv_column("X", [50.0, 50.1])
        # Only 2 values — value_profile won't trigger; should fall to unknown
        assert cl.signal_type is None


# ─────────────────────────────────────────────────────────────────────────────
# Confidence threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceThreshold:
    def test_high_confidence_no_confirmation(self) -> None:
        cl = classify_csv_column("frequency")
        assert cl.confidence >= CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is False

    def test_at_threshold_no_confirmation(self) -> None:
        # "System Demand" → 0.85 ≥ 0.80
        cl = classify_csv_column("System Demand")
        assert cl.confidence >= CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is False

    def test_below_threshold_requires_confirmation(self) -> None:
        cl = classify_csv_column("Tie-Line")
        assert cl.confidence < CONFIRMATION_THRESHOLD
        assert cl.requires_user_confirmation is True


# ─────────────────────────────────────────────────────────────────────────────
# Unknown / unrecognised columns
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownColumn:
    def test_unknown_name_returns_other_group(self) -> None:
        cl = classify_csv_column("XYZ_random_col")
        assert cl.display_group == "other"

    def test_unknown_name_no_signal_type(self) -> None:
        cl = classify_csv_column("XYZ_random_col")
        assert cl.signal_type is None
        assert cl.unit is None

    def test_unknown_requires_confirmation(self) -> None:
        cl = classify_csv_column("XYZ_random_col")
        assert cl.requires_user_confirmation is True

    def test_unknown_inferred_from(self) -> None:
        cl = classify_csv_column("XYZ_random_col")
        assert cl.inferred_from == "unknown"

    def test_empty_name_returns_other(self) -> None:
        cl = classify_csv_column("")
        assert cl.display_group == "other"


# ─────────────────────────────────────────────────────────────────────────────
# classify_csv_columns (DataFrame API)
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyDataFrame:
    def test_skips_timestamp_column(self) -> None:
        df = pd.DataFrame({"Time": [1, 2, 3], "Frequency": [50.0, 50.1, 49.9]})
        result = classify_csv_columns(df, timestamp_column="Time")
        assert "Time" not in result
        assert "Frequency" in result

    def test_classifies_data_columns(self) -> None:
        df = pd.DataFrame({
            "Time": [1, 2, 3],
            "MW": [100.0, 110.0, 105.0],
            "MVar": [10.0, 12.0, 11.0],
            "Frequency": [50.0, 50.1, 49.9],
        })
        result = classify_csv_columns(df, timestamp_column="Time")
        assert result["MW"].signal_type == "active_power"
        assert result["MVar"].signal_type == "reactive_power"
        assert result["Frequency"].signal_type == "frequency"

    def test_returns_all_non_timestamp_columns(self) -> None:
        df = pd.DataFrame({"ts": [1], "A": [1.0], "B": [2.0], "C": [3.0]})
        result = classify_csv_columns(df, timestamp_column="ts")
        assert set(result.keys()) == {"A", "B", "C"}

    def test_no_timestamp_column_classifies_all(self) -> None:
        df = pd.DataFrame({"MW": [100.0], "MVar": [10.0]})
        result = classify_csv_columns(df)
        assert len(result) == 2

    def test_pulu_csv_columns_classified(self) -> None:
        """Mirror the pulu_20260306.csv column set."""
        df = pd.DataFrame({
            "Time": ["3/6/2026 17:25"],
            "Time.1": ["17:25"],
            "System Demand": [18738.85],
            "Tie-Line": [108.16],
            "Frequency": [50.02],
        })
        result = classify_csv_columns(df, timestamp_column="Time")
        assert result["Frequency"].signal_type == "frequency"
        assert result["System Demand"].signal_type == "active_power"
        assert result["System Demand"].requires_user_confirmation is False
        assert result["Tie-Line"].requires_user_confirmation is True
        assert result["Time.1"].signal_type is None   # unrecognised artifact
