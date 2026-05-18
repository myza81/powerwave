"""Tests for Phase 8.55B: column_detector.py."""
from __future__ import annotations

import pytest

from app.import_wizard.column_detector import (
    _classify_by_values,
    _suggested_name,
    classify_by_name,
    detect_column_mappings,
)
from app.import_wizard.column_mapping import ParameterType


# ─────────────────────────────────────────────────────────────────────────────
# classify_by_name
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyByName:
    @pytest.mark.parametrize("name,expected_type", [
        ("Voltage_A", ParameterType.VOLTAGE),
        ("voltage", ParameterType.VOLTAGE),
        ("Va", ParameterType.VOLTAGE),
        ("kV_L1", ParameterType.VOLTAGE),
        ("Current_A", ParameterType.CURRENT),
        ("Ia", ParameterType.CURRENT),
        ("amps", ParameterType.CURRENT),
        ("MW_Total", ParameterType.MW),
        ("ActivePower", ParameterType.MW),
        ("MVAR_Q", ParameterType.MVAR),
        ("ReactivePower", ParameterType.MVAR),
        ("Freq", ParameterType.FREQUENCY),
        ("frequency", ParameterType.FREQUENCY),
        ("Hz_1", ParameterType.FREQUENCY),
        ("ROCOF", ParameterType.ROCOF),
        ("df_dt", ParameterType.ROCOF),
        ("DigitalInput", ParameterType.DIGITAL),
        ("CB_Status", ParameterType.DIGITAL),
        ("TripAlarm", ParameterType.DIGITAL),
        ("Timestamp", ParameterType.TIMESTAMP),
        ("datetime", ParameterType.TIMESTAMP),
        ("Date", ParameterType.TIMESTAMP),
    ])
    def test_known_types(self, name, expected_type):
        ptype, _, confidence, _ = classify_by_name(name)
        assert ptype == expected_type
        assert confidence > 0.0

    @pytest.mark.parametrize("name", [
        "Channel_1", "Signal_X", "Data", "Col_99",
    ])
    def test_unknown_type_for_ambiguous_names(self, name):
        ptype, _, _, _ = classify_by_name(name)
        assert ptype == ParameterType.UNKNOWN

    def test_strips_unit_from_parentheses(self):
        ptype, unit, _, _ = classify_by_name("Voltage (kV)")
        assert ptype == ParameterType.VOLTAGE

    def test_case_insensitive(self):
        p1, _, _, _ = classify_by_name("VOLTAGE")
        p2, _, _, _ = classify_by_name("voltage")
        assert p1 == p2 == ParameterType.VOLTAGE

    def test_confidence_positive(self):
        _, _, confidence, _ = classify_by_name("Voltage")
        assert confidence > 0.0

    def test_reason_non_empty(self):
        _, _, _, reason = classify_by_name("Voltage")
        assert reason != ""


# ─────────────────────────────────────────────────────────────────────────────
# _classify_by_values
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyByValues:
    def test_binary_values_are_digital(self):
        samples = ["0", "1", "0", "1", "0", "1", "0", "1", "0", "0"]
        ptype, boost = _classify_by_values(samples)
        assert ptype == ParameterType.DIGITAL
        assert boost > 0.0

    def test_frequency_range(self):
        samples = [50.0, 50.01, 49.99, 50.02, 50.0]
        ptype, boost = _classify_by_values(samples)
        assert ptype == ParameterType.FREQUENCY
        assert boost > 0.0

    def test_returns_none_for_too_few_samples(self):
        ptype, boost = _classify_by_values(["50.0", "50.1"])
        assert ptype is None
        assert boost == 0.0

    def test_returns_none_for_empty(self):
        ptype, boost = _classify_by_values([])
        assert ptype is None
        assert boost == 0.0

    def test_voltage_range(self):
        # ~230 V nominal
        samples = [229.8, 230.0, 230.1, 230.2, 229.9]
        ptype, boost = _classify_by_values(samples)
        assert ptype == ParameterType.VOLTAGE

    def test_non_numeric_returns_none(self):
        samples = ["abc", "def", "ghi", "jkl"]
        ptype, boost = _classify_by_values(samples)
        assert ptype is None


# ─────────────────────────────────────────────────────────────────────────────
# _suggested_name
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestedName:
    def test_voltage_prefix(self):
        name = _suggested_name("V_A", ParameterType.VOLTAGE, set())
        assert name.startswith("voltage")

    def test_unique_name_when_collision(self):
        used = {"voltage_v_a"}
        name = _suggested_name("V_A", ParameterType.VOLTAGE, used)
        assert name not in used

    def test_unknown_uses_channel_prefix(self):
        name = _suggested_name("Col_1", ParameterType.UNKNOWN, set())
        assert "channel" in name or "col" in name.lower()

    def test_timestamp_prefix(self):
        name = _suggested_name("ts", ParameterType.TIMESTAMP, set())
        assert "timestamp" in name


# ─────────────────────────────────────────────────────────────────────────────
# detect_column_mappings
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectColumnMappings:
    def _make_rows(self, n: int = 5) -> list[list]:
        return [
            [f"2024-01-01 00:00:0{i}", 230.0 + i, 100.0 + i, 23.0 + i * 0.1, 50.0 + i * 0.01]
            for i in range(n)
        ]

    def test_returns_one_mapping_per_column(self):
        cols = ["Timestamp", "Voltage_A", "Current_A", "MW", "Freq"]
        rows = self._make_rows()
        mappings = detect_column_mappings(cols, rows)
        assert len(mappings) == len(cols)

    def test_source_indices_correct(self):
        cols = ["Timestamp", "Voltage_A", "Current_A"]
        rows = self._make_rows()
        mappings = detect_column_mappings(cols, rows)
        for i, m in enumerate(mappings):
            assert m.source_index == i

    def test_voltage_classified_correctly(self):
        cols = ["Timestamp", "Voltage_A"]
        rows = self._make_rows()
        mappings = detect_column_mappings(cols, rows, timestamp_column_names={"Timestamp"})
        voltage = [m for m in mappings if m.source_name == "Voltage_A"]
        assert len(voltage) == 1
        assert voltage[0].parameter_type == ParameterType.VOLTAGE

    def test_timestamp_columns_marked_as_timestamp(self):
        cols = ["Timestamp", "Voltage_A"]
        rows = self._make_rows()
        mappings = detect_column_mappings(cols, rows, timestamp_column_names={"Timestamp"})
        ts = [m for m in mappings if m.source_name == "Timestamp"]
        assert ts[0].parameter_type == ParameterType.TIMESTAMP

    def test_suggested_names_are_unique(self):
        cols = ["Voltage", "Voltage", "Voltage"]
        rows = [[230.0, 231.0, 232.0] for _ in range(5)]
        mappings = detect_column_mappings(cols, rows)
        names = [m.suggested_name for m in mappings]
        assert len(names) == len(set(names))

    def test_confidence_values_populated(self):
        cols = ["Timestamp", "Voltage_A", "Current_A"]
        rows = self._make_rows()
        mappings = detect_column_mappings(cols, rows)
        for m in mappings:
            assert 0.0 <= m.confidence <= 1.0

    def test_unit_extracted_from_parenthetical(self):
        cols = ["Voltage (kV)", "Current (A)"]
        rows = [[230.0, 100.0] for _ in range(5)]
        mappings = detect_column_mappings(cols, rows)
        voltage = [m for m in mappings if "Voltage" in m.source_name][0]
        assert voltage.unit is not None

    def test_empty_inputs(self):
        mappings = detect_column_mappings([], [])
        assert mappings == []

    def test_classification_reason_non_empty(self):
        cols = ["Voltage_A", "Current_A"]
        rows = [[230.0, 100.0] for _ in range(5)]
        mappings = detect_column_mappings(cols, rows)
        for m in mappings:
            assert m.classification_reason != ""

    def test_frequency_detected_by_value(self):
        cols = ["Channel_X"]
        rows = [[50.0 + i * 0.01] for i in range(10)]
        mappings = detect_column_mappings(cols, rows)
        # Value analysis should detect frequency range
        assert mappings[0].parameter_type in (ParameterType.FREQUENCY, ParameterType.UNKNOWN)

    def test_digital_detected_by_value(self):
        cols = ["Trip"]
        rows = [[0], [1], [0], [0], [1], [0], [1], [0], [0], [1]]
        mappings = detect_column_mappings(cols, rows)
        assert mappings[0].parameter_type == ParameterType.DIGITAL

    def test_mvar_column_classified(self):
        cols = ["MVAR_Total"]
        rows = [[10.5 + i] for i in range(5)]
        mappings = detect_column_mappings(cols, rows)
        assert mappings[0].parameter_type == ParameterType.MVAR
