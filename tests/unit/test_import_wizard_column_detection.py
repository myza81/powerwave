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


def _mapping_for(name: str, values: list):
    rows = [[str(v)] for v in values]
    return detect_column_mappings([name], rows)[0]


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

    def test_magnitude_alone_no_longer_authoritative_for_voltage(self):
        # ~230 V nominal values with no name evidence must NOT be assigned an
        # authoritative type from magnitude alone (policy: numeric magnitude
        # must not independently assign voltage/current/MW/MVAr/ROCOF).
        samples = [229.8, 230.0, 230.1, 230.2, 229.9]
        ptype, boost = _classify_by_values(samples)
        assert ptype is None
        assert boost == 0.0

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


# ─────────────────────────────────────────────────────────────────────────────
# Shared semantic classifier integration — demand/power terminology the
# Wizard's own name rules do not cover on their own. Generic headers; none of
# these are tied to any specific fixture's exact values.
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedClassifierIntegration:
    @pytest.mark.parametrize("name", [
        "System Demand", "Total Demand", "Tie-Line", "Real Power",
        "Net Generation", "Plant Output", "Import Power", "Export Power",
    ])
    def test_demand_and_power_terms_resolve_to_mw(self, name):
        m = _mapping_for(name, [123.4, 125.1, 124.0, 126.7])
        assert m.parameter_type == ParameterType.MW
        assert m.unit == "MW"
        assert "voltage" not in m.suggested_name

    def test_grid_demand_low_confidence_still_mw_not_voltage(self):
        m = _mapping_for("Grid Demand", [18738.85, 18751.21, 18739.43, 18771.59])
        assert m.parameter_type == ParameterType.MW
        assert m.confidence < 0.80

    def test_tie_line_confidence_reflects_shared_classifier_uncertainty(self):
        # The shared classifier marks "Tie-Line" uncertain (0.70); the Wizard
        # must retain that uncertainty rather than forcing high confidence.
        m = _mapping_for("Tie-Line", [108.16, 80.64, 80.32, 57.28])
        assert m.parameter_type == ParameterType.MW
        assert m.confidence == pytest.approx(0.70)

    def test_reactive_power_terms_resolve_to_mvar(self):
        m = _mapping_for("Reactive Power", [45.2, 44.8, 46.1, 45.5])
        assert m.parameter_type == ParameterType.MVAR
        assert m.unit == "Mvar" or m.unit == "MVAr"

    @pytest.mark.parametrize("name", ["Va", "Vb", "Vc", "Vab", "Vbc", "Vca"])
    def test_relay_voltage_names_via_shared_classifier(self, name):
        # These names are already handled by the Wizard's own _NAME_RULES;
        # confirm the shared-classifier integration doesn't regress them.
        m = _mapping_for(name, [230.1, 230.4, 229.8, 230.0])
        assert m.parameter_type == ParameterType.VOLTAGE

    def test_sequence_component_names_not_previously_covered_by_wizard(self):
        # V1 (positive-sequence voltage) is not matched by the Wizard's own
        # _NAME_RULES regex; this must now resolve via the shared classifier.
        assert classify_by_name("V1")[0] == ParameterType.UNKNOWN
        m = _mapping_for("V1", [230.1, 230.4, 229.8, 230.0])
        assert m.parameter_type == ParameterType.VOLTAGE

    def test_p_total_resolves_to_mw_with_low_confidence(self):
        m = _mapping_for("P Total", [123.4, 125.1, 124.0, 126.7])
        assert m.parameter_type == ParameterType.MW
        assert m.confidence < 0.80

    def test_user_override_still_wins_over_shared_classifier(self):
        cols = ["System Demand"]
        rows = [[str(v)] for v in [18738.85, 18751.21, 18739.43, 18771.59]]
        mapping = detect_column_mappings(cols, rows)[0]
        assert mapping.parameter_type == ParameterType.MW  # automatic result
        mapping.user_type_override = ParameterType.VOLTAGE
        mapping.user_unit_override = "kV"
        assert mapping.effective_type == ParameterType.VOLTAGE
        assert mapping.effective_unit == "kV"
        # The automatic classification itself is left untouched by the override.
        assert mapping.parameter_type == ParameterType.MW


# ─────────────────────────────────────────────────────────────────────────────
# Unsafe magnitude fallback removed — numeric-only evidence with a neutral
# header must not authoritatively become voltage/current/MW/MVAr/ROCOF.
# ─────────────────────────────────────────────────────────────────────────────


class TestMagnitudeIsNotAuthoritative:
    @pytest.mark.parametrize("values", [
        [1.0, 1.01, 0.99, 1.02],            # per-unit-like
        [132.0, 131.8, 132.2, 132.1],       # transmission-voltage-like
        [275.0, 274.5, 275.3, 275.1],       # transmission-voltage-like
        [400.0, 401.5, 398.2, 399.0],       # current-like
        [18738.85, 18751.21, 18739.43, 18771.59],  # demand-like
    ])
    def test_neutral_header_stays_unknown(self, values):
        m = _mapping_for("Channel_1", values)
        assert m.parameter_type == ParameterType.UNKNOWN

    def test_per_unit_like_does_not_become_rocof(self):
        m = _mapping_for("Channel_1", [0.98, 1.01, 0.99, 1.02])
        assert m.parameter_type != ParameterType.ROCOF

    def test_large_magnitude_does_not_become_voltage(self):
        m = _mapping_for("Channel_1", [132.0, 131.8, 132.2, 132.1])
        assert m.parameter_type != ParameterType.VOLTAGE

    def test_current_like_magnitude_does_not_become_voltage(self):
        m = _mapping_for("Channel_1", [400.0, 401.5, 398.2, 399.0])
        assert m.parameter_type != ParameterType.VOLTAGE

    def test_demand_like_magnitude_does_not_become_voltage(self):
        m = _mapping_for("Channel_1", [18738.85, 18751.21, 18739.43, 18771.59])
        assert m.parameter_type != ParameterType.VOLTAGE

    def test_frequency_near_50_remains_a_safe_narrow_exception(self):
        # Frequency is the one retained magnitude exception (narrow 45-65 Hz
        # band, existing-test-justified) -- confirm it still works.
        m = _mapping_for("Channel_1", [50.0, 50.02, 49.98, 50.01])
        assert m.parameter_type == ParameterType.FREQUENCY

    def test_digital_binary_detection_unaffected(self):
        m = _mapping_for("Channel_1", [0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        assert m.parameter_type == ParameterType.DIGITAL


# ─────────────────────────────────────────────────────────────────────────────
# Phase B1 -- boundary-aware matching and status/control qualifier suppression
# ─────────────────────────────────────────────────────────────────────────────


class TestValidAnalogNamesStillClassify:
    """Regression guard for genuine analog measurement names."""

    @pytest.mark.parametrize("name,expected", [
        ("Voltage", ParameterType.VOLTAGE),
        ("Bus Voltage", ParameterType.VOLTAGE),
        ("Va", ParameterType.VOLTAGE),
        ("Vab", ParameterType.VOLTAGE),
        ("Current", ParameterType.CURRENT),
        ("Phase Current", ParameterType.CURRENT),
        ("Ia", ParameterType.CURRENT),
        ("Active Power", ParameterType.MW),
        ("System Demand", ParameterType.MW),
        ("Reactive Power", ParameterType.MVAR),
        ("Frequency", ParameterType.FREQUENCY),
        ("ROCOF", ParameterType.ROCOF),
    ])
    def test_valid_names(self, name: str, expected: ParameterType) -> None:
        m = _mapping_for(name, [100.0, 105.0, 98.0, 102.0, 99.0])
        assert m.parameter_type == expected


class TestStatusControlSuppression:
    """Names combining a measurement word with a status/control qualifier
    must not become that analog measurement -- see
    app.data.channel_name_matching.has_status_qualifier. The existing
    digital pattern (unchanged in this task) may still classify them as
    DIGITAL when it already would (e.g. "status"/"alarm" are pre-existing
    digital keywords); otherwise the safe result is UNKNOWN.
    """

    @pytest.mark.parametrize("name", [
        "Voltage Status", "VoltageStatus", "Voltage Alarm",
        "Current State", "CurrentState",
        "Frequency Alarm", "MW Status", "MWStatus",
        "Active Power Alarm", "Reactive Power Status",
    ])
    def test_not_an_analog_measurement(self, name: str) -> None:
        m = _mapping_for(name, [100.0, 105.0, 98.0, 102.0, 99.0])
        assert m.parameter_type not in (
            ParameterType.VOLTAGE, ParameterType.CURRENT, ParameterType.MW,
            ParameterType.MVAR, ParameterType.FREQUENCY, ParameterType.ROCOF,
        )

    def test_status_qualified_names_may_fall_back_to_existing_digital_pattern(self) -> None:
        # "status"/"alarm" are pre-existing digital keywords (unchanged by
        # this task) -- once the analog measurement match is suppressed,
        # these fall through to the same digital rule that already
        # classifies "Breaker Status"/"Trip"/"CB Open".
        m = _mapping_for("Voltage Status", [100.0, 105.0, 98.0, 102.0, 99.0])
        assert m.parameter_type == ParameterType.DIGITAL

    def test_state_qualified_name_has_no_pre_existing_digital_keyword(self) -> None:
        # "state" (unlike "status") was never one of the existing digital
        # keywords, and this task does not add one -- the safe result here
        # is UNKNOWN, not a forced/new digital classification.
        m = _mapping_for("Current State", [100.0, 105.0, 98.0, 102.0, 99.0])
        assert m.parameter_type == ParameterType.UNKNOWN


class TestExistingDigitalNamesUnaffected:
    @pytest.mark.parametrize("name", ["Breaker Status", "Trip", "CB Open"])
    def test_still_digital(self, name: str) -> None:
        m = _mapping_for(name, [0, 1, 0, 1, 0])
        assert m.parameter_type == ParameterType.DIGITAL


class TestSubstringCollisionsNoLongerClassify:
    @pytest.mark.parametrize("name", [
        "Occurrence", "Example", "Input", "Index", "Interval", "Info",
        "Variable", "Pump", "Impulse",
    ])
    def test_unrelated_words_stay_unknown(self, name: str) -> None:
        m = _mapping_for(name, [100.0, 105.0, 98.0, 102.0, 99.0])
        assert m.parameter_type == ParameterType.UNKNOWN


class TestUserOverrideStillWins:
    """A qualifier-suppressed or collision-suppressed suggestion must still
    be fully overridable by the user, exactly like any other suggestion --
    this task does not change override precedence.
    """

    def test_user_can_override_a_suppressed_suggestion(self) -> None:
        from app.import_wizard.models import ColumnMappingCandidate

        m = _mapping_for("Voltage Status", [100.0, 105.0, 98.0, 102.0, 99.0])
        candidate = ColumnMappingCandidate(
            source_name=m.source_name,
            source_index=m.source_index,
            suggested_name=m.suggested_name,
            parameter_type=m.parameter_type,
            unit=m.unit,
            confidence=m.confidence,
            classification_reason=m.classification_reason,
        )
        candidate.user_type_override = ParameterType.VOLTAGE
        candidate.user_unit_override = "kV"

        assert candidate.effective_type == ParameterType.VOLTAGE
        assert candidate.effective_unit == "kV"
        assert candidate.has_user_override is True
