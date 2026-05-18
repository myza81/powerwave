"""Unit tests for Phase 8.55A import wizard architecture and data contracts.

Coverage:
  1.  ImportWizardSession — initialization, advance_to, errors/warnings/infos,
      best_timestamp_candidate, is_ready_to_normalize, add/clear_message
  2.  RawPreviewModel — construction, defaults, parse_warnings accumulation
  3.  TimestampCandidate — representation, confidence, user_selected flag
  4.  ColumnMappingCandidate — effective_name/unit/type with and without overrides,
      has_user_override, excluded flag
  5.  NormalizationPlan — is_executable guards (all three blocking conditions),
      errors/warnings helpers, effective_name rename lookup
  6.  WizardStep transition rules — forward strict, forward skip, backward always
  7.  ValidationMessage — frozen, severity filtering, code/message/column/action
  8.  Serialization-safe — frozen models survive copy, no numpy/pandas/Qt imports
      leaked into contract layer

No QApplication or real file I/O required.
"""
from __future__ import annotations

import copy

import pytest

from app.import_wizard import (
    ColumnMappingCandidate,
    ImportWizardSession,
    NormalizationPlan,
    ParameterType,
    RawPreviewModel,
    TimestampCandidate,
    TimestampRepairPlan,
    TimestampRepairStrategy,
    ValidationMessage,
    ValidationSeverity,
    WizardStep,
    can_transition,
    next_step,
    step_index,
    steps_before,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ok_repair_plan() -> TimestampRepairPlan:
    return TimestampRepairPlan(
        strategy=TimestampRepairStrategy.NO_REPAIR,
        repair_validated=True,
    )


def _ok_normalization_plan() -> NormalizationPlan:
    return NormalizationPlan(
        timestamp_plan=_ok_repair_plan(),
        selected_columns=["VA", "IA"],
    )


def _session() -> ImportWizardSession:
    return ImportWizardSession(source_path="/data/test.csv", provider_type="csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ImportWizardSession
# ─────────────────────────────────────────────────────────────────────────────


class TestImportWizardSession:
    def test_default_step_is_load_file(self) -> None:
        s = _session()
        assert s.current_step == WizardStep.LOAD_FILE

    def test_source_path_stored(self) -> None:
        s = _session()
        assert s.source_path == "/data/test.csv"

    def test_provider_type_stored(self) -> None:
        s = _session()
        assert s.provider_type == "csv"

    def test_defaults_all_none_or_empty(self) -> None:
        s = _session()
        assert s.sheet_name is None
        assert s.delimiter is None
        assert s.raw_preview is None
        assert s.timestamp_candidates == []
        assert s.selected_timestamp_column is None
        assert s.timestamp_repair_plan is None
        assert s.column_mappings == []
        assert s.normalization_plan is None
        assert s.validation_messages == []

    def test_advance_to_next_step_succeeds(self) -> None:
        s = _session()
        assert s.advance_to(WizardStep.RAW_PREVIEW) is True
        assert s.current_step == WizardStep.RAW_PREVIEW

    def test_advance_to_same_step_is_allowed(self) -> None:
        s = _session()
        assert s.advance_to(WizardStep.LOAD_FILE) is True

    def test_advance_skip_two_steps_fails(self) -> None:
        s = _session()
        assert s.advance_to(WizardStep.TIMESTAMP_SELECT) is False
        assert s.current_step == WizardStep.LOAD_FILE  # unchanged

    def test_advance_skip_with_allow_skip_succeeds(self) -> None:
        s = _session()
        assert s.advance_to(WizardStep.TIMESTAMP_SELECT, allow_skip=True) is True
        assert s.current_step == WizardStep.TIMESTAMP_SELECT

    def test_advance_backward_always_succeeds(self) -> None:
        s = _session()
        s.advance_to(WizardStep.COLUMN_MAPPING, allow_skip=True)
        assert s.advance_to(WizardStep.LOAD_FILE) is True
        assert s.current_step == WizardStep.LOAD_FILE

    def test_add_message_appends(self) -> None:
        s = _session()
        msg = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="W001",
            message="Duplicate column",
        )
        s.add_message(msg)
        assert len(s.validation_messages) == 1

    def test_clear_messages_empties(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.INFO, "I001", "ok"))
        s.clear_messages()
        assert s.validation_messages == []

    def test_errors_filters_errors_only(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.ERROR, "E001", "bad"))
        s.add_message(ValidationMessage(ValidationSeverity.WARNING, "W001", "warn"))
        s.add_message(ValidationMessage(ValidationSeverity.INFO, "I001", "info"))
        assert len(s.errors()) == 1
        assert s.errors()[0].code == "E001"

    def test_warnings_filters_warnings_only(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.ERROR, "E001", "bad"))
        s.add_message(ValidationMessage(ValidationSeverity.WARNING, "W001", "warn"))
        assert len(s.warnings()) == 1

    def test_infos_filters_infos_only(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.INFO, "I001", "note"))
        s.add_message(ValidationMessage(ValidationSeverity.ERROR, "E001", "bad"))
        assert len(s.infos()) == 1

    def test_has_errors_true_when_error_present(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.ERROR, "E001", "bad"))
        assert s.has_errors() is True

    def test_has_errors_false_with_only_warnings(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.WARNING, "W001", "warn"))
        assert s.has_errors() is False

    def test_is_ready_to_normalize_false_with_no_plan(self) -> None:
        assert not _session().is_ready_to_normalize()

    def test_is_ready_to_normalize_true_with_valid_plan(self) -> None:
        s = _session()
        s.normalization_plan = _ok_normalization_plan()
        assert s.is_ready_to_normalize() is True

    def test_best_timestamp_candidate_none_when_empty(self) -> None:
        assert _session().best_timestamp_candidate() is None

    def test_best_timestamp_candidate_returns_highest_confidence(self) -> None:
        s = _session()
        s.timestamp_candidates = [
            TimestampCandidate("t1", 0, confidence=0.6),
            TimestampCandidate("t2", 1, confidence=0.9),
            TimestampCandidate("t3", 2, confidence=0.4),
        ]
        best = s.best_timestamp_candidate()
        assert best is not None
        assert best.column_name == "t2"

    def test_best_timestamp_candidate_user_selected_wins_over_confidence(self) -> None:
        s = _session()
        s.timestamp_candidates = [
            TimestampCandidate("t1", 0, confidence=0.9),
            TimestampCandidate("t2", 1, confidence=0.4, user_selected=True),
        ]
        best = s.best_timestamp_candidate()
        assert best is not None
        assert best.column_name == "t2"


# ─────────────────────────────────────────────────────────────────────────────
# 2. RawPreviewModel
# ─────────────────────────────────────────────────────────────────────────────


class TestRawPreviewModel:
    def test_basic_construction(self) -> None:
        m = RawPreviewModel(
            column_names=["time", "VA", "IA"],
            preview_rows=[["0.0", "230.0", "5.0"]],
        )
        assert m.column_names == ["time", "VA", "IA"]
        assert len(m.preview_rows) == 1

    def test_defaults(self) -> None:
        m = RawPreviewModel(column_names=[], preview_rows=[])
        assert m.header_row_index == 0
        assert m.skipped_row_count == 0
        assert m.row_count_estimate == 0
        assert m.sheet_name is None
        assert m.parse_warnings == []

    def test_sheet_name_stored(self) -> None:
        m = RawPreviewModel(column_names=[], preview_rows=[], sheet_name="Sheet1")
        assert m.sheet_name == "Sheet1"

    def test_parse_warnings_accumulate(self) -> None:
        m = RawPreviewModel(column_names=[], preview_rows=[])
        m.parse_warnings.append("Row 3 has extra columns")
        m.parse_warnings.append("Row 7 is blank")
        assert len(m.parse_warnings) == 2

    def test_skipped_rows_and_header_index(self) -> None:
        m = RawPreviewModel(
            column_names=["a", "b"],
            preview_rows=[],
            header_row_index=3,
            skipped_row_count=3,
        )
        assert m.header_row_index == 3
        assert m.skipped_row_count == 3

    def test_large_row_count_estimate(self) -> None:
        m = RawPreviewModel(column_names=[], preview_rows=[], row_count_estimate=1_000_000)
        assert m.row_count_estimate == 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 3. TimestampCandidate
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestampCandidate:
    def test_required_fields(self) -> None:
        c = TimestampCandidate(column_name="Timestamp", column_index=0, confidence=0.95)
        assert c.column_name == "Timestamp"
        assert c.column_index == 0
        assert c.confidence == pytest.approx(0.95)

    def test_defaults(self) -> None:
        c = TimestampCandidate("ts", 0, 0.5)
        assert c.detected_format is None
        assert c.example_values == []
        assert c.invalid_sample_count == 0
        assert c.timezone_detected is None
        assert c.user_selected is False

    def test_detected_format_stored(self) -> None:
        c = TimestampCandidate("ts", 0, 0.8, detected_format="%Y-%m-%d %H:%M:%S")
        assert c.detected_format == "%Y-%m-%d %H:%M:%S"

    def test_example_values_stored(self) -> None:
        c = TimestampCandidate("ts", 0, 0.7, example_values=["2024-01-01", "2024-01-02"])
        assert len(c.example_values) == 2

    def test_user_selected_flag(self) -> None:
        c = TimestampCandidate("ts", 2, 0.3, user_selected=True)
        assert c.user_selected is True

    def test_invalid_sample_count(self) -> None:
        c = TimestampCandidate("ts", 0, 0.6, invalid_sample_count=5)
        assert c.invalid_sample_count == 5

    def test_timezone_detected(self) -> None:
        c = TimestampCandidate("ts", 0, 0.9, timezone_detected="UTC")
        assert c.timezone_detected == "UTC"

    def test_ranking_by_confidence(self) -> None:
        candidates = [
            TimestampCandidate("c1", 0, 0.3),
            TimestampCandidate("c2", 1, 0.9),
            TimestampCandidate("c3", 2, 0.6),
        ]
        ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        assert ranked[0].column_name == "c2"
        assert ranked[-1].column_name == "c1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ColumnMappingCandidate — override behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnMappingCandidate:
    def test_effective_name_uses_suggested_when_no_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Va", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
        )
        assert c.effective_name == "VA"

    def test_effective_name_uses_user_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Va", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            user_name_override="Phase_A_Voltage",
        )
        assert c.effective_name == "Phase_A_Voltage"

    def test_effective_unit_uses_auto_when_no_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Va", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            unit="kV",
        )
        assert c.effective_unit == "kV"

    def test_effective_unit_uses_user_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Va", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            unit="kV", user_unit_override="V",
        )
        assert c.effective_unit == "V"

    def test_effective_unit_none_when_unset(self) -> None:
        c = ColumnMappingCandidate(
            source_name="X", source_index=0,
            suggested_name="X", parameter_type=ParameterType.UNKNOWN,
        )
        assert c.effective_unit is None

    def test_effective_type_uses_auto_when_no_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Ia", source_index=1,
            suggested_name="IA", parameter_type=ParameterType.CURRENT,
        )
        assert c.effective_type == ParameterType.CURRENT

    def test_effective_type_uses_user_override(self) -> None:
        c = ColumnMappingCandidate(
            source_name="Ia", source_index=1,
            suggested_name="IA", parameter_type=ParameterType.UNKNOWN,
            user_type_override=ParameterType.CURRENT,
        )
        assert c.effective_type == ParameterType.CURRENT

    def test_has_user_override_false_when_clean(self) -> None:
        c = ColumnMappingCandidate(
            source_name="VA", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
        )
        assert c.has_user_override is False

    def test_has_user_override_true_when_name_overridden(self) -> None:
        c = ColumnMappingCandidate(
            source_name="VA", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            user_name_override="PhaseA",
        )
        assert c.has_user_override is True

    def test_has_user_override_true_when_unit_overridden(self) -> None:
        c = ColumnMappingCandidate(
            source_name="VA", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            user_unit_override="pu",
        )
        assert c.has_user_override is True

    def test_has_user_override_true_when_type_overridden(self) -> None:
        c = ColumnMappingCandidate(
            source_name="col", source_index=0,
            suggested_name="col", parameter_type=ParameterType.UNKNOWN,
            user_type_override=ParameterType.MW,
        )
        assert c.has_user_override is True

    def test_excluded_flag(self) -> None:
        c = ColumnMappingCandidate(
            source_name="junk", source_index=5,
            suggested_name="junk", parameter_type=ParameterType.UNKNOWN,
            excluded=True,
        )
        assert c.excluded is True

    def test_all_parameter_types_representable(self) -> None:
        for pt in ParameterType:
            c = ColumnMappingCandidate(
                source_name="x", source_index=0,
                suggested_name="x", parameter_type=pt,
            )
            assert c.effective_type == pt

    def test_confidence_stored(self) -> None:
        c = ColumnMappingCandidate(
            source_name="VA", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            confidence=0.85,
        )
        assert c.confidence == pytest.approx(0.85)

    def test_classification_reason_stored(self) -> None:
        c = ColumnMappingCandidate(
            source_name="VA", source_index=0,
            suggested_name="VA", parameter_type=ParameterType.VOLTAGE,
            classification_reason="unit:kV",
        )
        assert c.classification_reason == "unit:kV"


# ─────────────────────────────────────────────────────────────────────────────
# 5. NormalizationPlan — executable readiness
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizationPlan:
    def test_not_executable_with_no_timestamp_plan(self) -> None:
        plan = NormalizationPlan(selected_columns=["VA"])
        assert plan.is_executable is False

    def test_not_executable_with_unvalidated_timestamp_plan(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=TimestampRepairPlan(
                strategy=TimestampRepairStrategy.NO_REPAIR,
                repair_validated=False,
            ),
            selected_columns=["VA"],
        )
        assert plan.is_executable is False

    def test_not_executable_with_no_selected_columns(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=[],
        )
        assert plan.is_executable is False

    def test_not_executable_with_error_message(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            validation_messages=[
                ValidationMessage(ValidationSeverity.ERROR, "E001", "bad column")
            ],
        )
        assert plan.is_executable is False

    def test_executable_when_all_conditions_met(self) -> None:
        assert _ok_normalization_plan().is_executable is True

    def test_executable_with_warnings_only(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            validation_messages=[
                ValidationMessage(ValidationSeverity.WARNING, "W001", "minor issue")
            ],
        )
        assert plan.is_executable is True

    def test_errors_helper_filters_errors(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            validation_messages=[
                ValidationMessage(ValidationSeverity.ERROR, "E001", "bad"),
                ValidationMessage(ValidationSeverity.WARNING, "W001", "warn"),
            ],
        )
        assert len(plan.errors()) == 1

    def test_warnings_helper_filters_warnings(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            validation_messages=[
                ValidationMessage(ValidationSeverity.WARNING, "W001", "warn"),
                ValidationMessage(ValidationSeverity.INFO, "I001", "ok"),
            ],
        )
        assert len(plan.warnings()) == 1

    def test_effective_name_applies_rename(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            column_renames={"Va": "VA"},
        )
        assert plan.effective_name("Va") == "VA"

    def test_effective_name_identity_when_no_rename(self) -> None:
        plan = _ok_normalization_plan()
        assert plan.effective_name("VA") == "VA"

    def test_column_types_stored(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            column_types={"VA": ParameterType.VOLTAGE},
        )
        assert plan.column_types["VA"] == ParameterType.VOLTAGE

    def test_output_path_suggestion(self) -> None:
        plan = NormalizationPlan(
            timestamp_plan=_ok_repair_plan(),
            selected_columns=["VA"],
            output_path_suggestion="/out/normalized.csv",
        )
        assert plan.output_path_suggestion == "/out/normalized.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 6. WizardStep transition rules
# ─────────────────────────────────────────────────────────────────────────────


class TestWizardStepTransitions:
    def test_forward_one_step_allowed(self) -> None:
        assert can_transition(WizardStep.LOAD_FILE, WizardStep.RAW_PREVIEW) is True

    def test_forward_two_steps_blocked(self) -> None:
        assert can_transition(WizardStep.LOAD_FILE, WizardStep.TIMESTAMP_SELECT) is False

    def test_forward_two_steps_allowed_with_skip(self) -> None:
        assert can_transition(
            WizardStep.LOAD_FILE, WizardStep.TIMESTAMP_SELECT, allow_skip=True
        ) is True

    def test_forward_to_last_step_with_skip_allowed(self) -> None:
        assert can_transition(
            WizardStep.LOAD_FILE, WizardStep.RENDER_WAVEFORM, allow_skip=True
        ) is True

    def test_backward_always_allowed(self) -> None:
        for target in list(WizardStep)[:-1]:
            assert can_transition(WizardStep.RENDER_WAVEFORM, target) is True

    def test_same_step_always_allowed(self) -> None:
        for step in WizardStep:
            assert can_transition(step, step) is True

    def test_next_step_returns_following_step(self) -> None:
        assert next_step(WizardStep.LOAD_FILE) == WizardStep.RAW_PREVIEW
        assert next_step(WizardStep.RAW_PREVIEW) == WizardStep.TIMESTAMP_SELECT

    def test_next_step_returns_none_at_end(self) -> None:
        assert next_step(WizardStep.RENDER_WAVEFORM) is None

    def test_steps_before_load_file_is_empty(self) -> None:
        assert steps_before(WizardStep.LOAD_FILE) == []

    def test_steps_before_third_step_has_two(self) -> None:
        before = steps_before(WizardStep.TIMESTAMP_SELECT)
        assert WizardStep.LOAD_FILE in before
        assert WizardStep.RAW_PREVIEW in before
        assert WizardStep.TIMESTAMP_SELECT not in before

    def test_step_index_is_sequential(self) -> None:
        indices = [step_index(s) for s in WizardStep]
        assert indices == list(range(len(WizardStep)))

    def test_all_steps_reachable_forward_with_skip(self) -> None:
        start = WizardStep.LOAD_FILE
        for step in WizardStep:
            assert can_transition(start, step, allow_skip=True) is True

    def test_session_advance_to_follows_transition_rules(self) -> None:
        s = ImportWizardSession(source_path="x.csv", provider_type="csv")
        assert s.advance_to(WizardStep.COLUMN_MAPPING) is False
        assert s.advance_to(WizardStep.RAW_PREVIEW) is True
        assert s.advance_to(WizardStep.TIMESTAMP_SELECT) is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. ValidationMessage — structure
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationMessage:
    def test_all_fields_stored(self) -> None:
        m = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="W042",
            message="Missing unit for column VA",
            affected_column="VA",
            suggested_action="Set unit to kV",
        )
        assert m.severity == ValidationSeverity.WARNING
        assert m.code == "W042"
        assert m.message == "Missing unit for column VA"
        assert m.affected_column == "VA"
        assert m.suggested_action == "Set unit to kV"

    def test_optional_fields_default_none(self) -> None:
        m = ValidationMessage(ValidationSeverity.INFO, "I001", "All good")
        assert m.affected_column is None
        assert m.suggested_action is None

    def test_message_is_frozen(self) -> None:
        m = ValidationMessage(ValidationSeverity.ERROR, "E001", "fatal")
        with pytest.raises((AttributeError, TypeError)):
            m.code = "E002"  # type: ignore[misc]

    def test_all_severities_representable(self) -> None:
        for sev in ValidationSeverity:
            m = ValidationMessage(sev, "X", "test")
            assert m.severity == sev

    def test_error_severity_value(self) -> None:
        assert ValidationSeverity.ERROR.value == "error"

    def test_warning_severity_value(self) -> None:
        assert ValidationSeverity.WARNING.value == "warning"

    def test_info_severity_value(self) -> None:
        assert ValidationSeverity.INFO.value == "info"


# ─────────────────────────────────────────────────────────────────────────────
# 8. TimestampRepairPlan — strategy coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestampRepairPlan:
    def test_no_repair_strategy(self) -> None:
        p = TimestampRepairPlan(strategy=TimestampRepairStrategy.NO_REPAIR)
        assert p.strategy == TimestampRepairStrategy.NO_REPAIR
        assert p.repair_validated is False

    def test_parse_detected_format(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format="%Y-%m-%d %H:%M:%S",
            repair_validated=True,
        )
        assert p.detected_format == "%Y-%m-%d %H:%M:%S"
        assert p.repair_validated is True

    def test_parse_user_format(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
            user_format="%d/%m/%Y %H:%M",
        )
        assert p.user_format == "%d/%m/%Y %H:%M"

    def test_reconstruct_from_interval(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
            sampling_interval_seconds=0.0002,
        )
        assert p.sampling_interval_seconds == pytest.approx(0.0002)

    def test_combine_date_time_columns(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.COMBINE_DATE_TIME_COLUMNS,
            date_column="Date",
            time_column="Time",
        )
        assert p.date_column == "Date"
        assert p.time_column == "Time"

    def test_timezone_alignment(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.TIMEZONE_ALIGNMENT,
            source_timezone="America/New_York",
            target_timezone="UTC",
        )
        assert p.source_timezone == "America/New_York"
        assert p.target_timezone == "UTC"

    def test_excel_serial_conversion_strategy(self) -> None:
        p = TimestampRepairPlan(strategy=TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION)
        assert p.strategy == TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION

    def test_plan_is_frozen(self) -> None:
        p = TimestampRepairPlan(strategy=TimestampRepairStrategy.NO_REPAIR)
        with pytest.raises((AttributeError, TypeError)):
            p.repair_validated = True  # type: ignore[misc]

    def test_all_strategies_representable(self) -> None:
        for strat in TimestampRepairStrategy:
            p = TimestampRepairPlan(strategy=strat)
            assert p.strategy == strat


# ─────────────────────────────────────────────────────────────────────────────
# 9. Serialization-safe — frozen models survive copy; no Qt/numpy leakage
# ─────────────────────────────────────────────────────────────────────────────


class TestSerializationSafe:
    def test_validation_message_survives_deepcopy(self) -> None:
        m = ValidationMessage(ValidationSeverity.WARNING, "W001", "test", "col_A")
        m2 = copy.deepcopy(m)
        assert m2.code == "W001"
        assert m2.affected_column == "col_A"

    def test_timestamp_repair_plan_survives_deepcopy(self) -> None:
        p = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format="%Y-%m-%d",
            repair_validated=True,
        )
        p2 = copy.deepcopy(p)
        assert p2.detected_format == "%Y-%m-%d"
        assert p2.repair_validated is True

    def test_normalization_plan_survives_deepcopy(self) -> None:
        plan = _ok_normalization_plan()
        plan2 = copy.deepcopy(plan)
        assert plan2.selected_columns == ["VA", "IA"]
        assert plan2.is_executable is True

    def test_session_survives_deepcopy(self) -> None:
        s = _session()
        s.add_message(ValidationMessage(ValidationSeverity.INFO, "I001", "ok"))
        s2 = copy.deepcopy(s)
        assert s2.source_path == "/data/test.csv"
        assert len(s2.validation_messages) == 1

    def test_no_numpy_in_contracts_module(self) -> None:
        import app.import_wizard.contracts as m
        assert not hasattr(m, "np"), "contracts.py must not import numpy"

    def test_no_numpy_in_wizard_state_module(self) -> None:
        import app.import_wizard.wizard_state as m
        assert not hasattr(m, "np"), "wizard_state.py must not import numpy"

    def test_no_pyqt_in_contracts_module(self) -> None:
        import app.import_wizard.contracts as m
        import sys
        # PyQt6 must not be imported as a side effect of importing contracts
        assert "PyQt6" not in sys.modules or True  # module may exist globally;
        # the critical check: contracts has no PyQt import
        source = m.__file__
        assert source is not None
        with open(source) as f:
            content = f.read()
        assert "PyQt6" not in content

    def test_no_pandas_in_timestamp_contracts(self) -> None:
        import app.import_wizard.timestamp_contracts as m
        assert not hasattr(m, "pd"), "timestamp_contracts.py must not import pandas"

    def test_wizard_step_enum_has_all_required_steps(self) -> None:
        required = {
            "LOAD_FILE", "RAW_PREVIEW", "TIMESTAMP_SELECT", "TIMESTAMP_REPAIR",
            "COLUMN_MAPPING", "NORMALIZATION_REVIEW", "SAVE_NORMALIZED", "RENDER_WAVEFORM",
        }
        names = {s.name for s in WizardStep}
        assert required <= names

    def test_parameter_type_enum_has_required_types(self) -> None:
        required = {"VOLTAGE", "CURRENT", "MW", "MVAR", "FREQUENCY", "ROCOF", "DIGITAL", "UNKNOWN"}
        names = {t.name for t in ParameterType}
        assert required <= names
