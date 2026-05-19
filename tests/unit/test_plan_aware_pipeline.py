"""Unit tests for Phase 8.55H plan-aware pipeline.

Covers:
- build_execution_plan(): all validation rules + column authority
- run_import_pipeline_with_plan(): authoritative execution behaviour
"""
from __future__ import annotations

from pathlib import Path

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.models import (
    ColumnMappingCandidate,
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.pipeline_plan_builder import PlanBuildResult, build_execution_plan
from app.import_wizard.import_pipeline import run_import_pipeline_with_plan, ImportPipelineOptions
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_candidate(
    col: str = "timestamp",
    fmt: str | None = "%Y-%m-%d %H:%M:%S",
    confidence: float = 0.95,
    user_selected: bool = True,
) -> TimestampCandidate:
    return TimestampCandidate(
        column_name=col,
        column_index=0,
        confidence=confidence,
        detected_format=fmt,
        user_selected=user_selected,
    )


def _make_mapping(
    source: str,
    suggested: str | None = None,
    ptype: ParameterType = ParameterType.MW,
    unit: str = "MW",
    excluded: bool = False,
    name_override: str | None = None,
    type_override: ParameterType | None = None,
    unit_override: str | None = None,
) -> ColumnMappingCandidate:
    return ColumnMappingCandidate(
        source_name=source,
        source_index=0,
        suggested_name=suggested or source,
        parameter_type=ptype,
        unit=unit,
        excluded=excluded,
        user_name_override=name_override,
        user_type_override=type_override,
        user_unit_override=unit_override,
    )


def _make_session(
    candidate: TimestampCandidate | None = None,
    repair_plan: TimestampRepairPlan | None = None,
) -> ImportWizardSession:
    c = candidate or _make_candidate()
    session = ImportWizardSession(source_path="/fake/data.csv", provider_type="csv")
    session.timestamp_candidates = [c]
    session.selected_timestamp_column = c.column_name
    session.timestamp_repair_plan = repair_plan or TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
        detected_format=c.detected_format,
        repair_validated=True,
    )
    return session


# ─────────────────────────────────────────────────────────────────────────────
# build_execution_plan — validation rules
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildExecutionPlanValidation:
    def test_no_timestamp_returns_error(self) -> None:
        session = ImportWizardSession(source_path="/f.csv", provider_type="csv")
        # No candidates → best_timestamp_candidate() returns None
        result = build_execution_plan(session, [])
        assert not result.is_executable
        codes = [m.code for m in result.errors()]
        assert "PLAN_NO_TIMESTAMP" in codes

    def test_no_timestamp_plan_is_none(self) -> None:
        session = ImportWizardSession(source_path="/f.csv", provider_type="csv")
        result = build_execution_plan(session, [])
        assert result.normalization_plan is None
        assert result.timestamp_candidate is None

    def test_no_data_columns_returns_error(self) -> None:
        session = _make_session()
        # All columns excluded
        mappings = [_make_mapping("mw", excluded=True)]
        result = build_execution_plan(session, mappings)
        assert not result.is_executable
        codes = [m.code for m in result.errors()]
        assert "PLAN_NO_DATA_COLUMNS" in codes

    def test_all_columns_excluded_via_timestamp_type(self) -> None:
        session = _make_session()
        # Mark column as TIMESTAMP type → auto-excluded
        mappings = [_make_mapping("ts2", ptype=ParameterType.TIMESTAMP)]
        result = build_execution_plan(session, mappings)
        assert not result.is_executable
        assert any(m.code == "PLAN_NO_DATA_COLUMNS" for m in result.errors())

    def test_duplicate_canonical_name_returns_error(self) -> None:
        session = _make_session()
        mappings = [
            _make_mapping("mw_a", suggested="power", ptype=ParameterType.MW),
            _make_mapping("mw_b", suggested="power", ptype=ParameterType.MW),
        ]
        result = build_execution_plan(session, mappings)
        assert not result.is_executable
        codes = [m.code for m in result.errors()]
        assert "PLAN_DUPLICATE_NAME" in codes

    def test_duplicate_via_user_name_override(self) -> None:
        session = _make_session()
        mappings = [
            _make_mapping("va", name_override="voltage"),
            _make_mapping("vb", name_override="voltage"),
        ]
        result = build_execution_plan(session, mappings)
        assert not result.is_executable
        assert any(m.code == "PLAN_DUPLICATE_NAME" for m in result.errors())

    def test_clean_plan_is_executable(self) -> None:
        session = _make_session()
        mappings = [
            _make_mapping("mw", ptype=ParameterType.MW, unit="MW"),
            _make_mapping("va", ptype=ParameterType.VOLTAGE, unit="kV"),
        ]
        result = build_execution_plan(session, mappings)
        assert result.is_executable
        assert not result.errors()

    def test_unknown_type_emits_warning_not_error(self) -> None:
        session = _make_session()
        mappings = [_make_mapping("mystery", ptype=ParameterType.UNKNOWN, unit=None)]
        result = build_execution_plan(session, mappings)
        assert result.is_executable
        assert any(m.code == "PLAN_UNKNOWN_COLUMN" for m in result.warnings())
        assert not result.errors()


# ─────────────────────────────────────────────────────────────────────────────
# build_execution_plan — column authority
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildExecutionPlanColumnAuthority:
    def test_user_selected_candidate_preserved(self) -> None:
        candidate = _make_candidate(col="event_time", user_selected=True)
        session = _make_session(candidate=candidate)
        mappings = [_make_mapping("mw")]
        result = build_execution_plan(session, mappings)
        assert result.timestamp_candidate is candidate

    def test_excluded_column_in_excluded_list(self) -> None:
        session = _make_session()
        mappings = [
            _make_mapping("mw", excluded=False),
            _make_mapping("noise", excluded=True),
        ]
        result = build_execution_plan(session, mappings)
        assert result.is_executable
        plan = result.normalization_plan
        assert "noise" not in plan.selected_columns
        assert "noise" in plan.excluded_columns

    def test_renamed_channel_in_renames(self) -> None:
        session = _make_session()
        mappings = [_make_mapping("mw_raw", name_override="active_power")]
        result = build_execution_plan(session, mappings)
        plan = result.normalization_plan
        assert "mw_raw" in plan.column_renames
        assert plan.column_renames["mw_raw"] == "active_power"
        assert "mw_raw" in plan.selected_columns

    def test_type_override_stored_in_plan(self) -> None:
        session = _make_session()
        mappings = [_make_mapping("trip", ptype=ParameterType.UNKNOWN, type_override=ParameterType.DIGITAL)]
        result = build_execution_plan(session, mappings)
        plan = result.normalization_plan
        assert plan.column_types.get("trip") == ParameterType.DIGITAL

    def test_unit_override_stored_in_plan(self) -> None:
        session = _make_session()
        mappings = [_make_mapping("va", ptype=ParameterType.VOLTAGE, unit="V", unit_override="kV")]
        result = build_execution_plan(session, mappings)
        plan = result.normalization_plan
        assert plan.column_units.get("va") == "kV"

    def test_timestamp_column_always_excluded(self) -> None:
        candidate = _make_candidate(col="ts")
        session = _make_session(candidate=candidate)
        # Include a mapping with the same name as the timestamp column
        mappings = [
            _make_mapping("ts", ptype=ParameterType.TIMESTAMP),
            _make_mapping("mw"),
        ]
        result = build_execution_plan(session, mappings)
        plan = result.normalization_plan
        assert "ts" not in plan.selected_columns
        assert "ts" in plan.excluded_columns

    def test_session_repair_plan_preserved(self) -> None:
        custom_plan = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION,
            repair_validated=True,
        )
        session = _make_session(repair_plan=custom_plan)
        mappings = [_make_mapping("mw")]
        result = build_execution_plan(session, mappings)
        assert result.normalization_plan.timestamp_plan is custom_plan

    def test_repair_plan_derived_when_session_has_none(self) -> None:
        candidate = _make_candidate(fmt="%d/%m/%Y %H:%M")
        session = ImportWizardSession(source_path="/f.csv", provider_type="csv")
        session.timestamp_candidates = [candidate]
        session.selected_timestamp_column = candidate.column_name
        # No repair plan on session — builder must derive one
        mappings = [_make_mapping("mw")]
        result = build_execution_plan(session, mappings)
        rp = result.normalization_plan.timestamp_plan
        assert rp is not None
        assert rp.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT
        assert rp.detected_format == "%d/%m/%Y %H:%M"

    def test_excel_serial_format_selects_correct_strategy(self) -> None:
        candidate = _make_candidate(fmt="excel_serial")
        session = ImportWizardSession(source_path="/f.xlsx", provider_type="excel")
        session.timestamp_candidates = [candidate]
        session.selected_timestamp_column = candidate.column_name
        mappings = [_make_mapping("mw")]
        result = build_execution_plan(session, mappings)
        rp = result.normalization_plan.timestamp_plan
        assert rp.strategy == TimestampRepairStrategy.EXCEL_SERIAL_CONVERSION

    def test_no_format_selects_no_repair(self) -> None:
        candidate = _make_candidate(fmt=None)
        session = ImportWizardSession(source_path="/f.csv", provider_type="csv")
        session.timestamp_candidates = [candidate]
        session.selected_timestamp_column = candidate.column_name
        mappings = [_make_mapping("mw")]
        result = build_execution_plan(session, mappings)
        rp = result.normalization_plan.timestamp_plan
        assert rp.strategy == TimestampRepairStrategy.NO_REPAIR


# ─────────────────────────────────────────────────────────────────────────────
# run_import_pipeline_with_plan — authoritative execution
# ─────────────────────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: int = 10) -> None:
    lines = ["timestamp,va,ia,mw,mvar,trip"]
    for i in range(rows):
        ts = f"2024-01-01 00:00:{i:02d}.000"
        lines.append(f"{ts},{230 + i * 0.1:.3f},{1.0 + i * 0.01:.3f},{100.0},{5.0},{i % 2}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_session(path: Path) -> tuple[ImportWizardSession, list[ColumnMappingCandidate]]:
    """Return a session + mappings matching _write_csv() column layout."""
    session = ImportWizardSession(source_path=str(path), provider_type="csv")
    candidate = TimestampCandidate(
        column_name="timestamp",
        column_index=0,
        confidence=0.99,
        detected_format="%Y-%m-%d %H:%M:%S.%f",
        user_selected=True,
    )
    session.timestamp_candidates = [candidate]
    session.selected_timestamp_column = "timestamp"
    session.timestamp_repair_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
        detected_format="%Y-%m-%d %H:%M:%S.%f",
        repair_validated=True,
    )
    session.raw_preview = RawPreviewModel(
        column_names=["timestamp", "va", "ia", "mw", "mvar", "trip"],
        preview_rows=[],
        header_row_index=0,
    )
    mappings = [
        ColumnMappingCandidate("va",   1, "va",   ParameterType.VOLTAGE, unit="kV"),
        ColumnMappingCandidate("ia",   2, "ia",   ParameterType.CURRENT, unit="A"),
        ColumnMappingCandidate("mw",   3, "mw",   ParameterType.MW,      unit="MW"),
        ColumnMappingCandidate("mvar", 4, "mvar", ParameterType.MVAR,    unit="Mvar"),
        ColumnMappingCandidate("trip", 5, "trip", ParameterType.DIGITAL, unit=None),
    ]
    return session, mappings


class TestRunImportPipelineWithPlan:
    def test_success_with_clean_plan(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        plan_result = build_execution_plan(session, mappings)
        assert plan_result.is_executable

        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        assert result.record is not None

    def test_user_selected_candidate_used(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # The candidate column is "timestamp" — verify it is used
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.selected_candidate is not None
        assert result.selected_candidate.column_name == "timestamp"

    def test_user_selected_repair_plan_used(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Inject a specific repair plan; verify it appears in the result
        custom_repair = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format="%Y-%m-%d %H:%M:%S.%f",
            repair_validated=True,
            repair_notes="custom",
        )
        session.timestamp_repair_plan = custom_repair
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.repair_plan is not None
        assert result.repair_plan.repair_notes == "custom"

    def test_excluded_column_absent_from_record(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Exclude 'mvar' from the GUI
        for m in mappings:
            if m.source_name == "mvar":
                m.excluded = True
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        record = result.record
        all_names = [c.name for c in record.analog_channels + record.digital_channels]
        assert "mvar" not in all_names

    def test_renamed_channel_in_record(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Rename 'mw' → 'active_power'
        for m in mappings:
            if m.source_name == "mw":
                m.user_name_override = "active_power"
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        all_names = [c.name for c in result.record.analog_channels]
        assert "active_power" in all_names
        assert "mw" not in all_names

    def test_type_override_digital_creates_digital_channel(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Override 'mvar' to digital
        for m in mappings:
            if m.source_name == "mvar":
                m.user_type_override = ParameterType.DIGITAL
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        digital_names = [c.name for c in result.record.digital_channels]
        assert "mvar" in digital_names

    def test_type_override_analog_creates_analog_channel(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Override 'trip' (originally digital) to MW (analog)
        for m in mappings:
            if m.source_name == "trip":
                m.user_type_override = ParameterType.MW
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        analog_names = [c.name for c in result.record.analog_channels]
        assert "trip" in analog_names

    def test_unit_override_in_diagnostics(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        for m in mappings:
            if m.source_name == "va":
                m.user_unit_override = "pu"
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        # Unit should be stored in the dataset's parameter metadata
        if result.dataset is not None:
            units = {p.canonical_name: p.unit for p in result.dataset.parameters}
            assert units.get("va") == "pu"

    def test_no_timestamp_candidate_returns_failure(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session = ImportWizardSession(source_path=str(csv), provider_type="csv")
        # No candidates at all
        plan = NormalizationPlan(
            timestamp_plan=TimestampRepairPlan(
                strategy=TimestampRepairStrategy.NO_REPAIR, repair_validated=True
            ),
            selected_columns=["mw"],
        )
        mappings = [_make_mapping("mw")]
        result = run_import_pipeline_with_plan(str(csv), session, plan, mappings)
        assert not result.success
        codes = [m.code for m in result.validation_messages]
        assert "PIPELINE_NO_TIMESTAMP" in codes

    def test_missing_ts_column_in_data_returns_failure(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Override selected column to a name not in the file
        session.timestamp_candidates[0] = TimestampCandidate(
            column_name="nonexistent_ts",
            column_index=99,
            confidence=0.99,
            detected_format="%Y-%m-%d %H:%M:%S",
            user_selected=True,
        )
        session.selected_timestamp_column = "nonexistent_ts"
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert not result.success
        codes = [m.code for m in result.validation_messages]
        assert "PIPELINE_TS_COLUMN_MISSING" in codes

    def test_session_raw_preview_used_for_header_row(self, tmp_path) -> None:
        # CSV with a metadata row before the real header
        csv = tmp_path / "meta.csv"
        csv.write_text(
            "# source: lab\n"
            "timestamp,mw\n"
            "2024-01-01 00:00:00,100.0\n"
            "2024-01-01 00:00:01,101.0\n",
            encoding="utf-8",
        )
        session = ImportWizardSession(source_path=str(csv), provider_type="csv")
        candidate = TimestampCandidate(
            column_name="timestamp", column_index=0, confidence=0.99,
            detected_format="%Y-%m-%d %H:%M:%S", user_selected=True,
        )
        session.timestamp_candidates = [candidate]
        session.selected_timestamp_column = "timestamp"
        session.timestamp_repair_plan = TimestampRepairPlan(
            strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
            detected_format="%Y-%m-%d %H:%M:%S",
            repair_validated=True,
        )
        # header_row_index=1 tells the loader to skip the metadata row
        session.raw_preview = RawPreviewModel(
            column_names=["timestamp", "mw"],
            preview_rows=[],
            header_row_index=1,
        )
        mappings = [ColumnMappingCandidate("mw", 1, "mw", ParameterType.MW, unit="MW")]
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        assert result.record.sample_count() == 2

    def test_all_columns_excluded_produces_warning(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Exclude everything except trip — then exclude trip too
        for m in mappings:
            m.excluded = True
        plan_result = build_execution_plan(session, mappings)
        # is_executable should be False; no exception raised
        assert not plan_result.is_executable
        assert any(m.code == "PLAN_NO_DATA_COLUMNS" for m in plan_result.errors())

    def test_diagnostics_reflects_authoritative_plan(self, tmp_path) -> None:
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)
        # Exclude one analog channel
        for m in mappings:
            if m.source_name == "mvar":
                m.excluded = True
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        # 4 analog (va, ia, mw, trip-excluded; wait — trip is digital)
        # va, ia, mw = 3 analog; trip = 1 digital; mvar excluded
        assert result.diagnostics.analog_channel_count == 3
        assert result.diagnostics.digital_channel_count == 1

    def test_stale_auto_plan_not_used(self, tmp_path) -> None:
        """run_import_pipeline_with_plan must use the provided plan, not re-auto-generate."""
        csv = tmp_path / "data.csv"
        _write_csv(csv)
        session, mappings = _base_session(csv)

        # Build plan with mvar renamed to 'reactive_power'
        for m in mappings:
            if m.source_name == "mvar":
                m.user_name_override = "reactive_power"
        plan_result = build_execution_plan(session, mappings)
        result = run_import_pipeline_with_plan(
            str(csv), session, plan_result.normalization_plan, mappings
        )
        assert result.success
        analog_names = [c.name for c in result.record.analog_channels]
        # The authoritative plan must have been used — not a re-generated auto-plan
        assert "reactive_power" in analog_names
        assert "mvar" not in analog_names
