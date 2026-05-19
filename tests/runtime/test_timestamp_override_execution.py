"""Runtime coverage for manual timestamp override execution."""
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.models import (
    ColumnMappingCandidate,
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.normalization_plan import NormalizationPlan
from app.import_wizard.pipeline_plan_builder import build_execution_plan
from app.import_wizard.import_pipeline import run_import_pipeline_with_plan
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy


@pytest.fixture()
def local_tmp():
    path = Path("test_artifacts") / f"timestamp_override_exec_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_ambiguous_csv(path: Path) -> None:
    path.write_text(
        "Time,MW\n"
        "01/02/2026 00:00:00,100.0\n"
        "01/02/2026 00:00:01,101.0\n"
        "01/02/2026 00:00:02,102.0\n",
        encoding="utf-8",
    )


def _session_and_mappings(path: Path, *, repair_plan: TimestampRepairPlan) -> tuple[
    ImportWizardSession,
    list[ColumnMappingCandidate],
]:
    candidate = TimestampCandidate(
        column_name="Time",
        column_index=0,
        confidence=0.99,
        detected_format="%m/%d/%Y %H:%M:%S",
        example_values=[
            "01/02/2026 00:00:00",
            "01/02/2026 00:00:01",
            "01/02/2026 00:00:02",
        ],
        user_selected=True,
    )
    session = ImportWizardSession(source_path=str(path), provider_type="csv")
    session.timestamp_candidates = [candidate]
    session.selected_timestamp_column = "Time"
    session.timestamp_repair_plan = repair_plan
    session.raw_preview = RawPreviewModel(
        column_names=["Time", "MW"],
        preview_rows=[],
        header_row_index=0,
    )
    mappings = [ColumnMappingCandidate("MW", 1, "MW", ParameterType.MW, unit="MW")]
    return session, mappings


def test_user_format_used_and_detected_format_ignored(local_tmp: Path) -> None:
    csv = local_tmp / "ambiguous.csv"
    _write_ambiguous_csv(csv)
    user_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
        detected_format="%m/%d/%Y %H:%M:%S",
        user_format="%d/%m/%Y %H:%M:%S",
        repair_validated=True,
    )
    session, mappings = _session_and_mappings(csv, repair_plan=user_plan)
    plan_result = build_execution_plan(session, mappings)

    result = run_import_pipeline_with_plan(
        str(csv),
        session,
        plan_result.normalization_plan,
        mappings,
    )

    assert result.success
    assert result.repair_plan.strategy == TimestampRepairStrategy.PARSE_USER_FORMAT
    assert result.repair_plan.user_format == "%d/%m/%Y %H:%M:%S"
    assert result.normalization_result.normalized.iloc[0].month == 2
    assert result.normalization_result.normalized.iloc[0].day == 1


def test_disturbance_record_timing_correct_after_override(local_tmp: Path) -> None:
    csv = local_tmp / "timing.csv"
    _write_ambiguous_csv(csv)
    user_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
        detected_format="%m/%d/%Y %H:%M:%S",
        user_format="%d/%m/%Y %H:%M:%S",
        repair_validated=True,
    )
    session, mappings = _session_and_mappings(csv, repair_plan=user_plan)
    plan_result = build_execution_plan(session, mappings)

    result = run_import_pipeline_with_plan(
        str(csv),
        session,
        plan_result.normalization_plan,
        mappings,
    )

    assert result.success
    record = result.record
    assert record.timing_info.start_time.month == 2
    assert record.timing_info.start_time.day == 1
    assert record.duration_seconds() == 2.0
    assert list(record.waveform_data["time"]) == [0.0, 1.0, 2.0]


def test_complete_parse_failure_blocks_before_execution(local_tmp: Path) -> None:
    csv = local_tmp / "bad_override.csv"
    _write_ambiguous_csv(csv)
    invalid_user_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
        detected_format="%m/%d/%Y %H:%M:%S",
        user_format="%Y-%m-%d %H:%M:%S",
        repair_validated=False,
    )
    session, mappings = _session_and_mappings(csv, repair_plan=invalid_user_plan)
    plan = NormalizationPlan(
        timestamp_plan=invalid_user_plan,
        selected_columns=["MW"],
    )

    result = run_import_pipeline_with_plan(str(csv), session, plan, mappings)

    assert not result.success
    assert result.normalization_result is None
    assert any(m.code == "PLAN_INVALID_TIMESTAMP_FORMAT" for m in result.validation_messages)


def test_detected_format_restored_when_override_cleared(local_tmp: Path) -> None:
    csv = local_tmp / "detected.csv"
    _write_ambiguous_csv(csv)
    detected_plan = TimestampRepairPlan(
        strategy=TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
        detected_format="%m/%d/%Y %H:%M:%S",
        repair_validated=True,
    )
    session, mappings = _session_and_mappings(csv, repair_plan=detected_plan)
    plan_result = build_execution_plan(session, mappings)

    result = run_import_pipeline_with_plan(
        str(csv),
        session,
        plan_result.normalization_plan,
        mappings,
    )

    assert result.success
    assert result.repair_plan.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT
    assert result.normalization_result.normalized.iloc[0].month == 1
    assert result.normalization_result.normalized.iloc[0].day == 2
