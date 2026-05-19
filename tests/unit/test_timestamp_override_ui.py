"""Unit coverage for Import Wizard timestamp format overrides."""
from __future__ import annotations

import os
import sys
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationSeverity
from app.import_wizard.file_profiler import FileProfileResult
from app.import_wizard.models import ColumnMappingCandidate, RawPreviewModel, TimestampCandidate
from app.import_wizard.pipeline_plan_builder import build_execution_plan
from app.import_wizard.timestamp_contracts import TimestampRepairStrategy
from app.import_wizard.timestamp_format_validator import validate_user_timestamp_format
from app.ui.import_wizard.import_wizard_dialog import ImportWizardDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture()
def local_tmp():
    path = Path("test_artifacts") / f"timestamp_override_ui_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


def _profile() -> FileProfileResult:
    preview = RawPreviewModel(
        column_names=["Time", "MW"],
        preview_rows=[
            ["01/02/2026 03:04:05", "10.0"],
            ["01/02/2026 03:04:06", "10.1"],
            ["bad", "10.2"],
        ],
        row_count_estimate=3,
    )
    return FileProfileResult(
        raw_preview=preview,
        provider_type="csv",
        delimiter=",",
        timestamp_candidates=[
            TimestampCandidate(
                "Time",
                0,
                0.95,
                detected_format="%Y-%m-%d %H:%M:%S",
                example_values=[
                    "01/02/2026 03:04:05",
                    "01/02/2026 03:04:06",
                    "bad",
                ],
                user_selected=True,
            )
        ],
        column_mappings=[
            ColumnMappingCandidate("Time", 0, "Time", ParameterType.TIMESTAMP),
            ColumnMappingCandidate("MW", 1, "MW", ParameterType.MW, unit="MW"),
        ],
    )


def _dialog(monkeypatch, local_tmp) -> ImportWizardDialog:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    path = local_tmp / "override.csv"
    path.write_text(
        "Time,MW\n"
        "01/02/2026 03:04:05,10.0\n"
        "01/02/2026 03:04:06,10.1\n"
        "bad,10.2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dialog_module, "profile_import_file", lambda *a, **k: _profile())
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg.set_source_path(str(path))
    dlg.profile_selected_file()
    return dlg


def test_valid_override_preview_feedback() -> None:
    result = validate_user_timestamp_format(
        "%d/%m/%Y %H:%M:%S",
        ["01/02/2026 03:04:05", "01/02/2026 03:04:06"],
        affected_column="Time",
    )

    assert result.is_valid
    assert result.parsed_count == 2
    assert result.validation_messages[0].code == "PLAN_TIMESTAMP_FORMAT_VALID"


def test_invalid_override_validation_error() -> None:
    result = validate_user_timestamp_format(
        "%Y-%m-%d %H:%M:%S",
        ["01/02/2026 03:04:05", "01/02/2026 03:04:06"],
        affected_column="Time",
    )

    assert not result.is_valid
    assert result.validation_messages[0].severity == ValidationSeverity.ERROR
    assert result.validation_messages[0].code == "PLAN_TIMESTAMP_PARSE_FAILED"


def test_partial_override_validation_warning() -> None:
    result = validate_user_timestamp_format(
        "%d/%m/%Y %H:%M:%S",
        ["01/02/2026 03:04:05", "bad"],
        affected_column="Time",
    )

    assert result.is_valid
    assert result.parsed_count == 1
    assert result.validation_messages[0].severity == ValidationSeverity.WARNING
    assert result.validation_messages[0].code == "PLAN_PARTIAL_TIMESTAMP_PARSE"


def test_manual_override_stored_in_session(monkeypatch, qapp, local_tmp) -> None:
    dlg = _dialog(monkeypatch, local_tmp)
    try:
        dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")

        plan = dlg.session.timestamp_repair_plan
        assert plan.strategy == TimestampRepairStrategy.PARSE_USER_FORMAT
        assert plan.user_format == "%d/%m/%Y %H:%M:%S"
        assert plan.detected_format == "%Y-%m-%d %H:%M:%S"
    finally:
        dlg.close()
        qapp.processEvents()


def test_clearing_override_restores_detected_behavior(monkeypatch, qapp, local_tmp) -> None:
    dlg = _dialog(monkeypatch, local_tmp)
    try:
        dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")
        dlg.timestamp_page.override_edit.clear()

        plan = dlg.session.timestamp_repair_plan
        assert plan.strategy == TimestampRepairStrategy.PARSE_DETECTED_FORMAT
        assert plan.detected_format == "%Y-%m-%d %H:%M:%S"
        assert plan.user_format is None
    finally:
        dlg.close()
        qapp.processEvents()


def test_builder_selects_user_format_strategy(monkeypatch, qapp, local_tmp) -> None:
    dlg = _dialog(monkeypatch, local_tmp)
    try:
        dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")
        result = build_execution_plan(dlg.session, list(dlg.column_model.mappings))

        assert result.is_executable
        assert result.normalization_plan.timestamp_plan.strategy == TimestampRepairStrategy.PARSE_USER_FORMAT
        assert result.normalization_plan.timestamp_plan.user_format == "%d/%m/%Y %H:%M:%S"
    finally:
        dlg.close()
        qapp.processEvents()


def test_invalid_override_blocks_dialog_execution(monkeypatch, qapp, local_tmp) -> None:
    dlg = _dialog(monkeypatch, local_tmp)
    try:
        dlg.timestamp_page.override_edit.setText("%Y-%m-%d %H:%M:%S")

        with patch("app.ui.import_wizard.import_wizard_dialog.QMessageBox.critical") as critical:
            dlg.run_import()
            assert critical.called

        assert dlg.pipeline_result is None
        assert dlg.plan_build_result is not None
        assert not dlg.plan_build_result.is_executable
    finally:
        dlg.close()
        qapp.processEvents()


def test_override_change_invalidates_stale_plan(monkeypatch, qapp, local_tmp) -> None:
    dlg = _dialog(monkeypatch, local_tmp)
    try:
        dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")
        clean = dlg._rebuild_execution_plan()
        assert clean is not None and clean.is_executable

        dlg.timestamp_page.override_edit.setText("%Y-%m-%d %H:%M:%S")

        assert dlg.plan_build_result is None
        assert dlg.pipeline_result is None
        assert dlg.session.normalization_plan is None
    finally:
        dlg.close()
        qapp.processEvents()
