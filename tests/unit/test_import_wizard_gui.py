"""Unit tests for the Qt Import Wizard skeleton."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.import_wizard import (
    FileProfileResult,
    ImportPipelineResult,
    ImportWizardSession,
    PipelineDiagnostics,
    RawPreviewModel,
    TimestampCandidate,
    ValidationMessage,
    ValidationSeverity,
)
from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.models import ColumnMappingCandidate
from app.ui.import_wizard import (
    ColumnMappingTableModel,
    ImportWizardDialog,
    PreviewTableModel,
    TimestampCandidateTableModel,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


def _profile(path: str = "sample.csv") -> FileProfileResult:
    preview = RawPreviewModel(
        column_names=["Time", "VA", "Trip"],
        preview_rows=[
            ["2026-01-01 00:00:00", "1.0", "0"],
            ["2026-01-01 00:00:01", "2.0", "1"],
        ],
        row_count_estimate=2,
    )
    return FileProfileResult(
        raw_preview=preview,
        timestamp_candidates=[
            TimestampCandidate(
                "Time",
                0,
                0.95,
                detected_format="%Y-%m-%d %H:%M:%S",
            )
        ],
        column_mappings=[
            ColumnMappingCandidate("Time", 0, "Time", ParameterType.TIMESTAMP, confidence=0.95),
            ColumnMappingCandidate("VA", 1, "VA", ParameterType.VOLTAGE, unit="V", confidence=0.9),
            ColumnMappingCandidate("Trip", 2, "Trip", ParameterType.DIGITAL, confidence=0.85),
        ],
        provider_type="csv",
        delimiter=",",
    )


def test_preview_table_model_populates_cells(qapp) -> None:
    model = PreviewTableModel(_profile().raw_preview)

    assert model.rowCount() == 2
    assert model.columnCount() == 3
    assert model.headerData(1, Qt.Orientation.Horizontal) == "VA"
    assert model.data(model.index(1, 2)) == "1"


def test_timestamp_candidate_model_selects_one(qapp) -> None:
    candidates = [
        TimestampCandidate("A", 0, 0.2),
        TimestampCandidate("B", 1, 0.9),
    ]
    model = TimestampCandidateTableModel(candidates)

    assert model.selected_candidate().column_name == "B"
    model.setData(model.index(0, 0), Qt.CheckState.Checked.value, Qt.ItemDataRole.CheckStateRole)
    assert model.selected_candidate().column_name == "A"


def test_column_mapping_model_edits_overrides(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("VA raw", 0, "VA", ParameterType.VOLTAGE, unit="V")
    ])

    assert model.setData(model.index(0, 2), "VA_EDITED")
    assert model.setData(model.index(0, 3), "current")
    assert model.setData(model.index(0, 4), "A")
    assert model.setData(model.index(0, 0), Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole)

    mapping = model.mappings[0]
    assert mapping.effective_name == "VA_EDITED"
    assert mapping.effective_type == ParameterType.CURRENT
    assert mapping.effective_unit == "A"
    assert mapping.excluded is True


def test_dialog_profiles_file_and_transitions(monkeypatch, qapp, tmp_path) -> None:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    path = tmp_path / "sample.csv"
    path.write_text("Time,VA\n2026-01-01 00:00:00,1\n", encoding="utf-8")
    monkeypatch.setattr(dialog_module, "profile_import_file", lambda *a, **k: _profile(str(path)))

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(path))
        dlg.profile_selected_file()

        assert dlg.session is not None
        assert dlg.session.provider_type == "csv"
        assert dlg.preview_model.rowCount() == 2
        assert dlg.current_step.value == "raw_preview"
    finally:
        dlg.close()
        qapp.processEvents()


def test_dialog_builds_normalization_plan(monkeypatch, qapp, tmp_path) -> None:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    path = tmp_path / "sample.csv"
    path.write_text("Time,VA,Trip\n2026-01-01 00:00:00,1,0\n", encoding="utf-8")
    monkeypatch.setattr(dialog_module, "profile_import_file", lambda *a, **k: _profile(str(path)))

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(path))
        dlg.profile_selected_file()
        dlg._set_step(dialog_module.WizardStep.COLUMN_MAPPING)
        plan = dlg._build_normalization_plan()

        assert plan is not None
        assert "VA" in plan.selected_columns
        assert "Trip" in plan.selected_columns
        assert "Time" in plan.excluded_columns
        assert plan.is_executable
    finally:
        dlg.close()
        qapp.processEvents()


def test_dialog_graceful_profile_failure(monkeypatch, qapp, tmp_path) -> None:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    path = tmp_path / "broken.csv"
    path.write_text("not useful\n", encoding="utf-8")
    result = FileProfileResult(
        raw_preview=RawPreviewModel(column_names=[], preview_rows=[]),
        validation_messages=[
            ValidationMessage(ValidationSeverity.ERROR, "BROKEN", "Malformed CSV")
        ],
    )
    monkeypatch.setattr(dialog_module, "profile_import_file", lambda *a, **k: result)

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(path))
        dlg.profile_selected_file()

        assert dlg.current_step.value == "load_file"
        assert "Malformed CSV" in dlg.load_page.message_view.toPlainText()
    finally:
        dlg.close()
        qapp.processEvents()


def test_dialog_pipeline_success_emits_record(monkeypatch, qapp, tmp_path) -> None:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module
    from app.data.synthetic import make_high_rate_record

    path = tmp_path / "sample.csv"
    path.write_text("Time,VA\n2026-01-01 00:00:00,1\n", encoding="utf-8")
    record = make_high_rate_record(duration_s=0.02, sampling_rate_hz=1000.0).record
    profile = _profile(str(path))
    pipeline_result = ImportPipelineResult(
        session=ImportWizardSession(str(path), "csv"),
        profile=profile,
        selected_candidate=profile.timestamp_candidates[0],
        repair_plan=None,
        normalization_result=None,
        dataset=None,
        bridge_result=None,
        record=record,
        diagnostics=PipelineDiagnostics(
            source_file_path=str(path),
            provider_type="csv",
            normalized_row_count=2,
            analog_channel_count=1,
        ),
        success=True,
        validation_messages=[],
    )
    monkeypatch.setattr(dialog_module, "profile_import_file", lambda *a, **k: profile)
    monkeypatch.setattr(dialog_module, "run_import_pipeline", lambda *a, **k: pipeline_result)

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        dlg.set_source_path(str(path))
        dlg.profile_selected_file()
        dlg._set_step(dialog_module.WizardStep.NORMALIZATION_REVIEW)
        dlg.run_import()
        dlg.open_waveform()

        assert dlg.pipeline_result is pipeline_result
        assert emitted == [record]
    finally:
        dlg.close()
        qapp.processEvents()
