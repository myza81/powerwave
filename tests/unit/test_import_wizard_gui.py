"""Unit tests for the Qt Import Wizard skeleton."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QHeaderView, QStyleOptionViewItem

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
from app.ui.import_wizard.column_mapping_model import ParameterTypeDelegate, UnitDelegate
from app.ui.import_wizard.wizard_pages import ColumnMappingPage


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


def test_column_mapping_type_sets_default_unit(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Tie Line", 0, "tie_line", ParameterType.UNKNOWN)
    ])

    assert model.setData(model.index(0, 3), "mw")

    mapping = model.mappings[0]
    assert mapping.effective_type == ParameterType.MW
    assert mapping.effective_unit == "MW"


@pytest.mark.parametrize(
    ("ptype", "expected_unit"),
    [
        (ParameterType.ROCOF, "Hz/s"),
        (ParameterType.FREQUENCY, "Hz"),
        (ParameterType.VOLTAGE, "V"),
        (ParameterType.CURRENT, "A"),
        (ParameterType.MW, "MW"),
        (ParameterType.MVAR, "Mvar"),
        (ParameterType.DIGITAL, ""),
    ],
)
def test_column_mapping_type_change_updates_unit_default(qapp, ptype, expected_unit) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Signal", 0, "signal", ParameterType.UNKNOWN, unit="custom")
    ])

    assert model.setData(model.index(0, 3), ptype.value)

    mapping = model.mappings[0]
    assert mapping.effective_type == ptype
    assert mapping.effective_unit == expected_unit
    assert model.data(model.index(0, 4), Qt.ItemDataRole.EditRole) == expected_unit


def test_column_mapping_type_change_notifies_unit_cell(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Signal", 0, "signal", ParameterType.UNKNOWN)
    ])
    changed: list[tuple[int, int]] = []
    model.dataChanged.connect(lambda top_left, bottom_right, _roles: changed.append((top_left.column(), bottom_right.column())))

    assert model.setData(model.index(0, 3), ParameterType.ROCOF.value)

    assert (4, 4) in changed


def test_column_mapping_unit_can_override_type_default_afterward(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Voltage", 0, "voltage", ParameterType.UNKNOWN)
    ])

    assert model.setData(model.index(0, 3), ParameterType.VOLTAGE.value)
    assert model.data(model.index(0, 4), Qt.ItemDataRole.EditRole) == "V"
    assert model.setData(model.index(0, 4), "kV")

    assert model.mappings[0].effective_unit == "kV"


def test_column_mapping_model_displays_readable_type_label(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Tie Line", 0, "tie_line", ParameterType.ROCOF)
    ])

    type_index = model.index(0, 3)

    assert model.data(type_index) == "ROCOF (Hz/s)"
    assert model.data(type_index, Qt.ItemDataRole.EditRole) == "rocof"
    assert "ROCOF (Hz/s)" in model.data(type_index, Qt.ItemDataRole.ToolTipRole)


def test_column_mapping_model_hides_timestamp_rows_from_table(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Time", 0, "timestamp_time", ParameterType.TIMESTAMP, confidence=0.95),
        ColumnMappingCandidate("Frequency", 1, "frequency_1", ParameterType.FREQUENCY, unit="Hz"),
        ColumnMappingCandidate("Tie Line", 2, "rocof_2", ParameterType.ROCOF, unit="Hz/s"),
    ])

    assert len(model.mappings) == 3
    assert model.rowCount() == 2
    assert model.hidden_time_axis_count() == 1
    assert [mapping.source_name for mapping in model.visible_mappings] == ["Frequency", "Tie Line"]
    assert model.visible_row_for_source("Tie Line") == 1
    assert model.visible_row_for_source("Time") == -1
    assert model.data(model.index(0, 1)) == "Frequency"


def test_column_mapping_page_sizes_editable_columns_for_dropdowns(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("2 - TIE LINE 1", 0, "rocof_2_tie_line_1", ParameterType.ROCOF)
    ])
    page = ColumnMappingPage(model)
    try:
        header = page.table.horizontalHeader()

        assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
        assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
        assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Fixed
        assert header.sectionResizeMode(4) == QHeaderView.ResizeMode.Fixed
        assert page.table.minimumWidth() < 860
        assert page.table.columnWidth(3) >= 220
        assert page.table.columnWidth(4) >= 140
        assert page.table.verticalHeader().defaultSectionSize() >= 34
        assert page.table.textElideMode() == Qt.TextElideMode.ElideMiddle
    finally:
        page.deleteLater()


def test_column_mapping_page_shows_type_and_unit_as_persistent_selects(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("2 - TIE LINE 1", 0, "rocof_2_tie_line_1", ParameterType.ROCOF, unit="Hz/s")
    ])
    page = ColumnMappingPage(model)
    try:
        page._show_mapping_selects()
        qapp.processEvents()

        assert page.table.isPersistentEditorOpen(model.index(0, 3))
        assert page.table.isPersistentEditorOpen(model.index(0, 4))
    finally:
        page.deleteLater()


def test_column_mapping_page_summarizes_hidden_time_axis(qapp) -> None:
    session = ImportWizardSession(source_path="sample.csv", provider_type="csv")
    session.selected_timestamp_column = "Time"
    session.column_mappings = [
        ColumnMappingCandidate("Time", 0, "timestamp_time", ParameterType.TIMESTAMP, confidence=0.95),
        ColumnMappingCandidate("Frequency", 1, "frequency_1", ParameterType.FREQUENCY, unit="Hz"),
    ]
    model = ColumnMappingTableModel(session.column_mappings)
    page = ColumnMappingPage(model)
    try:
        page.refresh(session)

        assert page.time_axis_label.text() == "Time axis: Time"
        assert "1 included column(s)" in page.message_label.text()
        assert "1 time-axis column hidden" in page.message_label.text()
    finally:
        page.deleteLater()


def test_unit_delegate_uses_editable_combo(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("VA", 0, "VA", ParameterType.VOLTAGE, unit="kV")
    ])
    delegate = UnitDelegate()
    editor = delegate.createEditor(None, QStyleOptionViewItem(), model.index(0, 4))
    try:
        assert editor.isEditable()
        assert editor.findText("kV") >= 0
        assert editor.findText("MW") >= 0
    finally:
        editor.deleteLater()


def test_unit_delegate_editor_covers_full_cell(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Tie Line", 0, "tie_line", ParameterType.ROCOF, unit="Hz/s")
    ])
    delegate = UnitDelegate()
    option = QStyleOptionViewItem()
    option.rect = option.rect.adjusted(0, 0, 140, 34)
    editor = delegate.createEditor(None, option, model.index(0, 4))
    try:
        delegate.setEditorData(editor, model.index(0, 4))
        delegate.updateEditorGeometry(editor, option, model.index(0, 4))

        assert editor.isEditable()
        assert editor.currentText() == "Hz/s"
        assert editor.autoFillBackground()
        assert editor.geometry() == option.rect
    finally:
        editor.deleteLater()


def test_type_delegate_editor_covers_full_cell(qapp) -> None:
    model = ColumnMappingTableModel([
        ColumnMappingCandidate("Tie Line", 0, "tie_line", ParameterType.ROCOF)
    ])
    delegate = ParameterTypeDelegate()
    option = QStyleOptionViewItem()
    option.rect = option.rect.adjusted(0, 0, 220, 34)
    editor = delegate.createEditor(None, option, model.index(0, 3))
    try:
        delegate.setEditorData(editor, model.index(0, 3))
        delegate.updateEditorGeometry(editor, option, model.index(0, 3))

        assert editor.currentText() == "ROCOF (Hz/s)"
        assert editor.autoFillBackground()
        assert editor.geometry() == option.rect
    finally:
        editor.deleteLater()


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
        plan_result = dlg._rebuild_execution_plan()

        assert plan_result is not None
        plan = plan_result.normalization_plan
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
    monkeypatch.setattr(dialog_module, "run_import_pipeline_with_plan", lambda *a, **k: pipeline_result)

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


def test_dialog_finished_signal_accepts_the_qdialog(qapp) -> None:
    """ImportWizardDialog is now a thin QDialog wrapper around ImportWizardWidget;
    confirm the wizard's `finished` signal still drives the wrapper's own Qt
    accept state, since that's the only modal contract callers/tests rely on.
    """
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        assert dlg.result() != dlg.DialogCode.Accepted
        dlg.wizard.finished.emit()
        assert dlg.result() == dlg.DialogCode.Accepted
    finally:
        qapp.processEvents()


def test_dialog_reject_declined_keeps_dialog_open(monkeypatch, qapp) -> None:
    """The wrapper's reject() must go through the wizard's own discard-risk
    check (request_close) rather than closing unconditionally — otherwise a
    caller relying on the old modal QDialog.reject() contract could lose an
    in-progress import without any confirmation.
    """
    from PyQt6.QtWidgets import QMessageBox

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.wizard._import_running = True
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
        )

        dlg.reject()

        assert dlg.wizard._closing is False
        assert dlg.result() != dlg.DialogCode.Accepted
    finally:
        qapp.processEvents()


def test_dialog_reject_confirmed_closes_dialog(monkeypatch, qapp) -> None:
    from PyQt6.QtWidgets import QMessageBox

    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.wizard._import_running = True
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )

        dlg.reject()

        assert dlg.wizard._closing is True
    finally:
        qapp.processEvents()


def test_dialog_attribute_access_proxies_to_underlying_wizard(qapp) -> None:
    """Regression test for the __getattr__/__setattr__ proxy that keeps the
    dialog wrapper API-compatible with the embeddable ImportWizardWidget.
    """
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        assert dlg.load_page is dlg.wizard.load_page
        assert dlg.timestamp_page is dlg.wizard.timestamp_page

        dlg.set_source_path("proxied.csv")
        assert dlg.wizard.load_page.path_edit.text() == "proxied.csv"
    finally:
        dlg.close()
        qapp.processEvents()
