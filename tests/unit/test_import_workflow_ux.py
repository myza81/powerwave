"""Unit coverage for Import Wizard workflow UX hardening."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.import_wizard import WizardStep
from app.import_wizard.column_mapping import ParameterType
from app.ui.import_wizard.import_wizard_dialog import ImportWizardDialog
from app.ui.import_wizard.workflow_state import evaluate_workflow_actions


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


def _write_csv(path: Path) -> Path:
    path.write_text(
        "Time,VA,Trip\n"
        "2026-01-01 00:00:00.000,1.0,0\n"
        "2026-01-01 00:00:00.020,1.1,1\n",
        encoding="utf-8",
    )
    return path


def _profiled_dialog(qapp, tmp_path: Path) -> ImportWizardDialog:
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg.set_source_path(str(_write_csv(tmp_path / "event.csv")))
    dlg.profile_selected_file()
    qapp.processEvents()
    assert dlg.session is not None
    return dlg


def _imported_dialog(qapp, tmp_path: Path) -> ImportWizardDialog:
    dlg = _profiled_dialog(qapp, tmp_path)
    dlg._set_step(WizardStep.NORMALIZATION_REVIEW)
    dlg.run_import()
    qapp.processEvents()
    assert dlg.pipeline_result is not None
    assert dlg.pipeline_result.success
    return dlg


def test_workflow_state_blocks_invalid_actions() -> None:
    state = evaluate_workflow_actions(
        step=WizardStep.TIMESTAMP_SELECT,
        step_index=2,
        has_path=True,
        has_profile=True,
        profile_has_errors=False,
        has_timestamp_candidates=False,
        has_included_columns=True,
        plan_is_executable=False,
        import_running=False,
        export_running=False,
        import_success=False,
        export_ready=False,
        settings_dirty_since_import=False,
    )

    assert not state.can_go_next
    assert "No timestamp candidate" in state.status_text


def test_button_enablement_follows_current_step(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        dlg._set_step(WizardStep.TIMESTAMP_SELECT)
        assert dlg.next_button.isEnabled()
        assert not dlg.import_button.isVisible()

        dlg._set_step(WizardStep.COLUMN_MAPPING)
        assert dlg.next_button.isEnabled()
        dlg.go_next()
        assert dlg.current_step == WizardStep.NORMALIZATION_REVIEW
        assert dlg.import_button.isEnabled()
    finally:
        dlg.close()
        qapp.processEvents()


def test_timestamp_override_invalidates_import_and_export_state(qapp, tmp_path) -> None:
    dlg = _imported_dialog(qapp, tmp_path)
    try:
        assert dlg.complete_page.save_normalized_button.isEnabled()
        dlg.timestamp_page.override_edit.setText("%Y-%m-%d %H:%M:%S.%f")
        qapp.processEvents()

        assert dlg.pipeline_result is None
        assert dlg.export_result is None
        assert not dlg.complete_page.save_normalized_button.isEnabled()
        assert "Re-import required" in dlg.workflow_status_label.text()
        assert dlg.timestamp_page.override_status_label.text() == "User Override"
    finally:
        dlg.close()
        qapp.processEvents()


def test_mapping_edit_invalidates_export_readiness(qapp, tmp_path) -> None:
    dlg = _imported_dialog(qapp, tmp_path)
    try:
        assert dlg.complete_page.save_normalized_button.isEnabled()
        row = next(i for i, mapping in enumerate(dlg.column_model.mappings) if mapping.source_name == "VA")
        assert dlg.column_model.setData(dlg.column_model.index(row, 2), "VA_USER")

        assert dlg.pipeline_result is None
        assert not dlg.complete_page.save_normalized_button.isEnabled()
        assert "User Override" in str(dlg.column_model.data(dlg.column_model.index(row, 2)))
    finally:
        dlg.close()
        qapp.processEvents()


def test_new_file_resets_workflow_state(qapp, tmp_path) -> None:
    dlg = _imported_dialog(qapp, tmp_path)
    try:
        assert dlg.pipeline_result is not None
        new_path = _write_csv(tmp_path / "next.csv")
        dlg.set_source_path(str(new_path))

        assert dlg.session is None
        assert dlg.pipeline_result is None
        assert dlg.export_result is None
        assert dlg.current_step == WizardStep.LOAD_FILE
        assert not dlg.complete_page.save_normalized_button.isEnabled()
    finally:
        dlg.close()
        qapp.processEvents()


def test_discard_prompt_on_explicit_close_action(monkeypatch, qapp, tmp_path) -> None:
    dlg = _imported_dialog(qapp, tmp_path)
    calls = []

    def fake_question(*args, **kwargs):
        calls.append(args)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(
        "app.ui.import_wizard.import_wizard_dialog.QMessageBox.question",
        fake_question,
    )
    try:
        dlg.request_close()
        assert calls
        assert dlg._closing is False
    finally:
        dlg.close()
        qapp.processEvents()


def test_user_override_visibility_in_mapping_model(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        row = next(i for i, mapping in enumerate(dlg.column_model.mappings) if mapping.source_name == "Trip")
        type_index = dlg.column_model.index(row, 3)
        unit_index = dlg.column_model.index(row, 4)
        include_index = dlg.column_model.index(row, 0)

        assert dlg.column_model.setData(type_index, ParameterType.DIGITAL.value)
        assert dlg.column_model.setData(unit_index, "state")
        assert dlg.column_model.setData(include_index, Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole)

        assert "User Override" in str(dlg.column_model.data(type_index))
        assert "User Override" in str(dlg.column_model.data(unit_index))
        tooltip = dlg.column_model.data(unit_index, Qt.ItemDataRole.ToolTipRole)
        assert "Engineering unit is a user override" in tooltip
        assert "excluded" in tooltip
    finally:
        dlg.close()
        qapp.processEvents()
