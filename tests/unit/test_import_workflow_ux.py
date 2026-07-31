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


def _write_duplicate_time_csv(path: Path) -> Path:
    path.write_text(
        "Time,Value\n"
        "7/18/2026 12:39,53.63\n"
        "7/18/2026 12:39,-224.84\n"
        "7/18/2026 12:39,-234.07\n"
        "7/18/2026 12:40,-254.95\n",
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


def _profiled_duplicate_time_dialog(qapp, tmp_path: Path) -> ImportWizardDialog:
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg.set_source_path(str(_write_duplicate_time_csv(tmp_path / "duplicate_time.csv")))
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


def test_sample_index_mode_builds_executable_plan(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("sample_index")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        dlg._set_step(WizardStep.TIMESTAMP_SELECT)
        assert dlg.next_button.isEnabled()

        dlg.go_next()
        dlg.go_next()

        assert dlg.current_step == WizardStep.NORMALIZATION_REVIEW
        assert dlg.plan_build_result is not None
        assert dlg.plan_build_result.is_executable
        plan = dlg.plan_build_result.normalization_plan
        assert plan is not None
        assert plan.timestamp_plan.strategy.value == "generate_sample_index"
        review_text = dlg.review_page.summary.toPlainText()
        assert "Time axis mode: sample_index" in review_text
        assert "Timestamp column: None" in review_text
    finally:
        dlg.close()
        qapp.processEvents()


def test_sample_index_mode_shows_axis_details_not_format_override(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("sample_index")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        qapp.processEvents()

        assert dlg.timestamp_page.override_group.title() == "Time Axis Details"
        assert dlg.timestamp_page.selected_column_label.text() == "Not used"
        assert dlg.timestamp_page.detected_format_caption.text() == "Axis basis"
        assert dlg.timestamp_page.detected_format_label.text() == "Sample index"
        assert dlg.timestamp_page.override_status_label.text() == "User-selected mode"
        assert dlg.timestamp_page.manual_format_caption.isHidden()
        assert dlg.timestamp_page.override_edit.isHidden()
        assert dlg.timestamp_page.reset_button.isHidden()
    finally:
        dlg.close()
        qapp.processEvents()


def test_elapsed_mode_shows_user_selected_unit_details(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("elapsed")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        unit_idx = dlg.timestamp_page.elapsed_unit_combo.findData("elapsed_milliseconds")
        dlg.timestamp_page.elapsed_unit_combo.setCurrentIndex(unit_idx)
        qapp.processEvents()

        assert dlg.timestamp_page.override_group.title() == "Time Axis Details"
        assert dlg.timestamp_page.selected_column_label.text() == "Time"
        assert dlg.timestamp_page.detected_format_caption.text() == "Elapsed unit"
        assert dlg.timestamp_page.detected_format_label.text() == "Milliseconds"
        assert dlg.timestamp_page.override_status_label.text() == "User-selected mode"
        assert dlg.timestamp_page.manual_format_caption.isHidden()
        assert dlg.timestamp_page.override_edit.isHidden()
        assert dlg.timestamp_page.reset_button.isHidden()
    finally:
        dlg.close()
        qapp.processEvents()


def test_sample_index_mode_imports_with_sequence_axis(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("sample_index")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        dlg._set_step(WizardStep.TIMESTAMP_SELECT)
        dlg.go_next()
        dlg.go_next()
        dlg.run_import()
        qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
        assert dlg.pipeline_result.dataset is not None
        assert dlg.pipeline_result.dataset.time_axis_mode == "sample_index"
        assert dlg.pipeline_result.record is not None
        assert dlg.pipeline_result.record.timing_info.timing_reference == "sample_index"
        assert dlg.pipeline_result.record.waveform_data["time"].tolist()[:2] == pytest.approx([0.0, 1.0])
    finally:
        dlg.close()
        qapp.processEvents()


def test_synthetic_elapsed_mode_requires_positive_timing_value(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("synthetic_elapsed")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        dlg.timestamp_page.synthetic_value_edit.clear()
        dlg._set_step(WizardStep.NORMALIZATION_REVIEW)

        assert dlg.plan_build_result is not None
        assert not dlg.plan_build_result.is_executable
        assert "Synthetic time requires a positive sample rate" in dlg.review_page.summary.toPlainText()

        dlg.timestamp_page.synthetic_value_edit.setText("100")
        dlg._set_step(WizardStep.NORMALIZATION_REVIEW)

        assert dlg.plan_build_result is not None
        assert dlg.plan_build_result.is_executable
        plan = dlg.plan_build_result.normalization_plan
        assert plan is not None
        assert plan.timestamp_plan.strategy.value == "generate_synthetic_elapsed"
        assert plan.timestamp_plan.sample_rate_hz == pytest.approx(100.0)
    finally:
        dlg.close()
        qapp.processEvents()


def test_synthetic_elapsed_mode_imports_with_generated_seconds(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        idx = dlg.timestamp_page.time_axis_mode_combo.findData("synthetic_elapsed")
        dlg.timestamp_page.time_axis_mode_combo.setCurrentIndex(idx)
        dlg.timestamp_page.synthetic_value_edit.setText("50")
        dlg._set_step(WizardStep.TIMESTAMP_SELECT)
        dlg.go_next()
        dlg.go_next()
        dlg.run_import()
        qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
        assert dlg.pipeline_result.dataset is not None
        assert dlg.pipeline_result.dataset.time_axis_mode == "synthetic_elapsed"
        assert dlg.pipeline_result.record is not None
        assert dlg.pipeline_result.record.timing_info.timing_reference == "synthetic_elapsed"
        assert dlg.pipeline_result.record.waveform_data["time"].tolist()[:2] == pytest.approx([0.0, 0.02])
    finally:
        dlg.close()
        qapp.processEvents()


def test_review_step_builds_plan_when_reached_from_step_list(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        dlg._set_step(WizardStep.NORMALIZATION_REVIEW)

        assert dlg.plan_build_result is not None
        assert dlg.plan_build_result.is_executable
        assert dlg.import_button.isEnabled()
        assert "Plan executable: yes" in dlg.review_page.summary.toPlainText()
        assert dlg.workflow_status_label.text() == "Plan is ready to import."
    finally:
        dlg.close()
        qapp.processEvents()


def test_review_step_explains_non_executable_plan(qapp, tmp_path) -> None:
    dlg = _profiled_dialog(qapp, tmp_path)
    try:
        for mapping in dlg.column_model.visible_mappings:
            row = dlg.column_model.visible_row_for_source(mapping.source_name)
            assert row >= 0
            assert dlg.column_model.setData(
                dlg.column_model.index(row, 0),
                Qt.CheckState.Unchecked.value,
                Qt.ItemDataRole.CheckStateRole,
            )

        dlg._set_step(WizardStep.NORMALIZATION_REVIEW)

        text = dlg.review_page.summary.toPlainText()
        assert not dlg.import_button.isEnabled()
        assert "Plan executable: no" in text
        assert "Blocking issues:" in text
        assert "No signal/data columns are selected for import." in text
        assert dlg.workflow_status_label.text() == "No data columns are included. Include at least one column before importing."
    finally:
        dlg.close()
        qapp.processEvents()


def test_review_step_warns_about_duplicate_timestamps_before_import(qapp, tmp_path) -> None:
    dlg = _profiled_duplicate_time_dialog(qapp, tmp_path)
    try:
        dlg._set_step(WizardStep.NORMALIZATION_REVIEW)

        text = dlg.review_page.summary.toPlainText()
        assert dlg.plan_build_result is not None
        assert dlg.plan_build_result.is_executable
        assert dlg.import_button.isEnabled()
        assert "Warning details:" in text
        assert "duplicate timestamp" in text
        assert "Sample index" in text
        assert "#c62828" in dlg.review_page.summary.toHtml().lower()
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
        row = dlg.column_model.visible_row_for_source("VA")
        assert row >= 0
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
        row = dlg.column_model.visible_row_for_source("Trip")
        assert row >= 0
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
