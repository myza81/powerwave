"""Unit coverage for Import Wizard normalized export UI integration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from app.import_wizard import (
    ExportWriteOptions,
    ExportWriteResult,
    FileProfileResult,
    ImportPipelineResult,
    ImportWizardSession,
    PipelineDiagnostics,
    RawPreviewModel,
    TimestampCandidate,
    ValidationMessage,
    ValidationSeverity,
    WizardStep,
)
from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.export_metadata import metadata_sidecar_path
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.ui.import_wizard.import_wizard_dialog import ImportWizardDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


class DeferredThreadPool:
    def __init__(self) -> None:
        self.workers = []

    def start(self, worker) -> None:
        self.workers.append(worker)


def _dataset(source_path: Path | None = None) -> NormalizedDataset:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="20ms"),
            "VA": [1.0, 1.1, 0.9],
            "Trip": [0, 0, 1],
        }
    )
    params = [
        ParameterMetadata("VA", ParameterType.VOLTAGE, "V", "VA", 1, confidence=0.9),
        ParameterMetadata("Trip", ParameterType.DIGITAL, None, "Trip", 2, confidence=0.8),
    ]
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params,
        excluded_columns=["Time"],
        validation_messages=[],
        diagnostics=AssemblyDiagnostics(total_rows=3, normalized_rows=3),
        source_path=str(source_path) if source_path else None,
        source_file_name=source_path.name if source_path else "source.csv",
        timestamp_repair_strategy="parse_detected_format",
        is_valid=True,
    )


def _pipeline_result(dataset: NormalizedDataset) -> ImportPipelineResult:
    profile = FileProfileResult(
        raw_preview=RawPreviewModel(column_names=["Time", "VA"], preview_rows=[]),
        timestamp_candidates=[
            TimestampCandidate("Time", 0, 0.95, detected_format="%Y-%m-%d %H:%M:%S")
        ],
    )
    return ImportPipelineResult(
        session=ImportWizardSession(dataset.source_path or "source.csv", "csv"),
        profile=profile,
        selected_candidate=profile.timestamp_candidates[0],
        repair_plan=None,
        normalization_result=None,
        dataset=dataset,
        bridge_result=None,
        record=None,
        diagnostics=PipelineDiagnostics(
            source_file_path=dataset.source_path or "source.csv",
            provider_type="csv",
            normalized_row_count=3,
            analog_channel_count=1,
            digital_channel_count=1,
        ),
        success=True,
        validation_messages=[],
    )


def _dialog_with_result(qapp, dataset: NormalizedDataset) -> ImportWizardDialog:
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg._pipeline_result = _pipeline_result(dataset)
    dlg.complete_page.set_result(dlg.pipeline_result)
    dlg._set_step(WizardStep.RENDER_WAVEFORM)
    qapp.processEvents()
    return dlg


def test_save_normalized_action_enabled_on_success(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    try:
        assert dlg.complete_page.save_normalized_button.isEnabled()
    finally:
        dlg.close()
        qapp.processEvents()


def test_export_dialog_default_filename(monkeypatch, qapp, tmp_path) -> None:
    source = tmp_path / "event.csv"
    dlg = _dialog_with_result(qapp, _dataset(source))
    captured = {}

    def fake_save(parent, title, default_path, filters, selected_filter):
        captured["default_path"] = default_path
        captured["selected_filter"] = selected_filter
        return str(tmp_path / "chosen.csv"), "CSV (*.csv)"

    monkeypatch.setattr(
        "app.ui.import_wizard.import_wizard_dialog.QFileDialog.getSaveFileName",
        fake_save,
    )
    try:
        dlg.save_normalized_file()
        assert Path(captured["default_path"]).name == "event_normalized.csv"
        assert captured["selected_filter"] == "CSV (*.csv)"
    finally:
        dlg.close()
        qapp.processEvents()


def test_csv_export_success_through_gui(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    out = tmp_path / "out.csv"
    try:
        dlg.save_normalized_file(str(out), export_format="csv")
        assert dlg.export_result is not None
        assert dlg.export_result.success
        assert out.exists()
        assert "Export complete" in dlg.complete_page.export_status.text()
    finally:
        dlg.close()
        qapp.processEvents()


def test_metadata_sidecar_creation_through_gui(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    out = tmp_path / "out.csv"
    try:
        dlg.save_normalized_file(str(out), export_format="csv", include_metadata_sidecar=True)
        sidecar = metadata_sidecar_path(out)
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["row_count"] == 3
        assert str(sidecar) in dlg.complete_page.export_status.text()
    finally:
        dlg.close()
        qapp.processEvents()


def test_overwrite_false_warning_path(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    out = tmp_path / "out.csv"
    try:
        dlg.save_normalized_file(str(out), export_format="csv", overwrite=True)
        dlg.save_normalized_file(str(out), export_format="csv", overwrite=False)
        assert dlg.export_result is not None
        assert not dlg.export_result.success
        assert any(m.code == "EXPORT_FILE_EXISTS" for m in dlg.export_result.errors())
        assert "EXPORT_FILE_EXISTS" in dlg.complete_page.export_status.text()
    finally:
        dlg.close()
        qapp.processEvents()


def test_overwrite_true_behavior(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    out = tmp_path / "out.csv"
    try:
        dlg.save_normalized_file(str(out), export_format="csv", overwrite=True)
        first = out.read_text(encoding="utf-8")
        out.write_text("old", encoding="utf-8")
        dlg.save_normalized_file(str(out), export_format="csv", overwrite=True)
        assert dlg.export_result is not None
        assert dlg.export_result.success
        assert out.read_text(encoding="utf-8") == first
    finally:
        dlg.close()
        qapp.processEvents()


def test_unsupported_format_handling(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    try:
        dlg.save_normalized_file(str(tmp_path / "bad.xlsx"), export_format="xlsx")
        assert dlg.export_result is not None
        assert not dlg.export_result.success
        assert any(m.code == "EXPORT_UNSUPPORTED_FORMAT" for m in dlg.export_result.errors())
    finally:
        dlg.close()
        qapp.processEvents()


def test_dependency_missing_handling(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    try:
        with patch("app.import_wizard.export_writer._PARQUET_AVAILABLE", False):
            dlg.save_normalized_file(str(tmp_path / "out.parquet"), export_format="parquet")
        assert dlg.export_result is not None
        assert not dlg.export_result.success
        assert any(m.code == "EXPORT_DEPENDENCY_MISSING" for m in dlg.export_result.errors())
    finally:
        dlg.close()
        qapp.processEvents()


def test_export_worker_completion_handling(qapp, tmp_path) -> None:
    pool = DeferredThreadPool()
    dlg = ImportWizardDialog(thread_pool=pool)
    dlg._pipeline_result = _pipeline_result(_dataset(tmp_path / "source.csv"))
    dlg.complete_page.set_result(dlg.pipeline_result)
    try:
        dlg.save_normalized_file(str(tmp_path / "out.csv"), export_format="csv")
        assert pool.workers
        assert dlg._export_running is True
        pool.workers[0].run()
        assert dlg._export_running is False
        assert dlg.export_result is not None
        assert dlg.export_result.success
    finally:
        dlg.close()
        qapp.processEvents()


def test_export_failure_graceful_handling(qapp, tmp_path, monkeypatch) -> None:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))

    def failing_writer(*args, **kwargs):
        return ExportWriteResult(
            success=False,
            output_path=None,
            format_used=None,
            rows_written=0,
            columns_written=0,
            metadata_path=None,
            validation_messages=[
                ValidationMessage(ValidationSeverity.ERROR, "EXPORT_WRITE_ERROR", "permission denied")
            ],
            diagnostics_summary="Export failed during validation.",
        )

    monkeypatch.setattr(dialog_module, "write_normalized_export", failing_writer)
    try:
        dlg.save_normalized_file(str(tmp_path / "out.csv"), export_format="csv")
        assert dlg.export_result is not None
        assert not dlg.export_result.success
        assert "permission denied" in dlg.complete_page.export_status.text()
        assert dlg.current_step.value == "render_waveform"
    finally:
        dlg.close()
        qapp.processEvents()


def test_sidecar_warning_visibility(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    out = tmp_path / "out.csv"
    sidecar = metadata_sidecar_path(out)
    sidecar.write_text("old", encoding="utf-8")
    try:
        dlg.save_normalized_file(str(out), export_format="csv", include_metadata_sidecar=True)
        assert dlg.export_result is not None
        assert dlg.export_result.success
        assert any(m.code == "EXPORT_SIDECAR_EXISTS" for m in dlg.export_result.warnings())
        assert "EXPORT_SIDECAR_EXISTS" in dlg.complete_page.export_status.text()
    finally:
        dlg.close()
        qapp.processEvents()


def test_export_summary_display(qapp, tmp_path) -> None:
    dlg = _dialog_with_result(qapp, _dataset(tmp_path / "source.csv"))
    try:
        dlg.save_normalized_file(str(tmp_path / "out.csv"), export_format="csv")
        status = dlg.complete_page.export_status.text()
        assert "3 rows" in status
        assert "3 column" in status
        assert "CSV" in status
    finally:
        dlg.close()
        qapp.processEvents()
