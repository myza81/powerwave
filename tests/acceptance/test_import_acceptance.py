"""Acceptance workflows for the Import Wizard subsystem."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.import_wizard.export_metadata import metadata_sidecar_path
from app.ui.import_wizard import ImportWizardDialog


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


class DeferredThreadPool:
    def __init__(self) -> None:
        self.workers = []

    def start(self, worker) -> None:
        self.workers.append(worker)


def _write_csv(path: Path) -> Path:
    path.write_text(
        "Time,Voltage A (kV),Current A (A),MW Total,Trip\n"
        "2026-01-01 00:00:00.000,132.0,450.0,95.0,0\n"
        "2026-01-01 00:00:00.020,132.2,452.0,95.2,0\n"
        "2026-01-01 00:00:00.040,131.8,448.0,94.8,1\n",
        encoding="utf-8",
    )
    return path


def _write_ambiguous_csv(path: Path) -> Path:
    path.write_text(
        "Timestamp,MW Total\n"
        "01/02/2026 00:00:00,100.0\n"
        "01/02/2026 00:00:01,101.0\n"
        "01/02/2026 00:00:02,102.0\n",
        encoding="utf-8",
    )
    return path


def _write_malformed_csv(path: Path) -> Path:
    path.write_text(
        "Timestamp,Voltage A (kV),MW Total\n"
        "2026-01-01 00:00:00.000,132.0,95.0\n"
        "2026-01-01 00:00:00.020,132.1,95.1,EXTRA_FIELD\n"
        "2026-01-01 00:00:00.040,132.2,95.2\n",
        encoding="utf-8",
    )
    return path


def _write_xlsx(path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Event"
    ws.append(["Time", "Voltage A (kV)", "MW Total", "Trip"])
    ws.append(["2026-01-01 00:00:00.000", 132.0, 95.0, 0])
    ws.append(["2026-01-01 00:00:00.020", 132.1, 95.1, 0])
    ws.append(["2026-01-01 00:00:00.040", 131.9, 94.9, 1])
    wb.save(path)
    wb.close()
    return path


def _profile_and_import(path: Path, runtime_qapp, *, thread_pool=None) -> ImportWizardDialog:
    dlg = ImportWizardDialog(thread_pool=thread_pool or ImmediateThreadPool())
    dlg.set_source_path(str(path))
    dlg.profile_selected_file()
    dlg.run_import()
    runtime_qapp.processEvents()
    assert dlg.pipeline_result is not None
    return dlg


def test_csv_import_opens_waveform(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "event.csv")
    dlg = _profile_and_import(source, runtime_qapp)
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        assert dlg.pipeline_result.success
        assert dlg.open_button.isEnabled()

        dlg.open_waveform()
        runtime_qapp.processEvents()

        assert emitted
        record = emitted[0]
        assert record.sample_count() == 3
        assert "time" in record.waveform_data.columns
        assert list(record.waveform_data["time"]) == [0.0, 0.02, 0.04]
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_xlsx_import_exports_normalized_csv_and_sidecar(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_xlsx(runtime_tmp_path / "event.xlsx")
    out = runtime_tmp_path / "event_normalized.csv"
    dlg = _profile_and_import(source, runtime_qapp)
    try:
        assert dlg.pipeline_result.success
        assert dlg.complete_page.save_normalized_button.isEnabled()

        dlg.save_normalized_file(str(out), export_format="csv", include_metadata_sidecar=True)
        runtime_qapp.processEvents()

        assert dlg.export_result is not None
        assert dlg.export_result.success
        assert out.exists()
        assert metadata_sidecar_path(out).exists()
        exported = pd.read_csv(out)
        assert len(exported) == 3
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_timestamp_override_is_authoritative_for_acceptance(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_ambiguous_csv(runtime_tmp_path / "ambiguous.csv")
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(source))
        dlg.profile_selected_file()
        dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")
        runtime_qapp.processEvents()

        assert "User Override" in dlg.timestamp_page.override_status_label.text()
        dlg.run_import()
        runtime_qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
        assert dlg.pipeline_result.record.timing_info.start_time.month == 2
        assert dlg.pipeline_result.record.timing_info.start_time.day == 1
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_malformed_import_shows_diagnostics_without_export(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_malformed_csv(runtime_tmp_path / "malformed.csv")
    dlg = _profile_and_import(source, runtime_qapp)
    try:
        assert not dlg.pipeline_result.success
        assert not dlg.open_button.isEnabled()
        assert not dlg.complete_page.save_normalized_button.isEnabled()
        assert "Import did not complete successfully" in dlg.workflow_status_label.text()
        assert "failed" in dlg.complete_page.diagnostics_panel.plain_text().lower()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_repeated_import_export_acceptance(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "repeat.csv")
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        for idx in range(2):
            dlg.set_source_path(str(source))
            dlg.profile_selected_file()
            dlg.run_import()
            runtime_qapp.processEvents()

            assert dlg.pipeline_result is not None
            assert dlg.pipeline_result.success
            out = runtime_tmp_path / f"repeat_{idx}.csv"
            dlg.save_normalized_file(str(out), export_format="csv", include_metadata_sidecar=True, overwrite=True)
            runtime_qapp.processEvents()

            assert dlg.export_result is not None
            assert dlg.export_result.success
            assert out.exists()
            assert metadata_sidecar_path(out).exists()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_dialog_close_during_pending_worker_is_safe(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "pending.csv")
    pool = DeferredThreadPool()
    dlg = ImportWizardDialog(thread_pool=pool)
    try:
        dlg.set_source_path(str(source))
        dlg.profile_selected_file()
        assert pool.workers
        pool.workers.pop(0).run()
        runtime_qapp.processEvents()

        dlg.run_import()
        runtime_qapp.processEvents()
        assert pool.workers

        dlg.close()
        runtime_qapp.processEvents()
        pool.workers.pop(0).run()
        runtime_qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
    finally:
        dlg.close()
        runtime_qapp.processEvents()
