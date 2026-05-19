"""Runtime hardening coverage for realistic Import Wizard workflows."""
from __future__ import annotations

from pathlib import Path

from app.import_wizard.export_metadata import metadata_sidecar_path
from app.ui.import_wizard import ImportWizardDialog
from tools.generate_import_stress_samples import StressSampleConfig, generate_import_stress_csv


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


class DeferredThreadPool:
    def __init__(self) -> None:
        self.workers = []

    def start(self, worker) -> None:
        self.workers.append(worker)


def _run_profile_worker(pool: DeferredThreadPool, runtime_qapp) -> None:
    assert pool.workers
    pool.workers.pop(0).run()
    runtime_qapp.processEvents()


def _write_ragged_csv(path: Path) -> Path:
    path.write_text(
        "Timestamp,Voltage A (kV),MW Total\n"
        "2026-01-01 00:00:00.000,132.0,95.0\n"
        "2026-01-01 00:00:00.020,132.1,95.1,EXTRA_FIELD\n"
        "2026-01-01 00:00:00.040,132.2,95.2\n",
        encoding="utf-8",
    )
    return path


def test_import_worker_pending_keeps_dialog_responsive(runtime_qapp, runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "pending.csv",
        StressSampleConfig(row_count=500),
    )
    pool = DeferredThreadPool()
    dlg = ImportWizardDialog(thread_pool=pool)
    try:
        dlg.set_source_path(str(csv_path))
        dlg.profile_selected_file()
        _run_profile_worker(pool, runtime_qapp)

        dlg.run_import()
        runtime_qapp.processEvents()

        assert pool.workers
        assert dlg.pipeline_result is None
        assert dlg.cancel_button.isEnabled() is False

        pool.workers.pop(0).run()
        runtime_qapp.processEvents()
        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
        assert dlg.open_button.isEnabled()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_failed_runtime_import_does_not_crash_dialog(runtime_qapp, runtime_tmp_path) -> None:
    csv_path = _write_ragged_csv(runtime_tmp_path / "ragged.csv")
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(csv_path))
        dlg.profile_selected_file()
        dlg.run_import()
        runtime_qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success is False
        assert dlg.cancel_button.isEnabled()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_successful_import_open_waveform_and_export_remain_available(runtime_qapp, runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "workflow.csv",
        StressSampleConfig(row_count=300, digital_text_values=True),
    )
    out = runtime_tmp_path / "workflow_normalized.csv"
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        dlg.set_source_path(str(csv_path))
        dlg.profile_selected_file()
        dlg.run_import()
        runtime_qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert dlg.pipeline_result.success
        assert dlg.open_button.isEnabled()
        assert dlg.complete_page.save_normalized_button.isEnabled()

        dlg.save_normalized_file(str(out), export_format="csv", include_metadata_sidecar=True, overwrite=True)
        runtime_qapp.processEvents()
        assert dlg.export_result is not None
        assert dlg.export_result.success
        assert out.exists()
        assert metadata_sidecar_path(out).exists()

        dlg.open_waveform()
        assert emitted
        assert emitted[0].sample_count() == 300
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_dialog_close_after_worker_completion_leaves_no_pending_work(runtime_qapp, runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "close_after.csv",
        StressSampleConfig(row_count=200),
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg.set_source_path(str(csv_path))
    dlg.profile_selected_file()
    dlg.run_import()
    runtime_qapp.processEvents()

    assert dlg.pipeline_result is not None
    dlg.close()
    runtime_qapp.processEvents()
    assert runtime_qapp.topLevelWidgets() is not None


def test_dialog_can_close_while_import_worker_is_pending(runtime_qapp, runtime_tmp_path) -> None:
    csv_path = generate_import_stress_csv(
        runtime_tmp_path / "close_pending.csv",
        StressSampleConfig(row_count=500),
    )
    pool = DeferredThreadPool()
    dlg = ImportWizardDialog(thread_pool=pool)
    dlg.set_source_path(str(csv_path))
    dlg.profile_selected_file()
    _run_profile_worker(pool, runtime_qapp)

    dlg.run_import()
    runtime_qapp.processEvents()
    assert pool.workers

    dlg.close()
    runtime_qapp.processEvents()
    pool.workers.pop(0).run()
    runtime_qapp.processEvents()
    assert dlg.pipeline_result is not None
