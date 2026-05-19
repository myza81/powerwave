"""Runtime coverage for Import Wizard workflow hardening."""
from __future__ import annotations

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


def test_repeated_import_export_cycles_stay_stable(runtime_qapp, runtime_tmp_path) -> None:
    source = generate_import_stress_csv(
        runtime_tmp_path / "repeat.csv",
        StressSampleConfig(row_count=200, digital_text_values=True),
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        for idx in range(2):
            dlg.set_source_path(str(source))
            dlg.profile_selected_file()
            dlg.run_import()
            runtime_qapp.processEvents()

            assert dlg.pipeline_result is not None
            assert dlg.pipeline_result.success
            assert dlg.open_button.isEnabled()
            assert dlg.complete_page.save_normalized_button.isEnabled()

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


def test_worker_completion_after_close_is_safe(runtime_qapp, runtime_tmp_path) -> None:
    source = generate_import_stress_csv(
        runtime_tmp_path / "pending_close.csv",
        StressSampleConfig(row_count=200),
    )
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


def test_rapid_navigation_does_not_enable_invalid_import(runtime_qapp, runtime_tmp_path) -> None:
    source = generate_import_stress_csv(
        runtime_tmp_path / "nav.csv",
        StressSampleConfig(row_count=50),
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(source))
        dlg.profile_selected_file()
        for _ in range(3):
            dlg.go_next()
            runtime_qapp.processEvents()
        assert dlg.current_step.value in {"column_mapping", "normalization_review"}
        if dlg.current_step.value == "normalization_review":
            assert dlg.import_button.isEnabled()
        else:
            assert not dlg.import_button.isVisible()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_failed_import_shows_operational_state(runtime_qapp, runtime_tmp_path) -> None:
    bad = runtime_tmp_path / "bad.csv"
    bad.write_text(
        "Timestamp,Voltage A (kV),MW Total\n"
        "2026-01-01 00:00:00.000,132.0,95.0\n"
        "2026-01-01 00:00:00.020,132.1,95.1,EXTRA_FIELD\n"
        "2026-01-01 00:00:00.040,132.2,95.2\n",
        encoding="utf-8",
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    try:
        dlg.set_source_path(str(bad))
        dlg.profile_selected_file()
        dlg.run_import()
        runtime_qapp.processEvents()

        assert dlg.pipeline_result is not None
        assert not dlg.pipeline_result.success
        assert not dlg.open_button.isEnabled()
        assert not dlg.complete_page.save_normalized_button.isEnabled()
        assert "Import did not complete successfully" in dlg.workflow_status_label.text()
    finally:
        dlg.close()
        runtime_qapp.processEvents()
