"""Runtime coverage for Import Wizard export UI integration."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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
        "Time,VA,IA,Trip\n"
        "2026-01-01 00:00:00.000,1.0,10.0,0\n"
        "2026-01-01 00:00:00.020,1.1,10.2,0\n"
        "2026-01-01 00:00:00.040,0.9,10.1,1\n",
        encoding="utf-8",
    )
    return path


def _run_import(path: Path, runtime_qapp, *, thread_pool=None) -> ImportWizardDialog:
    dlg = ImportWizardDialog(thread_pool=thread_pool or ImmediateThreadPool())
    dlg.set_source_path(str(path))
    dlg.profile_selected_file()
    dlg.run_import()
    runtime_qapp.processEvents()
    assert dlg.pipeline_result is not None
    assert dlg.pipeline_result.success
    assert dlg.pipeline_result.dataset is not None
    return dlg


def test_export_ui_runtime_csv_and_sidecar(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "event.csv")
    out = runtime_tmp_path / "event_normalized.csv"
    dlg = _run_import(source, runtime_qapp)
    try:
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


def test_export_ui_runtime_repeatability(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "repeat.csv")
    dlg = _run_import(source, runtime_qapp)
    try:
        for idx in range(3):
            out = runtime_tmp_path / f"repeat_{idx}.csv"
            dlg.save_normalized_file(str(out), export_format="csv")
            runtime_qapp.processEvents()
            assert dlg.export_result is not None
            assert dlg.export_result.success
            assert out.exists()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_export_ui_no_freeze_while_worker_pending(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "pending.csv")
    import_pool = ImmediateThreadPool()
    dlg = _run_import(source, runtime_qapp, thread_pool=import_pool)
    export_pool = DeferredThreadPool()
    dlg._thread_pool = export_pool
    out = runtime_tmp_path / "pending_normalized.csv"
    try:
        dlg.save_normalized_file(str(out), export_format="csv")
        runtime_qapp.processEvents()

        assert export_pool.workers
        assert dlg._export_running is True
        assert dlg.complete_page.save_normalized_button.isEnabled() is False
        assert dlg.cancel_button.isEnabled() is False

        export_pool.workers[0].run()
        runtime_qapp.processEvents()
        assert dlg._export_running is False
        assert dlg.export_result is not None
        assert dlg.export_result.success
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_export_does_not_interfere_with_waveform_handoff(runtime_qapp, runtime_tmp_path) -> None:
    source = _write_csv(runtime_tmp_path / "handoff.csv")
    dlg = _run_import(source, runtime_qapp)
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        dlg.save_normalized_file(str(runtime_tmp_path / "handoff_normalized.csv"), export_format="csv")
        dlg.open_waveform()
        assert emitted
        assert emitted[0].sample_count() == 3
    finally:
        dlg.close()
        runtime_qapp.processEvents()
