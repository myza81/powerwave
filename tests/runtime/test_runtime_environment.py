"""Runtime environment hygiene tests for Phase 8.55J."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRunnable, QThreadPool

from app.import_wizard.import_pipeline import run_import_pipeline
from app.testing.temp_runtime import isolated_runtime_dir, runtime_temp_dir, safe_rmtree
from app.ui.import_wizard import ImportWizardDialog


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


class NoopRunnable(QRunnable):
    def run(self) -> None:
        return


def _write_csv(path: Path) -> Path:
    path.write_text(
        "Time,VA,IA,Trip\n"
        "2026-01-01 00:00:00.000,1.0,10.0,0\n"
        "2026-01-01 00:00:00.020,1.1,10.2,0\n"
        "2026-01-01 00:00:00.040,0.9,10.1,1\n",
        encoding="utf-8",
    )
    return path


def _write_xlsx(path: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Samples"
    ws.append(["Time", "VA", "IA", "Trip"])
    ws.append(["2026-01-01 00:00:00.000", 1.0, 10.0, 0])
    ws.append(["2026-01-01 00:00:00.020", 1.1, 10.2, 0])
    ws.append(["2026-01-01 00:00:00.040", 0.9, 10.1, 1])
    wb.save(path)
    wb.close()
    return path


def _run_dialog_import(path: Path, runtime_qapp) -> int:
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        dlg.set_source_path(str(path))
        dlg.profile_selected_file()
        dlg.run_import()
        dlg.open_waveform()
        assert emitted
        return emitted[0].sample_count()
    finally:
        dlg.close()
        runtime_qapp.processEvents()


def test_isolated_runtime_temp_creation_and_cleanup() -> None:
    path = isolated_runtime_dir("runtime-env")
    marker = path / "marker.txt"
    marker.write_text("ok", encoding="utf-8")

    result = safe_rmtree(path)

    assert result.removed
    assert not path.exists()


def test_runtime_temp_context_cleans_csv_artifacts() -> None:
    with runtime_temp_dir("csv-cleanup") as root:
        csv = _write_csv(root / "recording.csv")
        assert csv.exists()
        result = run_import_pipeline(str(csv))
        assert result.success
        captured_root = root

    assert not captured_root.exists()


def test_runtime_temp_context_cleans_xlsx_artifacts() -> None:
    with runtime_temp_dir("xlsx-cleanup") as root:
        xlsx = _write_xlsx(root / "recording.xlsx")
        assert xlsx.exists()
        result = run_import_pipeline(str(xlsx))
        assert result.success
        captured_root = root

    assert not captured_root.exists()


def test_qthreadpool_worker_cleanup(runtime_qapp) -> None:
    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    pool.start(NoopRunnable())

    assert pool.waitForDone(5000)
    runtime_qapp.processEvents()


def test_import_wizard_runtime_repeatability(runtime_qapp) -> None:
    with runtime_temp_dir("wizard-repeat") as root:
        first = _write_csv(root / "first.csv")
        second = _write_csv(root / "second.csv")

        assert _run_dialog_import(first, runtime_qapp) == 3
        assert _run_dialog_import(second, runtime_qapp) == 3
        captured_root = root

    assert not captured_root.exists()


def test_timestamp_override_runtime_repeatability(runtime_qapp) -> None:
    with runtime_temp_dir("override-repeat") as root:
        csv = root / "override.csv"
        csv.write_text(
            "Time,MW\n"
            "01/02/2026 00:00:00,100.0\n"
            "01/02/2026 00:00:01,101.0\n",
            encoding="utf-8",
        )

        for _ in range(2):
            dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
            try:
                dlg.set_source_path(str(csv))
                dlg.profile_selected_file()
                dlg.timestamp_page.override_edit.setText("%d/%m/%Y %H:%M:%S")
                dlg.run_import()
                assert dlg.pipeline_result is not None
                assert dlg.pipeline_result.success
                assert dlg.pipeline_result.record.timing_info.start_time.month == 2
            finally:
                dlg.close()
                runtime_qapp.processEvents()
        captured_root = root

    assert not captured_root.exists()


def test_safe_cleanup_reports_locked_path_without_raising(runtime_tmp_path) -> None:
    locked = runtime_tmp_path / "locked.txt"
    locked.write_text("held", encoding="utf-8")

    with locked.open("r", encoding="utf-8") as handle:
        result = safe_rmtree(locked, retries=0)
        if result.removed:
            assert not locked.exists()
        else:
            assert result.error is not None
            assert handle.read() == "held"
