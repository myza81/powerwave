"""Runtime Qt tests for Import Wizard end-to-end skeleton behavior."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.ui.import_wizard import ImportWizardDialog


def _qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


def test_import_wizard_runtime_imports_csv_and_emits_record(tmp_path) -> None:
    app = _qapp()
    csv_path = tmp_path / "runtime_import.csv"
    csv_path.write_text(
        "Time,VA,IA,Trip\n"
        "2026-01-01 00:00:00.000,1.0,10.0,0\n"
        "2026-01-01 00:00:00.020,1.1,10.2,0\n"
        "2026-01-01 00:00:00.040,0.9,10.1,1\n",
        encoding="utf-8",
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    emitted = []
    dlg.import_completed.connect(emitted.append)
    try:
        dlg.set_source_path(str(csv_path))
        dlg.profile_selected_file()
        dlg.run_import()
        dlg.open_waveform()

        assert emitted
        record = emitted[0]
        assert record.sample_count() == 3
        assert record.analog_channels
    finally:
        dlg.close()
        app.processEvents()
