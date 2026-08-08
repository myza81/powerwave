"""Sprint 1E — visible ambiguous-timestamp interpretation.

Powerwave's DD/MM/YYYY ambiguous-date default (app.data.timestamp_disambiguation)
is unchanged by this sprint; only its visibility to the user changes:

* CsvProvider/ExcelProvider preserve the ambiguity sample on
  RecordingMetadata.timestamp_ambiguity_sample (tested directly in
  test_csv_provider.py / test_excel_provider.py's TestAmbiguousDateMetadata).
* A source loaded directly (bypassing the Import Wizard) shows a one-time
  QMessageBox.information naming the source, right after load.
* The Import Wizard's TimestampSelectPage shows a persistent banner while
  the automatic default is in effect, which disappears the moment a manual
  format override makes it inapplicable.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.file_profiler import FileProfileResult
from app.import_wizard.models import ColumnMappingCandidate, RawPreviewModel, TimestampCandidate
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.import_wizard.import_wizard_dialog import ImportWizardDialog
from app.ui.main_window.main_window import PowerwaveMainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _record(source_file: str, ambiguity_sample: str | None, n: int = 10) -> DisturbanceRecord:
    t = np.linspace(0, 1, n)
    df = pd.DataFrame({"time": t, "A": np.arange(n, dtype=float)})
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file=source_file, provider_type="csv", nominal_frequency=50.0,
            timestamp_ambiguity_sample=ambiguity_sample,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name="A", unit="MW", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2026, 6, 3), trigger_time=datetime(2026, 6, 3)),
    )


def _build_window(qapp: QApplication) -> PowerwaveMainWindow:
    win = PowerwaveMainWindow()
    win._active_session = EventAnalysisSession()
    win._session_canvas_action.setEnabled(True)
    win._ensure_session_panel().refresh_all(win._active_session)
    return win


# ─────────────────────────────────────────────────────────────────────────────
# Direct-load notice (main_window._on_session_import_record_ready)
# ─────────────────────────────────────────────────────────────────────────────


class TestDirectLoadAmbiguityNotice:
    def test_ambiguous_source_shows_notice(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                assert mock_info.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_notice_names_the_source(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                args, _kwargs = mock_info.call_args
                message = args[2]
                assert "Relay_A" in message
        finally:
            win.close()
            qapp.processEvents()

    def test_notice_states_ddmmyyyy(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                args, _kwargs = mock_info.call_args
                message = args[2]
                assert "DD/MM/YYYY" in message
        finally:
            win.close()
            qapp.processEvents()

    def test_notice_shows_real_sample_not_hardcoded(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "12/11/2026")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                args, _kwargs = mock_info.call_args
                message = args[2]
                assert "12/11/2026" in message
                assert "3/6/2026" not in message
        finally:
            win.close()
            qapp.processEvents()

    def test_notice_does_not_call_it_an_error(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                args, kwargs = mock_info.call_args
                title, message = args[1], args[2]
                assert "error" not in title.lower()
                assert "error" not in message.lower()
                assert "wrong" not in message.lower()
        finally:
            win.close()
            qapp.processEvents()

    def test_non_ambiguous_source_shows_no_notice(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", None)
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                assert mock_info.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_notice_shown_exactly_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                qapp.processEvents()
                assert mock_info.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_two_ambiguous_sources_each_get_one_notice(self, qapp: QApplication) -> None:
        """Step 9: one notification per newly-loaded ambiguous source, not
        consolidated into zero, and not repeated on later refresh."""
        win = _build_window(qapp)
        try:
            record_a = _record("Relay_A.csv", "3/6/2026 17:25")
            record_b = _record("Relay_B.csv", "12/11/2026")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record_a)
                win._on_session_import_record_ready(record_b)
                assert mock_info.call_count == 2
        finally:
            win.close()
            qapp.processEvents()

    def test_refreshing_session_panel_does_not_reshow_notice(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            record = _record("Relay_A.csv", "3/6/2026 17:25")
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                assert mock_info.call_count == 1
                # Ordinary refresh -- must not re-trigger the notice.
                win._ensure_session_panel().refresh_all(win._active_session)
                win._refresh_timing_assessment()
                assert mock_info.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_comtrade_like_record_never_shows_notice(self, qapp: QApplication) -> None:
        """COMTRADE records never set timestamp_ambiguity_sample -- the
        field defaults to None, so this handler is a no-op for them."""
        win = _build_window(qapp)
        try:
            record = DisturbanceRecord(
                metadata=RecordingMetadata(
                    station_name="S", recorder_name="R", source_file="event.cfg",
                    provider_type="comtrade", nominal_frequency=50.0,
                ),
                waveform_data=pd.DataFrame({"time": np.linspace(0, 1, 5), "A": np.arange(5, dtype=float)}),
                analog_channels=[AnalogChannel(name="A", unit="MW", index=0)],
                digital_channels=[],
                sampling_info=SamplingInformation(sampling_rates=[5.0], samples_per_rate=[5]),
                timing_info=TimingInformation(start_time=datetime(2026, 6, 3), trigger_time=datetime(2026, 6, 3)),
            )
            with patch(
                "app.ui.main_window.main_window.QMessageBox.information",
            ) as mock_info:
                win._on_session_import_record_ready(record)
                assert mock_info.call_count == 0
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Import Wizard banner (TimestampSelectPage)
# ─────────────────────────────────────────────────────────────────────────────


class ImmediateThreadPool:
    def start(self, worker) -> None:
        worker.run()


@pytest.fixture()
def local_tmp():
    path = Path("test_artifacts") / f"ambiguous_timestamp_notice_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _wizard_profile(example_values: list[str], detected_format: str | None) -> FileProfileResult:
    preview = RawPreviewModel(
        column_names=["Time", "MW"],
        preview_rows=[[v, "10.0"] for v in example_values],
        row_count_estimate=len(example_values),
    )
    return FileProfileResult(
        raw_preview=preview,
        provider_type="csv",
        delimiter=",",
        timestamp_candidates=[
            TimestampCandidate(
                "Time",
                0,
                0.95,
                detected_format=detected_format,
                example_values=example_values,
                user_selected=True,
            )
        ],
        column_mappings=[
            ColumnMappingCandidate("Time", 0, "Time", ParameterType.TIMESTAMP),
            ColumnMappingCandidate("MW", 1, "MW", ParameterType.MW, unit="MW"),
        ],
    )


def _wizard_dialog(
    monkeypatch, local_tmp, example_values: list[str], detected_format: str | None = "%Y-%m-%d %H:%M:%S"
) -> ImportWizardDialog:
    import app.ui.import_wizard.import_wizard_dialog as dialog_module

    path = local_tmp / "wizard_ambiguity.csv"
    lines = ["Time,MW"] + [f"{v},10.0" for v in example_values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        dialog_module, "profile_import_file",
        lambda *a, **k: _wizard_profile(example_values, detected_format),
    )
    dlg = ImportWizardDialog(thread_pool=ImmediateThreadPool())
    dlg.set_source_path(str(path))
    dlg.profile_selected_file()
    return dlg


class TestImportWizardAmbiguityBanner:
    def test_ambiguous_date_shows_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["3/6/2026 17:25", "3/6/2026 17:26"])
        try:
            assert not dlg.timestamp_page.ambiguity_banner.isHidden()
            text = dlg.timestamp_page.ambiguity_banner_label.text()
            assert "Ambiguous date format detected" in text
            assert "DD/MM/YYYY" in text
            assert "3/6/2026 17:25 → 3 June 2026" in text
        finally:
            dlg.close()
            qapp.processEvents()

    def test_second_ambiguous_example_also_shown(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["12/11/2026 00:00:00", "12/11/2026 00:00:01"])
        try:
            text = dlg.timestamp_page.ambiguity_banner_label.text()
            assert "12/11/2026" in text
        finally:
            dlg.close()
            qapp.processEvents()

    def test_day_over_twelve_no_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["13/6/2026 17:25", "13/6/2026 17:26"])
        try:
            assert dlg.timestamp_page.ambiguity_banner.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_month_over_twelve_no_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["6/13/2026 17:25", "6/13/2026 17:26"])
        try:
            assert dlg.timestamp_page.ambiguity_banner.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_iso_date_no_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["2026-06-03 17:25:00", "2026-06-03 17:25:01"])
        try:
            assert dlg.timestamp_page.ambiguity_banner.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_manual_override_hides_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["3/6/2026 17:25", "3/6/2026 17:26"])
        try:
            assert not dlg.timestamp_page.ambiguity_banner.isHidden()
            dlg.timestamp_page.override_edit.setText("%m/%d/%Y %H:%M:%S")
            assert dlg.timestamp_page.ambiguity_banner.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_clearing_override_restores_banner(self, monkeypatch, qapp, local_tmp) -> None:
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["3/6/2026 17:25", "3/6/2026 17:26"])
        try:
            dlg.timestamp_page.override_edit.setText("%m/%d/%Y %H:%M:%S")
            assert dlg.timestamp_page.ambiguity_banner.isHidden()
            dlg.timestamp_page.override_edit.clear()
            assert not dlg.timestamp_page.ambiguity_banner.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_advanced_timestamp_repair_still_reachable(self, monkeypatch, qapp, local_tmp) -> None:
        """The banner must sit alongside the existing override route, not
        replace or hide it."""
        dlg = _wizard_dialog(monkeypatch, local_tmp, ["3/6/2026 17:25", "3/6/2026 17:26"])
        try:
            assert not dlg.timestamp_page.ambiguity_banner.isHidden()
            assert not dlg.timestamp_page._recon_group.isHidden()
            assert not dlg.timestamp_page.override_edit.isHidden()
        finally:
            dlg.close()
            qapp.processEvents()

    def test_successful_plan_build_still_works_with_banner_visible(self, monkeypatch, qapp, local_tmp) -> None:
        from app.import_wizard.pipeline_plan_builder import build_execution_plan

        dlg = _wizard_dialog(monkeypatch, local_tmp, ["3/6/2026 17:25", "3/6/2026 17:26"])
        try:
            assert not dlg.timestamp_page.ambiguity_banner.isHidden()
            result = build_execution_plan(dlg.session, list(dlg.column_model.mappings))
            assert result.is_executable
        finally:
            dlg.close()
            qapp.processEvents()
