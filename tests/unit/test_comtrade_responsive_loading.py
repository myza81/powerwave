"""Sprint 1F — COMTRADE loading responsiveness and user feedback.

Large COMTRADE files were previously loaded synchronously on the main Qt
thread via PowerwaveMainWindow._on_add_to_session(), which could freeze the
UI with no progress or cancellation path. This sprint moves that single
call (ProviderManager.load()) onto a background QRunnable
(_ComtradeLoadWorker), shows a busy QProgressDialog with Cancel, and uses a
request-id to discard any result that arrives after the request was
cancelled, superseded by a session replacement, or the window closed.

No COMTRADE parsing behavior changes -- these tests assert the worker path
produces byte-for-byte identical DisturbanceRecord data to the old
synchronous path, and focus on the new orchestration: threading boundary,
event-loop responsiveness, cancellation, stale-result protection, error
handling, and duplicate-load prevention.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from app.providers.base.exceptions import ProviderLoadError
from app.providers.comtrade.comtrade_provider import ComtradeProvider
from app.providers.base.provider_manager import ProviderManager
from app.sessions.event_session import EventAnalysisSession
from app.ui.main_window.main_window import PowerwaveMainWindow, _ComtradeLoadWorker


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Minimal, self-contained COMTRADE fixture (1991 ASCII, 1 analog + 1 digital)
# ─────────────────────────────────────────────────────────────────────────────

_CFG_OK = (
    "SUBST_A,IED_01\n"
    "2,1A,1D\n"
    "1,VA,A,,kV,1.0,0.0,0,-32768,32767\n"
    "1,CB_A,,,0\n"
    "50\n"
    "1\n"
    "1000,4\n"
    "01/01/2024,12:00:00.000000\n"
    "01/01/2024,12:00:00.001000\n"
    "ASCII\n"
)
_DAT_OK = (
    "1,0,100,0\n"
    "2,1000,200,1\n"
    "3,2000,300,1\n"
    "4,3000,400,0\n"
)


def _write_comtrade(tmp_path: Path, stem: str = "test") -> Path:
    cfg_path = tmp_path / f"{stem}.cfg"
    dat_path = tmp_path / f"{stem}.dat"
    cfg_path.write_text(_CFG_OK, encoding="latin-1")
    dat_path.write_text(_DAT_OK, encoding="latin-1")
    return cfg_path


def _write_malformed_comtrade(tmp_path: Path, stem: str = "bad") -> Path:
    """CFG references a DAT file that doesn't exist -- ProviderLoadError."""
    cfg_path = tmp_path / f"{stem}.cfg"
    cfg_path.write_text(_CFG_OK, encoding="latin-1")
    return cfg_path


class ImmediateThreadPool:
    """Runs a QRunnable synchronously -- for deterministic success/error/cancel
    tests where thread-boundary itself isn't what's under test."""

    def start(self, worker) -> None:
        worker.run()


def _build_window(qapp: QApplication, thread_pool=None) -> PowerwaveMainWindow:
    win = PowerwaveMainWindow(comtrade_thread_pool=thread_pool or ImmediateThreadPool())
    win._active_session = EventAnalysisSession()
    win._session_canvas_action.setEnabled(True)
    win._ensure_session_panel().refresh_all(win._active_session)
    return win


def _pump_until(qapp: QApplication, predicate, timeout_s: float = 5.0) -> bool:
    """Process Qt events until predicate() is True or timeout_s elapses.
    Returns whether predicate ultimately became True."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture: worker thread vs main thread boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestThreadBoundary:
    def test_provider_load_runs_off_main_thread(self, qapp: QApplication, tmp_path: Path) -> None:
        main_thread_id = threading.get_ident()
        observed_thread_id = {}

        real_load = ProviderManager.load

        def _spy_load(self, path):
            observed_thread_id["id"] = threading.get_ident()
            return real_load(self, path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _spy_load):
                win._start_comtrade_load(cfg_path)
                assert _pump_until(qapp, lambda: "id" in observed_thread_id)
            assert observed_thread_id["id"] != main_thread_id
        finally:
            win.close()
            qapp.processEvents()

    def test_session_mutation_happens_on_main_thread(self, qapp: QApplication, tmp_path: Path) -> None:
        main_thread_id = threading.get_ident()
        insertion_thread_id = {}

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            original_add_source = EventAnalysisSession.add_source

            def _spy_add_source(self, *a, **k):
                insertion_thread_id["id"] = threading.get_ident()
                return original_add_source(self, *a, **k)

            with patch.object(EventAnalysisSession, "add_source", _spy_add_source):
                win._start_comtrade_load(cfg_path)
                assert _pump_until(qapp, lambda: "id" in insertion_thread_id)
            assert insertion_thread_id["id"] == main_thread_id
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Responsiveness
# ─────────────────────────────────────────────────────────────────────────────


class TestResponsiveness:
    def test_event_loop_stays_responsive_during_slow_load(self, qapp: QApplication, tmp_path: Path) -> None:
        """A slow provider.load() must not block the Qt event loop -- prove
        it by incrementing a counter via processEvents() while the (real,
        threaded) worker is still running."""
        def _slow_load(self, path):
            time.sleep(0.4)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                ticks = 0
                deadline = time.monotonic() + 2.0
                while win._active_comtrade_request_id is not None and time.monotonic() < deadline:
                    qapp.processEvents()
                    ticks += 1
                    time.sleep(0.005)
                assert ticks > 5, "event loop should have processed many events while worker ran"
        finally:
            win.close()
            qapp.processEvents()

    def test_progress_dialog_visible_during_load(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                assert win._comtrade_progress_dialog is not None
                assert not win._comtrade_progress_dialog.isHidden()
                assert win._comtrade_progress_dialog.minimum() == 0
                assert win._comtrade_progress_dialog.maximum() == 0  # indeterminate/busy
                _pump_until(qapp, lambda: win._active_comtrade_request_id is None)
        finally:
            win.close()
            qapp.processEvents()

    def test_no_fabricated_percentage(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", lambda self, path: (time.sleep(0.2), ComtradeProvider().load(path))[1]):
                win._start_comtrade_load(cfg_path)
                assert win._comtrade_progress_dialog.minimum() == win._comtrade_progress_dialog.maximum() == 0
                _pump_until(qapp, lambda: win._active_comtrade_request_id is None)
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Successful background load
# ─────────────────────────────────────────────────────────────────────────────


class TestSuccessfulLoad:
    def test_source_inserted_exactly_once(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_comtrade(tmp_path)
            win._start_comtrade_load(cfg_path)
            assert len(win._active_session.list_sources()) == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_record_numerically_identical_to_synchronous_load(self, qapp: QApplication, tmp_path: Path) -> None:
        cfg_path = _write_comtrade(tmp_path)
        sync_record = ComtradeProvider().load(cfg_path)

        win = _build_window(qapp)
        try:
            win._start_comtrade_load(cfg_path)
            async_source = win._active_session.list_sources()[0]
            async_record = async_source.record

            assert async_record.timing_info.start_time == sync_record.timing_info.start_time
            assert async_record.timing_info.trigger_time == sync_record.timing_info.trigger_time
            assert async_record.analog_channel_names() == sync_record.analog_channel_names()
            assert async_record.digital_channel_names() == sync_record.digital_channel_names()
            import numpy as np
            np.testing.assert_array_equal(
                async_record.waveform_data["time"].to_numpy(),
                sync_record.waveform_data["time"].to_numpy(),
            )
            for ch in sync_record.analog_channel_names():
                np.testing.assert_array_equal(
                    async_record.waveform_data[ch].to_numpy(),
                    sync_record.waveform_data[ch].to_numpy(),
                )
        finally:
            win.close()
            qapp.processEvents()

    def test_session_panel_and_canvas_refreshed(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_comtrade(tmp_path)
            win._start_comtrade_load(cfg_path)
            sid = win._active_session.list_sources()[0].source_id
            assert sid in win._session_panel._source_rows
            assert win._session_canvas_active
        finally:
            win.close()
            qapp.processEvents()

    def test_progress_dialog_closed_after_success(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_comtrade(tmp_path)
            win._start_comtrade_load(cfg_path)
            assert win._comtrade_progress_dialog is None
            assert win._active_comtrade_request_id is None
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation
# ─────────────────────────────────────────────────────────────────────────────


class TestCancellation:
    def test_cancel_discards_late_result_no_source_inserted(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                dialog = win._comtrade_progress_dialog
                assert dialog is not None
                # Simulate the user clicking Cancel: emit the same signal the
                # real Cancel button emits (QProgressDialog.cancel() called
                # programmatically does not itself re-emit canceled() in this
                # Qt build -- only an actual button click / close() does).
                dialog.canceled.emit()
                assert win._active_comtrade_request_id is None
                assert win._comtrade_progress_dialog is None
                # Let the abandoned worker actually finish in the background.
                time.sleep(0.5)
                qapp.processEvents()
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_shows_no_late_success_or_error_message(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load), patch(
                "app.ui.main_window.main_window.QMessageBox.critical"
            ) as mock_critical:
                win._start_comtrade_load(cfg_path)
                win._comtrade_progress_dialog.canceled.emit()
                time.sleep(0.5)
                qapp.processEvents()
                assert mock_critical.call_count == 0
            assert "cancelled" in win.statusBar().currentMessage().lower()
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_does_not_crash(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.2)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                win._comtrade_progress_dialog.canceled.emit()
                time.sleep(0.4)
                qapp.processEvents()  # must not raise
        finally:
            win.close()
            qapp.processEvents()

    def test_dialog_close_button_behaves_like_cancel(self, qapp: QApplication, tmp_path: Path) -> None:
        """QProgressDialog's default behavior maps the window-close (X)
        button to cancel() -- verify our handler treats it identically."""
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                dialog = win._comtrade_progress_dialog
                dialog.close()  # QProgressDialog maps close -> cancel()
                assert win._active_comtrade_request_id is None
                time.sleep(0.5)
                qapp.processEvents()
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Session replacement while loading
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionReplacement:
    def test_new_session_discards_stale_result(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load), patch(
                "app.ui.main_window.main_window.confirm_destructive_action", return_value=True,
            ):
                win._start_comtrade_load(cfg_path)
                original_session = win._active_session
                win._on_new_session()  # replaces the session mid-load
                time.sleep(0.5)
                qapp.processEvents()
            assert win._active_session is not original_session
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()

    def test_cleared_session_discards_stale_result(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                win._on_session_cleared()  # simulates Clear Session confirmed
                time.sleep(0.5)
                qapp.processEvents()
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_missing_dat_shows_clean_gui_error(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_malformed_comtrade(tmp_path)
            with patch("app.ui.main_window.main_window.QMessageBox.critical") as mock_critical:
                win._start_comtrade_load(cfg_path)
                assert mock_critical.call_count == 1
                _args, kwargs = mock_critical.call_args, {}
                args = mock_critical.call_args.args
                message = args[2]
                assert "Traceback" not in message
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()

    def test_error_leaves_no_partial_source(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_malformed_comtrade(tmp_path)
            with patch("app.ui.main_window.main_window.QMessageBox.critical"):
                win._start_comtrade_load(cfg_path)
            assert win._active_session.list_sources() == []
            assert win._session_panel._source_rows == {}
        finally:
            win.close()
            qapp.processEvents()

    def test_error_closes_progress_dialog(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_malformed_comtrade(tmp_path)
            with patch("app.ui.main_window.main_window.QMessageBox.critical"):
                win._start_comtrade_load(cfg_path)
            assert win._comtrade_progress_dialog is None
            assert win._active_comtrade_request_id is None
        finally:
            win.close()
            qapp.processEvents()

    def test_cancelled_request_does_not_show_late_error(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_fail(self, path):
            time.sleep(0.3)
            raise ProviderLoadError("simulated parser failure")

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_fail), patch(
                "app.ui.main_window.main_window.QMessageBox.critical"
            ) as mock_critical:
                win._start_comtrade_load(cfg_path)
                win._comtrade_progress_dialog.canceled.emit()
                time.sleep(0.5)
                qapp.processEvents()
                assert mock_critical.call_count == 0
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Window close during load
# ─────────────────────────────────────────────────────────────────────────────


class TestCloseDuringLoad:
    def test_close_during_load_does_not_crash(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        cfg_path = _write_comtrade(tmp_path)
        with patch.object(ProviderManager, "load", _slow_load):
            win._start_comtrade_load(cfg_path)
            win.close()  # must not raise
            qapp.processEvents()
            time.sleep(0.5)
            qapp.processEvents()  # worker's late signal must be a safe no-op

    def test_close_invalidates_active_request(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        cfg_path = _write_comtrade(tmp_path)

        def _slow_load(self, path):
            time.sleep(0.2)
            return ComtradeProvider().load(path)

        with patch.object(ProviderManager, "load", _slow_load):
            win._start_comtrade_load(cfg_path)
            assert win._active_comtrade_request_id is not None
            win.close()
            assert win._active_comtrade_request_id is None
            time.sleep(0.4)
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate request prevention
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateLoadPrevention:
    def test_second_load_rejected_while_first_active(self, qapp: QApplication, tmp_path: Path) -> None:
        def _slow_load(self, path):
            time.sleep(0.3)
            return ComtradeProvider().load(path)

        win = _build_window(qapp, thread_pool=QThreadPool.globalInstance())
        try:
            cfg_path = _write_comtrade(tmp_path)
            with patch.object(ProviderManager, "load", _slow_load):
                win._start_comtrade_load(cfg_path)
                first_request_id = win._active_comtrade_request_id
                win._start_comtrade_load(cfg_path)  # second attempt while first active
                assert win._active_comtrade_request_id == first_request_id
                assert "already loading" in win.statusBar().currentMessage().lower()
                _pump_until(qapp, lambda: win._active_comtrade_request_id is None)
            assert len(win._active_session.list_sources()) == 1  # only the first load landed
        finally:
            win.close()
            qapp.processEvents()

    def test_sequential_loads_both_succeed(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_a = _write_comtrade(tmp_path, "a")
            cfg_b = _write_comtrade(tmp_path, "b")
            win._start_comtrade_load(cfg_a)
            win._start_comtrade_load(cfg_b)
            assert len(win._active_session.list_sources()) == 2
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Regression: small COMTRADE, malformed COMTRADE, CSV/Excel Wizard untouched
# ─────────────────────────────────────────────────────────────────────────────


class TestRegression:
    def test_small_comtrade_still_loads_correctly(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_comtrade(tmp_path)
            win._start_comtrade_load(cfg_path)
            source = win._active_session.list_sources()[0]
            assert source.record.analog_channel_names() == ["VA"]
            assert source.record.digital_channel_names() == ["CB_A"]
        finally:
            win.close()
            qapp.processEvents()

    def test_malformed_comtrade_still_errors_cleanly(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            cfg_path = _write_malformed_comtrade(tmp_path)
            with patch("app.ui.main_window.main_window.QMessageBox.critical") as mock_critical:
                win._start_comtrade_load(cfg_path)
                assert mock_critical.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_csv_excel_still_routes_to_import_wizard_not_worker(self, qapp: QApplication, tmp_path: Path) -> None:
        win = _build_window(qapp)
        try:
            csv_path = tmp_path / "data.csv"
            csv_path.write_text("time,VA\n0,1.0\n1,2.0\n", encoding="utf-8")
            with patch.object(
                PowerwaveMainWindow, "_show_embedded_import_wizard"
            ) as mock_wizard, patch.object(
                PowerwaveMainWindow, "_start_comtrade_load"
            ) as mock_comtrade:
                with patch(
                    "app.ui.main_window.main_window.QFileDialog.getOpenFileName",
                    return_value=(str(csv_path), ""),
                ):
                    win._on_add_to_session()
                assert mock_wizard.call_count == 1
                assert mock_comtrade.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_worker_class_wraps_provider_manager_load_only(self) -> None:
        """_ComtradeLoadWorker stays intentionally narrow -- no session or
        UI logic lives inside run()."""
        import inspect

        source = inspect.getsource(_ComtradeLoadWorker.run)
        assert "EventAnalysisSession" not in source
        assert "SessionPanel" not in source
