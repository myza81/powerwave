"""Regression tests for the embedded (non-modal) Import Wizard integration.

The Import Wizard moved from a modal ``QDialog.exec()`` flow to an embedded
``ImportWizardWidget`` hosted as the main window's central widget. Modality
used to make it impossible to trigger a second "Open" action while a wizard
was in progress; embedding removes that guarantee, so these tests cover the
lifecycle paths that are newly reachable: reopening over an active wizard,
cancelling, completing an import, and repeated open/close cycles.
"""
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.sessions import EventAnalysisSession
from app.ui.import_wizard.import_wizard_dialog import ImportWizardWidget
from app.ui.main_window.main_window import PowerwaveMainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _process_events(qapp: QApplication, count: int = 3) -> None:
    for _ in range(count):
        qapp.processEvents()


def test_show_embedded_import_wizard_sets_central_widget(qapp: QApplication) -> None:
    win = PowerwaveMainWindow()
    try:
        win._show_embedded_import_wizard("does_not_exist.csv")
        _process_events(qapp)

        assert isinstance(win._embedded_import_wizard, ImportWizardWidget)
        assert win.centralWidget() is win._embedded_import_wizard
    finally:
        win.close()
        qapp.processEvents()


def test_reopen_without_discard_risk_replaces_silently(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh wizard with no user progress should be replaced without a prompt."""
    win = PowerwaveMainWindow()
    try:
        win._show_embedded_import_wizard("first.csv")
        _process_events(qapp)
        first_wizard = win._embedded_import_wizard
        assert first_wizard is not None

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("QMessageBox.question should not be shown when there is no discard risk")

        monkeypatch.setattr(QMessageBox, "question", _fail_if_called)

        win._show_embedded_import_wizard("second.csv")
        _process_events(qapp)

        assert win._embedded_import_wizard is not None
        assert win._embedded_import_wizard is not first_wizard
        assert win.centralWidget() is win._embedded_import_wizard
    finally:
        win.close()
        qapp.processEvents()


def test_reopen_with_discard_risk_declined_keeps_original_wizard(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: before embedding, a second Open/Add-Source action was
    unreachable while the wizard was modal. Now it is reachable (File > Open,
    or the Session Panel's Add Source button, both remain clickable), so
    declining the discard prompt must leave the in-progress wizard intact
    instead of silently replacing it.
    """
    win = PowerwaveMainWindow()
    try:
        win._show_embedded_import_wizard("first.csv")
        _process_events(qapp)
        first_wizard = win._embedded_import_wizard
        assert first_wizard is not None
        first_wizard._import_running = True  # simulate in-progress work worth protecting

        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
        )

        win._show_embedded_import_wizard("second.csv")
        _process_events(qapp)

        assert win._embedded_import_wizard is first_wizard
        assert win.centralWidget() is first_wizard
    finally:
        win.close()
        qapp.processEvents()


def test_reopen_with_discard_risk_confirmed_replaces_wizard(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    win = PowerwaveMainWindow()
    try:
        win._show_embedded_import_wizard("first.csv")
        _process_events(qapp)
        first_wizard = win._embedded_import_wizard
        assert first_wizard is not None
        first_wizard._import_running = True

        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )

        win._show_embedded_import_wizard("second.csv")
        _process_events(qapp)

        assert win._embedded_import_wizard is not None
        assert win._embedded_import_wizard is not first_wizard
        assert win.centralWidget() is win._embedded_import_wizard
    finally:
        win.close()
        qapp.processEvents()


def test_cancelling_wizard_with_no_active_session_shows_placeholder(qapp: QApplication) -> None:
    win = PowerwaveMainWindow()
    try:
        win._show_embedded_import_wizard("first.csv")
        _process_events(qapp)
        wizard = win._embedded_import_wizard
        assert wizard is not None

        wizard.request_close()  # no discard risk -> closes immediately
        _process_events(qapp)

        assert win._embedded_import_wizard is None
        assert win.centralWidget() is not wizard
        assert win.centralWidget() is not None
    finally:
        win.close()
        qapp.processEvents()


def test_completing_import_clears_wizard_and_adds_source(qapp: QApplication) -> None:
    from app.data.synthetic import make_high_rate_record

    record = make_high_rate_record(duration_s=0.5, sampling_rate_hz=500.0).record

    win = PowerwaveMainWindow()
    try:
        win._active_session = EventAnalysisSession()
        win._show_embedded_import_wizard("first.csv")
        _process_events(qapp)
        wizard = win._embedded_import_wizard
        assert wizard is not None

        # Simulate a completed pipeline result the user chose to open.
        # open_waveform() only reads .success and .record, so a lightweight
        # stand-in avoids constructing every field of the real dataclass.
        wizard._pipeline_result = types.SimpleNamespace(success=True, record=record)
        wizard.open_waveform()
        _process_events(qapp)

        assert win._embedded_import_wizard is None
        assert win._active_session is not None
        assert len(win._active_session.list_sources()) == 1
        assert win._session_canvas_active is True
    finally:
        win.close()
        qapp.processEvents()


def test_repeated_open_close_cycles_do_not_error(qapp: QApplication) -> None:
    win = PowerwaveMainWindow()
    try:
        for i in range(4):
            win._show_embedded_import_wizard(f"file_{i}.csv")
            _process_events(qapp)
            assert win._embedded_import_wizard is not None
            win._embedded_import_wizard.request_close()
            _process_events(qapp)
            assert win._embedded_import_wizard is None
    finally:
        win.close()
        qapp.processEvents()
