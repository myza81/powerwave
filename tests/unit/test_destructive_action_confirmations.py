"""Sprint 1D — destructive-action confirmation tests: Remove Source, Clear
Session, New/Open Session, and Delete Calculated Signal.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication). Confirmation itself is exercised in two
ways: the real confirm_destructive_action() helper is tested directly
against a mocked QMessageBox.question() (dialog defaults), while the
main-window/session-panel wiring tests mock confirm_destructive_action at
its imported call site to simulate Cancel/Confirm without a real modal.
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculationStatus,
    ChannelRef,
)
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.dialogs.confirm import confirm_destructive_action
from app.ui.main_window.main_window import PowerwaveMainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _record(analog: dict[str, str], n: int = 20) -> DisturbanceRecord:
    time = np.linspace(0, 1, n)
    data: dict[str, object] = {"time": time}
    for name in analog:
        data[name] = np.sin(time)
    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[20.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _build_window(qapp: QApplication) -> PowerwaveMainWindow:
    win = PowerwaveMainWindow()
    win._active_session = EventAnalysisSession()
    win._session_canvas_action.setEnabled(True)
    win._ensure_session_panel().refresh_all(win._active_session)
    return win


def _add_source(win: PowerwaveMainWindow, analog: dict[str, str], name: str) -> str:
    sess = win._active_session
    sid = sess.add_source(_record(analog), name, "csv")
    sess.default_layout()
    win._ensure_session_panel().refresh_all(sess)
    return sid


def _add_calc(win: PowerwaveMainWindow, calc_id: str, name: str, expr: str, bindings: dict, ref: str) -> None:
    sess = win._active_session
    defn = CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression=expr,
        variable_bindings=bindings, reference_variable=ref,
    )
    sess.add_calculated_signal(defn)
    CalculatedSignalResolutionService(sess).resolve_one(calc_id)
    win._ensure_session_panel().refresh_calculated_signals(sess)


# ─────────────────────────────────────────────────────────────────────────────
# The confirmation helper itself
# ─────────────────────────────────────────────────────────────────────────────


class TestConfirmDestructiveActionHelper:
    def test_yes_returns_true(self, qapp: QApplication, monkeypatch) -> None:
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        assert confirm_destructive_action(None, title="t", message="m") is True

    def test_no_returns_false(self, qapp: QApplication, monkeypatch) -> None:
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        assert confirm_destructive_action(None, title="t", message="m") is False

    def test_default_button_is_no(self, qapp: QApplication, monkeypatch) -> None:
        captured = {}

        def _fake_question(parent, title, message, buttons, default):
            captured["buttons"] = buttons
            captured["default"] = default
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", _fake_question)
        confirm_destructive_action(None, title="t", message="m")
        assert captured["default"] == QMessageBox.StandardButton.No
        assert captured["buttons"] == (
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

    def test_title_and_message_are_forwarded(self, qapp: QApplication, monkeypatch) -> None:
        captured = {}

        def _fake_question(parent, title, message, buttons, default):
            captured["title"] = title
            captured["message"] = message
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", _fake_question)
        confirm_destructive_action(None, title="Remove source?", message="detail text")
        assert captured["title"] == "Remove source?"
        assert captured["message"] == "detail text"


# ─────────────────────────────────────────────────────────────────────────────
# Remove Source
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoveSourceConfirmation:
    def test_confirmation_shown_no_dependents(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                win._on_session_remove_source(sid)
                assert mock_confirm.call_count == 1
                _, kwargs = mock_confirm.call_args
                assert kwargs["title"] == "Remove source?"
                assert "Relay A" in kwargs["message"]
                assert "stale" not in kwargs["message"]
        finally:
            win.close()
            qapp.processEvents()

    def test_confirmation_shown_with_dependents_mentions_stale(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Relay A")
            sid_b = _add_source(win, {"B": "MW"}, "Relay B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                win._on_session_remove_source(sid_b)
                _, kwargs = mock_confirm.call_args
                assert "Relay B" in kwargs["message"]
                assert "stale" in kwargs["message"].lower()
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_leaves_source_and_calc_untouched(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Relay A")
            sid_b = _add_source(win, {"B": "MW"}, "Relay B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
            result_before = win._active_session.get_calculated_signal_result("c1")

            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ):
                win._on_session_remove_source(sid_b)

            assert win._active_session.get_source(sid_b) is not None
            assert win._active_session.list_sources().__len__() == 2
            result_after = win._active_session.get_calculated_signal_result("c1")
            assert result_after.status == CalculationStatus.OK
            np.testing.assert_array_equal(result_after.values, result_before.values)
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_removes_source_exactly_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ), patch.object(
                EventAnalysisSession, "remove_source", autospec=True,
                side_effect=EventAnalysisSession.remove_source,
            ) as spy:
                win._on_session_remove_source(sid)
                assert spy.call_count == 1
            assert win._active_session.get_source(sid) is None
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_marks_dependent_calc_missing_source(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Relay A")
            sid_b = _add_source(win, {"B": "MW"}, "Relay B")
            _add_calc(win, "c1", "AminusB", "a - b",
                      {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")

            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ):
                win._on_session_remove_source(sid_b)

            result = win._active_session.get_calculated_signal_result("c1")
            assert result is not None  # retained, not deleted
            assert result.status == CalculationStatus.STALE
            status = win._active_session.get_dependency_status("c1")
            assert sid_b in status.missing_sources
            assert win._active_session.get_calculated_signal_definition("c1") is not None
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Clear Session
# ─────────────────────────────────────────────────────────────────────────────


class TestClearSessionConfirmation:
    def test_empty_session_clears_without_confirmation(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            panel = win._session_panel
            with patch(
                "app.ui.session.session_panel.confirm_destructive_action",
            ) as mock_confirm:
                panel._on_clear_session()
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_populated_session_shows_confirmation(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            panel = win._session_panel
            with patch(
                "app.ui.session.session_panel.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                panel._on_clear_session()
                assert mock_confirm.call_count == 1
                _, kwargs = mock_confirm.call_args
                assert kwargs["title"] == "Clear current session?"
        finally:
            win.close()
            qapp.processEvents()

    def test_populated_with_only_calc_signal_shows_confirmation(self, qapp: QApplication) -> None:
        """A session with zero sources but a lingering calculated signal
        entry is still meaningful work."""
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "DoubleA", "a + a", {"a": ChannelRef(sid_a, "A")}, "a")
            panel = win._session_panel
            with patch(
                "app.ui.session.session_panel.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                panel._on_clear_session()
                assert mock_confirm.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_leaves_rows_and_session_untouched(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid_a = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "DoubleA", "a + a", {"a": ChannelRef(sid_a, "A")}, "a")
            panel = win._session_panel

            with patch(
                "app.ui.session.session_panel.confirm_destructive_action",
                return_value=False,
            ):
                panel._on_clear_session()

            assert sid_a in panel._source_rows
            assert "c1" in panel._calc_rows
            assert win._active_session.get_source(sid_a) is not None
            assert win._active_session.get_calculated_signal_definition("c1") is not None
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_clears_exactly_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            panel = win._session_panel

            cleared = []
            panel.session_cleared.connect(lambda: cleared.append(1))

            with patch(
                "app.ui.session.session_panel.confirm_destructive_action",
                return_value=True,
            ):
                panel._on_clear_session()

            assert cleared == [1]
            assert panel._source_rows == {}
            assert panel._calc_rows == {}
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# New / Open Session
# ─────────────────────────────────────────────────────────────────────────────


class TestNewOpenSessionConfirmation:
    def test_no_active_session_proceeds_without_confirmation(self, qapp: QApplication) -> None:
        win = PowerwaveMainWindow()
        try:
            win._active_session = None
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
            ) as mock_confirm:
                assert win._confirm_replace_active_session() is True
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_empty_session_proceeds_without_confirmation(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
            ) as mock_confirm:
                assert win._confirm_replace_active_session() is True
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_populated_session_shows_confirmation(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                assert win._confirm_replace_active_session() is False
                assert mock_confirm.call_count == 1
                _, kwargs = mock_confirm.call_args
                assert kwargs["title"] == "Start a new session?"
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_via_open_session_leaves_session_untouched(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            original_session = win._active_session
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ):
                win._on_open_session()
            assert win._active_session is original_session
            assert win._active_session.get_source(sid) is not None
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_via_open_session_replaces_session(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            original_session = win._active_session
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ):
                win._on_open_session()
            assert win._active_session is not original_session
            assert win._active_session.list_sources() == []
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_via_unified_open_file_does_not_show_file_dialog(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            original_session = win._active_session
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ), patch(
                "app.ui.main_window.main_window.QFileDialog.getOpenFileName",
            ) as mock_dialog:
                win._open_unified_file()
                assert mock_dialog.call_count == 0
            assert win._active_session is original_session
            assert win._active_session.get_source(sid) is not None
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_via_unified_open_file_proceeds_to_file_dialog(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            original_session = win._active_session
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ), patch(
                "app.ui.main_window.main_window.QFileDialog.getOpenFileName",
                return_value=("", ""),
            ) as mock_dialog:
                win._open_unified_file()
                assert mock_dialog.call_count == 1
            assert win._active_session is not original_session
        finally:
            win.close()
            qapp.processEvents()

    def test_no_double_prompt_single_confirm_call(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ) as mock_confirm, patch(
                "app.ui.main_window.main_window.QFileDialog.getOpenFileName",
                return_value=("", ""),
            ):
                win._open_unified_file()
                assert mock_confirm.call_count == 1
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Delete Calculated Signal
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteCalculatedSignalConfirmation:
    def test_confirmation_shown_with_name(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "Net System Power", "a + a", {"a": ChannelRef(sid, "A")}, "a")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ) as mock_confirm:
                win._on_calc_signal_delete("c1")
                assert mock_confirm.call_count == 1
                _, kwargs = mock_confirm.call_args
                assert kwargs["title"] == "Delete calculated signal?"
                assert "Net System Power" in kwargs["message"]
        finally:
            win.close()
            qapp.processEvents()

    def test_cancel_leaves_definition_result_and_curve(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "Net System Power", "a + a", {"a": ChannelRef(sid, "A")}, "a")

            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=False,
            ):
                win._on_calc_signal_delete("c1")

            assert win._active_session.get_calculated_signal_definition("c1") is not None
            assert win._active_session.get_calculated_signal_result("c1") is not None
            assert "c1" in win._session_panel._calc_rows
        finally:
            win.close()
            qapp.processEvents()

    def test_confirm_removes_definition_result_and_leaves_sources(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "Net System Power", "a + a", {"a": ChannelRef(sid, "A")}, "a")

            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ):
                win._on_calc_signal_delete("c1")

            assert win._active_session.get_calculated_signal_definition("c1") is None
            assert win._active_session.get_calculated_signal_result("c1") is None
            assert win._active_session.get_source(sid) is not None  # source untouched
        finally:
            win.close()
            qapp.processEvents()

    def test_delete_executes_exactly_once(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "Net System Power", "a + a", {"a": ChannelRef(sid, "A")}, "a")

            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ), patch.object(
                EventAnalysisSession, "remove_calculated_signal", autospec=True,
                side_effect=EventAnalysisSession.remove_calculated_signal,
            ) as spy:
                win._on_calc_signal_delete("c1")
                assert spy.call_count == 1
        finally:
            win.close()
            qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation spam avoidance (Step 9)
# ─────────────────────────────────────────────────────────────────────────────


class TestNoConfirmationSpam:
    def test_offset_edit_does_not_confirm(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
            ) as mock_confirm:
                win._on_session_offset_changed(sid, 0.5)
                win._on_session_offset_edit_finished(sid)
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_recalculate_all_does_not_confirm(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            _add_calc(win, "c1", "DoubleA", "a + a", {"a": ChannelRef(sid, "A")}, "a")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
            ) as mock_confirm:
                win._on_calc_signal_recalculate_all()
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_source_activation_toggle_does_not_confirm(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sid = _add_source(win, {"A": "MW"}, "Relay A")
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
            ) as mock_confirm:
                win._on_session_source_active(sid, False)
                assert mock_confirm.call_count == 0
        finally:
            win.close()
            qapp.processEvents()
