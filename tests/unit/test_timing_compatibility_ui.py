"""UI tests for Sprint 1B's session timing-reference compatibility warning:
the SessionPanel banner, TimingDetailsDialog, and the main-window refresh
wiring that keeps the banner in sync with session state.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication, QPushButton

from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.main_window.main_window import PowerwaveMainWindow
from app.ui.session.session_panel import SessionPanel
from app.ui.session.timing_details_dialog import TimingDetailsDialog

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _timing(
    *,
    start: datetime = datetime(2026, 1, 1, 10, 0, 0),
    timezone: str | None = None,
    timing_reference: str = "absolute",
) -> TimingInformation:
    return TimingInformation(
        start_time=start, trigger_time=start, timezone=timezone,
        timing_reference=timing_reference,
    )


def _record(timing: TimingInformation, n: int = 10) -> DisturbanceRecord:
    t = np.linspace(0, 1, n)
    df = pd.DataFrame({"time": t, "A": np.arange(n, dtype=float)})
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name="A", unit="MW", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=timing,
    )


def _add(sess: EventAnalysisSession, timing: TimingInformation, name: str) -> str:
    return sess.add_source(_record(timing), name, "csv")


# ─────────────────────────────────────────────────────────────────────────────
# SessionPanel banner
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionPanelBanner:
    def test_hidden_for_one_source(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        assert panel._timing_banner.isHidden()

    def test_hidden_for_zero_sources(self, qapp) -> None:
        sess = EventAnalysisSession()
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        assert panel._timing_banner.isHidden()

    def test_hidden_for_two_compatible_sources(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(timezone="UTC"), "A")
        _add(sess, _timing(timezone="UTC"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        assert panel._timing_banner.isHidden()

    def test_shown_for_mixed_timing_references(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        assert not panel._timing_banner.isHidden()
        assert panel._timing_banner_label.text() != ""

    def test_shown_for_unknown_timing_reference(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="totally_novel_strategy"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        assert not panel._timing_banner.isHidden()

    def test_message_persists_across_repeated_refreshes(self, qapp) -> None:
        """Unlike a transient status-bar message, the banner stays visible
        for as long as the underlying condition holds, across any number
        of refresh calls (e.g. repeated unrelated session refreshes)."""
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        panel = SessionPanel()
        for _ in range(5):
            panel.refresh_timing_assessment(sess)
            assert not panel._timing_banner.isHidden()

    def test_no_automatic_repair_action_exists(self, qapp) -> None:
        """The banner offers Details only -- no Fix/Repair/Align action."""
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)

        buttons_in_banner = panel._timing_banner.findChildren(QPushButton)
        labels = {b.text().lower() for b in buttons_in_banner}
        assert labels == {"details"}
        for forbidden in ("fix", "repair", "align", "correct", "apply"):
            assert forbidden not in labels


# ─────────────────────────────────────────────────────────────────────────────
# Details dialog
# ─────────────────────────────────────────────────────────────────────────────


class TestDetailsDialog:
    def test_details_action_opens_with_correct_row_count(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)

        dlg = TimingDetailsDialog(panel._timing_assessment)
        assert dlg._source_table.rowCount() == 2

    def test_source_details_readable(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _timing(), "Relay A")
        sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "SCADA")
        sess.set_time_offset(sid_b, -0.42, method="manual")

        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        dlg = TimingDetailsDialog(panel._timing_assessment)

        names = {dlg._source_table.item(r, 0).text() for r in range(dlg._source_table.rowCount())}
        assert names == {"Relay A", "SCADA"}
        offsets = {dlg._source_table.item(r, 2).text() for r in range(dlg._source_table.rowCount())}
        assert "-0.420 s" in offsets
        assert "+0.000 s" in offsets

    def test_details_reflect_current_offset_after_change(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _timing(), "A")
        sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")

        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        dlg1 = TimingDetailsDialog(panel._timing_assessment)
        offsets1 = [dlg1._source_table.item(r, 2).text() for r in range(dlg1._source_table.rowCount())]
        assert all(o == "+0.000 s" for o in offsets1)

        sess.set_time_offset(sid_b, 2.5, method="manual")
        panel.refresh_timing_assessment(sess)
        dlg2 = TimingDetailsDialog(panel._timing_assessment)
        offsets2 = {dlg2._source_table.item(r, 2).text() for r in range(dlg2._source_table.rowCount())}
        assert "+2.500 s" in offsets2

    def test_manual_alignment_note_shown_when_applicable(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _timing(), "A")
        sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        sess.set_time_offset(sid_b, 1.0, method="manual")

        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        dlg = TimingDetailsDialog(panel._timing_assessment)
        assert panel._timing_assessment.manual_alignment_present

    def test_no_repair_button_in_details_dialog(self, qapp) -> None:
        sess = EventAnalysisSession()
        _add(sess, _timing(), "A")
        _add(sess, _timing(timing_reference="relative_elapsed"), "B")
        panel = SessionPanel()
        panel.refresh_timing_assessment(sess)
        dlg = TimingDetailsDialog(panel._timing_assessment)

        labels = {b.text().lower() for b in dlg.findChildren(QPushButton)}
        for forbidden in ("fix", "repair", "align", "apply"):
            assert forbidden not in labels


# ─────────────────────────────────────────────────────────────────────────────
# main_window refresh wiring
# ─────────────────────────────────────────────────────────────────────────────


def _build_window(qapp: QApplication) -> PowerwaveMainWindow:
    win = PowerwaveMainWindow()
    win._active_session = EventAnalysisSession()
    win._session_canvas_action.setEnabled(True)
    win._activate_session_canvas()
    qapp.processEvents()
    return win


class TestMainWindowRefreshWiring:
    def test_add_source_refreshes_banner(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sess.default_layout()
            win._activate_session_canvas()  # first source: banner should stay hidden (1 source)
            assert win._session_panel._timing_banner.isHidden()

            record_b = _record(_timing(timing_reference="relative_elapsed"))
            win._on_session_import_record_ready(record_b)
            qapp.processEvents()
            assert not win._session_panel._timing_banner.isHidden()
        finally:
            win.close()
            qapp.processEvents()

    def test_remove_source_refreshes_banner(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
            sess.default_layout()
            win._activate_session_canvas()
            assert not win._session_panel._timing_banner.isHidden()

            # Sprint 1D: Remove Source now confirms first -- simulate the
            # user confirming so this test keeps exercising the removal.
            with patch(
                "app.ui.main_window.main_window.confirm_destructive_action",
                return_value=True,
            ):
                win._on_session_remove_source(sid_b)
            qapp.processEvents()
            assert win._session_panel._timing_banner.isHidden()
        finally:
            win.close()
            qapp.processEvents()

    def test_activation_change_refreshes_banner(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
            sess.default_layout()
            win._activate_session_canvas()
            assert not win._session_panel._timing_banner.isHidden()

            win._on_session_source_active(sid_b, False)
            qapp.processEvents()
            assert win._session_panel._timing_banner.isHidden()

            win._on_session_source_active(sid_b, True)
            qapp.processEvents()
            assert not win._session_panel._timing_banner.isHidden()
        finally:
            win.close()
            qapp.processEvents()

    def test_offset_change_refreshes_details_content(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
            sess.default_layout()
            win._activate_session_canvas()

            win._on_session_offset_changed(sid_b, 4.0)
            qapp.processEvents()
            assessment = win._session_panel._timing_assessment
            by_id = {p.source_id: p for p in assessment.source_profiles}
            assert by_id[sid_b].time_offset_s == pytest.approx(4.0)
        finally:
            win.close()
            qapp.processEvents()

    def test_clear_session_hides_banner(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
            sess.default_layout()
            win._activate_session_canvas()
            assert not win._session_panel._timing_banner.isHidden()

            win._on_session_cleared()
            qapp.processEvents()
            assert win._session_panel._timing_banner.isHidden()
        finally:
            win.close()
            qapp.processEvents()

    def test_no_mutation_from_full_ui_refresh_cycle(self, qapp: QApplication) -> None:
        win = _build_window(qapp)
        try:
            sess = win._active_session
            sid_a = _add(sess, _timing(), "A")
            sid_b = _add(sess, _timing(timing_reference="relative_elapsed"), "B")
            sess.default_layout()
            win._activate_session_canvas()

            offsets_before = {s.source_id: s.time_offset_s for s in sess.list_sources()}
            win._refresh_timing_assessment()
            win._refresh_timing_assessment()
            offsets_after = {s.source_id: s.time_offset_s for s in sess.list_sources()}
            assert offsets_before == offsets_after
        finally:
            win.close()
            qapp.processEvents()
