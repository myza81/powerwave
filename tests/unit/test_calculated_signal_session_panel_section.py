"""Unit tests for the Calculated Signals section of SessionPanel (Phase 3B)
in app.ui.session.session_panel / app.ui.session.calculated_signal_row_widget.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).

Uses only generic, synthetic session fixtures -- no filename, station, or
event identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication

from app.calculated_signals.models import CalculatedSignalDefinition, ChannelRef
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.session.session_panel import SessionPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_record(analog: dict[str, str], n: int = 10) -> DisturbanceRecord:
    t = np.linspace(0, 1, n)
    data: dict[str, object] = {"time": t}
    for name in analog:
        data[name] = np.arange(n, dtype=float)
    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _session_with_source() -> tuple[EventAnalysisSession, str]:
    sess = EventAnalysisSession()
    sid = sess.add_source(_make_record({"Va": "kV"}), "Source A", "csv")
    sess.default_layout()
    return sess, sid


def _create_and_resolve(sess: EventAnalysisSession, calc_id: str, name: str, sid: str) -> None:
    defn = CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression="a + a",
        variable_bindings={"a": ChannelRef(sid, "Va")}, reference_variable="a",
    )
    sess.add_calculated_signal(defn)
    CalculatedSignalResolutionService(sess).resolve_one(calc_id)


class TestCalculatedSignalsSectionAppearance:
    def test_created_signal_appears(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert "c1" in panel._calc_rows

    def test_section_hidden_when_no_signals(self, qapp) -> None:
        sess, sid = _session_with_source()
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert panel._calc_signals_group.isHidden()

    def test_section_shown_once_a_signal_exists(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert not panel._calc_signals_group.isHidden()

    def test_marker_prefix_shown_in_row_label(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert panel._calc_rows["c1"]._name_lbl.text().startswith("ƒ ")
        assert "NetVoltage" in panel._calc_rows["c1"]._name_lbl.text()

    def test_multiple_calculated_signals_all_appear(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "First", sid)
        _create_and_resolve(sess, "c2", "Second", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert set(panel._calc_rows.keys()) == {"c1", "c2"}

    def test_source_rows_remain_unchanged(self, qapp) -> None:
        sess, sid = _session_with_source()
        panel = SessionPanel()
        panel.refresh_all(sess)
        assert sid in panel._source_rows
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel.refresh_calculated_signals(sess)
        assert sid in panel._source_rows
        assert len(panel._source_rows) == 1  # no fake source row added for c1


class TestCalculatedSignalsSectionVisibilityToggle:
    def test_toggling_checkbox_emits_signal(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)

        received = []
        panel.calculated_signal_visibility_changed.connect(
            lambda cid, v: received.append((cid, v))
        )
        row = panel._calc_rows["c1"]
        row._visible_cb.setChecked(False)
        assert received == [("c1", False)]

    def test_refresh_does_not_spuriously_emit_visibility_signal(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)

        received = []
        panel.calculated_signal_visibility_changed.connect(
            lambda cid, v: received.append((cid, v))
        )
        sess.set_calculated_signal_visible("c1", False)
        panel.refresh_calculated_signals(sess)  # programmatic refresh
        assert received == []  # refresh() must not re-emit the signal it's reacting to


class TestCalculatedSignalsSectionStaleLabel:
    def test_stale_status_label_shown(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert panel._calc_rows["c1"]._status_lbl.text() == "OK"

        sess.set_time_offset(sid, 0.5)
        panel.refresh_calculated_signals(sess)
        assert panel._calc_rows["c1"]._status_lbl.text() == "stale"

    def test_never_calculated_status_label(self, qapp) -> None:
        sess, sid = _session_with_source()
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="NeverRun", expression="a + a",
            variable_bindings={"a": ChannelRef(sid, "Va")}, reference_variable="a",
        )
        sess.add_calculated_signal(defn)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert panel._calc_rows["c1"]._status_lbl.text() == "not calculated"


class TestCalculatedSignalsSectionDelete:
    def test_delete_button_emits_signal(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)

        received = []
        panel.calculated_signal_delete_requested.connect(received.append)
        panel._calc_rows["c1"]._delete_btn.click()
        assert received == ["c1"]

    def test_row_removed_after_session_deletion_and_refresh(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert "c1" in panel._calc_rows

        sess.remove_calculated_signal("c1")
        panel.refresh_calculated_signals(sess)
        assert "c1" not in panel._calc_rows
        assert panel._calc_signals_group.isHidden()


class TestCalculatedSignalsSectionRecalculate:
    def test_recalculate_button_emits_signal(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)

        received = []
        panel.calculated_signal_recalculate_requested.connect(received.append)
        panel._calc_rows["c1"]._recalc_btn.click()
        assert received == ["c1"]

    def test_recalculate_all_button_emits_signal(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "NetVoltage", sid)
        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)

        received = []
        panel.calculated_signal_recalculate_all_requested.connect(lambda: received.append(True))
        panel._recalc_all_btn.click()
        assert received == [True]


class TestNameCollisionHandledByBackend:
    def test_duplicate_name_rejected_by_session_not_ui(self, qapp) -> None:
        sess, sid = _session_with_source()
        _create_and_resolve(sess, "c1", "SameName", sid)
        dup = CalculatedSignalDefinition(
            calc_id="c2", name="SameName", expression="a + a",
            variable_bindings={"a": ChannelRef(sid, "Va")}, reference_variable="a",
        )
        with pytest.raises(ValueError, match="already exists"):
            sess.add_calculated_signal(dup)

        panel = SessionPanel()
        panel.refresh_calculated_signals(sess)
        assert set(panel._calc_rows.keys()) == {"c1"}  # rejected signal never reaches the UI
