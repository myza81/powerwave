"""Integration tests for the per-curve measurement architecture (Sprint 1A)
through the real rendering path: SessionCanvasController.compute_current_
measurements() reading directly from SessionCanvasWidget's already-painted
curves.

These exercise the exact bug the EAV flagged: a panel containing curves at
different sample rates (or a calculated signal alongside a real channel)
must measure every visible curve, never silently drop one because its
length differs from another's.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).
"""
from __future__ import annotations

import sys
import time as time_module
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
from app.ui.session.session_canvas_controller import SessionCanvasController


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_record(
    analog: dict[str, str],
    time: np.ndarray,
    values: dict[str, np.ndarray] | None = None,
) -> DisturbanceRecord:
    values = values or {}
    n = len(time)
    data: dict[str, object] = {"time": time}
    for name in analog:
        data[name] = values.get(name, np.sin(time))
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


def _rebuild_and_measure(ctrl: SessionCanvasController, sess: EventAnalysisSession, panel_id: str, t_a: float, t_b: float):
    canvas = ctrl._canvases[panel_id]
    canvas.set_measurement_mode(True)
    canvas.set_cursor_a_pos(t_a)
    canvas.set_cursor_b_pos(t_b)
    ctrl.set_measurement_mode(True, sess)
    return ctrl.compute_current_measurements(canvas, sess)


class TestSameSampleRateRegression:
    def test_two_same_rate_channels_one_source(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 2, 2000)
        sid = sess.add_source(
            _make_record({"A": "MW", "B": "MW"}, t, {"A": t * 2, "B": t * 3}),
            "Source A", "csv",
        )
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        panel_id = next(p.panel_id for p in sess.list_panels() if p.panel_id == "power")
        result = _rebuild_and_measure(ctrl, sess, panel_id, 0.5, 1.0)
        assert result is not None
        assert len(result.channels) == 2
        assert all(ch.available for ch in result.channels)


class TestMixedSampleRatesThroughCanvas:
    def test_two_real_channels_different_rates_both_measured(self, qapp) -> None:
        """The core EAV bug: a fast and a slow real channel sharing a panel
        used to silently drop whichever one didn't match the first
        channel's decimated array length."""
        sess = EventAnalysisSession()
        t_fast = np.linspace(0, 2, 20000)   # decimates to 4000 pts
        t_slow = np.linspace(0, 2, 100)     # stays at 100 pts (PMU-like)
        sid_a = sess.add_source(_make_record({"Fast": "MW"}, t_fast), "Relay", "csv")
        sid_b = sess.add_source(_make_record({"Slow": "MW"}, t_slow), "PMU", "csv")
        sess.default_layout()
        # Force both onto the same panel so they genuinely share one canvas.
        power_panel = next(p for p in sess.list_panels() if p.panel_id == "power")
        sess.set_channel_panel(sid_b, "Slow", power_panel.panel_id)

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 0.5, 1.0)
        assert result is not None
        names = {ch.name: ch for ch in result.channels}
        assert set(names) == {"Fast", "Slow"}
        assert names["Fast"].available
        assert names["Slow"].available

    def test_five_mixed_rates_all_measured(self, qapp) -> None:
        """6400 sps, 3200 sps, 1600 sps, PMU 50 sps, and a full-resolution
        calculated signal, all sharing one panel."""
        sess = EventAnalysisSession()
        span = (0.0, 2.0)
        sid_a = sess.add_source(_make_record({"R6400": "MW"}, np.linspace(*span, 12800)), "S6400", "csv")
        sid_b = sess.add_source(_make_record({"R3200": "MW"}, np.linspace(*span, 6400)), "S3200", "csv")
        sid_c = sess.add_source(_make_record({"R1600": "MW"}, np.linspace(*span, 3200)), "S1600", "csv")
        sid_d = sess.add_source(_make_record({"PMU50": "MW"}, np.linspace(*span, 100)), "SPMU", "csv")
        sess.default_layout()
        power_panel = next(p for p in sess.list_panels() if p.panel_id == "power")
        for sid, ch in [(sid_b, "R3200"), (sid_c, "R1600"), (sid_d, "PMU50")]:
            sess.set_channel_panel(sid, ch, power_panel.panel_id)

        defn = CalculatedSignalDefinition(
            calc_id="calc1", name="NetPower", expression="a + a",
            variable_bindings={"a": ChannelRef(sid_a, "R6400")}, reference_variable="a",
        )
        sess.add_calculated_signal(defn)
        CalculatedSignalResolutionService(sess).resolve_one("calc1")

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        placement = sess.ensure_calculated_signal_panel("calc1")
        assert placement == ("power", False)  # panel already exists (real MW channels routed here)
        ctrl.refresh_calculated_signals(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 0.5, 1.5)
        assert result is not None
        assert len(result.channels) == 5
        for ch in result.channels:
            assert ch.available, f"{ch.name} unexpectedly unavailable: {ch.unavailable_reason}"


class TestCalculatedSignalRequiresNoSpecialCode:
    def test_calc_signal_alone_in_panel_measured_like_any_curve(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 500)
        sid = sess.add_source(_make_record({"A": "MW"}, t), "Source A", "csv")
        sess.default_layout()
        defn = CalculatedSignalDefinition(
            calc_id="calc1", name="Doubled", expression="a + a",
            variable_bindings={"a": ChannelRef(sid, "A")}, reference_variable="a",
        )
        sess.add_calculated_signal(defn)
        CalculatedSignalResolutionService(sess).resolve_one("calc1")

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        ctrl.refresh_calculated_signals(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 0.2, 0.6)
        assert result is not None
        calc_rows = [ch for ch in result.channels if "Doubled" in ch.name]
        assert len(calc_rows) == 1
        assert calc_rows[0].available
        assert calc_rows[0].name.startswith("ƒ")

    def test_stale_calc_signal_still_measured(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 500)
        sid = sess.add_source(_make_record({"A": "MW"}, t), "Source A", "csv")
        sess.default_layout()
        defn = CalculatedSignalDefinition(
            calc_id="calc1", name="Doubled", expression="a + a",
            variable_bindings={"a": ChannelRef(sid, "A")}, reference_variable="a",
        )
        sess.add_calculated_signal(defn)
        CalculatedSignalResolutionService(sess).resolve_one("calc1")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        ctrl.refresh_calculated_signals(sess)

        sess.set_time_offset(sid, 0.1)  # marks calc1 STALE, keeps prior arrays
        ctrl.refresh_calculated_signals(sess, ["calc1"])

        result = _rebuild_and_measure(ctrl, sess, "power", 0.2, 0.6)
        calc_rows = [ch for ch in result.channels if "Doubled" in ch.name]
        assert calc_rows[0].available
        assert "(stale)" in calc_rows[0].name


class TestLabelDisambiguation:
    def test_colliding_channel_names_across_sources_get_prefixed(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 200)
        sid_a = sess.add_source(_make_record({"Va": "kV"}, t), "Source A", "csv")
        sid_b = sess.add_source(_make_record({"Va": "kV"}, t), "Source B", "csv")
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        result = _rebuild_and_measure(ctrl, sess, "voltage", 0.2, 0.6)
        names = {ch.name for ch in result.channels}
        assert names == {"Source A/Va", "Source B/Va"}

    def test_non_colliding_names_keep_plain_labels_even_with_multiple_sources(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 200)
        sid_a = sess.add_source(_make_record({"Fast": "MW"}, t), "Relay", "csv")
        sid_b = sess.add_source(_make_record({"Slow": "MW"}, t), "PMU", "csv")
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 0.2, 0.6)
        names = {ch.name for ch in result.channels}
        assert names == {"Fast", "Slow"}


class TestHiddenCurveExcluded:
    def test_hidden_real_channel_excluded_from_measurement(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 500)
        sid = sess.add_source(_make_record({"A": "MW", "B": "MW"}, t), "Source A", "csv")
        sess.default_layout()
        sess.set_channel_visibility(sid, "B", False)

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 0.2, 0.6)
        names = {ch.name for ch in result.channels}
        assert "A" in names
        assert "B" not in names


class TestMultiplePanelsIndependent:
    def test_cursor_in_one_panel_does_not_affect_another(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 500)
        sid_a = sess.add_source(_make_record({"Va": "kV"}, t), "Source A", "csv")
        sid_b = sess.add_source(_make_record({"Ia": "A"}, t), "Source B", "csv")
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        voltage_result = _rebuild_and_measure(ctrl, sess, "voltage", 0.1, 0.3)
        current_result = _rebuild_and_measure(ctrl, sess, "current", 0.6, 0.9)

        assert voltage_result is not None and current_result is not None
        assert voltage_result.t_a == pytest.approx(0.1)
        assert current_result.t_a == pytest.approx(0.6)
        assert {ch.name for ch in voltage_result.channels} == {"Va"}
        assert {ch.name for ch in current_result.channels} == {"Ia"}


class TestCursorOutsideAllCurves:
    def test_cursors_beyond_all_data_report_unavailable_not_none(self, qapp) -> None:
        sess = EventAnalysisSession()
        t = np.linspace(0, 1, 500)
        sid = sess.add_source(_make_record({"A": "MW"}, t), "Source A", "csv")
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        result = _rebuild_and_measure(ctrl, sess, "power", 50.0, 60.0)
        assert result is not None  # curves exist, just unavailable -- not None
        assert len(result.channels) == 1
        assert not result.channels[0].available


class TestPerformanceThroughController:
    def test_large_real_channel_measurement_is_fast(self, qapp) -> None:
        sess = EventAnalysisSession()
        n = 1_000_000
        t = np.linspace(0, 100, n)
        sid = sess.add_source(_make_record({"A": "MW"}, t), "Source A", "csv")
        sess.default_layout()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        started = time_module.perf_counter()
        result = _rebuild_and_measure(ctrl, sess, "power", 10.0, 20.0)
        elapsed = time_module.perf_counter() - started
        assert result is not None
        assert result.channels[0].available
        assert elapsed < 5.0, f"measurement took {elapsed:.2f}s"
