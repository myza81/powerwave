"""Unit tests for Calculated Signals Session Canvas integration (Phase 3B)
in app.ui.session.session_canvas_controller.

Uses the repository's offscreen Qt conventions (QT_QPA_PLATFORM=offscreen is
set process-wide by tests/conftest.py; a module-scoped `qapp` fixture
provides a single QApplication).

Uses only generic, synthetic session fixtures -- no filename, station, or
event identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

import sys
import time as time_module
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from PyQt6.QtWidgets import QApplication

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)
from app.calculated_signals.resolver import CalculatedSignalResolutionService
from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.ui.session.session_canvas_controller import (
    SessionCanvasController,
    _calc_curve_key,
    _is_calc_curve_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# QApplication fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Generic synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    analog: dict[str, str | None],
    digital: list[str] | None = None,
    time: np.ndarray | None = None,
    values: dict[str, np.ndarray] | None = None,
    n: int = 200,
) -> DisturbanceRecord:
    digital = digital or []
    values = values or {}
    if time is not None:
        n = len(time)
    else:
        time = np.linspace(0, 10, n)

    data: dict[str, object] = {"time": time}
    for name in analog:
        data[name] = values.get(name, np.sin(time))
    for name in digital:
        data[name] = np.zeros(n, dtype=np.int8)

    df = pd.DataFrame(data)
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name=n, unit=u, index=i) for i, (n, u) in enumerate(analog.items())],
        digital_channels=[DigitalChannel(name=n, index=i) for i, n in enumerate(digital)],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
    )


def _two_source_session() -> tuple[EventAnalysisSession, str, str]:
    sess = EventAnalysisSession()
    sid_a = sess.add_source(_make_record({"A": "MW"}), "Source A", "csv")
    sid_b = sess.add_source(_make_record({"B": "MW"}), "Source B", "csv")
    sess.default_layout()
    return sess, sid_a, sid_b


def _create_and_resolve(
    sess: EventAnalysisSession, calc_id: str, name: str,
    expression: str, bindings: dict[str, ChannelRef], reference: str,
    output_unit: str | None = None,
) -> None:
    defn = CalculatedSignalDefinition(
        calc_id=calc_id, name=name, expression=expression,
        variable_bindings=bindings, reference_variable=reference,
        output_unit=output_unit,
    )
    sess.add_calculated_signal(defn)
    CalculatedSignalResolutionService(sess).resolve_one(calc_id)


def _sync(ctrl: SessionCanvasController, sess: EventAnalysisSession, calc_ids=None) -> None:
    """Mirror main_window._sync_calculated_signals_to_canvas for tests:
    ensure panels, rebuild layout if a brand-new panel appeared, repaint."""
    ids = calc_ids if calc_ids is not None else [
        e.definition.calc_id for e in sess.list_calculated_signals()
    ]
    needs_rebuild = False
    for calc_id in ids:
        placement = sess.ensure_calculated_signal_panel(calc_id)
        if placement is not None and placement[1]:
            needs_rebuild = True
    if needs_rebuild:
        ctrl.rebuild_layout(sess)
    ctrl.refresh_calculated_signals(sess, calc_ids)


def _canvas_for(ctrl: SessionCanvasController, sess: EventAnalysisSession, calc_id: str):
    panel_id = ctrl._calc_panel_by_id[calc_id]
    return ctrl._canvases[panel_id]


# ─────────────────────────────────────────────────────────────────────────────
# Curve identity
# ─────────────────────────────────────────────────────────────────────────────


class TestCalcCurveIdentity:
    def test_calc_curve_key_uses_prefix_and_name(self) -> None:
        key = _calc_curve_key("abc-123", "Net Power")
        assert key == ("calc:abc-123", "Net Power")

    def test_is_calc_curve_key_true_for_calc_keys(self) -> None:
        assert _is_calc_curve_key(("calc:abc-123", "Net Power"))

    def test_is_calc_curve_key_false_for_real_source_keys(self) -> None:
        assert not _is_calc_curve_key(("7e5acefa-1a0b-4913-b1c7-2f75e1bd21a6", "Va"))

    def test_calc_prefix_cannot_collide_with_real_uuid_source_id(self) -> None:
        import uuid
        for _ in range(1000):
            real_id = str(uuid.uuid4())
            assert not real_id.startswith("calc:")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────


class TestCalcCurveRendering:
    def test_newly_created_signal_rendered(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a - b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)

        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert key in canvas._curves

    def test_correct_time_and_value_arrays(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a - b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)

        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        result = sess.get_calculated_signal_result("c1")
        # getOriginalDataset() -- the raw arrays passed to setData(), before
        # pyqtgraph's own clip-to-view/downsampling reduces getData()'s
        # on-screen representation (see TestPerformance for that distinction).
        x_data, y_data = canvas._curves[key].getOriginalDataset()
        np.testing.assert_array_almost_equal(x_data, result.time)
        np.testing.assert_array_almost_equal(y_data, result.values)

    def test_panel_routed_by_unit_power(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"MWChan": "MW"}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "TotalPower", "a + a", {"a": ChannelRef(sid, "MWChan")}, "a")
        assert sess.get_calculated_signal_result("c1").unit == "MW"
        panel_id, _ = sess.ensure_calculated_signal_panel("c1")
        assert panel_id == "power"

    def test_panel_routed_by_unit_voltage(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"Va": "kV"}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "NetVoltage", "a + a", {"a": ChannelRef(sid, "Va")}, "a")
        panel_id, _ = sess.ensure_calculated_signal_panel("c1")
        assert panel_id == "voltage"

    def test_panel_routed_by_unit_current(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"Ia": "A"}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "NetCurrent", "a + a", {"a": ChannelRef(sid, "Ia")}, "a")
        panel_id, _ = sess.ensure_calculated_signal_panel("c1")
        assert panel_id == "current"

    def test_panel_routed_by_unit_frequency(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"Freq": "Hz"}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "NetFreq", "a + a", {"a": ChannelRef(sid, "Freq")}, "a")
        panel_id, _ = sess.ensure_calculated_signal_panel("c1")
        assert panel_id == "frequency"

    def test_panel_routed_to_other_for_unknown_unit(self, qapp) -> None:
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"Weird": None}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "WeirdCalc", "a + 1", {"a": ChannelRef(sid, "Weird")}, "a")
        panel_id, _ = sess.ensure_calculated_signal_panel("c1")
        assert panel_id == "other"

    def test_new_panel_created_when_none_existing(self, qapp) -> None:
        """A calculated MW result with no existing Power panel in the
        session must still get one created, plus a canvas widget for it."""
        sess = EventAnalysisSession()
        sid = sess.add_source(_make_record({"SomeVal": "MW"}), "A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "Derived", "a + a", {"a": ChannelRef(sid, "SomeVal")}, "a")

        # Simulate "no existing power panel" (e.g. deactivated/removed real
        # channel scenarios elsewhere already leave the calc signal's panel
        # missing) by clearing the panel registry outright.
        sess._panels.clear()
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        assert "power" not in ctrl._canvases
        _sync(ctrl, sess)
        assert "power" in ctrl._canvases
        key = _calc_curve_key("c1", "Derived")
        assert key in ctrl._canvases["power"]._curves


# ─────────────────────────────────────────────────────────────────────────────
# Visibility
# ─────────────────────────────────────────────────────────────────────────────


class TestCalcCurveVisibility:
    def test_hide_shows_curve_invisible(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert canvas._curves[key].isVisible()

        sess.set_calculated_signal_visible("c1", False)
        _sync(ctrl, sess, ["c1"])
        assert not canvas._curves[key].isVisible()

    def test_show_restores_without_recalculation(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        sess.set_calculated_signal_visible("c1", False)
        _sync(ctrl, sess)

        first_computed_at = sess.get_calculated_signal_result("c1").computed_at
        sess.set_calculated_signal_visible("c1", True)
        _sync(ctrl, sess, ["c1"])
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert canvas._curves[key].isVisible()
        assert sess.get_calculated_signal_result("c1").computed_at == first_computed_at


# ─────────────────────────────────────────────────────────────────────────────
# Deletion
# ─────────────────────────────────────────────────────────────────────────────


class TestCalcCurveDeletion:
    def test_deletion_removes_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert key in canvas._curves

        ctrl.remove_calculated_signal_curve("c1")
        assert key not in canvas._curves
        sess.remove_calculated_signal("c1")
        assert sess.get_calculated_signal("c1") is None

    def test_deletion_does_not_remove_real_source(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        ctrl.remove_calculated_signal_curve("c1")
        sess.remove_calculated_signal("c1")
        assert sess.get_source(sid_a) is not None
        assert sess.get_source(sid_b) is not None


# ─────────────────────────────────────────────────────────────────────────────
# STALE / ERROR / no-result handling
# ─────────────────────────────────────────────────────────────────────────────


class TestStaleAndErrorHandling:
    def test_stale_curve_retained_with_label(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        sess.set_time_offset(sid_b, 0.25)  # marks c1 STALE, preserves arrays
        assert sess.get_calculated_signal_result("c1").status == CalculationStatus.STALE
        _sync(ctrl, sess, ["c1"])

        key = _calc_curve_key("c1", "NetPower")
        assert key in canvas._curves  # retained, not removed
        meta = canvas._metadata[key]
        assert "(stale)" in meta.display_name

    def test_error_or_no_result_does_not_fabricate_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="NeverCalculated", expression="a + b",
            variable_bindings={"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")},
            reference_variable="a",
        )
        sess.add_calculated_signal(defn)  # never resolved -- result is None
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        ctrl.refresh_calculated_signals(sess)

        key = _calc_curve_key("c1", "NeverCalculated")
        for canvas in ctrl._canvases.values():
            assert key not in canvas._curves

    def test_error_result_removes_existing_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert key in canvas._curves

        error_result = CalculatedSignalResult(
            calc_id="c1", time=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]),
            validity_mask=np.array([True, True]), unit="MW",
            status=CalculationStatus.ERROR, error_message="forced error",
            computed_at=datetime.now().astimezone(),
        )
        sess.set_calculated_signal_result("c1", error_result)
        ctrl.refresh_calculated_signals(sess, ["c1"])
        assert key not in canvas._curves


# ─────────────────────────────────────────────────────────────────────────────
# refresh_all() does not sweep calc curves
# ─────────────────────────────────────────────────────────────────────────────


class TestRefreshAllDoesNotSweepCalcCurves:
    def test_refresh_all_leaves_calc_curve_intact(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        assert key in canvas._curves

        ctrl.refresh_all(sess)  # real-channel refresh sweep
        assert key in canvas._curves  # still present -- not swept


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────


class TestThemeCompatibility:
    def test_theme_switch_after_creation_does_not_remove_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")

        ctrl.set_canvas_theme("light")
        assert key in canvas._curves
        ctrl.set_canvas_theme("dark")
        assert key in canvas._curves


# ─────────────────────────────────────────────────────────────────────────────
# Downsampling / clip-to-view reuse
# ─────────────────────────────────────────────────────────────────────────────


class TestDownsamplingReuse:
    def test_primary_axis_calc_curve_gets_downsampling_enabled(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        curve = canvas._curves[key]
        # Same PlotDataItem downsampling/clip-to-view path real curves use --
        # SessionCanvasWidget.update_curve() applies this identically.
        opts = curve.opts
        assert opts.get("autoDownsample") or opts.get("downsample")


# ─────────────────────────────────────────────────────────────────────────────
# Crosshair
# ─────────────────────────────────────────────────────────────────────────────


class TestCrosshairCompatibility:
    def test_crosshair_values_include_calc_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        received = []
        canvas.crosshair_values_changed.connect(lambda t, values: received.append((t, values)))
        canvas._emit_crosshair_values(1.0)
        assert received
        labels = [v[0] for v in received[0][1]]
        assert any(k.startswith("calc:") for k in labels)

    def test_crosshair_ignores_hidden_calc_curve(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        sess.set_calculated_signal_visible("c1", False)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        received = []
        canvas.crosshair_values_changed.connect(lambda t, values: received.append((t, values)))
        canvas._emit_crosshair_values(1.0)
        if received:
            labels = [v[0] for v in received[0][1]]
            assert not any(k.startswith("calc:") for k in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────


class TestMeasurementCompatibility:
    def test_point_measurement_includes_calc_signal(self, qapp) -> None:
        sess = EventAnalysisSession()
        time = np.linspace(0, 10, 500)
        sid_a = sess.add_source(_make_record({"A": "MW"}, time=time), "Source A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "DoubleA", "a + a", {"a": ChannelRef(sid_a, "A")}, "a")

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        ctrl.set_measurement_mode(True, sess)
        canvas.set_measurement_mode(True)
        canvas.set_cursor_a_pos(1.0)
        canvas.set_cursor_b_pos(2.0)
        result = ctrl.compute_current_measurements(canvas, sess)
        assert result is not None
        names = [c.name for c in result.channels]
        assert any("DoubleA" in n for n in names)

    def test_stale_result_remains_measurable(self, qapp) -> None:
        sess = EventAnalysisSession()
        time = np.linspace(0, 10, 500)
        sid_a = sess.add_source(_make_record({"A": "MW"}, time=time), "Source A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "DoubleA", "a + a", {"a": ChannelRef(sid_a, "A")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        sess.set_time_offset(sid_a, 0.1)  # marks stale, preserves arrays
        _sync(ctrl, sess, ["c1"])

        ctrl.set_measurement_mode(True, sess)
        canvas.set_measurement_mode(True)
        canvas.set_cursor_a_pos(1.0)
        canvas.set_cursor_b_pos(2.0)
        result = ctrl.compute_current_measurements(canvas, sess)
        assert result is not None
        names = [c.name for c in result.channels]
        assert any("(stale)" in n for n in names)


# ─────────────────────────────────────────────────────────────────────────────
# Recalculation replaces curve data without duplicate creation
# ─────────────────────────────────────────────────────────────────────────────


class TestRecalculationInPlace:
    def test_recalculation_updates_same_curve_object(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "NetPower")
        curve_before = canvas._curves[key]

        sess.set_time_offset(sid_b, 0.2)
        CalculatedSignalResolutionService(sess).resolve_for_source(sid_b)
        _sync(ctrl, sess, list(sess.get_calculated_dependents_for_source(sid_b)))

        curve_after = canvas._curves[key]
        assert curve_before is curve_after  # same PlotDataItem -- no duplicate
        assert len([c for c in canvas._curves if c == key]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Architecture invariants (no synthetic source/record)
# ─────────────────────────────────────────────────────────────────────────────


class TestNoSyntheticSourceRegression:
    def test_calc_signal_never_enters_sources(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        assert "calc:c1" not in sess._sources
        assert "c1" not in sess._sources
        assert len(sess.list_sources()) == 2

    def test_no_disturbance_record_or_source_created_for_calc_signal(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        before_ids = {s.source_id for s in sess.list_sources()}
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        after_ids = {s.source_id for s in sess.list_sources()}
        assert before_ids == after_ids  # no new source, no provider_type invented


# ─────────────────────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_large_calc_curve_renders_without_python_loops(self, qapp) -> None:
        n = 500_000
        time = np.linspace(0, 100, n)
        sess = EventAnalysisSession()
        sid_a = sess.add_source(_make_record({"A": "MW"}, time=time), "Source A", "csv")
        sess.default_layout()
        _create_and_resolve(sess, "c1", "BigCalc", "a + a", {"a": ChannelRef(sid_a, "A")}, "a")

        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)

        started = time_module.perf_counter()
        _sync(ctrl, sess)
        elapsed = time_module.perf_counter() - started

        canvas = _canvas_for(ctrl, sess, "c1")
        key = _calc_curve_key("c1", "BigCalc")
        # getOriginalDataset() is the raw array passed to setData() -- full
        # resolution, before pyqtgraph's own clip-to-view/auto-downsampling
        # reduces what getData() returns for on-screen rendering.
        x_data, _ = canvas._curves[key].getOriginalDataset()
        assert len(x_data) == n  # full resolution, no decimation of the underlying data
        assert elapsed < 10.0, f"refresh_calculated_signals took {elapsed:.2f}s for {n} samples"

    def test_refreshing_one_calc_signal_does_not_repaint_source_curves(self, qapp) -> None:
        sess, sid_a, sid_b = _two_source_session()
        _create_and_resolve(sess, "c1", "NetPower", "a + b",
                             {"a": ChannelRef(sid_a, "A"), "b": ChannelRef(sid_b, "B")}, "a")
        ctrl = SessionCanvasController()
        ctrl.rebuild_layout(sess)
        ctrl.refresh_all(sess)
        _sync(ctrl, sess)
        canvas = _canvas_for(ctrl, sess, "c1")

        real_key = (sid_a, "A")
        curve_before = canvas._curves[real_key]
        ctrl.refresh_calculated_signals(sess, ["c1"])
        curve_after = canvas._curves[real_key]
        assert curve_before is curve_after  # untouched by a scoped calc refresh
