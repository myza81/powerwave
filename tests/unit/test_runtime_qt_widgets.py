"""Runtime smoke tests for Qt/pyqtgraph widgets.

These tests intentionally instantiate real widgets. Import-only tests cannot
catch pyqtgraph constructor/API mismatches or scene lifecycle issues.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# FlexiblePlotCanvas — direct widget tests (canvas-level, no main window)
# ─────────────────────────────────────────────────────────────────────────────

def test_flexible_plot_canvas_instantiates(qapp: QApplication) -> None:
    pg.setConfigOptions(
        useOpenGL=True,
        antialias=False,
        foreground="w",
        background="#1E1E1E",
    )
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    canvas = FlexiblePlotCanvas()
    try:
        assert canvas._primary_plot is not None
    finally:
        canvas.close()
        qapp.processEvents()


def test_flexible_plot_canvas_set_record_before_show_supports_multi_axis(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_mixed_disturbance_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    result = make_mixed_disturbance_record()
    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(result.record)
        assert len(canvas._axis_manager.parameter_names()) > 1
    finally:
        canvas.close()
        qapp.processEvents()


def test_sparse_low_rate_record_uses_markers_and_neighbor_context(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.models import SamplingInformation
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    base = make_high_rate_record().record
    channel = dataclasses.replace(base.analog_channels[0], name="MW", unit="MW")
    record = dataclasses.replace(
        base,
        waveform_data=pd.DataFrame({
            "time": [2340.0, 2400.0],
            "MW": [100.0, 101.0],
        }),
        analog_channels=[channel],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / 60.0],
            samples_per_rate=[2],
        ),
    )

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        canvas._primary_plot.setXRange(2348.0, 2355.0, padding=0)
        qapp.processEvents()

        curve = canvas._axis_manager.get_curves()["MW"]
        x, y = curve.getData()
        y_range = curve.getViewBox().viewRange()[1]
        assert curve.opts["symbol"] == "o"
        assert list(x) == [2340.0, 2400.0]
        assert list(y) == [100.0, 101.0]
        assert y_range[0] < 100.0
        assert y_range[1] > 101.0
    finally:
        canvas.close()
        qapp.processEvents()


def test_sparse_low_rate_record_initial_view_shows_full_time_extent(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.models import SamplingInformation
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    base = make_high_rate_record().record
    channel = dataclasses.replace(base.analog_channels[0], name="MW", unit="MW")
    record = dataclasses.replace(
        base,
        waveform_data=pd.DataFrame({
            "time": [0.0, 60.0, 120.0],
            "MW": [100.0, 101.0, 102.0],
        }),
        analog_channels=[channel],
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / 60.0],
            samples_per_rate=[3],
        ),
    )

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        qapp.processEvents()

        x_range = canvas._primary_plot.getViewBox().viewRange()[0]
        curve = canvas._axis_manager.get_curves()["MW"]
        x, y = curve.getData()

        assert x_range[0] == pytest.approx(0.0)
        assert x_range[1] == pytest.approx(120.0)
        assert list(x) == [0.0, 60.0, 120.0]
        assert list(y) == [100.0, 101.0, 102.0]
    finally:
        canvas.close()
        qapp.processEvents()


def test_high_rate_record_initial_view_remains_trigger_zoomed(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record().record
    trigger_s = (
        record.timing_info.trigger_time - record.timing_info.start_time
    ).total_seconds()

    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        qapp.processEvents()

        x_start, x_end = canvas._primary_plot.getViewBox().viewRange()[0]
        assert x_start == pytest.approx(max(0.0, trigger_s - 0.2))
        assert x_end == pytest.approx(min(float(canvas._time_cache[-1]), trigger_s + 0.2))
    finally:
        canvas.close()
        qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# SynchronizationManager — runtime wiring tests (no main window)
# ─────────────────────────────────────────────────────────────────────────────

def test_synchronization_manager_runtime_x_range_and_cursor(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.managers.synchronization_manager import SynchronizationManager
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    master = FlexiblePlotCanvas()
    follower = FlexiblePlotCanvas()
    manager = SynchronizationManager()
    try:
        master.set_record(record)
        follower.set_record(record)
        manager.register_many([master, follower], master_canvas=master)

        manager.synchronize_x_range(master, (0.2, 0.4))
        manager.synchronize_cursor(master, 0.3)
        qapp.processEvents()

        assert follower._primary_plot.getViewBox().viewRange()[0] == pytest.approx([0.2, 0.4])
        assert follower._cursor.value() == pytest.approx(0.3)
    finally:
        manager.clear()
        master.close()
        follower.close()
        qapp.processEvents()


def test_synchronization_manager_runtime_digital_timeline_follows_canvas(
    qapp: QApplication,
) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.visualization.managers.synchronization_manager import SynchronizationManager
    from app.visualization.widgets.digital_event_timeline import DigitalEventTimeline
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    canvas = FlexiblePlotCanvas()
    timeline = DigitalEventTimeline()
    manager = SynchronizationManager()
    try:
        canvas.set_record(record)
        timeline.set_record(record)
        manager.register_canvas(canvas, set_as_master=True)
        manager.register_canvas(timeline)

        manager.synchronize_x_range(canvas, (0.1, 0.35))
        manager.synchronize_cursor(canvas, 0.22)
        qapp.processEvents()

        assert timeline.getPlotItem().getViewBox().viewRange()[0] == pytest.approx([0.1, 0.35])
        assert timeline._cursor.value() == pytest.approx(0.22)
    finally:
        manager.clear()
        canvas.close()
        timeline.close()
        qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# FlexiblePlotCanvas — RMS and channel visibility (direct canvas tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_visibility_removes_rms_overlay_with_hidden_channel(
    qapp: QApplication,
) -> None:
    from app.analytics.rms.rms_models import RMSDisplayMode
    from app.data.synthetic import make_high_rate_record
    from app.visualization.widgets.flexible_plot_canvas import FlexiblePlotCanvas

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    canvas = FlexiblePlotCanvas()
    try:
        canvas.set_record(record)
        canvas.set_rms_display_mode(RMSDisplayMode.OVERLAY)
        qapp.processEvents()

        visible = canvas.visible_channel_names()
        hidden = visible[0]
        assert hidden in canvas._rms_curves

        canvas.set_visible_channels(visible[1:])
        qapp.processEvents()

        assert hidden not in canvas._axis_manager.get_curves()
        assert hidden not in canvas._rms_curves
        assert set(canvas._rms_curves).issubset(set(canvas.visible_channel_names()))
    finally:
        canvas.close()
        qapp.processEvents()


# ─────────────────────────────────────────────────────────────────────────────
# Session canvas — smoke tests via PowerwaveMainWindow
# ─────────────────────────────────────────────────────────────────────────────

def _process_events(qapp: QApplication, count: int = 5) -> None:
    for _ in range(count):
        qapp.processEvents()


def test_session_canvas_activates_on_synthetic_record(qapp: QApplication) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    win = PowerwaveMainWindow()
    try:
        win._load_record_into_session(record, time_mode=TimeDisplayMode.RELATIVE)
        _process_events(qapp)

        assert win._session_canvas_active is True
        assert win._active_session is not None
        # _load_record_into_session also injects synthetic sequence/harmonic sources,
        # so total source count is >= 1 (1 primary + 0-4 synthetic).
        sources = win._active_session.list_sources()
        primary = [s for s in sources if not s.provider_type.startswith("synthetic:")]
        assert len(primary) == 1
        canvases = win._session_canvas_controller.active_canvases()
        assert len(canvases) >= 1
        total_curves = sum(c.curve_count for c in canvases)
        assert total_curves > 0
    finally:
        win.close()
        qapp.processEvents()


def test_session_canvas_activates_on_comtrade_load(qapp: QApplication) -> None:
    from app.providers import ComtradeProvider, ProviderManager
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    manager = ProviderManager()
    manager.register_provider(ComtradeProvider())
    record = manager.load(Path("samples/comtrade/pulu_20260306.cfg"))
    win = PowerwaveMainWindow()
    try:
        win._load_record_into_session(record, time_mode=TimeDisplayMode.RELATIVE)
        _process_events(qapp)

        assert win._session_canvas_active is True
        sources = win._active_session.list_sources()
        primary = [s for s in sources if not s.provider_type.startswith("synthetic:")]
        assert len(primary) == 1
        canvases = win._session_canvas_controller.active_canvases()
        assert len(canvases) >= 1
        total_curves = sum(c.curve_count for c in canvases)
        assert total_curves > 0
    finally:
        win.close()
        qapp.processEvents()


def test_session_canvas_channel_visibility_toggle_updates_model(qapp: QApplication) -> None:
    from app.data.synthetic import make_high_rate_record
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    win = PowerwaveMainWindow()
    try:
        win._load_record_into_session(record, time_mode=TimeDisplayMode.RELATIVE)
        _process_events(qapp)

        assert win._active_session is not None
        session = win._active_session
        all_sources = session.list_sources()
        source_id = next(
            s.source_id for s in all_sources if not s.provider_type.startswith("synthetic:")
        )
        channels = [
            c for c in session.list_analog_channels(active_only=False)
            if c.source_id == source_id
        ]
        assert len(channels) > 0
        ch = channels[0]
        assert ch.is_visible is True

        session.set_channel_visibility(source_id, ch.channel_name, False)
        _process_events(qapp)

        updated = {
            c.channel_name: c
            for c in session.list_analog_channels(active_only=False)
            if c.source_id == source_id
        }
        assert updated[ch.channel_name].is_visible is False
    finally:
        win.close()
        qapp.processEvents()


def test_session_canvas_two_source_session_has_both_sources(qapp: QApplication) -> None:
    from app.data.synthetic import make_high_rate_record, make_mixed_disturbance_record
    from app.sessions.event_session import EventAnalysisSession
    from app.ui.main_window.main_window import PowerwaveMainWindow

    rec_a = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    rec_b = make_mixed_disturbance_record().record
    win = PowerwaveMainWindow()
    try:
        session = EventAnalysisSession()
        session.add_source(rec_a, "source_a", "synthetic", None)
        session.add_source(rec_b, "source_b", "synthetic", None)
        session.default_layout()
        win._active_session = session
        win._session_canvas_action.setEnabled(True)
        win._save_manifest_action.setEnabled(True)
        win._ensure_session_panel().refresh_all(session)
        win._activate_session_canvas()
        _process_events(qapp)

        assert win._session_canvas_active is True
        sources = win._active_session.list_sources()
        assert len(sources) == 2
        assert {s.display_name for s in sources} == {"source_a", "source_b"}
        canvases = win._session_canvas_controller.active_canvases()
        assert len(canvases) >= 1
    finally:
        win.close()
        qapp.processEvents()


def test_session_canvas_deactivate_shows_placeholder(qapp: QApplication) -> None:
    from PyQt6.QtWidgets import QLabel
    from app.data.synthetic import make_high_rate_record
    from app.ui.main_window.main_window import PowerwaveMainWindow
    from app.visualization.axis.datetime_axis import TimeDisplayMode

    record = make_high_rate_record(duration_s=1.0, sampling_rate_hz=1000.0).record
    win = PowerwaveMainWindow()
    try:
        win._load_record_into_session(record, time_mode=TimeDisplayMode.RELATIVE)
        _process_events(qapp)
        assert win._session_canvas_active is True

        win._deactivate_session_canvas()
        _process_events(qapp)

        assert win._session_canvas_active is False
        assert isinstance(win.centralWidget(), QLabel)
    finally:
        win.close()
        qapp.processEvents()
