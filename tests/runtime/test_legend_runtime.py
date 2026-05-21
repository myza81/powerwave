"""Runtime tests for Phase 9E — Legend, Labels & Colour Management.

Tests exercise the full Qt widget stack with a live QApplication and a real
EventAnalysisSession.  Each test creates its own widgets; runtime_qapp
closes all top-level widgets after each test.

Coverage
--------
 1.  Legend visible after session canvas creation
 2.  Colour change updates live curve pen (no setData)
 3.  Display-name edit updates legend row text
 4.  Hide/show legend stable — legend survives setVisible cycles
 5.  Remove source removes legend rows
 6.  100+ legend rows do not crash
 7.  Repeated refresh_all does not create duplicate legend rows
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.sessions import EventAnalysisSession
from app.ui.session.session_canvas_controller import SessionCanvasController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runtime_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    for w in list(QApplication.topLevelWidgets()):
        w.close()
        w.deleteLater()
    app.processEvents()
    app.sendPostedEvents()
    app.processEvents()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_record(n_analog: int = 3, n_samples: int = 100):
    from app.models.channels import AnalogChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    dt = 0.001
    t = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    names = [f"VA{i}" for i in range(n_analog)]
    data = {"time": t}
    for name in names:
        data[name] = np.sin(2 * np.pi * 50 * t)

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="RT_LEGEND",
            recorder_name="REC",
            source_file="legend.cfg",
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=data,
        analog_channels=[AnalogChannel(name=n, unit="kV", index=i) for i, n in enumerate(names)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[1.0 / dt], samples_per_rate=[n_samples]),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1),
            trigger_time=datetime(2024, 1, 1),
        ),
        disturbance_info=None,
    )


def _build_session(n_sources: int = 2, n_analog: int = 3) -> EventAnalysisSession:
    session = EventAnalysisSession()
    for i in range(n_sources):
        rec = _make_record(n_analog=n_analog)
        session.add_source(rec, f"Source{i + 1}", "comtrade")
    session.default_layout()
    return session


# ---------------------------------------------------------------------------
# 1. Legend visible after session canvas creation
# ---------------------------------------------------------------------------


def test_legend_visible_after_canvas_creation(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)
    splitter.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    for canvas in ctrl._canvases.values():
        assert canvas.legend.isVisible()
        assert canvas.legend.row_count > 0

    splitter.close()


# ---------------------------------------------------------------------------
# 2. Colour change updates live curve pen without setData
# ---------------------------------------------------------------------------


def test_colour_change_updates_pen(runtime_qapp) -> None:
    session = _build_session(n_sources=1, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    channels = list(session.list_analog_channels(active_only=False))
    ch = channels[0]

    # Count setData calls to confirm no data refetch
    setdata_count = [0]
    for canvas in ctrl._canvases.values():
        for curve in canvas._curves.values():
            orig = curve.setData

            def counting(*a, _orig=orig, **kw):
                setdata_count[0] += 1
                _orig(*a, **kw)

            curve.setData = counting  # type: ignore[method-assign]

    ctrl.on_colour_changed(ch.source_id, ch.channel_name, "#ff0000")
    runtime_qapp.processEvents()

    assert setdata_count[0] == 0, "colour change must not call setData"

    # Check pen was updated
    for canvas in ctrl._canvases.values():
        curve = canvas._curves.get((ch.source_id, ch.channel_name))
        if curve is not None:
            pen_color = curve.opts["pen"].color().name()
            assert pen_color == "#ff0000"


# ---------------------------------------------------------------------------
# 3. Display-name edit updates legend row text
# ---------------------------------------------------------------------------


def test_display_name_edit_updates_legend(runtime_qapp) -> None:
    session = _build_session(n_sources=1, n_analog=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    channels = list(session.list_analog_channels(active_only=False))
    ch = channels[0]

    # Simulate legend name edit via internal handler
    ctrl._handle_legend_name(ch.source_id, ch.channel_name, "Custom Name")
    runtime_qapp.processEvents()

    # Check session updated
    updated_ch = session.get_channel(ch.source_id, ch.channel_name)
    assert updated_ch is not None
    assert updated_ch.display_name == "Custom Name"

    # Check legend row updated
    for canvas in ctrl._canvases.values():
        row = canvas.legend._rows.get((ch.source_id, ch.channel_name))
        if row is not None:
            assert "Custom Name" in row._name_lbl.text()


# ---------------------------------------------------------------------------
# 4. Hide/show legend is stable
# ---------------------------------------------------------------------------


def test_hide_show_legend_stable(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)
    splitter.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    count_before = sum(c.legend.row_count for c in ctrl._canvases.values())

    ctrl.set_legend_visible(False)
    runtime_qapp.processEvents()
    for canvas in ctrl._canvases.values():
        assert not canvas.legend.isVisible()

    ctrl.set_legend_visible(True)
    runtime_qapp.processEvents()
    for canvas in ctrl._canvases.values():
        assert canvas.legend.isVisible()

    count_after = sum(c.legend.row_count for c in ctrl._canvases.values())
    assert count_after == count_before

    splitter.close()


# ---------------------------------------------------------------------------
# 5. Remove source removes legend rows
# ---------------------------------------------------------------------------


def test_remove_source_removes_legend_rows(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    sources = session.list_sources()
    sid = sources[0].source_id

    ctrl.on_source_removed(sid)
    runtime_qapp.processEvents()

    for canvas in ctrl._canvases.values():
        for key in canvas.legend._rows:
            assert key[0] != sid


# ---------------------------------------------------------------------------
# 6. 100+ legend rows do not crash
# ---------------------------------------------------------------------------


def test_hundred_legend_rows_no_crash(runtime_qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(n_analog=60, n_samples=50)
    session.add_source(rec, "BigSource1", "comtrade")
    rec2 = _make_record(n_analog=60, n_samples=50)
    session.add_source(rec2, "BigSource2", "comtrade")
    session.default_layout()

    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)
    splitter.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    total_rows = sum(c.legend.row_count for c in ctrl._canvases.values())
    assert total_rows >= 2

    splitter.close()


# ---------------------------------------------------------------------------
# 7. Repeated refresh_all does not create duplicate legend rows
# ---------------------------------------------------------------------------


def test_repeated_refresh_no_duplicate_legend_rows(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=3)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    for _ in range(5):
        ctrl.refresh_all(session)
        runtime_qapp.processEvents()

    for canvas in ctrl._canvases.values():
        # Row count must equal the number of unique (source_id, channel_name) refs
        # rendered in this panel (channels that are active and visible)
        assert len(canvas.legend._rows) == canvas.curve_count
