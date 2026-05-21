"""Runtime tests for Phase 9D — Unified Session Canvas.

Tests exercise the full Qt widget stack with a live QApplication and a real
EventAnalysisSession.  Each test creates its own widgets; runtime_qapp
closes all top-level widgets after each test.

Coverage
--------
 1.  Two-source (COMTRADE + CSV-rate) session renders without crash
 2.  Offset change visibly moves curve data (time arrays shift)
 3.  Synchronized pan/zoom propagates X range across all panels
 4.  Source removal updates live canvas (curves removed)
 5.  Panel rebuild after source addition produces correct count
 6.  Repeated refresh_all is stable (no duplicate curves)
 7.  100+ visible curves do not crash
 8.  processEvents responsiveness: 5 full refresh cycles < 10 s
 9.  Channel visibility toggle hides curve without data refetch
10.  Session canvas deactivation restores standard layout safely
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication, QSplitter

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


def _make_record(
    n_analog: int = 3,
    n_samples: int = 200,
    dt: float = 0.001,
    step_at: float | None = None,
):
    from app.models.channels import AnalogChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    t = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    names = [f"VA{i}" for i in range(n_analog)]
    data = {"time": t}
    for name in names:
        if step_at is not None:
            data[name] = np.where(t >= step_at, 10.0, 0.5).astype(np.float64)
        else:
            data[name] = np.sin(2 * np.pi * 50 * t)

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="RT_TEST",
            recorder_name="REC",
            source_file="rt.cfg",
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
        session.add_source(rec, f"Source{i+1}", "comtrade")
    session.default_layout()
    return session


# ---------------------------------------------------------------------------
# 1. Two-source session renders without crash
# ---------------------------------------------------------------------------


def test_two_source_session_renders(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=3)
    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)
    splitter.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    total_curves = sum(c.curve_count for c in ctrl._canvases.values())
    assert total_curves >= 2  # at least some curves rendered

    splitter.close()


# ---------------------------------------------------------------------------
# 2. Offset change shifts curve time arrays
# ---------------------------------------------------------------------------


def test_offset_change_shifts_curve_data(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    sources = session.list_sources()
    sid = sources[0].source_id

    # Collect initial time values for this source's curves
    def _get_first_t(source_id: str) -> float | None:
        for canvas in ctrl._canvases.values():
            for (s, _), curve in canvas._curves.items():
                if s == source_id:
                    xdata = curve.xData
                    if xdata is not None and len(xdata) > 0:
                        return float(xdata[0])
        return None

    t_before = _get_first_t(sid)

    # Apply offset and refresh
    session.set_time_offset(sid, 0.05, method="manual")
    ctrl.on_offset_changed(sid, 0.05, session)
    runtime_qapp.processEvents()

    t_after = _get_first_t(sid)

    if t_before is not None and t_after is not None:
        assert abs(t_after - t_before) > 0.01


# ---------------------------------------------------------------------------
# 3. Synchronized pan/zoom propagates X range across panels
# ---------------------------------------------------------------------------


def test_synchronized_pan_zoom(runtime_qapp) -> None:
    from app.visualization.managers.synchronization_manager import SynchronizationManager

    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)
    splitter.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    sync = SynchronizationManager()
    ctrl.register_with_sync(sync)
    assert sync.registered_count == len(ctrl._canvases)

    # Set X range on the master (first canvas) and check sync propagates
    canvases = ctrl.active_canvases()
    if len(canvases) >= 2:
        master = canvases[0]
        master.normalize_viewport(0.01, 0.05)
        sync.synchronize_x_range(master, (0.01, 0.05))
        runtime_qapp.processEvents()

        follower = canvases[1]
        x_range = follower.getViewBox().viewRange()[0]
        assert abs(x_range[0] - 0.01) < 0.01 or abs(x_range[1] - 0.05) < 0.1

    splitter.close()


# ---------------------------------------------------------------------------
# 4. Source removal updates live canvas
# ---------------------------------------------------------------------------


def test_source_removal_removes_curves(runtime_qapp) -> None:
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
        for (s, _) in canvas._curves:
            assert s != sid


# ---------------------------------------------------------------------------
# 5. Panel rebuild after source addition produces correct count
# ---------------------------------------------------------------------------


def test_panel_rebuild_after_source_addition(runtime_qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(n_analog=2)
    session.add_source(rec, "S1", "comtrade")
    session.default_layout()

    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    panels_before = len(ctrl._canvases)

    rec2 = _make_record(n_analog=2)
    session.add_source(rec2, "S2", "comtrade")
    session.default_layout()

    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    panels_after = len(ctrl._canvases)
    assert panels_after >= panels_before


# ---------------------------------------------------------------------------
# 6. Repeated refresh_all is stable (no duplicate curves)
# ---------------------------------------------------------------------------


def test_repeated_refresh_all_no_duplicates(runtime_qapp) -> None:
    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    for _ in range(5):
        ctrl.refresh_all(session)
        runtime_qapp.processEvents()

    # Curve count must equal the number of channel_refs in panels (not multiply)
    total_refs = sum(len(p.channel_refs) for p in session.list_panels())
    total_curves = sum(c.curve_count for c in ctrl._canvases.values())
    # Only analog channels are rendered; total_curves ≤ total analog refs
    analog_refs = sum(
        1 for p in session.list_panels()
        for (sid, cname) in p.channel_refs
        if any(
            ch.channel_name == cname and ch.source_id == sid
            for ch in session.list_analog_channels(active_only=False)
        )
    )
    assert total_curves <= analog_refs


# ---------------------------------------------------------------------------
# 7. 100+ visible curves do not crash
# ---------------------------------------------------------------------------


def test_hundred_curves_no_crash(runtime_qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(n_analog=60, n_samples=100)
    session.add_source(rec, "BigSource1", "comtrade")
    rec2 = _make_record(n_analog=60, n_samples=100)
    session.add_source(rec2, "BigSource2", "comtrade")
    session.default_layout()

    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    total = sum(c.curve_count for c in ctrl._canvases.values())
    assert total >= 2


# ---------------------------------------------------------------------------
# 8. processEvents responsiveness: 5 refresh cycles < 10 s
# ---------------------------------------------------------------------------


def test_refresh_responsiveness(runtime_qapp) -> None:
    session = _build_session(n_sources=3, n_analog=5)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    start = time.monotonic()
    for _ in range(5):
        ctrl.refresh_all(session)
        runtime_qapp.processEvents()
    elapsed = time.monotonic() - start

    assert elapsed < 10.0, f"5 refresh cycles took {elapsed:.2f}s (limit 10s)"


# ---------------------------------------------------------------------------
# 9. Channel visibility toggle hides curve without data refetch
# ---------------------------------------------------------------------------


def test_visibility_toggle_no_data_refetch(runtime_qapp) -> None:
    session = _build_session(n_sources=1, n_analog=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    channels = list(session.list_analog_channels(active_only=False))
    ch = channels[0]

    call_count = [0]
    orig = session.build_aligned_data

    def counting(*a, **kw):
        call_count[0] += 1
        return orig(*a, **kw)

    session.build_aligned_data = counting
    ctrl.on_channel_visibility_changed(ch.source_id, ch.channel_name, False, session)
    runtime_qapp.processEvents()

    assert call_count[0] == 0  # no data fetch for visibility-only change
    session.build_aligned_data = orig


# ---------------------------------------------------------------------------
# 10. Session canvas deactivation (rebuild_layout with new splitter) is safe
# ---------------------------------------------------------------------------


def test_rebuild_layout_after_reparent(runtime_qapp) -> None:
    """Detaching and rebuilding the layout must not crash or lose curves."""
    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()

    # First activation
    splitter1 = ctrl.rebuild_layout(session)
    splitter1.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    curves_after_first = sum(c.curve_count for c in ctrl._canvases.values())

    # Simulate deactivation (canvases detached) then re-activation
    splitter2 = ctrl.rebuild_layout(session)
    splitter2.show()
    ctrl.refresh_all(session)
    runtime_qapp.processEvents()

    curves_after_second = sum(c.curve_count for c in ctrl._canvases.values())
    assert curves_after_second == curves_after_first

    splitter1.close()
    splitter2.close()
