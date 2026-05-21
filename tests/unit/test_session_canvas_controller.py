"""Unit tests for SessionCanvasController and SessionCanvasWidget — Phase 9D.

Coverage
--------
 1.  rebuild_layout creates correct number of panel canvases
 2.  rebuild_layout reuses existing canvas widgets (topology unchanged)
 3.  rebuild_layout destroys stale canvases when panels are removed
 4.  refresh_all populates curves via build_aligned_data
 5.  on_offset_changed refreshes only the affected source channels
 6.  on_channel_visibility_changed hides/shows one curve, no data fetch
 7.  on_source_removed clears all curves and zero line for that source
 8.  NaN values in aligned data are passed through without crash
 9.  zero marker created for each active source on refresh_all
10.  zero marker removed when source is deactivated
11.  curve registry reuse: same PlotDataItem after second refresh
12.  source color assignment is stable across repeated refresh calls
13.  SessionCanvasWidget.update_curve creates then reuses PlotDataItem
14.  SessionCanvasWidget.set_curve_visible toggles visibility
15.  SessionCanvasWidget.remove_source clears curves and zero line
16.  SessionCanvasWidget.clear_all empties both registries
17.  common_grid: all channels from same source share time length
18.  _session_window returns valid range from active source offsets
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication, QSplitter

from app.sessions import EventAnalysisSession
from app.ui.session.session_canvas_controller import (
    SessionCanvasController,
    _session_window,
)
from app.visualization.widgets.session_canvas import SessionCanvasWidget


# ---------------------------------------------------------------------------
# QApplication fixture (module-scoped — shared across all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------


def _make_record(n_analog: int = 3, n_samples: int = 100, dt: float = 0.001):
    from app.models.channels import AnalogChannel, DigitalChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    t = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    data = {"time": t}
    names = [f"VA{i}" for i in range(n_analog)]
    for name in names:
        data[name] = np.sin(2 * np.pi * 50 * t)

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="REC",
            source_file="test.cfg",
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
# 1. rebuild_layout creates correct panel count
# ---------------------------------------------------------------------------


def test_rebuild_layout_panel_count(qapp) -> None:
    session = _build_session(n_sources=2)
    ctrl = SessionCanvasController()
    splitter = ctrl.rebuild_layout(session)

    n_panels = len([p for p in session.list_panels() if p.is_visible])
    assert len(ctrl._canvases) == n_panels
    assert n_panels > 0


# ---------------------------------------------------------------------------
# 2. rebuild_layout reuses existing canvases when topology is unchanged
# ---------------------------------------------------------------------------


def test_rebuild_layout_reuses_canvases(qapp) -> None:
    session = _build_session(n_sources=2)
    ctrl = SessionCanvasController()

    splitter1 = ctrl.rebuild_layout(session)   # keep alive — simulates setCentralWidget
    first_ids = {pid: id(c) for pid, c in ctrl._canvases.items()}

    splitter2 = ctrl.rebuild_layout(session)  # same topology
    second_ids = {pid: id(c) for pid, c in ctrl._canvases.items()}

    assert first_ids == second_ids
    splitter1.close()
    splitter2.close()


# ---------------------------------------------------------------------------
# 3. rebuild_layout returns a QSplitter
# ---------------------------------------------------------------------------


def test_rebuild_layout_returns_splitter(qapp) -> None:
    session = _build_session(n_sources=1)
    ctrl = SessionCanvasController()
    result = ctrl.rebuild_layout(session)
    assert isinstance(result, QSplitter)


# ---------------------------------------------------------------------------
# 4. refresh_all populates curves from build_aligned_data
# ---------------------------------------------------------------------------


def test_refresh_all_populates_curves(qapp) -> None:
    session = _build_session(n_sources=2, n_analog=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    total = sum(c.curve_count for c in ctrl._canvases.values())
    # 2 sources × 2 channels = 4 channel refs spread across panels
    assert total >= 4


# ---------------------------------------------------------------------------
# 5. on_offset_changed refreshes only the affected source
# ---------------------------------------------------------------------------


def test_on_offset_changed_refreshes_only_affected_source(qapp) -> None:
    session = _build_session(n_sources=2, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    sources = session.list_sources()
    sid1 = sources[0].source_id

    # Capture which keys are updated by patching update_curve on all canvases
    updated_sources: set[str] = set()
    for canvas in ctrl._canvases.values():
        original = canvas.update_curve

        def patched(source_id, channel_name, time, values, *, color, visible, _orig=original, _tracking=updated_sources, **kwargs):
            _tracking.add(source_id)
            _orig(source_id, channel_name, time, values, color=color, visible=visible, **kwargs)

        canvas.update_curve = patched

    session.set_time_offset(sid1, 0.01, method="manual")
    ctrl.on_offset_changed(sid1, 0.01, session)

    sources_updated = updated_sources
    assert sid1 in sources_updated
    # The second source's curves should NOT have been updated
    sid2 = sources[1].source_id
    assert sid2 not in sources_updated


# ---------------------------------------------------------------------------
# 6. on_channel_visibility_changed toggles visibility, no data fetch
# ---------------------------------------------------------------------------


def test_on_channel_visibility_changed_no_data_fetch(qapp) -> None:
    session = _build_session(n_sources=1, n_analog=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    channels = list(session.list_analog_channels(active_only=False))
    assert channels, "expected at least one channel"
    ch = channels[0]

    build_call_count = [0]
    original_build = session.build_aligned_data

    def counting_build(*args, **kwargs):
        build_call_count[0] += 1
        return original_build(*args, **kwargs)

    session.build_aligned_data = counting_build

    ctrl.on_channel_visibility_changed(ch.source_id, ch.channel_name, False, session)
    assert build_call_count[0] == 0  # no data fetch occurred

    session.build_aligned_data = original_build


# ---------------------------------------------------------------------------
# 7. on_source_removed clears curves for that source
# ---------------------------------------------------------------------------


def test_on_source_removed_clears_curves(qapp) -> None:
    session = _build_session(n_sources=2, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    sources = session.list_sources()
    sid = sources[0].source_id

    ctrl.on_source_removed(sid)

    for canvas in ctrl._canvases.values():
        for (s, _) in canvas._curves:
            assert s != sid


# ---------------------------------------------------------------------------
# 8. NaN values in aligned data pass through without crash
# ---------------------------------------------------------------------------


def test_nan_aligned_data_no_crash(qapp) -> None:
    session = _build_session(n_sources=1, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    # Inject NaN into a channel
    source = session.list_sources()[0]
    ch = list(session.list_analog_channels(active_only=False))[0]

    # Monkey-patch build_aligned_data to return NaN values
    from app.sessions.session_models import AlignedChannelData

    def nan_build(sid, channel_name, t_start, t_end, max_points=4000):
        t = np.linspace(t_start, t_end, 50)
        v = np.full(50, np.nan)
        return AlignedChannelData(
            source_id=sid, channel_name=channel_name,
            time=t, values=v,
            original_sample_rate_hz=1000.0,
            time_offset_s=0.0, unit="kV",
        )

    orig = session.build_aligned_data
    session.build_aligned_data = nan_build
    ctrl.refresh_all(session)   # must not raise
    session.build_aligned_data = orig


# ---------------------------------------------------------------------------
# 9. Zero marker created for each active source
# ---------------------------------------------------------------------------


def test_zero_markers_created_for_active_sources(qapp) -> None:
    session = _build_session(n_sources=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    # Every canvas should have a zero line for each active source
    active_sids = {s.source_id for s in session.list_sources() if s.is_active}
    for canvas in ctrl._canvases.values():
        assert set(canvas._zero_lines.keys()) == active_sids


# ---------------------------------------------------------------------------
# 10. Zero marker removed when source is deactivated
# ---------------------------------------------------------------------------


def test_zero_marker_removed_when_source_deactivated(qapp) -> None:
    session = _build_session(n_sources=2)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    sources = session.list_sources()
    sid = sources[0].source_id
    session.set_source_active(sid, False)
    ctrl.refresh_all(session)

    for canvas in ctrl._canvases.values():
        assert sid not in canvas._zero_lines


# ---------------------------------------------------------------------------
# 11. Curve registry reuse: same PlotDataItem after second refresh
# ---------------------------------------------------------------------------


def test_curve_registry_reuse(qapp) -> None:
    session = _build_session(n_sources=1, n_analog=1)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    # Collect curve ids after first refresh
    ids_before: dict[tuple, int] = {}
    for canvas in ctrl._canvases.values():
        for key, curve in canvas._curves.items():
            ids_before[key] = id(curve)

    ctrl.refresh_all(session)  # second pass

    for canvas in ctrl._canvases.values():
        for key, curve in canvas._curves.items():
            assert id(curve) == ids_before[key], f"PlotDataItem recreated for {key}"


# ---------------------------------------------------------------------------
# 12. Source color assignment is stable
# ---------------------------------------------------------------------------


def test_source_color_stable(qapp) -> None:
    session = _build_session(n_sources=3)
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    colors_first = dict(ctrl._source_color_idx)

    ctrl.refresh_all(session)

    assert ctrl._source_color_idx == colors_first


# ---------------------------------------------------------------------------
# 13. SessionCanvasWidget.update_curve creates then reuses PlotDataItem
# ---------------------------------------------------------------------------


def test_session_canvas_widget_update_curve_reuse(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 100)
    v = np.sin(2 * np.pi * 50 * t)

    canvas.update_curve("s1", "VA", t, v, color="#ff0000", visible=True)
    item_id_first = id(canvas._curves[("s1", "VA")])

    canvas.update_curve("s1", "VA", t * 2, v * 0.5, color="#ff0000", visible=True)
    item_id_second = id(canvas._curves[("s1", "VA")])

    assert item_id_first == item_id_second
    canvas.clear_all()


# ---------------------------------------------------------------------------
# 14. SessionCanvasWidget.set_curve_visible toggles visibility
# ---------------------------------------------------------------------------


def test_session_canvas_widget_set_curve_visible(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 50)
    v = np.ones(50)

    canvas.update_curve("s1", "VA", t, v)
    curve = canvas._curves[("s1", "VA")]
    assert curve.isVisible()

    canvas.set_curve_visible("s1", "VA", False)
    assert not curve.isVisible()

    canvas.set_curve_visible("s1", "VA", True)
    assert curve.isVisible()

    canvas.clear_all()


# ---------------------------------------------------------------------------
# 15. SessionCanvasWidget.remove_source clears that source's curves + line
# ---------------------------------------------------------------------------


def test_session_canvas_widget_remove_source(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 50)
    v = np.ones(50)

    canvas.update_curve("s1", "VA0", t, v)
    canvas.update_curve("s1", "VA1", t, v)
    canvas.update_curve("s2", "VA0", t, v)
    canvas.update_zero_line("s1", "Source1", 0.0, "#1f77b4")

    canvas.remove_source("s1")

    assert ("s1", "VA0") not in canvas._curves
    assert ("s1", "VA1") not in canvas._curves
    assert ("s2", "VA0") in canvas._curves
    assert "s1" not in canvas._zero_lines

    canvas.clear_all()


# ---------------------------------------------------------------------------
# 16. SessionCanvasWidget.clear_all empties both registries
# ---------------------------------------------------------------------------


def test_session_canvas_widget_clear_all(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    t = np.linspace(0, 0.1, 50)

    canvas.update_curve("s1", "VA", t, np.ones(50))
    canvas.update_zero_line("s1", "S1", 0.0, "#ff0000")
    assert canvas.curve_count == 1
    assert canvas.zero_line_count == 1

    canvas.clear_all()
    assert canvas.curve_count == 0
    assert canvas.zero_line_count == 0


# ---------------------------------------------------------------------------
# 17. Common grid: all channels from same source share time length
# ---------------------------------------------------------------------------


def test_common_grid_same_source_same_length(qapp) -> None:
    session = _build_session(n_sources=1, n_analog=3)
    sources = session.list_sources()
    sid = sources[0].source_id
    t_start, t_end = -1.0, 1.0

    channels = [ch for ch in session.list_analog_channels(active_only=False) if ch.source_id == sid]
    results = [session.build_aligned_data(sid, ch.channel_name, t_start, t_end) for ch in channels]

    lengths = [len(r.time) for r in results]
    assert len(set(lengths)) == 1  # all same length


# ---------------------------------------------------------------------------
# 18. _session_window returns valid range from active sources
# ---------------------------------------------------------------------------


def test_session_window_valid_range(qapp) -> None:
    session = _build_session(n_sources=2)
    t_start, t_end = _session_window(session)
    assert t_start < t_end
    assert np.isfinite(t_start)
    assert np.isfinite(t_end)


def test_session_window_with_offset(qapp) -> None:
    session = _build_session(n_sources=1)
    sid = session.list_sources()[0].source_id

    t_start_before, t_end_before = _session_window(session)
    session.set_time_offset(sid, 5.0, method="manual")
    t_start_after, t_end_after = _session_window(session)

    assert t_start_after > t_start_before
    assert t_end_after > t_end_before


def test_session_window_fallback_when_no_active_sources(qapp) -> None:
    session = _build_session(n_sources=1)
    sid = session.list_sources()[0].source_id
    session.set_source_active(sid, False)

    t_start, t_end = _session_window(session)
    assert t_start == -1.0
    assert t_end == 1.0
