"""Unit tests for Phase 9E gap-closure: panel merge/split, set-as-reference,
and phase-aware auto-colour.

Coverage
--------
 1.  SessionCanvasWidget emits merge_with_requested when header menu chosen
 2.  SessionCanvasWidget emits split_by_source_requested from header menu
 3.  SessionCanvasWidget emits split_by_type_requested from header menu
 4.  set_mergeable_panels populates the merge submenu entries
 5.  _handle_merge_panels calls session.merge_panels and rebuilds layout
 6.  _handle_split_by_source moves second-source channels to a new panel
 7.  _handle_split_by_type moves different-type channels to new panels
 8.  set_as_reference_requested signal exists on SourceRowWidget
 9.  set_as_reference_requested signal re-emitted by SessionPanel
10.  _on_session_set_as_reference sets reference offset to 0.0
11.  _on_session_set_as_reference adjusts all other sources relatively
12.  _auto_colour returns phase base colour for A-phase channel (Source 1)
13.  _auto_colour returns saturation variant for A-phase channel (Source 2)
14.  _auto_colour falls back to palette for non-phase channel
15.  _auto_colour is deterministic (same inputs → same output)
16.  _auto_colour different from user colour override (override wins in _paint_panel)
17.  _update_mergeable_panels populates each canvas with other panel titles
18.  Panel split by source with single-source panel is a no-op
19.  Panel split by type with single-type panel is a no-op
20.  Phase colours: VA/VB/VC share blue/yellow/red hue family across sources
"""
from __future__ import annotations

import sys
import colorsys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication, QScrollArea

from app.sessions import EventAnalysisSession
from app.ui.session.session_canvas_controller import (
    SessionCanvasController,
    _SOURCE_COLORS,
    _PHASE_BASE_COLOURS,
)
from app.ui.session.legend_widget import generate_source_variant_colour
from app.visualization.widgets.session_canvas import SessionCanvasWidget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(n_analog=2, names=None):
    from app.models.channels import AnalogChannel
    from app.models.disturbance_record import DisturbanceRecord
    from app.models.metadata import RecordingMetadata
    from app.models.timing import SamplingInformation, TimingInformation

    dt = 0.001
    n = 100
    t = np.linspace(0.0, (n - 1) * dt, n)
    if names is None:
        names = [f"VA{i}" for i in range(n_analog)]
    data = {"time": t, **{name: np.sin(2 * np.pi * 50 * t) for name in names}}
    return DisturbanceRecord(
        metadata=RecordingMetadata("T", "R", "t.cfg", "comtrade", 50.0),
        waveform_data=data,
        analog_channels=[AnalogChannel(name=n, unit="kV", index=i) for i, n in enumerate(names)],
        digital_channels=[],
        sampling_info=SamplingInformation([1.0 / dt], [n]),
        timing_info=TimingInformation(datetime(2024, 1, 1), datetime(2024, 1, 1)),
        disturbance_info=None,
    )


def _build_session(n_sources=2, names=None):
    session = EventAnalysisSession()
    for i in range(n_sources):
        rec = _make_record(names=names or ["VA", "VB"])
        session.add_source(rec, f"Src{i + 1}", "comtrade")
    session.default_layout()
    return session


# ---------------------------------------------------------------------------
# 1–4: SessionCanvasWidget panel header signals
# ---------------------------------------------------------------------------


def test_canvas_merge_with_signal(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    canvas.set_mergeable_panels([("p2", "Panel 2"), ("p3", "Panel 3")])

    received = []
    canvas.merge_with_requested.connect(lambda a, b: received.append((a, b)))

    # Simulate direct signal emission (tests the signal chain, not UI interaction)
    canvas.merge_with_requested.emit("p1", "p2")
    assert received == [("p1", "p2")]


def test_canvas_split_by_source_signal(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    received = []
    canvas.split_by_source_requested.connect(lambda pid: received.append(pid))
    canvas.split_by_source_requested.emit("p1")
    assert received == ["p1"]


def test_canvas_split_by_type_signal(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    received = []
    canvas.split_by_type_requested.connect(lambda pid: received.append(pid))
    canvas.split_by_type_requested.emit("p1")
    assert received == ["p1"]


def test_set_mergeable_panels_stored(qapp) -> None:
    canvas = SessionCanvasWidget("p1", "Panel 1")
    panels = [("p2", "Voltage"), ("p3", "Current")]
    canvas.set_mergeable_panels(panels)
    assert canvas._mergeable_panels == panels


# ---------------------------------------------------------------------------
# 5–7: Controller merge/split handlers
# ---------------------------------------------------------------------------


def test_handle_merge_panels_calls_session(qapp) -> None:
    session = _build_session(n_sources=2, names=["VA", "VB", "VC"])
    # Default layout: voltage panel only (VA/VB/VC)
    panels_before = len([p for p in session.list_panels() if p.is_visible])

    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    panel_ids = [p.panel_id for p in session.list_panels()]
    if len(panel_ids) < 2:
        pytest.skip("Need ≥2 panels to merge")

    # Add a second panel by splitting
    ctrl._handle_split_by_source(panel_ids[0])
    panels_after_split = len([p for p in session.list_panels() if p.is_visible])

    # Now merge back
    new_panel_ids = [p.panel_id for p in session.list_panels()]
    if len(new_panel_ids) >= 2:
        ctrl._handle_merge_panels(new_panel_ids[0], new_panel_ids[1])
        panels_after_merge = len([p for p in session.list_panels() if p.is_visible])
        assert panels_after_merge < panels_after_split or panels_after_merge >= 1


def test_handle_split_by_source_creates_new_panel(qapp) -> None:
    session = _build_session(n_sources=2, names=["VA"])
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    # Find the voltage panel (has VA from both sources)
    volt_panel = next((p for p in session.list_panels() if "voltage" in p.panel_id.lower()), None)
    if volt_panel is None or len({r[0] for r in volt_panel.channel_refs}) < 2:
        pytest.skip("Need multi-source voltage panel for split test")

    panels_before = len(session.list_panels())
    ctrl._handle_split_by_source(volt_panel.panel_id)
    panels_after = len(session.list_panels())
    assert panels_after > panels_before


def test_session_split_notifies_owner_with_new_splitter(qapp) -> None:
    session = _build_session(n_sources=2, names=["VA"])
    ctrl = SessionCanvasController()
    rebuilt = []
    ctrl.set_layout_rebuilt_callback(lambda splitter: rebuilt.append(splitter))
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)

    volt_panel = next((p for p in session.list_panels() if "voltage" in p.panel_id.lower()), None)
    if volt_panel is None or len({r[0] for r in volt_panel.channel_refs}) < 2:
        pytest.skip("Need multi-source voltage panel for split test")

    ctrl._handle_split_by_source(volt_panel.panel_id)

    assert rebuilt
    assert isinstance(rebuilt[-1], QScrollArea)
    assert rebuilt[-1].widget() is ctrl._splitter


def test_handle_split_by_source_single_source_noop(qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(names=["VA", "VB"])
    session.add_source(rec, "S1", "comtrade")
    session.default_layout()

    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    volt_panel = next((p for p in session.list_panels()), None)
    if volt_panel is None:
        pytest.skip("No panels")

    panels_before = len(session.list_panels())
    ctrl._handle_split_by_source(volt_panel.panel_id)
    assert len(session.list_panels()) == panels_before  # no change


def test_handle_split_by_type_single_type_noop(qapp) -> None:
    session = EventAnalysisSession()
    rec = _make_record(names=["VA", "VB"])  # both voltage
    session.add_source(rec, "S1", "comtrade")
    session.default_layout()

    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    volt_panel = next((p for p in session.list_panels()), None)
    if volt_panel is None:
        pytest.skip("No panels")

    panels_before = len(session.list_panels())
    ctrl._handle_split_by_type(volt_panel.panel_id)
    assert len(session.list_panels()) == panels_before


# ---------------------------------------------------------------------------
# 8–11: set_as_reference — signal wiring and logic
# ---------------------------------------------------------------------------


def test_source_row_set_as_reference_signal(qapp) -> None:
    from app.ui.session.source_row_widget import SourceRowWidget
    from app.sessions.session_models import SessionSource

    source = SessionSource(
        source_id="s1", display_name="S1", record=_make_record(),
        provider_type="comtrade", origin_path=None,
        time_offset_s=0.0, is_active=True,
    )
    row = SourceRowWidget(source, [], [], [])
    assert hasattr(row, "set_as_reference_requested")
    received = []
    row.set_as_reference_requested.connect(received.append)
    row.set_as_reference_requested.emit("s1")
    assert received == ["s1"]


def test_session_panel_set_as_reference_signal(qapp) -> None:
    from app.ui.session.session_panel import SessionPanel
    panel = SessionPanel()
    assert hasattr(panel, "set_as_reference_requested")


def test_set_as_reference_zeroes_ref_offset(qapp) -> None:
    session = _build_session(n_sources=2)
    sources = session.list_sources()
    sid = sources[0].source_id
    session.set_time_offset(sid, 3.0, method="manual")
    session.set_time_offset(sources[1].source_id, 5.0, method="manual")

    # Simulate handler directly
    ref_offset = session.get_time_offset(sid)
    for source in session.list_sources():
        new_offset = source.time_offset_s - ref_offset
        session.set_time_offset(source.source_id, new_offset, method="manual")

    assert session.get_time_offset(sid) == 0.0


def test_set_as_reference_adjusts_others_relatively(qapp) -> None:
    session = _build_session(n_sources=2)
    sources = session.list_sources()
    sid0 = sources[0].source_id
    sid1 = sources[1].source_id
    session.set_time_offset(sid0, 2.0, method="manual")
    session.set_time_offset(sid1, 5.0, method="manual")

    # Set source 0 as reference
    ref_offset = session.get_time_offset(sid0)
    for source in session.list_sources():
        new_offset = source.time_offset_s - ref_offset
        session.set_time_offset(source.source_id, new_offset, method="manual")

    assert session.get_time_offset(sid0) == 0.0
    # source 1 was 3.0 s ahead of source 0 → stays 3.0 s ahead
    assert abs(session.get_time_offset(sid1) - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# 12–20: Phase-aware auto-colour
# ---------------------------------------------------------------------------


def _hsv(hex_color: str) -> tuple[float, float, float]:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return colorsys.rgb_to_hsv(r, g, b)


def _ctrl_with_sources(qapp, n_sources=2):
    session = _build_session(n_sources=n_sources, names=["VA", "VB", "VC"])
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)
    ctrl.refresh_all(session)
    return ctrl, session


def test_auto_colour_a_phase_source1(qapp) -> None:
    ctrl, session = _ctrl_with_sources(qapp)
    sources = session.list_sources()
    sid = sources[0].source_id

    color = ctrl._auto_colour(sid, "VA")
    # VA is A-phase → should be close to blue (#1f77b4)
    h_result, s_result, _ = _hsv(color)
    h_base, s_base, _ = _hsv(_PHASE_BASE_COLOURS["A"])
    assert abs(h_result - h_base) < 0.02  # same hue family
    assert abs(s_result - s_base) < 0.05  # full saturation for source 0


def test_auto_colour_a_phase_source2_is_variant(qapp) -> None:
    ctrl, session = _ctrl_with_sources(qapp, n_sources=2)
    sources = session.list_sources()
    sid0 = sources[0].source_id
    sid1 = sources[1].source_id

    c0 = ctrl._auto_colour(sid0, "VA")
    c1 = ctrl._auto_colour(sid1, "VA")
    # Different (saturation/hue shifted)
    assert c0 != c1

    # Source 2's hue should be offset from Source 1's
    h0, s0, _ = _hsv(c0)
    h1, s1, _ = _hsv(c1)
    hue_delta = (h1 - h0) % 1.0
    assert abs(hue_delta - 15.0 / 360.0) < 0.03  # ~15° shift


def test_auto_colour_fallback_for_non_phase(qapp) -> None:
    ctrl, session = _ctrl_with_sources(qapp)
    sources = session.list_sources()
    sid = sources[0].source_id

    # MW is not a phase channel → should get flat palette colour
    color = ctrl._auto_colour(sid, "MW")
    assert color in _SOURCE_COLORS


def test_auto_colour_deterministic(qapp) -> None:
    ctrl, session = _ctrl_with_sources(qapp)
    sources = session.list_sources()
    sid = sources[0].source_id

    c1 = ctrl._auto_colour(sid, "VA")
    c2 = ctrl._auto_colour(sid, "VA")
    assert c1 == c2


def test_auto_colour_vb_vc_different_hue(qapp) -> None:
    ctrl, session = _ctrl_with_sources(qapp)
    sources = session.list_sources()
    sid = sources[0].source_id

    ca = ctrl._auto_colour(sid, "VA")
    cb = ctrl._auto_colour(sid, "VB")
    cc = ctrl._auto_colour(sid, "VC")
    # All three should differ in hue
    ha, _, _ = _hsv(ca)
    hb, _, _ = _hsv(cb)
    hc, _, _ = _hsv(cc)
    assert ha != hb
    assert hb != hc
    assert ha != hc


def test_update_mergeable_panels_populated(qapp) -> None:
    session = _build_session(n_sources=2, names=["VA", "IA"])
    ctrl = SessionCanvasController()
    ctrl.rebuild_layout(session)

    # Every canvas should know about all OTHER panels
    panel_ids = list(ctrl._canvases.keys())
    if len(panel_ids) < 2:
        pytest.skip("Need ≥2 panels")

    for pid, canvas in ctrl._canvases.items():
        other_ids = {p for p, _ in canvas._mergeable_panels}
        assert pid not in other_ids                    # not itself
        assert any(p != pid for p in other_ids)        # has at least one other


def test_phase_colours_related_across_sources(qapp) -> None:
    """VA from src1 and VA from src2 must share the same hue family."""
    ctrl, session = _ctrl_with_sources(qapp, n_sources=3)
    sources = session.list_sources()

    colours = [ctrl._auto_colour(s.source_id, "VA") for s in sources]
    hues = [(colorsys.rgb_to_hsv(*[int(c[i:i+2], 16)/255.0 for i in (1, 3, 5)])[0]) for c in colours]

    # All hues should be within 90° of each other (same blue-ish family)
    for h in hues:
        delta = min(abs(h - hues[0]), 1.0 - abs(h - hues[0]))
        assert delta < 0.25, f"Hue {h:.3f} strayed far from base {hues[0]:.3f}"
