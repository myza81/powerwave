"""Unit tests for app/sessions/event_session.py — Phase 9A.

Coverage
--------
1.  Session construction — session_version=1, created_at set, immutable
2.  add_source — returns UUID, source stored, channels populated
3.  add_source — two sources get different UUIDs
4.  remove_source — source and channels removed; panels cleaned
5.  remove_source — unknown id is a no-op
6.  list_sources — reflects current state
7.  set_source_active / list filtering
8.  set_time_offset — offset stored; method and confidence stored
9.  set_time_offset — invalid method raises ValueError
10. get_time_offset — returns 0.0 for unknown source
11. reset_all_offsets — all offsets back to 0.0; method reset to 'none'
12. get_global_time_range — single source, no offset
13. get_global_time_range — two overlapping sources: intersection returned
14. get_global_time_range — offset shifts the range
15. get_global_time_range — no active sources returns fallback
16. get_global_time_range — non-overlapping sources returns union fallback
17. channel registry — analog and digital channels registered on add_source
18. list_analog_channels — active_only filter
19. list_digital_channels — active_only filter
20. set_channel_display_name — stores override
21. set_channel_colour — stores hex override
22. set_channel_visibility — stores visibility
23. set_channel_panel — reassigns channel; updates panel channel_refs
24. default_layout — creates expected panel set; channels assigned correctly
25. default_layout — digital channels go to digital panel
26. add_panel / remove_panel
27. rename_panel
28. merge_panels — channel_refs combined; panel_b removed
29. merge_panels — channels reassigned to panel_a
30. merge_panels — unknown panel raises KeyError
31. build_aligned_data — offset applied non-destructively
32. build_aligned_data — clipping to [t_start, t_end]
33. build_aligned_data — empty result when no overlap with window
34. build_aligned_data — max_points cap triggers decimation
35. build_aligned_data — time_is_uniform flag propagated
36. build_aligned_data — unknown source_id raises KeyError
37. get_source_quality_metrics — returns SourceQualityMetrics for known source
38. get_source_quality_metrics — lazily cached (same object on second call)
39. get_source_quality_metrics — cache invalidated when source removed and re-added
40. get_source_quality_metrics — unknown source_id raises KeyError
41. build_aligned_data — two sources with different sampling rates stay independent
42. add_source with digital-only record registers digital channels
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.sessions import EventAnalysisSession, SourceQualityMetrics
from app.sessions.session_models import PanelConfig, SessionChannel, SessionSource
from app.models.disturbance_record import DisturbanceRecord
from app.models.channels import AnalogChannel, DigitalChannel
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    n_samples: int = 200,
    dt: float = 0.02,
    analog_names: list[str] | None = None,
    digital_names: list[str] | None = None,
    t_start: float = 0.0,
) -> DisturbanceRecord:
    """Build a minimal DisturbanceRecord for testing."""
    if analog_names is None:
        analog_names = ["VA", "VB", "IA"]
    if digital_names is None:
        digital_names = []

    time = np.linspace(t_start, t_start + (n_samples - 1) * dt, n_samples)
    waveform_data: dict = {"time": time}
    for name in analog_names:
        waveform_data[name] = np.random.default_rng(0).uniform(-1, 1, n_samples)
    for name in digital_names:
        waveform_data[name] = np.zeros(n_samples)

    analog_channels = [
        AnalogChannel(name=n, unit="kV" if "V" in n else "A", index=i)
        for i, n in enumerate(analog_names)
    ]
    digital_channels = [
        DigitalChannel(name=n, index=i)
        for i, n in enumerate(digital_names)
    ]

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST",
            recorder_name="TEST_RECORDER",
            source_file="test.cfg",
            provider_type="csv",
            nominal_frequency=50.0,
        ),
        waveform_data=waveform_data,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=SamplingInformation(
            sampling_rates=[1.0 / dt],
            samples_per_rate=[n_samples],
        ),
        timing_info=TimingInformation(start_time=datetime(2024, 1, 1), trigger_time=datetime(2024, 1, 1)),
        disturbance_info=None,
    )


def _session_with_two_sources() -> tuple[EventAnalysisSession, str, str]:
    sess = EventAnalysisSession()
    r1 = _make_record(n_samples=100, dt=0.02)  # 0–1.98 s, 50 Hz
    r2 = _make_record(n_samples=50, dt=1.0)    # 0–49 s, 1 Hz
    sid1 = sess.add_source(r1, "comtrade", "comtrade")
    sid2 = sess.add_source(r2, "csv", "csv")
    return sess, sid1, sid2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Session construction
# ─────────────────────────────────────────────────────────────────────────────


def test_session_version_is_one() -> None:
    sess = EventAnalysisSession()
    assert sess.session_version == 1


def test_session_created_at_is_utc_datetime() -> None:
    sess = EventAnalysisSession()
    assert isinstance(sess.created_at, datetime)
    assert sess.created_at.tzinfo is not None
    assert sess.created_at.tzinfo == timezone.utc


def test_session_created_at_immutable_by_convention() -> None:
    sess = EventAnalysisSession()
    original = sess.created_at
    # Should not be mutated by normal operations
    r = _make_record()
    sess.add_source(r, "test", "csv")
    assert sess.created_at == original


# ─────────────────────────────────────────────────────────────────────────────
# 2–6. Source add / remove / list
# ─────────────────────────────────────────────────────────────────────────────


def test_add_source_returns_string_uuid() -> None:
    sess = EventAnalysisSession()
    r = _make_record()
    sid = sess.add_source(r, "test", "csv")
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID format


def test_add_source_stored_in_session() -> None:
    sess = EventAnalysisSession()
    r = _make_record()
    sid = sess.add_source(r, "test_source", "comtrade", "/data/test.cfg")
    source = sess.get_source(sid)
    assert source is not None
    assert source.source_id == sid
    assert source.display_name == "test_source"
    assert source.provider_type == "comtrade"
    assert source.origin_path == "/data/test.cfg"
    assert source.is_active is True
    assert source.time_offset_s == pytest.approx(0.0)
    assert source.alignment_method == "none"
    assert source.alignment_confidence is None


def test_add_source_two_sources_different_uuids() -> None:
    sess = EventAnalysisSession()
    sid1 = sess.add_source(_make_record(), "a", "csv")
    sid2 = sess.add_source(_make_record(), "b", "csv")
    assert sid1 != sid2


def test_remove_source_removes_source() -> None:
    sess = EventAnalysisSession()
    sid = sess.add_source(_make_record(), "x", "csv")
    sess.remove_source(sid)
    assert sess.get_source(sid) is None
    assert len(sess.list_sources()) == 0


def test_remove_source_removes_channels() -> None:
    sess = EventAnalysisSession()
    sid = sess.add_source(_make_record(analog_names=["VA", "IA"]), "x", "csv")
    sess.remove_source(sid)
    assert sess.get_channel(sid, "VA") is None
    assert sess.get_channel(sid, "IA") is None


def test_remove_source_unknown_is_noop() -> None:
    sess = EventAnalysisSession()
    sess.remove_source("nonexistent-uuid")   # must not raise


def test_list_sources_reflects_state() -> None:
    sess, sid1, sid2 = _session_with_two_sources()
    ids = {s.source_id for s in sess.list_sources()}
    assert ids == {sid1, sid2}


# ─────────────────────────────────────────────────────────────────────────────
# 7. set_source_active
# ─────────────────────────────────────────────────────────────────────────────


def test_set_source_active_false() -> None:
    sess, sid1, sid2 = _session_with_two_sources()
    sess.set_source_active(sid1, False)
    assert sess.get_source(sid1).is_active is False
    analogs = sess.list_analog_channels(active_only=True)
    active_ids = {ch.source_id for ch in analogs}
    assert sid1 not in active_ids
    assert sid2 in active_ids


def test_list_analog_channels_active_only_false() -> None:
    sess, sid1, sid2 = _session_with_two_sources()
    sess.set_source_active(sid1, False)
    all_ch = sess.list_analog_channels(active_only=False)
    ids = {ch.source_id for ch in all_ch}
    assert sid1 in ids


# ─────────────────────────────────────────────────────────────────────────────
# 8–11. Time offset
# ─────────────────────────────────────────────────────────────────────────────


def test_set_time_offset_stored_with_metadata() -> None:
    sess, sid1, _ = _session_with_two_sources()
    sess.set_time_offset(sid1, -2.5, method="manual", confidence=None)
    assert sess.get_time_offset(sid1) == pytest.approx(-2.5)
    assert sess.get_source(sid1).alignment_method == "manual"
    assert sess.get_source(sid1).alignment_confidence is None


def test_set_time_offset_auto_trigger_method() -> None:
    sess, sid1, _ = _session_with_two_sources()
    sess.set_time_offset(sid1, -1.0, method="auto_trigger", confidence=0.87)
    src = sess.get_source(sid1)
    assert src.alignment_method == "auto_trigger"
    assert src.alignment_confidence == pytest.approx(0.87)


def test_set_time_offset_invalid_method_raises() -> None:
    sess, sid1, _ = _session_with_two_sources()
    with pytest.raises(ValueError, match="Unknown alignment_method"):
        sess.set_time_offset(sid1, 0.0, method="bad_method")


def test_get_time_offset_unknown_source_returns_zero() -> None:
    sess = EventAnalysisSession()
    assert sess.get_time_offset("no-such-id") == pytest.approx(0.0)


def test_reset_all_offsets() -> None:
    sess, sid1, sid2 = _session_with_two_sources()
    sess.set_time_offset(sid1, 5.0, method="manual")
    sess.set_time_offset(sid2, -3.0, method="auto_trigger", confidence=0.9)
    sess.reset_all_offsets()
    assert sess.get_time_offset(sid1) == pytest.approx(0.0)
    assert sess.get_time_offset(sid2) == pytest.approx(0.0)
    assert sess.get_source(sid1).alignment_method == "none"
    assert sess.get_source(sid2).alignment_method == "none"
    assert sess.get_source(sid1).alignment_confidence is None
    assert sess.get_source(sid2).alignment_confidence is None


# ─────────────────────────────────────────────────────────────────────────────
# 12–16. Global time range
# ─────────────────────────────────────────────────────────────────────────────


def test_global_time_range_single_source() -> None:
    sess = EventAnalysisSession()
    r = _make_record(n_samples=100, dt=0.02, t_start=0.0)
    sid = sess.add_source(r, "x", "csv")
    lo, hi = sess.get_global_time_range()
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi == pytest.approx(1.98, abs=0.05)


def test_global_time_range_intersection() -> None:
    sess = EventAnalysisSession()
    # Source 1: 0–2 s
    r1 = _make_record(n_samples=101, dt=0.02, t_start=0.0)
    # Source 2: 1–5 s
    r2 = _make_record(n_samples=41, dt=0.1, t_start=1.0)
    sess.add_source(r1, "a", "comtrade")
    sess.add_source(r2, "b", "csv")
    lo, hi = sess.get_global_time_range()
    assert lo == pytest.approx(1.0, abs=0.05)
    assert hi <= 2.0 + 0.05  # must be at most end of r1


def test_global_time_range_offset_shifts() -> None:
    sess = EventAnalysisSession()
    r = _make_record(n_samples=100, dt=0.02, t_start=0.0)
    sid = sess.add_source(r, "x", "csv")
    sess.set_time_offset(sid, 10.0)
    lo, hi = sess.get_global_time_range()
    assert lo == pytest.approx(10.0, abs=0.05)


def test_global_time_range_no_active_sources_fallback() -> None:
    sess = EventAnalysisSession()
    lo, hi = sess.get_global_time_range()
    assert hi > lo  # fallback must be a valid range


def test_global_time_range_non_overlapping_returns_union() -> None:
    sess = EventAnalysisSession()
    r1 = _make_record(n_samples=50, dt=0.02, t_start=0.0)  # 0–0.98 s
    r2 = _make_record(n_samples=50, dt=0.02, t_start=5.0)  # 5–5.98 s
    sess.add_source(r1, "a", "csv")
    sess.add_source(r2, "b", "csv")
    lo, hi = sess.get_global_time_range()
    # Non-overlapping → fallback to union
    assert lo <= 0.1
    assert hi >= 5.9


# ─────────────────────────────────────────────────────────────────────────────
# 17–23. Channel registry
# ─────────────────────────────────────────────────────────────────────────────


def test_channels_registered_on_add_source() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA", "IA"], digital_names=["CB1"])
    sid = sess.add_source(r, "x", "comtrade")
    assert sess.get_channel(sid, "VA") is not None
    assert sess.get_channel(sid, "IA") is not None
    assert sess.get_channel(sid, "CB1") is not None


def test_analog_digital_channel_types() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], digital_names=["CB1"])
    sid = sess.add_source(r, "x", "comtrade")
    assert sess.get_channel(sid, "VA").channel_type == "analog"
    assert sess.get_channel(sid, "CB1").channel_type == "digital"


def test_list_digital_channels() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], digital_names=["CB1", "CB2"])
    sid = sess.add_source(r, "x", "comtrade")
    digi = sess.list_digital_channels()
    assert len(digi) == 2


def test_set_channel_display_name() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    sess.set_channel_display_name(sid, "VA", "Phase A Voltage")
    assert sess.get_channel(sid, "VA").display_name == "Phase A Voltage"


def test_set_channel_colour() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    sess.set_channel_colour(sid, "VA", "#FF0000")
    assert sess.get_channel(sid, "VA").color_hex == "#FF0000"


def test_set_channel_visibility() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    sess.set_channel_visibility(sid, "VA", False)
    assert sess.get_channel(sid, "VA").is_visible is False


def test_set_channel_panel() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    sess.default_layout()
    new_panel_id = sess.add_panel("Custom", "analog")
    sess.set_channel_panel(sid, "VA", new_panel_id)
    assert sess.get_channel(sid, "VA").panel_id == new_panel_id
    new_panel = sess.get_panel(new_panel_id) if hasattr(sess, "get_panel") else \
        next((p for p in sess.list_panels() if p.panel_id == new_panel_id), None)
    if new_panel is not None:
        assert (sid, "VA") in new_panel.channel_refs


# ─────────────────────────────────────────────────────────────────────────────
# 24–25. Default layout
# ─────────────────────────────────────────────────────────────────────────────


def test_default_layout_creates_panels_for_channel_types() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA", "VB", "IA"], digital_names=["CB1"])
    sess.add_source(r, "x", "comtrade")
    sess.default_layout()
    panels = {p.panel_id: p for p in sess.list_panels()}
    assert "voltage" in panels
    assert "current" in panels
    assert "digital" in panels


def test_default_layout_digital_to_digital_panel() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], digital_names=["CB1", "CB2"])
    sid = sess.add_source(r, "x", "comtrade")
    sess.default_layout()
    assert sess.get_channel(sid, "CB1").panel_id == "digital"
    assert sess.get_channel(sid, "CB2").panel_id == "digital"


def test_default_layout_voltage_channels_to_voltage_panel() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA", "VB"])
    sid = sess.add_source(r, "x", "csv")
    sess.default_layout()
    assert sess.get_channel(sid, "VA").panel_id == "voltage"
    assert sess.get_channel(sid, "VB").panel_id == "voltage"


def test_default_layout_two_sources_same_panel() -> None:
    sess = EventAnalysisSession()
    r1 = _make_record(analog_names=["VA"])
    r2 = _make_record(analog_names=["VA"])
    sid1 = sess.add_source(r1, "a", "comtrade")
    sid2 = sess.add_source(r2, "b", "csv")
    sess.default_layout()
    v_panel = next(p for p in sess.list_panels() if p.panel_id == "voltage")
    refs = v_panel.channel_refs
    assert (sid1, "VA") in refs
    assert (sid2, "VA") in refs


# ─────────────────────────────────────────────────────────────────────────────
# 26–30. Panels
# ─────────────────────────────────────────────────────────────────────────────


def test_add_remove_panel() -> None:
    sess = EventAnalysisSession()
    pid = sess.add_panel("Custom Panel", "analog")
    assert any(p.panel_id == pid for p in sess.list_panels())
    sess.remove_panel(pid)
    assert not any(p.panel_id == pid for p in sess.list_panels())


def test_rename_panel() -> None:
    sess = EventAnalysisSession()
    pid = sess.add_panel("Old Name", "analog")
    sess.rename_panel(pid, "New Name")
    panel = next(p for p in sess.list_panels() if p.panel_id == pid)
    assert panel.title == "New Name"


def test_merge_panels_combines_channel_refs() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA", "IA"])
    sid = sess.add_source(r, "x", "csv")
    pid_a = sess.add_panel("A", "analog")
    pid_b = sess.add_panel("B", "analog")
    # Manually place channels in panels
    panels = {p.panel_id: p for p in sess.list_panels()}
    panels[pid_a].channel_refs = [(sid, "VA")]
    panels[pid_b].channel_refs = [(sid, "IA")]
    sess.get_channel(sid, "VA").panel_id = pid_a
    sess.get_channel(sid, "IA").panel_id = pid_b

    merged_id = sess.merge_panels(pid_a, pid_b)
    assert merged_id == pid_a
    merged = next(p for p in sess.list_panels() if p.panel_id == pid_a)
    assert (sid, "VA") in merged.channel_refs
    assert (sid, "IA") in merged.channel_refs
    assert not any(p.panel_id == pid_b for p in sess.list_panels())


def test_merge_panels_channels_reassigned_to_a() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA", "IA"])
    sid = sess.add_source(r, "x", "csv")
    pid_a = sess.add_panel("A", "analog")
    pid_b = sess.add_panel("B", "analog")
    panels = {p.panel_id: p for p in sess.list_panels()}
    panels[pid_b].channel_refs = [(sid, "IA")]
    sess.get_channel(sid, "IA").panel_id = pid_b

    sess.merge_panels(pid_a, pid_b)
    assert sess.get_channel(sid, "IA").panel_id == pid_a


def test_merge_panels_unknown_panel_raises() -> None:
    sess = EventAnalysisSession()
    pid = sess.add_panel("A", "analog")
    with pytest.raises(KeyError):
        sess.merge_panels(pid, "nonexistent-uuid")


# ─────────────────────────────────────────────────────────────────────────────
# 31–41. build_aligned_data
# ─────────────────────────────────────────────────────────────────────────────


def test_build_aligned_data_offset_non_destructive() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=100, dt=0.02)
    original_time = r.waveform_data["time"].copy()
    sid = sess.add_source(r, "x", "csv")
    sess.set_time_offset(sid, 5.0)
    result = sess.build_aligned_data(sid, "VA", 5.0, 7.0)
    # Original record must not be mutated
    np.testing.assert_array_equal(r.waveform_data["time"], original_time)
    # Output time should be shifted
    if len(result.time) > 0:
        assert result.time[0] >= 5.0 - 0.01


def test_build_aligned_data_offset_shifts_output() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=100, dt=0.02, t_start=0.0)
    sid = sess.add_source(r, "x", "csv")
    sess.set_time_offset(sid, 10.0)
    # Window around the shifted range
    result = sess.build_aligned_data(sid, "VA", 10.0, 12.0)
    assert len(result.time) > 0
    assert result.time_offset_s == pytest.approx(10.0)
    assert result.time[0] >= 10.0 - 0.01


def test_build_aligned_data_clips_to_window() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=500, dt=0.02, t_start=0.0)
    sid = sess.add_source(r, "x", "csv")
    result = sess.build_aligned_data(sid, "VA", 1.0, 3.0)
    assert len(result.time) > 0
    assert result.time[0] >= 1.0 - 0.01
    assert result.time[-1] <= 3.0 + 0.01


def test_build_aligned_data_empty_when_no_overlap() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=100, dt=0.02, t_start=0.0)
    sid = sess.add_source(r, "x", "csv")
    # Request window way outside source range
    result = sess.build_aligned_data(sid, "VA", 100.0, 200.0)
    assert len(result.time) == 0
    assert len(result.values) == 0


def test_build_aligned_data_max_points_cap() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=10_000, dt=0.0001, t_start=0.0)
    sid = sess.add_source(r, "x", "comtrade")
    result = sess.build_aligned_data(sid, "VA", 0.0, 1.0, max_points=200)
    assert len(result.time) <= 200


def test_build_aligned_data_time_is_uniform_flag() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=200, dt=0.02)
    sid = sess.add_source(r, "x", "comtrade")
    result = sess.build_aligned_data(sid, "VA", 0.0, 4.0)
    assert result.time_is_uniform is True


def test_build_aligned_data_unknown_source_raises() -> None:
    sess = EventAnalysisSession()
    with pytest.raises(KeyError):
        sess.build_aligned_data("no-such-id", "VA", 0.0, 1.0)


def test_build_aligned_data_two_sources_independent() -> None:
    sess = EventAnalysisSession()
    r_fast = _make_record(analog_names=["VA"], n_samples=500, dt=0.02, t_start=0.0)
    r_slow = _make_record(analog_names=["VA"], n_samples=10, dt=1.0, t_start=0.0)
    sid_fast = sess.add_source(r_fast, "comtrade", "comtrade")
    sid_slow = sess.add_source(r_slow, "csv", "csv")
    r_fast_out = sess.build_aligned_data(sid_fast, "VA", 0.0, 2.0)
    r_slow_out = sess.build_aligned_data(sid_slow, "VA", 0.0, 9.0)
    # Fast source has more points in its window
    assert len(r_fast_out.time) > len(r_slow_out.time)
    # Original sample rates differ
    assert r_fast_out.original_sample_rate_hz > r_slow_out.original_sample_rate_hz


# ─────────────────────────────────────────────────────────────────────────────
# 37–40. Source quality metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_get_source_quality_metrics_returns_instance() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"], n_samples=100, dt=0.02)
    sid = sess.add_source(r, "x", "csv")
    q = sess.get_source_quality_metrics(sid)
    assert isinstance(q, SourceQualityMetrics)
    assert q.source_id == sid
    assert q.sample_count == 100


def test_get_source_quality_metrics_cached() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    q1 = sess.get_source_quality_metrics(sid)
    q2 = sess.get_source_quality_metrics(sid)
    assert q1 is q2  # same object — cached


def test_get_source_quality_metrics_cache_cleared_on_remove() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=["VA"])
    sid = sess.add_source(r, "x", "csv")
    _ = sess.get_source_quality_metrics(sid)
    sess.remove_source(sid)
    # Re-add with a different record
    r2 = _make_record(analog_names=["VA"], n_samples=50)
    sid2 = sess.add_source(r2, "y", "csv")
    q = sess.get_source_quality_metrics(sid2)
    assert q.sample_count == 50


def test_get_source_quality_metrics_unknown_raises() -> None:
    sess = EventAnalysisSession()
    with pytest.raises(KeyError):
        sess.get_source_quality_metrics("no-such-id")


# ─────────────────────────────────────────────────────────────────────────────
# 42. Digital-only record
# ─────────────────────────────────────────────────────────────────────────────


def test_add_source_digital_only_record() -> None:
    sess = EventAnalysisSession()
    r = _make_record(analog_names=[], digital_names=["CB1", "CB2", "CB3"])
    sid = sess.add_source(r, "relay_digi", "comtrade")
    digi = sess.list_digital_channels()
    assert len(digi) == 3
    analog = sess.list_analog_channels()
    assert len(analog) == 0
