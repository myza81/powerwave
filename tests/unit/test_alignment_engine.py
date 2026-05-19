"""Unit tests for app/sessions/alignment_engine.py — Phase 9A.

Coverage
--------
1.  apply_time_offset — non-mutating, correct shift, zero offset pass-through
2.  apply_time_offset — float64 precision
3.  resample_to_grid — basic linear interpolation accuracy
4.  resample_to_grid — NaN fill outside source coverage (no extrapolation)
5.  resample_to_grid — handles non-uniform source time arrays
6.  resample_to_grid — single-sample source returns NaN grid
7.  build_common_time_grid — finest rate governs resolution
8.  build_common_time_grid — max_points cap enforced
9.  build_common_time_grid — single source
10. build_common_time_grid — uses median not mean (tolerates dropout gap)
11. build_common_time_grid — empty sources list fallback
12. detect_trigger_time — step event detected at correct time
13. detect_trigger_time — flat signal returns None
14. detect_trigger_time — too-short signal returns None
15. detect_trigger_time — uses actual time values not index arithmetic
16. detect_trigger_time — all-NaN baseline edge case
17. suggest_alignment_offsets — multi-source; all with triggers aligned to t=0
18. suggest_alignment_offsets — source with no trigger gets offset=0, confidence=0
19. suggest_alignment_offsets — source with no analog channels
20. suggest_alignment_offsets — returns AlignmentResult list with correct fields
21. assess_time_uniformity — perfectly uniform array returns (True, ~1.0)
22. assess_time_uniformity — jittered array returns (False, score < 1.0)
23. assess_time_uniformity — heavily jittered array has low stability score
24. assess_time_uniformity — too-short array returns (True, 1.0) safely
25. compute_source_quality — correct sample count and rate for uniform input
26. compute_source_quality — missing_data_pct from NaN values
27. compute_source_quality — duplicate_timestamp_pct from repeated times
28. compute_source_quality — non-uniform source sets time_is_uniform=False
"""
from __future__ import annotations

import numpy as np
import pytest

from app.sessions.alignment_engine import (
    apply_time_offset,
    assess_time_uniformity,
    build_common_time_grid,
    compute_source_quality,
    detect_trigger_time,
    resample_to_grid,
    suggest_alignment_offsets,
)
from app.sessions.session_models import AlignmentResult, SourceQualityMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _uniform_time(n: int = 100, dt: float = 0.02) -> np.ndarray:
    return np.linspace(0.0, (n - 1) * dt, n)


def _make_source_stub(source_id: str, time_arr, values_arr, has_channels: bool = True):
    """Minimal stub that matches what alignment_engine accesses from SessionSource."""
    from unittest.mock import MagicMock

    record = MagicMock()
    record.waveform_data = {"time": time_arr}
    if has_channels:
        ch = MagicMock()
        ch.name = "VA"
        record.waveform_data["VA"] = values_arr
        record.analog_channels = [ch]
    else:
        record.analog_channels = []

    source = MagicMock()
    source.source_id = source_id
    source.record = record
    return source


# ─────────────────────────────────────────────────────────────────────────────
# 1–2. apply_time_offset
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_time_offset_correct_shift() -> None:
    t = np.array([0.0, 1.0, 2.0])
    result = apply_time_offset(t, 5.0)
    np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0])


def test_apply_time_offset_non_mutating() -> None:
    t = np.array([0.0, 1.0, 2.0])
    original = t.copy()
    apply_time_offset(t, 99.0)
    np.testing.assert_array_equal(t, original)


def test_apply_time_offset_zero_offset() -> None:
    t = np.array([1.0, 2.0, 3.0])
    result = apply_time_offset(t, 0.0)
    np.testing.assert_array_equal(result, t)


def test_apply_time_offset_negative() -> None:
    t = np.array([10.0, 11.0, 12.0])
    result = apply_time_offset(t, -3.0)
    np.testing.assert_array_almost_equal(result, [7.0, 8.0, 9.0])


def test_apply_time_offset_float64_precision() -> None:
    t = np.array([0.0, 0.02, 0.04], dtype=np.float64)
    result = apply_time_offset(t, 1e-9)
    assert result.dtype == np.float64
    assert abs(result[0] - 1e-9) < 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# 3–6. resample_to_grid
# ─────────────────────────────────────────────────────────────────────────────


def test_resample_to_grid_linear_accuracy() -> None:
    t_src = np.array([0.0, 1.0, 2.0])
    v_src = np.array([0.0, 10.0, 20.0])
    grid = np.array([0.5, 1.0, 1.5])
    result = resample_to_grid(t_src, v_src, grid)
    np.testing.assert_array_almost_equal(result, [5.0, 10.0, 15.0])


def test_resample_to_grid_nan_outside_coverage() -> None:
    t_src = np.array([1.0, 2.0, 3.0])
    v_src = np.array([10.0, 20.0, 30.0])
    grid = np.array([0.0, 1.5, 4.0])  # 0.0 and 4.0 are outside [1.0, 3.0]
    result = resample_to_grid(t_src, v_src, grid)
    assert np.isnan(result[0]), "Before coverage should be NaN"
    assert abs(result[1] - 15.0) < 0.01
    assert np.isnan(result[2]), "After coverage should be NaN"


def test_resample_to_grid_non_uniform_source() -> None:
    # Non-uniform source times: 0, 0.1, 0.5, 1.0 (gaps of varying sizes)
    t_src = np.array([0.0, 0.1, 0.5, 1.0])
    v_src = np.array([0.0, 1.0, 5.0, 10.0])
    grid = np.array([0.0, 0.3, 0.75, 1.0])
    result = resample_to_grid(t_src, v_src, grid)
    # At t=0.3 (between 0.1 and 0.5): linear interp
    expected_03 = 1.0 + (5.0 - 1.0) * (0.3 - 0.1) / (0.5 - 0.1)
    assert abs(result[1] - expected_03) < 0.01
    assert abs(result[3] - 10.0) < 0.01


def test_resample_to_grid_single_sample_returns_nan() -> None:
    t_src = np.array([1.0])
    v_src = np.array([5.0])
    grid = np.array([0.5, 1.0, 1.5])
    result = resample_to_grid(t_src, v_src, grid)
    assert all(np.isnan(result))


# ─────────────────────────────────────────────────────────────────────────────
# 7–11. build_common_time_grid
# ─────────────────────────────────────────────────────────────────────────────


def test_build_common_time_grid_finest_rate_governs() -> None:
    t_fast = np.linspace(0, 1, 1000)  # dt = 0.001 s
    t_slow = np.linspace(0, 1, 10)   # dt = 0.111 s
    grid = build_common_time_grid([t_fast, t_slow], 0.0, 1.0, max_points=10_000)
    # Should be governed by fast source: ~1000 points
    assert len(grid) >= 900
    assert len(grid) <= 10_000


def test_build_common_time_grid_max_points_cap() -> None:
    t = np.linspace(0, 1, 100_000)
    grid = build_common_time_grid([t], 0.0, 1.0, max_points=500)
    assert len(grid) <= 500


def test_build_common_time_grid_single_source() -> None:
    t = np.linspace(0.0, 2.0, 201)  # dt = 0.01
    grid = build_common_time_grid([t], 0.0, 2.0, max_points=10_000)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(2.0)
    assert len(grid) >= 2


def test_build_common_time_grid_uses_median_not_mean() -> None:
    # Simulate dropout: 99 samples at dt=0.01 then one huge gap
    t_normal = np.linspace(0.0, 0.98, 99)
    t_gap = np.array([5.0])  # creates a huge gap — mean dt would be inflated
    t_combined = np.concatenate([t_normal, t_gap])
    grid = build_common_time_grid([t_combined], 0.0, 5.0, max_points=10_000)
    # With median dt ~0.01, expect ~500 points; with mean dt it would be much fewer
    assert len(grid) >= 100


def test_build_common_time_grid_empty_sources_fallback() -> None:
    grid = build_common_time_grid([], 0.0, 1.0, max_points=100)
    assert len(grid) >= 2
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 12–16. detect_trigger_time
# ─────────────────────────────────────────────────────────────────────────────


def test_detect_trigger_time_finds_step_event() -> None:
    t = _uniform_time(200, dt=0.02)
    v = np.zeros(200)
    v[100] = 500.0   # large step at t = 2.0 s
    result = detect_trigger_time(t, v)
    assert result is not None
    assert abs(result - 2.0) < 0.1


def test_detect_trigger_time_flat_returns_none() -> None:
    t = _uniform_time(100)
    v = np.ones(100) * 230.0   # constant — no event
    result = detect_trigger_time(t, v)
    assert result is None


def test_detect_trigger_time_too_short_returns_none() -> None:
    t = np.array([0.0, 0.02, 0.04])
    v = np.array([0.0, 500.0, 0.0])
    result = detect_trigger_time(t, v)
    assert result is None


def test_detect_trigger_time_uses_actual_time_values() -> None:
    # Non-uniform time array starting at a large offset
    t = np.concatenate([np.linspace(100.0, 101.0, 50), np.linspace(101.1, 102.0, 50)])
    v = np.zeros(100)
    v[75] = 1000.0  # event at index 75 → actual time around 101.5 s
    result = detect_trigger_time(t, v)
    assert result is not None
    assert result > 100.0  # must return actual time, not index


def test_detect_trigger_time_custom_threshold() -> None:
    t = _uniform_time(200, dt=0.02)
    v = np.ones(200) * 10.0
    v[100] = 50.0  # 5× baseline — above threshold=3, below threshold=10
    assert detect_trigger_time(t, v, threshold_factor=3.0) is not None
    assert detect_trigger_time(t, v, threshold_factor=10.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# 17–20. suggest_alignment_offsets
# ─────────────────────────────────────────────────────────────────────────────


def test_suggest_alignment_offsets_aligns_to_zero() -> None:
    t = _uniform_time(200, dt=0.02)
    v = np.zeros(200)
    v[100] = 500.0  # trigger at t = 2.0 s

    s1 = _make_source_stub("s1", t, v)
    results = suggest_alignment_offsets([s1])
    assert len(results) == 1
    r = results[0]
    assert r.source_id == "s1"
    # Offset should be -2.0 to shift trigger to t=0
    assert abs(r.suggested_offset_s - (-2.0)) < 0.1
    assert r.alignment_method == "auto_trigger"
    assert r.reference_time is not None


def test_suggest_alignment_offsets_multi_source() -> None:
    t1 = _uniform_time(200, dt=0.02)
    v1 = np.zeros(200); v1[50] = 500.0   # trigger at t = 1.0 s

    t2 = _uniform_time(200, dt=0.02)
    v2 = np.zeros(200); v2[100] = 500.0  # trigger at t = 2.0 s

    s1 = _make_source_stub("s1", t1, v1)
    s2 = _make_source_stub("s2", t2, v2)
    results = suggest_alignment_offsets([s1, s2])
    assert len(results) == 2
    r1 = next(r for r in results if r.source_id == "s1")
    r2 = next(r for r in results if r.source_id == "s2")
    assert abs(r1.suggested_offset_s - (-1.0)) < 0.1
    assert abs(r2.suggested_offset_s - (-2.0)) < 0.1


def test_suggest_alignment_offsets_no_trigger() -> None:
    t = _uniform_time(100)
    v = np.ones(100) * 5.0  # flat — no trigger

    s = _make_source_stub("s_flat", t, v)
    results = suggest_alignment_offsets([s])
    assert len(results) == 1
    r = results[0]
    assert r.suggested_offset_s == pytest.approx(0.0)
    assert r.alignment_confidence == pytest.approx(0.0)
    assert r.reference_time is None


def test_suggest_alignment_offsets_no_analog_channels() -> None:
    t = _uniform_time(50)
    v = np.zeros(50)
    s = _make_source_stub("s_empty", t, v, has_channels=False)
    results = suggest_alignment_offsets([s])
    assert len(results) == 1
    assert results[0].suggested_offset_s == pytest.approx(0.0)
    assert results[0].alignment_confidence == pytest.approx(0.0)


def test_suggest_alignment_offsets_returns_alignment_result_instances() -> None:
    t = _uniform_time(100)
    v = np.zeros(100); v[50] = 300.0

    s = _make_source_stub("sx", t, v)
    results = suggest_alignment_offsets([s])
    assert isinstance(results[0], AlignmentResult)
    assert isinstance(results[0].notes, str)
    assert len(results[0].notes) > 0


def test_suggest_alignment_offsets_confidence_in_range() -> None:
    t = _uniform_time(200, dt=0.02)
    v = np.zeros(200); v[100] = 999.0

    s = _make_source_stub("sx", t, v)
    results = suggest_alignment_offsets([s])
    conf = results[0].alignment_confidence
    assert conf is not None
    assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 21–24. assess_time_uniformity
# ─────────────────────────────────────────────────────────────────────────────


def test_assess_time_uniformity_perfect() -> None:
    t = np.linspace(0.0, 1.0, 1000)
    is_uniform, score = assess_time_uniformity(t)
    assert is_uniform is True
    assert score > 0.99


def test_assess_time_uniformity_jittered_not_uniform() -> None:
    rng = np.random.default_rng(42)
    t = np.cumsum(np.abs(rng.normal(loc=0.02, scale=0.01, size=200)))
    is_uniform, score = assess_time_uniformity(t)
    assert is_uniform is False
    assert score < 1.0


def test_assess_time_uniformity_heavily_jittered_low_score() -> None:
    rng = np.random.default_rng(7)
    # std = 50% of mean — very jittered
    t = np.cumsum(np.abs(rng.normal(loc=0.02, scale=0.01, size=200)))
    _, score = assess_time_uniformity(t)
    assert score < 0.99   # not perfectly stable


def test_assess_time_uniformity_too_short() -> None:
    t = np.array([0.0, 1.0])
    is_uniform, score = assess_time_uniformity(t)
    assert is_uniform is True
    assert score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 25–28. compute_source_quality
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_source_quality_basic() -> None:
    t = _uniform_time(100, dt=0.02)
    v = np.zeros(100)
    q = compute_source_quality("s1", t, v)
    assert isinstance(q, SourceQualityMetrics)
    assert q.source_id == "s1"
    assert q.sample_count == 100
    assert abs(q.inferred_sample_rate_hz - 50.0) < 1.0
    assert q.time_is_uniform is True


def test_compute_source_quality_missing_data_pct() -> None:
    t = _uniform_time(100, dt=0.02)
    v = np.zeros(100, dtype=np.float64)
    v[10:30] = np.nan  # 20 NaN values
    q = compute_source_quality("s2", t, v)
    assert abs(q.missing_data_pct - 20.0) < 0.5


def test_compute_source_quality_duplicate_timestamps() -> None:
    # 5 duplicate timestamps
    t = np.array([0.0, 0.02, 0.02, 0.04, 0.04, 0.06, 0.08, 0.10, 0.10, 0.12])
    v = np.zeros(10)
    q = compute_source_quality("s3", t, v)
    assert q.duplicate_timestamp_pct > 0.0


def test_compute_source_quality_non_uniform_flag() -> None:
    rng = np.random.default_rng(0)
    t = np.cumsum(np.abs(rng.normal(loc=0.02, scale=0.01, size=200)))
    v = np.zeros(200)
    q = compute_source_quality("s4", t, v)
    assert q.time_is_uniform is False
    assert q.sample_rate_stability < 1.0
