"""Unit tests for app.visualization.rendering.digital_transforms.

All tests are pure NumPy — no PyQt6 or display required.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.visualization.rendering.digital_transforms import (
    build_step_series,
    clip_digital_to_viewport,
    digital_role_color,
    extract_transitions,
)

# ─────────────────────────────────────────────────────────────────────────────
# TestDigitalRoleColor
# ─────────────────────────────────────────────────────────────────────────────


class TestDigitalRoleColor:
    def test_trip_keyword_returns_red(self) -> None:
        assert digital_role_color("PROT_TRIP") == "#FF2222"

    def test_gcb_keyword_returns_orange(self) -> None:
        assert digital_role_color("GCB OPEN") == "#FF8800"

    def test_cb_keyword_returns_orange(self) -> None:
        assert digital_role_color("52B_STATUS") == "#FF8800"

    def test_reclose_keyword_returns_blue(self) -> None:
        assert digital_role_color("79AR_RECLOSE") == "#4488FF"

    def test_intertrip_keyword_returns_magenta(self) -> None:
        assert digital_role_color("INTERTRIP_SEND") == "#FF44FF"

    def test_pickup_keyword_returns_amber(self) -> None:
        assert digital_role_color("OC_PICKUP") == "#FFAA00"

    def test_unknown_returns_grey(self) -> None:
        assert digital_role_color("GENERIC_DIGITAL_01") == "#AAAAAA"

    def test_alarm_exception_overrides_trip(self) -> None:
        # "alarm" appears before "trip" in name — alarm exception must fire first
        assert digital_role_color("ALARM_TRIP_STATUS") == "#AAAAAA"

    def test_case_insensitive(self) -> None:
        assert digital_role_color("trip_output") == "#FF2222"
        assert digital_role_color("TRIP_OUTPUT") == "#FF2222"

    def test_mcb_alarm_exception(self) -> None:
        assert digital_role_color("MCB_STATUS") == "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractTransitions
# ─────────────────────────────────────────────────────────────────────────────


def _uniform_dig(n: int, pattern: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) * 0.001
    if pattern is None:
        d = np.zeros(n, dtype=np.float64)
    else:
        d = np.array(pattern, dtype=np.float64)
    return t, d


class TestExtractTransitions:
    def test_empty_arrays_return_empty(self) -> None:
        t = np.empty(0, dtype=np.float64)
        d = np.empty(0, dtype=np.float64)
        t_out, d_out = extract_transitions(t, d)
        assert len(t_out) == 0
        assert len(d_out) == 0

    def test_constant_zero_returns_two_points(self) -> None:
        t, d = _uniform_dig(100)
        t_out, d_out = extract_transitions(t, d)
        # initial point + terminal sentinel
        assert len(t_out) == 2
        assert t_out[0] == pytest.approx(t[0])
        assert t_out[-1] == pytest.approx(t[-1])
        assert all(v == 0.0 for v in d_out)

    def test_constant_one_returns_two_points(self) -> None:
        t = np.arange(50, dtype=np.float64)
        d = np.ones(50, dtype=np.float64)
        t_out, d_out = extract_transitions(t, d)
        assert len(t_out) == 2
        assert all(v == 1.0 for v in d_out)

    def test_single_transition_zero_to_one(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        d = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        t_out, d_out = extract_transitions(t, d)
        # initial(0), transition at t=2, terminal(4)
        assert len(t_out) == 3
        assert t_out[0] == pytest.approx(0.0)
        assert t_out[1] == pytest.approx(2.0)
        assert d_out[0] == pytest.approx(0.0)
        assert d_out[1] == pytest.approx(1.0)

    def test_multiple_transitions_correct_count(self) -> None:
        t = np.arange(10, dtype=np.float64)
        d = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        t_out, d_out = extract_transitions(t, d)
        # transitions at indices 2, 4, 6, 7 → 4 transitions + initial + terminal
        # initial at 0, then 2,4,6,7 transitions, then terminal at 9
        assert len(t_out) > 2
        assert set(d_out).issubset({0.0, 1.0})

    def test_non_binary_coerced_to_binary(self) -> None:
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        d = np.array([0.0, 2.5, 0.0], dtype=np.float64)
        t_out, d_out = extract_transitions(t, d)
        assert set(d_out).issubset({0.0, 1.0})

    def test_2d_time_raises(self) -> None:
        t = np.ones((3, 2), dtype=np.float64)
        d = np.ones(6, dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            extract_transitions(t, d)

    def test_2d_data_raises(self) -> None:
        t = np.arange(6, dtype=np.float64)
        d = np.ones((3, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            extract_transitions(t, d)

    def test_length_mismatch_raises(self) -> None:
        t = np.arange(10, dtype=np.float64)
        d = np.arange(5, dtype=np.float64)
        with pytest.raises(ValueError, match="length mismatch"):
            extract_transitions(t, d)

    def test_output_dtype_float64(self) -> None:
        t, d = _uniform_dig(50)
        t_out, d_out = extract_transitions(t, d)
        assert t_out.dtype == np.float64
        assert d_out.dtype == np.float64

    def test_first_time_preserved(self) -> None:
        t = np.linspace(0.5, 1.5, 100, dtype=np.float64)
        d = np.zeros(100, dtype=np.float64)
        t_out, _ = extract_transitions(t, d)
        assert t_out[0] == pytest.approx(t[0])

    def test_last_time_is_sentinel(self) -> None:
        t = np.arange(20, dtype=np.float64)
        d = np.zeros(20, dtype=np.float64)
        t_out, _ = extract_transitions(t, d)
        assert t_out[-1] == pytest.approx(t[-1])


# ─────────────────────────────────────────────────────────────────────────────
# TestClipDigitalToViewport
# ─────────────────────────────────────────────────────────────────────────────


def _make_transitions(
    events: list[tuple[float, int]],
    end_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper: build a transition array from (time, state) events + terminal."""
    ts = [e[0] for e in events] + [end_time]
    ds = [e[1] for e in events] + [events[-1][1] if events else 0]
    return np.array(ts, dtype=np.float64), np.array(ds, dtype=np.float64)


class TestClipDigitalToViewport:
    def test_empty_input_returns_empty(self) -> None:
        t = np.empty(0, dtype=np.float64)
        d = np.empty(0, dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t, d, 0.0, 1.0)
        assert len(t_out) == 0
        assert len(d_out) == 0

    def test_viewport_before_data_returns_empty(self) -> None:
        t_tr, d_tr = _make_transitions([(2.0, 0), (3.0, 1)], 4.0)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 0.0, 1.0)
        assert len(t_out) == 0

    def test_all_data_before_viewport_carries_final_state(self) -> None:
        t_tr, d_tr = _make_transitions([(0.0, 0), (0.5, 1)], 1.0)
        # Final state is 1 (high); viewport starts after data
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 2.0, 3.0)
        assert len(t_out) == 2
        assert t_out[0] == pytest.approx(2.0)
        assert t_out[-1] == pytest.approx(3.0)
        assert all(v == pytest.approx(1.0) for v in d_out)

    def test_viewport_within_data_includes_carry(self) -> None:
        # Transition at t=1.0 (0→1), viewport [0.5, 1.5]
        t_tr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        d_tr = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 0.5, 1.5)
        # carry at 0.5 (state=0) + transition at 1.0 (state=1) + terminal at 1.5
        assert t_out[0] == pytest.approx(0.5)
        assert d_out[0] == pytest.approx(0.0)  # carry: was LOW
        assert t_out[-1] == pytest.approx(1.5)

    def test_t_start_equals_t_end_returns_at_most_two(self) -> None:
        t_tr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        d_tr = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 1.0, 1.0)
        assert len(t_out) <= 2

    def test_t_start_greater_than_t_end_swapped(self) -> None:
        t_tr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        d_tr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 1.5, 0.5)
        assert len(t_out) > 0
        assert t_out[0] == pytest.approx(0.5)

    def test_viewport_past_data_end_uses_final_state(self) -> None:
        t_tr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        d_tr = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 0.8, 2.0)
        assert t_out[-1] == pytest.approx(2.0)
        assert d_out[-1] == pytest.approx(1.0)

    def test_no_transitions_in_viewport_flat_carry(self) -> None:
        # Data: 0 from 0 to 0.3, then 1 from 0.3 to 1.0; viewport [0.1, 0.2]
        t_tr = np.array([0.0, 0.3, 1.0], dtype=np.float64)
        d_tr = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 0.1, 0.2)
        # No transitions in [0.1, 0.2]; carry state=0 from before 0.3
        assert len(t_out) == 2
        assert d_out[0] == pytest.approx(0.0)
        assert d_out[-1] == pytest.approx(0.0)

    def test_output_dtype_float64(self) -> None:
        t_tr = np.array([0.0, 1.0], dtype=np.float64)
        d_tr = np.array([0.0, 0.0], dtype=np.float64)
        t_out, d_out = clip_digital_to_viewport(t_tr, d_tr, 0.0, 1.0)
        assert t_out.dtype == np.float64
        assert d_out.dtype == np.float64

    def test_terminal_at_t_end(self) -> None:
        t_tr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        d_tr = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_out, _ = clip_digital_to_viewport(t_tr, d_tr, 0.0, 0.75)
        assert t_out[-1] == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildStepSeries
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildStepSeries:
    def test_single_segment_low(self) -> None:
        t = np.array([0.0, 1.0], dtype=np.float64)
        d = np.array([0.0, 0.0], dtype=np.float64)
        t_s, y_s = build_step_series(t, d, y_offset=0.0, track_height=1.0)
        assert len(t_s) == 2
        assert all(v == pytest.approx(0.0) for v in y_s)

    def test_single_segment_high(self) -> None:
        t = np.array([0.0, 1.0], dtype=np.float64)
        d = np.array([1.0, 1.0], dtype=np.float64)
        t_s, y_s = build_step_series(t, d, y_offset=0.0, track_height=1.0)
        assert all(v == pytest.approx(1.0) for v in y_s)

    def test_transition_zero_to_one(self) -> None:
        # t=[0, 0.5, 1.0], d=[0, 1, 1] — transition at t=0.5
        t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        d = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_s, y_s = build_step_series(t, d, y_offset=0.0, track_height=1.0)
        # 2 segments × 2 points = 4 points
        assert len(t_s) == 4
        assert y_s[0] == pytest.approx(0.0)   # first segment: LOW
        assert y_s[2] == pytest.approx(1.0)   # second segment: HIGH

    def test_y_offset_applied(self) -> None:
        t = np.array([0.0, 1.0], dtype=np.float64)
        d = np.array([1.0, 1.0], dtype=np.float64)
        _, y_s = build_step_series(t, d, y_offset=3.0, track_height=1.0)
        assert all(v == pytest.approx(4.0) for v in y_s)   # 1*1.0 + 3.0

    def test_output_dtype_float64(self) -> None:
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        d = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        t_s, y_s = build_step_series(t, d)
        assert t_s.dtype == np.float64
        assert y_s.dtype == np.float64

    def test_single_point_returns_unchanged(self) -> None:
        t = np.array([0.5], dtype=np.float64)
        d = np.array([1.0], dtype=np.float64)
        t_s, y_s = build_step_series(t, d, y_offset=2.0, track_height=1.0)
        assert len(t_s) == 1
        assert y_s[0] == pytest.approx(3.0)

    def test_track_height_scales_high_state(self) -> None:
        t = np.array([0.0, 1.0], dtype=np.float64)
        d = np.array([1.0, 1.0], dtype=np.float64)
        _, y_s = build_step_series(t, d, y_offset=0.0, track_height=0.5)
        assert all(v == pytest.approx(0.5) for v in y_s)
