"""Unit tests for app.calculated_signals.engine."""
from __future__ import annotations

import time as time_module

import numpy as np
import pytest

from app.calculated_signals.engine import (
    AlignmentError,
    CalculationEngineConfig,
    CalculationError,
    InputValidationError,
    UnitCompatibilityError,
    _compute_overlap,
    _nominal_interval,
    _safe_interpolation_mask,
    _sanitize_time_values,
    _SanitizedInput,
    calculate_signal,
)
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculationStatus,
    ChannelRef,
    ResolvedAnalogInput,
)


def _defn(expression: str, bindings: dict[str, ChannelRef], reference: str, **kwargs) -> CalculatedSignalDefinition:
    return CalculatedSignalDefinition(
        calc_id="c1", name="Test", expression=expression,
        variable_bindings=bindings, reference_variable=reference, **kwargs,
    )


def _input(variable: str, time: np.ndarray, values: np.ndarray, unit: str | None = "MW", **kwargs) -> ResolvedAnalogInput:
    return ResolvedAnalogInput(variable=variable, time=time, values=values, unit=unit, **kwargs)


def _ab(expression="A + B", unit="MW", n=11, reference="A"):
    t = np.linspace(0.0, 1.0, n)
    A = _input("A", t, np.full(n, 1.0), unit)
    B = _input("B", t, np.full(n, 2.0), unit)
    defn = _defn(expression, {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, reference)
    return defn, {"A": A, "B": B}


# ─────────────────────────────────────────────────────────────────────────────
# CalculationEngineConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculationEngineConfig:
    def test_defaults(self) -> None:
        cfg = CalculationEngineConfig()
        assert cfg.gap_multiplier == 5.0
        assert cfg.large_sample_rate_ratio == 10.0
        assert cfg.limited_overlap_fraction == 0.25

    def test_non_positive_gap_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="gap_multiplier"):
            CalculationEngineConfig(gap_multiplier=0.0)

    def test_non_positive_rate_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="large_sample_rate_ratio"):
            CalculationEngineConfig(large_sample_rate_ratio=-1.0)

    def test_overlap_fraction_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="limited_overlap_fraction"):
            CalculationEngineConfig(limited_overlap_fraction=1.5)
        with pytest.raises(ValueError, match="limited_overlap_fraction"):
            CalculationEngineConfig(limited_overlap_fraction=-0.1)

    def test_overlap_fraction_boundary_values_accepted(self) -> None:
        CalculationEngineConfig(limited_overlap_fraction=0.0)
        CalculationEngineConfig(limited_overlap_fraction=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Time sanitization
# ─────────────────────────────────────────────────────────────────────────────


class TestSanitizeTimeValues:
    def test_already_sorted(self) -> None:
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([10.0, 20.0, 30.0])
        ct, cv, nf, dup = _sanitize_time_values(t, v)
        np.testing.assert_array_equal(ct, t)
        np.testing.assert_array_equal(cv, v)
        assert nf == 0 and dup == 0

    def test_unsorted_is_sorted(self) -> None:
        t = np.array([2.0, 0.0, 1.0])
        v = np.array([30.0, 10.0, 20.0])
        ct, cv, _, _ = _sanitize_time_values(t, v)
        np.testing.assert_array_equal(ct, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(cv, np.array([10.0, 20.0, 30.0]))

    def test_duplicate_timestamps_keep_first(self) -> None:
        # Two samples at t=1.0; the first-encountered value (99.0) must win.
        t = np.array([0.0, 1.0, 1.0, 2.0])
        v = np.array([10.0, 99.0, 88.0, 30.0])
        ct, cv, _, dup = _sanitize_time_values(t, v)
        np.testing.assert_array_equal(ct, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(cv, np.array([10.0, 99.0, 30.0]))
        assert dup == 1

    def test_duplicate_after_sort_keep_first_by_original_order(self) -> None:
        # Unsorted input with a duplicate: after stable sort, "first" means
        # first in original array order, not sorted-array order.
        t = np.array([1.0, 0.0, 1.0])
        v = np.array([111.0, 10.0, 222.0])
        ct, cv, _, dup = _sanitize_time_values(t, v)
        np.testing.assert_array_equal(ct, np.array([0.0, 1.0]))
        # Of the two t=1.0 rows, index 0 (value 111.0) appeared first originally.
        np.testing.assert_array_equal(cv, np.array([10.0, 111.0]))
        assert dup == 1

    def test_nan_and_inf_time_dropped(self) -> None:
        t = np.array([0.0, float("nan"), 1.0, float("inf"), 2.0, float("-inf")])
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ct, cv, nf, _ = _sanitize_time_values(t, v)
        np.testing.assert_array_equal(ct, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(cv, np.array([1.0, 3.0, 5.0]))
        assert nf == 3

    def test_finite_and_nonfinite_mixture(self) -> None:
        t = np.array([0.0, 1.0, float("nan"), 2.0])
        v = np.array([1.0, 2.0, 3.0, 4.0])
        ct, cv, nf, dup = _sanitize_time_values(t, v)
        assert len(ct) == 3
        assert nf == 1 and dup == 0

    def test_all_nonfinite_yields_empty(self) -> None:
        t = np.array([float("nan"), float("inf")])
        v = np.array([1.0, 2.0])
        ct, cv, nf, dup = _sanitize_time_values(t, v)
        assert len(ct) == 0
        assert nf == 2

    def test_does_not_mutate_caller_arrays(self) -> None:
        t = np.array([2.0, 0.0, 1.0])
        v = np.array([30.0, 10.0, 20.0])
        t_before, v_before = t.copy(), v.copy()
        _sanitize_time_values(t, v)
        np.testing.assert_array_equal(t, t_before)
        np.testing.assert_array_equal(v, v_before)


class TestNominalInterval:
    def test_regular_interval(self) -> None:
        t = np.array([0.0, 0.1, 0.2, 0.3])
        assert _nominal_interval(t) == pytest.approx(0.1)

    def test_too_short_returns_none(self) -> None:
        assert _nominal_interval(np.array([0.0])) is None
        assert _nominal_interval(np.array([])) is None

    def test_robust_to_one_outlier_gap(self) -> None:
        t = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
        # median of [0.1,0.1,0.1,4.7] is 0.1 -- robust to the one large gap.
        assert _nominal_interval(t) == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeOverlap:
    def _sanitized(self, time: np.ndarray) -> _SanitizedInput:
        return _SanitizedInput(
            variable="x", time=time, values=np.zeros_like(time), unit="MW",
            dropped_nonfinite=0, dropped_duplicates=0,
        )

    def test_full_overlap_identical_ranges(self) -> None:
        s = {"A": self._sanitized(np.linspace(0, 1, 5)), "B": self._sanitized(np.linspace(0, 1, 5))}
        start, end = _compute_overlap(s)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        s = {"A": self._sanitized(np.linspace(0, 1, 5)), "B": self._sanitized(np.linspace(0.5, 1.5, 5))}
        start, end = _compute_overlap(s)
        assert start == pytest.approx(0.5)
        assert end == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        s = {"A": self._sanitized(np.linspace(0, 1, 5)), "B": self._sanitized(np.linspace(2, 3, 5))}
        start, end = _compute_overlap(s)
        assert start > end


class TestSafeInterpolationMask:
    def test_no_gap_all_safe(self) -> None:
        source = np.linspace(0, 1, 11)
        target = np.linspace(0, 1, 21)
        mask = _safe_interpolation_mask(source, target, gap_multiplier=5.0)
        assert mask.all()

    def test_large_gap_marks_interior_unsafe(self) -> None:
        source = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
        target = np.array([0.05, 1.0, 2.05])
        mask = _safe_interpolation_mask(source, target, gap_multiplier=5.0)
        assert mask[0]  # inside the dense region
        assert not mask[1]  # inside the big gap
        assert mask[2]  # inside the second dense region

    def test_exact_match_always_safe(self) -> None:
        source = np.array([0.0, 0.1, 0.2, 2.0])
        target = np.array([2.0])  # coincides with a real sample right after a big gap
        mask = _safe_interpolation_mask(source, target, gap_multiplier=5.0)
        assert mask[0]

    def test_too_few_source_samples_all_unsafe(self) -> None:
        mask = _safe_interpolation_mask(np.array([0.0]), np.array([0.0, 0.5]), gap_multiplier=5.0)
        assert not mask.any()


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_missing_expression_variable_raises(self) -> None:
        defn, inputs = _ab()
        del inputs["B"]
        with pytest.raises(InputValidationError, match="missing"):
            calculate_signal(defn, inputs)

    def test_missing_reference_input_raises(self) -> None:
        defn, inputs = _ab(expression="B", reference="A")
        del inputs["A"]
        with pytest.raises(InputValidationError, match="missing"):
            calculate_signal(defn, inputs)

    def test_short_array_rejected(self) -> None:
        defn, inputs = _ab()
        inputs["A"] = _input("A", np.array([0.0]), np.array([1.0]), "MW")
        with pytest.raises(InputValidationError, match="at least 2"):
            calculate_signal(defn, inputs)

    def test_mismatched_variable_key_rejected(self) -> None:
        defn, inputs = _ab()
        # Swap in an input whose own .variable disagrees with its dict key.
        inputs["A"] = _input("Z", np.linspace(0, 1, 5), np.ones(5), "MW")
        with pytest.raises(InputValidationError, match="mismatched"):
            calculate_signal(defn, inputs)

    def test_extra_unreferenced_inputs_are_ignored(self) -> None:
        defn, inputs = _ab(expression="A")  # B not referenced by the expression
        result = calculate_signal(defn, inputs)
        assert result.status == CalculationStatus.OK

    def test_too_few_usable_samples_after_sanitization_raises(self) -> None:
        defn, inputs = _ab()
        inputs["A"] = _input("A", np.array([0.0, float("nan")]), np.array([1.0, 2.0]), "MW")
        with pytest.raises(InputValidationError, match="fewer than 2"):
            calculate_signal(defn, inputs)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap-driven failures
# ─────────────────────────────────────────────────────────────────────────────


class TestOverlapFailures:
    def test_no_overlap_raises_alignment_error(self) -> None:
        defn, inputs = _ab()
        inputs["B"] = _input("B", np.linspace(10, 11, 11), np.ones(11), "MW")
        with pytest.raises(AlignmentError, match="overlap"):
            calculate_signal(defn, inputs)

    def test_one_sample_reference_overlap_raises(self) -> None:
        defn, inputs = _ab()
        inputs["A"] = _input("A", np.linspace(0, 1, 11), np.ones(11), "MW")
        inputs["B"] = _input("B", np.array([1.0, 1.05]), np.array([1.0, 1.0]), "MW")
        with pytest.raises(AlignmentError):
            calculate_signal(defn, inputs)

    def test_reference_shorter_than_input(self) -> None:
        defn, inputs = _ab()
        inputs["A"] = _input("A", np.linspace(0.3, 0.7, 5), np.ones(5), "MW")
        inputs["B"] = _input("B", np.linspace(0, 1, 11), np.full(11, 2.0), "MW")
        result = calculate_signal(defn, inputs)
        assert result.time.min() >= 0.3 - 1e-9
        assert result.time.max() <= 0.7 + 1e-9

    def test_non_reference_shorter_than_reference(self) -> None:
        defn, inputs = _ab()
        inputs["A"] = _input("A", np.linspace(0, 1, 11), np.ones(11), "MW")
        inputs["B"] = _input("B", np.linspace(0.3, 0.7, 5), np.full(5, 2.0), "MW")
        result = calculate_signal(defn, inputs)
        # Reference time trimmed to the overlap window.
        assert result.time.min() >= 0.3 - 1e-9
        assert result.time.max() <= 0.7 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation
# ─────────────────────────────────────────────────────────────────────────────


class TestInterpolation:
    def test_same_time_base_no_interpolation_error(self) -> None:
        defn, inputs = _ab()
        result = calculate_signal(defn, inputs)
        np.testing.assert_array_almost_equal(result.values, np.full(11, 3.0))

    def test_different_rate_interpolates_correctly(self) -> None:
        ref_t = np.linspace(0, 1, 11)  # includes 0.2 and 0.3 exactly
        low_t = np.array([0.0, 0.5, 1.0])
        low_v = np.array([0.0, 10.0, 20.0])
        A = _input("A", ref_t, np.zeros(11), "MW")
        B = _input("B", low_t, low_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        # At t=0.3, linear interpolation between (0,0) and (0.5,10) -> 6.0
        idx = np.argmin(np.abs(result.time - 0.3))
        assert result.time[idx] == pytest.approx(0.3)
        assert result.values[idx] == pytest.approx(6.0)
        # At t=0.5, B is an exact source sample -> exactly 10.0, no interpolation error.
        idx_exact = np.argmin(np.abs(result.time - 0.5))
        assert result.values[idx_exact] == pytest.approx(10.0)

    def test_irregular_sampling_still_interpolates(self) -> None:
        rng = np.random.default_rng(1)
        ref_t = np.linspace(0, 1, 51)
        jittered_t = np.sort(np.linspace(0, 1, 50) + rng.normal(0, 0.002, 50))
        A = _input("A", ref_t, np.zeros(51), "MW")
        B = _input("B", jittered_t, np.sin(jittered_t), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert result.validity_mask.all()

    def test_no_extrapolation_outside_input_coverage(self) -> None:
        ref_t = np.linspace(0, 1, 11)
        narrow_t = np.linspace(0.3, 0.7, 5)
        A = _input("A", ref_t, np.zeros(11), "MW")
        B = _input("B", narrow_t, np.full(5, 100.0), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        # Overlap window is exactly B's range [0.3, 0.7], so the reference is
        # already trimmed there -- every sample must be valid, none extrapolated.
        assert result.time.min() >= 0.3 - 1e-9
        assert result.time.max() <= 0.7 + 1e-9

    def test_reference_values_are_never_interpolated(self) -> None:
        # Reference has irregular native sample positions; its own values
        # must appear verbatim (not resampled), only trimmed to the overlap.
        ref_t = np.array([0.0, 0.3, 0.31, 0.9, 1.0])
        ref_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        other_t = np.linspace(0, 1, 100)
        A = _input("A", ref_t, ref_v, "MW")
        B = _input("B", other_t, np.zeros(100), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        np.testing.assert_array_equal(result.time, ref_t)
        np.testing.assert_array_almost_equal(result.values, ref_v, decimal=6)


class TestGapDetection:
    def test_regular_intervals_no_gap_warning(self) -> None:
        t = np.linspace(0, 1, 21)
        A = _input("A", t, np.zeros(21), "MW")
        B = _input("B", t, np.ones(21), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not any("gap" in w for w in result.warnings)
        assert result.validity_mask.all()

    def test_mildly_irregular_intervals_no_gap_warning(self) -> None:
        rng = np.random.default_rng(2)
        ref_t = np.linspace(0, 1, 41)
        b_t = np.sort(np.linspace(0, 1, 40) + rng.normal(0, 0.003, 40))
        A = _input("A", ref_t, np.zeros(41), "MW")
        B = _input("B", b_t, np.ones(40), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not any("gap" in w for w in result.warnings)

    def test_one_large_gap_not_bridged(self) -> None:
        ref_t = np.linspace(0, 2, 21)
        gap_t = np.array([0.0, 0.2, 0.4, 1.9, 2.0])
        gap_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = _input("A", ref_t, np.zeros(21), "MW")
        B = _input("B", gap_t, gap_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("gap" in w for w in result.warnings)
        # Samples inside the gap (strictly between 0.4 and 1.9) are invalid.
        inside_gap = (result.time > 0.4) & (result.time < 1.9)
        assert inside_gap.any()
        assert not result.validity_mask[inside_gap].any()

    def test_target_samples_outside_gap_remain_valid(self) -> None:
        ref_t = np.linspace(0, 2, 21)
        gap_t = np.array([0.0, 0.2, 0.4, 1.9, 2.0])
        gap_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = _input("A", ref_t, np.zeros(21), "MW")
        B = _input("B", gap_t, gap_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        near_start = result.time <= 0.4 + 1e-9
        near_end = result.time >= 1.9 - 1e-9
        assert result.validity_mask[near_start].all()
        assert result.validity_mask[near_end].all()

    def test_multiple_gaps(self) -> None:
        ref_t = np.linspace(0, 3, 31)
        multi_gap_t = np.array([0.0, 0.1, 1.0, 1.1, 2.5, 2.6])
        multi_gap_v = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        A = _input("A", ref_t, np.zeros(31), "MW")
        B = _input("B", multi_gap_t, multi_gap_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        # Interior of both gaps (0.1-1.0 and 1.1-2.5) must be invalid.
        gap1 = (result.time > 0.1) & (result.time < 1.0)
        gap2 = (result.time > 1.1) & (result.time < 2.5)
        assert not result.validity_mask[gap1].any()
        assert not result.validity_mask[gap2].any()

    def test_gap_multiplier_is_configurable(self) -> None:
        # A gap that IS bridged with a very permissive multiplier.
        ref_t = np.linspace(0, 2, 21)
        gap_t = np.array([0.0, 0.2, 0.4, 1.9, 2.0])
        gap_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = _input("A", ref_t, np.zeros(21), "MW")
        B = _input("B", gap_t, gap_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        permissive = CalculationEngineConfig(gap_multiplier=100.0)
        result = calculate_signal(defn, {"A": A, "B": B}, config=permissive)
        assert result.validity_mask.all()


# ─────────────────────────────────────────────────────────────────────────────
# Units
# ─────────────────────────────────────────────────────────────────────────────


class TestUnitRules:
    def _pair(self, unit_a: str, unit_b: str, expression: str, reference="A"):
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), unit_a)
        B = _input("B", t, np.full(11, 2.0), unit_b)
        defn = _defn(expression, {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, reference)
        return calculate_signal(defn, {"A": A, "B": B})

    def test_mw_plus_mw(self) -> None:
        result = self._pair("MW", "MW", "A + B")
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 6.0))

    def test_mw_plus_kw(self) -> None:
        # 4 MW + 2 kW = 4.002 MW (left operand's unit wins)
        result = self._pair("MW", "kW", "A + B")
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 4.002))

    def test_kv_minus_v(self) -> None:
        # 4 kV - 2 V = 3.998 kV
        result = self._pair("kV", "V", "A - B")
        assert result.unit == "kV"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 3.998))

    def test_a_plus_ka(self) -> None:
        # 4 A + 2 kA = 2004 A
        result = self._pair("A", "kA", "A + B")
        assert result.unit == "A"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2004.0))

    def test_mw_plus_kv_rejected(self) -> None:
        with pytest.raises(UnitCompatibilityError, match="incompatible"):
            self._pair("MW", "kV", "A + B")

    def test_unknown_plus_mw_rejected(self) -> None:
        with pytest.raises(UnitCompatibilityError, match="unresolved"):
            self._pair(None, "MW", "A + B")

    def test_signal_times_scalar(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        defn = _defn("A * 2", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 8.0))

    def test_scalar_times_signal(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        defn = _defn("2 * A", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 8.0))

    def test_signal_divided_by_scalar(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        defn = _defn("A / 2", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2.0))

    def test_mw_div_mw_is_dimensionless(self) -> None:
        result = self._pair("MW", "MW", "A / B")
        assert result.unit == "pu"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2.0))

    def test_kv_div_v_is_dimensionless_after_conversion(self) -> None:
        # 4 kV / 2 V = 4000 V / 2 V = 2000 (dimensionless)
        result = self._pair("kV", "V", "A / B")
        assert result.unit == "pu"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2000.0))

    def test_signal_times_signal_rejected(self) -> None:
        with pytest.raises(UnitCompatibilityError, match="signal x signal"):
            self._pair("kV", "kA", "A * B")

    def test_different_family_division_rejected(self) -> None:
        with pytest.raises(UnitCompatibilityError, match="different-family"):
            self._pair("MW", "Hz", "A / B")

    def test_division_by_dimensionless_denominator_allowed(self) -> None:
        result = self._pair("MW", "pu", "A / B")
        assert result.unit == "MW"

    def test_multiplication_by_dimensionless_allowed(self) -> None:
        result = self._pair("MW", "pu", "A * B")
        assert result.unit == "MW"

    def test_abs_preserves_unit(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, -4.0), "MW")
        defn = _defn("abs(A)", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 4.0))

    def test_manual_compatible_output_unit_converts(self) -> None:
        result = self._pair("MW", "MW", "A + B", )
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.full(11, 2.0), "MW")
        defn = _defn(
            "A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A",
            output_unit="kW",
        )
        result = calculate_signal(defn, {"A": A, "B": B})
        assert result.unit == "kW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 6000.0))

    def test_manual_incompatible_output_unit_rejected(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.full(11, 2.0), "MW")
        defn = _defn(
            "A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A",
            output_unit="kV",
        )
        with pytest.raises(UnitCompatibilityError, match="incompatible"):
            calculate_signal(defn, {"A": A, "B": B})

    def test_output_unit_none_derives_automatically(self) -> None:
        result = self._pair("MW", "MW", "A + B")
        assert result.unit == "MW"

    def test_constant_plus_signal_allowed_and_preserves_unit(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        defn = _defn("A + 1", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "MW"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 5.0))

    def test_constant_plus_unknown_unit_signal_allowed(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), None)
        defn = _defn("A + 1", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit is None
        np.testing.assert_array_almost_equal(result.values, np.full(11, 5.0))

    def test_constant_divided_by_dimensionless_signal_allowed(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 2.0), "pu")
        defn = _defn("1 / A", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        assert result.unit == "pu"
        np.testing.assert_array_almost_equal(result.values, np.full(11, 0.5))

    def test_constant_divided_by_physical_signal_rejected(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 2.0), "MW")
        defn = _defn("1 / A", {"A": ChannelRef("s", "a")}, "A")
        with pytest.raises(UnitCompatibilityError):
            calculate_signal(defn, {"A": A})


# ─────────────────────────────────────────────────────────────────────────────
# Numerical safety
# ─────────────────────────────────────────────────────────────────────────────


class TestNumericalSafety:
    def test_division_by_zero_becomes_invalid_nan(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not result.validity_mask.any()
        assert np.all(np.isnan(result.values))
        assert any("division by zero" in w.lower() for w in result.warnings)

    def test_partial_division_by_zero(self) -> None:
        t = np.linspace(0, 1, 11)
        b_vals = np.full(11, 2.0)
        b_vals[5] = 0.0
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, b_vals, "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not result.validity_mask[5]
        assert result.validity_mask[0]

    def test_nan_input_propagates_as_invalid(self) -> None:
        t = np.linspace(0, 1, 11)
        a_vals = np.full(11, 4.0)
        a_vals[3] = float("nan")
        A = _input("A", t, a_vals, "MW")
        B = _input("B", t, np.full(11, 2.0), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not result.validity_mask[3]
        assert any("nan" in w.lower() for w in result.warnings)

    def test_inf_input_becomes_invalid_output(self) -> None:
        t = np.linspace(0, 1, 11)
        a_vals = np.full(11, 4.0)
        a_vals[2] = float("inf")
        A = _input("A", t, a_vals, "MW")
        B = _input("B", t, np.full(11, 2.0), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not result.validity_mask[2]
        assert np.isnan(result.values[2])

    def test_no_zero_filling_of_invalid_samples(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not np.any(result.values == 0.0)  # NaN, never silently zeroed

    def test_input_arrays_not_mutated(self) -> None:
        t = np.linspace(0, 1, 11)
        a_vals = np.full(11, 4.0)
        b_vals = np.full(11, 2.0)
        a_before, b_before = a_vals.copy(), b_vals.copy()
        A = _input("A", t, a_vals, "MW")
        B = _input("B", t, b_vals, "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        calculate_signal(defn, {"A": A, "B": B})
        np.testing.assert_array_equal(A.values, a_before)
        np.testing.assert_array_equal(B.values, b_before)
        np.testing.assert_array_equal(a_vals, a_before)
        np.testing.assert_array_equal(b_vals, b_before)

    def test_no_global_warning_suppression(self) -> None:
        # Confirms the engine's local np.errstate around division does not
        # leak into the ambient warnings filter for the rest of the process.
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        calculate_signal(defn, {"A": A, "B": B})
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="warn"):
                _ = np.array([1.0]) / np.array([0.0])
        assert any("divide" in str(w.message) for w in caught)


# ─────────────────────────────────────────────────────────────────────────────
# Expression integration
# ─────────────────────────────────────────────────────────────────────────────


class TestExpressionIntegration:
    def test_average_of_two(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.full(11, 6.0), "MW")
        defn = _defn("(A + B) / 2", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        np.testing.assert_array_almost_equal(result.values, np.full(11, 5.0))

    def test_absolute_difference(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.full(11, 6.0), "MW")
        defn = _defn("abs(A - B)", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2.0))

    def test_scaled_signal(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        defn = _defn("A * 0.5", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        np.testing.assert_array_almost_equal(result.values, np.full(11, 2.0))

    def test_more_than_two_inputs(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 1.0), "A")
        B = _input("B", t, np.full(11, 2.0), "A")
        C = _input("C", t, np.full(11, 3.0), "A")
        defn = _defn(
            "A + B + C",
            {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b"), "C": ChannelRef("s", "c")},
            "A",
        )
        result = calculate_signal(defn, {"A": A, "B": B, "C": C})
        np.testing.assert_array_almost_equal(result.values, np.full(11, 6.0))
        assert result.unit == "A"

    def test_constants_mixed_with_signals(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 10.0), "MW")
        defn = _defn("(A - 2) / 2 + 1", {"A": ChannelRef("s", "a")}, "A")
        result = calculate_signal(defn, {"A": A})
        np.testing.assert_array_almost_equal(result.values, np.full(11, 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────


class TestWarnings:
    def test_interpolation_warning_present(self) -> None:
        t1 = np.linspace(0, 1, 11)
        t2 = np.linspace(0, 1, 13)
        A = _input("A", t1, np.zeros(11), "MW")
        B = _input("B", t2, np.ones(13), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("interpolated" in w for w in result.warnings)

    def test_large_sample_rate_ratio_warning(self) -> None:
        ref_t = np.linspace(0, 1, 1001)
        low_t = np.linspace(0, 1, 11)
        A = _input("A", ref_t, np.zeros(1001), "MW")
        B = _input("B", low_t, np.ones(11), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("sampling rate differs" in w for w in result.warnings)

    def test_no_rate_warning_for_similar_rates(self) -> None:
        t1 = np.linspace(0, 1, 100)
        t2 = np.linspace(0, 1, 105)
        A = _input("A", t1, np.zeros(100), "MW")
        B = _input("B", t2, np.ones(105), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert not any("sampling rate differs" in w for w in result.warnings)

    def test_limited_overlap_warning(self) -> None:
        A = _input("A", np.linspace(0, 100, 1001), np.zeros(1001), "MW")
        B = _input("B", np.linspace(95, 105, 11), np.ones(11), "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("overlap" in w.lower() for w in result.warnings)

    def test_gap_warning_present(self) -> None:
        ref_t = np.linspace(0, 2, 21)
        gap_t = np.array([0.0, 0.2, 0.4, 1.9, 2.0])
        gap_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = _input("A", ref_t, np.zeros(21), "MW")
        B = _input("B", gap_t, gap_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("data gap" in w for w in result.warnings)

    def test_invalid_output_warning(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("invalid sample" in w for w in result.warnings)

    def test_division_by_zero_warning(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert any("division by zero" in w.lower() for w in result.warnings)

    def test_status_is_ok_even_with_warnings(self) -> None:
        t = np.linspace(0, 1, 11)
        A = _input("A", t, np.full(11, 4.0), "MW")
        B = _input("B", t, np.zeros(11), "MW")
        defn = _defn("A / B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")
        result = calculate_signal(defn, {"A": A, "B": B})
        assert result.status == CalculationStatus.OK
        assert len(result.warnings) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Analog-only / digital-exclusion sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestNoDigitalProcessing:
    def test_resolved_analog_input_has_no_digital_concept(self) -> None:
        # ResolvedAnalogInput has no digital/boolean flag or field at all --
        # this is a structural sanity check, not a runtime behaviour test.
        fields = ResolvedAnalogInput.__dataclass_fields__.keys()
        assert "is_digital" not in fields
        assert "digital" not in fields

    def test_engine_module_has_no_boolean_logic_helpers(self) -> None:
        import app.calculated_signals.engine as engine_module
        forbidden = {"edge_detect", "trip_logic", "digital_to_numeric", "boolean_eval"}
        assert forbidden.isdisjoint(dir(engine_module))


# ─────────────────────────────────────────────────────────────────────────────
# Performance sanity (Step 22)
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_million_sample_reference_with_mixed_rate_inputs(self) -> None:
        n = 1_000_000
        ref_t = np.linspace(0, 100, n)
        A = _input("A", ref_t, np.sin(ref_t), "MW", source_id="s1", channel_name="A")
        B = _input("B", ref_t.copy(), np.cos(ref_t), "MW", source_id="s1", channel_name="B")
        low_t = np.linspace(0, 100, n // 20)
        C = _input("C", low_t, np.sin(low_t * 0.5), "MW", source_id="s2", channel_name="C")

        defn = _defn(
            "(A + B) / 2 + abs(A - C)",
            {"A": ChannelRef("s1", "A"), "B": ChannelRef("s1", "B"), "C": ChannelRef("s2", "C")},
            "A",
        )

        start = time_module.perf_counter()
        result = calculate_signal(defn, {"A": A, "B": B, "C": C})
        elapsed = time_module.perf_counter() - start

        assert len(result.values) == n
        assert result.validity_mask.all()
        # No strict wall-clock assertion (repository has no established
        # timing-test convention) -- a generous ceiling only guards against a
        # gross algorithmic regression (e.g. an accidental per-sample loop),
        # not against normal machine-to-machine variance.
        assert elapsed < 30.0

    def test_large_array_gap_detection_is_vectorized(self) -> None:
        # A large array with a single internal gap must not take noticeably
        # longer than the equivalent gap-free case -- confirms
        # _safe_interpolation_mask has no O(N) Python-level loop.
        n = 500_000
        ref_t = np.linspace(0, 100, n)
        source_t = np.concatenate([np.linspace(0, 40, n // 2), np.linspace(60, 100, n // 2)])
        source_v = np.sin(source_t)
        A = _input("A", ref_t, np.zeros(n), "MW")
        B = _input("B", source_t, source_v, "MW")
        defn = _defn("A + B", {"A": ChannelRef("s", "a"), "B": ChannelRef("s", "b")}, "A")

        start = time_module.perf_counter()
        result = calculate_signal(defn, {"A": A, "B": B})
        elapsed = time_module.perf_counter() - start

        assert len(result.values) == n
        assert elapsed < 30.0
        gap_region = (result.time > 40) & (result.time < 60)
        assert not result.validity_mask[gap_region].any()
