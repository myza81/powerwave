"""The Calculated Signals numerical calculation engine (Phase 2B).

Pipeline implemented by ``calculate_signal()``::

    Reference analog input
            |
    Reference input's own aligned time base
            |
    Find common overlap across all inputs
            |
    Trim reference time to common overlap
            |
    Interpolate every non-reference input onto that time base
            |
    Normalize compatible units where required
            |
    Evaluate the Phase 2A ValidatedExpression tree (values + units together)
            |
    Replace invalid numerical results with NaN
            |
    Build validity mask + warnings
            |
    CalculatedSignalResult

This module knows nothing about EventAnalysisSession, SessionSource, or
ChannelRef -- it operates purely on ResolvedAnalogInput objects (see
models.py), which a later phase is responsible for producing from a real
session. It also knows nothing about digital channels: ResolvedAnalogInput
carries no digital flag, and this engine performs no boolean/status
arithmetic of any kind.

Interpolation reuses app.sessions.alignment_engine.resample_to_grid()
unchanged (verified linear, bounds_error=False, fill_value=np.nan, no
extrapolation, does not mutate its inputs) -- this module does not
implement a second interpolation routine.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ResolvedAnalogInput,
)
from app.calculated_signals.units import (
    DIMENSIONLESS_UNIT,
    NormalizedUnit,
    UnitFamily,
    are_compatible_units,
    convert_values,
    normalize_unit,
)
from app.sessions.alignment_engine import resample_to_grid

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class CalculatedSignalEngineError(ValueError):
    """Base class for all Calculated Signals engine errors."""


class InputValidationError(CalculatedSignalEngineError):
    """A ResolvedAnalogInput, or the binding/input correspondence, is invalid."""


class AlignmentError(CalculatedSignalEngineError):
    """Inputs could not be aligned on a common, sufficient time overlap."""


class UnitCompatibilityError(CalculatedSignalEngineError):
    """An operator was applied to operands with incompatible or unresolved units."""


class CalculationError(CalculatedSignalEngineError):
    """Expression evaluation failed for a reason other than input/unit validity."""


# ─────────────────────────────────────────────────────────────────────────────
# Engine configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CalculationEngineConfig:
    """Named, documented thresholds for the Phase 2B engine's V1 heuristics.

    No repository-wide precedent exists for "large sample-rate ratio" or
    "limited overlap fraction" thresholds (checked: no such constant exists
    anywhere in app/), so both are V1 heuristics, not reused conventions.

    gap_multiplier's default of 5.0 is likewise a new V1 heuristic, chosen
    deliberately larger than the closest existing precedent --
    app.import_wizard.interval_inference.infer_interval's 1.5x
    dominant-interval threshold for *counting* estimated missing samples
    during import diagnostics. That threshold is intentionally sensitive
    (it wants to flag even modest gaps for the user to review); this
    engine's gap_multiplier instead gates whether interpolation is
    performed at all, so it must be conservative enough that ordinary
    jittery/irregular sampling (which the V1 concept explicitly requires to
    keep interpolating normally) is not mistaken for a real data gap.
    """

    gap_multiplier: float = 5.0
    large_sample_rate_ratio: float = 10.0
    limited_overlap_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not (self.gap_multiplier > 0):
            raise ValueError("gap_multiplier must be positive")
        if not (self.large_sample_rate_ratio > 0):
            raise ValueError("large_sample_rate_ratio must be positive")
        if not (0.0 <= self.limited_overlap_fraction <= 1.0):
            raise ValueError("limited_overlap_fraction must be between 0.0 and 1.0")


_DEFAULT_CONFIG = CalculationEngineConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Time sanitization (Step 7)
# ─────────────────────────────────────────────────────────────────────────────


def _sanitize_time_values(
    time: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (clean_time, clean_values, dropped_nonfinite, dropped_duplicates).

    Policy (never mutates the caller's arrays -- always builds new ones):
      1. Drop samples whose time is not finite (NaN/+-inf).
      2. Stable-sort ascending by time.
      3. Collapse duplicate timestamps, keeping only the FIRST occurrence at
         each timestamp -- this matches the existing repository convention
         in app.import_wizard.interval_inference.detect_duplicates, which
         uses pandas' `duplicated(keep="first")` semantics. A stable sort in
         step 2 is what makes "first occurrence" well-defined: ties at the
         same timestamp keep their original relative order, so "first" means
         "first in the original array", not an arbitrary tie-break.

    Duplicate samples are dropped, never averaged -- there is no existing
    repository precedent for averaging duplicate timestamps.
    """
    finite_mask = np.isfinite(time)
    dropped_nonfinite = int(time.size - np.count_nonzero(finite_mask))
    t = time[finite_mask]
    v = values[finite_mask]

    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    v_sorted = v[order]

    if t_sorted.size == 0:
        return t_sorted, v_sorted, dropped_nonfinite, 0

    keep = np.ones(t_sorted.size, dtype=bool)
    keep[1:] = t_sorted[1:] != t_sorted[:-1]
    dropped_duplicates = int(t_sorted.size - np.count_nonzero(keep))

    return t_sorted[keep], v_sorted[keep], dropped_nonfinite, dropped_duplicates


# ─────────────────────────────────────────────────────────────────────────────
# Internal-gap detection (Step 9)
# ─────────────────────────────────────────────────────────────────────────────


def _nominal_interval(sorted_time: np.ndarray) -> float | None:
    """Return the median positive consecutive time delta, or None.

    A robust statistic (median, not mean) so a handful of outlier gaps do
    not skew the estimate of the source's "normal" sampling interval --
    mirrors the same choice already made by
    app.sessions.alignment_engine.build_common_time_grid and
    app.import_wizard.interval_inference.infer_interval.
    """
    if sorted_time.size < 2:
        return None
    diffs = np.diff(sorted_time)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive))


def _safe_interpolation_mask(
    source_time: np.ndarray, target_time: np.ndarray, gap_multiplier: float
) -> np.ndarray:
    """Return a boolean mask, True where interpolating *source_time*-based
    data onto *target_time* is safe (not bridging an internal data gap).

    A target timestamp is unsafe when the two real source samples bracketing
    it are farther apart than ``nominal_interval * gap_multiplier`` -- an
    ordinary jittery/irregular sampling interval stays well under this
    (target still bracketed by close samples), but a genuine dropout/gap in
    the source produces a bracketing interval many multiples of the nominal
    spacing. Fully vectorized (no per-sample Python loop) so this scales to
    multi-million-sample inputs.
    """
    n = source_time.size
    m = target_time.size
    if n < 2 or m == 0:
        return np.zeros(m, dtype=bool)

    nominal = _nominal_interval(source_time)
    if nominal is None or nominal <= 0:
        return np.zeros(m, dtype=bool)
    gap_threshold = nominal * gap_multiplier

    right_idx = np.searchsorted(source_time, target_time, side="left")
    right_safe = np.clip(right_idx, 0, n - 1)
    exact_match = (right_idx < n) & (source_time[right_safe] == target_time)

    left_idx = right_idx - 1
    in_range = (left_idx >= 0) & (right_idx < n)
    left_safe = np.clip(left_idx, 0, n - 1)
    local_interval = source_time[right_safe] - source_time[left_safe]

    within_threshold = in_range & (local_interval <= gap_threshold)
    return exact_match | within_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Overlap (Step 8)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _SanitizedInput:
    variable: str
    time: np.ndarray
    values: np.ndarray
    unit: str | None
    dropped_nonfinite: int
    dropped_duplicates: int


def _compute_overlap(sanitized: Mapping[str, _SanitizedInput]) -> tuple[float, float]:
    """Return (overlap_start, overlap_end) = (max of per-input minima, min
    of per-input maxima) across every sanitized input's time range."""
    starts = [float(s.time[0]) for s in sanitized.values()]
    ends = [float(s.time[-1]) for s in sanitized.values()]
    return max(starts), min(ends)


# ─────────────────────────────────────────────────────────────────────────────
# Unit-aware expression evaluation (Steps 12-13)
# ─────────────────────────────────────────────────────────────────────────────

_UNRESOLVED_UNIT = NormalizedUnit(raw=None, canonical=None, family=None, scale_to_base=None)
_DIMENSIONLESS_NORMALIZED = normalize_unit(DIMENSIONLESS_UNIT)


@dataclass(slots=True)
class _Operand:
    """A value paired with its propagated unit, used only inside this
    module's expression traversal -- never exposed publicly.

    is_constant marks a subtree built entirely from numeric literals (no
    Name reference anywhere below it): such a subtree carries no unit claim
    at all, and combining it with a real signal via +, -, *, or / is always
    permitted (a bare number added to/multiplied against/dividing a signal
    does not conflict with that signal's unit -- there is nothing to
    convert). Two real signals must have compatible (or explicitly
    dimensionless, for */÷) units instead; see _apply_add_sub / _apply_mult
    / _apply_div.
    """

    value: "np.ndarray | np.floating"
    unit: NormalizedUnit
    is_constant: bool


@dataclass(slots=True)
class _EvalContext:
    """Mutable counters threaded through the evaluation for warning generation."""

    division_by_zero_samples: int = 0


def _safe_div(a, b):
    """Divide using NumPy semantics without emitting a RuntimeWarning here.

    Local errstate only -- scoped to this one division, never a global
    warnings.simplefilter/np.seterr call. The numeric result (inf/nan for
    division by zero) is unchanged; only the warning emission is suppressed
    at this call site. Final NaN/inf sanitization still happens once, at the
    top level, in calculate_signal().
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def _apply_add_sub(left: _Operand, right: _Operand, subtract: bool) -> _Operand:
    op = np.subtract if subtract else np.add
    verb = "subtract" if subtract else "add"
    noun = "subtraction" if subtract else "addition"

    if left.is_constant and right.is_constant:
        return _Operand(op(left.value, right.value), _UNRESOLVED_UNIT, True)
    if left.is_constant:
        return _Operand(op(left.value, right.value), right.unit, False)
    if right.is_constant:
        return _Operand(op(left.value, right.value), left.unit, False)

    if not left.unit.is_resolved or not right.unit.is_resolved:
        raise UnitCompatibilityError(
            f"cannot {verb} signals with an unresolved unit "
            f"({left.unit.raw!r} and {right.unit.raw!r})"
        )
    if left.unit.family != right.unit.family:
        raise UnitCompatibilityError(
            f"incompatible units for {noun}: {left.unit.raw!r} "
            f"({left.unit.family.value}) and {right.unit.raw!r} ({right.unit.family.value})"
        )
    right_converted = convert_values(right.value, right.unit, left.unit)
    return _Operand(op(left.value, right_converted), left.unit, False)


def _apply_mult(left: _Operand, right: _Operand) -> _Operand:
    if left.is_constant and right.is_constant:
        return _Operand(left.value * right.value, _UNRESOLVED_UNIT, True)
    if left.is_constant:
        return _Operand(left.value * right.value, right.unit, False)
    if right.is_constant:
        return _Operand(left.value * right.value, left.unit, False)

    left_dimless = left.unit.is_resolved and left.unit.family == UnitFamily.DIMENSIONLESS
    right_dimless = right.unit.is_resolved and right.unit.family == UnitFamily.DIMENSIONLESS
    if left_dimless or right_dimless:
        out_unit = right.unit if left_dimless else left.unit
        return _Operand(left.value * right.value, out_unit, False)

    raise UnitCompatibilityError(
        "signal x signal is not supported in V1 unless one operand is dimensionless "
        f"({left.unit.raw!r} x {right.unit.raw!r})"
    )


def _apply_div(left: _Operand, right: _Operand, ctx: _EvalContext) -> _Operand:
    result = _safe_div(left.value, right.value)
    zero_mask = np.asarray(right.value) == 0
    ctx.division_by_zero_samples += int(np.count_nonzero(zero_mask))

    if right.is_constant:
        if left.is_constant:
            return _Operand(result, _UNRESOLVED_UNIT, True)
        return _Operand(result, left.unit, False)

    if left.is_constant:
        if right.unit.is_resolved and right.unit.family == UnitFamily.DIMENSIONLESS:
            return _Operand(result, right.unit, False)
        raise UnitCompatibilityError(
            "dividing a constant by a signal is not supported in V1 unless the "
            f"signal is dimensionless ({right.unit.raw!r})"
        )

    right_dimless = right.unit.is_resolved and right.unit.family == UnitFamily.DIMENSIONLESS
    if right_dimless:
        return _Operand(result, left.unit, False)

    if not left.unit.is_resolved or not right.unit.is_resolved:
        raise UnitCompatibilityError(
            f"cannot divide signals with an unresolved unit ({left.unit.raw!r} / {right.unit.raw!r})"
        )
    if left.unit.family == right.unit.family:
        left_converted = convert_values(left.value, left.unit, right.unit)
        return _Operand(_safe_div(left_converted, right.value), _DIMENSIONLESS_NORMALIZED, False)

    raise UnitCompatibilityError(
        "different-family signal division is not supported in V1 unless the "
        f"denominator is dimensionless ({left.unit.raw!r} / {right.unit.raw!r})"
    )


def _evaluate_with_units(node: ast.AST, env: Mapping[str, _Operand], ctx: _EvalContext) -> _Operand:
    """Walk the already-validated Phase 2A AST, computing value and unit
    together at every node. Reuses the exact tree produced by
    app.calculated_signals.expression.validate_expression -- this function
    does not re-implement or relax that module's security validation; it
    only adds unit propagation to the same restricted node types Phase 2A
    already proved safe.
    """
    if isinstance(node, ast.Expression):
        return _evaluate_with_units(node.body, env, ctx)

    if isinstance(node, ast.BinOp):
        left = _evaluate_with_units(node.left, env, ctx)
        right = _evaluate_with_units(node.right, env, ctx)
        if isinstance(node.op, ast.Add):
            return _apply_add_sub(left, right, subtract=False)
        if isinstance(node.op, ast.Sub):
            return _apply_add_sub(left, right, subtract=True)
        if isinstance(node.op, ast.Mult):
            return _apply_mult(left, right)
        return _apply_div(left, right, ctx)  # ast.Div is the only remaining allowed type

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_with_units(node.operand, env, ctx)
        value = +operand.value if isinstance(node.op, ast.UAdd) else -operand.value
        return _Operand(value, operand.unit, operand.is_constant)

    if isinstance(node, ast.Call):
        operand = _evaluate_with_units(node.args[0], env, ctx)
        return _Operand(np.abs(operand.value), operand.unit, operand.is_constant)

    if isinstance(node, ast.Name):
        return env[node.id]

    # ast.Constant is the only remaining allowed leaf type after Phase 2A validation.
    return _Operand(np.float64(node.value), _UNRESOLVED_UNIT, True)


# ─────────────────────────────────────────────────────────────────────────────
# Public engine entry point (Step 19)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_signal(
    definition: CalculatedSignalDefinition,
    inputs: Mapping[str, ResolvedAnalogInput],
    *,
    config: CalculationEngineConfig | None = None,
) -> CalculatedSignalResult:
    """Compute a CalculatedSignalResult from a definition and resolved inputs.

    Only ``ResolvedAnalogInput`` values are consumed -- this function never
    looks up a ChannelRef against a session; *inputs* must already contain
    one entry per required variable name, keyed by that variable name.

    Required variables are the reference variable plus every variable the
    expression actually references (``ValidatedExpression.referenced_
    variables``) -- not necessarily every key of
    ``definition.variable_bindings``, matching Phase 2A's own "declared but
    unused bindings are fine" policy. Extra entries in *inputs* beyond the
    required set are ignored, not rejected (same policy as Phase 2A's
    evaluate_expression).

    Raises one of this module's exceptions (InputValidationError,
    AlignmentError, UnitCompatibilityError, CalculationError) for every
    structural failure -- it never returns a CalculatedSignalResult with
    status=ERROR itself; converting a caught exception into an ERROR-status
    result (if desired) is left to the caller (a future phase), so this
    function never silently returns an empty/placeholder result.

    Returns status=CalculationStatus.OK on success, even when warnings were
    generated -- warnings describe assumptions and data-quality observations,
    not calculation validity.
    """
    cfg = config or _DEFAULT_CONFIG
    warnings: list[str] = []

    # ── Step 1: expression validation (reused unchanged from Phase 2A) ──
    from app.calculated_signals.expression import validate_expression

    validated = validate_expression(
        definition.expression, set(definition.variable_bindings.keys())
    )

    required_variables = set(validated.referenced_variables) | {definition.reference_variable}

    # ── Step 2: binding/input correspondence ──
    missing_inputs = sorted(v for v in required_variables if v not in inputs)
    if missing_inputs:
        raise InputValidationError(f"missing resolved input(s) for variable(s): {missing_inputs}")

    for var in required_variables:
        resolved = inputs[var]
        if not isinstance(resolved, ResolvedAnalogInput):
            raise InputValidationError(
                f"input {var!r} must be a ResolvedAnalogInput, got {type(resolved).__name__}"
            )
        if resolved.variable != var:
            raise InputValidationError(
                f"input keyed {var!r} has mismatched ResolvedAnalogInput.variable "
                f"{resolved.variable!r}"
            )
        if resolved.time.size < 2:
            raise InputValidationError(f"input {var!r} must have at least 2 samples")

    if definition.reference_variable not in required_variables:
        # Structurally unreachable given CalculatedSignalDefinition's own
        # validation (reference_variable is always a variable_bindings key,
        # and required_variables always includes it) -- defensive only.
        raise InputValidationError("reference_variable is not among the required variables")

    # ── Step 3: sanitize every required input's time/value arrays ──
    sanitized: dict[str, _SanitizedInput] = {}
    for var in required_variables:
        resolved = inputs[var]
        clean_t, clean_v, dropped_nf, dropped_dup = _sanitize_time_values(
            resolved.time, resolved.values
        )
        if clean_t.size < 2:
            raise InputValidationError(
                f"input {var!r} has fewer than 2 usable samples after removing "
                "non-finite timestamps and duplicate timestamps"
            )
        sanitized[var] = _SanitizedInput(
            variable=var, time=clean_t, values=clean_v, unit=resolved.unit,
            dropped_nonfinite=dropped_nf, dropped_duplicates=dropped_dup,
        )
        if dropped_nf or dropped_dup:
            warnings.append(
                f"Input {var!r} had {dropped_nf} non-finite timestamp(s) and "
                f"{dropped_dup} duplicate timestamp(s) removed before alignment "
                "(duplicate timestamps: first occurrence kept)."
            )
        if np.any(np.isnan(clean_v)):
            nan_pct = 100.0 * float(np.count_nonzero(np.isnan(clean_v))) / clean_v.size
            warnings.append(f"Input {var!r} contains NaN values ({nan_pct:.1f}% of samples).")

    # ── Step 4: common overlap ──
    overlap_start, overlap_end = _compute_overlap(sanitized)
    if overlap_start > overlap_end:
        raise AlignmentError(
            f"no time overlap exists across the required inputs "
            f"(overlap window would be [{overlap_start}, {overlap_end}])"
        )

    ref = sanitized[definition.reference_variable]
    ref_full_start, ref_full_end = float(ref.time[0]), float(ref.time[-1])
    ref_mask = (ref.time >= overlap_start) & (ref.time <= overlap_end)
    ref_time_trimmed = ref.time[ref_mask]
    ref_values_trimmed = ref.values[ref_mask]

    if ref_time_trimmed.size < 2:
        raise AlignmentError(
            f"common overlap [{overlap_start}, {overlap_end}] contains fewer than "
            "2 reference samples"
        )

    overlap_duration = overlap_end - overlap_start
    ref_full_duration = ref_full_end - ref_full_start
    if ref_full_duration > 0:
        overlap_fraction = overlap_duration / ref_full_duration
        if overlap_fraction < cfg.limited_overlap_fraction:
            warnings.append(
                f"Common overlap covers only {overlap_fraction * 100.0:.1f}% of the "
                "reference signal's own duration."
            )

    ref_nominal = _nominal_interval(ref.time)
    ref_rate = (1.0 / ref_nominal) if ref_nominal else None

    # ── Step 5: interpolate every non-reference required input ──
    unit_env: dict[str, NormalizedUnit] = {
        definition.reference_variable: normalize_unit(ref.unit),
    }
    value_env: dict[str, "np.ndarray | np.floating"] = {
        definition.reference_variable: ref_values_trimmed,
    }

    for var in required_variables:
        if var == definition.reference_variable:
            continue
        src = sanitized[var]
        interpolated = resample_to_grid(src.time, src.values, ref_time_trimmed)

        safe_mask = _safe_interpolation_mask(src.time, ref_time_trimmed, cfg.gap_multiplier)
        n_gap_invalidated = int(np.count_nonzero(~safe_mask & np.isfinite(interpolated)))
        if not np.all(safe_mask):
            interpolated = np.where(safe_mask, interpolated, np.nan)
        if n_gap_invalidated > 0:
            warnings.append(
                f"Input {var!r} has one or more internal data gaps larger than "
                f"{cfg.gap_multiplier}x its nominal sampling interval; "
                f"{n_gap_invalidated} reference sample(s) inside those gaps were "
                "marked invalid rather than interpolated."
            )

        warnings.append(
            f"Input {var!r} was linearly interpolated onto the reference time base "
            "(no extrapolation outside its own coverage)."
        )

        src_nominal = _nominal_interval(src.time)
        if ref_rate and src_nominal and src_nominal > 0:
            src_rate = 1.0 / src_nominal
            ratio = max(ref_rate, src_rate) / min(ref_rate, src_rate)
            if ratio > cfg.large_sample_rate_ratio:
                warnings.append(
                    f"Input {var!r}'s sampling rate differs from the reference "
                    f"signal's by a factor of {ratio:.1f}x."
                )

        value_env[var] = interpolated
        unit_env[var] = normalize_unit(src.unit)

    # ── Step 6: evaluate expression (value + unit together) ──
    operand_env: dict[str, _Operand] = {
        var: _Operand(value_env[var], unit_env[var], is_constant=False)
        for var in required_variables
    }
    ctx = _EvalContext()
    try:
        result_operand = _evaluate_with_units(validated.tree, operand_env, ctx)
    except CalculatedSignalEngineError:
        raise
    except Exception as exc:  # noqa: BLE001 -- wrap, never leak a raw traceback
        raise CalculationError(f"failed to evaluate expression: {exc}") from None

    raw_values = np.broadcast_to(
        np.asarray(result_operand.value, dtype=np.float64), ref_time_trimmed.shape
    ).copy()

    if ctx.division_by_zero_samples > 0:
        warnings.append(
            f"Division by zero occurred at {ctx.division_by_zero_samples} sample "
            "position(s); those output samples are invalid (NaN)."
        )

    # ── Step 7: sanitize invalid numerical output ──
    validity_mask = np.isfinite(raw_values)
    sanitized_values = np.where(validity_mask, raw_values, np.nan)

    n_invalid = int(np.count_nonzero(~validity_mask))
    if n_invalid > 0:
        invalid_pct = 100.0 * n_invalid / sanitized_values.size
        warnings.append(
            f"Output contains {n_invalid} invalid sample(s) ({invalid_pct:.1f}%), "
            "shown as NaN and excluded by the validity mask."
        )

    # ── Step 8: output-unit contract ──
    derived_unit = result_operand.unit
    if definition.output_unit is None:
        final_values = sanitized_values
        final_unit = derived_unit.canonical if derived_unit.is_resolved else None
        if not derived_unit.is_resolved and not result_operand.is_constant:
            warnings.append("Output unit could not be derived and was left unresolved.")
    else:
        requested = normalize_unit(definition.output_unit)
        if not requested.is_resolved:
            raise UnitCompatibilityError(
                f"output_unit {definition.output_unit!r} is not a recognized unit"
            )
        if not derived_unit.is_resolved:
            raise UnitCompatibilityError(
                f"cannot verify requested output_unit {definition.output_unit!r}: "
                "the calculated result's own unit could not be derived"
            )
        if not are_compatible_units(requested, derived_unit):
            raise UnitCompatibilityError(
                f"requested output_unit {definition.output_unit!r} "
                f"({requested.family.value}) is incompatible with the derived unit "
                f"{derived_unit.canonical!r} ({derived_unit.family.value})"
            )
        final_values = convert_values(sanitized_values, derived_unit, requested)
        final_unit = requested.canonical

    return CalculatedSignalResult(
        calc_id=definition.calc_id,
        time=ref_time_trimmed,
        values=final_values,
        validity_mask=validity_mask,
        unit=final_unit,
        status=CalculationStatus.OK,
        error_message=None,
        computed_at=datetime.now(timezone.utc),
        warnings=warnings,
    )
