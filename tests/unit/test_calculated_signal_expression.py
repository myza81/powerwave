"""Unit tests for app.calculated_signals.expression.

Covers: allowed-syntax acceptance, rejection of every disallowed construct,
structural DoS limits, and numerical correctness of evaluation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from app.calculated_signals.expression import (
    MAX_AST_NODE_COUNT,
    MAX_EXPRESSION_LENGTH,
    MAX_NESTING_DEPTH,
    MAX_VARIABLE_NAME_LENGTH,
    ExpressionEvaluationError,
    ExpressionSyntaxError,
    ExpressionValidationError,
    evaluate_expression,
    validate_expression,
)


def _validate(expr: str, variables=("A", "B", "C")):
    return validate_expression(expr, set(variables))


def _run(expr: str, **values):
    validated = _validate(expr, values.keys())
    return evaluate_expression(validated, values)


# ─────────────────────────────────────────────────────────────────────────────
# Allowed syntax
# ─────────────────────────────────────────────────────────────────────────────


class TestAllowedSyntax:
    @pytest.mark.parametrize("expr", [
        "A",
        "A + B",
        "A - B",
        "A * B",
        "A / B",
        "-A",
        "+A",
        "(A + B) / 2",
        "abs(A)",
        "abs(A - B)",
        "A * 0.5",
        "2 * A",
        "A + 1",
    ])
    def test_parses_and_validates(self, expr: str) -> None:
        validated = _validate(expr)
        assert validated.expression == expr

    def test_parentheses_need_no_dedicated_node_and_affect_precedence(self) -> None:
        assert _run("(A + B) * 2", A=1.0, B=2.0) == 6.0
        assert _run("A + B * 2", A=1.0, B=2.0) == 5.0

    def test_referenced_variables_collected(self) -> None:
        validated = _validate("A + B", variables=("A", "B", "C"))
        assert validated.referenced_variables == frozenset({"A", "B"})

    def test_unused_allowed_variable_is_not_an_error(self) -> None:
        # C is declared as allowed but not used in the expression.
        validated = validate_expression("A + B", {"A", "B", "C"})
        assert validated.referenced_variables == frozenset({"A", "B"})


# ─────────────────────────────────────────────────────────────────────────────
# Rejected constructs
# ─────────────────────────────────────────────────────────────────────────────


class TestRejectedConstructs:
    @pytest.mark.parametrize("expr", [
        "A ** 2",
        "A // B",
        "A % B",
        "A @ B",
        "A < B",
        "A and B",
        "A or B",
        "not A",
        "A if B else C",
        "A[0]",
        "A.shape",
        "A.__class__",
        '__import__("os")',
        'open("x")',
        "sum(A)",
        "min(A)",
        "max(A)",
        "round(A)",
        "np.abs(A)",
        "math.sin(A)",
        "lambda x: x",
        "[x for x in [A]]",
        "(A := B)",
        '{"x": A}',
        "[A, B]",
        "(A, B)",
        '"hello"',
        'b"bytes"',
        "True",
        "False",
        "None",
        "1j",
        "abs(A, B)",
        "abs(x=A)",
    ])
    def test_rejected(self, expr: str) -> None:
        with pytest.raises((ExpressionValidationError, ExpressionSyntaxError)):
            _validate(expr)

    def test_syntax_error_raises_expression_syntax_error(self) -> None:
        with pytest.raises(ExpressionSyntaxError):
            _validate("A +")

    def test_unclosed_parenthesis_raises_syntax_error(self) -> None:
        with pytest.raises(ExpressionSyntaxError):
            _validate("(A + B")

    def test_empty_expression_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError, match="non-empty"):
            _validate("")

    def test_whitespace_only_expression_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError, match="non-empty"):
            _validate("   ")

    def test_unknown_variable_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError, match="unknown variable"):
            _validate("A + Z", variables=("A", "B"))

    def test_missing_variable_at_evaluation_time(self) -> None:
        validated = _validate("A + B", variables=("A", "B"))
        with pytest.raises(ExpressionEvaluationError, match="missing"):
            evaluate_expression(validated, {"A": 1.0})

    def test_object_dtype_array_rejected_at_evaluation(self) -> None:
        validated = _validate("A", variables=("A",))
        with pytest.raises(ExpressionEvaluationError, match="numeric"):
            evaluate_expression(validated, {"A": np.array([1.0, "x"], dtype=object)})

    def test_string_array_rejected_at_evaluation(self) -> None:
        validated = _validate("A", variables=("A",))
        with pytest.raises(ExpressionEvaluationError, match="numeric"):
            evaluate_expression(validated, {"A": np.array(["1.0", "2.0"])})

    def test_bool_operand_rejected_at_evaluation(self) -> None:
        validated = _validate("A", variables=("A",))
        with pytest.raises(ExpressionEvaluationError, match="numeric"):
            evaluate_expression(validated, {"A": True})

    def test_excessively_long_expression_rejected(self) -> None:
        long_expr = "A" + " + A" * (MAX_EXPRESSION_LENGTH // 4 + 10)
        assert len(long_expr) > MAX_EXPRESSION_LENGTH
        with pytest.raises(ExpressionValidationError, match="length"):
            _validate(long_expr, variables=("A",))

    def test_expression_at_length_limit_is_accepted(self) -> None:
        # A single long numeric literal keeps the AST shallow (one BinOp,
        # one Name, one Constant) while pushing the string right up against
        # MAX_EXPRESSION_LENGTH -- isolates the length check from the
        # nesting-depth/node-count checks, which a long *chain* would trip
        # first (see test_excessive_nesting_rejected below).
        prefix = "A + "
        digits = "1" * (MAX_EXPRESSION_LENGTH - len(prefix))
        body = prefix + digits
        assert len(body) == MAX_EXPRESSION_LENGTH
        _validate(body, variables=("A",))  # must not raise

    def test_excessive_nesting_rejected(self) -> None:
        expr = "A"
        for _ in range(MAX_NESTING_DEPTH + 20):
            expr = f"({expr} + A)"
        with pytest.raises(ExpressionValidationError, match="nest"):
            _validate(expr, variables=("A",))

    @staticmethod
    def _balanced_sum(leaf: str, n: int) -> str:
        """Build a balanced binary '+' tree of n copies of *leaf*.

        Depth grows as O(log n) rather than O(n), unlike a flat left-
        associative chain -- this lets a node-count test exceed
        MAX_AST_NODE_COUNT while staying well under MAX_NESTING_DEPTH.
        """
        if n == 1:
            return leaf
        half = n // 2
        left = TestRejectedConstructs._balanced_sum(leaf, half)
        right = TestRejectedConstructs._balanced_sum(leaf, n - half)
        return f"({left} + {right})"

    def test_excessive_node_count_rejected(self) -> None:
        # 140 leaves -> ~139 BinOp + 140 Name nodes = ~279 nodes (> 256),
        # at a balanced depth of ceil(log2(140)) ~= 8 (well under 32) --
        # isolates the node-count check from the nesting-depth check.
        expr = self._balanced_sum("A", 140)
        with pytest.raises(ExpressionValidationError, match="complex"):
            _validate(expr, variables=("A",))

    def test_node_count_at_limit_with_shallow_nesting_is_accepted(self) -> None:
        # 90 leaves -> ~179 nodes (< 256), depth ~= 7 (< 32): must be accepted.
        expr = self._balanced_sum("A", 90)
        _validate(expr, variables=("A",))  # must not raise

    def test_forbidden_variable_name_underscore_prefix(self) -> None:
        with pytest.raises(ExpressionValidationError, match="_"):
            validate_expression("_A", {"_A"})

    def test_forbidden_variable_name_too_long(self) -> None:
        long_name = "A" * (MAX_VARIABLE_NAME_LENGTH + 1)
        with pytest.raises(ExpressionValidationError, match="too long"):
            validate_expression(long_name, {long_name})

    def test_python_keyword_variable_name_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError, match="keyword"):
            validate_expression("for", {"for"})

    def test_reserved_abs_binding_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError, match="reserved"):
            validate_expression("abs", {"abs"})

    def test_dunder_name_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError):
            _validate("__class__")

    def test_no_keyword_arguments_anywhere(self) -> None:
        with pytest.raises(ExpressionValidationError, match="keyword"):
            _validate("abs(x=A)")

    def test_starred_argument_rejected(self) -> None:
        with pytest.raises(ExpressionValidationError):
            _validate("abs(*A)", variables=("A",))


# ─────────────────────────────────────────────────────────────────────────────
# Numerical correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestNumericalCorrectness:
    def test_scalar_scalar_arithmetic(self) -> None:
        assert _run("A + B", A=2.0, B=3.0) == 5.0
        assert _run("A - B", A=5.0, B=3.0) == 2.0
        assert _run("A * B", A=2.0, B=3.0) == 6.0
        assert _run("A / B", A=6.0, B=3.0) == 2.0

    def test_array_array_arithmetic(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        result = _run("A + B", A=a, B=b)
        np.testing.assert_array_equal(result, np.array([11.0, 22.0, 33.0]))

    def test_scalar_array_arithmetic_broadcasts(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        result = _run("A * 2", A=a)
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_unary_operations(self) -> None:
        assert _run("-A", A=5.0) == -5.0
        assert _run("+A", A=5.0) == 5.0
        a = np.array([1.0, -2.0])
        np.testing.assert_array_equal(_run("-A", A=a), np.array([-1.0, 2.0]))

    def test_parentheses_precedence(self) -> None:
        assert _run("(A + B) / 2", A=4.0, B=6.0) == 5.0
        assert _run("A + B / 2", A=4.0, B=6.0) == 7.0

    def test_abs_scalar_and_array(self) -> None:
        assert _run("abs(A - B)", A=1.0, B=5.0) == 4.0
        a = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_array_equal(_run("abs(A)", A=a), np.array([1.0, 2.0, 3.0]))

    def test_broadcasting_behaviour(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        result = _run("A + B", A=a, B=b)
        np.testing.assert_array_equal(result, np.array([[11.0, 22.0], [13.0, 24.0]]))

    def test_shape_mismatch_raises_evaluation_error(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ExpressionEvaluationError):
            _run("A + B", A=a, B=b)

    def test_input_arrays_remain_unchanged(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        a_before = a.copy()
        b_before = b.copy()
        _run("abs(A - B) * 2 + A / B", A=a, B=b)
        np.testing.assert_array_equal(a, a_before)
        np.testing.assert_array_equal(b, b_before)

    def test_scalar_result_is_a_float64_scalar_not_ndarray(self) -> None:
        result = _run("A + B", A=1.0, B=2.0)
        assert isinstance(result, float)
        assert isinstance(result, np.floating)
        assert not isinstance(result, np.ndarray)

    def test_array_result_is_ndarray(self) -> None:
        result = _run("A + B", A=np.array([1.0]), B=np.array([2.0]))
        assert isinstance(result, np.ndarray)

    def test_complex_expression_correctness(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        c = 2.0
        result = _run("abs(A - B) / C + (A + B)", A=a, B=b, C=c)
        expected = np.abs(a - b) / c + (a + b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extra_variables_in_environment_are_ignored(self) -> None:
        # "policy: extra values are ignored, not rejected" -- C is supplied
        # but not referenced by the expression.
        validated = validate_expression("A + B", {"A", "B", "C"})
        result = evaluate_expression(validated, {"A": 1.0, "B": 2.0, "C": 999.0})
        assert result == 3.0

    def test_division_by_zero_follows_numpy_semantics_not_exception(self) -> None:
        # Explicitly documented as deferred to Phase 2B: no NaN substitution
        # happens here, and no exception is raised for scalar or array
        # division by zero -- NumPy's own inf/nan + RuntimeWarning behaviour
        # is left untouched.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _run("A / B", A=1.0, B=0.0)
        assert np.isinf(result)
        assert any("divide by zero" in str(w.message) for w in caught)

    def test_array_division_by_zero_produces_inf_with_warning(self) -> None:
        a = np.array([1.0, -1.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _run("A / B", A=a, B=b)
        assert np.isposinf(result[0])
        assert np.isneginf(result[1])
        assert np.isnan(result[2])  # 0/0
        assert len(caught) >= 1

    def test_abs_of_nan_propagates_nan(self) -> None:
        result = _run("abs(A)", A=np.array([float("nan"), -1.0]))
        assert np.isnan(result[0])
        assert result[1] == 1.0
