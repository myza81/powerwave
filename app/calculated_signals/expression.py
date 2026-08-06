"""A restricted, secure expression parser and evaluator for Calculated Signals.

Expressions are parsed with ``ast.parse(expr, mode="eval")`` and then walked
recursively against an explicit allow-list of AST node types before any
evaluation happens. Evaluation never calls Python's ``eval()``, ``exec()``,
or ``compile()`` on the expression string or its tree -- ``evaluate_expression``
interprets the already-validated AST node by node, dispatching each allowed
node type to plain NumPy arithmetic itself. Because the allow-list is
enforced structurally (every node type not explicitly recognised raises
before evaluation is ever attempted), no combination of syntax can reach
attribute access, subscripting, imports, comprehensions, arbitrary function
calls, or any other Python capability outside the five arithmetic operators,
unary +/-, parentheses (which need no dedicated AST node), and ``abs()``.

Expressions reference only short aliases (e.g. "A", "B") declared by the
caller's ``allowed_variables`` -- never source or channel names directly;
resolving an alias to a real channel is the responsibility of a future
phase's binding layer (CalculatedSignalDefinition.variable_bindings), not
this module.
"""
from __future__ import annotations

import ast
import keyword
from dataclasses import dataclass
from typing import Collection, Mapping

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class CalculatedSignalExpressionError(ValueError):
    """Base class for all Calculated Signals expression errors."""


class ExpressionSyntaxError(CalculatedSignalExpressionError):
    """The expression string is not valid Python expression syntax."""


class ExpressionValidationError(CalculatedSignalExpressionError):
    """The expression parsed but uses a construct, function, or variable not
    permitted by the Calculated Signals restricted grammar."""


class ExpressionEvaluationError(CalculatedSignalExpressionError):
    """The expression passed validation but evaluation against the supplied
    values failed (e.g. a missing variable or a NumPy arithmetic error)."""


# ─────────────────────────────────────────────────────────────────────────────
# Structural limits
# ─────────────────────────────────────────────────────────────────────────────
# Conservative defaults intended to block pathological/DoS-style expressions
# without interfering with any expression a Calculated Signals user would
# plausibly write (the V1 concept's entire grammar is five operators, unary
# +/-, parentheses, and abs()).

MAX_EXPRESSION_LENGTH = 1000     # characters
MAX_AST_NODE_COUNT = 256         # total AST nodes visited during validation
MAX_NESTING_DEPTH = 32           # recursive descent depth
MAX_VARIABLE_NAME_LENGTH = 64    # characters per variable name

_ABS_NAME = "abs"
_RESERVED_NAMES = frozenset({_ABS_NAME})

_ALLOWED_BINOP_TYPES = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOP_TYPES = (ast.UAdd, ast.USub)


# ─────────────────────────────────────────────────────────────────────────────
# Validated expression
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ValidatedExpression:
    """A parsed expression that has passed the restricted-grammar check.

    This is the only type ``evaluate_expression`` accepts, so evaluation can
    never run against an unvalidated AST -- there is no way to construct a
    ValidatedExpression except through ``validate_expression``.
    """

    expression: str
    tree: ast.Expression
    referenced_variables: frozenset[str]


def _is_reserved_or_invalid_name(name: str) -> str | None:
    """Return a human-readable rejection reason for *name*, or None if valid."""
    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        return f"variable name too long (> {MAX_VARIABLE_NAME_LENGTH} characters): {name!r}"
    if not name.isidentifier():
        return f"invalid variable name: {name!r}"
    if name.startswith("_"):
        return f"variable names starting with '_' are not allowed: {name!r}"
    if keyword.iskeyword(name):
        return f"{name!r} is a Python keyword and cannot be used as a variable"
    if name in _RESERVED_NAMES:
        return f"{name!r} is reserved for use as a function name and cannot be used as a variable"
    return None


class _NodeBudget:
    """Mutable counter threaded through recursive validation calls."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


def _validate_node(
    node: ast.AST,
    depth: int,
    allowed_variables: frozenset[str],
    budget: _NodeBudget,
    referenced: set[str],
) -> None:
    budget.count += 1
    if budget.count > MAX_AST_NODE_COUNT:
        raise ExpressionValidationError(
            f"expression is too complex (exceeds {MAX_AST_NODE_COUNT} AST nodes)"
        )
    if depth > MAX_NESTING_DEPTH:
        raise ExpressionValidationError(
            f"expression nesting is too deep (exceeds {MAX_NESTING_DEPTH} levels)"
        )

    if isinstance(node, ast.Expression):
        _validate_node(node.body, depth + 1, allowed_variables, budget, referenced)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOP_TYPES):
            raise ExpressionValidationError(
                f"unsupported operator: {type(node.op).__name__}"
            )
        _validate_node(node.left, depth + 1, allowed_variables, budget, referenced)
        _validate_node(node.right, depth + 1, allowed_variables, budget, referenced)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOP_TYPES):
            raise ExpressionValidationError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        _validate_node(node.operand, depth + 1, allowed_variables, budget, referenced)
        return

    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == _ABS_NAME):
            func_desc = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ExpressionValidationError(
                f"unsupported function call: {func_desc!r} (only abs() is allowed)"
            )
        if node.keywords:
            raise ExpressionValidationError("keyword arguments are not allowed")
        if len(node.args) != 1:
            raise ExpressionValidationError("abs() requires exactly one positional argument")
        _validate_node(node.args[0], depth + 1, allowed_variables, budget, referenced)
        return

    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise ExpressionValidationError("only variable reads are allowed")
        reason = _is_reserved_or_invalid_name(node.id)
        if reason is not None:
            raise ExpressionValidationError(reason)
        if node.id not in allowed_variables:
            raise ExpressionValidationError(f"unknown variable: {node.id!r}")
        referenced.add(node.id)
        return

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ExpressionValidationError("boolean literals are not allowed")
        if not isinstance(value, (int, float)):
            raise ExpressionValidationError(
                f"unsupported literal type: {type(value).__name__}"
            )
        return

    raise ExpressionValidationError(f"unsupported syntax: {type(node).__name__}")


def validate_expression(
    expression: str,
    allowed_variables: Collection[str],
) -> ValidatedExpression:
    """Parse and validate *expression* against the restricted grammar.

    Only ``+ - * /``, unary ``+``/``-``, parentheses, ``abs()``, numeric
    literals, and names present in *allowed_variables* are accepted.
    Anything else -- attribute access, subscripting, other function calls,
    imports, comprehensions, comparisons, boolean operators, assignment,
    string/boolean/None/complex literals, etc. -- raises
    ExpressionValidationError. Invalid Python syntax raises
    ExpressionSyntaxError. Neither error path ever calls eval/exec/compile
    on the expression.

    Variables declared in *allowed_variables* but not referenced by the
    expression are permitted (no "unused variable" error) -- a definition
    may bind more channels than a given expression currently needs.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionValidationError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionValidationError(
            f"expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters"
        )

    allowed = frozenset(allowed_variables)
    for name in allowed:
        reason = _is_reserved_or_invalid_name(name) if isinstance(name, str) else f"invalid variable name: {name!r}"
        if reason is not None:
            raise ExpressionValidationError(f"invalid allowed_variables entry: {reason}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionSyntaxError(f"invalid expression syntax: {exc.msg}") from None

    budget = _NodeBudget()
    referenced: set[str] = set()
    _validate_node(tree, 0, allowed, budget, referenced)

    return ValidatedExpression(
        expression=expression, tree=tree, referenced_variables=frozenset(referenced)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_operand(value: object, name: str) -> "np.ndarray | np.floating":
    """Return *value* as a NumPy array or NumPy scalar, or raise.

    Plain Python int/float are converted to np.float64 specifically so that
    division always follows NumPy semantics (inf/nan + RuntimeWarning, never
    ZeroDivisionError) regardless of whether the caller passed a Python
    scalar or a NumPy array -- see the module-level division note below.
    """
    if isinstance(value, bool):
        raise ExpressionEvaluationError(f"variable {name!r} must be numeric, got bool")
    if isinstance(value, (int, float)):
        return np.float64(value)

    arr = np.asarray(value)
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        raise ExpressionEvaluationError(
            f"variable {name!r} must be numeric, got dtype={arr.dtype}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ExpressionEvaluationError(
            f"variable {name!r} must be numeric, got dtype={arr.dtype}"
        )
    return arr


def _evaluate_node(node: ast.AST, env: Mapping[str, "np.ndarray | np.floating"]):
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, env)
        right = _evaluate_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        # ast.Div is the only remaining allowed BinOp type after validation.
        return left / right

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand

    if isinstance(node, ast.Call):
        return np.abs(_evaluate_node(node.args[0], env))

    if isinstance(node, ast.Name):
        return env[node.id]

    # ast.Constant is the only remaining allowed leaf type after validation.
    return np.float64(node.value)


def evaluate_expression(
    validated: ValidatedExpression,
    variables: Mapping[str, "np.ndarray | float | int"],
) -> "np.ndarray | float":
    """Evaluate *validated* against *variables* using vectorized NumPy arithmetic.

    Only the variables the expression actually references
    (``validated.referenced_variables``) are read from *variables* -- extra
    keys are silently ignored, not rejected, so a caller may pass a full
    binding-value map without first filtering it down.

    Every referenced variable must be present in *variables*, or
    ExpressionEvaluationError is raised naming the missing variable(s).
    Values must be a plain int/float or a numeric NumPy array (object and
    string-dtype arrays are rejected); arrays are never mutated -- every
    operator used here (+, -, *, /, np.abs) returns a new array/scalar.

    Division follows plain NumPy semantics: division by zero produces
    +/-inf (or nan for 0/0) plus a NumPy RuntimeWarning, it does not raise.
    Replacing infinities with NaN, or otherwise policing division-by-zero,
    is explicitly deferred to Phase 2B -- this function performs the
    arithmetic exactly as NumPy defines it and nothing more.

    Return contract: if every operand touched by the expression is a plain
    scalar (no array anywhere in the expression), the result is a NumPy
    float64 scalar -- which satisfies Python's `float` protocol
    (`isinstance(result, float)` is True) rather than a 0-dimensional
    ndarray. If any operand is an array, the result is a NumPy ndarray,
    broadcast according to standard NumPy broadcasting rules (mismatched,
    non-broadcastable shapes raise ExpressionEvaluationError).
    """
    missing = validated.referenced_variables - set(variables.keys())
    if missing:
        raise ExpressionEvaluationError(
            f"missing value(s) for variable(s): {sorted(missing)}"
        )

    env: dict[str, "np.ndarray | np.floating"] = {
        name: _coerce_operand(variables[name], name)
        for name in validated.referenced_variables
    }

    try:
        return _evaluate_node(validated.tree.body, env)
    except ExpressionEvaluationError:
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: wrap any
        # NumPy/arithmetic failure (e.g. broadcast shape mismatch) into our
        # own exception type rather than leaking a raw traceback to callers.
        raise ExpressionEvaluationError(f"failed to evaluate expression: {exc}") from None
