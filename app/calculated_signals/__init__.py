"""Calculated Signals -- Phase 2A: core data models and a safe expression engine.

This phase defines the durable "definition" and "result" models for a
user-created calculated analog signal, and a restricted-AST expression
parser/evaluator that never calls Python's eval(), exec(), or compile() on
user input. It has no dependency on EventAnalysisSession, providers,
alignment, unit conversion, or plotting -- those integrations belong to
later phases.

Public API:
  ChannelRef                  -- (source_id, channel_name) reference to one analog channel
  CalculatedSignalDefinition  -- immutable "what the user asked for"
  CalculatedSignalResult      -- mutable "what came out"
  CalculationStatus           -- OK / STALE / ERROR

  ValidatedExpression         -- a parsed, restricted-grammar expression
  validate_expression         -- parse + validate an expression string
  evaluate_expression         -- evaluate a ValidatedExpression against bound values

  CalculatedSignalExpressionError, ExpressionSyntaxError,
  ExpressionValidationError, ExpressionEvaluationError

  MAX_EXPRESSION_LENGTH, MAX_AST_NODE_COUNT, MAX_NESTING_DEPTH,
  MAX_VARIABLE_NAME_LENGTH  -- structural limits enforced during validation
"""
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)
from app.calculated_signals.expression import (
    MAX_AST_NODE_COUNT,
    MAX_EXPRESSION_LENGTH,
    MAX_NESTING_DEPTH,
    MAX_VARIABLE_NAME_LENGTH,
    CalculatedSignalExpressionError,
    ExpressionEvaluationError,
    ExpressionSyntaxError,
    ExpressionValidationError,
    ValidatedExpression,
    evaluate_expression,
    validate_expression,
)

__all__ = [
    "ChannelRef",
    "CalculatedSignalDefinition",
    "CalculatedSignalResult",
    "CalculationStatus",
    "ValidatedExpression",
    "validate_expression",
    "evaluate_expression",
    "CalculatedSignalExpressionError",
    "ExpressionSyntaxError",
    "ExpressionValidationError",
    "ExpressionEvaluationError",
    "MAX_EXPRESSION_LENGTH",
    "MAX_AST_NODE_COUNT",
    "MAX_NESTING_DEPTH",
    "MAX_VARIABLE_NAME_LENGTH",
]
