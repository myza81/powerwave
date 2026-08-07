"""Calculated Signals -- core data models, a safe expression engine (Phase
2A), and a numerical calculation engine (Phase 2B).

Phase 2A defines the durable "definition" and "result" models for a
user-created calculated analog signal, and a restricted-AST expression
parser/evaluator that never calls Python's eval(), exec(), or compile() on
user input.

Phase 2B adds ResolvedAnalogInput (a neutral, session-independent analog
waveform input), unit-family classification/conversion, and
calculate_signal() -- a pipeline that aligns inputs on a reference time
base, interpolates linearly with no extrapolation, blocks interpolation
across internal data gaps, enforces V1 unit-compatibility rules, and
produces a CalculatedSignalResult with a validity mask and factual warnings.

Neither phase has any dependency on EventAnalysisSession, providers, the
UI, Session Canvas, or persistence -- those integrations belong to later
phases. Neither phase accepts or processes digital channels in any form.

Phase 2C-3 adds CalculatedSignalResolutionService, the external service that
resolves a calculated signal's dependencies against a live
EventAnalysisSession (never through the display-oriented
build_aligned_data(), which clips and decimates) and invokes
calculate_signal() to (re)compute it. EventAnalysisSession itself never
calls calculate_signal(); this service is the only bridge between the two.

Public API:
  ChannelRef                  -- (source_id, channel_name) reference to one analog channel
  CalculatedSignalDefinition  -- immutable "what the user asked for"
  CalculatedSignalResult      -- mutable "what came out"
  CalculationStatus           -- OK / STALE / ERROR
  ResolvedAnalogInput         -- an already-aligned analog waveform, ready for calculate_signal()

  ValidatedExpression         -- a parsed, restricted-grammar expression
  validate_expression         -- parse + validate an expression string
  evaluate_expression         -- evaluate a ValidatedExpression against bound values (no units)

  CalculatedSignalExpressionError, ExpressionSyntaxError,
  ExpressionValidationError, ExpressionEvaluationError

  MAX_EXPRESSION_LENGTH, MAX_AST_NODE_COUNT, MAX_NESTING_DEPTH,
  MAX_VARIABLE_NAME_LENGTH  -- structural limits enforced during validation

  UnitFamily                  -- voltage / current / active_power / reactive_power / frequency / rocof / dimensionless
  NormalizedUnit               -- a classified unit (family + canonical spelling + scale)
  normalize_unit               -- classify a unit string
  are_compatible_units         -- same-family check
  convert_values                -- same-family value conversion
  DIMENSIONLESS_UNIT           -- canonical dimensionless unit string ("pu")

  CalculationEngineConfig      -- named V1 heuristic thresholds
  calculate_signal             -- the Phase 2B calculation pipeline

  CalculatedSignalEngineError, InputValidationError, AlignmentError,
  UnitCompatibilityError, CalculationError

  CalculatedSignalResolutionService  -- resolves a session's calculated signals and recalculates them
  ResolutionBatchResult              -- successful results + failures from a batch resolve
  ResolutionFailure                  -- one calc_id's resolution/calculation failure
  CalculatedSignalResolutionError    -- raised by the resolution service on an unresolvable/failed signal
"""
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
    ResolvedAnalogInput,
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
from app.calculated_signals.units import (
    DIMENSIONLESS_UNIT,
    NormalizedUnit,
    UnitFamily,
    are_compatible_units,
    convert_values,
    normalize_unit,
)
from app.calculated_signals.engine import (
    AlignmentError,
    CalculatedSignalEngineError,
    CalculationEngineConfig,
    CalculationError,
    InputValidationError,
    UnitCompatibilityError,
    calculate_signal,
)
from app.calculated_signals.resolver import (
    CalculatedSignalResolutionError,
    CalculatedSignalResolutionService,
    ResolutionBatchResult,
    ResolutionFailure,
)

__all__ = [
    "ChannelRef",
    "CalculatedSignalDefinition",
    "CalculatedSignalResult",
    "CalculationStatus",
    "ResolvedAnalogInput",
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
    "UnitFamily",
    "NormalizedUnit",
    "normalize_unit",
    "are_compatible_units",
    "convert_values",
    "DIMENSIONLESS_UNIT",
    "CalculationEngineConfig",
    "calculate_signal",
    "CalculatedSignalEngineError",
    "InputValidationError",
    "AlignmentError",
    "UnitCompatibilityError",
    "CalculationError",
    "CalculatedSignalResolutionService",
    "ResolutionBatchResult",
    "ResolutionFailure",
    "CalculatedSignalResolutionError",
]
