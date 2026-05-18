"""Column parameter-type classification enum.

Kept in its own module so it can be imported by both normalization_plan and
models without creating circular dependencies.
"""
from __future__ import annotations

import enum


class ParameterType(enum.Enum):
    """Semantic classification of a data column in a CSV/Excel import."""

    VOLTAGE   = "voltage"
    CURRENT   = "current"
    MW        = "mw"
    MVAR      = "mvar"
    FREQUENCY = "frequency"
    ROCOF     = "rocof"
    DIGITAL   = "digital"
    TIMESTAMP = "timestamp"
    UNKNOWN   = "unknown"
