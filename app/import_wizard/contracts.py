"""Validation message contract — base severity and message types.

No dependencies on other import_wizard modules; safe to import from anywhere.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ValidationSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    """Structured validation feedback produced during import wizard processing."""

    severity: ValidationSeverity
    code: str
    message: str
    affected_column: str | None = None
    suggested_action: str | None = None
