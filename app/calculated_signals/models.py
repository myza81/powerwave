"""Core data models for Calculated Signals (Phase 2A).

Defines the durable "what the user asked for" half of a calculated signal
(ChannelRef, CalculatedSignalDefinition) and the "what came out" half
(CalculatedSignalResult, CalculationStatus).

This module has no dependency on EventAnalysisSession, providers, alignment,
units, or plotting -- see the package docstring in __init__.py for the
Phase 2A scope boundary. ChannelRef intentionally carries only
(source_id, channel_name): analog-vs-digital eligibility cannot be checked
here (a bare string pair has no way to know its own channel kind) and must
instead be enforced by whichever future phase resolves this reference
against a live session.

No separate "provenance" model is introduced in this phase: a
CalculatedSignalDefinition's own expression + variable_bindings +
reference_variable already constitute its provenance. Alignment-specific
provenance (offsets applied, overlap window, interpolation percentage, unit
conversions performed) belongs to Phase 2B's result/engine layer, not here.
"""
from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

# Interpolation methods supported in Phase 2A. Extending this set is a
# Phase 2B decision (V1 concept only specifies linear interpolation).
_VALID_INTERPOLATIONS = frozenset({"linear"})

# Reserved: the one function name the expression grammar allows (see
# expression.py). A binding variable may not shadow it.
_RESERVED_FUNCTION_NAMES = frozenset({"abs"})


def _require_non_blank(value: str, field_name: str) -> None:
    """Raise ValueError unless *value* is a string with non-whitespace content.

    Validation only -- the original string is preserved verbatim in the
    owning dataclass field; this helper never trims or otherwise rewrites it.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")


def _require_tz_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


def _is_valid_binding_name(name: str) -> bool:
    """True if *name* is a legal, non-reserved expression variable identifier."""
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not name.startswith("_")
        and not keyword.iskeyword(name)
        and name not in _RESERVED_FUNCTION_NAMES
    )


@dataclass(frozen=True, slots=True)
class ChannelRef:
    """A reference to one analog channel, by session identity.

    Deliberately carries only (source_id, channel_name) -- no file path,
    filename, provider name, station name, or event identity belongs here;
    that information lives in the session layer (SessionSource), not in a
    calculated-signal definition. No digital/analog flag is stored either:
    this object cannot know a channel's kind by itself, so analog-only
    eligibility must be enforced later, when a future phase resolves this
    reference against a live EventAnalysisSession.
    """

    source_id: str
    channel_name: str

    def __post_init__(self) -> None:
        _require_non_blank(self.source_id, "source_id")
        _require_non_blank(self.channel_name, "channel_name")


@dataclass(frozen=True, slots=True)
class CalculatedSignalDefinition:
    """The user's intent for one calculated signal -- durable, independent of
    any computed result.

    Immutable: variable_bindings is defensively snapshotted into a
    MappingProxyType over a private dict at construction time, so mutating
    the caller's original dict afterward cannot change this definition, and
    the definition's own mapping cannot be mutated by callers either.
    """

    calc_id: str
    name: str
    expression: str
    variable_bindings: Mapping[str, ChannelRef]
    reference_variable: str
    output_unit: str | None = None
    interpolation: str = "linear"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _require_non_blank(self.calc_id, "calc_id")
        _require_non_blank(self.name, "name")
        _require_non_blank(self.expression, "expression")

        if not self.variable_bindings:
            raise ValueError("variable_bindings must not be empty")

        # Defensive snapshot: dict keys are already unique by construction
        # (a Python dict cannot hold two entries for the same key), so no
        # separate duplicate-name check is needed -- only per-name validity.
        snapshot: dict[str, ChannelRef] = {}
        for var_name, ref in self.variable_bindings.items():
            if not isinstance(var_name, str) or not var_name.isidentifier():
                raise ValueError(f"invalid binding variable name: {var_name!r}")
            if var_name.startswith("_"):
                raise ValueError(
                    f"binding variable name must not start with '_': {var_name!r}"
                )
            if keyword.iskeyword(var_name):
                raise ValueError(
                    f"binding variable name is a Python keyword: {var_name!r}"
                )
            if var_name in _RESERVED_FUNCTION_NAMES:
                raise ValueError(
                    f"{var_name!r} is reserved for use as a function name and "
                    "cannot be used as a binding variable name"
                )
            if not isinstance(ref, ChannelRef):
                raise ValueError(
                    f"binding {var_name!r} must be a ChannelRef, got {type(ref).__name__}"
                )
            snapshot[var_name] = ref

        object.__setattr__(self, "variable_bindings", MappingProxyType(snapshot))

        if self.reference_variable not in self.variable_bindings:
            raise ValueError(
                f"reference_variable {self.reference_variable!r} is not among "
                f"variable_bindings ({sorted(self.variable_bindings.keys())})"
            )

        if self.interpolation not in _VALID_INTERPOLATIONS:
            raise ValueError(
                f"unsupported interpolation {self.interpolation!r}; only "
                f"{sorted(_VALID_INTERPOLATIONS)} is supported in Phase 2A"
            )

        _require_tz_aware(self.created_at, "created_at")


class CalculationStatus(str, Enum):
    """Lifecycle status of a CalculatedSignalResult."""

    OK = "ok"
    STALE = "stale"
    ERROR = "error"


def _as_1d_numeric_array(value: object, field_name: str) -> np.ndarray:
    """Return a defensive float64 copy of *value*, or raise ValueError."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{field_name} must be a numeric array, got dtype={arr.dtype}")
    return arr.astype(np.float64, copy=True)


def _as_1d_bool_array(value: object, field_name: str) -> np.ndarray:
    """Return a defensive bool copy of *value*, or raise ValueError."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional, got shape={arr.shape}")
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        raise ValueError(
            f"{field_name} must be boolean or safely convertible to boolean, "
            f"got dtype={arr.dtype}"
        )
    return arr.astype(bool, copy=True)


@dataclass(slots=True)
class CalculatedSignalResult:
    """The most recent computed output for one calculated signal.

    Deliberately mutable and separate from CalculatedSignalDefinition: a
    result can be replaced, marked stale, or fail, without touching the
    user's original definition.

    Arrays are always copied and converted to a predictable dtype on
    construction -- time/values to float64, validity_mask to bool -- via
    np.asarray(..., dtype=...).copy() (see _as_1d_numeric_array /
    _as_1d_bool_array above). This object therefore never aliases a
    caller-owned array: callers remain free to mutate their own arrays after
    passing them in without affecting this result, and this result's arrays
    can be mutated by a future owner without affecting the caller's copy.

    No engineering calculation logic (interpolation, alignment, unit
    conversion) belongs in this model -- see Phase 2B.
    """

    calc_id: str
    time: np.ndarray
    values: np.ndarray
    validity_mask: np.ndarray
    unit: str | None
    status: CalculationStatus
    error_message: str | None
    computed_at: datetime
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_non_blank(self.calc_id, "calc_id")

        self.time = _as_1d_numeric_array(self.time, "time")
        self.values = _as_1d_numeric_array(self.values, "values")
        self.validity_mask = _as_1d_bool_array(self.validity_mask, "validity_mask")

        n = len(self.time)
        if len(self.values) != n or len(self.validity_mask) != n:
            raise ValueError(
                "time, values, and validity_mask must have identical length "
                f"(got {len(self.time)}, {len(self.values)}, {len(self.validity_mask)})"
            )

        if not isinstance(self.status, CalculationStatus):
            raise ValueError(
                f"status must be a CalculationStatus, got {type(self.status).__name__}"
            )

        if self.status == CalculationStatus.ERROR:
            if not (self.error_message and self.error_message.strip()):
                raise ValueError("status=ERROR requires a non-empty error_message")

        _require_tz_aware(self.computed_at, "computed_at")

        # Defensive copy: never hold the caller's own list by reference.
        self.warnings = list(self.warnings)
