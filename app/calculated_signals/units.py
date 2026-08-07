"""Minimal unit-family classification and same-family conversion for
Calculated Signals (Phase 2B).

This is deliberately not a dimensional-analysis engine: it only recognises
the specific unit families already in active use elsewhere in Powerwave
(voltage, current, active/reactive power, frequency, ROCOF, dimensionless
per-unit) and only supports converting between two spellings/scales within
the *same* family (e.g. kV -> V). It has no concept of compound/derived
units (e.g. it cannot tell you that V x A = W) -- app.calculated_signals.
engine enforces that "signal x signal" is blocked in V1 rather than asking
this module to derive a new unit.

Canonical spellings mirror app.visualization.engineering_display's existing
defaults ("MW", "MVar", "pu") so a Calculated Signals output unit reads the
same way the rest of the application already displays these quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class UnitFamily(str, Enum):
    """A family of mutually convertible engineering units."""

    VOLTAGE = "voltage"
    CURRENT = "current"
    ACTIVE_POWER = "active_power"
    REACTIVE_POWER = "reactive_power"
    FREQUENCY = "frequency"
    ROCOF = "rocof"
    DIMENSIONLESS = "dimensionless"


# Canonical dimensionless unit produced by same-family signal division (e.g.
# MW / MW) -- matches the spelling already used throughout the codebase for
# per-unit quantities (app.visualization.engineering_display's default
# EngineeringDisplayPreferences.per_unit == "pu").
DIMENSIONLESS_UNIT = "pu"

# Each table maps a lowercased alias -> (canonical spelling, scale_to_base).
# scale_to_base: multiplying a value in this unit by scale_to_base converts
# it to the family's base unit (V, A, W, var, Hz, Hz/s, or pu -- all 1.0).
_VOLTAGE_ALIASES: dict[str, tuple[str, float]] = {
    "v": ("V", 1.0), "volt": ("V", 1.0), "volts": ("V", 1.0),
    "kv": ("kV", 1_000.0),
    "mv": ("mV", 0.001),
}
_CURRENT_ALIASES: dict[str, tuple[str, float]] = {
    "a": ("A", 1.0), "amp": ("A", 1.0), "amps": ("A", 1.0),
    "ampere": ("A", 1.0), "amperes": ("A", 1.0),
    "ka": ("kA", 1_000.0),
    "ma": ("mA", 0.001),
}
_ACTIVE_POWER_ALIASES: dict[str, tuple[str, float]] = {
    "w": ("W", 1.0), "watt": ("W", 1.0), "watts": ("W", 1.0),
    "kw": ("kW", 1_000.0),
    "mw": ("MW", 1_000_000.0),
    "gw": ("GW", 1_000_000_000.0),
}
_REACTIVE_POWER_ALIASES: dict[str, tuple[str, float]] = {
    "var": ("var", 1.0),
    "kvar": ("kVAr", 1_000.0),
    "mvar": ("MVAr", 1_000_000.0),
    "mvarr": ("MVAr", 1_000_000.0),  # tolerated alias, matches engineering_display.py
    "gvar": ("GVAr", 1_000_000_000.0),
}
_FREQUENCY_ALIASES: dict[str, tuple[str, float]] = {
    "hz": ("Hz", 1.0),
}
_ROCOF_ALIASES: dict[str, tuple[str, float]] = {
    "hz/s": ("Hz/s", 1.0),
    "hz/sec": ("Hz/s", 1.0),
    "hzpersecond": ("Hz/s", 1.0),
}
_DIMENSIONLESS_ALIASES: dict[str, tuple[str, float]] = {
    "pu": (DIMENSIONLESS_UNIT, 1.0),
    "p.u.": (DIMENSIONLESS_UNIT, 1.0),
    "per_unit": (DIMENSIONLESS_UNIT, 1.0),
    "perunit": (DIMENSIONLESS_UNIT, 1.0),
    "dimensionless": (DIMENSIONLESS_UNIT, 1.0),
    "1": (DIMENSIONLESS_UNIT, 1.0),
}

_FAMILY_TABLES: dict[UnitFamily, dict[str, tuple[str, float]]] = {
    UnitFamily.VOLTAGE: _VOLTAGE_ALIASES,
    UnitFamily.CURRENT: _CURRENT_ALIASES,
    UnitFamily.ACTIVE_POWER: _ACTIVE_POWER_ALIASES,
    UnitFamily.REACTIVE_POWER: _REACTIVE_POWER_ALIASES,
    UnitFamily.FREQUENCY: _FREQUENCY_ALIASES,
    UnitFamily.ROCOF: _ROCOF_ALIASES,
    UnitFamily.DIMENSIONLESS: _DIMENSIONLESS_ALIASES,
}


@dataclass(frozen=True, slots=True)
class NormalizedUnit:
    """The result of classifying a unit string.

    ``family`` and ``scale_to_base`` are None when the unit is unresolved --
    either because *raw* was None (no unit supplied at all) or because the
    string did not match any known alias. An unresolved unit is treated
    conservatively everywhere in the engine (see engine.py's operator
    rules): it is never assumed compatible with anything, including another
    unresolved unit.
    """

    raw: str | None
    canonical: str | None
    family: UnitFamily | None
    scale_to_base: float | None

    @property
    def is_resolved(self) -> bool:
        return self.family is not None


def normalize_unit(unit: str | None) -> NormalizedUnit:
    """Classify *unit* into a family + canonical spelling, or leave it
    unresolved. Never raises -- an unrecognised string is a legitimate,
    representable outcome (family=None), not an error; callers (the
    engine's operator rules) decide what to do with an unresolved unit.
    """
    if unit is None:
        return NormalizedUnit(raw=None, canonical=None, family=None, scale_to_base=None)

    compact = unit.strip().lower()
    for family, table in _FAMILY_TABLES.items():
        if compact in table:
            canonical, scale = table[compact]
            return NormalizedUnit(raw=unit, canonical=canonical, family=family, scale_to_base=scale)

    return NormalizedUnit(raw=unit, canonical=None, family=None, scale_to_base=None)


def are_compatible_units(a: NormalizedUnit, b: NormalizedUnit) -> bool:
    """True iff both units resolved to the same family.

    Two unresolved units are never considered compatible with each other --
    "unknown" is not itself a family, so it cannot match anything, including
    another unknown.
    """
    return a.is_resolved and b.is_resolved and a.family == b.family


def convert_values(values: np.ndarray | float, from_unit: NormalizedUnit, to_unit: NormalizedUnit) -> np.ndarray:
    """Convert *values* from *from_unit* to *to_unit*.

    Both units must be resolved and belong to the same family, or ValueError
    is raised. Returns a new array/scalar -- *values* is never mutated.
    """
    if not are_compatible_units(from_unit, to_unit):
        raise ValueError(
            f"cannot convert {from_unit.raw!r} to {to_unit.raw!r}: incompatible or unresolved units"
        )
    factor = from_unit.scale_to_base / to_unit.scale_to_base  # type: ignore[operator]
    return np.asarray(values, dtype=np.float64) * factor
