"""Phasor channel classification and phase identification.

Classification priority chain (same philosophy as frequency_overlay.py):
  1. operator force_role override
  2. SignalMetadata.measurement_kind  ("voltage_phasor" / "current_phasor")
  3. SignalMetadata.electrical_type   ("voltage" / "current")
  4. Unit heuristics                  (kV / V → voltage; A / kA → current)
  5. Name heuristics                  (V-prefix → voltage; I-prefix → current)
  6. Default: UNKNOWN (conservative)

Phase identification uses a separate chain:
  1. AnalogChannel.phase field (authoritative if set)
  2. SignalMetadata.phase_reference  (operator-set)
  3. Name suffix heuristics (A/B/C, R/Y/B)
  4. Default: PhaseLabel.UNKNOWN
"""
from __future__ import annotations

from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorChannelResult,
    PhasorChannelRole,
)
from app.data.signal_metadata import SignalMetadata

# ─────────────────────────────────────────────────────────────────────────────
# Classification tables
# ─────────────────────────────────────────────────────────────────────────────

# measurement_kind → PhasorChannelRole
_MEASUREMENT_KIND_MAP: dict[str, PhasorChannelRole] = {
    "voltage_phasor": PhasorChannelRole.VOLTAGE_PHASOR,
    "voltage":        PhasorChannelRole.VOLTAGE_PHASOR,
    "current_phasor": PhasorChannelRole.CURRENT_PHASOR,
    "current":        PhasorChannelRole.CURRENT_PHASOR,
}

# electrical_type → PhasorChannelRole
_ELECTRICAL_TYPE_MAP: dict[str, PhasorChannelRole] = {
    "voltage": PhasorChannelRole.VOLTAGE_PHASOR,
    "current": PhasorChannelRole.CURRENT_PHASOR,
}

# Unit strings that unambiguously indicate voltage (lower-cased)
_VOLTAGE_UNITS = frozenset(["kv", "v", "volts", "volt"])

# Unit strings that unambiguously indicate current (lower-cased)
_CURRENT_UNITS = frozenset(["a", "ka", "amps", "amp"])

# Name fragments that identify voltage channels (lower-cased, checked after current)
_VOLTAGE_NAME_FRAGMENTS = (
    "voltage", "volt",
)
# Short voltage name prefixes: channel starts with V (case-insensitive)
_VOLTAGE_PREFIXES = ("v",)

# Name fragments that identify current channels (lower-cased)
_CURRENT_NAME_FRAGMENTS = (
    "current",
)
# Short current name prefixes: channel starts with I (case-insensitive)
_CURRENT_PREFIXES = ("i",)

# Display units by role
_DISPLAY_UNITS: dict[PhasorChannelRole, str] = {
    PhasorChannelRole.VOLTAGE_PHASOR: "kV",
    PhasorChannelRole.CURRENT_PHASOR: "A",
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase identification tables
# ─────────────────────────────────────────────────────────────────────────────

# Suffixes that identify Phase A — 2-char (check first)
_PHASE_A_SUFFIX2 = frozenset(["an", "ag"])
# Suffixes that identify Phase B — 2-char
_PHASE_B_SUFFIX2 = frozenset(["bn", "bg"])
# Suffixes that identify Phase C — 2-char
_PHASE_C_SUFFIX2 = frozenset(["cn", "cg"])

# Unambiguous 1-char ABC/numeric phase suffixes (match on any channel name)
_PHASE_A_SUFFIX1 = frozenset(["a", "1"])
_PHASE_B_SUFFIX1 = frozenset(["b", "2"])
_PHASE_C_SUFFIX1 = frozenset(["c", "3"])

# RYB 1-char suffixes: only match when channel name is short (≤4 chars)
# or the suffix is preceded by a separator. "Channel_XY" → "y" is NOT a phase.
_PHASE_A_RYB = frozenset(["r"])
_PHASE_B_RYB = frozenset(["y"])

# Name fragments that identify phase (after stripping signal-type prefix)
_PHASE_A_FRAGMENTS = frozenset(["_a", "-a", "pha", "phasea", "phase_a", "red"])
_PHASE_B_FRAGMENTS = frozenset(["_b", "-b", "phb", "phaseb", "phase_b", "yellow", "yel"])
_PHASE_C_FRAGMENTS = frozenset(["_c", "-c", "phc", "phasec", "phase_c", "blue"])


# ─────────────────────────────────────────────────────────────────────────────
# Public classification API
# ─────────────────────────────────────────────────────────────────────────────


def classify_phasor_role(
    channel_name: str,
    unit: str | None = None,
    signal_meta: SignalMetadata | None = None,
    *,
    force_role: PhasorChannelRole | None = None,
) -> PhasorChannelResult:
    """Classify one channel as VOLTAGE_PHASOR, CURRENT_PHASOR, or UNKNOWN.

    Args:
        channel_name: Raw channel name string.
        unit:         Engineering unit string, or None.
        signal_meta:  Optional SignalMetadata for authoritative overrides.
        force_role:   Operator override; always wins when not None.

    Returns:
        Immutable PhasorChannelResult.
    """
    # 1. Operator override
    if force_role is not None:
        return PhasorChannelResult(
            role=force_role,
            phase=_identify_phase(channel_name, signal_meta),
            reason="operator_override",
            auto_classified=False,
            display_unit=_DISPLAY_UNITS.get(force_role),
        )

    # 2. measurement_kind metadata
    if signal_meta is not None and signal_meta.measurement_kind:
        role = _MEASUREMENT_KIND_MAP.get(signal_meta.measurement_kind.lower())
        if role is not None:
            return PhasorChannelResult(
                role=role,
                phase=_identify_phase(channel_name, signal_meta),
                reason=f"measurement_kind:{signal_meta.measurement_kind}",
                auto_classified=False,
                display_unit=_DISPLAY_UNITS.get(role),
            )

    # 3. electrical_type metadata
    if signal_meta is not None and signal_meta.electrical_type:
        role = _ELECTRICAL_TYPE_MAP.get(signal_meta.electrical_type.lower())
        if role is not None:
            return PhasorChannelResult(
                role=role,
                phase=_identify_phase(channel_name, signal_meta),
                reason=f"electrical_type:{signal_meta.electrical_type}",
                auto_classified=True,
                display_unit=_DISPLAY_UNITS.get(role),
            )

    # 4. Unit heuristics
    if unit:
        u = unit.strip().lower()
        if u in _VOLTAGE_UNITS:
            return PhasorChannelResult(
                role=PhasorChannelRole.VOLTAGE_PHASOR,
                phase=_identify_phase(channel_name, signal_meta),
                reason=f"unit:{unit}",
                auto_classified=True,
                display_unit="kV",
            )
        if u in _CURRENT_UNITS:
            return PhasorChannelResult(
                role=PhasorChannelRole.CURRENT_PHASOR,
                phase=_identify_phase(channel_name, signal_meta),
                reason=f"unit:{unit}",
                auto_classified=True,
                display_unit="A",
            )

    # 5. Name heuristics
    role = _classify_by_name(channel_name)
    if role is not None:
        return PhasorChannelResult(
            role=role,
            phase=_identify_phase(channel_name, signal_meta),
            reason=f"name:{channel_name}",
            auto_classified=True,
            display_unit=_DISPLAY_UNITS.get(role),
        )

    # 6. Default: UNKNOWN (conservative)
    return PhasorChannelResult(
        role=PhasorChannelRole.UNKNOWN,
        phase=PhaseLabel.UNKNOWN,
        reason="no_match",
        auto_classified=True,
        display_unit=None,
    )


def identify_phase(
    channel_name: str,
    signal_meta: SignalMetadata | None = None,
    channel_phase: str | None = None,
    *,
    force_phase: PhaseLabel | None = None,
) -> PhaseLabel:
    """Public API for phase identification with operator override support.

    Args:
        channel_name:  Raw channel name string.
        signal_meta:   Optional SignalMetadata.
        channel_phase: AnalogChannel.phase field value (highest priority after force).
        force_phase:   Operator override; always wins when not None.

    Returns:
        PhaseLabel.
    """
    if force_phase is not None:
        return force_phase

    # AnalogChannel.phase field takes highest non-override priority
    if channel_phase:
        mapped = _map_phase_string(channel_phase)
        if mapped != PhaseLabel.UNKNOWN:
            return mapped

    return _identify_phase(channel_name, signal_meta)


def is_voltage_channel(
    channel_name: str,
    unit: str | None = None,
    signal_meta: SignalMetadata | None = None,
) -> bool:
    """Return True if the channel is classified as a voltage phasor."""
    return (
        classify_phasor_role(channel_name, unit, signal_meta).role
        == PhasorChannelRole.VOLTAGE_PHASOR
    )


def is_current_channel(
    channel_name: str,
    unit: str | None = None,
    signal_meta: SignalMetadata | None = None,
) -> bool:
    """Return True if the channel is classified as a current phasor."""
    return (
        classify_phasor_role(channel_name, unit, signal_meta).role
        == PhasorChannelRole.CURRENT_PHASOR
    )


# ─────────────────────────────────────────────────────────────────────────────
# Three-phase group detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_three_phase_groups(
    channel_names: list[str],
    signal_metadata: dict[str, SignalMetadata] | None = None,
    channel_phases: dict[str, str] | None = None,
) -> list[dict[str, str | None]]:
    """Identify three-phase A/B/C groups among a list of channel names.

    Groups channels that share a common prefix and differ only in phase suffix.
    Returns a list of dicts, each with keys: 'a', 'b', 'c', 'signal_type', 'prefix'.
    Only complete groups (all three phases identified) are returned.

    Args:
        channel_names:   All analog channel names in the record.
        signal_metadata: Optional per-channel SignalMetadata for phase hints.
        channel_phases:  Optional AnalogChannel.phase values keyed by name.

    Returns:
        List of three-phase group dicts.
    """
    meta = signal_metadata or {}
    phases_map = channel_phases or {}

    # Build {channel_name: PhaseLabel} for all channels
    phase_by_channel: dict[str, PhaseLabel] = {}
    role_by_channel: dict[str, PhasorChannelRole] = {}

    for name in channel_names:
        ch_phase = phases_map.get(name)
        ch_meta = meta.get(name)
        result = classify_phasor_role(name, signal_meta=ch_meta)
        role_by_channel[name] = result.role
        phase_by_channel[name] = identify_phase(
            name, signal_meta=ch_meta, channel_phase=ch_phase
        )

    # Group channels by (signal_type, detected_prefix)
    # Prefix = channel name with phase suffix stripped
    groups: dict[tuple[str, str], dict[PhaseLabel, str]] = {}
    for name in channel_names:
        role = role_by_channel[name]
        if role == PhasorChannelRole.UNKNOWN:
            continue
        sig_type = "voltage" if role == PhasorChannelRole.VOLTAGE_PHASOR else "current"
        phase = phase_by_channel[name]
        if phase == PhaseLabel.UNKNOWN:
            continue
        prefix = _strip_phase_suffix(name)
        key = (sig_type, prefix)
        if key not in groups:
            groups[key] = {}
        groups[key][phase] = name

    result_groups = []
    for (sig_type, prefix), phase_map in groups.items():
        if (
            PhaseLabel.A in phase_map
            and PhaseLabel.B in phase_map
            and PhaseLabel.C in phase_map
        ):
            result_groups.append({
                "a": phase_map[PhaseLabel.A],
                "b": phase_map[PhaseLabel.B],
                "c": phase_map[PhaseLabel.C],
                "signal_type": sig_type,
                "prefix": prefix,
            })

    return result_groups


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _classify_by_name(name: str) -> PhasorChannelRole | None:
    """Name-based heuristic. Returns None if no match."""
    lower = name.lower().strip()

    # Explicit fragment matches first
    for frag in _CURRENT_NAME_FRAGMENTS:
        if frag in lower:
            return PhasorChannelRole.CURRENT_PHASOR

    for frag in _VOLTAGE_NAME_FRAGMENTS:
        if frag in lower:
            return PhasorChannelRole.VOLTAGE_PHASOR

    # Short prefix checks: "I..." = current, "V..." = voltage
    # Must be at least 2 chars to avoid single-letter false positives
    if len(name) >= 2:
        first = name[0].lower()
        if first == "i":
            return PhasorChannelRole.CURRENT_PHASOR
        if first == "v":
            return PhasorChannelRole.VOLTAGE_PHASOR

    return None


def _identify_phase(
    channel_name: str,
    signal_meta: SignalMetadata | None,
) -> PhaseLabel:
    """Internal phase identification from metadata + name heuristics."""
    # phase_reference in SignalMetadata
    if signal_meta is not None:
        ref = getattr(signal_meta, "phase_reference", None)
        if ref:
            mapped = _map_phase_string(str(ref))
            if mapped != PhaseLabel.UNKNOWN:
                return mapped

    return _phase_from_name(channel_name)


def _phase_from_name(name: str) -> PhaseLabel:
    """Extract phase label from channel name using suffix/fragment heuristics."""
    lower = name.lower().strip()

    # Fragment checks (e.g. "phase_a", "red")
    for frag in _PHASE_A_FRAGMENTS:
        if frag in lower:
            return PhaseLabel.A
    for frag in _PHASE_B_FRAGMENTS:
        if frag in lower:
            return PhaseLabel.B
    for frag in _PHASE_C_FRAGMENTS:
        if frag in lower:
            return PhaseLabel.C

    # 2-char suffix checks (higher specificity — try first)
    if len(lower) >= 2:
        suffix2 = lower[-2:]
        if suffix2 in _PHASE_A_SUFFIX2:
            return PhaseLabel.A
        if suffix2 in _PHASE_B_SUFFIX2:
            return PhaseLabel.B
        if suffix2 in _PHASE_C_SUFFIX2:
            return PhaseLabel.C

    # 1-char ABC/numeric suffixes — match universally
    if len(lower) >= 1:
        suffix1 = lower[-1]
        if suffix1 in _PHASE_A_SUFFIX1:
            return PhaseLabel.A
        if suffix1 in _PHASE_B_SUFFIX1:
            return PhaseLabel.B
        if suffix1 in _PHASE_C_SUFFIX1:
            return PhaseLabel.C

        # RYB suffixes (r/y): only accept for short names (≤4 chars)
        # or when preceded by a separator. Prevents "Channel_XY" → Phase B.
        _sep = {"_", "-", " ", "/", "."}
        _ryb_ok = len(lower) <= 4 or (len(lower) >= 2 and lower[-2] in _sep)
        if _ryb_ok:
            if suffix1 in _PHASE_A_RYB:
                return PhaseLabel.A
            if suffix1 in _PHASE_B_RYB:
                return PhaseLabel.B

    return PhaseLabel.UNKNOWN


def _map_phase_string(phase_str: str) -> PhaseLabel:
    """Map a raw phase string ('A', 'B', 'C', 'R', 'Y', etc.) to PhaseLabel."""
    s = phase_str.strip().upper()
    if s in ("A", "R", "RED", "1"):
        return PhaseLabel.A
    if s in ("B", "Y", "YELLOW", "YEL", "2"):
        return PhaseLabel.B
    if s in ("C", "BLUE", "BLU", "3"):
        return PhaseLabel.C
    return PhaseLabel.UNKNOWN


def _strip_phase_suffix(name: str) -> str:
    """Remove trailing phase suffix from a channel name to obtain a prefix.

    Examples: "VA" → "V", "IA" → "I", "VAN" → "V", "V_A" → "V_"
    """
    lower = name.lower()

    # Remove 2-char phase suffixes
    for sfx in ("an", "bn", "cn", "ag", "bg", "cg"):
        if lower.endswith(sfx) and len(name) > len(sfx):
            return name[: -len(sfx)]

    # Remove 1-char phase suffix (a/b/c/r/y/1/2/3)
    all_phase_chars = {"a", "b", "c", "r", "y", "1", "2", "3"}
    if lower and lower[-1] in all_phase_chars and len(name) > 1:
        return name[:-1]

    return name
