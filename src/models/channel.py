"""
src/models/channel.py

AnalogueChannel and DigitalChannel dataclasses — the atomic unit of every
disturbance record loaded by PowerWave Analyst.

Architecture: Data layer — imported by parsers/ and engine/ only.
              Never import from ui/ or engine/ here (LAW 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Signal Role Constants ────────────────────────────────────────────────────
# Analogue roles — used as signal_role values on AnalogueChannel

class SignalRole:
    """String constants for analogue and digital signal roles."""

    # Analogue
    V_PHASE     = "V_PHASE"      # Phase-to-earth voltage          kV   A/B/C
    V_LINE      = "V_LINE"       # Phase-to-phase voltage          kV   AB/BC/CA
    V_RESIDUAL  = "V_RESIDUAL"   # Residual / zero-seq voltage     kV   N
    I_PHASE     = "I_PHASE"      # Phase current                   A/kA A/B/C
    I_EARTH     = "I_EARTH"      # Earth / neutral / residual cur  A/kA N
    V1_PMU      = "V1_PMU"       # Positive-seq voltage (PMU)      kV   Pos-seq
    I1_PMU      = "I1_PMU"       # Positive-seq current (PMU)      A/kA Pos-seq
    P_MW        = "P_MW"         # Active power (pre-calc or PMU)  MW   3-phase
    Q_MVAR      = "Q_MVAR"       # Reactive power                  MVAR 3-phase
    FREQ        = "FREQ"         # System frequency                Hz
    ROCOF       = "ROCOF"        # Rate of change of frequency     Hz/s
    DC_FIELD_I  = "DC_FIELD_I"   # Generator field current (DC)   A    non-zero offset
    DC_FIELD_V  = "DC_FIELD_V"   # Generator field voltage (DC)   V    non-zero offset
    MECH_SPEED  = "MECH_SPEED"   # Mechanical shaft speed          RPM  non-zero offset
    MECH_VALVE  = "MECH_VALVE"   # Valve / gate / governor pos     %    non-zero offset
    SEQ_RMS     = "SEQ_RMS"      # Sequence component RMS          kV   pos/neg/zero
    ANALOGUE    = "ANALOGUE"     # Generic — unknown or other

    # Digital
    DIG_TRIP      = "DIG_TRIP"      # Protection trip output
    DIG_CB        = "DIG_CB"        # Circuit breaker / switch status
    DIG_PICKUP    = "DIG_PICKUP"    # Protection element pickup / alarm
    DIG_AR        = "DIG_AR"        # Auto-reclose
    DIG_INTERTRIP = "DIG_INTERTRIP" # Teleprotection / breaker-fail / inter-trip
    DIG_TRIGGER   = "DIG_TRIGGER"   # External trigger input
    DIG_GENERIC   = "DIG_GENERIC"   # Alarm, supervision, comms-fail, enable, MCB

    # All analogue roles as a frozenset for membership tests
    ANALOGUE_ROLES: frozenset[str] = frozenset({
        V_PHASE, V_LINE, V_RESIDUAL, I_PHASE, I_EARTH,
        V1_PMU, I1_PMU, P_MW, Q_MVAR, FREQ, ROCOF,
        DC_FIELD_I, DC_FIELD_V, MECH_SPEED, MECH_VALVE,
        SEQ_RMS, ANALOGUE,
    })

    # All digital roles as a frozenset
    DIGITAL_ROLES: frozenset[str] = frozenset({
        DIG_TRIP, DIG_CB, DIG_PICKUP, DIG_AR,
        DIG_INTERTRIP, DIG_TRIGGER, DIG_GENERIC,
    })


class RoleConfidence:
    """Detection confidence levels assigned by signal_role_detector."""
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ── Colour Map ───────────────────────────────────────────────────────────────
# Keyed by phase letter or signal role string.
# Used to set AnalogueChannel.colour and DigitalChannel.colour at parse time.

COLOUR_MAP: dict[str, str] = {
    # Phase colours
    "A":   "#FF4444",   # Phase A — red
    "B":   "#FFCC00",   # Phase B — yellow
    "C":   "#4488FF",   # Phase C — blue
    "N":   "#44BB44",   # Earth / neutral — green
    # Analogue role overrides (take precedence over phase when no phase context)
    SignalRole.FREQ:       "#00DDDD",
    SignalRole.ROCOF:      "#00DDDD",
    SignalRole.P_MW:       "#FFAA44",
    SignalRole.Q_MVAR:     "#AA44FF",
    SignalRole.DC_FIELD_I: "#AAAAAA",
    SignalRole.DC_FIELD_V: "#AAAAAA",
    SignalRole.MECH_SPEED: "#AAAAAA",
    SignalRole.MECH_VALVE: "#AAAAAA",
    # Digital role colours
    SignalRole.DIG_TRIP:      "#FF2222",
    SignalRole.DIG_CB:        "#FF8800",
    SignalRole.DIG_PICKUP:    "#FFAA00",
    SignalRole.DIG_AR:        "#44AAFF",
    SignalRole.DIG_INTERTRIP: "#FF44FF",
    SignalRole.DIG_TRIGGER:   "#FFFFFF",
    SignalRole.DIG_GENERIC:   "#888888",
}

# Fallback colour when role/phase is unknown
_COLOUR_FALLBACK = "#CCCCCC"


def default_colour_for(signal_role: str, phase: str = "") -> str:
    """Return the display colour for a channel given its role and phase.

    Phase takes priority for voltage/current analogue channels so that
    Va/Vb/Vc each get their phase colour. Role-level overrides apply
    for FREQ, power, DC/mechanical, and all digital channels.

    Args:
        signal_role: One of the SignalRole constants.
        phase:       Phase letter (A, B, C, N, AB, BC, CA, etc.) or empty string.

    Returns:
        Hex colour string, e.g. "#FF4444".
    """
    # Role-level override takes priority for specific roles
    if signal_role in COLOUR_MAP:
        return COLOUR_MAP[signal_role]

    # Phase-based colour for V/I analogue channels
    if phase:
        phase_key = phase[0].upper() if phase else ""
        if phase_key in COLOUR_MAP:
            return COLOUR_MAP[phase_key]

    return _COLOUR_FALLBACK


# ── AnalogueChannel ──────────────────────────────────────────────────────────

@dataclass
class AnalogueChannel:
    """One analogue channel inside a DisturbanceRecord.

    ``raw_data`` holds float32 samples **after** applying the COMTRADE
    (multiplier × raw_int + offset) scaling.  Use the ``physical_data``
    property when you need the engineering-unit values — it re-applies
    the stored multiplier and offset for transparency, but in practice
    raw_data is already scaled so multiplier=1.0 / offset=0.0 is the
    post-parse default unless the parser explicitly stores unscaled ints.

    LAW 10: physical = (raw × multiplier) + offset.
    Offset MUST always be applied — it may be non-zero for DC_FIELD_I/V,
    MECH_SPEED, MECH_VALVE, and pre-calculated P/Q channels.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    channel_id: int
    name: str
    phase: str                    # A/B/C/N/AB/BC/CA/Pos-seq/3-phase or ""
    unit: str                     # kV / A / kA / MW / MVAR / Hz / RPM / % / ""

    # ── CT/VT Scaling (from COMTRADE CFG) ────────────────────────────────
    multiplier: float = 1.0       # 'a' coefficient in CFG
    offset: float = 0.0           # 'b' coefficient in CFG — may be non-zero
    primary: Optional[float] = None    # primary rated value; None if absent in CFG
    secondary: Optional[float] = None  # secondary rated value; None if absent in CFG
    ps_flag: str = "S"            # 'P' = primary values stored, 'S' = secondary

    # ── Sample Data ───────────────────────────────────────────────────────
    raw_data: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    # float32 — samples after CFG scaling; physical_data applies offset on top

    # ── Signal Role Detection ─────────────────────────────────────────────
    signal_role: str = SignalRole.ANALOGUE
    role_confidence: str = RoleConfidence.LOW
    role_confirmed: bool = False  # True once engineer confirms via mapping dialog

    # ── Display ───────────────────────────────────────────────────────────
    colour: str = _COLOUR_FALLBACK
    visible: bool = True

    # ── Grouping ──────────────────────────────────────────────────────────
    bay_name: str = ""            # extracted from channel name (BEN32 multi-bay)

    # ── Derived-value Flag ────────────────────────────────────────────────
    is_derived: bool = False      # True for pre-calculated MW/MVAr in BEN32 slow record

    def __post_init__(self) -> None:
        """Validate types and set colour from role/phase if not already set."""
        if not isinstance(self.raw_data, np.ndarray):
            self.raw_data = np.asarray(self.raw_data, dtype=np.float32)
        elif self.raw_data.dtype != np.float32:
            self.raw_data = self.raw_data.astype(np.float32)

        # Auto-assign colour if caller left it at fallback
        if self.colour == _COLOUR_FALLBACK:
            self.colour = default_colour_for(self.signal_role, self.phase)

    @property
    def physical_data(self) -> np.ndarray:
        """Return engineering-unit values: (raw_data × multiplier) + offset.

        LAW 10 compliance: offset is always applied, even when zero.
        Returns float64 array for maximum precision in calculations.
        """
        return self.raw_data.astype(np.float64) * self.multiplier + self.offset

    @property
    def n_samples(self) -> int:
        """Number of samples in this channel."""
        return len(self.raw_data)


# ── DigitalChannel ───────────────────────────────────────────────────────────

@dataclass
class DigitalChannel:
    """One digital (binary) channel inside a DisturbanceRecord.

    Samples are stored as uint8 (0 or 1).  ``normal_state`` is the
    quiescent state from the COMTRADE CFG; an XOR with normal_state
    gives the 'changed' mask used for event detection.

    Complementary pairs (GCB OPEN + GCB CLOSED) are detected by the
    parser and linked via ``complementary_channel_id``.  The UI renders
    a complementary pair as a single state bar.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    channel_id: int
    name: str
    phase: str = ""               # phase association if known, else ""

    # ── COMTRADE CFG Fields ────────────────────────────────────────────────
    normal_state: int = 0         # 0 or 1 — quiescent state from CFG
    ccbm: str = ""                # circuit-breaker contact multiplier (BEN32 5-field)

    # ── Sample Data ───────────────────────────────────────────────────────
    data: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )

    # ── Signal Role Detection ─────────────────────────────────────────────
    signal_role: str = SignalRole.DIG_GENERIC
    role_confirmed: bool = False

    # ── Display ───────────────────────────────────────────────────────────
    colour: str = _COLOUR_FALLBACK
    visible: bool = True

    # ── Grouping ──────────────────────────────────────────────────────────
    bay_name: str = ""

    # ── Complementary Pair Tracking ───────────────────────────────────────
    is_complementary: bool = False
    complementary_channel_id: Optional[int] = None
    # When True, this channel is the PRIMARY of the pair (the OPEN side).
    # The UI renders this pair as a single merged state bar.
    is_primary_of_pair: bool = False

    def __post_init__(self) -> None:
        """Validate data dtype and auto-assign colour from role."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data, dtype=np.uint8)
        elif self.data.dtype != np.uint8:
            self.data = self.data.astype(np.uint8)

        if self.colour == _COLOUR_FALLBACK:
            self.colour = COLOUR_MAP.get(self.signal_role, _COLOUR_FALLBACK)

    @property
    def n_samples(self) -> int:
        """Number of samples in this channel."""
        return len(self.data)

    @property
    def active_mask(self) -> np.ndarray:
        """Boolean mask: True where channel is in the non-normal (active) state.

        XORs the raw data with normal_state so that channels with
        normal_state=1 (normally closed) are correctly interpreted.
        """
        return (self.data ^ self.normal_state).astype(bool)
