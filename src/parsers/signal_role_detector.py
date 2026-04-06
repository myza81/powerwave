"""
src/parsers/signal_role_detector.py

Signal role auto-detection for AnalogueChannel and DigitalChannel objects.
Runs after CFG+DAT parsing is complete (raw_data is populated).

Assigns signal_role, phase, role_confidence, and colour to every channel
in-place.  Also detects is_derived (pre-calculated P/Q) and complementary
DIG_CB pairs (GCB OPEN + GCB CLOSED).

Architecture: Data layer (parsers/) — imports models/ only.
              Never import from ui/ or engine/ here (LAW 1).

Detection priority (analogue, first match wins):
  1. Unit field — most reliable structural signal
  2. Non-zero offset signature — DC/mechanical channels
  3. Power channel name patterns
  4. Frequency channel name patterns
  5. Sequence RMS name patterns
  6. Direct signal code lookup (case-sensitive then upper-cased)
  7. Regex pattern matching on full channel name
  8. Magnitude heuristic from raw_data
  Fallback: ANALOGUE, LOW

Detection order (digital):
  ALARM EXCEPTION first (overrides all), then CB → AR → INTERTRIP →
  TRIGGER → TRIP → PICKUP → DIG_GENERIC fallback.
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np

from models.channel import (
    AnalogueChannel,
    DigitalChannel,
    RoleConfidence,
    SignalRole,
    default_colour_for,
)

# ── Analogue: direct signal-code lookup table ─────────────────────────────────
# Maps signal code string → (role, phase).
# Case-sensitive entries are checked first so Chinese lowercase (Ia, Ub …)
# match before the upper-cased fallback sweep.

_SIGNAL_CODE_TO_ROLE: dict[str, tuple[str, str]] = {
    # ── R/Y/B convention (BEN32 / Malaysian / TNB) ────────────────────────
    'VR': (SignalRole.V_PHASE,    'A'),
    'VY': (SignalRole.V_PHASE,    'B'),
    'VB': (SignalRole.V_PHASE,    'C'),
    'UR': (SignalRole.V_PHASE,    'A'),
    'UY': (SignalRole.V_PHASE,    'B'),
    'UB': (SignalRole.V_PHASE,    'C'),
    'IR': (SignalRole.I_PHASE,    'A'),
    'IY': (SignalRole.I_PHASE,    'B'),
    'IB': (SignalRole.I_PHASE,    'C'),
    'IN': (SignalRole.I_EARTH,    'N'),
    'VN': (SignalRole.V_RESIDUAL, 'N'),
    'UN': (SignalRole.V_RESIDUAL, 'N'),
    # ── IEC A/B/C convention ──────────────────────────────────────────────
    'VA': (SignalRole.V_PHASE,    'A'),
    'VC': (SignalRole.V_PHASE,    'C'),
    'IA': (SignalRole.I_PHASE,    'A'),
    'IC': (SignalRole.I_PHASE,    'C'),
    # ── Chinese IEC convention (NARI) — lowercase, case-sensitive ─────────
    'Ia': (SignalRole.I_PHASE,    'A'),
    'Ib': (SignalRole.I_PHASE,    'B'),
    'Ic': (SignalRole.I_PHASE,    'C'),
    'Ua': (SignalRole.V_PHASE,    'A'),
    'Ub': (SignalRole.V_PHASE,    'B'),
    'Uc': (SignalRole.V_PHASE,    'C'),
    'ia': (SignalRole.I_PHASE,    'A'),
    'ib': (SignalRole.I_PHASE,    'B'),
    'ic': (SignalRole.I_PHASE,    'C'),
    'ua': (SignalRole.V_PHASE,    'A'),
    'ub': (SignalRole.V_PHASE,    'B'),
    'uc': (SignalRole.V_PHASE,    'C'),
    # ── Zero / residual sequence ──────────────────────────────────────────
    '3I0':  (SignalRole.I_EARTH,    'N'),
    '3IO':  (SignalRole.I_EARTH,    'N'),   # letter-O misread
    '3U0':  (SignalRole.V_RESIDUAL, 'N'),
    '3UO':  (SignalRole.V_RESIDUAL, 'N'),
    'I0':   (SignalRole.I_EARTH,    'N'),
    'U0':   (SignalRole.V_RESIDUAL, 'N'),
    'V0':   (SignalRole.V_RESIDUAL, 'N'),
    'UX':   (SignalRole.V_RESIDUAL, 'N'),
    'Ux':   (SignalRole.V_RESIDUAL, 'N'),
    'ux':   (SignalRole.V_RESIDUAL, 'N'),
}

# Upper-cased aliases so that e.g. "VB" and "vb" both match when the
# case-sensitive pass misses.  Built once at import time.
_SIGNAL_CODE_UPPER: dict[str, tuple[str, str]] = {
    k.upper(): v for k, v in _SIGNAL_CODE_TO_ROLE.items()
}

# ── Analogue: unit-field mapping ──────────────────────────────────────────────
# Maps upper-cased unit string → (role, phase_hint).
# Phase is '' here; _extract_phase / signal-code lookup refines it.

_UNIT_TO_ROLE: dict[str, tuple[str, str]] = {
    'KV':   (SignalRole.V_PHASE,    ''),
    'V':    (SignalRole.V_PHASE,    ''),
    'KA':   (SignalRole.I_PHASE,    ''),
    'A':    (SignalRole.I_PHASE,    ''),
    'HZ':   (SignalRole.FREQ,       ''),
    'MW':   (SignalRole.P_MW,       '3-phase'),
    'MVAR': (SignalRole.Q_MVAR,     '3-phase'),
    'MVAR': (SignalRole.Q_MVAR,     '3-phase'),  # noqa: duplicate intentional
    'RPM':  (SignalRole.MECH_SPEED, ''),
    '%':    (SignalRole.MECH_VALVE, ''),
}

# ── Analogue: residual / earth refinement keywords ───────────────────────────
_RESIDUAL_V_KW: tuple[str, ...] = (
    'RESID', 'ZERO', '3U0', 'U0', 'V0', 'UX', 'VN', 'UN', 'O SEQ',
)
_EARTH_I_KW: tuple[str, ...] = (
    'NEUTR', 'EARTH', '3I0', 'I0', 'IN', 'RESID',
)

# ── Digital: keyword sets ─────────────────────────────────────────────────────

# ALARM EXCEPTION — checked first, overrides every other digital rule.
_ALARM_KW: tuple[str, ...] = (
    'COMMFAIL', 'COMM FAIL', 'COMM_FAIL', '_FAIL', 'FAIL ',
    'ALARM', 'WARNING', 'MCB', 'VTS', 'BI_EN', 'BI_MCB',
    'EN_Z1', 'EN_Z', '_ENABLE', 'ENABLE_', 'HEALTY', 'HEALTHY',
    'UNUSED', 'VIRTUAL',
)

_CB_KW: tuple[str, ...] = (
    'CB_R', 'CB_Y', 'CB_B',
    'CB OPEN', 'CB CLOSE', 'CB_OPEN', 'CB_CLOSE',
    'GCB OPEN', 'GCB CLOSED', 'GCB_OPEN', 'GCB_CLOSED',
    'FCB OPEN', 'FCB CLOSED', 'FCB_OPEN', 'FCB_CLOSED',
    'BI_52B', 'BI_52A', '52B', '52A',
)

_AR_KW: tuple[str, ...] = (
    '79AR', '79 AR', 'AR_BLOCK', 'AR_ATTEMPTED', 'AR ATTEMPTED',
    'AR_UNSUCCESSFUL', 'AR UNSUCCESSFUL', '79AR_L/O', 'AR_L/O',
    'AR_LOCKOUT', 'AR LOCKOUT', 'RECLOSE', 'AR_INPROGRESS',
    'AUTORECLOSE', 'AUTO RECLOSE',
)

_INTERTRIP_KW: tuple[str, ...] = (
    '50BF', 'BF_SEND', 'BF_REC', 'BF_STG', 'BF_INTR',
    '50BF_STG1', '50BF_STG2', '50BF_INTR', 'BF STAGE',
    'SEND1', 'RECV1', 'SEND2', 'RECV2', 'SEND3', 'RECV3',
    'INTERTRIP', 'INTER-TRIP', 'DIRECT TRIP', 'REMOTE TRIP',
    'DIRECT TRIP FROM REMOTE', 'INTERTRIP RECEIVE',
    '_CS', '_CR', '_REC', 'TELEPROTECT',
    '87L/1/2_L1', '87L/1/2_L2', '87L/1/2_L3',
    'DEF M1/M2', 'POTT',
    # NARI relay SEND/RECV channels (case-insensitive, so 'send1'/'recv1' match)
    'SEND', 'RECV',
    # BEN32 "85 Intertrip" receive/send channels e.g. "LGNG1 85INTR_REC1"
    '85INTR', 'INTR_REC', 'INTR_SND',
)

_TRIGGER_KW: tuple[str, ...] = (
    'TRIGGER', 'TRIG ', 'EXT TRIG', 'EXTERNAL TRIG',
)

_TRIP_KW: tuple[str, ...] = (
    # Differential
    '87L', '87T', '87B', '87G', '87N', '87M', '87STUB',
    # Distance
    '21ZBU', 'BU_21Z', '21Z', 'OP_Z1', 'OP_Z2', 'OP_Z3', 'OP_Z_REV', 'OP_Z_SOTF',
    'OP_Z_TELEP', 'OP_Z_TELEP', 'OP_Z_TELE',
    'M1 Z1', 'M2 Z1', 'M1 21ZBU', 'M2 21ZBU',
    'M1 GEN', 'M1 GENERAL', 'M2 GEN', 'GENERAL TRIP',
    'DELAY TRIP', 'M1 DELAY', 'M2 DELAY',
    # Overcurrent / ROC
    'OP_ROC', 'OP_OC', 'OP_OVLD', 'OP_ROC_SOTF', 'OP_ROC_POTT',
    'OP_ROC_TELEP', 'OP_WE_POTT',
    'TOL STAGE', 'BACKUP TOL', 'THERMAL OL', 'OVLD_TRP',
    # SOTF
    'SOTF', 'M1/M2_SOTF', 'M1/M2 SOTF',
    # Generic
    'TRIP', 'OPERATED', 'GEN TRIP',
    # NARI prefix (Op_ at start)
    'OP_',
    # Differential current trip
    'IDIFF TRIP',
)

_PICKUP_KW: tuple[str, ...] = (
    'OVER ', 'UNDER ', 'OVERVOLTAGE', 'UNDERVOLTAGE', 'OVERCURRENT',
    'FD,', ' FD', '_FD', 'FAULT DET', 'FAULT DETECT',
    'PICKUP', 'PICK UP', 'START ', 'ELEMENT START',
    'AR_INPROGRESS', 'AR INPROG',
    'VEBI_DISTP', 'VEBI_ROC',
    'BI_EXTRP',
    # Standalone "FD" (NARI digital channel name)
    'FD',
    # NARI "PhComp Blk" = phase comparator block = pickup-level event
    'PHCOMP BLK',
    # "Idiff CZ Start"
    'IDIFF CZ START',
    # "Ext CBF" = external CB-fail initiation (pickup level)
    'EXT CBF',
)

# ── Regex patterns for PRIORITY 7 ────────────────────────────────────────────
_CURRENT_RE = re.compile(r'^I[_\-]?[RYBABC]|^IL\d|CURR|AMPERE', re.IGNORECASE)
_VOLTAGE_RE = re.compile(r'^[VU][_\-]?[RYBABC]|^[VU]\d|VOLT', re.IGNORECASE)


# ── Phase extraction helper ───────────────────────────────────────────────────

def _extract_phase(name: str) -> str:
    """Extract phase letter from a channel name.

    Handles R/Y/B (BEN32), A/B/C (IEC), L1/L2/L3 (European), and
    neutral/earth keywords.  Returns '' when no phase can be determined.
    """
    n = name.upper().strip()
    parts = n.split()
    last = parts[-1] if parts else n

    # R/Y/B convention
    if last.endswith('R') or '_R' in n or 'PHASE R' in n:
        return 'A'
    if last.endswith('Y') or '_Y' in n or 'PHASE Y' in n:
        return 'B'
    # Avoid matching 'IB' ending as phase C — only bare suffix 'B'
    if last.endswith('B') and not last.endswith('IB'):
        return 'C'

    # A/B/C convention (last char of last token)
    _PHASE_MAP = {'A': 'A', 'B': 'B', 'C': 'C'}
    if len(last) >= 2 and last[-1] in _PHASE_MAP:
        return _PHASE_MAP[last[-1]]

    # Numeric L1/L2/L3
    if last.endswith('1') or 'L1' in n:
        return 'A'
    if last.endswith('2') or 'L2' in n:
        return 'B'
    if last.endswith('3') or 'L3' in n:
        return 'C'

    # Neutral / earth
    if any(k in n for k in ('IN', 'UN', 'VN', 'NEUTRAL', 'EARTH', 'GROUND', 'N ')):
        return 'N'

    return ''


def _strip_bay_signal(name: str) -> tuple[str, str]:
    """Return (signal_raw, signal_upper) — the last space-separated token."""
    parts = name.strip().split()
    if not parts:
        return ('', '')
    raw = parts[-1]
    return (raw, raw.upper())


# ── Core analogue detector ───────────────────────────────────────────────────

def detect_analogue_role(ch: AnalogueChannel) -> tuple[str, str, str]:
    """Detect (signal_role, phase, confidence) for one analogue channel.

    Rules are applied in strict priority order; first match wins.
    Also sets ch.is_derived=True when a P_MW or Q_MVAR channel is
    detected (indicates a pre-calculated power value stored in the file).

    Args:
        ch: AnalogueChannel with name, unit, offset, and raw_data set.

    Returns:
        Tuple of (signal_role, phase, confidence) strings.
    """
    name = ch.name.strip()
    name_up = name.upper()
    unit_up = (ch.unit or '').upper().strip()

    # ── PRIORITY 1: Unit field ────────────────────────────────────────────
    if unit_up in _UNIT_TO_ROLE:
        role, phase = _UNIT_TO_ROLE[unit_up]

        # Refine V_PHASE / I_PHASE with signal-code or phase extraction
        if role in (SignalRole.V_PHASE, SignalRole.I_PHASE):
            sig_raw, sig_up = _strip_bay_signal(name)
            if sig_raw in _SIGNAL_CODE_TO_ROLE:
                role, phase = _SIGNAL_CODE_TO_ROLE[sig_raw]
            elif sig_up in _SIGNAL_CODE_UPPER:
                role, phase = _SIGNAL_CODE_UPPER[sig_up]
            else:
                phase = _extract_phase(name)

            # Check for residual / earth within voltage/current
            if role == SignalRole.V_PHASE and any(k in name_up for k in _RESIDUAL_V_KW):
                role = SignalRole.V_RESIDUAL
                phase = 'N'
            if role == SignalRole.I_PHASE and any(k in name_up for k in _EARTH_I_KW):
                role = SignalRole.I_EARTH
                phase = 'N'

        # MVAR written as 'MVAr' or 'MVAR' with mixed case
        if unit_up in ('MVAR',):
            role = SignalRole.Q_MVAR
            phase = '3-phase'

        if role in (SignalRole.P_MW, SignalRole.Q_MVAR):
            ch.is_derived = True

        return (role, phase, RoleConfidence.HIGH)

    # ── PRIORITY 2: Non-zero offset → DC / mechanical ─────────────────────
    if ch.offset != 0.0:
        if any(k in name_up for k in ('FIELD CURRENT', 'FIELD I', 'IF ')):
            return (SignalRole.DC_FIELD_I, '', RoleConfidence.HIGH)
        if any(k in name_up for k in ('FIELD VOLTAGE', 'FIELD V', 'VF ')):
            return (SignalRole.DC_FIELD_V, '', RoleConfidence.HIGH)
        if any(k in name_up for k in ('SPEED', 'RPM', 'MECH')):
            return (SignalRole.MECH_SPEED, '', RoleConfidence.HIGH)
        if any(k in name_up for k in ('VALVE', 'GATE', 'GOVERNOR', 'POSITION')):
            return (SignalRole.MECH_VALVE, '', RoleConfidence.HIGH)
        # Frequency channels often have a non-zero offset in BEN32 slow records
        if any(k in name_up for k in ('FREQ', 'FREQUENCY')):
            return (SignalRole.FREQ, '', RoleConfidence.HIGH)

    # ── PRIORITY 3: Power channel name patterns ────────────────────────────
    # Q_MVAR checked first — "REACTIVE POWER UNIT" must not match P_MW keywords
    if any(k in name_up for k in ('R.POWER', 'REACTIVE', 'MVAR')):
        ch.is_derived = True
        return (SignalRole.Q_MVAR, '3-phase', RoleConfidence.HIGH)
    if any(k in name_up for k in ('POWER UNIT', 'ACTIVE POWER', 'REAL POWER')):
        ch.is_derived = True
        return (SignalRole.P_MW, '3-phase', RoleConfidence.HIGH)
    # "POWER <BAY>" as used in BEN32 slow records (e.g. "POWER PGPS 2")
    if name_up.startswith('POWER '):
        ch.is_derived = True
        return (SignalRole.P_MW, '3-phase', RoleConfidence.HIGH)

    # ── PRIORITY 4: Frequency channel name patterns ────────────────────────
    if any(k in name_up for k in ('DF/DT', 'ROCOF')):
        return (SignalRole.ROCOF, '', RoleConfidence.HIGH)
    if any(k in name_up for k in ('FREQ ', 'FREQUENCY')):
        return (SignalRole.FREQ, '', RoleConfidence.HIGH)
    # "FREQ<something>" at start (e.g. "FREQ PGPS 2", "FREQ UY PLTG 1")
    if name_up.startswith('FREQ'):
        return (SignalRole.FREQ, '', RoleConfidence.HIGH)

    # ── PRIORITY 5: Sequence RMS name patterns ─────────────────────────────
    if any(k in name_up for k in (
        'O SEQ RMS', 'ZERO SEQ', 'NEG SEQ', 'POS SEQ', 'SEQ RMS', '0SEQ',
    )):
        phase_hint = ''
        if 'NEG' in name_up:
            phase_hint = 'neg'
        elif 'POS' in name_up:
            phase_hint = 'pos'
        else:
            phase_hint = 'zero'
        return (SignalRole.SEQ_RMS, phase_hint, RoleConfidence.HIGH)

    # ── PRIORITY 6: Direct signal-code lookup ─────────────────────────────
    sig_raw, sig_up = _strip_bay_signal(name)
    if sig_raw in _SIGNAL_CODE_TO_ROLE:
        role, phase = _SIGNAL_CODE_TO_ROLE[sig_raw]
        return (role, phase, RoleConfidence.HIGH)
    if sig_up in _SIGNAL_CODE_UPPER:
        role, phase = _SIGNAL_CODE_UPPER[sig_up]
        return (role, phase, RoleConfidence.HIGH)

    # ── PRIORITY 7: Regex pattern matching ────────────────────────────────
    if _CURRENT_RE.search(name_up):
        return (SignalRole.I_PHASE, _extract_phase(name), RoleConfidence.MEDIUM)
    if _VOLTAGE_RE.search(name_up):
        return (SignalRole.V_PHASE, _extract_phase(name), RoleConfidence.MEDIUM)

    # ── PRIORITY 8: Magnitude heuristic ───────────────────────────────────
    if ch.raw_data is not None and len(ch.raw_data) > 0:
        max_abs = float(np.nanmax(np.abs(ch.raw_data.astype(np.float64))))
        if max_abs > 1000:
            return (SignalRole.V_PHASE, '', RoleConfidence.LOW)
        if 0.1 < max_abs < 50:
            return (SignalRole.I_PHASE, '', RoleConfidence.LOW)

    # ── FALLBACK ──────────────────────────────────────────────────────────
    return (SignalRole.ANALOGUE, '', RoleConfidence.LOW)


# ── Core digital detector ────────────────────────────────────────────────────

def detect_digital_role(ch: DigitalChannel) -> tuple[str, str]:
    """Detect (signal_role, confidence) for one digital channel.

    ALARM EXCEPTION is evaluated first — it overrides all other rules.

    Args:
        ch: DigitalChannel with name set.

    Returns:
        Tuple of (signal_role, confidence) strings.
    """
    name_up = ch.name.upper().strip()

    # ── ALARM EXCEPTION (must be first) ───────────────────────────────────
    if any(k in name_up for k in _ALARM_KW):
        return (SignalRole.DIG_GENERIC, RoleConfidence.HIGH)

    # ── Circuit breaker status ─────────────────────────────────────────────
    if any(k in name_up for k in _CB_KW):
        return (SignalRole.DIG_CB, RoleConfidence.HIGH)

    # ── Auto-reclose ──────────────────────────────────────────────────────
    if any(k in name_up for k in _AR_KW):
        return (SignalRole.DIG_AR, RoleConfidence.HIGH)

    # ── Intertrip / breaker-fail ──────────────────────────────────────────
    if any(k in name_up for k in _INTERTRIP_KW):
        return (SignalRole.DIG_INTERTRIP, RoleConfidence.HIGH)

    # ── External trigger ──────────────────────────────────────────────────
    if any(k in name_up for k in _TRIGGER_KW):
        return (SignalRole.DIG_TRIGGER, RoleConfidence.HIGH)

    # ── Protection trip (after specific checks above) ─────────────────────
    if any(k in name_up for k in _TRIP_KW):
        return (SignalRole.DIG_TRIP, RoleConfidence.HIGH)

    # ── Protection pickup ─────────────────────────────────────────────────
    if any(k in name_up for k in _PICKUP_KW):
        return (SignalRole.DIG_PICKUP, RoleConfidence.MEDIUM)

    # ── Fallback ──────────────────────────────────────────────────────────
    return (SignalRole.DIG_GENERIC, RoleConfidence.LOW)


# ── Complementary CB pair detector ───────────────────────────────────────────

def detect_complementary_cb_pairs(
    digital_channels: list[DigitalChannel],
) -> list[tuple[int, int]]:
    """Find GCB/FCB/CB OPEN + CLOSED complementary pairs among DIG_CB channels.

    Two DIG_CB channels form a complementary pair when:
    - One name contains 'OPEN'
    - The other contains 'CLOSED' or 'CLOSE'
    - Their base names (with OPEN/CLOSED stripped) match after normalisation

    Modifies channels in-place:
    - is_complementary = True on both channels
    - is_primary_of_pair = True on the OPEN side
    - complementary_channel_id set on both channels

    Returns:
        List of (open_channel_id, closed_channel_id) tuples.
    """
    pairs: list[tuple[int, int]] = []
    cb_channels = [c for c in digital_channels if c.signal_role == SignalRole.DIG_CB]

    for i, ch_open in enumerate(cb_channels):
        name_up = ch_open.name.upper()
        if 'OPEN' not in name_up:
            continue
        base_open = (
            name_up
            .replace('_OPEN', '')
            .replace(' OPEN', '')
            .strip()
        )
        for ch_close in cb_channels[i + 1:]:
            close_up = ch_close.name.upper()
            if 'CLOSED' in close_up or 'CLOSE' in close_up:
                base_close = (
                    close_up
                    .replace('_CLOSED', '')
                    .replace(' CLOSED', '')
                    .replace('_CLOSE', '')
                    .replace(' CLOSE', '')
                    .strip()
                )
                if base_open == base_close:
                    # Mark both channels
                    ch_open.is_complementary = True
                    ch_open.is_primary_of_pair = True
                    ch_open.complementary_channel_id = ch_close.channel_id

                    ch_close.is_complementary = True
                    ch_close.is_primary_of_pair = False
                    ch_close.complementary_channel_id = ch_open.channel_id

                    pairs.append((ch_open.channel_id, ch_close.channel_id))

    return pairs


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_signal_roles(
    channels: list[Union[AnalogueChannel, DigitalChannel]],
) -> list[Union[AnalogueChannel, DigitalChannel]]:
    """Auto-detect signal roles for all channels in a DisturbanceRecord.

    Modifies every channel object in-place:
    - signal_role, phase (analogue), role_confidence
    - colour (via default_colour_for)
    - is_derived (analogue P_MW / Q_MVAR only)
    - is_complementary, complementary_channel_id, is_primary_of_pair (digital DIG_CB)

    Args:
        channels: Combined list of AnalogueChannel and DigitalChannel objects.
                  Typically record.analogue_channels + record.digital_channels.

    Returns:
        The same list (mutated in place).
    """
    digital_channels: list[DigitalChannel] = []

    for ch in channels:
        if isinstance(ch, AnalogueChannel):
            role, phase, confidence = detect_analogue_role(ch)
            ch.signal_role = role
            ch.phase = phase if phase else ch.phase
            ch.role_confidence = confidence
            ch.colour = default_colour_for(role, ch.phase)

        elif isinstance(ch, DigitalChannel):
            role, confidence = detect_digital_role(ch)
            ch.signal_role = role
            ch.role_confidence = confidence  # type: ignore[attr-defined]
            ch.colour = default_colour_for(role, ch.phase)
            digital_channels.append(ch)

    # Complementary CB pair detection runs after all digital roles are assigned
    detect_complementary_cb_pairs(digital_channels)

    return channels
