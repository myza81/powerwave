"""Fault type classification from detected events and waveform data.

Uses symmetrical components (Fortescue transform) at the fault midpoint to
classify the fault as SLG, LL, DLG, or 3-phase, and to identify which
phases are involved.

Algorithm:
  1. Select the primary fault event (deepest voltage dip or highest overcurrent).
  2. Identify voltage channels for phases A, B, C by name heuristics.
  3. Compute prefault and fault-window RMS for each phase to find depressions.
  4. Compute a single-cycle DFT phasor at the fault midpoint for each phase.
  5. Apply Fortescue transform → V1, V2, V0 sequence magnitudes.
  6. Classify fault type using sequence ratios + per-phase depression.

Returns None when three-phase voltage channels cannot be identified.
"""
from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np

from app.analytics.events.event_models import DetectedEvent, EventType
from app.analytics.phasors.symmetrical_components import (
    compute_sequence_components,
    sequence_magnitudes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fault type taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class FaultType(str, Enum):
    A_G   = "A-g"
    B_G   = "B-g"
    C_G   = "C-g"
    AB    = "AB"
    BC    = "BC"
    CA    = "CA"
    AB_G  = "AB-g"
    BC_G  = "BC-g"
    CA_G  = "CA-g"
    ABC   = "ABC"
    ABC_G = "ABC-g"
    UNKNOWN = "Unknown"

    @property
    def description(self) -> str:
        return {
            FaultType.A_G:   "Single-Line-to-Ground (Phase A)",
            FaultType.B_G:   "Single-Line-to-Ground (Phase B)",
            FaultType.C_G:   "Single-Line-to-Ground (Phase C)",
            FaultType.AB:    "Line-to-Line (Phases A–B)",
            FaultType.BC:    "Line-to-Line (Phases B–C)",
            FaultType.CA:    "Line-to-Line (Phases C–A)",
            FaultType.AB_G:  "Double-Line-to-Ground (Phases A–B)",
            FaultType.BC_G:  "Double-Line-to-Ground (Phases B–C)",
            FaultType.CA_G:  "Double-Line-to-Ground (Phases C–A)",
            FaultType.ABC:   "Three-Phase Fault",
            FaultType.ABC_G: "Three-Phase-to-Ground Fault",
            FaultType.UNKNOWN: "Unclassified",
        }[self]

    @property
    def faulted_phases(self) -> list[str]:
        return {
            FaultType.A_G:   ["A"],
            FaultType.B_G:   ["B"],
            FaultType.C_G:   ["C"],
            FaultType.AB:    ["A", "B"],
            FaultType.BC:    ["B", "C"],
            FaultType.CA:    ["C", "A"],
            FaultType.AB_G:  ["A", "B"],
            FaultType.BC_G:  ["B", "C"],
            FaultType.CA_G:  ["C", "A"],
            FaultType.ABC:   ["A", "B", "C"],
            FaultType.ABC_G: ["A", "B", "C"],
            FaultType.UNKNOWN: [],
        }[self]

    @property
    def is_ground_fault(self) -> bool:
        return self in {
            FaultType.A_G, FaultType.B_G, FaultType.C_G,
            FaultType.AB_G, FaultType.BC_G, FaultType.CA_G,
            FaultType.ABC_G,
        }

    @property
    def color(self) -> str:
        if self == FaultType.UNKNOWN:
            return "#888888"
        if self == FaultType.ABC or self == FaultType.ABC_G:
            return "#FF8800"
        if self.is_ground_fault:
            return "#FF5555"
        return "#FFAA33"


@dataclasses.dataclass
class FaultCharacterisation:
    """Result of fault type classification."""

    fault_type: FaultType
    v1_pu: float           # positive sequence ratio at fault (prefault = 1.0)
    v2_pu: float           # negative sequence ratio
    v0_pu: float           # zero sequence ratio
    unbalance_pct: float   # |V2| / |V1| × 100 %
    fault_depth_pct: float # (1 − v1_pu) × 100 % depression
    t_fault_mid: float     # time used for phasor extraction (seconds)
    description: str       # human-readable summary

    @property
    def summary_line(self) -> str:
        ground = " (ground)" if self.fault_type.is_ground_fault else ""
        return (
            f"{self.fault_type.value}{ground} — "
            f"depth {self.fault_depth_pct:.0f}%, "
            f"unbal {self.unbalance_pct:.0f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase channel identification
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_A_KEYWORDS = ["van", "va", "ua", "u_a", "v_a", "vr", "vl1", "v1_", "_a"]
_PHASE_B_KEYWORDS = ["vbn", "vb", "ub", "u_b", "v_b", "vy", "vl2", "v2_", "_b"]
_PHASE_C_KEYWORDS = ["vcn", "vc", "uc", "u_c", "v_c", "vbl", "vl3", "v3_", "_c"]


def _match_phase_keyword(name_l: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if name_l == kw:
            return True
        if name_l.endswith(kw) and (len(name_l) == len(kw) or not name_l[-len(kw) - 1].isalpha()):
            return True
        if name_l.startswith(kw) and (len(name_l) == len(kw) or not name_l[len(kw)].isalpha()):
            return True
    return False


def _find_phase_channel(
    channels: list[str],
    keywords: list[str],
    exclude: set[str],
) -> str | None:
    for ch in channels:
        if ch in exclude:
            continue
        if _match_phase_keyword(ch.lower(), keywords):
            return ch
    return None


def identify_voltage_phase_channels(
    analog_channels: list,
) -> dict[str, str] | None:
    """Return {channel_name: 'A'/'B'/'C'} for a 3-phase voltage set, or None.

    Args:
        analog_channels: List of analog channel objects with `.name` and `.unit`.

    Returns:
        Mapping of three channel names to their phases, or None if 3-phase
        voltage cannot be identified.
    """
    from app.analytics.events.event_detector import _infer_role  # noqa: PLC0415

    voltage_ch = [
        ch.name for ch in analog_channels
        if _infer_role(ch.name, getattr(ch, "unit", "") or "") == "voltage"
    ]

    if len(voltage_ch) < 3:
        return None

    used: set[str] = set()
    ch_a = _find_phase_channel(voltage_ch, _PHASE_A_KEYWORDS, used)
    if ch_a:
        used.add(ch_a)
    ch_b = _find_phase_channel(voltage_ch, _PHASE_B_KEYWORDS, used)
    if ch_b:
        used.add(ch_b)
    ch_c = _find_phase_channel(voltage_ch, _PHASE_C_KEYWORDS, used)

    if ch_a and ch_b and ch_c:
        return {ch_a: "A", ch_b: "B", ch_c: "C"}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phasor computation
# ─────────────────────────────────────────────────────────────────────────────

def _dft_phasor(data: np.ndarray, sample_rate: float, nominal_hz: float) -> complex:
    """Compute a fundamental-frequency RMS phasor from a 1-cycle data window."""
    n = len(data)
    if n < 2:
        return 0.0 + 0.0j
    # Replace NaN with 0 so bad samples don't propagate
    clean = np.where(np.isfinite(data), data, 0.0)
    idx = np.arange(n)
    k = nominal_hz / sample_rate  # fractional cycles per sample
    phasor_peak = 2.0 / n * np.sum(clean * np.exp(-1j * 2.0 * np.pi * k * idx))
    return complex(phasor_peak / math.sqrt(2))  # peak → RMS


def _window_rms(data: np.ndarray) -> float:
    """RMS of finite values in a data window."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite ** 2)))


def _time_to_index(time: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(time, t, side="left"))
    return max(0, min(idx, len(time) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Classification logic
# ─────────────────────────────────────────────────────────────────────────────

_DIP_PHASE_THRESHOLD = 0.85   # per-phase ratio < this → faulted
_V0_GROUND_THRESHOLD = 0.08   # V0/V1_prefault > this → ground path present
_UNBALANCE_THRESHOLD = 0.05   # V2/V1_prefault > this → unbalanced fault


def _classify_type(
    faulted: list[str],   # e.g. ["A", "B"]
    v0_pu: float,
) -> FaultType:
    n = len(faulted)
    is_ground = v0_pu > _V0_GROUND_THRESHOLD

    if n == 0:
        # No phase depressed — likely three-phase balanced or no clear fault
        return FaultType.ABC_G if is_ground else FaultType.ABC

    if n == 1:
        phase = faulted[0]
        return {"A": FaultType.A_G, "B": FaultType.B_G, "C": FaultType.C_G}[phase]

    if n == 2:
        pair = frozenset(faulted)
        ll_map = {
            frozenset(["A", "B"]): (FaultType.AB, FaultType.AB_G),
            frozenset(["B", "C"]): (FaultType.BC, FaultType.BC_G),
            frozenset(["C", "A"]): (FaultType.CA, FaultType.CA_G),
        }
        ll, dlg = ll_map.get(pair, (FaultType.UNKNOWN, FaultType.UNKNOWN))
        return dlg if is_ground else ll

    # n >= 3
    return FaultType.ABC_G if is_ground else FaultType.ABC


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_fault_from_events(
    events: list[DetectedEvent],
    time: np.ndarray,
    data_by_channel: dict[str, np.ndarray],
    analog_channels: list,
    *,
    nominal_hz: float = 50.0,
) -> FaultCharacterisation | None:
    """Classify the fault type from detected events and waveform data.

    Args:
        events:           Detected events (from detect_events).
        time:             1-D time array (seconds).
        data_by_channel:  channel_name → sample array.
        analog_channels:  List of analog channel objects (for phase ID).
        nominal_hz:       System nominal frequency.

    Returns:
        FaultCharacterisation or None if 3-phase voltages not identifiable.
    """
    if not events or len(time) < 4:
        return None

    # Need a primary fault event (voltage dip or overcurrent)
    dip_events = [
        e for e in events
        if e.event_type in (EventType.VOLTAGE_DIP, EventType.OVERCURRENT)
    ]
    if not dip_events:
        return None

    # Select the most severe event (deepest dip / highest OC)
    primary = min(
        dip_events,
        key=lambda e: e.ratio if e.event_type == EventType.VOLTAGE_DIP else -e.ratio,
    )
    t_fault_mid = (primary.t_start + primary.t_end) / 2.0

    # Identify 3-phase voltage channels
    phase_map = identify_voltage_phase_channels(analog_channels)
    if phase_map is None:
        return None

    sample_rate = 1.0 / float(np.median(np.diff(time)))
    cycle_samples = max(4, int(round(sample_rate / nominal_hz)))

    # Build prefault window: first quarter of recording or before first event
    t_first_event = min(e.t_start for e in events)
    prefault_end = t_first_event - 1.0 / nominal_hz  # one cycle before first event
    if prefault_end <= time[0]:
        prefault_end = time[int(len(time) * 0.25)]

    i_pf_end = _time_to_index(time, prefault_end)
    pf_start = max(0, i_pf_end - cycle_samples)
    prefault_slice = slice(pf_start, i_pf_end)

    # Build fault window: one cycle centred on fault midpoint
    i_mid = _time_to_index(time, t_fault_mid)
    half = cycle_samples // 2
    fault_slice = slice(max(0, i_mid - half), min(len(time), i_mid + half))

    # Compute phasors and RMS for each phase
    phasors_fault: dict[str, complex] = {}
    phasors_prefault: dict[str, complex] = {}
    rms_fault: dict[str, float] = {}
    rms_prefault: dict[str, float] = {}

    for ch, phase in phase_map.items():
        data = data_by_channel.get(ch)
        if data is None:
            continue
        pf_data = data[prefault_slice]
        ft_data = data[fault_slice]
        phasors_prefault[phase] = _dft_phasor(pf_data, sample_rate, nominal_hz)
        phasors_fault[phase] = _dft_phasor(ft_data, sample_rate, nominal_hz)
        rms_prefault[phase] = _window_rms(pf_data)
        rms_fault[phase] = _window_rms(ft_data)

    if not all(p in phasors_fault for p in ["A", "B", "C"]):
        return None

    # Sequence components at fault midpoint
    v1_f, v2_f, v0_f = compute_sequence_components(
        np.array([phasors_fault["A"]]),
        np.array([phasors_fault["B"]]),
        np.array([phasors_fault["C"]]),
    )
    v1_pf, _, _ = compute_sequence_components(
        np.array([phasors_prefault["A"]]),
        np.array([phasors_prefault["B"]]),
        np.array([phasors_prefault["C"]]),
    )

    mag_v1_f  = float(np.abs(v1_f[0]))
    mag_v2_f  = float(np.abs(v2_f[0]))
    mag_v0_f  = float(np.abs(v0_f[0]))
    mag_v1_pf = float(np.abs(v1_pf[0]))

    ref = mag_v1_pf if mag_v1_pf > 1e-9 else max(mag_v1_f, 1e-9)
    v1_pu = mag_v1_f / ref
    v2_pu = mag_v2_f / ref
    v0_pu = mag_v0_f / ref
    unbalance_pct = (mag_v2_f / mag_v1_f * 100.0) if mag_v1_f > 1e-9 else 0.0
    fault_depth_pct = max(0.0, (1.0 - v1_pu) * 100.0)

    # Per-phase depression
    faulted_phases: list[str] = []
    for phase in ["A", "B", "C"]:
        pf_rms = rms_prefault.get(phase, 0.0)
        ft_rms = rms_fault.get(phase, 0.0)
        if pf_rms > 1e-9:
            ratio = ft_rms / pf_rms
            if ratio < _DIP_PHASE_THRESHOLD:
                faulted_phases.append(phase)
        elif ft_rms < 1e-9:
            faulted_phases.append(phase)

    # Override for 3-phase fault: if all depress together use sequence-based check
    if len(faulted_phases) == 0 and fault_depth_pct > 5.0:
        faulted_phases = ["A", "B", "C"]

    fault_type = _classify_type(faulted_phases, v0_pu)

    description = (
        f"{fault_type.description}; "
        f"V₁={v1_pu:.2f} pu, V₂={v2_pu:.2f} pu, V₀={v0_pu:.2f} pu, "
        f"depth {fault_depth_pct:.0f}%"
    )

    return FaultCharacterisation(
        fault_type=fault_type,
        v1_pu=v1_pu,
        v2_pu=v2_pu,
        v0_pu=v0_pu,
        unbalance_pct=unbalance_pct,
        fault_depth_pct=fault_depth_pct,
        t_fault_mid=t_fault_mid,
        description=description,
    )
