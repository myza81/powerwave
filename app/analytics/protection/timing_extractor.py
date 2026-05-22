"""Protection relay timing extraction — pure numpy, no Qt.

Extracts key protection timings from digital channel state transitions
and analog waveform signatures:

  Fault inception   — first voltage dip / overcurrent event start
  Relay pickup      — first digital channel departure from normal state,
                      on a channel identified as a pickup/start contact
  Trip command      — first operated transition on a trip-labelled channel
  CB open / clear   — first operated transition on a CB-status channel,
                      OR the sample where fault current drops to near-zero
  Arc extinction    — confirmed zero current for ≥ 0.5 cycles post-clearing
  Reclosure         — voltage/current recovers after a dead-time interval

All times are in the same second-based reference as the waveform time array.
Relative timings are expressed in milliseconds, referenced to fault inception.

Returns None when no fault inception event is present.
"""
from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

from app.analytics.events.event_models import DetectedEvent, EventType


# ─────────────────────────────────────────────────────────────────────────────
# Result models
# ─────────────────────────────────────────────────────────────────────────────

class ProtectionRole(str, Enum):
    PICKUP  = "pickup"
    TRIP    = "trip"
    CB      = "cb"
    RECLOSE = "reclose"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class TimingEvent:
    """One extracted protection timing milestone."""

    label: str              # e.g. "Relay Pickup", "Trip Command"
    t: float                # absolute time (seconds, same base as waveform)
    t_relative_ms: float    # milliseconds after fault inception
    channel: str            # source channel name
    role: ProtectionRole    # classification of the channel


@dataclasses.dataclass
class ProtectionTimingResult:
    """Extracted protection relay timing summary."""

    t_fault: float                  # fault inception (seconds)
    t_pickup: float | None          # relay pickup (seconds)
    t_trip: float | None            # trip command issued (seconds)
    t_clearing: float | None        # fault current extinguished (seconds)
    t_reclose: float | None         # reclosure / voltage recovery (seconds)

    pickup_delay_ms: float | None   # t_pickup - t_fault (ms)
    trip_delay_ms: float | None     # t_trip   - t_fault (ms)
    clearing_time_ms: float | None  # t_clearing - t_fault (ms)
    reclose_interval_ms: float | None  # t_reclose - t_clearing (ms)

    events: list[TimingEvent]       # all milestone events in chronological order
    notes: list[str]                # diagnostic notes


# ─────────────────────────────────────────────────────────────────────────────
# Digital channel classification
# ─────────────────────────────────────────────────────────────────────────────

_PICKUP_KW  = {"pickup", "start", "strt", "pk", "pku", "prot", "86", "21start", "relay"}
_TRIP_KW    = {"trip", "trp", "tr", "87", "51", "50", "67", "21", "cb_trip", "breaker_trip",
               "t1", "t2", "tc", "tout"}
_CB_KW      = {"cb", "breaker", "bkr", "52", "cbaux", "52a", "52b", "open", "closed",
               "close", "aux", "switch"}
_RECLOSE_KW = {"reclose", "ar", "auto_reclose", "recl", "79", "reclose_in_progress"}


def _classify_digital(name: str, description: str | None) -> ProtectionRole:
    combined = (name + " " + (description or "")).lower()
    tokens = set(combined.replace("_", " ").replace("-", " ").split())

    # Check pickup first (more specific)
    if tokens & _PICKUP_KW:
        return ProtectionRole.PICKUP
    if tokens & _TRIP_KW:
        return ProtectionRole.TRIP
    if tokens & _CB_KW:
        return ProtectionRole.CB
    if tokens & _RECLOSE_KW:
        return ProtectionRole.RECLOSE

    # Fall back to prefix/substring matching for codes like "D1", "D2"
    for kw in _TRIP_KW:
        if kw in combined:
            return ProtectionRole.TRIP
    for kw in _PICKUP_KW:
        if kw in combined:
            return ProtectionRole.PICKUP
    for kw in _CB_KW:
        if kw in combined:
            return ProtectionRole.CB

    return ProtectionRole.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Transition detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_first_operation(
    data: np.ndarray,
    time: np.ndarray,
    normal_state: int,
    t_min: float,
    t_max: float | None = None,
) -> float | None:
    """Return time of the first transition away from normal_state after t_min."""
    i_start = int(np.searchsorted(time, t_min, side="right"))
    i_end = int(np.searchsorted(time, t_max, side="left")) if t_max is not None else len(time)
    i_end = min(i_end, len(time))
    operated = 1 - normal_state
    for i in range(i_start, i_end):
        if data[i] == operated and (i == 0 or data[i - 1] == normal_state):
            return float(time[i])
    return None


def _find_first_recovery(
    data: np.ndarray,
    time: np.ndarray,
    normal_state: int,
    t_min: float,
) -> float | None:
    """Return time of first return to normal_state after an operation (t > t_min)."""
    i_start = int(np.searchsorted(time, t_min, side="right"))
    for i in range(i_start, len(time)):
        if data[i] == normal_state and (i == 0 or data[i - 1] != normal_state):
            return float(time[i])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Current extinction detection
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_rms_fast(data: np.ndarray, window: int) -> np.ndarray:
    """O(N) cumulative-sum sliding RMS, right-aligned."""
    sq = data ** 2
    cs = np.cumsum(sq)
    cs_w = cs[window:] - cs[:-window]
    result = np.sqrt(cs_w / window)
    pad = np.full(window, result[0] if len(result) else 0.0)
    return np.concatenate([pad, result])


def _find_current_extinction(
    current_data: dict[str, np.ndarray],
    time: np.ndarray,
    t_fault: float,
    prefault_rms: dict[str, float],
    *,
    cycle_samples: int,
    threshold_pct: float = 0.05,
    sustained_cycles: float = 0.5,
) -> float | None:
    """Return time when all current channels drop below threshold_pct of prefault."""
    if not current_data:
        return None

    sustained = max(1, int(sustained_cycles * cycle_samples))
    i_fault = int(np.searchsorted(time, t_fault, side="right"))

    # Build a mask: True when all currents are below threshold simultaneously
    low_mask = np.ones(len(time), dtype=bool)
    for name, data in current_data.items():
        baseline = prefault_rms.get(name, 0.0)
        if baseline < 1e-9:
            continue
        rms = _sliding_rms_fast(np.where(np.isfinite(data), data, 0.0), cycle_samples)
        low_mask &= (rms < threshold_pct * baseline)

    # Find first sustained run of True after t_fault
    for i in range(i_fault, len(time) - sustained):
        if np.all(low_mask[i:i + sustained]):
            return float(time[i])
    return None


def _find_current_recovery(
    current_data: dict[str, np.ndarray],
    time: np.ndarray,
    prefault_rms: dict[str, float],
    t_min: float,
    *,
    cycle_samples: int,
    threshold_pct: float = 0.10,
) -> float | None:
    """Return time when any current channel rises above threshold_pct of prefault after t_min."""
    if not current_data:
        return None
    i_min = int(np.searchsorted(time, t_min, side="right"))
    for name, data in current_data.items():
        baseline = prefault_rms.get(name, 0.0)
        if baseline < 1e-9:
            continue
        rms = _sliding_rms_fast(np.where(np.isfinite(data), data, 0.0), cycle_samples)
        for i in range(i_min, len(time) - 1):
            if rms[i] < threshold_pct * baseline and rms[i + 1] >= threshold_pct * baseline:
                return float(time[i + 1])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_protection_timing(
    events: list[DetectedEvent],
    time: np.ndarray,
    analog_data: dict[str, np.ndarray],
    digital_data: dict[str, np.ndarray],
    digital_channels: list,
    analog_channels: list,
    *,
    nominal_hz: float = 50.0,
) -> ProtectionTimingResult | None:
    """Extract protection relay timing from digital channels and waveform data.

    Args:
        events:           Detected events (from detect_events — Phase 2).
        time:             1-D float64 time array (seconds).
        analog_data:      channel_name → array for all analog channels with roles.
        digital_data:     channel_name → array (0/1 integers) for digital channels.
        digital_channels: List of DigitalChannel objects (for normal_state info).
        analog_channels:  List of AnalogChannel objects (for role inference).
        nominal_hz:       Nominal system frequency.

    Returns:
        ProtectionTimingResult or None when no fault inception found.
    """
    if not events or len(time) < 4:
        return None

    # ── 1. Fault inception ────────────────────────────────────────────────────
    fault_events = [
        e for e in events
        if e.event_type in (EventType.VOLTAGE_DIP, EventType.OVERCURRENT)
    ]
    if not fault_events:
        return None

    t_fault = float(min(e.t_start for e in fault_events))
    notes: list[str] = []
    timing_events: list[TimingEvent] = [
        TimingEvent(
            label="Fault Inception",
            t=t_fault,
            t_relative_ms=0.0,
            channel=fault_events[0].channel,
            role=ProtectionRole.UNKNOWN,
        )
    ]

    sample_rate = 1.0 / float(np.median(np.diff(time)))
    cycle_samples = max(4, int(round(sample_rate / nominal_hz)))

    # ── 2. Identify current channels + prefault RMS ───────────────────────────
    from app.analytics.events.event_detector import _infer_role  # noqa: PLC0415

    current_ch_names = [
        ch.name for ch in analog_channels
        if _infer_role(ch.name, getattr(ch, "unit", "") or "") == "current"
        and ch.name in analog_data
    ]
    current_data = {n: np.asarray(analog_data[n], dtype=float) for n in current_ch_names}

    # Pre-fault RMS: first quarter of recording or before t_fault - 2 cycles
    pf_end_t = t_fault - 2.0 / nominal_hz
    i_pf_end = int(np.searchsorted(time, pf_end_t, side="left"))
    if i_pf_end < cycle_samples * 2:
        i_pf_end = min(int(len(time) * 0.25), len(time) - 1)

    prefault_rms: dict[str, float] = {}
    for name, data in current_data.items():
        pf_slice = data[max(0, i_pf_end - cycle_samples):i_pf_end]
        finite = pf_slice[np.isfinite(pf_slice)]
        prefault_rms[name] = float(np.sqrt(np.mean(finite ** 2))) if len(finite) else 0.0

    # ── 3. Digital channel classification and transition detection ────────────
    normal_states = {ch.name: ch.normal_state for ch in digital_channels}
    descriptions  = {ch.name: (ch.description or "") for ch in digital_channels}

    classified: dict[str, ProtectionRole] = {}
    for name in digital_data:
        classified[name] = _classify_digital(name, descriptions.get(name))

    # Look-ahead window: fault to end of recording but max 2 seconds
    t_end = float(time[-1])
    t_window_end = min(t_end, t_fault + 2.0)

    t_pickup:  float | None = None
    t_trip:    float | None = None
    t_cb_open: float | None = None
    t_reclose: float | None = None

    # Ordered priority: pickup before trip before CB
    for role in (ProtectionRole.PICKUP, ProtectionRole.TRIP, ProtectionRole.CB,
                 ProtectionRole.RECLOSE):
        for name, ch_role in classified.items():
            if ch_role != role:
                continue
            data = digital_data[name]
            normal = normal_states.get(name, 0)
            t_op = _find_first_operation(data, time, normal, t_fault, t_window_end)
            if t_op is None:
                continue

            rel_ms = (t_op - t_fault) * 1000.0
            label = {
                ProtectionRole.PICKUP:  "Relay Pickup",
                ProtectionRole.TRIP:    "Trip Command",
                ProtectionRole.CB:      "CB Open",
                ProtectionRole.RECLOSE: "Reclosure",
            }[role]

            timing_events.append(TimingEvent(
                label=label, t=t_op, t_relative_ms=rel_ms, channel=name, role=role,
            ))

            if role == ProtectionRole.PICKUP and t_pickup is None:
                t_pickup = t_op
            elif role == ProtectionRole.TRIP and t_trip is None:
                t_trip = t_op
            elif role == ProtectionRole.CB and t_cb_open is None:
                t_cb_open = t_op
            elif role == ProtectionRole.RECLOSE and t_reclose is None:
                t_reclose = t_op

    if not classified:
        notes.append("No digital channels available — digital timing not extracted")

    # ── 4. Current extinction (fault clearing from waveform) ──────────────────
    t_waveform_clear: float | None = None
    if current_data:
        t_waveform_clear = _find_current_extinction(
            current_data, time, t_fault, prefault_rms,
            cycle_samples=cycle_samples,
        )
        if t_waveform_clear is not None:
            rel_ms = (t_waveform_clear - t_fault) * 1000.0
            timing_events.append(TimingEvent(
                label="Arc Extinction",
                t=t_waveform_clear,
                t_relative_ms=rel_ms,
                channel=", ".join(current_ch_names[:2]),
                role=ProtectionRole.UNKNOWN,
            ))
    else:
        notes.append("No current channels found — waveform clearing time not extracted")

    # ── 5. Reclosure from current recovery ────────────────────────────────────
    t_clear = t_cb_open or t_waveform_clear
    if t_clear is not None and t_reclose is None and current_data:
        t_reclose_wave = _find_current_recovery(
            current_data, time, prefault_rms, t_clear,
            cycle_samples=cycle_samples,
        )
        if t_reclose_wave is not None:
            rel_ms = (t_reclose_wave - t_fault) * 1000.0
            timing_events.append(TimingEvent(
                label="Reclosure (waveform)",
                t=t_reclose_wave,
                t_relative_ms=rel_ms,
                channel=", ".join(current_ch_names[:2]),
                role=ProtectionRole.RECLOSE,
            ))
            t_reclose = t_reclose_wave

    # ── 6. Assemble result ────────────────────────────────────────────────────
    # Use CB open as primary clearing time; fall back to waveform extinction
    t_clearing = t_cb_open or t_waveform_clear

    def _delay_ms(t: float | None) -> float | None:
        return (t - t_fault) * 1000.0 if t is not None else None

    def _interval_ms(t_end: float | None, t_start: float | None) -> float | None:
        if t_end is not None and t_start is not None:
            return (t_end - t_start) * 1000.0
        return None

    timing_events.sort(key=lambda e: e.t)

    return ProtectionTimingResult(
        t_fault=t_fault,
        t_pickup=t_pickup,
        t_trip=t_trip,
        t_clearing=t_clearing,
        t_reclose=t_reclose,
        pickup_delay_ms=_delay_ms(t_pickup),
        trip_delay_ms=_delay_ms(t_trip),
        clearing_time_ms=_delay_ms(t_clearing),
        reclose_interval_ms=_interval_ms(t_reclose, t_clearing),
        events=timing_events,
        notes=notes,
    )
