"""Phasor and sequence component analytics — Phase 6A.

Public surface:

Models:
  PhasorDisplayMode    — OFF / MAGNITUDE / ANGLE / SEQUENCE_COMPONENTS
  PhasorWindowMode     — HALF_CYCLE / ONE_CYCLE / TWO_CYCLE
  PhaseLabel           — A / B / C / UNKNOWN
  PhasorChannelRole    — VOLTAGE_PHASOR / CURRENT_PHASOR / UNKNOWN
  PhasorChannelResult  — immutable classification result
  PhasorConfig         — nominal_hz + window_mode
  ThreePhaseGroup      — complete three-phase set (A/B/C channel names)

Extraction:
  extract_phasor             — sliding DFT phasor from waveform arrays
  extract_phasor_from_record — convenience wrapper for DataFrame columns
  compute_phasor_window_samples — engineering window size in samples

Symmetrical components:
  compute_sequence_components        — Fortescue transform (Va, Vb, Vc → V1, V2, V0)
  compute_sequence_from_phasor_arrays — end-to-end helper from extract_phasor() tuples
  sequence_magnitudes                — |V1|, |V2|, |V0| from complex arrays
  unbalance_factor                   — NSVUF = |V2|/|V1| × 100 %

Classification:
  classify_phasor_role       — priority-chain channel role classification
  identify_phase             — phase A/B/C identification with override support
  detect_three_phase_groups  — find complete A/B/C groups in a channel list
  is_voltage_channel         — convenience bool helper
  is_current_channel         — convenience bool helper

Registry:
  PhasorRegistry             — session-level mutable registry with cache

Cache:
  PhasorCache                — per-record result store (avoids redundant recompute)
"""
from app.analytics.phasors.phasor_cache import PhasorCache
from app.analytics.phasors.phasor_extraction import (
    compute_phasor_window_samples,
    extract_phasor,
    extract_phasor_from_record,
)
from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorChannelResult,
    PhasorChannelRole,
    PhasorConfig,
    PhasorDisplayMode,
    PhasorWindowMode,
    ThreePhaseGroup,
)
from app.analytics.phasors.phasor_overlay import (
    classify_phasor_role,
    detect_three_phase_groups,
    identify_phase,
    is_current_channel,
    is_voltage_channel,
)
from app.analytics.phasors.phasor_registry import PhasorRegistry
from app.analytics.phasors.symmetrical_components import (
    compute_sequence_components,
    compute_sequence_from_phasor_arrays,
    sequence_magnitudes,
    unbalance_factor,
)

__all__ = [
    # Models
    "PhasorDisplayMode",
    "PhasorWindowMode",
    "PhaseLabel",
    "PhasorChannelRole",
    "PhasorChannelResult",
    "PhasorConfig",
    "ThreePhaseGroup",
    # Extraction
    "extract_phasor",
    "extract_phasor_from_record",
    "compute_phasor_window_samples",
    # Symmetrical components
    "compute_sequence_components",
    "compute_sequence_from_phasor_arrays",
    "sequence_magnitudes",
    "unbalance_factor",
    # Classification
    "classify_phasor_role",
    "identify_phase",
    "detect_three_phase_groups",
    "is_voltage_channel",
    "is_current_channel",
    # Registry
    "PhasorRegistry",
    # Cache
    "PhasorCache",
]
