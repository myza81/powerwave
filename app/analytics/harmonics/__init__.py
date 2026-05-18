"""Harmonic analysis foundation — Phase 7.

Public surface:

Models:
  HarmonicDisplayMode   — OFF / HARMONIC_MAGNITUDE / THD / SPECTRUM
  HarmonicWindowMode    — ONE_CYCLE / TWO_CYCLE / FOUR_CYCLE
  HarmonicChannelRole   — VOLTAGE_HARMONIC / CURRENT_HARMONIC / UNKNOWN
  HarmonicChannelResult — immutable classification result (frozen dataclass)
  HarmonicConfig        — nominal_hz, max_order, window_mode, overlap, …
  HarmonicResult        — sliding-window extraction result (time + magnitudes)

Extraction:
  compute_harmonic_window_samples — window size in samples from config + sample rate
  extract_harmonics               — vectorized sliding-window FFT extraction

Metrics:
  compute_thd              — scalar THD from {order: magnitude} dict
  compute_thd_array        — vectorized time-varying THD from magnitude arrays
  compute_thd_from_result  — time-varying THD from a HarmonicResult
  individual_harmonic_distortion — H_n / H_1 ratio

Classification:
  classify_harmonic_role — priority-chain channel role classification
  is_harmonic_eligible   — convenience bool helper

Registry:
  HarmonicRegistry — session-level mutable registry with classification cache

Cache:
  HarmonicCache — per-record result store (avoids redundant recompute)
"""
from app.analytics.harmonics.harmonic_cache import HarmonicCache
from app.analytics.harmonics.harmonic_extraction import (
    compute_harmonic_window_samples,
    extract_harmonics,
)
from app.analytics.harmonics.harmonic_metrics import (
    compute_thd,
    compute_thd_array,
    compute_thd_from_result,
    individual_harmonic_distortion,
)
from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelResult,
    HarmonicChannelRole,
    HarmonicConfig,
    HarmonicDisplayMode,
    HarmonicResult,
    HarmonicWindowMode,
)
from app.analytics.harmonics.harmonic_overlay import (
    classify_harmonic_role,
    is_harmonic_eligible,
)
from app.analytics.harmonics.harmonic_registry import HarmonicRegistry

__all__ = [
    # Models
    "HarmonicDisplayMode",
    "HarmonicWindowMode",
    "HarmonicChannelRole",
    "HarmonicChannelResult",
    "HarmonicConfig",
    "HarmonicResult",
    # Extraction
    "compute_harmonic_window_samples",
    "extract_harmonics",
    # Metrics
    "compute_thd",
    "compute_thd_array",
    "compute_thd_from_result",
    "individual_harmonic_distortion",
    # Classification
    "classify_harmonic_role",
    "is_harmonic_eligible",
    # Registry
    "HarmonicRegistry",
    # Cache
    "HarmonicCache",
]
