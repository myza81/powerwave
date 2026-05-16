"""RMS overlay analytics - Phase 5A foundation.

Public surface:
  RMSDisplayMode       - OFF / OVERLAY / RMS_ONLY enum
  RMSWindowMode        - HALF_CYCLE / ONE_CYCLE / TWO_CYCLE / CUSTOM_SAMPLES
  RMSConfig            - nominal frequency plus selected RMS window mode
  RMSEligibilityResult - eligible flag + reason + auto_classified
  RMSCache             - per-canvas result store (avoids redundant recompute)
  compute_rms_overlay  - vectorized (rms_time, rms_values) from raw waveform
  compute_rms_window_samples - config-aware engineering sample count
  compute_window_samples - legacy engineering-parameter sample count helper
  classify_rms_eligibility - channel eligibility heuristic + override support
"""
from app.analytics.rms.rms_cache import RMSCache
from app.analytics.rms.rms_models import (
    RMSConfig,
    RMSDisplayMode,
    RMSEligibilityResult,
    RMSWindowMode,
)
from app.analytics.rms.rms_overlay import classify_rms_eligibility
from app.analytics.rms.sliding_rms import (
    compute_rms_overlay,
    compute_rms_window_samples,
    compute_window_samples,
)

__all__ = [
    "RMSCache",
    "RMSConfig",
    "RMSDisplayMode",
    "RMSEligibilityResult",
    "RMSWindowMode",
    "classify_rms_eligibility",
    "compute_rms_overlay",
    "compute_rms_window_samples",
    "compute_window_samples",
]
