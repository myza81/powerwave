"""Frequency and ROCOF analytics — Phase 5C.

Provider-neutral integration layer for existing frequency and ROCOF channels.
Does NOT implement waveform-derived frequency estimation, PLL, zero-crossing
detection, or DFT frequency tracking. Those belong to later phases.

Public surface:
  FrequencyDisplayMode     — OFF / OVERLAY / PANEL_ONLY visibility enum
  FrequencyChannelRole     — FREQUENCY / ROCOF / UNKNOWN classification
  FrequencyChannelResult   — role + reason + auto_classified + display_unit
  FrequencyConfig          — nominal Hz configuration holder
  FrequencyRegistry        — session-level classification cache and mode tracker
  classify_frequency_role  — pure classification function (test-friendly)
  is_frequency_channel     — convenience bool check
  is_rocof_channel         — convenience bool check (re-exported from overlay)
  rocof_display_label      — ROCOF curve label helper
  rocof_axis_label         — ROCOF shared-axis title helper
  frequency_display_label  — frequency curve label helper
  frequency_axis_label     — frequency shared-axis title helper
"""
from app.analytics.frequency.frequency_models import (
    FrequencyChannelResult,
    FrequencyChannelRole,
    FrequencyConfig,
    FrequencyDisplayMode,
)
from app.analytics.frequency.frequency_overlay import (
    classify_frequency_role,
    is_frequency_channel,
    is_rocof_channel,
)
from app.analytics.frequency.frequency_registry import FrequencyRegistry
from app.analytics.frequency.rocof_overlay import (
    frequency_axis_label,
    frequency_display_label,
    rocof_axis_label,
    rocof_display_label,
)

__all__ = [
    "FrequencyChannelResult",
    "FrequencyChannelRole",
    "FrequencyConfig",
    "FrequencyDisplayMode",
    "FrequencyRegistry",
    "classify_frequency_role",
    "frequency_axis_label",
    "frequency_display_label",
    "is_frequency_channel",
    "is_rocof_channel",
    "rocof_axis_label",
    "rocof_display_label",
]
