"""ScalingRegistry — priority-chain scaling configuration.

Priority order (highest first):
  1. Per-signal override (operator-specified for one channel)
  2. Global session config (applies to all channels)

Per-signal configs allow one channel to carry a different transformer ratio
or voltage base than the rest of the recording — for example, a 33 kV feeder
mixed with a 275 kV bus, or a recording that mixes HV and LV CT secondaries.
"""
from __future__ import annotations

from app.analytics.scaling.engineering_scaling import compute_scaling_factor
from app.analytics.scaling.scaling_models import (
    EngineeringScalingMode,
    GlobalScalingConfig,
    ScalingResult,
    SignalScalingConfig,
)


class ScalingRegistry:
    """Runtime holder for session-level and per-channel scaling configuration.

    Mutable by design — the operator can update PT/CT ratios or per-signal
    overrides during an active session without reloading data.
    """

    def __init__(self) -> None:
        self._global: GlobalScalingConfig = GlobalScalingConfig()
        self._per_signal: dict[str, SignalScalingConfig] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration setters
    # ─────────────────────────────────────────────────────────────────────────

    def set_global_config(self, config: GlobalScalingConfig) -> None:
        """Replace the session-level default scaling config."""
        self._global = config

    def set_signal_config(self, channel_name: str, config: SignalScalingConfig) -> None:
        """Add or replace a per-signal override for *channel_name*."""
        self._per_signal[channel_name] = config

    def clear_signal_config(self, channel_name: str) -> None:
        """Remove a per-signal override, falling back to global config."""
        self._per_signal.pop(channel_name, None)

    def clear_all_signal_configs(self) -> None:
        """Remove all per-signal overrides."""
        self._per_signal.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Scaling computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_scaling_result(
        self,
        channel_name: str,
        unit: str | None,
        mode: EngineeringScalingMode,
    ) -> ScalingResult:
        """Return the resolved scaling factor for one channel at *mode*."""
        per_signal = self._per_signal.get(channel_name)
        return compute_scaling_factor(channel_name, unit, mode, self._global, per_signal)

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def global_config(self) -> GlobalScalingConfig:
        return self._global

    def per_signal_config(self, channel_name: str) -> SignalScalingConfig | None:
        return self._per_signal.get(channel_name)

    def has_signal_override(self, channel_name: str) -> bool:
        return channel_name in self._per_signal
