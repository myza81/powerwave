"""Session-level frequency/ROCOF channel registry — Phase 5C.

FrequencyRegistry is the mutable session holder for frequency/ROCOF channel
state. It:
  - Caches channel role classifications (to avoid repeated heuristic runs)
  - Tracks the active FrequencyDisplayMode
  - Provides list helpers for extracting frequency/ROCOF channel names

One registry instance is expected per session and is passed to visualization
components that need to route or hide frequency/ROCOF channels.

The registry owns no waveform data, performs no computation, and has no Qt
dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.frequency.frequency_models import (
    FrequencyChannelResult,
    FrequencyChannelRole,
    FrequencyConfig,
    FrequencyDisplayMode,
)
from app.analytics.frequency.frequency_overlay import classify_frequency_role

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata


# Panel keys used by the main window for frequency/ROCOF grouped panels.
# Used by consumers to determine which panel canvases to show/hide.
FREQUENCY_PANEL_KEY = "frequency"
ROCOF_PANEL_KEY = "rocof"

# Suffix pattern used in multi-source panel keys ({source_id}/{group})
_FREQUENCY_SUFFIXES = ("/" + FREQUENCY_PANEL_KEY, "/" + ROCOF_PANEL_KEY)


class FrequencyRegistry:
    """Session-level tracker for frequency/ROCOF channel state.

    Usage::

        registry = FrequencyRegistry()
        result = registry.classify("SysFreq", unit="Hz")
        if registry.is_frequency("SysFreq"):
            ...
        registry.set_display_mode(FrequencyDisplayMode.OFF)
    """

    def __init__(
        self,
        config: FrequencyConfig | None = None,
        display_mode: FrequencyDisplayMode = FrequencyDisplayMode.PANEL_ONLY,
    ) -> None:
        self._config = config or FrequencyConfig()
        self._display_mode = display_mode
        self._role_cache: dict[str, FrequencyChannelResult] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def config(self) -> FrequencyConfig:
        return self._config

    @property
    def display_mode(self) -> FrequencyDisplayMode:
        return self._display_mode

    def set_display_mode(self, mode: FrequencyDisplayMode) -> None:
        """Change the active display mode. Clears the role cache."""
        if not isinstance(mode, FrequencyDisplayMode):
            raise TypeError(f"Expected FrequencyDisplayMode, got {type(mode)!r}")
        self._display_mode = mode
        # Cache remains valid — role classification is independent of display mode.

    def set_config(self, config: FrequencyConfig) -> None:
        """Replace the active FrequencyConfig (e.g. change nominal Hz)."""
        self._config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: "SignalMetadata | None" = None,
        *,
        force_role: FrequencyChannelRole | None = None,
    ) -> FrequencyChannelResult:
        """Classify a channel and cache the result.

        Cached results keyed by channel_name only. If a channel needs
        reclassification after metadata changes, call clear_cache() first.

        Args:
            channel_name: Channel / column name.
            unit:         Optional unit string.
            signal_meta:  Optional SignalMetadata for priority-chain fields.
            force_role:   If set, always returns this role (operator override).
                          The forced result is also cached.
        """
        cache_key = channel_name
        if cache_key in self._role_cache and force_role is None:
            return self._role_cache[cache_key]
        result = classify_frequency_role(
            channel_name, unit, signal_meta, force_role=force_role
        )
        self._role_cache[cache_key] = result
        return result

    def is_frequency(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if the channel is classified as FREQUENCY."""
        return (
            self.classify(channel_name, unit, signal_meta).role
            == FrequencyChannelRole.FREQUENCY
        )

    def is_rocof(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if the channel is classified as ROCOF."""
        return (
            self.classify(channel_name, unit, signal_meta).role
            == FrequencyChannelRole.ROCOF
        )

    def is_frequency_or_rocof(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if the channel is FREQUENCY or ROCOF."""
        role = self.classify(channel_name, unit, signal_meta).role
        return role in {FrequencyChannelRole.FREQUENCY, FrequencyChannelRole.ROCOF}

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk helpers
    # ─────────────────────────────────────────────────────────────────────────

    def frequency_channels(
        self,
        channel_names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: dict[str, "SignalMetadata"] | None = None,
    ) -> list[str]:
        """Return the subset of channel_names classified as FREQUENCY."""
        units = units or {}
        signal_metadata = signal_metadata or {}
        return [
            name
            for name in channel_names
            if self.classify(
                name,
                units.get(name),
                signal_metadata.get(name),
            ).role == FrequencyChannelRole.FREQUENCY
        ]

    def rocof_channels(
        self,
        channel_names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: dict[str, "SignalMetadata"] | None = None,
    ) -> list[str]:
        """Return the subset of channel_names classified as ROCOF."""
        units = units or {}
        signal_metadata = signal_metadata or {}
        return [
            name
            for name in channel_names
            if self.classify(
                name,
                units.get(name),
                signal_metadata.get(name),
            ).role == FrequencyChannelRole.ROCOF
        ]

    def frequency_panel_keys(self, all_panel_keys: list[str]) -> list[str]:
        """Return panel keys that belong to frequency/ROCOF groups.

        Matches both direct keys ("frequency", "rocof") and multi-source
        suffixed keys ("{source_id}/frequency", "{source_id}/rocof").

        Args:
            all_panel_keys: All active panel canvas keys in the session.

        Returns:
            Subset of keys that correspond to frequency/ROCOF panels.
        """
        result: list[str] = []
        for key in all_panel_keys:
            if key in {FREQUENCY_PANEL_KEY, ROCOF_PANEL_KEY}:
                result.append(key)
            elif any(key.endswith(suffix) for suffix in _FREQUENCY_SUFFIXES):
                result.append(key)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Cache management
    # ─────────────────────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear the classification cache (e.g. after metadata reload)."""
        self._role_cache.clear()

    @property
    def cached_roles(self) -> dict[str, FrequencyChannelRole]:
        """Read-only view of cached roles (for testing/debugging)."""
        return {k: v.role for k, v in self._role_cache.items()}
