"""Session-level harmonic analytics registry — Phase 7.

Mirrors the design of PhasorRegistry and FrequencyRegistry:
  - mutable, session-scoped, not thread-safe (call from UI thread only)
  - caches classification results by channel name
  - holds active HarmonicConfig and HarmonicDisplayMode
  - provides bulk helpers for filtering eligible channels
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.analytics.harmonics.harmonic_models import (
    HarmonicChannelRole,
    HarmonicConfig,
    HarmonicDisplayMode,
)
from app.analytics.harmonics.harmonic_overlay import (
    HarmonicChannelResult,
    classify_harmonic_role,
)

if TYPE_CHECKING:
    from app.data.signal_metadata import SignalMetadata


class HarmonicRegistry:
    """Session-level harmonic classification and display-mode registry.

    Caches classification results to avoid repeated heuristic runs.
    Cache is keyed by channel name only.  Changing HarmonicConfig does NOT
    automatically invalidate the role cache (call ``clear_cache()`` explicitly
    when changing config if classification may be affected).
    """

    def __init__(
        self,
        config: HarmonicConfig | None = None,
        display_mode: HarmonicDisplayMode = HarmonicDisplayMode.OFF,
    ) -> None:
        self._config: HarmonicConfig = config or HarmonicConfig()
        self._display_mode: HarmonicDisplayMode = display_mode
        self._cached_roles: dict[str, HarmonicChannelRole] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: "SignalMetadata | None" = None,
        *,
        force_role: HarmonicChannelRole | None = None,
    ) -> HarmonicChannelResult:
        """Classify one channel and cache the role.

        Args:
            channel_name: Raw channel name.
            unit:         Engineering unit, or None.
            signal_meta:  Optional per-channel SignalMetadata.
            force_role:   Operator override.

        Returns:
            Immutable HarmonicChannelResult.
        """
        result = classify_harmonic_role(
            channel_name,
            unit=unit,
            signal_meta=signal_meta,
            force_role=force_role,
        )
        self._cached_roles[channel_name] = result.role
        return result

    def is_harmonic_eligible(
        self,
        name: str,
        unit: str | None = None,
        meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if channel is eligible (VOLTAGE_HARMONIC or CURRENT_HARMONIC)."""
        role = self.classify(name, unit, meta).role
        return role in (
            HarmonicChannelRole.VOLTAGE_HARMONIC,
            HarmonicChannelRole.CURRENT_HARMONIC,
        )

    def is_voltage(
        self,
        name: str,
        unit: str | None = None,
        meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if channel is classified as VOLTAGE_HARMONIC."""
        return self.classify(name, unit, meta).role == HarmonicChannelRole.VOLTAGE_HARMONIC

    def is_current(
        self,
        name: str,
        unit: str | None = None,
        meta: "SignalMetadata | None" = None,
    ) -> bool:
        """Return True if channel is classified as CURRENT_HARMONIC."""
        return self.classify(name, unit, meta).role == HarmonicChannelRole.CURRENT_HARMONIC

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk helpers
    # ─────────────────────────────────────────────────────────────────────────

    def harmonic_eligible_channels(
        self,
        names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: "dict[str, SignalMetadata] | None" = None,
    ) -> list[str]:
        """Return names classified as eligible (order preserved).

        Args:
            names:           All channel names to scan.
            units:           Optional {name: unit} mapping.
            signal_metadata: Optional {name: SignalMetadata} mapping.

        Returns:
            Subset of ``names`` that are VOLTAGE_HARMONIC or CURRENT_HARMONIC.
        """
        units = units or {}
        meta = signal_metadata or {}
        return [
            n for n in names
            if self.classify(n, units.get(n), meta.get(n)).role
            in (HarmonicChannelRole.VOLTAGE_HARMONIC, HarmonicChannelRole.CURRENT_HARMONIC)
        ]

    def voltage_harmonic_channels(
        self,
        names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: "dict[str, SignalMetadata] | None" = None,
    ) -> list[str]:
        """Return names classified as VOLTAGE_HARMONIC (order preserved)."""
        units = units or {}
        meta = signal_metadata or {}
        return [
            n for n in names
            if self.classify(n, units.get(n), meta.get(n)).role
            == HarmonicChannelRole.VOLTAGE_HARMONIC
        ]

    def current_harmonic_channels(
        self,
        names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: "dict[str, SignalMetadata] | None" = None,
    ) -> list[str]:
        """Return names classified as CURRENT_HARMONIC (order preserved)."""
        units = units or {}
        meta = signal_metadata or {}
        return [
            n for n in names
            if self.classify(n, units.get(n), meta.get(n)).role
            == HarmonicChannelRole.CURRENT_HARMONIC
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration and mode
    # ─────────────────────────────────────────────────────────────────────────

    def set_display_mode(self, mode: HarmonicDisplayMode) -> None:
        """Set the session-level harmonic display mode.

        Raises:
            TypeError: If mode is not a HarmonicDisplayMode instance.
        """
        if not isinstance(mode, HarmonicDisplayMode):
            raise TypeError(
                f"Expected HarmonicDisplayMode, got {type(mode).__name__!r}"
            )
        self._display_mode = mode

    def set_config(self, config: HarmonicConfig) -> None:
        """Replace the harmonic configuration.

        Note: does not automatically clear the classification cache because
        config changes do not affect role classification (only extraction
        parameters).  Call clear_cache() explicitly if needed.
        """
        self._config = config

    def clear_cache(self) -> None:
        """Remove all cached classification entries."""
        self._cached_roles.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def display_mode(self) -> HarmonicDisplayMode:
        return self._display_mode

    @property
    def config(self) -> HarmonicConfig:
        return self._config

    @property
    def cached_roles(self) -> dict[str, HarmonicChannelRole]:
        """Read-only snapshot of the classification cache."""
        return dict(self._cached_roles)
