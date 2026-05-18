"""Session-level phasor analytics registry.

Holds:
  - classification cache (channel name → PhasorChannelRole)
  - phasor configuration (nominal_hz, window_mode)
  - display mode (OFF / MAGNITUDE / ANGLE / SEQUENCE_COMPONENTS)
  - three-phase group detection cache

Mirrors the design of FrequencyRegistry: mutable, session-scoped, not thread-safe
(call only from the UI thread or from a single analytics worker).
"""
from __future__ import annotations

from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorChannelRole,
    PhasorChannelResult,
    PhasorConfig,
    PhasorDisplayMode,
    ThreePhaseGroup,
)
from app.analytics.phasors.phasor_overlay import (
    classify_phasor_role,
    detect_three_phase_groups,
    identify_phase,
)
from app.data.signal_metadata import SignalMetadata

# Standard panel key constants (matching channel_grouper display groups)
VOLTAGE_PHASOR_PANEL_KEY = "voltage_raw"
CURRENT_PHASOR_PANEL_KEY = "current_raw"

# Sequence component panel keys (introduced by Phase 6A)
SEQUENCE_VOLTAGE_PANEL_KEY = "sequence_voltage"
SEQUENCE_CURRENT_PANEL_KEY = "sequence_current"

# Multi-source suffix patterns for panel key detection
_PHASOR_PANEL_SUFFIXES = (
    "/voltage_raw", "/current_raw",
    "/sequence_voltage", "/sequence_current",
)


class PhasorRegistry:
    """Session-level phasor classification and display-mode registry.

    Caches classification results to avoid repeated heuristic runs.
    Cache is keyed by channel name only — changing the PhasorConfig requires
    calling clear_cache() to force reclassification.
    """

    def __init__(
        self,
        config: PhasorConfig | None = None,
        display_mode: PhasorDisplayMode = PhasorDisplayMode.OFF,
    ) -> None:
        self._config: PhasorConfig = config or PhasorConfig()
        self._display_mode: PhasorDisplayMode = display_mode
        self._cached_roles: dict[str, PhasorChannelRole] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        channel_name: str,
        unit: str | None = None,
        signal_meta: SignalMetadata | None = None,
        *,
        force_role: PhasorChannelRole | None = None,
    ) -> PhasorChannelResult:
        """Classify one channel and cache the result.

        Args:
            channel_name: Raw channel name.
            unit:         Engineering unit, or None.
            signal_meta:  Optional per-channel SignalMetadata.
            force_role:   Operator override; cached and returned immediately.

        Returns:
            Immutable PhasorChannelResult.
        """
        result = classify_phasor_role(
            channel_name, unit=unit, signal_meta=signal_meta, force_role=force_role
        )
        self._cached_roles[channel_name] = result.role
        return result

    def is_voltage(
        self,
        name: str,
        unit: str | None = None,
        meta: SignalMetadata | None = None,
    ) -> bool:
        """Return True if channel is classified as VOLTAGE_PHASOR."""
        return self.classify(name, unit, meta).role == PhasorChannelRole.VOLTAGE_PHASOR

    def is_current(
        self,
        name: str,
        unit: str | None = None,
        meta: SignalMetadata | None = None,
    ) -> bool:
        """Return True if channel is classified as CURRENT_PHASOR."""
        return self.classify(name, unit, meta).role == PhasorChannelRole.CURRENT_PHASOR

    def is_phasor_eligible(
        self,
        name: str,
        unit: str | None = None,
        meta: SignalMetadata | None = None,
    ) -> bool:
        """Return True if channel is voltage or current (eligible for phasor extraction)."""
        role = self.classify(name, unit, meta).role
        return role in (PhasorChannelRole.VOLTAGE_PHASOR, PhasorChannelRole.CURRENT_PHASOR)

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk helpers
    # ─────────────────────────────────────────────────────────────────────────

    def voltage_channels(
        self,
        names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: dict[str, SignalMetadata] | None = None,
    ) -> list[str]:
        """Return the subset of names classified as VOLTAGE_PHASOR (order preserved)."""
        units = units or {}
        meta = signal_metadata or {}
        return [
            n for n in names
            if self.classify(n, units.get(n), meta.get(n)).role
            == PhasorChannelRole.VOLTAGE_PHASOR
        ]

    def current_channels(
        self,
        names: list[str],
        units: dict[str, str] | None = None,
        signal_metadata: dict[str, SignalMetadata] | None = None,
    ) -> list[str]:
        """Return the subset of names classified as CURRENT_PHASOR (order preserved)."""
        units = units or {}
        meta = signal_metadata or {}
        return [
            n for n in names
            if self.classify(n, units.get(n), meta.get(n)).role
            == PhasorChannelRole.CURRENT_PHASOR
        ]

    def phasor_panel_keys(self, all_panel_keys: list[str]) -> list[str]:
        """Return panel keys that correspond to phasor or sequence component panels.

        Matches both direct keys ('voltage_raw', 'sequence_voltage') and
        multi-source keys ('CSV/voltage_raw', 'COMTRADE/sequence_current').

        Args:
            all_panel_keys: All panel keys from the current session layout.

        Returns:
            Subset of all_panel_keys that are phasor or sequence panels.
        """
        direct_keys = {
            VOLTAGE_PHASOR_PANEL_KEY,
            CURRENT_PHASOR_PANEL_KEY,
            SEQUENCE_VOLTAGE_PANEL_KEY,
            SEQUENCE_CURRENT_PANEL_KEY,
        }
        result = []
        for key in all_panel_keys:
            if key in direct_keys:
                result.append(key)
                continue
            for suffix in _PHASOR_PANEL_SUFFIXES:
                if key.endswith(suffix):
                    result.append(key)
                    break
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Three-phase group detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_three_phase_groups(
        self,
        channel_names: list[str],
        signal_metadata: dict[str, SignalMetadata] | None = None,
        channel_phases: dict[str, str] | None = None,
    ) -> list[ThreePhaseGroup]:
        """Identify complete three-phase A/B/C groups among channel names.

        Returns a list of ThreePhaseGroup objects where all three phases
        (A, B, C) have been identified. Incomplete groups are excluded.

        Args:
            channel_names:   Analog channel names to scan.
            signal_metadata: Optional per-channel SignalMetadata.
            channel_phases:  Optional AnalogChannel.phase field values.

        Returns:
            List of complete ThreePhaseGroup objects.
        """
        raw = detect_three_phase_groups(channel_names, signal_metadata, channel_phases)
        return [
            ThreePhaseGroup(
                phase_a=g["a"],
                phase_b=g["b"],
                phase_c=g["c"],
                signal_type=str(g["signal_type"]),
                prefix=str(g["prefix"]),
            )
            for g in raw
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration and mode
    # ─────────────────────────────────────────────────────────────────────────

    def set_display_mode(self, mode: PhasorDisplayMode) -> None:
        """Set the session-level phasor display mode.

        Raises:
            TypeError: If mode is not a PhasorDisplayMode instance.
        """
        if not isinstance(mode, PhasorDisplayMode):
            raise TypeError(
                f"Expected PhasorDisplayMode, got {type(mode).__name__!r}"
            )
        self._display_mode = mode

    def set_config(self, config: PhasorConfig) -> None:
        """Replace the phasor configuration.

        Callers should call clear_cache() after changing config to ensure
        window-size-dependent phasor results are recomputed.
        """
        self._config = config

    def clear_cache(self) -> None:
        """Remove all classification cache entries."""
        self._cached_roles.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def display_mode(self) -> PhasorDisplayMode:
        return self._display_mode

    @property
    def config(self) -> PhasorConfig:
        return self._config

    @property
    def cached_roles(self) -> dict[str, PhasorChannelRole]:
        """Read-only view of the classification cache."""
        return dict(self._cached_roles)
