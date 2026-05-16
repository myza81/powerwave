"""Runtime signal visibility policy for Powerwave displays.

This module is intentionally provider-neutral. It works with channel-like
objects from DisturbanceRecord and returns channel names only; it never mutates
records, waveform arrays, or analytics results.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS = 8
DEFAULT_MAX_VISIBLE_DIGITAL_SIGNALS = 16


def _channel_names(channels: Iterable[Any]) -> list[str]:
    return [str(ch.name) for ch in channels if getattr(ch, "name", None)]


def default_visible_analog_names(
    channels: Iterable[Any],
    *,
    max_visible: int = DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS,
) -> list[str]:
    """Return deterministic default-visible analog channel names.

    Small panels remain fully visible. Large panels start with the first
    channels in record order so the initial view is readable and explainable,
    while the Signal Browser can reveal the rest without reloading data.
    """
    names = _channel_names(channels)
    if max_visible <= 0 or len(names) <= max_visible:
        return names
    return names[:max_visible]

def default_visible_digital_names(
    channels: Iterable[Any],
    *,
    max_visible: int = DEFAULT_MAX_VISIBLE_DIGITAL_SIGNALS,
) -> list[str]:
    """Return deterministic default-visible digital channel names."""
    names = _channel_names(channels)
    if max_visible <= 0 or len(names) <= max_visible:
        return names
    return names[:max_visible]
