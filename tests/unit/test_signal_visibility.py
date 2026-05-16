from __future__ import annotations

import dataclasses

from app.visualization.signal_visibility import (
    DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS,
    DEFAULT_MAX_VISIBLE_DIGITAL_SIGNALS,
    default_visible_analog_names,
    default_visible_digital_names,
)


@dataclasses.dataclass(frozen=True)
class _Channel:
    name: str


def _channels(count: int) -> list[_Channel]:
    return [_Channel(f"CH{index}") for index in range(count)]


def test_default_analog_visibility_keeps_small_panels_fully_visible() -> None:
    channels = _channels(DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS)

    assert default_visible_analog_names(channels) == [ch.name for ch in channels]


def test_default_analog_visibility_limits_large_panels_deterministically() -> None:
    channels = _channels(DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS + 3)

    assert default_visible_analog_names(channels) == [
        ch.name for ch in channels[:DEFAULT_MAX_VISIBLE_ANALOG_SIGNALS]
    ]


def test_default_digital_visibility_limits_large_timelines_deterministically() -> None:
    channels = _channels(DEFAULT_MAX_VISIBLE_DIGITAL_SIGNALS + 4)

    assert default_visible_digital_names(channels) == [
        ch.name for ch in channels[:DEFAULT_MAX_VISIBLE_DIGITAL_SIGNALS]
    ]
