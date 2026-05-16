"""Domain-aware engineering display policy for visualization labels.

This module intentionally does not implement generic SI auto-scaling. Powerwave
uses operational power-system units so axis labels keep their engineering
meaning stable during zoom, pan, and value changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineeringDisplayPreferences:
    """Future preference hook for engineering display conventions."""

    active_power_unit: str = "MW"
    reactive_power_unit: str = "MVar"
    frequency_unit: str = "Hz"
    rocof_unit: str = "Hz/s"
    per_unit: str = "pu"


@dataclass(frozen=True, slots=True)
class EngineeringAxisLabel:
    """Resolved label text and fixed operational unit for a plot axis."""

    text: str
    unit: str


DEFAULT_ENGINEERING_DISPLAY_PREFERENCES = EngineeringDisplayPreferences()

_GROUP_TITLES = {
    "voltage_raw": "Voltage Waveforms",
    "current_raw": "Current Waveforms",
    "voltage_rms": "Voltage RMS",
    "current_rms": "Current RMS",
    "power": "Power",
    "frequency": "Frequency",
    "rocof": "ROCOF",
    "other": "Other Analog Channels",
}


def infer_signal_type(channel_name: str, unit: str | None = None) -> str | None:
    """Infer a coarse engineering signal role from name/unit hints."""
    name = channel_name.strip().lower()
    unit_norm = (unit or "").strip().lower().replace(" ", "")

    if "rocof" in name or unit_norm in {"hz/s", "hz/sec", "hzpersecond"}:
        return "rocof"
    if "freq" in name or name in {"f", "hz"} or unit_norm in {"hz", "khz", "mhz"}:
        return "frequency"
    if "mvar" in name or "q_" in name or unit_norm in {"mvar", "mvarr"}:
        return "reactive_power"
    if "mw" in name or "demand" in name or "p_" in name or unit_norm in {"mw", "kmw", "mmw"}:
        return "active_power"
    if "pu" in name or unit_norm in {"pu", "p.u.", "perunit"}:
        return "per_unit"
    if name.startswith("v") or "volt" in name or unit_norm in {"v", "kv"}:
        return "voltage"
    if name.startswith("i") or "current" in name or unit_norm in {"a", "ka", "amp", "amps"}:
        return "current"
    return None


def normalize_engineering_unit(
    unit: str | None,
    *,
    signal_type: str | None = None,
    channel_name: str | None = None,
    preferences: EngineeringDisplayPreferences = DEFAULT_ENGINEERING_DISPLAY_PREFERENCES,
) -> str:
    """Return a fixed operational display unit for a channel.

    The return value is for display only; waveform values are not scaled here.
    """
    raw = (unit or "").strip()
    compact = raw.lower().replace(" ", "")
    role = signal_type or infer_signal_type(channel_name or "", raw)

    if role in {"active_power", "power"}:
        return preferences.active_power_unit
    if role == "reactive_power":
        return preferences.reactive_power_unit
    if role == "frequency":
        return preferences.frequency_unit
    if role == "rocof":
        return preferences.rocof_unit
    if role == "per_unit":
        return preferences.per_unit
    if role in {"voltage", "voltage_rms"}:
        if compact == "v":
            return "V"
        return "kV"
    if role in {"current", "current_rms"}:
        if compact == "ka":
            return "kA"
        return "A"

    known = {
        "mw": "MW",
        "mvar": preferences.reactive_power_unit,
        "mvarr": preferences.reactive_power_unit,
        "hz": preferences.frequency_unit,
        "hz/s": preferences.rocof_unit,
        "kv": "kV",
        "v": "V",
        "ka": "kA",
        "a": "A",
        "pu": preferences.per_unit,
        "p.u.": preferences.per_unit,
        "perunit": preferences.per_unit,
    }
    return known.get(compact, raw or "unknown")


def format_axis_label(
    channel_name: str,
    unit: str | None,
    *,
    signal_type: str | None = None,
) -> EngineeringAxisLabel:
    """Return a stable engineering label for a channel axis."""
    role = signal_type or infer_signal_type(channel_name, unit)
    return EngineeringAxisLabel(
        text=_display_name(channel_name, role),
        unit=normalize_engineering_unit(unit, signal_type=role, channel_name=channel_name),
    )


def format_panel_title(group_name: str, channel_count: int = 0, source_id: str | None = None) -> str:
    """Return a consistent grouped-panel title."""
    base = _GROUP_TITLES.get(group_name, _title_from_token(group_name))
    if group_name == "other" and channel_count:
        base = f"{base} ({channel_count})"
    if source_id:
        return f"{source_id} - {base}"
    return base


def format_rms_curve_label(channel_name: str, unit: str | None) -> str:
    """Return a readable label for an RMS overlay curve."""
    axis_label = format_axis_label(channel_name, unit)
    return f"{axis_label.text} RMS ({axis_label.unit})"


def format_rms_mode_title(base_title: str | None, mode_value: str) -> str:
    """Return the panel title with clear RMS mode indication."""
    suffix = {
        "overlay": "RMS Overlay",
        "rms_only": "RMS Only",
    }.get(mode_value)
    base = (base_title or "").strip()
    if suffix is None:
        return base
    return f"{base} - {suffix}" if base else suffix


def _display_name(channel_name: str, signal_type: str | None) -> str:
    name = channel_name.strip() or "Signal"
    if signal_type == "frequency" and name.lower() in {"f", "freq", "frequency"}:
        return "Frequency"
    if signal_type == "rocof":
        return "ROCOF"
    return name


def _title_from_token(token: str) -> str:
    return " ".join(part.capitalize() for part in token.replace("/", " ").replace("_", " ").split())
