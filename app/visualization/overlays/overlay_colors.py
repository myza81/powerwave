from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import Qt

from app.analytics.phasors.phasor_models import (
    PhaseLabel,
    PhasorDisplayMode,
)
from app.analytics.phasors.phasor_overlay import identify_phase


@dataclass(frozen=True, slots=True)
class OverlayTheme:
    """Central color/style policy for visualization overlays on a dark canvas."""

    voltage_phase_colors: dict[PhaseLabel, str] = field(default_factory=lambda: {
        PhaseLabel.A: "#FF7A70",
        PhaseLabel.B: "#FFE066",
        PhaseLabel.C: "#6FA8FF",
        PhaseLabel.UNKNOWN: "#D0D0D0",
    })
    current_phase_colors: dict[PhaseLabel, str] = field(default_factory=lambda: {
        PhaseLabel.A: "#FF9B8F",
        PhaseLabel.B: "#FFE98F",
        PhaseLabel.C: "#92BDFF",
        PhaseLabel.UNKNOWN: "#D0D0D0",
    })
    sequence_colors: dict[str, str] = field(default_factory=lambda: {
        "positive": "#80D17A",
        "negative": "#FF9F40",
        "zero": "#B58CFF",
    })
    magnitude_width: float = 1.7
    angle_width: float = 1.4
    sequence_width: float = 1.5


DEFAULT_OVERLAY_THEME = OverlayTheme()


_WAVEFORM_MEASUREMENT_KINDS = frozenset({
    "",
    "instantaneous",
    "sampled",
    "waveform",
    "raw",
    "unknown",
})
_NON_WAVEFORM_SIGNAL_TYPES = frozenset({
    "frequency",
    "rocof",
    "power",
    "rms",
    "telemetry",
    "calculated",
    "average",
})


def is_waveform_phasor_candidate(signal_meta: Any | None) -> bool:
    """Return False for RMS/telemetry/calculated channels unsuitable for DFT overlays."""
    if signal_meta is None:
        return True
    kind = str(getattr(signal_meta, "measurement_kind", "") or "").strip().lower()
    if kind and kind not in _WAVEFORM_MEASUREMENT_KINDS:
        return False
    signal_type = str(getattr(signal_meta, "signal_type", "") or "").strip().lower()
    display_group = str(getattr(signal_meta, "display_group", "") or "").strip().lower()
    electrical_type = str(getattr(signal_meta, "electrical_type", "") or "").strip().lower()
    if signal_type in _NON_WAVEFORM_SIGNAL_TYPES:
        return False
    if display_group in _NON_WAVEFORM_SIGNAL_TYPES:
        return False
    if electrical_type in {"frequency", "rocof", "power"}:
        return False
    return True


def phasor_color(
    channel_name: str,
    mode: PhasorDisplayMode,
    *,
    theme: OverlayTheme | None = None,
    signal_role: str | None = None,
    phase: PhaseLabel | None = None,
    base_color: str | None = None,
) -> str:
    """Return a deterministic overlay color for a phasor channel."""
    theme = theme or DEFAULT_OVERLAY_THEME
    if base_color is not None:
        if not _valid_hex(base_color):
            return "#44FFFF" if mode == PhasorDisplayMode.ANGLE else "#FFFFFF"
        if mode == PhasorDisplayMode.ANGLE:
            return _blend_hex(base_color, "#00FFFF", 0.40)
        return _blend_hex(base_color, "#FFFFFF", 0.60)

    phase = phase or identify_phase(channel_name)
    role = (signal_role or "").lower()
    if role == "current":
        color = theme.current_phase_colors.get(phase, theme.current_phase_colors[PhaseLabel.UNKNOWN])
    elif role == "voltage":
        color = theme.voltage_phase_colors.get(phase, theme.voltage_phase_colors[PhaseLabel.UNKNOWN])
    elif base_color is not None:
        color = base_color
    else:
        color = theme.voltage_phase_colors.get(phase, theme.voltage_phase_colors[PhaseLabel.UNKNOWN])

    if mode == PhasorDisplayMode.ANGLE:
        return _blend_hex(color, "#00FFFF", 0.35)
    return color


def phasor_pen(
    channel_name: str,
    mode: PhasorDisplayMode,
    *,
    theme: OverlayTheme | None = None,
    signal_role: str | None = None,
    phase: PhaseLabel | None = None,
    base_color: str | None = None,
) -> Any:
    """Return a PyQtGraph pen for a phasor magnitude or angle overlay."""
    theme = theme or DEFAULT_OVERLAY_THEME
    color = phasor_color(
        channel_name,
        mode,
        theme=theme,
        signal_role=signal_role,
        phase=phase,
        base_color=base_color,
    )
    if mode == PhasorDisplayMode.ANGLE:
        return pg.mkPen(color, width=theme.angle_width, style=Qt.PenStyle.DashDotLine)
    return pg.mkPen(color, width=theme.magnitude_width, style=Qt.PenStyle.DotLine)


def sequence_color(sequence_type: str, theme: OverlayTheme | None = None) -> str:
    """Return a deterministic color for positive, negative, or zero sequence."""
    theme = theme or DEFAULT_OVERLAY_THEME
    normalized = _normalize_sequence_type(sequence_type)
    return theme.sequence_colors.get(normalized, theme.sequence_colors["positive"])


def sequence_pen(sequence_type: str, theme: OverlayTheme | None = None) -> Any:
    """Return a PyQtGraph pen for sequence component curves."""
    theme = theme or DEFAULT_OVERLAY_THEME
    return pg.mkPen(
        sequence_color(sequence_type, theme),
        width=theme.sequence_width,
        style=Qt.PenStyle.SolidLine,
    )


def phasor_curve_label(channel_name: str, mode: PhasorDisplayMode) -> str:
    """Return a compact deterministic label for a phasor overlay curve."""
    suffix = "Magnitude" if mode == PhasorDisplayMode.MAGNITUDE else "Angle"
    return f"{channel_name} {suffix}"


def sequence_curve_label(signal_prefix: str, sequence_type: str) -> str:
    """Return a compact deterministic label such as V1 Magnitude or I2 Magnitude."""
    normalized = _normalize_sequence_type(sequence_type)
    index = {"positive": "1", "negative": "2", "zero": "0"}[normalized]
    prefix = signal_prefix.strip().upper()[:1] or "V"
    return f"{prefix}{index} Magnitude"


def _normalize_sequence_type(sequence_type: str) -> str:
    lower = str(sequence_type).strip().lower()
    if lower in {"positive", "pos", "1", "v1", "i1"} or lower.endswith("1"):
        return "positive"
    if lower in {"negative", "neg", "2", "v2", "i2"} or lower.endswith("2"):
        return "negative"
    if lower in {"zero", "0", "v0", "i0"} or lower.endswith("0"):
        return "zero"
    return "positive"


def _blend_hex(hex_color: str, target_hex: str, amount: float) -> str:
    source = hex_color.lstrip("#")
    target = target_hex.lstrip("#")
    if len(source) != 6 or len(target) != 6:
        return hex_color
    ratio = max(0.0, min(1.0, float(amount)))
    sr, sg, sb = int(source[0:2], 16), int(source[2:4], 16), int(source[4:6], 16)
    tr, tg, tb = int(target[0:2], 16), int(target[2:4], 16), int(target[4:6], 16)
    r = int(sr + (tr - sr) * ratio)
    g = int(sg + (tg - sg) * ratio)
    b = int(sb + (tb - sb) * ratio)
    return f"#{r:02X}{g:02X}{b:02X}"


def _valid_hex(hex_color: str) -> bool:
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic overlay colors (Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

_HARMONIC_ORDER_COLORS: dict[int, str] = {
    1:  "#808080",
    2:  "#FF9900",
    3:  "#FF6600",
    4:  "#FFCC00",
    5:  "#FF00CC",
    6:  "#CC00FF",
    7:  "#00CCFF",
    8:  "#00FFEE",
    9:  "#00FF88",
    10: "#88FF00",
    11: "#AA88FF",
    12: "#FF88FF",
    13: "#FF88AA",
}
_HARMONIC_DEFAULT_COLOR = "#BBBBBB"
_THD_VOLTAGE_COLOR = "#FF7070"
_THD_CURRENT_COLOR = "#70AAFF"


def harmonic_order_color(order: int) -> str:
    """Return a deterministic hex color for a given harmonic order."""
    return _HARMONIC_ORDER_COLORS.get(order, _HARMONIC_DEFAULT_COLOR)


def harmonic_order_pen(order: int, *, width: float = 1.5) -> Any:
    """Return a PyQtGraph pen for a harmonic magnitude overlay curve."""
    return pg.mkPen(
        harmonic_order_color(order),
        width=width,
        style=Qt.PenStyle.SolidLine,
    )


def thd_pen(role: str = "voltage", *, width: float = 1.7) -> Any:
    """Return a PyQtGraph pen for a THD trend curve."""
    color = _THD_VOLTAGE_COLOR if role.strip().lower() != "current" else _THD_CURRENT_COLOR
    return pg.mkPen(color, width=width, style=Qt.PenStyle.SolidLine)


def harmonic_curve_label(channel_name: str, order: int) -> str:
    """Return a compact deterministic label for a harmonic magnitude curve."""
    return f"{channel_name} H{order}"


def thd_curve_label(channel_name: str) -> str:
    """Return a compact deterministic label for a THD trend curve."""
    return f"{channel_name} THD%"
