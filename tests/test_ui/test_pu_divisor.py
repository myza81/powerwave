"""
tests/test_ui/test_pu_divisor.py

Focused tests for legacy PU conversion policy.  These avoid constructing Qt
widgets and exercise the divisor logic directly on lightweight instances.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.channel import AnalogueChannel, SignalRole
from ui.rms_converter_dock import RmsConverterDock, SQRT3 as RMS_SQRT3
from ui.unified_canvas import UnifiedCanvasWidget, SQRT3 as CANVAS_SQRT3


def _channel(
    role: str,
    unit: str = "kV",
    channel_id: int = 1,
    phase: str = "A",
) -> AnalogueChannel:
    return AnalogueChannel(
        channel_id=channel_id,
        name="V",
        phase=phase,
        unit=unit,
        signal_role=role,
    )


def _canvas_with(ch: AnalogueChannel, convention: str = "auto") -> UnifiedCanvasWidget:
    canvas = UnifiedCanvasWidget.__new__(UnifiedCanvasWidget)
    canvas._files = {
        "f": SimpleNamespace(
            record=SimpleNamespace(analogue_channels=[ch]),
            voltage_convention=convention,
        )
    }
    canvas._base_kv = {("f", ch.channel_id): 275.0}
    canvas._pu_mode = True
    return canvas


def _dock_with(ch: AnalogueChannel, convention: str = "auto") -> RmsConverterDock:
    dock = RmsConverterDock.__new__(RmsConverterDock)
    dock._files = {
        "f": SimpleNamespace(
            record=SimpleNamespace(analogue_channels=[ch]),
            voltage_convention=convention,
        )
    }
    dock._base_kv = {("f", ch.channel_id): 275.0}
    return dock


class TestUnifiedCanvasPuDivisor:
    def test_per_unit_channel_is_not_divided_by_base_voltage(self) -> None:
        canvas = _canvas_with(_channel(SignalRole.V_PHASE, unit="pu"))

        assert canvas._get_pu_divisor("f", 1) == pytest.approx(1.0)
        assert canvas._display_unit_for_channel("f", canvas._files["f"].record.analogue_channels[0]) == "pu"

    def test_auto_phase_voltage_uses_line_to_earth_base(self) -> None:
        canvas = _canvas_with(_channel(SignalRole.V_PHASE))

        assert canvas._get_pu_divisor("f", 1) == pytest.approx(275.0 / CANVAS_SQRT3)

    def test_auto_line_voltage_uses_line_to_line_base(self) -> None:
        canvas = _canvas_with(_channel(SignalRole.V_LINE, phase="AB"))

        assert canvas._get_pu_divisor("f", 1) == pytest.approx(275.0)


class TestRmsConverterPuDivisor:
    def test_per_unit_channel_is_not_divided_by_base_voltage(self) -> None:
        dock = _dock_with(_channel(SignalRole.V_PHASE, unit="p.u."))

        assert dock._get_pu_divisor("f", 1) == pytest.approx(1.0)

    def test_auto_phase_voltage_uses_line_to_earth_base(self) -> None:
        dock = _dock_with(_channel(SignalRole.V_PHASE))

        assert dock._get_pu_divisor("f", 1) == pytest.approx(275.0 / RMS_SQRT3)

    def test_auto_positive_sequence_pmu_uses_line_to_line_base(self) -> None:
        dock = _dock_with(_channel(SignalRole.V1_PMU, phase="Pos-seq"))

        assert dock._get_pu_divisor("f", 1) == pytest.approx(275.0)
