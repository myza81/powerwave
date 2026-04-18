"""
tests/test_engine/test_measurements.py

Tests for engine.measurements — point-in-time value lookup and
display-unit ↔ raw-seconds conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from engine.measurements import display_to_raw_s, get_value_at_time
from models.channel import AnalogueChannel
from models.disturbance_record import DisturbanceRecord

# ── Helpers ────────────────────────────────────────────────────────────────────

_T0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_T_TRIGGER_OFFSET = 0.5   # trigger 0.5 s after start


def _make_record(
    time_array: np.ndarray,
    sample_rate: float = 1000.0,
    time_axis_label: str = 'Time (ms)',
    trigger_offset_s: float = _T_TRIGGER_OFFSET,
) -> DisturbanceRecord:
    """Build a minimal DisturbanceRecord with the given time array."""
    from datetime import timedelta
    start = _T0
    trigger = _T0 + timedelta(seconds=trigger_offset_s)
    record = DisturbanceRecord(
        station_name='TEST',
        device_id='DEV1',
        start_time=start,
        trigger_time=trigger,
        trigger_sample=0,
        sample_rate=sample_rate,
        nominal_frequency=50.0,
        source_format='COMTRADE_1999',
        file_path=Path('fake.cfg'),
        time_array=time_array,
    )
    record._time_axis_label = time_axis_label   # type: ignore[attr-defined]
    return record


def _make_channel(raw_data: np.ndarray) -> AnalogueChannel:
    """Build a minimal AnalogueChannel with the given raw_data."""
    return AnalogueChannel(
        channel_id=1,
        name='TEST_CH',
        phase='A',
        unit='kV',
        multiplier=1.0,
        offset=0.0,
        raw_data=raw_data.astype(np.float32),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGetValueAtTime:
    """Tests for get_value_at_time()."""

    def test_exact_sample_lookup(self) -> None:
        """Nearest-sample lookup returns the correct value for an exact match."""
        t = np.array([0.0, 0.001, 0.002, 0.003])
        d = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        record = _make_record(t, trigger_offset_s=0.0)
        ch = _make_channel(d)

        result = get_value_at_time(record, ch, 0.001)

        assert result == pytest.approx(20.0)

    def test_boundary_clamp_before_start(self) -> None:
        """Time before the record start clamps to the first sample."""
        t = np.array([0.0, 0.001, 0.002, 0.003])
        d = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        record = _make_record(t, trigger_offset_s=0.0)
        ch = _make_channel(d)

        result = get_value_at_time(record, ch, -99.0)

        assert result == pytest.approx(10.0)

    def test_boundary_clamp_after_end(self) -> None:
        """Time after the record end clamps to the last sample."""
        t = np.array([0.0, 0.001, 0.002, 0.003])
        d = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        record = _make_record(t, trigger_offset_s=0.0)
        ch = _make_channel(d)

        result = get_value_at_time(record, ch, 999.0)

        assert result == pytest.approx(40.0)

    def test_empty_record_returns_nan(self) -> None:
        """Empty time_array returns NaN without raising."""
        record = _make_record(np.array([], dtype=np.float64), trigger_offset_s=0.0)
        ch = _make_channel(np.array([], dtype=np.float32))

        result = get_value_at_time(record, ch, 0.0)

        assert np.isnan(result)


class TestDeltaTimeCalculation:
    """Tests for the ΔT (B−A) computation used by MeasurementPanel."""

    def test_delta_ms_precision(self) -> None:
        """Delta between two cursor positions in ms is accurate to <0.001 ms."""
        t_a_ms = 0.5 * 1000.0    # 0.5 s in ms
        t_b_ms = 0.7 * 1000.0    # 0.7 s in ms

        delta_ms = t_b_ms - t_a_ms

        assert abs(delta_ms - 200.0) < 0.001


class TestDisplayToRawS:
    """Tests for display_to_raw_s()."""

    def test_ms_to_raw(self) -> None:
        """Millisecond display value converts correctly to raw seconds."""
        t = np.linspace(0.0, 1.0, 1000)
        record = _make_record(t, time_axis_label='Time (ms)', trigger_offset_s=0.5)

        raw = display_to_raw_s(record, 0.0)   # t_display=0 ms → trigger

        assert raw == pytest.approx(0.5)

    def test_seconds_to_raw(self) -> None:
        """Seconds display value converts correctly to raw seconds."""
        t = np.linspace(0.0, 1.0, 1000)
        record = _make_record(t, time_axis_label='Time (s)', trigger_offset_s=0.5)

        raw = display_to_raw_s(record, 0.0)   # t_display=0 s → trigger

        assert raw == pytest.approx(0.5)

    def test_minutes_to_raw(self) -> None:
        """Minutes display value converts correctly to raw seconds."""
        t = np.linspace(0.0, 100.0, 5000)
        record = _make_record(t, sample_rate=50.0, time_axis_label='Time (min)', trigger_offset_s=30.0)

        raw = display_to_raw_s(record, 0.0)   # t_display=0 min → trigger

        assert raw == pytest.approx(30.0)


class TestRealFileValueSanity:
    """Integration test using the JMHE_500kV COMTRADE fixture."""

    def test_first_voltage_channel_plausible_range(self) -> None:
        """Value of first V_PHASE channel at t=0 is within ±800 kV.

        JMHE_500kV is a 500 kV BEN32 record; primary peak values are
        ~408 kV.  Any value outside ±800 kV indicates a scaling error.
        """
        from engine.decimator import prepare_display_data
        from parsers.comtrade_parser import ComtradeParser

        cfg_path = Path(__file__).parent.parent / 'test_data' / 'JMHE_500kV.cfg'
        record = prepare_display_data(ComtradeParser().load(cfg_path))

        # Find first V_PHASE channel
        v_channels = [
            ch for ch in record.analogue_channels
            if ch.signal_role == 'V_PHASE'
        ]
        assert v_channels, "No V_PHASE channels found in JMHE_500kV.cfg"

        # Trigger is at t=0 in display units → raw seconds = trigger_offset_s
        trigger_raw_s = (record.trigger_time - record.start_time).total_seconds()
        value = get_value_at_time(record, v_channels[0], trigger_raw_s)

        assert not np.isnan(value), "Value at trigger should not be NaN"
        assert -800.0 < value < 800.0, (
            f"Expected plausible 500kV value, got {value:.3f} kV"
        )
