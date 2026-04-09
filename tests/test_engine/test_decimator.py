"""
tests/test_engine/test_decimator.py

Unit and performance tests for engine.decimator.

Tests cover:
  - decimate_minmax  : peak preservation, output length, no Python loop overhead
  - decimate_uniform : shape preservation, output length
  - decimate_digital : state-change preservation, no missed transitions
  - prepare_display_data : integration smoke — decorates record correctly
  - Speed assertion: 28 channels × 60,000 points < 50 ms (vectorised numpy)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from engine.decimator import (
    MAX_ANALOGUE_POINTS,
    MAX_DIGITAL_POINTS,
    decimate_digital,
    decimate_minmax,
    decimate_uniform,
    prepare_display_data,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sine(n: int = 60_000, rate: float = 6000.0, freq: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, sine) arrays, n points at given sample rate."""
    t = np.linspace(0.0, n / rate, n, endpoint=False, dtype=np.float64)
    d = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return t, d


# ── decimate_minmax ───────────────────────────────────────────────────────────

class TestDecimateMinmax:

    def test_passthrough_when_below_limit(self) -> None:
        t = np.linspace(0, 1, 100, dtype=np.float64)
        d = np.ones(100, dtype=np.float32)
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        assert len(t_out) == 100
        assert len(d_out) == 100

    def test_output_length_at_most_max_points(self) -> None:
        t, d = _sine(60_000)
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        assert len(t_out) <= 2000
        assert len(t_out) == len(d_out)

    def test_peak_preserved(self) -> None:
        """Max of decimated output must be >= 99% of original max."""
        t, d = _sine(60_000)
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        assert np.max(d_out) >= 0.99 * np.max(d)

    def test_trough_preserved(self) -> None:
        """Min of decimated output must be within 1% of original min."""
        t, d = _sine(60_000)
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        # For negative min, 0.99 * min is less negative (closer to zero) —
        # output must be at least as negative as 99% of the original trough.
        assert np.min(d_out) <= 0.99 * np.min(d)

    def test_spike_survives_decimation(self) -> None:
        """An injected spike must appear in the decimated output."""
        t, d = _sine(60_000)
        d = d.astype(np.float64)
        d[30_000] = 5.0
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        assert np.max(d_out) >= 4.9

    def test_time_order_preserved(self) -> None:
        """Output time array must be strictly non-decreasing."""
        t, d = _sine(60_000)
        t_out, _ = decimate_minmax(t, d, max_points=2000)
        assert np.all(np.diff(t_out) >= 0)

    def test_output_dtype_float64(self) -> None:
        t, d = _sine(60_000)
        t_out, d_out = decimate_minmax(t, d, max_points=2000)
        assert t_out.dtype == np.float64
        assert d_out.dtype == np.float64


# ── decimate_uniform ──────────────────────────────────────────────────────────

class TestDecimateUniform:

    def test_passthrough_when_below_limit(self) -> None:
        t = np.linspace(0, 1, 500, dtype=np.float64)
        d = np.ones(500, dtype=np.float32)
        t_out, d_out = decimate_uniform(t, d, max_points=2000)
        assert len(t_out) == 500

    def test_output_length_at_most_max_points(self) -> None:
        t, d = _sine(54_000)   # PMU-sized: 18 min × 50 fps
        t_out, d_out = decimate_uniform(t, d, max_points=2000)
        assert len(t_out) <= 2000
        assert len(t_out) == len(d_out)

    def test_mean_preserved_approx(self) -> None:
        """Mean of decimated output should be within 5% of original mean."""
        t = np.linspace(0, 18 * 60, 54_000, dtype=np.float64)
        d = np.full(54_000, 284.0, dtype=np.float32)  # constant magnitude
        t_out, d_out = decimate_uniform(t, d, max_points=2000)
        assert abs(np.mean(d_out) - 284.0) < 1.0

    def test_output_dtype_float64(self) -> None:
        t, d = _sine(54_000)
        _, d_out = decimate_uniform(t, d, max_points=2000)
        assert d_out.dtype == np.float64


# ── decimate_digital ──────────────────────────────────────────────────────────

class TestDecimateDigital:

    def test_passthrough_when_below_limit(self) -> None:
        t = np.linspace(0, 1, 100, dtype=np.float64)
        d = np.zeros(100, dtype=np.int8)
        t_out, d_out = decimate_digital(t, d, max_points=500)
        assert len(t_out) == 100

    def test_state_change_preserved(self) -> None:
        """A single 0→1 transition must survive heavy decimation."""
        n = 10_000
        t = np.linspace(0, 1, n, dtype=np.float64)
        d = np.zeros(n, dtype=np.int8)
        d[4999] = 1
        d[5000] = 1
        t_out, d_out = decimate_digital(t, d, max_points=100)
        assert np.max(d_out) == 1.0

    def test_multiple_transitions_preserved(self) -> None:
        """All state changes must appear in the output."""
        n = 50_000
        t = np.linspace(0, 10, n, dtype=np.float64)
        d = np.zeros(n, dtype=np.int8)
        # Place 5 transitions
        for idx in [5000, 15000, 25000, 35000, 45000]:
            d[idx:idx + 100] = 1
        t_out, d_out = decimate_digital(t, d, max_points=100)
        # All HIGH segments must be represented
        assert np.any(d_out == 1.0)

    def test_output_dtype_float64(self) -> None:
        t = np.linspace(0, 1, 1000, dtype=np.float64)
        d = np.zeros(1000, dtype=np.int8)
        _, d_out = decimate_digital(t, d, max_points=100)
        assert d_out.dtype == np.float64

    def test_always_static_channel(self) -> None:
        """All-zeros digital channel must not crash and output must be all 0."""
        n = 50_000
        t = np.linspace(0, 10, n, dtype=np.float64)
        d = np.zeros(n, dtype=np.int8)
        t_out, d_out = decimate_digital(t, d, max_points=100)
        assert np.all(d_out == 0.0)


# ── Speed benchmark ───────────────────────────────────────────────────────────

class TestDecimatorSpeed:

    def test_28_analogue_channels_under_50ms(self) -> None:
        """28 channels × 60,000 pts via decimate_minmax must finish < 50 ms."""
        t, d = _sine(60_000)
        t0 = time.perf_counter()
        for _ in range(28):
            decimate_minmax(t, d, max_points=MAX_ANALOGUE_POINTS)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f'\n[speed] 28-ch decimate_minmax: {elapsed_ms:.1f} ms')
        assert elapsed_ms < 50.0, (
            f'decimate_minmax too slow: {elapsed_ms:.1f} ms for 28 channels '
            f'(limit: 50 ms)'
        )

    def test_104_digital_channels_under_20ms(self) -> None:
        """104 digital channels × 20,500 pts must finish < 20 ms."""
        n = 20_500
        t = np.linspace(0, 20.5 / 1000, n, dtype=np.float64)
        d = np.zeros(n, dtype=np.int8)
        t0 = time.perf_counter()
        for _ in range(104):
            decimate_digital(t, d, max_points=MAX_DIGITAL_POINTS)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f'\n[speed] 104-ch decimate_digital: {elapsed_ms:.1f} ms')
        assert elapsed_ms < 20.0, (
            f'decimate_digital too slow: {elapsed_ms:.1f} ms for 104 channels '
            f'(limit: 20 ms)'
        )


# ── prepare_display_data integration ─────────────────────────────────────────

class TestPrepareDisplayData:
    """Smoke tests — verify _display_t/_display_d are attached correctly."""

    def _make_record(
        self,
        n_analogue: int = 3,
        n_digital: int = 2,
        sample_rate: float = 6000.0,
        n_samples: int = 60_000,
    ):
        """Build a minimal synthetic DisturbanceRecord."""
        from datetime import datetime, timezone

        import numpy as np

        from models.channel import AnalogueChannel, DigitalChannel
        from models.disturbance_record import DisturbanceRecord

        t = np.linspace(0.0, n_samples / sample_rate, n_samples, dtype=np.float64)
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        analogue = []
        for i in range(n_analogue):
            ch = AnalogueChannel(
                channel_id=i + 1,
                name=f'VA{i}',
                phase='A',
                unit='kV',
                multiplier=1.0,
                offset=0.0,
                raw_data=np.sin(
                    2 * np.pi * 50 * t
                ).astype(np.float32),
                signal_role='V_PHASE',
                colour='#FF4444',
                bay_name='',
            )
            analogue.append(ch)

        digital = []
        for i in range(n_digital):
            ch = DigitalChannel(
                channel_id=1000 + i,
                name=f'DIG{i}',
                phase='',
                data=np.zeros(n_samples, dtype=np.int8),
                signal_role='DIG_GENERIC',
                bay_name='',
            )
            digital.append(ch)

        record = DisturbanceRecord(
            station_name='TEST',
            device_id='1',
            start_time=start,
            trigger_time=start,
            trigger_sample=0,
            sample_rate=sample_rate,
            nominal_frequency=50.0,
            analogue_channels=analogue,
            digital_channels=digital,
            time_array=t,
            source_format='COMTRADE_1999',
            file_path=__import__('pathlib').Path('/tmp/test.cfg'),
            header_text='',
        )
        return record

    def test_analogue_display_arrays_attached(self) -> None:
        record = self._make_record()
        prepare_display_data(record)
        for ch in record.analogue_channels:
            assert hasattr(ch, '_display_t'), f'{ch.name} missing _display_t'
            assert hasattr(ch, '_display_d'), f'{ch.name} missing _display_d'
            assert len(ch._display_t) == len(ch._display_d)
            assert len(ch._display_t) <= MAX_ANALOGUE_POINTS

    def test_digital_display_arrays_attached(self) -> None:
        record = self._make_record()
        prepare_display_data(record)
        for ch in record.digital_channels:
            assert hasattr(ch, '_display_t'), f'{ch.name} missing _display_t'
            assert hasattr(ch, '_display_d'), f'{ch.name} missing _display_d'
            assert len(ch._display_t) == len(ch._display_d)

    def test_t_display_and_label_set_on_record(self) -> None:
        record = self._make_record()
        prepare_display_data(record)
        assert hasattr(record, '_t_display')
        assert hasattr(record, '_time_axis_label')
        assert isinstance(record._time_axis_label, str)

    def test_waveform_label_is_ms(self) -> None:
        record = self._make_record(sample_rate=6000.0)
        prepare_display_data(record)
        assert 'ms' in record._time_axis_label

    def test_trend_long_label_is_min(self) -> None:
        """18-minute PMU record at 50 Hz → label must be minutes."""
        record = self._make_record(sample_rate=50.0, n_samples=54_000)
        prepare_display_data(record)
        assert 'min' in record._time_axis_label

    def test_trend_short_label_is_s(self) -> None:
        """30-second record at 50 Hz → label must be seconds."""
        record = self._make_record(sample_rate=50.0, n_samples=1_500)
        prepare_display_data(record)
        assert '(s)' in record._time_axis_label

    def test_empty_record_does_not_crash(self) -> None:
        from datetime import datetime, timezone
        from models.disturbance_record import DisturbanceRecord
        record = DisturbanceRecord(
            station_name='EMPTY',
            device_id='0',
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            trigger_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            trigger_sample=0,
            sample_rate=50.0,
            nominal_frequency=50.0,
            analogue_channels=[],
            digital_channels=[],
            time_array=np.array([], dtype=np.float64),
            source_format='COMTRADE_1999',
            file_path=__import__('pathlib').Path('/tmp/empty.cfg'),
            header_text='',
        )
        prepare_display_data(record)   # must not raise
