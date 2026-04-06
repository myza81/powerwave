"""
tests/test_parsers/test_pmu_csv_parser.py

Integration tests for PmuCsvParser (src/parsers/pmu_csv_parser.py).

Real PMU files in tests/test_data/:
  500JMJG_U5.csv   — PMU ID 241, 500 kV, broken timestamp (GPS fault)
  275BAHS_KAWA1.csv — PMU ID 571, 275 kV, KAWA1_ column prefix, valid timestamps
  230PLTG_WAV81.csv — PMU ID 813, 230 kV, per-phase VA/VB/VC columns

Coverage:
  - is_pmu_csv() detection: True for PMU files, False for generic CSV
  - Metadata extraction: pmu_id, station_name from row 0
  - Column prefix stripping: KAWA1_V1 Magnitude → V1 Magnitude
  - Broken timestamp detection → synthetic time array at 50 fps
  - Valid timestamp parsing → UTC time array derived from SGT (UTC+8)
  - Voltage scaling: raw Volts → kV (÷ 1000)
  - Current scaling: raw Amps → kA (÷ 1000)
  - Signal role assignment: V1_PMU, I1_PMU, FREQ, ROCOF, V_PHASE
  - Phase assignment: Pos-seq for V1/I1, A/B/C for VA/VB/VC
  - Status digital channel creation
  - DisturbanceRecord fields: sample_rate=50, display_mode='TREND', source_format='PMU_CSV'
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from parsers.pmu_csv_parser import PmuCsvParser, is_pmu_csv
from models.channel import SignalRole
from models.disturbance_record import SourceFormat

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "test_data"

FILE_500KV   = DATA_DIR / "500JMJG_U5.csv"
FILE_275KV   = DATA_DIR / "275BAHS_KAWA1.csv"
FILE_230KV   = DATA_DIR / "230PLTG_WAV81.csv"
FILE_GENERIC = DATA_DIR / "synthetic_ambiguous.csv"


@pytest.fixture(scope="module")
def record_500kv():
    return PmuCsvParser().load(FILE_500KV)


@pytest.fixture(scope="module")
def record_275kv():
    return PmuCsvParser().load(FILE_275KV)


@pytest.fixture(scope="module")
def record_230kv():
    return PmuCsvParser().load(FILE_230KV)


# ── Helper ────────────────────────────────────────────────────────────────────

def _channel_by_role(record, role: str):
    """Return the first analogue channel matching signal_role, or None."""
    return next((ch for ch in record.analogue_channels if ch.signal_role == role), None)


def _channel_by_name(record, name: str):
    """Return the first analogue channel whose name equals name, or None."""
    return next((ch for ch in record.analogue_channels if ch.name == name), None)


# ── Group 1: is_pmu_csv() detection ──────────────────────────────────────────

class TestIsPmuCsv:
    def test_500kv_detected(self):
        assert is_pmu_csv(FILE_500KV) is True

    def test_275kv_detected(self):
        assert is_pmu_csv(FILE_275KV) is True

    def test_230kv_detected(self):
        assert is_pmu_csv(FILE_230KV) is True

    def test_generic_csv_not_detected(self):
        assert is_pmu_csv(FILE_GENERIC) is False


# ── Group 2: 500JMJG_U5.csv — PMU 241, 500 kV, broken timestamp ──────────────

class TestPmu500kv:

    def test_loads_without_error(self, record_500kv):
        assert record_500kv is not None

    def test_station_name(self, record_500kv):
        assert '500JMJG' in record_500kv.station_name

    def test_device_id(self, record_500kv):
        assert record_500kv.device_id == '241'

    def test_sample_rate(self, record_500kv):
        assert 40.0 <= record_500kv.sample_rate <= 60.0

    def test_display_mode_trend(self, record_500kv):
        assert record_500kv.display_mode == 'TREND'

    def test_source_format_pmu_csv(self, record_500kv):
        assert record_500kv.source_format == SourceFormat.PMU_CSV

    def test_time_array_length(self, record_500kv):
        assert len(record_500kv.time_array) == 54000

    def test_time_array_starts_at_zero(self, record_500kv):
        assert record_500kv.time_array[0] == pytest.approx(0.0, abs=1e-6)

    def test_time_array_dtype(self, record_500kv):
        assert record_500kv.time_array.dtype == np.float64

    def test_time_array_duration(self, record_500kv):
        # 54000 samples at 50 fps = 1080 s minus one interval = 1079.98 s
        assert record_500kv.time_array[-1] == pytest.approx(1079.98, abs=0.1)

    def test_time_array_monotonic(self, record_500kv):
        diffs = np.diff(record_500kv.time_array)
        assert (diffs >= 0).all(), "time_array must be non-decreasing"

    def test_time_step_approx_20ms(self, record_500kv):
        median_dt = float(np.median(np.diff(record_500kv.time_array)))
        assert median_dt == pytest.approx(0.020, abs=0.005)

    def test_n_analogue_channels(self, record_500kv):
        assert record_500kv.n_analogue == 6

    def test_freq_channel_present(self, record_500kv):
        ch = _channel_by_role(record_500kv, SignalRole.FREQ)
        assert ch is not None, "FREQ channel must be present"
        assert ch.unit == 'Hz'

    def test_freq_values_around_50hz(self, record_500kv):
        ch = _channel_by_role(record_500kv, SignalRole.FREQ)
        assert 45.0 <= float(ch.raw_data.mean()) <= 55.0

    def test_v1_pmu_channel_present(self, record_500kv):
        ch = _channel_by_name(record_500kv, 'V1 Magnitude')
        assert ch is not None, "V1 Magnitude channel must be present"
        assert ch.signal_role == SignalRole.V1_PMU
        assert ch.unit == 'kV'
        assert ch.phase == 'Pos-seq'

    def test_v1_magnitude_in_kv_range(self, record_500kv):
        ch = _channel_by_name(record_500kv, 'V1 Magnitude')
        max_val = float(np.abs(ch.raw_data).max())
        assert 400.0 <= max_val <= 600.0, f"V1 Magnitude max={max_val:.1f} kV not in 400–600 kV range"

    def test_i1_pmu_channel_present(self, record_500kv):
        ch = _channel_by_name(record_500kv, 'I1 Magnitude')
        assert ch is not None, "I1 Magnitude channel must be present"
        assert ch.signal_role == SignalRole.I1_PMU
        assert ch.unit == 'kA'

    def test_i1_magnitude_in_ka_range(self, record_500kv):
        ch = _channel_by_name(record_500kv, 'I1 Magnitude')
        max_val = float(np.abs(ch.raw_data).max())
        assert 0.5 <= max_val <= 5.0, f"I1 Magnitude max={max_val:.4f} kA not in 0.5–5.0 kA range"

    def test_gps_fault_noted_in_header(self, record_500kv):
        assert 'GPS: LOW' in record_500kv.header_text

    def test_status_digital_channel_present(self, record_500kv):
        assert record_500kv.n_digital >= 1
        assert record_500kv.digital_channels[0].name == 'Status'


# ── Group 3: 275BAHS_KAWA1.csv — PMU 571, 275 kV, KAWA1_ prefix ──────────────

class TestPmu275kv:

    def test_station_name(self, record_275kv):
        assert '275BAHS' in record_275kv.station_name

    def test_device_id(self, record_275kv):
        assert record_275kv.device_id == '571'

    def test_kawa1_prefix_stripped(self, record_275kv):
        # Column was "KAWA1_V1 Magnitude" — must appear as "V1 Magnitude"
        names = [ch.name for ch in record_275kv.analogue_channels]
        assert 'V1 Magnitude' in names, f"Expected 'V1 Magnitude' in {names}"
        assert not any('KAWA1' in n for n in names), \
            "KAWA1_ prefix must be stripped from all channel names"

    def test_v1_magnitude_in_275kv_range(self, record_275kv):
        ch = _channel_by_name(record_275kv, 'V1 Magnitude')
        max_val = float(np.abs(ch.raw_data).max())
        assert 200.0 <= max_val <= 320.0, f"V1 max={max_val:.1f} kV not in 200–320 kV range"

    def test_valid_timestamp_gps_ok(self, record_275kv):
        assert 'GPS: OK' in record_275kv.header_text

    def test_time_step_20ms(self, record_275kv):
        dt = record_275kv.time_array[1] - record_275kv.time_array[0]
        assert dt == pytest.approx(0.020, abs=0.001)

    def test_time_array_length(self, record_275kv):
        assert len(record_275kv.time_array) == 54000

    def test_source_format(self, record_275kv):
        assert record_275kv.source_format == SourceFormat.PMU_CSV

    def test_display_mode_trend(self, record_275kv):
        assert record_275kv.display_mode == 'TREND'


# ── Group 4: 230PLTG_WAV81.csv — PMU 813, 230 kV, per-phase voltages ─────────

class TestPmu230kv:

    def test_loads_without_error(self, record_230kv):
        assert record_230kv is not None

    def test_n_analogue_channels(self, record_230kv):
        assert record_230kv.n_analogue == 12

    def test_va_channel_present(self, record_230kv):
        ch = _channel_by_name(record_230kv, 'VA Magnitude')
        assert ch is not None, "VA Magnitude channel must be present"
        assert ch.signal_role == SignalRole.V_PHASE
        assert ch.phase == 'A'
        assert ch.unit == 'kV'

    def test_vb_channel_present(self, record_230kv):
        ch = _channel_by_name(record_230kv, 'VB Magnitude')
        assert ch is not None
        assert ch.phase == 'B'

    def test_vc_channel_present(self, record_230kv):
        ch = _channel_by_name(record_230kv, 'VC Magnitude')
        assert ch is not None
        assert ch.phase == 'C'

    def test_v1_magnitude_in_230kv_range(self, record_230kv):
        ch = _channel_by_name(record_230kv, 'V1 Magnitude')
        assert ch is not None
        max_val = float(np.abs(ch.raw_data).max())
        assert 180.0 <= max_val <= 280.0, f"V1 max={max_val:.1f} kV not in 180–280 kV range"

    def test_station_name(self, record_230kv):
        assert '230PLTG' in record_230kv.station_name

    def test_device_id(self, record_230kv):
        assert record_230kv.device_id == '813'

    def test_gps_fault_noted(self, record_230kv):
        assert 'GPS: LOW' in record_230kv.header_text
