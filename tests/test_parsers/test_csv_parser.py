"""
tests/test_parsers/test_csv_parser.py

Unit and integration tests for CsvParser (src/parsers/csv_parser.py).

Synthetic test files in tests/test_data/:
  synthetic_waveform_1000hz.csv  — datetime timestamps, V/I channels, 1000 Hz
  synthetic_trend_50hz.csv       — datetime timestamps, P/Q/Freq channels, 50 Hz
  synthetic_semicolon.csv        — semicolon separator, same V/I data
  synthetic_no_time_header.csv   — first column is seconds (header = "t_sec")
  synthetic_ambiguous.csv        — generic column names (ch1..ch4)

Coverage:
  - Separator auto-detection (comma and semicolon)
  - Time column detection (datetime and numeric seconds variants)
  - sample_rate derived from time deltas
  - display_mode WAVEFORM for ≥200 Hz, TREND for <200 Hz (LAW 9)
  - Unit inference from bracket syntax
  - signal_role_detector integration — at least one channel gets non-ANALOGUE role
  - NeedsMappingDialog raised when all headers are ambiguous
  - column_map provided: roles applied, dialog NOT raised
  - DisturbanceRecord fields populated correctly (LAW 5)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from models.channel import AnalogueChannel, RoleConfidence, SignalRole
from models.disturbance_record import DisturbanceRecord, SourceFormat
from parsers.csv_parser import CsvParser
from parsers.parser_exceptions import NeedsMappingDialog

# ── Path helper ───────────────────────────────────────────────────────────────

TEST_DATA = Path('tests/test_data')

WAVEFORM_CSV   = TEST_DATA / 'synthetic_waveform_1000hz.csv'
TREND_CSV      = TEST_DATA / 'synthetic_trend_50hz.csv'
SEMICOLON_CSV  = TEST_DATA / 'synthetic_semicolon.csv'
NO_TIME_CSV    = TEST_DATA / 'synthetic_no_time_header.csv'
AMBIGUOUS_CSV  = TEST_DATA / 'synthetic_ambiguous.csv'

# Real PMU CSV files (three-phase / positive-sequence recorder exports)
PLTG_WAV81_CSV = TEST_DATA / '230PLTG_WAV81.csv'
BAHS_KAWA1_CSV = TEST_DATA / '275BAHS_KAWA1.csv'
JMJG_U5_CSV    = TEST_DATA / '500JMJG_U5.csv'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, column_map=None) -> DisturbanceRecord:
    """Shorthand: load a CSV and return the DisturbanceRecord."""
    return CsvParser().load(path, column_map=column_map)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSeparatorDetection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeparatorDetection:
    """_detect_separator must identify comma and semicolon delimiters."""

    def test_comma_separator_default(self):
        parser = CsvParser()
        sep = parser._detect_separator(WAVEFORM_CSV)
        assert sep == ','

    def test_semicolon_separator_detected(self):
        parser = CsvParser()
        sep = parser._detect_separator(SEMICOLON_CSV)
        assert sep == ';'


# ═══════════════════════════════════════════════════════════════════════════════
# TestUnitInference
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnitInference:
    """_infer_unit must extract unit from bracket syntax."""

    def setup_method(self):
        self.parser = CsvParser()

    def test_parentheses_kv(self):
        assert self.parser._infer_unit('Va (kV)') == 'kV'

    def test_parentheses_ka(self):
        assert self.parser._infer_unit('Ia (kA)') == 'kA'

    def test_square_brackets_a(self):
        assert self.parser._infer_unit('Ia [A]') == 'A'

    def test_mw_unit(self):
        assert self.parser._infer_unit('P_MW (MW)') == 'MW'

    def test_mvar_unit(self):
        assert self.parser._infer_unit('Q_MVAR (MVAR)') == 'MVAR'

    def test_hz_unit(self):
        assert self.parser._infer_unit('Freq (Hz)') == 'Hz'

    def test_no_brackets_returns_empty(self):
        assert self.parser._infer_unit('Channel1') == ''

    def test_plain_name_returns_empty(self):
        assert self.parser._infer_unit('ch1') == ''


# ═══════════════════════════════════════════════════════════════════════════════
# TestTimeColumnDetection
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeColumnDetection:
    """_detect_time_column must identify the time column from header keywords."""

    def test_timestamp_header_detected(self):
        """'Timestamp' contains 'time' keyword → detected as datetime."""
        import pandas as pd
        df = pd.DataFrame({'Timestamp': ['2024-01-01 00:00:00'], 'Va (kV)': ['1.0']})
        col, ttype = CsvParser()._detect_time_column(df, None)
        assert col == 'Timestamp'
        assert ttype == 'datetime'

    def test_seconds_column_type(self):
        """Column with pure-numeric values guessed as 'seconds'."""
        import pandas as pd
        df = pd.DataFrame({'t_sec': ['0.0', '0.001', '0.002'], 'Va (kV)': ['1', '2', '3']})
        col, ttype = CsvParser()._detect_time_column(df, None)
        assert col == 't_sec'
        assert ttype == 'seconds'

    def test_column_map_overrides_detection(self):
        """column_map['time_column'] takes precedence over keyword scan."""
        import pandas as pd
        df = pd.DataFrame({'t': ['0.0', '0.001'], 'channel': ['1.0', '2.0']})
        col, ttype = CsvParser()._detect_time_column(df, {'time_column': 't'})
        assert col == 't'

    def test_fallback_to_first_column(self):
        """When no time keyword found, first column is used."""
        import pandas as pd
        df = pd.DataFrame({'ch1': ['0.0', '0.001'], 'ch2': ['1.0', '2.0']})
        col, ttype = CsvParser()._detect_time_column(df, None)
        assert col == 'ch1'


# ═══════════════════════════════════════════════════════════════════════════════
# TestSampleRateComputation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleRateComputation:
    """_compute_sample_rate must derive Hz from median time deltas."""

    def setup_method(self):
        self.parser = CsvParser()

    def test_1000hz(self):
        t = np.arange(100) * 0.001
        assert abs(self.parser._compute_sample_rate(t) - 1000.0) < 1.0

    def test_50hz(self):
        t = np.arange(500) * 0.02
        assert abs(self.parser._compute_sample_rate(t) - 50.0) < 0.1

    def test_single_sample_returns_default(self):
        t = np.array([0.0])
        assert self.parser._compute_sample_rate(t) == 50.0

    def test_empty_array_returns_default(self):
        t = np.array([])
        assert self.parser._compute_sample_rate(t) == 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestWaveformCsv — 1000 Hz datetime-based file
# ═══════════════════════════════════════════════════════════════════════════════

class TestWaveformCsv:
    """Standard CSV with datetime time column and V/I channels."""

    def setup_method(self):
        self.record = _load(WAVEFORM_CSV)

    def test_returns_disturbance_record(self):
        assert isinstance(self.record, DisturbanceRecord)

    def test_source_format_csv(self):
        assert self.record.source_format == SourceFormat.CSV

    def test_station_name_from_filename(self):
        assert self.record.station_name == 'synthetic_waveform_1000hz'

    def test_channel_count(self):
        # Va Vb Vc Ia Ib Ic = 6 analogue channels
        assert self.record.n_analogue == 6

    def test_no_digital_channels(self):
        assert self.record.n_digital == 0

    def test_sample_rate_approx_1000hz(self):
        assert abs(self.record.sample_rate - 1000.0) < 5.0

    def test_display_mode_waveform(self):
        # 1000 Hz >= 200 Hz threshold → WAVEFORM (LAW 9)
        assert self.record.display_mode == 'WAVEFORM'

    def test_time_array_length(self):
        assert len(self.record.time_array) == 100

    def test_time_array_starts_at_zero(self):
        assert self.record.time_array[0] == pytest.approx(0.0, abs=1e-9)

    def test_time_array_dtype_float64(self):
        assert self.record.time_array.dtype == np.float64

    def test_raw_data_dtype_float32(self):
        for ch in self.record.analogue_channels:
            assert ch.raw_data.dtype == np.float32

    def test_raw_data_length_matches_time_array(self):
        n = len(self.record.time_array)
        for ch in self.record.analogue_channels:
            assert len(ch.raw_data) == n

    def test_va_channel_unit_kv(self):
        va = next(c for c in self.record.analogue_channels if 'Va' in c.name)
        assert va.unit == 'kV'

    def test_ia_channel_unit_ka(self):
        ia = next(c for c in self.record.analogue_channels if 'Ia' in c.name)
        assert ia.unit == 'kA'

    def test_at_least_one_non_analogue_role(self):
        """signal_role_detector must assign at least one recognised role."""
        roles = {ch.signal_role for ch in self.record.analogue_channels}
        assert roles != {SignalRole.ANALOGUE}, (
            f"All channels have generic ANALOGUE role: {[c.name for c in self.record.analogue_channels]}"
        )

    def test_va_channel_role_v_phase(self):
        va = next(c for c in self.record.analogue_channels if 'Va' in c.name)
        assert va.signal_role == SignalRole.V_PHASE

    def test_ia_channel_role_i_phase(self):
        ia = next(c for c in self.record.analogue_channels if 'Ia' in c.name)
        assert ia.signal_role == SignalRole.I_PHASE

    def test_va_amplitude_reasonable(self):
        """Va should swing near ±230 kV."""
        va = next(c for c in self.record.analogue_channels if 'Va' in c.name)
        max_val = float(np.max(np.abs(va.raw_data)))
        assert 200.0 < max_val < 260.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestTrendCsv — 50 Hz file → TREND display mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrendCsv:
    """50 Hz CSV with P/Q/Freq channels → TREND mode."""

    def setup_method(self):
        self.record = _load(TREND_CSV)

    def test_sample_rate_approx_50hz(self):
        assert abs(self.record.sample_rate - 50.0) < 1.0

    def test_display_mode_trend(self):
        # 50 Hz < 200 Hz threshold → TREND (LAW 9)
        assert self.record.display_mode == 'TREND'

    def test_channel_count(self):
        # P_MW, Q_MVAR, Freq = 3 channels
        assert self.record.n_analogue == 3

    def test_time_array_length(self):
        assert len(self.record.time_array) == 500

    def test_p_mw_role(self):
        p = next(c for c in self.record.analogue_channels if 'P_MW' in c.name)
        assert p.signal_role == SignalRole.P_MW

    def test_q_mvar_role(self):
        q = next(c for c in self.record.analogue_channels if 'Q_MVAR' in c.name)
        assert q.signal_role == SignalRole.Q_MVAR

    def test_freq_role(self):
        f = next(c for c in self.record.analogue_channels if 'Freq' in c.name)
        assert f.signal_role == SignalRole.FREQ

    def test_nominal_frequency_default_50(self):
        assert self.record.nominal_frequency == 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestSemicolonCsv
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemicolonCsv:
    """Semicolon-separated CSV parsed correctly."""

    def setup_method(self):
        self.record = _load(SEMICOLON_CSV)

    def test_channel_count(self):
        assert self.record.n_analogue == 6

    def test_sample_rate_approx_1000hz(self):
        assert abs(self.record.sample_rate - 1000.0) < 5.0

    def test_display_mode_waveform(self):
        assert self.record.display_mode == 'WAVEFORM'

    def test_va_values_match_waveform(self):
        """Values in semicolon file should match comma file."""
        rec_comma = _load(WAVEFORM_CSV)
        rec_semi  = self.record
        va_comma = next(c for c in rec_comma.analogue_channels if 'Va' in c.name)
        va_semi  = next(c for c in rec_semi.analogue_channels  if 'Va' in c.name)
        np.testing.assert_allclose(
            va_comma.raw_data[:40], va_semi.raw_data[:40], rtol=1e-4
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestNoTimeHeaderCsv
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoTimeHeaderCsv:
    """CSV where first column is numeric seconds, not a datetime string."""

    def setup_method(self):
        self.record = _load(NO_TIME_CSV)

    def test_loads_without_error(self):
        assert isinstance(self.record, DisturbanceRecord)

    def test_channel_count(self):
        # Va and Vb — "t_sec" is the time column; not included as a channel
        assert self.record.n_analogue == 2

    def test_time_array_starts_at_zero(self):
        assert self.record.time_array[0] == pytest.approx(0.0, abs=1e-9)

    def test_sample_rate_approx_1000hz(self):
        # t_sec steps are 0.001 → 1000 Hz
        assert abs(self.record.sample_rate - 1000.0) < 5.0

    def test_va_role_detected(self):
        va = next(c for c in self.record.analogue_channels if 'Va' in c.name)
        assert va.signal_role == SignalRole.V_PHASE


# ═══════════════════════════════════════════════════════════════════════════════
# TestAmbiguousCsv — triggers NeedsMappingDialog
# ═══════════════════════════════════════════════════════════════════════════════

class TestAmbiguousCsv:
    """Generic column names (ch1..ch4) — NeedsMappingDialog must be raised."""

    def test_raises_needs_mapping_dialog(self):
        with pytest.raises(NeedsMappingDialog) as exc_info:
            _load(AMBIGUOUS_CSV)
        exc = exc_info.value
        assert isinstance(exc.columns, list)
        assert len(exc.columns) > 0

    def test_exception_carries_column_names(self):
        with pytest.raises(NeedsMappingDialog) as exc_info:
            _load(AMBIGUOUS_CSV)
        # ch2, ch3, ch4 should appear (ch1 is used as time fallback)
        cols = exc_info.value.columns
        assert any('ch' in c for c in cols)

    def test_with_column_map_no_exception(self):
        """Providing column_map bypasses dialog entirely."""
        column_map = {
            'time_column': 'ch1',
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
                'ch3': {'role': SignalRole.V_PHASE, 'phase': 'B', 'unit': 'kV'},
                'ch4': {'role': SignalRole.I_PHASE, 'phase': 'A', 'unit': 'A'},
            }
        }
        record = _load(AMBIGUOUS_CSV, column_map=column_map)
        assert isinstance(record, DisturbanceRecord)

    def test_column_map_roles_applied(self):
        """Roles from column_map are applied to channels."""
        column_map = {
            'time_column': 'ch1',
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
                'ch3': {'role': SignalRole.V_PHASE, 'phase': 'B', 'unit': 'kV'},
                'ch4': {'role': SignalRole.I_PHASE, 'phase': 'A', 'unit': 'A'},
            }
        }
        record = _load(AMBIGUOUS_CSV, column_map=column_map)
        ch2 = next(c for c in record.analogue_channels if c.name == 'ch2')
        assert ch2.signal_role == SignalRole.V_PHASE
        assert ch2.phase == 'A'
        assert ch2.unit == 'kV'
        assert ch2.role_confirmed is True

    def test_column_map_station_name_override(self):
        """column_map['station_name'] overrides the filename stem."""
        column_map = {
            'time_column': 'ch1',
            'station_name': 'TEST_STATION',
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
            }
        }
        record = _load(AMBIGUOUS_CSV, column_map=column_map)
        assert record.station_name == 'TEST_STATION'

    def test_column_map_nominal_frequency_override(self):
        """column_map['nominal_frequency'] overrides the 50.0 default."""
        column_map = {
            'time_column': 'ch1',
            'nominal_frequency': 60.0,
            'channels': {
                'ch2': {'role': SignalRole.V_PHASE, 'phase': 'A', 'unit': 'kV'},
            }
        }
        record = _load(AMBIGUOUS_CSV, column_map=column_map)
        assert record.nominal_frequency == 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestDisturbanceRecordContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestDisturbanceRecordContract:
    """Cross-cutting LAW 5 checks — every load must satisfy the model contract."""

    def test_file_path_stored(self):
        record = _load(WAVEFORM_CSV)
        assert record.file_path == WAVEFORM_CSV

    def test_trigger_sample_within_bounds(self):
        record = _load(WAVEFORM_CSV)
        assert 0 <= record.trigger_sample < len(record.time_array)

    def test_channel_ids_are_unique(self):
        record = _load(WAVEFORM_CSV)
        ids = [c.channel_id for c in record.analogue_channels]
        assert len(ids) == len(set(ids))

    def test_channel_ids_start_at_one(self):
        record = _load(WAVEFORM_CSV)
        assert record.analogue_channels[0].channel_id == 1

    def test_physical_data_shape_matches_raw(self):
        """physical_data property (LAW 10) must return same shape as raw_data."""
        record = _load(WAVEFORM_CSV)
        for ch in record.analogue_channels:
            assert ch.physical_data.shape == ch.raw_data.shape

    def test_multiplier_and_offset_defaults(self):
        """CSV channels have identity scaling (multiplier=1.0, offset=0.0)."""
        record = _load(WAVEFORM_CSV)
        for ch in record.analogue_channels:
            assert ch.multiplier == pytest.approx(1.0)
            assert ch.offset == pytest.approx(0.0)

    def test_colours_assigned(self):
        """Auto-assigned colours must be non-empty hex strings."""
        record = _load(WAVEFORM_CSV)
        for ch in record.analogue_channels:
            assert ch.colour.startswith('#')
            assert len(ch.colour) == 7


# ═══════════════════════════════════════════════════════════════════════════════
# TestRealPmuCsv_230PLTG — 230PLTG_WAV81.csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealPmuCsv_230PLTG:
    """Real PMU CSV from 230 kV PLTG station (recorder ID 813).

    CsvParser limitation with this file:
      The first line is a PMU metadata row
      ("ID: 813, Station Name: 230PLTG-WAV81,,,,,...") which pandas
      reads as column headers.  The actual data header row ("Date,
      Time(Asia),status,...") becomes the first data row.  Because the
      first column ('ID: 813') contains date strings ("10/15/25") that
      are non-numeric, the time array is empty and sample_rate falls
      back to the 50 Hz default.  CsvParser must NOT crash; it produces
      a degenerate but structurally valid DisturbanceRecord.
      Proper parsing of this file requires pmu_csv_parser.py.

    Expected channel layout after CsvParser:
      Unnamed:2..14 → 13 numeric data columns
      (status/freq/df/dt/V1-mag/ang/VA/VB/VC-mag/ang/I1-mag/ang).
    """

    def setup_method(self):
        self.record = _load(PLTG_WAV81_CSV)

    def test_loads_without_exception(self):
        assert isinstance(self.record, DisturbanceRecord)

    def test_source_format_csv(self):
        assert self.record.source_format == SourceFormat.CSV

    def test_sample_rate_positive(self):
        # True sample rate (50 fps) cannot be determined from the broken time
        # column; CsvParser defaults to 50 Hz.  Assert only that it is > 0.
        assert self.record.sample_rate > 0.0

    def test_display_mode_trend(self):
        # Default 50 Hz < 200 Hz threshold → TREND (LAW 9)
        assert self.record.display_mode == 'TREND'

    def test_channel_count(self):
        # 13 unnamed numeric columns survive _build_analogue_channels
        assert self.record.n_analogue == 13

    def test_at_least_one_non_analogue_role(self):
        """signal_role_detector assigns V_PHASE/I_PHASE from data values."""
        roles = {ch.signal_role for ch in self.record.analogue_channels}
        assert roles != {SignalRole.ANALOGUE}

    def test_time_array_dtype_float64(self):
        assert self.record.time_array.dtype == np.float64

    def test_time_array_monotonically_non_decreasing(self):
        ta = self.record.time_array
        if len(ta) > 1:
            assert np.all(np.diff(ta) >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRealPmuCsv_275BAHS — 275BAHS_KAWA1.csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealPmuCsv_275BAHS:
    """Real PMU CSV from 275 kV BAHS/KAWA1 station (recorder ID 571).

    CsvParser limitation with this file:
      The metadata row has exactly two comma-separated fields
      ("ID: 571, Station Name: 275BAHS-KAWA1"), so pandas creates a
      two-column DataFrame, silently truncating all data rows to their
      last two fields.  No column name contains a time keyword so the
      first column ('ID: 571') is used as time (treated as seconds).
      After the actual header row ('KAWA1_I1 Magnitude') is discarded as
      non-numeric, only one data column remains and all channels are
      ANALOGUE → NeedsMappingDialog is raised.  This is the correct
      graceful-degradation path for an unresolvable CSV file.
    """

    def test_raises_needs_mapping_dialog(self):
        """CsvParser raises NeedsMappingDialog — correct fallback for this file."""
        with pytest.raises(NeedsMappingDialog) as exc_info:
            _load(BAHS_KAWA1_CSV)
        assert isinstance(exc_info.value.columns, list)
        assert len(exc_info.value.columns) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestRealPmuCsv_500JMJG — 500JMJG_U5.csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealPmuCsv_500JMJG:
    """Real PMU CSV from 500 kV JMJG station unit 5 (recorder ID 241).

    CsvParser limitation with this file:
      Same metadata-row problem as 230PLTG: the first line
      ("ID: 241, Station Name: 500JMJG-U5,,,,,,") becomes column
      headers.  All timestamps are the broken "12:00.0" format (known
      PMU 241 GPS issue per CLAUDE.md), stored only in the date column
      which is non-numeric → empty time array and default 50 Hz.
      CsvParser must NOT crash.  Proper parsing requires pmu_csv_parser.py.

    Expected channel layout after CsvParser:
      Status column ('Unnamed: 2') is "00 00" → non-numeric → skipped.
      Unnamed:3..8 → 6 numeric channels
      (freq/df/dt/V1-mag/ang/I1-mag/ang).
    """

    def setup_method(self):
        self.record = _load(JMJG_U5_CSV)

    def test_loads_without_exception(self):
        assert isinstance(self.record, DisturbanceRecord)

    def test_source_format_csv(self):
        assert self.record.source_format == SourceFormat.CSV

    def test_sample_rate_positive(self):
        # True sample rate (50 fps) cannot be determined from the broken time
        # column; CsvParser defaults to 50 Hz.  Assert only that it is > 0.
        assert self.record.sample_rate > 0.0

    def test_display_mode_trend(self):
        # Default 50 Hz < 200 Hz threshold → TREND (LAW 9)
        assert self.record.display_mode == 'TREND'

    def test_channel_count(self):
        # 6 unnamed numeric columns survive _build_analogue_channels
        assert self.record.n_analogue == 6

    def test_at_least_one_non_analogue_role(self):
        """signal_role_detector assigns V_PHASE/I_PHASE from data values."""
        roles = {ch.signal_role for ch in self.record.analogue_channels}
        assert roles != {SignalRole.ANALOGUE}

    def test_time_array_dtype_float64(self):
        assert self.record.time_array.dtype == np.float64

    def test_time_array_monotonically_non_decreasing(self):
        ta = self.record.time_array
        if len(ta) > 1:
            assert np.all(np.diff(ta) >= 0.0)
