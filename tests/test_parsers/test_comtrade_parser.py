"""
tests/test_parsers/test_comtrade_parser.py

Step-1 integration tests for ComtradeParser (CFG parsing only — no DAT).
Each test loads a real disturbance-record file from tests/test_data/ and
verifies that the parser extracts the correct metadata.

Files under test:
  JMHE_500kV.cfg  — BEN32 fast (5000 Hz), 28A/168D, multi-bay
  NARI_relay.CFG  — Unknown IED, variable-rate (nrates=0), 22A/32D, 13-field
  PMJY_275.cfg    — BEN32 slow (20 Hz), 12A/12D, TREND mode, name-first channels
  PTAI_275.cfg    — BEN32 fast (5000 Hz), 42A/162D, multi-bay, uppercase KV/KA
  Relay.cfg       — NARI relay, multi-rate 4 sections, 9A/32D, short-field format
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.parsers.comtrade_parser import (
    ComtradeParser,
    build_time_array,
    extract_bay_from_analogue_name,
    extract_bay_from_digital_name,
    parse_rev_year,
    _parse_timestamp,
)

# ── Path helper ───────────────────────────────────────────────────────────────

TEST_DATA = Path('tests/test_data')


# ── Unit tests for helper functions ──────────────────────────────────────────

class TestParseRevYear:
    """parse_rev_year must normalise all non-standard values to '1999'."""

    def test_valid_1991(self):
        assert parse_rev_year('1991') == '1991'

    def test_valid_1999(self):
        assert parse_rev_year('1999') == '1999'

    def test_valid_2013(self):
        assert parse_rev_year('2013') == '2013'

    def test_missing_field(self):
        # NARI omits the third field on line 1
        assert parse_rev_year('') == '1999'

    def test_whitespace_only(self):
        assert parse_rev_year('   ') == '1999'

    def test_ben32_calendar_year_2005(self):
        # BEN32 writes the recording year instead of the standard year
        assert parse_rev_year('2005') == '1999'

    def test_ben32_calendar_year_2025(self):
        assert parse_rev_year('2025') == '1999'

    def test_ben32_calendar_year_2001(self):
        # 2001 > 2013 is false; but 2001 is not in VALID_REV_YEARS → default
        assert parse_rev_year('2001') == '1999'


class TestExtractBayFromAnalogueName:
    """Bay extraction for BEN32 analogue channel names."""

    # Suffix format: "BAYNAME SIGNALCODE"
    def test_suffix_vr(self):
        bay, sig = extract_bay_from_analogue_name('SJTC2 VR')
        assert bay == 'SJTC2'
        assert sig == 'VR'

    def test_suffix_ib(self):
        bay, sig = extract_bay_from_analogue_name('KPAR IB')
        assert bay == 'KPAR'
        assert sig == 'IB'

    def test_suffix_multitoken_bay(self):
        bay, sig = extract_bay_from_analogue_name('GT UNIT1 VB')
        assert bay == 'GT UNIT1'
        assert sig == 'VB'

    # Name-first format: "SIGNALTYPE BAYNAME ..."
    def test_name_first_power(self):
        bay, sig = extract_bay_from_analogue_name('POWER PGPS 2')
        assert bay == 'PGPS 2'
        assert sig == 'POWER'

    def test_name_first_freq_simple(self):
        bay, sig = extract_bay_from_analogue_name('FREQ PGPS 2')
        assert bay == 'PGPS 2'
        assert sig == 'FREQ'

    def test_name_first_freq_with_phase_code(self):
        # "FREQ UY PLTG 1" → phase code UY is stripped; bay = "PLTG 1"
        bay, sig = extract_bay_from_analogue_name('FREQ UY PLTG 1')
        assert bay == 'PLTG 1'
        assert sig == 'FREQ UY'

    def test_name_first_freq_ur(self):
        bay, sig = extract_bay_from_analogue_name('FREQ UR PGPS 1')
        assert bay == 'PGPS 1'
        assert sig == 'FREQ UR'

    def test_name_first_freq_ub(self):
        bay, sig = extract_bay_from_analogue_name('FREQ UB KTBR 2')
        assert bay == 'KTBR 2'
        assert sig == 'FREQ UB'

    # No structure detected
    def test_no_structure_single_word(self):
        bay, sig = extract_bay_from_analogue_name('Ia')
        assert bay == ''
        assert sig == 'Ia'

    def test_no_structure_nari_name(self):
        bay, sig = extract_bay_from_analogue_name('3I0')
        assert bay == ''


class TestExtractBayFromDigitalName:
    """Bay name lookup in digital channel names."""

    def test_prefix_match(self):
        result = extract_bay_from_digital_name('SJTC2 87L/1', {'SJTC2', 'KPAR'})
        assert result == 'SJTC2'

    def test_embedded_match(self):
        result = extract_bay_from_digital_name('OVER UR SJTC2 TRIP', {'SJTC2', 'KPAR'})
        assert result == 'SJTC2'

    def test_no_match_returns_empty(self):
        result = extract_bay_from_digital_name('SPARE', {'SJTC2', 'KPAR'})
        assert result == ''

    def test_empty_known_bays(self):
        result = extract_bay_from_digital_name('SJTC2 TRIP', set())
        assert result == ''


class TestBuildTimeArray:
    """build_time_array: correct lengths and uniformity checks."""

    def test_single_rate_section(self):
        sections = [{'rate': 1000.0, 'end_sample': 5000}]
        t = build_time_array(sections)
        assert len(t) == 5000
        assert t.dtype == np.float64
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(4999 / 1000.0)

    def test_two_rate_sections_non_uniform(self):
        sections = [
            {'rate': 1200.0, 'end_sample': 192},
            {'rate': 600.0,  'end_sample': 2992},
        ]
        t = build_time_array(sections)
        assert len(t) == 2992
        dt = np.diff(t)
        # Array is NOT uniform — two distinct dt values exist
        assert not np.allclose(dt, dt[0], rtol=0.01)

    def test_four_rate_sections_relay(self):
        # Relay.cfg: 4 sections matching the real file
        sections = [
            {'rate': 1200.0, 'end_sample': 192},
            {'rate': 600.0,  'end_sample': 1872},
            {'rate': 1200.0, 'end_sample': 2064},
            {'rate': 600.0,  'end_sample': 2800},
        ]
        t = build_time_array(sections)
        assert len(t) == 2800
        assert t.dtype == np.float64

    def test_variable_rate_returns_empty(self):
        sections = [{'rate': 0.0, 'end_sample': 6024}]
        t = build_time_array(sections)
        assert len(t) == 0


class TestParseTimestamp:
    """_parse_timestamp: standard and non-standard date formats."""

    def test_standard_dd_mm_yyyy(self):
        dt = _parse_timestamp('30/03/2023', '17:23:49.252330')
        assert dt == datetime(2023, 3, 30, 17, 23, 49, 252330)

    def test_non_standard_mm_dd_yy(self):
        # Relay.cfg writes "06/13/04" — day 13 is not a valid month → MM/DD/YY
        dt = _parse_timestamp('06/13/04', '18:58:32.770000')
        assert dt == datetime(2004, 6, 13, 18, 58, 32, 770000)

    def test_microseconds_padded(self):
        dt = _parse_timestamp('01/01/2020', '00:00:00.1')
        assert dt.microsecond == 100000   # '1' padded to '100000'

    def test_no_fractional_seconds(self):
        dt = _parse_timestamp('15/07/2021', '12:30:00')
        assert dt.second == 0
        assert dt.microsecond == 0


# ── Integration tests against real CFG files ─────────────────────────────────

class TestJmhe500kV:
    """JMHE_500kV.cfg — BEN32 fast record, 5000 Hz, 28A/168D."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(TEST_DATA / 'JMHE_500kV.cfg')

    def test_station_name(self, record):
        assert record.station_name == 'JMHE (IF)'

    def test_device_id(self, record):
        assert record.device_id == '1205'

    def test_rev_year_in_source_format(self, record):
        assert record.source_format == 'COMTRADE_1999'

    def test_analogue_count(self, record):
        assert record.n_analogue == 28

    def test_digital_count(self, record):
        assert record.n_digital == 168

    def test_sample_rate(self, record):
        assert record.sample_rate == pytest.approx(5000.0)

    def test_display_mode_waveform(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_nominal_frequency(self, record):
        assert record.nominal_frequency == pytest.approx(50.0)

    def test_start_time(self, record):
        assert record.start_time == datetime(2023, 3, 30, 17, 23, 49, 252330)

    def test_trigger_time(self, record):
        assert record.trigger_time == datetime(2023, 3, 30, 17, 23, 49, 525233)

    def test_first_analogue_channel(self, record):
        ch = record.analogue_channels[0]
        assert ch.channel_id == 1
        assert ch.name == 'LGNG1 VR'
        assert ch.phase == 'A'
        assert ch.unit == 'kV'
        assert ch.multiplier == pytest.approx(0.0274660960)
        assert ch.offset == pytest.approx(0.0)
        assert ch.primary == pytest.approx(500.0)
        assert ch.ps_flag == 'P'
        assert ch.bay_name == 'LGNG1'

    def test_bay_names_populated(self, record):
        # JMHE has 4 bays: LGNG1, LGNG2, GT1, GT2
        assert 'LGNG1' in record.bay_names
        assert 'LGNG2' in record.bay_names
        assert 'GT1' in record.bay_names
        assert 'GT2' in record.bay_names

    def test_time_array_length(self, record):
        # 5000 Hz, 17087 end_sample
        assert len(record.time_array) == 17087

    def test_analogue_data_populated(self, record):
        # Step 2 — DAT read: all 17087 samples present on every analogue channel
        assert len(record.analogue_channels[0].raw_data) == 17087

    def test_digital_data_populated(self, record):
        assert len(record.digital_channels[0].data) == 17087


class TestNariRelayCfg:
    """NARI_relay.CFG — unknown IED, variable rate (nrates=0), 22A/32D, 13-field."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(TEST_DATA / 'NARI_relay.CFG')

    def test_station_name(self, record):
        assert record.station_name == 'PULU275'

    def test_device_id(self, record):
        assert record.device_id == '32'

    def test_analogue_count(self, record):
        assert record.n_analogue == 22

    def test_digital_count(self, record):
        assert record.n_digital == 32

    def test_rev_year_normalised(self, record):
        # rev_year '2001' → normalised to '1999'
        assert record.source_format == 'COMTRADE_1999'

    def test_nominal_frequency(self, record):
        assert record.nominal_frequency == pytest.approx(50.0)

    def test_first_analogue_channel_13field(self, record):
        # 13-field parse: primary and secondary should be populated
        ch = record.analogue_channels[0]
        assert ch.channel_id == 1
        assert ch.name == 'IX-T1'
        assert ch.unit == 'A'
        assert ch.multiplier == pytest.approx(8.287)
        assert ch.primary == pytest.approx(1.0)
        assert ch.ps_flag == 'P'

    def test_digital_5field(self, record):
        # 5-field digital parse
        ch = record.digital_channels[0]
        assert ch.channel_id == 1
        assert ch.name == 'Idiff Trip Z1'
        assert ch.normal_state == 0

    def test_variable_rate_resolved_from_dat(self, record):
        # nrates=0: after DAT reading, sample_rate is derived from timestamp deltas.
        # DAT timestamps are 0, 833, 1666 µs → dt=833µs → ~1200 Hz (WAVEFORM)
        assert record.sample_rate == pytest.approx(1200.0, abs=50.0)
        assert record.display_mode == 'WAVEFORM'

    def test_time_array_populated_from_dat_timestamps(self, record):
        # Variable-rate: time_array built from actual DAT timestamps, not CFG sections
        assert len(record.time_array) == 6024
        assert record.time_array[0] == pytest.approx(0.0)
        assert record.time_array[-1] == pytest.approx(5.012, abs=0.01)


class TestPmjy275:
    """PMJY_275.cfg — BEN32 slow record (20 Hz), TREND mode, 12A/12D."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(TEST_DATA / 'PMJY_275.cfg')

    def test_station_name(self, record):
        assert record.station_name == 'Ben_488'

    def test_device_id(self, record):
        assert record.device_id == '488'

    def test_analogue_count(self, record):
        assert record.n_analogue == 12

    def test_digital_count(self, record):
        assert record.n_digital == 12

    def test_sample_rate(self, record):
        assert record.sample_rate == pytest.approx(20.0)

    def test_display_mode_trend(self, record):
        assert record.display_mode == 'TREND'

    def test_rev_year_normalised(self, record):
        # rev_year '2005' → '1999'
        assert record.source_format == 'COMTRADE_1999'

    def test_freq_channel_offset(self, record):
        # FREQ channels store offset=50.0 (centred on nominal freq)
        freq_ch = next(c for c in record.analogue_channels if 'FREQ' in c.name)
        assert freq_ch.offset == pytest.approx(50.0)

    def test_bay_name_freq_pgps2(self, record):
        # "FREQ PGPS 2" → bay = "PGPS 2"
        ch = next(c for c in record.analogue_channels if c.name == 'FREQ PGPS 2')
        assert ch.bay_name == 'PGPS 2'

    def test_bay_name_freq_with_phase_code(self, record):
        # "FREQ UY PLTG 1" → bay = "PLTG 1" (phase code UY stripped)
        ch = next(c for c in record.analogue_channels if c.name == 'FREQ UY PLTG 1')
        assert ch.bay_name == 'PLTG 1'

    def test_bay_name_freq_ur(self, record):
        # "FREQ UR PGPS 1" → bay = "PGPS 1"
        ch = next(c for c in record.analogue_channels if c.name == 'FREQ UR PGPS 1')
        assert ch.bay_name == 'PGPS 1'

    def test_bay_names_unique_set(self, record):
        # 6 unique bays: PGPS 2, PLTG 1, KTBR 2, PGPS 1, PLTG 2, KTBR 1
        assert len(record.bay_names) == 6

    def test_time_array_length(self, record):
        # 20 Hz, 1344 end_sample
        assert len(record.time_array) == 1344

    def test_start_time(self, record):
        assert record.start_time == datetime(2022, 7, 27, 12, 49, 26, 958066)


class TestPtai275:
    """PTAI_275.cfg — BEN32 fast record (5000 Hz), 42A/162D, uppercase KV/KA units."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(TEST_DATA / 'PTAI_275.cfg')

    def test_station_name(self, record):
        assert record.station_name == 'PTAI (BEN5K)'

    def test_device_id(self, record):
        assert record.device_id == '1736'

    def test_analogue_count(self, record):
        assert record.n_analogue == 42

    def test_digital_count(self, record):
        assert record.n_digital == 162

    def test_sample_rate(self, record):
        assert record.sample_rate == pytest.approx(5000.0)

    def test_display_mode_waveform(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_rev_year_normalised(self, record):
        # '2005' → '1999'
        assert record.source_format == 'COMTRADE_1999'

    def test_uppercase_unit_preserved(self, record):
        # PTAI uses 'KV' and 'KA' — stored as-is from CFG
        ch = record.analogue_channels[0]
        assert ch.unit in ('KV', 'kV')   # accept either case

    def test_first_analogue_bay(self, record):
        # "KULN1 VR" → bay = "KULN1"
        ch = record.analogue_channels[0]
        assert ch.bay_name == 'KULN1'

    def test_multiple_bays_present(self, record):
        assert 'KULN1' in record.bay_names
        assert 'SLKS1' in record.bay_names
        assert 'BFLD1' in record.bay_names

    def test_time_array_length(self, record):
        # 5000 Hz, 151713 end_sample
        assert len(record.time_array) == 151713


class TestRelayCfg:
    """Relay.cfg — NARI relay, 4-section multi-rate, 9A/32D, short-field format."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(TEST_DATA / 'Relay.cfg')

    def test_station_name(self, record):
        assert record.station_name == 'NARI-RELAYS'

    def test_device_id(self, record):
        assert record.device_id == '0'

    def test_analogue_count(self, record):
        assert record.n_analogue == 9

    def test_digital_count(self, record):
        assert record.n_digital == 32

    def test_rev_year_missing_defaults_to_1999(self, record):
        # Line 1 has only 2 fields — rev_year absent → '1999'
        assert record.source_format == 'COMTRADE_1999'

    def test_sample_rate_from_first_section(self, record):
        # First section rate = 1200 Hz
        assert record.sample_rate == pytest.approx(1200.0)

    def test_display_mode_waveform(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_nominal_frequency(self, record):
        assert record.nominal_frequency == pytest.approx(50.0)

    def test_analogue_10field_no_primary(self, record):
        # NARI short format: primary and secondary absent → None, ps_flag → 'S'
        ch = record.analogue_channels[0]
        assert ch.name.strip() == '3I0'
        assert ch.unit == 'A'
        assert ch.multiplier == pytest.approx(0.005413)
        assert ch.primary is None
        assert ch.secondary is None
        assert ch.ps_flag == 'S'

    def test_digital_3field(self, record):
        # NARI short digital format: ch_num, name, normal_state
        ch = record.digital_channels[0]
        assert ch.name == 'FD'
        assert ch.normal_state == 0
        assert ch.phase == ''
        assert ch.ccbm == ''

    def test_time_array_length(self, record):
        # 4 sections: 192 + 1680 + 192 + 736 = 2800 total samples
        assert len(record.time_array) == 2800

    def test_time_array_non_uniform(self, record):
        # Two distinct dt values exist (1/1200 and 1/600)
        dt = np.diff(record.time_array)
        assert not np.allclose(dt, dt[0], rtol=0.01)

    def test_start_time_mm_dd_yy(self, record):
        # CFG writes "06/13/04" → month=6, day=13, year=2004
        assert record.start_time == datetime(2004, 6, 13, 18, 58, 32, 770000)

    def test_nari_digital_op_prefix(self, record):
        # Op_* channels should be present
        op_channels = [c for c in record.digital_channels if c.name.startswith('Op_')]
        assert len(op_channels) > 0

    def test_analogue_data_populated(self, record):
        assert len(record.analogue_channels[0].raw_data) == 2800

    def test_digital_data_populated(self, record):
        assert len(record.digital_channels[0].data) == 2800


# ── Step 2 DAT reading integration tests ─────────────────────────────────────

class TestDatReadingIntegration:
    """Verify DAT-reading correctness: physical values, offset application,
    variable-rate resolution, and complete load for all real test files."""

    def test_ptai275_first_analogue_plausible_for_275kv(self):
        """PTAI_275 is a 275 kV BEN32 station.

        First channel 'KULN1 VR' (phase-to-earth voltage):
          multiplier=0.0118743504, offset=0.0
          Nominal peak phase voltage ≈ 275 kV / √3 × √2 ≈ 225 kV.
          Assert max absolute value is within 100–500 kV.
        """
        record = ComtradeParser().load(TEST_DATA / 'PTAI_275.cfg')
        ch = record.analogue_channels[0]
        assert ch.name == 'KULN1 VR'
        max_abs = float(np.max(np.abs(ch.raw_data)))
        assert 100.0 < max_abs < 500.0

    def test_pmjy275_freq_offset_applied(self):
        """PMJY_275 is a BEN32 slow record (20 Hz) with FREQ channels.

        FREQ channels have offset=50.0 Hz in the CFG.  Without offset the
        raw values produce ~0.46 Hz (nonsense).  With offset applied the
        physical values must be in the plausible system frequency band 48–52 Hz.
        """
        record = ComtradeParser().load(TEST_DATA / 'PMJY_275.cfg')
        freq_ch = next(c for c in record.analogue_channels if 'FREQ' in c.name)
        assert freq_ch.offset == pytest.approx(50.0)
        mean_freq = float(np.mean(freq_ch.raw_data))
        assert 48.0 < mean_freq < 52.0, (
            f"FREQ mean={mean_freq:.3f} Hz — offset not applied correctly"
        )

    def test_nari_relay_variable_rate_resolved(self):
        """NARI_relay.CFG has nrates=0 (variable rate).

        After DAT reading, sample_rate must be derived from timestamp deltas
        (~833 µs → ~1200 Hz) and time_array populated from actual timestamps.
        display_mode must flip from TREND (50 Hz placeholder) to WAVEFORM.
        """
        record = ComtradeParser().load(TEST_DATA / 'NARI_relay.CFG')
        assert record.sample_rate == pytest.approx(1200.0, abs=50.0), (
            f"Expected ~1200 Hz from DAT timestamps, got {record.sample_rate:.1f}"
        )
        assert record.display_mode == 'WAVEFORM'
        assert len(record.time_array) == 6024
        assert record.time_array[0] == pytest.approx(0.0)

    @pytest.mark.parametrize('cfg_name', [
        'JMHE_500kV.cfg',
        'NARI_relay.CFG',
        'PMJY_275.cfg',
        'PTAI_275.cfg',
        'Relay.cfg',
    ])
    def test_all_real_files_load_without_error(self, cfg_name):
        """Every real test file must load completely without raising an exception."""
        record = ComtradeParser().load(TEST_DATA / cfg_name)
        assert record is not None
        assert record.n_analogue > 0
        assert record.n_digital >= 0
        # At least one channel must have sample data populated
        assert len(record.analogue_channels[0].raw_data) > 0
