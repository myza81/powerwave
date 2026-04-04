"""
tests/test_parsers/test_comtrade_integration.py

Milestone 1B Step 4 — End-to-end integration tests.

Each test loads a real CFG/DAT pair through the complete pipeline
(CFG parse → DAT read → signal role detection) and asserts engineering
correctness against known properties of the actual recordings.

Real test files
---------------
JMHE_500kV   — BEN32 fast record, 500 kV substation, ASCII, 5000 Hz
PMJY_275     — BEN32 slow record, FREQ + MW channels, ASCII, 20 Hz
Relay        — NARI multi-rate (4 sections: 1200/600/1200/600 Hz), MM/DD/YY date
NARI_relay   — NARI variable-rate (nrates=0), sample_rate from DAT timestamps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.comtrade_parser import ComtradeParser

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA = Path('tests/test_data')

JMHE    = DATA / 'JMHE_500kV.cfg'
PMJY    = DATA / 'PMJY_275.cfg'
RELAY   = DATA / 'Relay.cfg'           # NARI multi-rate, 4 sections
NARI_VR = DATA / 'NARI_relay.CFG'      # NARI variable-rate, nrates=0

ALL_CFG = [JMHE, PMJY, RELAY, NARI_VR]

# ── Helpers ───────────────────────────────────────────────────────────────────

def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


# ── Test 1: BEN32 fast record ─────────────────────────────────────────────────

class TestBEN32FastRecord:
    """JMHE_500kV — BEN32 500 kV substation fast record, 5000 Hz, ASCII."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(JMHE)

    def test_display_mode_is_waveform(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_sample_rate(self, record):
        assert record.sample_rate == 5000.0

    def test_channel_counts(self, record):
        assert len(record.analogue_channels) == 28
        assert len(record.digital_channels) == 168

    def test_bay_names_from_analogue_channels(self, record):
        # Z1230 appears only in digital names — not expected in bay_names
        assert set(record.bay_names) == {'LGNG1', 'LGNG2', 'GT1', 'GT2'}
        assert len(record.bay_names) == 4

    def test_bay_names_order_matches_channel_appearance(self, record):
        # LGNG1 is the first bay encountered in analogue channels
        assert record.bay_names[0] == 'LGNG1'

    def test_time_array_shape_and_dtype(self, record):
        assert record.time_array.dtype == np.float64
        assert len(record.time_array) == 17087

    def test_source_format(self, record):
        assert record.source_format == 'COMTRADE_1999'

    def test_station_and_device_id(self, record):
        assert record.station_name == 'JMHE (IF)'
        assert record.device_id == '1205'

    def test_trigger_sample_positive(self, record):
        assert record.trigger_sample > 0

    def test_all_analogue_channels_high_confidence(self, record):
        low_conf = [
            c.name for c in record.analogue_channels
            if c.role_confidence != 'HIGH'
        ]
        assert low_conf == [], f"Low-confidence analogue channels: {low_conf}"

    def test_v_phase_channels_detected(self, record):
        v_phase = [c for c in record.analogue_channels if c.signal_role == 'V_PHASE']
        # 4 bays × 3 phases = 12 V_PHASE channels
        assert len(v_phase) == 12

    def test_i_phase_channels_detected(self, record):
        i_phase = [c for c in record.analogue_channels if c.signal_role == 'I_PHASE']
        # 4 bays × 3 phases = 12 I_PHASE channels
        assert len(i_phase) == 12

    def test_i_earth_channels_detected(self, record):
        i_earth = [c for c in record.analogue_channels if c.signal_role == 'I_EARTH']
        # 4 bays × 1 earth = 4 I_EARTH channels
        assert len(i_earth) == 4

    def test_v_phase_phases_assigned(self, record):
        for ch in record.analogue_channels:
            if ch.signal_role == 'V_PHASE':
                assert ch.phase in ('A', 'B', 'C'), \
                    f"V_PHASE channel {ch.name!r} has unexpected phase {ch.phase!r}"

    def test_prefault_rms_lgng1_vr_plausible_for_500kv(self, record):
        """Engineering check: pre-fault RMS ≈ 500 / √3 ≈ 289 kV."""
        vr = next(
            c for c in record.analogue_channels
            if c.signal_role == 'V_PHASE' and c.phase == 'A'
            and c.bay_name == 'LGNG1'
        )
        prefault = vr.raw_data[:500]   # 500 samples = 0.1 s pre-trigger
        v_rms = rms(prefault)
        assert 150.0 < v_rms < 380.0, \
            f"LGNG1 VR pre-fault RMS = {v_rms:.1f} kV, expected 150–380 kV for 500 kV system"

    def test_digital_bay_assignment_from_known_bays(self, record):
        # Channels prefixed with a known bay name get that bay assigned
        lgng1_digs = [
            c for c in record.digital_channels if c.bay_name == 'LGNG1'
        ]
        assert len(lgng1_digs) > 0

    def test_digital_trip_roles_detected(self, record):
        trip_names = {c.name for c in record.digital_channels if c.signal_role == 'DIG_TRIP'}
        assert 'LGNG1 87L/1' in trip_names
        assert 'LGNG1 87L/2' in trip_names
        assert 'LGNG1 BU_21Z' in trip_names

    def test_digital_ar_roles_detected(self, record):
        ar_names = {c.name for c in record.digital_channels if c.signal_role == 'DIG_AR'}
        assert any('AR' in n or '79' in n for n in ar_names), \
            f"No AR channels found in: {ar_names}"

    def test_digital_intertrip_roles_detected(self, record):
        it_names = {c.name for c in record.digital_channels if c.signal_role == 'DIG_INTERTRIP'}
        assert any('85INTR' in n or '50BF' in n for n in it_names), \
            f"No INTERTRIP channels found in: {it_names}"

    def test_digital_commfail_is_generic(self, record):
        # "87L/1_COMMFAIL" must be DIG_GENERIC, not DIG_TRIP (alarm exception first)
        commfail = next(
            c for c in record.digital_channels if 'COMMFAIL' in c.name
        )
        assert commfail.signal_role == 'DIG_GENERIC'

    def test_no_derived_channels(self, record):
        # BEN32 fast record has no pre-calculated P/Q channels
        assert not any(c.is_derived for c in record.analogue_channels)

    def test_no_zero_offset_analogue(self, record):
        # All analogue offsets in this file are zero — verify offset=0 stored
        for ch in record.analogue_channels:
            assert ch.offset == 0.0, f"Channel {ch.name!r} has unexpected offset {ch.offset}"


# ── Test 2: BEN32 slow record ─────────────────────────────────────────────────

class TestBEN32SlowRecord:
    """PMJY_275 — BEN32 slow record, 20 Hz, FREQ + MW channels."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(PMJY)

    def test_display_mode_is_trend(self, record):
        assert record.display_mode == 'TREND'

    def test_sample_rate(self, record):
        assert record.sample_rate == 20.0

    def test_channel_counts(self, record):
        assert len(record.analogue_channels) == 12
        assert len(record.digital_channels) == 12

    def test_time_array_dtype(self, record):
        assert record.time_array.dtype == np.float64

    def test_long_duration(self, record):
        """Slow records capture extended events — duration should be >> 10 s."""
        assert record.duration > 10.0

    def test_no_v_phase_channels(self, record):
        v_phase = [c for c in record.analogue_channels if c.signal_role == 'V_PHASE']
        assert v_phase == []

    def test_no_i_phase_channels(self, record):
        i_phase = [c for c in record.analogue_channels if c.signal_role == 'I_PHASE']
        assert i_phase == []

    def test_freq_channels_detected(self, record):
        freq_chs = [c for c in record.analogue_channels if c.signal_role == 'FREQ']
        assert len(freq_chs) > 0

    def test_freq_channels_have_nonzero_offset(self, record):
        """BEN32 slow-record FREQ channels store deviation around 50 Hz — offset = 50.0."""
        for ch in record.analogue_channels:
            if ch.signal_role == 'FREQ':
                assert ch.offset != 0.0, \
                    f"FREQ channel {ch.name!r} expected non-zero offset, got {ch.offset}"

    def test_freq_channel_values_plausible(self, record):
        """After offset applied, FREQ values should be in normal grid range 49–51 Hz."""
        freq_ch = next(c for c in record.analogue_channels if c.signal_role == 'FREQ')
        assert float(freq_ch.raw_data.min()) > 49.0, \
            f"FREQ min {freq_ch.raw_data.min():.3f} Hz below 49 Hz"
        assert float(freq_ch.raw_data.max()) < 51.5, \
            f"FREQ max {freq_ch.raw_data.max():.3f} Hz above 51.5 Hz"

    def test_mw_channels_detected(self, record):
        mw_chs = [c for c in record.analogue_channels if c.signal_role == 'P_MW']
        assert len(mw_chs) > 0

    def test_mw_channels_are_derived(self, record):
        """MW values in slow record are pre-calculated — is_derived must be True."""
        for ch in record.analogue_channels:
            if ch.signal_role == 'P_MW':
                assert ch.is_derived is True, \
                    f"P_MW channel {ch.name!r} should have is_derived=True"

    def test_all_analogue_high_confidence(self, record):
        low = [c.name for c in record.analogue_channels if c.role_confidence != 'HIGH']
        assert low == [], f"Low-confidence channels: {low}"

    def test_source_format(self, record):
        assert record.source_format == 'COMTRADE_1999'


# ── Test 3: NARI multi-rate ───────────────────────────────────────────────────

class TestNARIMultiRate:
    """Relay.cfg — NARI relay, 4 rate sections (1200/600/1200/600 Hz), MM/DD/YY date."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(RELAY)

    def test_display_mode_is_waveform(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_primary_sample_rate(self, record):
        assert record.sample_rate == 1200.0

    def test_channel_counts(self, record):
        assert len(record.analogue_channels) == 9
        assert len(record.digital_channels) == 32

    def test_time_array_dtype(self, record):
        assert record.time_array.dtype == np.float64

    def test_time_array_nonuniform(self, record):
        """Multi-rate sections produce non-uniform dt (1/1200 and 1/600 Hz)."""
        dt = np.diff(record.time_array)
        assert not np.allclose(dt, dt[0], rtol=0.01), \
            "Expected non-uniform time array for multi-rate NARI record"

    def test_time_array_has_two_distinct_dt_values(self, record):
        dt = np.diff(record.time_array)
        # Two rates: 1200 Hz → dt≈0.000833 s, 600 Hz → dt≈0.001667 s
        unique_dts = np.unique(np.round(dt, 5))
        assert len(unique_dts) == 2, \
            f"Expected 2 distinct dt values, got {len(unique_dts)}: {unique_dts}"

    def test_mm_dd_yy_date_parsed_correctly(self, record):
        """06/13/04 must be parsed as MM/DD/YY → 2004-06-13, not day 6 month 13."""
        assert record.start_time.year == 2004
        assert record.start_time.month == 6
        assert record.start_time.day == 13

    def test_3i0_detected_as_i_earth(self, record):
        ch = next(c for c in record.analogue_channels if '3I0' in c.name)
        assert ch.signal_role == 'I_EARTH'
        assert ch.phase == 'N'
        assert ch.role_confidence == 'HIGH'

    def test_3u0_detected_as_v_residual(self, record):
        ch = next(c for c in record.analogue_channels if '3U0' in c.name)
        assert ch.signal_role == 'V_RESIDUAL'
        assert ch.phase == 'N'
        assert ch.role_confidence == 'HIGH'

    def test_chinese_lowercase_ia_ib_ic(self, record):
        """NARI lowercase Ia/Ib/Ic must map to I_PHASE A/B/C."""
        expected = {'Ia': 'A', 'Ib': 'B', 'Ic': 'C'}
        for name_stripped, expected_phase in expected.items():
            ch = next(
                (c for c in record.analogue_channels if c.name.strip() == name_stripped),
                None,
            )
            assert ch is not None, f"Channel {name_stripped!r} not found"
            assert ch.signal_role == 'I_PHASE', \
                f"{name_stripped}: role={ch.signal_role}"
            assert ch.phase == expected_phase, \
                f"{name_stripped}: phase={ch.phase}, expected {expected_phase}"

    def test_chinese_lowercase_ua_ub_uc(self, record):
        """NARI lowercase Ua/Ub/Uc must map to V_PHASE A/B/C."""
        expected = {'Ua': 'A', 'Ub': 'B', 'Uc': 'C'}
        for name_stripped, expected_phase in expected.items():
            ch = next(
                (c for c in record.analogue_channels if c.name.strip() == name_stripped),
                None,
            )
            assert ch is not None, f"Channel {name_stripped!r} not found"
            assert ch.signal_role == 'V_PHASE', \
                f"{name_stripped}: role={ch.signal_role}"
            assert ch.phase == expected_phase, \
                f"{name_stripped}: phase={ch.phase}, expected {expected_phase}"

    def test_ux_detected_as_v_residual(self, record):
        ch = next(c for c in record.analogue_channels if c.name.strip() == 'Ux')
        assert ch.signal_role == 'V_RESIDUAL'
        assert ch.phase == 'N'

    def test_all_analogue_high_confidence(self, record):
        low = [c.name for c in record.analogue_channels if c.role_confidence != 'HIGH']
        assert low == [], f"Low-confidence: {low}"

    def test_nari_op_prefix_trip_channels(self, record):
        """Op_ prefix channels → DIG_TRIP."""
        op_trips = [
            c for c in record.digital_channels
            if c.name.startswith('Op_') and c.signal_role == 'DIG_TRIP'
        ]
        assert len(op_trips) > 0, "No Op_ trip channels detected"

    def test_nari_bi_prefix_generic(self, record):
        """BI_MCB_VT → alarm exception → DIG_GENERIC."""
        mcb = next(
            (c for c in record.digital_channels if 'MCB' in c.name), None
        )
        if mcb is not None:
            assert mcb.signal_role == 'DIG_GENERIC'

    def test_nari_send_recv_intertrip(self, record):
        """Send1/Recv1 → DIG_INTERTRIP."""
        intertrip = [c for c in record.digital_channels if c.signal_role == 'DIG_INTERTRIP']
        assert len(intertrip) > 0, "No INTERTRIP channels found"

    def test_source_format(self, record):
        assert record.source_format == 'COMTRADE_1999'


# ── Test 4: NARI variable-rate ────────────────────────────────────────────────

class TestNARIVariableRate:
    """NARI_relay.CFG — variable-rate (nrates=0), sample_rate from DAT timestamps."""

    @pytest.fixture(scope='class')
    def record(self):
        return ComtradeParser().load(NARI_VR)

    def test_sample_rate_derived_from_dat(self, record):
        """nrates=0 → sample_rate must be derived from actual DAT timestamps."""
        assert record.sample_rate > 0.0
        # Not the DEFAULT_VARIABLE_RATE placeholder (50 Hz) — real data has ~1200 Hz
        assert record.sample_rate > 100.0

    def test_display_mode(self, record):
        assert record.display_mode == 'WAVEFORM'

    def test_channel_counts(self, record):
        assert len(record.analogue_channels) == 22
        assert len(record.digital_channels) == 32

    def test_time_array_dtype(self, record):
        assert record.time_array.dtype == np.float64

    def test_time_array_populated(self, record):
        assert len(record.time_array) > 0

    def test_time_array_monotonic(self, record):
        assert np.all(np.diff(record.time_array) > 0), \
            "time_array must be strictly monotonically increasing"

    def test_current_channels_detected(self, record):
        i_chs = [c for c in record.analogue_channels if c.signal_role == 'I_PHASE']
        assert len(i_chs) > 0

    def test_voltage_channels_detected(self, record):
        v_chs = [c for c in record.analogue_channels
                 if c.signal_role in ('V_PHASE', 'V_RESIDUAL')]
        assert len(v_chs) > 0

    def test_idiff_trip_detected(self, record):
        """'Idiff Trip Z1' → DIG_TRIP."""
        trips = {c.name for c in record.digital_channels if c.signal_role == 'DIG_TRIP'}
        assert any('Idiff Trip' in n for n in trips), \
            f"'Idiff Trip' channels not in DIG_TRIP: {trips}"

    def test_alarm_channels_generic(self, record):
        """Circuit-fail alarm channels → DIG_GENERIC (alarm exception first)."""
        alm = next(
            (c for c in record.digital_channels if 'Alm' in c.name or 'Healt' in c.name),
            None,
        )
        if alm is not None:
            assert alm.signal_role == 'DIG_GENERIC', \
                f"Alarm channel {alm.name!r} should be DIG_GENERIC, got {alm.signal_role}"

    def test_pickup_channels_detected(self, record):
        """'PhComp Blk' and 'Idiff CZ Start' → DIG_PICKUP."""
        pickups = {c.name for c in record.digital_channels if c.signal_role == 'DIG_PICKUP'}
        assert any('PhComp' in n for n in pickups) or any('CZ Start' in n for n in pickups), \
            f"No expected pickup channels found in: {pickups}"

    def test_unused_channels_generic(self, record):
        """'Unused' digital channels → DIG_GENERIC (alarm exception via 'UNUSED' kw)."""
        unused = [c for c in record.digital_channels if 'Unused' in c.name or 'unused' in c.name]
        for ch in unused:
            assert ch.signal_role == 'DIG_GENERIC', \
                f"Unused channel {ch.name!r} should be DIG_GENERIC, got {ch.signal_role}"

    def test_source_format(self, record):
        assert record.source_format == 'COMTRADE_1999'


# ── Test 5: All files load end-to-end ────────────────────────────────────────

class TestAllFilesLoadClean:
    """Smoke test: every real CFG/DAT pair loads without exception."""

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_load_returns_record(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        assert record is not None

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_has_analogue_channels(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        assert len(record.analogue_channels) > 0

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_sample_rate_positive(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        assert record.sample_rate > 0.0

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_time_array_float64(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        assert record.time_array.dtype == np.float64

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_nominal_frequency_valid(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        assert record.nominal_frequency in (50.0, 60.0)

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_every_analogue_has_signal_role(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        for ch in record.analogue_channels:
            assert ch.signal_role, \
                f"[{cfg_path.name}] Channel {ch.name!r} has no signal_role"

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_every_digital_has_signal_role(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        for ch in record.digital_channels:
            assert ch.signal_role, \
                f"[{cfg_path.name}] Digital channel {ch.name!r} has no signal_role"

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_every_analogue_has_colour(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        for ch in record.analogue_channels:
            assert ch.colour, \
                f"[{cfg_path.name}] Channel {ch.name!r} has no colour"

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_raw_data_is_float32(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        for ch in record.analogue_channels:
            assert ch.raw_data.dtype == np.float32, \
                f"[{cfg_path.name}] {ch.name!r} raw_data dtype={ch.raw_data.dtype}"

    @pytest.mark.parametrize('cfg_path', ALL_CFG, ids=[p.name for p in ALL_CFG])
    def test_raw_data_length_matches_time_array(self, cfg_path):
        record = ComtradeParser().load(cfg_path)
        n = len(record.time_array)
        if n == 0:
            pytest.skip("Empty time_array (variable-rate with no DAT rows)")
        for ch in record.analogue_channels:
            assert len(ch.raw_data) == n, \
                f"[{cfg_path.name}] {ch.name!r}: raw_data len {len(ch.raw_data)} ≠ time_array len {n}"
