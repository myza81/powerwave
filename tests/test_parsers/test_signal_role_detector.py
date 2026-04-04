"""
tests/test_parsers/test_signal_role_detector.py

Unit tests for src/parsers/signal_role_detector.py

Coverage:
  - detect_analogue_role() — all 8 priority rules
  - detect_digital_role()  — alarm exception + all role categories
  - detect_complementary_cb_pairs() — GCB/FCB OPEN+CLOSED wiring
  - detect_signal_roles()  — integration through the main entry point

Channel names drawn from real CFG files:
  JMHE_500kV.cfg  — BEN32 fast, R/Y/B naming, multi-bay
  PMJY_275.cfg    — BEN32 slow, POWER/FREQ name-first, MW channels
  Relay.cfg       — NARI relay, Chinese IEC (Ia/Ub/3I0), NARI digital prefixes
  NARI_relay.CFG  — IX-Tn / VXN current/voltage, Idiff digital channels
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.channel import (
    AnalogueChannel,
    DigitalChannel,
    RoleConfidence,
    SignalRole,
)
from src.parsers.signal_role_detector import (
    detect_analogue_role,
    detect_complementary_cb_pairs,
    detect_digital_role,
    detect_signal_roles,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_analogue(
    name: str,
    unit: str = '',
    offset: float = 0.0,
    raw_data: np.ndarray | None = None,
    phase: str = '',
) -> AnalogueChannel:
    """Create a minimal AnalogueChannel for testing."""
    return AnalogueChannel(
        channel_id=1,
        name=name,
        phase=phase,
        unit=unit,
        offset=offset,
        raw_data=raw_data if raw_data is not None else np.array([], dtype=np.float32),
    )


def _make_digital(name: str, channel_id: int = 1) -> DigitalChannel:
    """Create a minimal DigitalChannel for testing."""
    return DigitalChannel(channel_id=channel_id, name=name)


# ══════════════════════════════════════════════════════════════════════════════
# ANALOGUE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalogueRoleUnit:
    """Priority 1: unit field is the most reliable structural signal."""

    def test_kv_voltage_ben32(self):
        # "LGNG1 VR" with unit kV — from JMHE_500kV.cfg
        ch = _make_analogue('LGNG1 VR', unit='kV')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'A'
        assert conf == RoleConfidence.HIGH

    def test_kv_voltage_phase_b(self):
        ch = _make_analogue('LGNG1 VY', unit='kV')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'B'

    def test_kv_voltage_phase_c(self):
        ch = _make_analogue('GT1 VB', unit='kV')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'C'

    def test_ka_current_phase_a(self):
        # "LGNG1 IR" unit kA
        ch = _make_analogue('LGNG1 IR', unit='kA')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'A'
        assert conf == RoleConfidence.HIGH

    def test_ka_earth_current(self):
        ch = _make_analogue('LGNG1 IN', unit='kA')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_EARTH
        assert phase == 'N'

    def test_unit_hz_is_freq(self):
        # "FREQ PGPS 2" with unit Hz — PMJY_275.cfg
        ch = _make_analogue('FREQ PGPS 2', unit='Hz')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.FREQ
        assert conf == RoleConfidence.HIGH

    def test_unit_mw_is_p_mw(self):
        # "POWER PGPS 2" with unit MW — PMJY_275.cfg
        ch = _make_analogue('POWER PGPS 2', unit='MW')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.P_MW
        assert phase == '3-phase'
        assert conf == RoleConfidence.HIGH

    def test_unit_mw_sets_is_derived(self):
        ch = _make_analogue('POWER KTBR 1', unit='MW')
        detect_analogue_role(ch)
        assert ch.is_derived is True

    def test_unit_mvar_is_q_mvar(self):
        ch = _make_analogue('REACTIVE POWER UNIT1', unit='MVAR')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.Q_MVAR

    def test_unit_rpm_is_mech_speed(self):
        ch = _make_analogue('SHAFT SPEED', unit='RPM')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.MECH_SPEED
        assert conf == RoleConfidence.HIGH

    def test_unit_v_voltage(self):
        # NARI_relay.CFG uses V (not kV) for VXN channels
        ch = _make_analogue('VXN Z1', unit='V')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert conf == RoleConfidence.HIGH

    def test_unit_a_current(self):
        # NARI_relay.CFG uses A for IX-T channels
        ch = _make_analogue('IX-T1', unit='A')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert conf == RoleConfidence.HIGH


class TestAnalogueRoleOffset:
    """Priority 2: non-zero offset identifies DC/mechanical channels."""

    def test_field_current_by_offset_and_name(self):
        ch = _make_analogue('FIELD CURRENT', unit='', offset=-1875.0)
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.DC_FIELD_I
        assert conf == RoleConfidence.HIGH

    def test_field_voltage_by_offset_and_name(self):
        ch = _make_analogue('FIELD VOLTAGE', unit='', offset=120.0)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.DC_FIELD_V
        assert conf == RoleConfidence.HIGH

    def test_speed_by_offset_and_name(self):
        ch = _make_analogue('MECH SPEED', unit='', offset=3000.0)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.MECH_SPEED
        assert conf == RoleConfidence.HIGH

    def test_valve_by_offset_and_name(self):
        ch = _make_analogue('GOVERNOR VALVE', unit='', offset=50.0)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.MECH_VALVE
        assert conf == RoleConfidence.HIGH

    def test_freq_with_nonzero_offset(self):
        # BEN32 slow record FREQ channels have offset=50.0
        ch = _make_analogue('FREQ PGPS 2', unit='', offset=50.0)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.FREQ
        assert conf == RoleConfidence.HIGH


class TestAnalogueRolePowerPatterns:
    """Priority 3: power channel name patterns."""

    def test_power_prefix_no_unit(self):
        # "POWER PGPS 2" with blank unit (fallback path)
        ch = _make_analogue('POWER PGPS 2', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.P_MW
        assert conf == RoleConfidence.HIGH
        assert ch.is_derived is True

    def test_active_power_keyword(self):
        ch = _make_analogue('ACTIVE POWER UNIT', unit='')
        role, _, _ = detect_analogue_role(ch)
        assert role == SignalRole.P_MW

    def test_reactive_power_keyword(self):
        ch = _make_analogue('REACTIVE POWER UNIT', unit='')
        role, _, _ = detect_analogue_role(ch)
        assert role == SignalRole.Q_MVAR


class TestAnalogueRoleFreqPatterns:
    """Priority 4: frequency channel name patterns."""

    def test_freq_prefix(self):
        ch = _make_analogue('FREQ UY PLTG 1', unit='')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.FREQ
        assert conf == RoleConfidence.HIGH

    def test_rocof_keyword(self):
        ch = _make_analogue('DF/DT MEASUREMENT', unit='')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.ROCOF
        assert conf == RoleConfidence.HIGH

    def test_frequency_keyword(self):
        ch = _make_analogue('SYSTEM FREQUENCY', unit='')
        role, _, _ = detect_analogue_role(ch)
        assert role == SignalRole.FREQ


class TestAnalogueRoleSeqRms:
    """Priority 5: sequence RMS name patterns."""

    def test_zero_seq_rms(self):
        ch = _make_analogue('ZERO SEQ RMS', unit='kV')
        # unit kV fires priority 1 first; let's test without unit
        ch2 = _make_analogue('ZERO SEQ RMS', unit='')
        role, phase, conf = detect_analogue_role(ch2)
        assert role == SignalRole.SEQ_RMS
        assert conf == RoleConfidence.HIGH

    def test_neg_seq_rms(self):
        ch = _make_analogue('NEG SEQ RMS', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.SEQ_RMS
        assert phase == 'neg'

    def test_pos_seq_rms(self):
        ch = _make_analogue('POS SEQ RMS', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.SEQ_RMS
        assert phase == 'pos'


class TestAnalogueRoleSignalCode:
    """Priority 6: direct signal-code lookup."""

    # BEN32 R/Y/B codes
    def test_vr_ben32(self):
        ch = _make_analogue('SJTC2 VR', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'A'
        assert conf == RoleConfidence.HIGH

    def test_iy_ben32(self):
        ch = _make_analogue('KPAR IY', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'B'
        assert conf == RoleConfidence.HIGH

    def test_ib_ben32_phase_c(self):
        ch = _make_analogue('GT2 IB', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'C'

    def test_in_earth(self):
        ch = _make_analogue('GT1 IN', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.I_EARTH
        assert phase == 'N'

    # NARI Chinese convention (lowercase)
    def test_ia_nari(self):
        ch = _make_analogue(' Ia', unit='')    # leading space — as in Relay.cfg
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'A'
        assert conf == RoleConfidence.HIGH

    def test_ib_nari(self):
        ch = _make_analogue(' Ib', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'B'

    def test_ic_nari(self):
        ch = _make_analogue(' Ic', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert phase == 'C'

    def test_ua_nari(self):
        ch = _make_analogue(' Ua', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'A'

    def test_ub_nari(self):
        ch = _make_analogue(' Ub', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'B'

    def test_uc_nari(self):
        ch = _make_analogue(' Uc', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert phase == 'C'

    # Zero-sequence codes
    def test_3i0_earth(self):
        ch = _make_analogue('3I0', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_EARTH
        assert phase == 'N'
        assert conf == RoleConfidence.HIGH

    def test_3u0_residual(self):
        ch = _make_analogue('3U0', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_RESIDUAL
        assert phase == 'N'
        assert conf == RoleConfidence.HIGH

    def test_ux_residual(self):
        ch = _make_analogue(' Ux', unit='')
        role, phase, _ = detect_analogue_role(ch)
        assert role == SignalRole.V_RESIDUAL
        assert phase == 'N'


class TestAnalogueRolePatternMatch:
    """Priority 7: regex pattern matching on full name."""

    def test_ix_current_pattern_nari(self):
        # "IX-T1" without unit: 'X' is not in the [RYBABC] regex class, so
        # priority 7 does not match — falls to ANALOGUE/LOW.
        # In real files IX-Tn channels always carry unit='A' (priority 1).
        ch = _make_analogue('IX-T1', unit='')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.ANALOGUE
        assert conf == RoleConfidence.LOW

    def test_vxn_voltage_pattern(self):
        # "VXN Z1" without unit: 'X' not in [RYBABC], not a digit after V —
        # falls to ANALOGUE/LOW. Real files carry unit='V' (priority 1).
        ch = _make_analogue('VXN Z1', unit='')
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.ANALOGUE
        assert conf == RoleConfidence.LOW


class TestAnalogueRoleMagnitudeHeuristic:
    """Priority 8: magnitude heuristic from raw_data."""

    def test_high_magnitude_voltage(self):
        raw = np.array([132000.0, 133000.0, 131500.0], dtype=np.float32)
        ch = _make_analogue('UNKNOWN_CH', unit='', raw_data=raw)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.V_PHASE
        assert conf == RoleConfidence.LOW

    def test_low_magnitude_current(self):
        raw = np.array([1.2, 1.5, -1.3], dtype=np.float32)
        ch = _make_analogue('UNKNOWN_CH', unit='', raw_data=raw)
        role, _, conf = detect_analogue_role(ch)
        assert role == SignalRole.I_PHASE
        assert conf == RoleConfidence.LOW

    def test_no_data_fallback(self):
        ch = _make_analogue('MYSTERY', unit='')
        role, phase, conf = detect_analogue_role(ch)
        assert role == SignalRole.ANALOGUE
        assert conf == RoleConfidence.LOW


# ══════════════════════════════════════════════════════════════════════════════
# DIGITAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestDigitalAlarmException:
    """ALARM EXCEPTION must fire before any other rule."""

    def test_commfail_overrides_trip(self):
        # "87L/1_COMMFAIL" — contains both 87L (trip) and COMMFAIL (alarm)
        # Alarm exception must win.
        ch = _make_digital('LGNG1 87L/1_COMMFAIL')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC
        assert conf == RoleConfidence.HIGH

    def test_alarm_keyword(self):
        ch = _make_digital('Cct Fail Z1 Alm')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC

    def test_mcb_keyword(self):
        ch = _make_digital('BI_MCB_VT')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC
        assert conf == RoleConfidence.HIGH

    def test_vts_keyword(self):
        ch = _make_digital('Op_OC1_VTS')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC

    def test_unused_channel(self):
        ch = _make_digital('Unused')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC

    def test_virtual_input(self):
        ch = _make_digital('Virtual Input 40')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC

    def test_bi_en_z1(self):
        ch = _make_digital('BI_En_Z1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC

    def test_relay_healthy(self):
        ch = _make_digital('L32 Relay Healty')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC


class TestDigitalCB:
    """DIG_CB detection."""

    def test_cb_r_open_ben32(self):
        ch = _make_digital('LGNG1 CB_BUS_R_OPEN')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_CB
        assert conf == RoleConfidence.HIGH

    def test_gcb_open(self):
        ch = _make_digital('GCB OPEN')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_CB

    def test_gcb_closed(self):
        ch = _make_digital('GCB CLOSED')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_CB

    def test_bi_52b_nari(self):
        ch = _make_digital('BI_52b')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_CB


class TestDigitalAR:
    """DIG_AR detection."""

    def test_79ar_attempted_ben32(self):
        ch = _make_digital('LGNG1 79AR_BUS_ATMPT')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_AR
        assert conf == RoleConfidence.HIGH

    def test_79ar_lockout(self):
        ch = _make_digital('LGNG1 79AR_BUS_L/O')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_AR


class TestDigitalIntertrip:
    """DIG_INTERTRIP detection."""

    def test_50bf_stage_ben32(self):
        ch = _make_digital('Z1230 50BF_CTR_STG1/STG2')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_INTERTRIP
        assert conf == RoleConfidence.HIGH

    def test_85intr_receive_ben32(self):
        ch = _make_digital('LGNG1 85INTR_REC1')
        role, conf = detect_digital_role(ch)
        # '85INTR_REC1' contains 'RECV' equivalent... let's check _INTERTRIP_KW
        # 'RECV1' and '_CR' are in the set; '85INTR_REC1' contains 'REC1'
        # '_REC' is in _INTERTRIP_KW
        assert role == SignalRole.DIG_INTERTRIP

    def test_send1_nari(self):
        ch = _make_digital('Send1')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_INTERTRIP

    def test_recv1_nari(self):
        ch = _make_digital('Recv1')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_INTERTRIP

    def test_pott_nari(self):
        ch = _make_digital('Op_WE_POTT')
        # POTT is in _INTERTRIP_KW — but Op_ is in _TRIP_KW.
        # INTERTRIP is checked before TRIP, so POTT wins.
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_INTERTRIP

    def test_op_roc_pott_nari(self):
        ch = _make_digital('Op_ROC_POTT')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_INTERTRIP


class TestDigitalTrip:
    """DIG_TRIP detection."""

    def test_87l_differential_ben32(self):
        ch = _make_digital('LGNG1 87L/1')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP
        assert conf == RoleConfidence.HIGH

    def test_87stub_differential(self):
        ch = _make_digital('LGNG1 87STUB_NORTH')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_op_z1_nari(self):
        ch = _make_digital('Op_Z1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_op_z_sotf_nari(self):
        ch = _make_digital('Op_Z_SOTF')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_op_roc_nari(self):
        ch = _make_digital('Op_ROC1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_op_ovld_trp_nari(self):
        ch = _make_digital('Op_Ovld_Trp')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_trip_generic(self):
        ch = _make_digital('Trip')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_idiff_trip_nari(self):
        ch = _make_digital('Idiff Trip Z1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP

    def test_bu_21z_distance(self):
        ch = _make_digital('LGNG1 BU_21Z')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP


class TestDigitalPickup:
    """DIG_PICKUP detection."""

    def test_fd_standalone_nari(self):
        ch = _make_digital('FD')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_vebi_distp_nari(self):
        ch = _make_digital('VEBI_DistP')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_vebi_roc_nari(self):
        ch = _make_digital('VEBI_ROC')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_phcomp_blk_nari(self):
        ch = _make_digital('PhComp Blk Z1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_idiff_cz_start_nari(self):
        ch = _make_digital('Idiff CZ Start')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_ext_cbf_nari(self):
        ch = _make_digital('Ext CBF Z1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_bi_extrp_nari(self):
        ch = _make_digital('BI_ExTrp_TeleP')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_over_voltage_pickup(self):
        ch = _make_digital('OVER VOLTAGE UNIT1')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_PICKUP

    def test_bu_21z_sotf_is_trip(self):
        """BU_21Z_SOTF should be trip, not pickup — SOTF is a trip keyword."""
        ch = _make_digital('LGNG1 BU_21Z_SOTF')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_TRIP


class TestDigitalGenericFallback:
    """Channels that don't match any specific rule fall back to DIG_GENERIC."""

    def test_bbp_off_falls_back(self):
        ch = _make_digital('L31 BBP OFF')
        role, conf = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC
        assert conf == RoleConfidence.LOW

    def test_diff_blked_generic(self):
        ch = _make_digital('Diff Z1 Blked')
        role, _ = detect_digital_role(ch)
        assert role == SignalRole.DIG_GENERIC


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEMENTARY CB PAIR DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class TestComplementaryCbPairs:
    """detect_complementary_cb_pairs must link OPEN/CLOSED DIG_CB pairs."""

    def _make_cb(self, name: str, channel_id: int) -> DigitalChannel:
        ch = _make_digital(name, channel_id=channel_id)
        ch.signal_role = SignalRole.DIG_CB
        return ch

    def test_gcb_open_closed_pair(self):
        ch_open   = self._make_cb('GCB OPEN',   channel_id=10)
        ch_closed = self._make_cb('GCB CLOSED',  channel_id=11)
        pairs = detect_complementary_cb_pairs([ch_open, ch_closed])

        assert pairs == [(10, 11)]
        assert ch_open.is_complementary is True
        assert ch_open.is_primary_of_pair is True
        assert ch_open.complementary_channel_id == 11
        assert ch_closed.is_complementary is True
        assert ch_closed.is_primary_of_pair is False
        assert ch_closed.complementary_channel_id == 10

    def test_fcb_open_closed_pair(self):
        ch_open   = self._make_cb('FCB OPEN',   channel_id=20)
        ch_closed = self._make_cb('FCB CLOSED', channel_id=21)
        pairs = detect_complementary_cb_pairs([ch_open, ch_closed])
        assert len(pairs) == 1
        assert ch_open.is_complementary is True
        assert ch_closed.is_complementary is True

    def test_underscore_variant(self):
        ch_open   = self._make_cb('GCB_OPEN',   channel_id=30)
        ch_closed = self._make_cb('GCB_CLOSED', channel_id=31)
        pairs = detect_complementary_cb_pairs([ch_open, ch_closed])
        assert len(pairs) == 1

    def test_no_pair_when_only_open(self):
        ch_open = self._make_cb('CB_BUS_R_OPEN', channel_id=40)
        pairs = detect_complementary_cb_pairs([ch_open])
        assert pairs == []
        assert ch_open.is_complementary is False

    def test_non_cb_channels_ignored(self):
        ch = _make_digital('GCB OPEN', channel_id=50)
        ch.signal_role = SignalRole.DIG_TRIP   # wrong role — not DIG_CB
        pairs = detect_complementary_cb_pairs([ch])
        assert pairs == []


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: detect_signal_roles() — combined entry point
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectSignalRolesIntegration:
    """detect_signal_roles() runs over a mixed list and mutates in-place."""

    def test_returns_same_list(self):
        channels = [_make_analogue('SJTC2 VR', unit='kV')]
        result = detect_signal_roles(channels)
        assert result is channels

    def test_analogue_role_assigned(self):
        ch = _make_analogue('GT1 IY', unit='kA')
        detect_signal_roles([ch])
        assert ch.signal_role == SignalRole.I_PHASE
        assert ch.phase == 'B'
        assert ch.role_confidence == RoleConfidence.HIGH

    def test_digital_role_assigned(self):
        ch = _make_digital('LGNG1 87L/1')
        detect_signal_roles([ch])
        assert ch.signal_role == SignalRole.DIG_TRIP

    def test_colour_assigned_after_detection(self):
        ch = _make_analogue('LGNG1 VR', unit='kV')
        detect_signal_roles([ch])
        # Phase A → red
        assert ch.colour == '#FF4444'

    def test_colour_digital_trip(self):
        ch = _make_digital('LGNG1 87L/1')
        detect_signal_roles([ch])
        assert ch.colour == '#FF2222'

    def test_complementary_pair_wired_via_main_entry(self):
        ch_open   = _make_digital('GCB OPEN',   channel_id=1)
        ch_closed = _make_digital('GCB CLOSED', channel_id=2)
        detect_signal_roles([ch_open, ch_closed])
        assert ch_open.signal_role == SignalRole.DIG_CB
        assert ch_closed.signal_role == SignalRole.DIG_CB
        assert ch_open.is_complementary is True
        assert ch_closed.is_complementary is True

    def test_is_derived_set_for_power_channel(self):
        ch = _make_analogue('POWER PGPS 2', unit='MW')
        detect_signal_roles([ch])
        assert ch.is_derived is True

    def test_real_jmhe_analogue_batch(self):
        """Spot-check five channels from JMHE_500kV.cfg."""
        channels = [
            _make_analogue('LGNG1 VR', unit='kV'),   # V_PHASE A
            _make_analogue('LGNG1 VY', unit='kV'),   # V_PHASE B
            _make_analogue('LGNG1 VB', unit='kV'),   # V_PHASE C
            _make_analogue('LGNG1 IR', unit='kA'),   # I_PHASE A
            _make_analogue('LGNG1 IN', unit='kA'),   # I_EARTH N
        ]
        detect_signal_roles(channels)

        assert channels[0].signal_role == SignalRole.V_PHASE and channels[0].phase == 'A'
        assert channels[1].signal_role == SignalRole.V_PHASE and channels[1].phase == 'B'
        assert channels[2].signal_role == SignalRole.V_PHASE and channels[2].phase == 'C'
        assert channels[3].signal_role == SignalRole.I_PHASE and channels[3].phase == 'A'
        assert channels[4].signal_role == SignalRole.I_EARTH and channels[4].phase == 'N'

    def test_real_pmjy_analogue_batch(self):
        """Spot-check POWER and FREQ channels from PMJY_275.cfg."""
        channels = [
            _make_analogue('FREQ PGPS 2',  unit='Hz'),  # FREQ
            _make_analogue('POWER PGPS 2', unit='MW'),  # P_MW, is_derived
            _make_analogue('FREQ UY PLTG 1', unit='Hz'),  # FREQ
        ]
        detect_signal_roles(channels)

        assert channels[0].signal_role == SignalRole.FREQ
        assert channels[1].signal_role == SignalRole.P_MW
        assert channels[1].is_derived is True
        assert channels[2].signal_role == SignalRole.FREQ

    def test_real_relay_cfg_nari_analogue(self):
        """NARI Relay.cfg Chinese-convention analogue channels."""
        channels = [
            _make_analogue('3I0', unit='A'),   # I_EARTH N (unit A → refine via code)
            _make_analogue('3U0', unit='V'),   # V_RESIDUAL N
            _make_analogue(' Ia', unit='A'),   # I_PHASE A
            _make_analogue(' Ib', unit='A'),   # I_PHASE B
            _make_analogue(' Ic', unit='A'),   # I_PHASE C
            _make_analogue(' Ua', unit='V'),   # V_PHASE A
            _make_analogue(' Ux', unit='V'),   # V_RESIDUAL N
        ]
        detect_signal_roles(channels)

        assert channels[0].signal_role == SignalRole.I_EARTH
        assert channels[0].phase == 'N'
        assert channels[1].signal_role == SignalRole.V_RESIDUAL
        assert channels[2].signal_role == SignalRole.I_PHASE and channels[2].phase == 'A'
        assert channels[3].signal_role == SignalRole.I_PHASE and channels[3].phase == 'B'
        assert channels[4].signal_role == SignalRole.I_PHASE and channels[4].phase == 'C'
        assert channels[5].signal_role == SignalRole.V_PHASE and channels[5].phase == 'A'
        assert channels[6].signal_role == SignalRole.V_RESIDUAL

    def test_real_relay_cfg_nari_digital(self):
        """Key digital channels from Relay.cfg."""
        channels = [
            _make_digital('FD'),             # DIG_PICKUP
            _make_digital('Send1'),           # DIG_INTERTRIP
            _make_digital('Recv1'),           # DIG_INTERTRIP
            _make_digital('Trip'),            # DIG_TRIP
            _make_digital('Op_ROC1'),         # DIG_TRIP
            _make_digital('Op_Z1'),           # DIG_TRIP
            _make_digital('BI_52b'),          # DIG_CB
            _make_digital('BI_MCB_VT'),       # DIG_GENERIC (alarm exception)
            _make_digital('BI_En_Z1'),        # DIG_GENERIC (alarm exception)
            _make_digital('VEBI_DistP'),      # DIG_PICKUP
            _make_digital('VEBI_ROC'),        # DIG_PICKUP
        ]
        detect_signal_roles(channels)

        assert channels[0].signal_role == SignalRole.DIG_PICKUP
        assert channels[1].signal_role == SignalRole.DIG_INTERTRIP
        assert channels[2].signal_role == SignalRole.DIG_INTERTRIP
        assert channels[3].signal_role == SignalRole.DIG_TRIP
        assert channels[4].signal_role == SignalRole.DIG_TRIP
        assert channels[5].signal_role == SignalRole.DIG_TRIP
        assert channels[6].signal_role == SignalRole.DIG_CB
        assert channels[7].signal_role == SignalRole.DIG_GENERIC
        assert channels[8].signal_role == SignalRole.DIG_GENERIC
        assert channels[9].signal_role == SignalRole.DIG_PICKUP
        assert channels[10].signal_role == SignalRole.DIG_PICKUP

    def test_real_jmhe_digital_alarm_exception(self):
        """COMMFAIL channels from JMHE_500kV.cfg must be DIG_GENERIC."""
        channels = [
            _make_digital('LGNG1 87L/1'),          # DIG_TRIP
            _make_digital('LGNG1 87L/1_COMMFAIL'),  # DIG_GENERIC — alarm exception
            _make_digital('LGNG1 87L/2_COMMFAIL'),  # DIG_GENERIC
        ]
        detect_signal_roles(channels)

        assert channels[0].signal_role == SignalRole.DIG_TRIP
        assert channels[1].signal_role == SignalRole.DIG_GENERIC
        assert channels[2].signal_role == SignalRole.DIG_GENERIC
