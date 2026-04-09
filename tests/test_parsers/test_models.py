"""
tests/test_parsers/test_models.py

Unit tests for:
  src/models/channel.py          — AnalogueChannel, DigitalChannel, SignalRole,
                                    RoleConfidence, COLOUR_MAP, default_colour_for()
  src/models/disturbance_record.py — DisturbanceRecord, SourceFormat
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from models.channel import (
    AnalogueChannel,
    COLOUR_MAP,
    DigitalChannel,
    RoleConfidence,
    SignalRole,
    default_colour_for,
)


# ── SignalRole constants ─────────────────────────────────────────────────────

class TestSignalRoleConstants:
    """All role constants must exist and be non-empty strings."""

    def test_analogue_roles_exist(self) -> None:
        for attr in (
            "V_PHASE", "V_LINE", "V_RESIDUAL", "I_PHASE", "I_EARTH",
            "V1_PMU", "I1_PMU", "P_MW", "Q_MVAR", "FREQ", "ROCOF",
            "DC_FIELD_I", "DC_FIELD_V", "MECH_SPEED", "MECH_VALVE",
            "SEQ_RMS", "ANALOGUE",
        ):
            assert isinstance(getattr(SignalRole, attr), str)
            assert getattr(SignalRole, attr) != ""

    def test_digital_roles_exist(self) -> None:
        for attr in (
            "DIG_TRIP", "DIG_CB", "DIG_PICKUP", "DIG_AR",
            "DIG_INTERTRIP", "DIG_TRIGGER", "DIG_GENERIC",
        ):
            assert isinstance(getattr(SignalRole, attr), str)
            assert getattr(SignalRole, attr) != ""

    def test_analogue_roles_frozenset_complete(self) -> None:
        expected = {
            SignalRole.V_PHASE, SignalRole.V_LINE, SignalRole.V_RESIDUAL,
            SignalRole.I_PHASE, SignalRole.I_EARTH,
            SignalRole.V1_PMU, SignalRole.I1_PMU,
            SignalRole.P_MW, SignalRole.Q_MVAR,
            SignalRole.FREQ, SignalRole.ROCOF,
            SignalRole.DC_FIELD_I, SignalRole.DC_FIELD_V,
            SignalRole.MECH_SPEED, SignalRole.MECH_VALVE,
            SignalRole.SEQ_RMS, SignalRole.ANALOGUE,
        }
        assert expected == SignalRole.ANALOGUE_ROLES

    def test_digital_roles_frozenset_complete(self) -> None:
        expected = {
            SignalRole.DIG_TRIP, SignalRole.DIG_CB, SignalRole.DIG_PICKUP,
            SignalRole.DIG_AR, SignalRole.DIG_INTERTRIP,
            SignalRole.DIG_TRIGGER, SignalRole.DIG_GENERIC,
        }
        assert expected == SignalRole.DIGITAL_ROLES


# ── RoleConfidence constants ─────────────────────────────────────────────────

class TestRoleConfidence:
    def test_values(self) -> None:
        assert RoleConfidence.HIGH   == "HIGH"
        assert RoleConfidence.MEDIUM == "MEDIUM"
        assert RoleConfidence.LOW    == "LOW"


# ── COLOUR_MAP ───────────────────────────────────────────────────────────────

class TestColourMap:
    def test_phase_colours_present(self) -> None:
        for phase in ("A", "B", "C", "N"):
            assert phase in COLOUR_MAP
            assert COLOUR_MAP[phase].startswith("#")

    def test_digital_role_colours_present(self) -> None:
        for role in (
            SignalRole.DIG_TRIP, SignalRole.DIG_CB, SignalRole.DIG_PICKUP,
            SignalRole.DIG_AR, SignalRole.DIG_INTERTRIP, SignalRole.DIG_GENERIC,
        ):
            assert role in COLOUR_MAP, f"{role} missing from COLOUR_MAP"

    def test_special_analogue_colours_present(self) -> None:
        for role in (
            SignalRole.FREQ, SignalRole.ROCOF,
            SignalRole.P_MW, SignalRole.Q_MVAR,
            SignalRole.DC_FIELD_I, SignalRole.DC_FIELD_V,
            SignalRole.MECH_SPEED, SignalRole.MECH_VALVE,
        ):
            assert role in COLOUR_MAP, f"{role} missing from COLOUR_MAP"

    def test_all_colours_are_valid_hex(self) -> None:
        for key, colour in COLOUR_MAP.items():
            assert colour.startswith("#"), f"{key}: colour {colour!r} missing '#'"
            assert len(colour) == 7, f"{key}: colour {colour!r} not 7 chars"


# ── default_colour_for() ─────────────────────────────────────────────────────

class TestDefaultColourFor:
    def test_phase_a_voltage_returns_red(self) -> None:
        colour = default_colour_for(SignalRole.V_PHASE, "A")
        # V_PHASE is not in COLOUR_MAP so phase takes effect
        assert colour == "#FF4444"

    def test_freq_role_returns_cyan(self) -> None:
        colour = default_colour_for(SignalRole.FREQ, "")
        assert colour == "#00DDDD"

    def test_mw_role_returns_orange(self) -> None:
        assert default_colour_for(SignalRole.P_MW, "") == "#FFAA44"

    def test_mvar_role_returns_purple(self) -> None:
        assert default_colour_for(SignalRole.Q_MVAR, "") == "#AA44FF"

    def test_dc_quantities_return_grey(self) -> None:
        assert default_colour_for(SignalRole.DC_FIELD_I, "") == "#AAAAAA"
        assert default_colour_for(SignalRole.DC_FIELD_V, "") == "#AAAAAA"
        assert default_colour_for(SignalRole.MECH_SPEED, "") == "#AAAAAA"
        assert default_colour_for(SignalRole.MECH_VALVE, "") == "#AAAAAA"

    def test_unknown_role_no_phase_returns_fallback(self) -> None:
        colour = default_colour_for("UNKNOWN_ROLE", "")
        assert colour.startswith("#")


# ── AnalogueChannel construction ────────────────────────────────────────────

class TestAnalogueChannelDefaults:
    def test_minimal_construction(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.channel_id == 1
        assert ch.name == "Va"
        assert ch.phase == "A"
        assert ch.unit == "kV"

    def test_default_scaling(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.multiplier == 1.0
        assert ch.offset == 0.0

    def test_default_primary_secondary_none(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.primary is None
        assert ch.secondary is None

    def test_default_role_and_confidence(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.signal_role == SignalRole.ANALOGUE
        assert ch.role_confidence == RoleConfidence.LOW
        assert ch.role_confirmed is False

    def test_default_visible_true(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.visible is True

    def test_default_is_derived_false(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.is_derived is False

    def test_raw_data_defaults_to_empty_float32(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        assert ch.raw_data.dtype == np.float32
        assert len(ch.raw_data) == 0

    def test_raw_data_coerced_to_float32(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=data)
        assert ch.raw_data.dtype == np.float32

    def test_raw_data_list_coerced_to_ndarray(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=[1.0, 2.0, 3.0])
        assert isinstance(ch.raw_data, np.ndarray)
        assert ch.raw_data.dtype == np.float32

    def test_colour_auto_assigned_for_phase_a(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             signal_role=SignalRole.V_PHASE)
        assert ch.colour == "#FF4444"

    def test_n_samples(self) -> None:
        data = np.zeros(500, dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=data)
        assert ch.n_samples == 500


# ── AnalogueChannel.physical_data (LAW 10) ───────────────────────────────────

class TestAnalogueChannelPhysicalData:
    """LAW 10: physical = (raw × multiplier) + offset — always applied."""

    def test_identity_transform(self) -> None:
        """multiplier=1, offset=0 → physical equals raw."""
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=raw, multiplier=1.0, offset=0.0)
        np.testing.assert_allclose(ch.physical_data, raw.astype(np.float64))

    def test_multiplier_applied(self) -> None:
        raw = np.array([100.0, 200.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=raw, multiplier=0.1, offset=0.0)
        np.testing.assert_allclose(ch.physical_data, [10.0, 20.0], rtol=1e-5)

    def test_offset_applied(self) -> None:
        """Non-zero offset — as seen on DC_FIELD_I, MECH_SPEED etc."""
        raw = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Field_I", phase="",
                             unit="A", signal_role=SignalRole.DC_FIELD_I,
                             raw_data=raw, multiplier=1.0, offset=250.0)
        np.testing.assert_allclose(ch.physical_data, [250.0, 300.0, 350.0],
                                   rtol=1e-5)

    def test_multiplier_and_offset_combined(self) -> None:
        """physical = raw * 0.5 + 10.0"""
        raw = np.array([0.0, 10.0, 20.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="RPM", phase="",
                             unit="RPM", signal_role=SignalRole.MECH_SPEED,
                             raw_data=raw, multiplier=0.5, offset=10.0)
        np.testing.assert_allclose(ch.physical_data, [10.0, 15.0, 20.0],
                                   rtol=1e-5)

    def test_physical_data_returns_float64(self) -> None:
        raw = np.array([1.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=raw)
        assert ch.physical_data.dtype == np.float64

    def test_zero_offset_still_applied_uniformly(self) -> None:
        """Even zero offset must be applied — result same as raw in this case."""
        raw = np.array([5.0, 10.0], dtype=np.float32)
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV",
                             raw_data=raw, multiplier=2.0, offset=0.0)
        np.testing.assert_allclose(ch.physical_data, [10.0, 20.0], rtol=1e-5)


# ── DigitalChannel construction ──────────────────────────────────────────────

class TestDigitalChannelDefaults:
    def test_minimal_construction(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP_A")
        assert ch.channel_id == 1
        assert ch.name == "TRIP_A"

    def test_default_normal_state_zero(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP_A")
        assert ch.normal_state == 0

    def test_default_signal_role_generic(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP_A")
        assert ch.signal_role == SignalRole.DIG_GENERIC

    def test_data_coerced_to_uint8(self) -> None:
        data = np.array([0, 1, 0, 1], dtype=np.int32)
        ch = DigitalChannel(channel_id=1, name="TRIP_A", data=data)
        assert ch.data.dtype == np.uint8

    def test_data_list_coerced_to_ndarray(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP_A", data=[0, 1, 1, 0])
        assert isinstance(ch.data, np.ndarray)
        assert ch.data.dtype == np.uint8

    def test_default_not_complementary(self) -> None:
        ch = DigitalChannel(channel_id=1, name="GCB_OPEN")
        assert ch.is_complementary is False
        assert ch.complementary_channel_id is None

    def test_colour_auto_assigned_from_role(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP_A",
                            signal_role=SignalRole.DIG_TRIP)
        assert ch.colour == "#FF2222"

    def test_n_samples(self) -> None:
        ch = DigitalChannel(channel_id=1, name="CB",
                            data=np.zeros(100, dtype=np.uint8))
        assert ch.n_samples == 100

    def test_visible_default_true(self) -> None:
        ch = DigitalChannel(channel_id=1, name="CB")
        assert ch.visible is True


# ── DigitalChannel.active_mask ───────────────────────────────────────────────

class TestDigitalChannelActiveMask:
    def test_normal_state_0_active_when_1(self) -> None:
        """normal_state=0 → channel is active when data=1."""
        ch = DigitalChannel(channel_id=1, name="TRIP",
                            signal_role=SignalRole.DIG_TRIP,
                            normal_state=0,
                            data=np.array([0, 0, 1, 1, 0], dtype=np.uint8))
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(ch.active_mask, expected)

    def test_normal_state_1_active_when_0(self) -> None:
        """normal_state=1 (normally closed) → channel is active when data=0."""
        ch = DigitalChannel(channel_id=1, name="GCB_CLOSED",
                            signal_role=SignalRole.DIG_CB,
                            normal_state=1,
                            data=np.array([1, 1, 0, 0, 1], dtype=np.uint8))
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(ch.active_mask, expected)

    def test_active_mask_returns_bool_array(self) -> None:
        ch = DigitalChannel(channel_id=1, name="CB",
                            data=np.array([0, 1], dtype=np.uint8))
        assert ch.active_mask.dtype == bool


# ── Complementary pair linkage ────────────────────────────────────────────────

class TestComplementaryPair:
    """GCB OPEN + GCB CLOSED linked by the parser."""

    def test_mark_pair(self) -> None:
        open_ch = DigitalChannel(
            channel_id=10, name="GCB OPEN",
            signal_role=SignalRole.DIG_CB,
            is_complementary=True,
            complementary_channel_id=11,
            is_primary_of_pair=True,
        )
        closed_ch = DigitalChannel(
            channel_id=11, name="GCB CLOSED",
            signal_role=SignalRole.DIG_CB,
            is_complementary=True,
            complementary_channel_id=10,
            is_primary_of_pair=False,
        )
        assert open_ch.is_primary_of_pair
        assert not closed_ch.is_primary_of_pair
        assert open_ch.complementary_channel_id == closed_ch.channel_id
        assert closed_ch.complementary_channel_id == open_ch.channel_id


# ═════════════════════════════════════════════════════════════════════════════
# DisturbanceRecord tests
# ═════════════════════════════════════════════════════════════════════════════

from models.disturbance_record import (
    DisturbanceRecord,
    SourceFormat,
    WAVEFORM_THRESHOLD,
    VALID_NOMINAL_FREQUENCIES,
)

# ── Shared fixture ────────────────────────────────────────────────────────────

_T0 = datetime(2024, 6, 1, 12, 0, 0)
_T_TRIGGER = datetime(2024, 6, 1, 12, 0, 2)


def _make_record(**overrides) -> DisturbanceRecord:
    """Return a minimal valid DisturbanceRecord, with optional field overrides."""
    defaults = dict(
        station_name="TEST_STATION",
        device_id="RELAY_01",
        start_time=_T0,
        trigger_time=_T_TRIGGER,
        trigger_sample=12000,
        sample_rate=6000.0,
        nominal_frequency=50.0,
        source_format=SourceFormat.COMTRADE_1999,
        file_path=Path("/data/test.cfg"),
    )
    defaults.update(overrides)
    return DisturbanceRecord(**defaults)


# ── SourceFormat constants ────────────────────────────────────────────────────

class TestSourceFormat:
    def test_all_constants_exist(self) -> None:
        for attr in ("COMTRADE_1991", "COMTRADE_1999", "COMTRADE_2013",
                     "CSV", "EXCEL", "PMU_CSV"):
            assert isinstance(getattr(SourceFormat, attr), str)
            assert getattr(SourceFormat, attr) != ""

    def test_all_frozenset_complete(self) -> None:
        expected = {
            SourceFormat.COMTRADE_1991, SourceFormat.COMTRADE_1999,
            SourceFormat.COMTRADE_2013, SourceFormat.CSV,
            SourceFormat.EXCEL, SourceFormat.PMU_CSV,
        }
        assert expected == SourceFormat.ALL


# ── Module-level constants ────────────────────────────────────────────────────

class TestModuleConstants:
    def test_waveform_threshold(self) -> None:
        assert WAVEFORM_THRESHOLD == 200.0

    def test_valid_nominal_frequencies(self) -> None:
        assert 50.0 in VALID_NOMINAL_FREQUENCIES
        assert 60.0 in VALID_NOMINAL_FREQUENCIES
        assert len(VALID_NOMINAL_FREQUENCIES) == 2


# ── DisturbanceRecord: construction and defaults ──────────────────────────────

class TestDisturbanceRecordDefaults:
    def test_minimal_construction(self) -> None:
        rec = _make_record()
        assert rec.station_name == "TEST_STATION"
        assert rec.device_id == "RELAY_01"
        assert rec.start_time == _T0
        assert rec.trigger_time == _T_TRIGGER
        assert rec.trigger_sample == 12000
        assert rec.sample_rate == 6000.0
        assert rec.nominal_frequency == 50.0
        assert rec.source_format == SourceFormat.COMTRADE_1999
        assert rec.file_path == Path("/data/test.cfg")

    def test_analogue_channels_default_empty(self) -> None:
        assert _make_record().analogue_channels == []

    def test_digital_channels_default_empty(self) -> None:
        assert _make_record().digital_channels == []

    def test_bay_names_default_empty(self) -> None:
        assert _make_record().bay_names == []

    def test_header_text_default_empty_string(self) -> None:
        assert _make_record().header_text == ""

    def test_time_array_default_empty_float64(self) -> None:
        rec = _make_record()
        assert rec.time_array.dtype == np.float64
        assert len(rec.time_array) == 0

    def test_channel_lists_are_independent_instances(self) -> None:
        """Two records must not share the same list object."""
        r1 = _make_record()
        r2 = _make_record()
        assert r1.analogue_channels is not r2.analogue_channels
        assert r1.digital_channels is not r2.digital_channels
        assert r1.bay_names is not r2.bay_names


# ── display_mode auto-derivation (LAW 9) ─────────────────────────────────────

class TestDisplayMode:
    def test_waveform_at_threshold(self) -> None:
        rec = _make_record(sample_rate=200.0)
        assert rec.display_mode == "WAVEFORM"

    def test_waveform_above_threshold(self) -> None:
        for rate in (201.0, 1200.0, 6000.0):
            assert _make_record(sample_rate=rate).display_mode == "WAVEFORM"

    def test_trend_below_threshold(self) -> None:
        for rate in (50.0, 100.0, 199.9):
            assert _make_record(sample_rate=rate).display_mode == "TREND", \
                f"Expected TREND for {rate} Hz"

    def test_display_mode_ignores_caller_value(self) -> None:
        """Caller cannot override display_mode — it is always derived (LAW 9)."""
        rec = _make_record(sample_rate=50.0, display_mode="WAVEFORM")
        assert rec.display_mode == "TREND"

    def test_ben32_fast_record_is_waveform(self) -> None:
        rec = _make_record(sample_rate=1200.0)
        assert rec.display_mode == "WAVEFORM"

    def test_ben32_slow_record_is_trend(self) -> None:
        rec = _make_record(sample_rate=50.0)
        assert rec.display_mode == "TREND"


# ── nominal_frequency validation ─────────────────────────────────────────────

class TestNominalFrequencyValidation:
    def test_50hz_accepted(self) -> None:
        rec = _make_record(nominal_frequency=50.0)
        assert rec.nominal_frequency == 50.0

    def test_60hz_accepted(self) -> None:
        rec = _make_record(nominal_frequency=60.0)
        assert rec.nominal_frequency == 60.0

    def test_49hz_raises(self) -> None:
        with pytest.raises(ValueError, match="nominal_frequency"):
            _make_record(nominal_frequency=49.0)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="nominal_frequency"):
            _make_record(nominal_frequency=0.0)

    def test_55hz_raises(self) -> None:
        with pytest.raises(ValueError, match="nominal_frequency"):
            _make_record(nominal_frequency=55.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="nominal_frequency"):
            _make_record(nominal_frequency=-50.0)


# ── sample_rate validation ────────────────────────────────────────────────────

class TestSampleRateValidation:
    def test_positive_rate_accepted(self) -> None:
        assert _make_record(sample_rate=50.0).sample_rate == 50.0
        assert _make_record(sample_rate=6000.0).sample_rate == 6000.0

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            _make_record(sample_rate=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            _make_record(sample_rate=-100.0)


# ── time_array dtype coercion ─────────────────────────────────────────────────

class TestTimeArrayDtype:
    def test_float64_input_unchanged(self) -> None:
        t = np.linspace(0.0, 1.0, 6001, dtype=np.float64)
        rec = _make_record(time_array=t)
        assert rec.time_array.dtype == np.float64

    def test_float32_coerced_to_float64(self) -> None:
        t = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        rec = _make_record(time_array=t)
        assert rec.time_array.dtype == np.float64

    def test_int_array_coerced_to_float64(self) -> None:
        t = np.arange(100, dtype=np.int32)
        rec = _make_record(time_array=t)
        assert rec.time_array.dtype == np.float64

    def test_list_coerced_to_float64_ndarray(self) -> None:
        rec = _make_record(time_array=[0.0, 0.001, 0.002])
        assert isinstance(rec.time_array, np.ndarray)
        assert rec.time_array.dtype == np.float64

    def test_values_preserved_after_coercion(self) -> None:
        t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        rec = _make_record(time_array=t)
        np.testing.assert_allclose(rec.time_array, [0.0, 0.5, 1.0], rtol=1e-5)


# ── Private cache initialisation ─────────────────────────────────────────────

class TestCacheInitialisation:
    def test_rms_cache_is_empty_dict(self) -> None:
        rec = _make_record()
        assert isinstance(rec._rms_cache, dict)
        assert len(rec._rms_cache) == 0

    def test_phasor_cache_is_empty_dict(self) -> None:
        rec = _make_record()
        assert isinstance(rec._phasor_cache, dict)
        assert len(rec._phasor_cache) == 0

    def test_caches_are_independent_across_instances(self) -> None:
        r1 = _make_record()
        r2 = _make_record()
        r1._rms_cache["ch1"] = [1, 2, 3]
        assert "ch1" not in r2._rms_cache

    def test_caches_not_in_init_signature(self) -> None:
        """_rms_cache and _phasor_cache must not be constructor parameters."""
        import inspect
        sig = inspect.signature(DisturbanceRecord.__init__)
        assert "_rms_cache" not in sig.parameters
        assert "_phasor_cache" not in sig.parameters


# ── Convenience properties ────────────────────────────────────────────────────

class TestConvenienceProperties:
    def test_n_analogue_empty(self) -> None:
        assert _make_record().n_analogue == 0

    def test_n_analogue_with_channels(self) -> None:
        ch = AnalogueChannel(channel_id=1, name="Va", phase="A", unit="kV")
        rec = _make_record(analogue_channels=[ch])
        assert rec.n_analogue == 1

    def test_n_digital_empty(self) -> None:
        assert _make_record().n_digital == 0

    def test_n_digital_with_channels(self) -> None:
        ch = DigitalChannel(channel_id=1, name="TRIP")
        rec = _make_record(digital_channels=[ch])
        assert rec.n_digital == 1

    def test_duration_empty_array(self) -> None:
        assert _make_record().duration == 0.0

    def test_duration_single_sample(self) -> None:
        rec = _make_record(time_array=np.array([0.5]))
        assert rec.duration == 0.0

    def test_duration_calculated_correctly(self) -> None:
        t = np.array([0.0, 0.5, 1.0, 2.5])
        rec = _make_record(time_array=t)
        assert rec.duration == pytest.approx(2.5)

    def test_get_analogue_channel_found(self) -> None:
        ch = AnalogueChannel(channel_id=3, name="Ia", phase="A", unit="kA")
        rec = _make_record(analogue_channels=[ch])
        result = rec.get_analogue_channel(3)
        assert result is ch

    def test_get_analogue_channel_not_found(self) -> None:
        rec = _make_record()
        assert rec.get_analogue_channel(99) is None

    def test_get_digital_channel_found(self) -> None:
        ch = DigitalChannel(channel_id=7, name="TRIP_A")
        rec = _make_record(digital_channels=[ch])
        assert rec.get_digital_channel(7) is ch

    def test_get_digital_channel_not_found(self) -> None:
        assert _make_record().get_digital_channel(0) is None
