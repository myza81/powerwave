"""Unit tests for ComtradeProvider — CFG parsing, DAT parsing, normalization.

All tests use in-memory fixture data written to pytest tmp_path.
No real COMTRADE files are required.
"""

from __future__ import annotations

import math
import struct
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.providers.base.exceptions import ProviderLoadError
from app.providers.comtrade.comtrade_provider import (
    ComtradeProvider,
    _CfgData,
    _apply_analog_scaling,
    _extract_digital_channels,
    _parse_analog_line,
    _parse_ascii_dat,
    _parse_binary_dat,
    _parse_cfg,
    _parse_digital_line,
    _parse_timestamp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_cfg_dat(
    tmp_path: Path,
    cfg_text: str,
    dat_content: str | bytes,
    stem: str = "test",
) -> Path:
    """Write CFG and DAT files; return the CFG path."""
    cfg_path = tmp_path / f"{stem}.cfg"
    dat_path = tmp_path / f"{stem}.dat"
    cfg_path.write_text(cfg_text, encoding="latin-1")
    if isinstance(dat_content, str):
        dat_path.write_text(dat_content, encoding="latin-1")
    else:
        dat_path.write_bytes(dat_content)
    return cfg_path


def _make_binary_rows(
    rows: list[tuple[int, int, list[int], list[int]]],
) -> bytes:
    """Build Binary DAT bytes from (n, ts, analogs_int16, dwords_uint16) tuples."""
    buf = b""
    for n, ts, analogs, dwords in rows:
        buf += struct.pack("<II", n, ts)
        for a in analogs:
            buf += struct.pack("<h", a)
        for d in dwords:
            buf += struct.pack("<H", d)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Minimal CFG fixtures
# ─────────────────────────────────────────────────────────────────────────────

# 1991 format: no rev_yr, 10-field analog, 5-field digital, 1 rate
_CFG_1991 = (
    "SUBST_A,IED_01\n"
    "2,1A,1D\n"
    "1,VA,A,,kV,1.0,0.0,0,-32768,32767\n"
    "1,CB_A,,,0\n"
    "50\n"
    "1\n"
    "1000,4\n"
    "01/01/2024,12:00:00.000000\n"
    "01/01/2024,12:00:00.001000\n"
    "ASCII\n"
)

# 1999 format: 13-field analog, TIMEMULT=0.1, PS=P
_CFG_1999 = (
    "SUBST_B,RELAY_02,1999\n"
    "2,1A,1D\n"
    "1,VA,A,,kV,0.5,10.0,0,-32768,32767,110.0,0.11,P\n"
    "1,CB_A,,,0\n"
    "50\n"
    "1\n"
    "500,3\n"
    "15/03/2023,08:30:00.000000\n"
    "15/03/2023,08:30:00.000200\n"
    "ASCII\n"
    "0.1\n"
)

# Multi-rate CFG: 2 rate sections
_CFG_MULTIRATE = (
    "SUBST_C,IED_03,1999\n"
    "2,1A,0D\n"
    "1,IA,A,,A,1.0,0.0,0,-32768,32767,100.0,1.0,S\n"
    "50\n"
    "2\n"
    "5000,3\n"
    "500,5\n"
    "01/06/2023,00:00:00.000000\n"
    "01/06/2023,00:00:00.000200\n"
    "ASCII\n"
    "1\n"
)

# Binary format CFG
_CFG_BINARY = (
    "SUBST_D,IED_04,1999\n"
    "2,1A,1D\n"
    "1,VB,B,,kV,2.0,0.0,0,-32768,32767\n"
    "1,CB_B,,,1\n"
    "50\n"
    "1\n"
    "1000,3\n"
    "10/10/2023,10:00:00.000000\n"
    "10/10/2023,10:00:00.001000\n"
    "BINARY\n"
    "1\n"
)

# Corresponding ASCII DAT content for _CFG_1991 (4 samples, 1 analog, 1 dword)
_DAT_1991_ASCII = (
    "1,0,100,0\n"
    "2,1000,200,1\n"
    "3,2000,300,1\n"
    "4,3000,400,0\n"
)

# ASCII DAT for _CFG_1999 (3 samples, TIMEMULT=0.1)
_DAT_1999_ASCII = (
    "1,0,200,0\n"
    "2,1000,400,1\n"
    "3,2000,600,0\n"
)

# ASCII DAT for _CFG_MULTIRATE (5 samples, 1 analog, 0 digital)
_DAT_MULTIRATE_ASCII = (
    "1,0,10\n"
    "2,200,20\n"
    "3,400,30\n"
    "4,2000,40\n"
    "5,4000,50\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# TestComtradeProviderCanLoad
# ─────────────────────────────────────────────────────────────────────────────


class TestComtradeProviderCanLoad:
    def test_accepts_cfg(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.cfg")) is True

    def test_accepts_cfg_uppercase(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.CFG")) is True

    def test_accepts_comtrade(self) -> None:
        assert ComtradeProvider().can_load(Path("fault.comtrade")) is True

    def test_rejects_csv(self) -> None:
        assert ComtradeProvider().can_load(Path("data.csv")) is False

    def test_rejects_dat(self) -> None:
        assert ComtradeProvider().can_load(Path("data.dat")) is False


# ─────────────────────────────────────────────────────────────────────────────
# TestCfgParsing
# ─────────────────────────────────────────────────────────────────────────────


class TestCfgParsing:
    def test_1991_station_and_device(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.station_name == "SUBST_A"
        assert cfg.rec_dev_id == "IED_01"

    def test_1991_defaults_rev_yr_to_1999(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.rev_yr == "1999"

    def test_1991_channel_counts(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.n_analog == 1
        assert cfg.n_digital == 1

    def test_1991_analog_def_fields(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        adef = cfg.analog_defs[0]
        assert adef.ch_id == "VA"
        assert adef.ph == "A"
        assert adef.uu == "kV"
        assert adef.a == pytest.approx(1.0)
        assert adef.b == pytest.approx(0.0)

    def test_1999_rev_yr_parsed(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.rev_yr == "1999"

    def test_1999_analog_primary_secondary_ps(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        cfg = _parse_cfg(cfg_path)
        adef = cfg.analog_defs[0]
        assert adef.primary == pytest.approx(110.0)
        assert adef.secondary == pytest.approx(0.11)
        assert adef.ps_flag == "P"

    def test_1999_timemult_parsed(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.timemult == pytest.approx(0.1)

    def test_1991_timemult_defaults_to_1(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.timemult == pytest.approx(1.0)

    def test_multirate_sampling_rates(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_MULTIRATE, _DAT_MULTIRATE_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.n_rates == 2
        assert cfg.sampling_rates == pytest.approx([5000.0, 500.0])

    def test_multirate_samples_per_rate(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_MULTIRATE, _DAT_MULTIRATE_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.samples_per_rate == [3, 2]
        assert cfg.total_samples == 5

    def test_start_and_trigger_time(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.start_time == datetime(2024, 1, 1, 12, 0, 0)
        assert cfg.trigger_time == datetime(2024, 1, 1, 12, 0, 0, 1000)

    def test_nominal_frequency(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.nominal_freq == pytest.approx(50.0)

    def test_dat_format_ascii(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        cfg = _parse_cfg(cfg_path)
        assert cfg.dat_format == "ASCII"

    def test_dat_format_binary(self, tmp_path: Path) -> None:
        binary_dat = _make_binary_rows([(1, 0, [100], [0])])
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        cfg = _parse_cfg(cfg_path)
        assert cfg.dat_format == "BINARY"

    def test_digital_normal_state_parsed(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, _make_binary_rows([]))
        # Normal state for CB_B is 1 in _CFG_BINARY
        cfg = _parse_cfg(cfg_path)
        assert cfg.digital_defs[0].y == 1

    def test_missing_cfg_raises_provider_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderLoadError):
            _parse_cfg(tmp_path / "nonexistent.cfg")

    def test_invalid_ft_raises_provider_load_error(self, tmp_path: Path) -> None:
        bad_cfg = _CFG_1991.replace("ASCII", "WEIRDFORMAT")
        cfg_path = _write_cfg_dat(tmp_path, bad_cfg, _DAT_1991_ASCII)
        with pytest.raises(ProviderLoadError, match="WEIRDFORMAT"):
            _parse_cfg(cfg_path)

    def test_malformed_timestamp_raises_provider_load_error(self, tmp_path: Path) -> None:
        bad_cfg = _CFG_1991.replace(
            "01/01/2024,12:00:00.000000", "NOT-A-TIMESTAMP"
        )
        cfg_path = _write_cfg_dat(tmp_path, bad_cfg, _DAT_1991_ASCII)
        with pytest.raises(ProviderLoadError):
            _parse_cfg(cfg_path)

    def test_missing_dat_raises_provider_load_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "orphan.cfg"
        cfg_path.write_text(_CFG_1991, encoding="latin-1")
        # No .dat file written
        with pytest.raises(ProviderLoadError):
            _parse_cfg(cfg_path)

    def test_unknown_rev_yr_warns_and_defaults(self, tmp_path: Path) -> None:
        bad_rev = _CFG_1999.replace(",1999\n", ",2099\n")
        cfg_path = _write_cfg_dat(tmp_path, bad_rev, _DAT_1999_ASCII)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _parse_cfg(cfg_path)
        assert cfg.rev_yr == "1999"
        assert any("2099" in str(w.message) for w in caught)


# ─────────────────────────────────────────────────────────────────────────────
# TestParseAnalogLine (unit tests for the helper)
# ─────────────────────────────────────────────────────────────────────────────


class TestParseAnalogLine:
    def test_10_field_1991_format(self) -> None:
        line = "1,VA,A,,kV,0.001,0.0,0,-32768,32767"
        adef = _parse_analog_line(line, 1)
        assert adef.ch_id == "VA"
        assert adef.a == pytest.approx(0.001)
        assert adef.primary is None

    def test_13_field_1999_format(self) -> None:
        line = "1,IA,A,,kA,0.001,0.0,0,-32768,32767,1000.0,1.0,P"
        adef = _parse_analog_line(line, 1)
        assert adef.primary == pytest.approx(1000.0)
        assert adef.secondary == pytest.approx(1.0)
        assert adef.ps_flag == "P"

    def test_empty_ch_id_uses_generated_name(self) -> None:
        line = "3,,A,,kV,1.0,0.0,0,-32768,32767"
        adef = _parse_analog_line(line, 3)
        assert adef.ch_id == "A3"

    def test_zero_a_warns(self) -> None:
        line = "1,VA,A,,kV,0.0,0.0,0,-32768,32767"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adef = _parse_analog_line(line, 1)
        assert adef.a == pytest.approx(0.0)
        assert any("a=0.0" in str(w.message) for w in caught)


# ─────────────────────────────────────────────────────────────────────────────
# TestParseDigitalLine (unit tests for the helper)
# ─────────────────────────────────────────────────────────────────────────────


class TestParseDigitalLine:
    def test_5_field_format(self) -> None:
        line = "1,52A,A,,0"
        ddef = _parse_digital_line(line, 1)
        assert ddef.ch_id == "52A"
        assert ddef.ph == "A"
        assert ddef.y == 0

    def test_3_field_format(self) -> None:
        line = "2,CB_B,1"
        ddef = _parse_digital_line(line, 2)
        assert ddef.ch_id == "CB_B"
        assert ddef.ph is None
        assert ddef.y == 1

    def test_empty_ch_id_generates_name(self) -> None:
        line = "4,,,, 0"
        ddef = _parse_digital_line(line, 4)
        assert ddef.ch_id == "D4"


# ─────────────────────────────────────────────────────────────────────────────
# TestParseTimestamp (unit tests for the helper)
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTimestamp:
    def test_full_microsecond_precision(self) -> None:
        dt = _parse_timestamp("01/01/2024,12:00:00.000100", Path("x.cfg"))
        assert dt == datetime(2024, 1, 1, 12, 0, 0, 100)

    def test_whole_second_no_subsecond(self) -> None:
        dt = _parse_timestamp("15/06/2023,08:30:00", Path("x.cfg"))
        assert dt == datetime(2023, 6, 15, 8, 30, 0)

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ProviderLoadError):
            _parse_timestamp("not-a-date", Path("x.cfg"))


# ─────────────────────────────────────────────────────────────────────────────
# TestAsciiDatParsing
# ─────────────────────────────────────────────────────────────────────────────


class TestAsciiDatParsing:
    def _cfg_for_ascii(self, tmp_path: Path) -> _CfgData:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        return _parse_cfg(cfg_path)

    def test_time_array_length(self, tmp_path: Path) -> None:
        cfg = self._cfg_for_ascii(tmp_path)
        time_arr, _, _ = _parse_ascii_dat(cfg.dat_file, cfg)
        assert len(time_arr) == 4

    def test_time_array_values_microseconds(self, tmp_path: Path) -> None:
        # TIMEMULT=1.0 (default for 1991), timestamps 0,1000,2000,3000 µs
        cfg = self._cfg_for_ascii(tmp_path)
        time_arr, _, _ = _parse_ascii_dat(cfg.dat_file, cfg)
        expected = np.array([0.0, 0.001, 0.002, 0.003])
        np.testing.assert_allclose(time_arr, expected)

    def test_analog_raw_values(self, tmp_path: Path) -> None:
        cfg = self._cfg_for_ascii(tmp_path)
        _, analog_raw, _ = _parse_ascii_dat(cfg.dat_file, cfg)
        expected = np.array([[100.0], [200.0], [300.0], [400.0]])
        np.testing.assert_allclose(analog_raw, expected)

    def test_digital_words_extracted(self, tmp_path: Path) -> None:
        # DAT digital word column: 0,1,1,0
        cfg = self._cfg_for_ascii(tmp_path)
        _, _, digital_words = _parse_ascii_dat(cfg.dat_file, cfg)
        expected = np.array([[0], [1], [1], [0]], dtype=np.uint16)
        np.testing.assert_array_equal(digital_words, expected)

    def test_timemult_applied(self, tmp_path: Path) -> None:
        # TIMEMULT=0.1, timestamps 0,1000,2000 → 0, 100µs, 200µs → 0, 0.0001, 0.0002 s
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        cfg = _parse_cfg(cfg_path)
        time_arr, _, _ = _parse_ascii_dat(cfg.dat_file, cfg)
        expected = np.array([0.0, 0.0001, 0.0002])
        np.testing.assert_allclose(time_arr, expected)

    def test_column_mismatch_raises(self, tmp_path: Path) -> None:
        bad_dat = "1,0,100\n2,1000,200\n"  # missing digital word column
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, bad_dat)
        cfg = _parse_cfg(cfg_path)
        with pytest.raises(ProviderLoadError, match="column count mismatch"):
            _parse_ascii_dat(cfg.dat_file, cfg)

    def test_empty_dat_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, "")
        cfg = _parse_cfg(cfg_path)
        with pytest.raises(ProviderLoadError):
            _parse_ascii_dat(cfg.dat_file, cfg)

    def test_no_digital_channels(self, tmp_path: Path) -> None:
        # Use multi-rate CFG which has 0 digital channels
        cfg_path = _write_cfg_dat(tmp_path, _CFG_MULTIRATE, _DAT_MULTIRATE_ASCII)
        cfg = _parse_cfg(cfg_path)
        time_arr, analog_raw, digital_words = _parse_ascii_dat(cfg.dat_file, cfg)
        assert len(time_arr) == 5
        assert analog_raw.shape == (5, 1)
        assert digital_words.shape == (5, 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestBinaryDatParsing
# ─────────────────────────────────────────────────────────────────────────────


class TestBinaryDatParsing:
    def _cfg_and_dat(self, tmp_path: Path) -> tuple[_CfgData, bytes]:
        # 3 samples: n=1/2/3, ts=0/1000/2000, analog=int16(50/100/150), dword=uint16(0/1/0)
        rows = [
            (1, 0, [50], [0]),
            (2, 1000, [100], [1]),
            (3, 2000, [150], [0]),
        ]
        binary_dat = _make_binary_rows(rows)
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        return _parse_cfg(cfg_path), binary_dat

    def test_time_array_length(self, tmp_path: Path) -> None:
        cfg, _ = self._cfg_and_dat(tmp_path)
        time_arr, _, _ = _parse_binary_dat(cfg.dat_file, cfg)
        assert len(time_arr) == 3

    def test_time_array_values(self, tmp_path: Path) -> None:
        # TIMEMULT=1.0, ts=0,1000,2000 µs
        cfg, _ = self._cfg_and_dat(tmp_path)
        time_arr, _, _ = _parse_binary_dat(cfg.dat_file, cfg)
        np.testing.assert_allclose(time_arr, [0.0, 0.001, 0.002])

    def test_analog_raw_values(self, tmp_path: Path) -> None:
        cfg, _ = self._cfg_and_dat(tmp_path)
        _, analog_raw, _ = _parse_binary_dat(cfg.dat_file, cfg)
        expected = np.array([[50.0], [100.0], [150.0]])
        np.testing.assert_allclose(analog_raw, expected)

    def test_digital_words_extracted(self, tmp_path: Path) -> None:
        cfg, _ = self._cfg_and_dat(tmp_path)
        _, _, digital_words = _parse_binary_dat(cfg.dat_file, cfg)
        expected = np.array([[0], [1], [0]], dtype=np.uint16)
        np.testing.assert_array_equal(digital_words, expected)

    def test_negative_analog_raw(self, tmp_path: Path) -> None:
        rows = [(1, 0, [-1000], [0])]
        binary_dat = _make_binary_rows(rows)
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        cfg = _parse_cfg(cfg_path)
        _, analog_raw, _ = _parse_binary_dat(cfg.dat_file, cfg)
        assert analog_raw[0, 0] == pytest.approx(-1000.0)

    def test_size_mismatch_raises(self, tmp_path: Path) -> None:
        # Truncate by 1 byte to cause mismatch
        rows = [(1, 0, [100], [0]), (2, 1000, [200], [1]), (3, 2000, [300], [0])]
        bad_bytes = _make_binary_rows(rows)[:-1]
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, bad_bytes)
        cfg = _parse_cfg(cfg_path)
        with pytest.raises(ProviderLoadError, match="not a multiple"):
            _parse_binary_dat(cfg.dat_file, cfg)

    def test_empty_dat_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, b"")
        cfg = _parse_cfg(cfg_path)
        with pytest.raises(ProviderLoadError):
            _parse_binary_dat(cfg.dat_file, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# TestAnalogScaling
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalogScaling:
    def _make_defs(self, params: list[tuple[float, float]]) -> list:
        from app.providers.comtrade.comtrade_provider import _AnalogDef

        return [
            _AnalogDef(
                index=i + 1,
                ch_id=f"CH{i}",
                ph=None,
                uu="kV",
                a=a,
                b=b,
                skew=0.0,
                primary=None,
                secondary=None,
                ps_flag=None,
            )
            for i, (a, b) in enumerate(params)
        ]

    def test_single_channel_scaling(self) -> None:
        raw = np.array([[100.0], [200.0], [300.0]])
        defs = self._make_defs([(0.5, 10.0)])
        result = _apply_analog_scaling(raw, defs)
        expected = np.array([[60.0], [110.0], [160.0]])
        np.testing.assert_allclose(result, expected)

    def test_multi_channel_scaling(self) -> None:
        raw = np.array([[10.0, 20.0], [30.0, 40.0]])
        defs = self._make_defs([(2.0, 0.0), (1.0, 5.0)])
        result = _apply_analog_scaling(raw, defs)
        expected = np.array([[20.0, 25.0], [60.0, 45.0]])
        np.testing.assert_allclose(result, expected)

    def test_identity_scaling_unchanged(self) -> None:
        raw = np.array([[1.0, 2.0], [3.0, 4.0]])
        defs = self._make_defs([(1.0, 0.0), (1.0, 0.0)])
        result = _apply_analog_scaling(raw, defs)
        np.testing.assert_allclose(result, raw)

    def test_negative_raw_values(self) -> None:
        raw = np.array([[-32768.0]])
        defs = self._make_defs([(0.001, 0.0)])
        result = _apply_analog_scaling(raw, defs)
        np.testing.assert_allclose(result, [[-32.768]])

    def test_empty_analog_array_passthrough(self) -> None:
        raw = np.empty((5, 0), dtype=np.float64)
        result = _apply_analog_scaling(raw, [])
        assert result.shape == (5, 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestDigitalExtraction
# ─────────────────────────────────────────────────────────────────────────────


class TestDigitalExtraction:
    def test_single_channel_bit0(self) -> None:
        # word=0b0001 → bit 0 = 1; word=0b0000 → bit 0 = 0
        words = np.array([[1], [0], [1]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=1)
        expected = np.array([[1], [0], [1]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_multi_channel_same_word(self) -> None:
        # 3 channels in 1 word: word=0b101 = 5 → ch0=1, ch1=0, ch2=1
        words = np.array([[5]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=3)
        expected = np.array([[1, 0, 1]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_channels_across_two_words(self) -> None:
        # 17 channels require 2 words
        # word0 = 0xFFFF (all bits 1), word1 = 0x0001 (only bit 0 set)
        # ch 0..15 → all 1; ch 16 → 1
        words = np.array([[0xFFFF, 0x0001]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=17)
        assert result.shape == (1, 17)
        assert all(result[0, :16] == 1)
        assert result[0, 16] == 1

    def test_channel_17_only_bit0_of_second_word(self) -> None:
        # word0=0, word1=0b10 → ch16 = bit0 of word1 = 0; ch17 = bit1 of word1 = 1
        words = np.array([[0, 2]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=18)
        assert result[0, 16] == 0   # bit 0 of word1
        assert result[0, 17] == 1   # bit 1 of word1

    def test_zero_digital_channels_empty_array(self) -> None:
        words = np.empty((4, 0), dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=0)
        assert result.shape == (4, 0)

    def test_output_dtype_is_int8(self) -> None:
        words = np.array([[3]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=2)
        assert result.dtype == np.int8

    def test_normal_state_not_inverted(self) -> None:
        # normal_state=1 means channel is normally closed. Raw bit=1 → stored as 1.
        # The inversion (active = raw XOR normal_state) is NOT applied here.
        words = np.array([[1]], dtype=np.uint16)
        result = _extract_digital_channels(words, n_digital=1)
        assert result[0, 0] == 1  # raw bit, not inverted


# ─────────────────────────────────────────────────────────────────────────────
# TestFullLoad — end-to-end integration through ComtradeProvider.load()
# ─────────────────────────────────────────────────────────────────────────────


class TestFullLoad:
    def test_1991_ascii_returns_disturbance_record(self, tmp_path: Path) -> None:
        from app.models import DisturbanceRecord

        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert isinstance(record, DisturbanceRecord)

    def test_1991_ascii_validate_passes(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert record.validate() == []

    def test_1991_ascii_metadata(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert record.metadata.station_name == "SUBST_A"
        assert record.metadata.recorder_name == "IED_01"
        assert record.metadata.provider_type == "COMTRADE"
        assert record.metadata.nominal_frequency == pytest.approx(50.0)
        assert record.metadata.timezone is None

    def test_1991_ascii_time_column(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert "time" in record.waveform_data.columns
        np.testing.assert_allclose(
            record.waveform_data["time"].to_numpy(),
            [0.0, 0.001, 0.002, 0.003],
        )

    def test_1991_ascii_analog_scaling_applied(self, tmp_path: Path) -> None:
        # a=1.0, b=0.0 → physical == raw
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        expected = [100.0, 200.0, 300.0, 400.0]
        np.testing.assert_allclose(record.waveform_data["VA"].to_numpy(), expected)

    def test_1991_ascii_digital_channel_values(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        expected = [0, 1, 1, 0]
        np.testing.assert_array_equal(record.waveform_data["CB_A"].to_numpy(), expected)

    def test_1999_ascii_with_timemult_and_scaling(self, tmp_path: Path) -> None:
        # a=0.5, b=10.0; TIMEMULT=0.1
        # raw=[200,400,600] → physical=[110,210,310]
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        record = ComtradeProvider().load(cfg_path)
        np.testing.assert_allclose(
            record.waveform_data["VA"].to_numpy(), [110.0, 210.0, 310.0]
        )

    def test_1999_ascii_timing_info(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert record.timing_info.start_time == datetime(2023, 3, 15, 8, 30, 0)
        assert record.timing_info.time_multiplier == pytest.approx(0.1)
        assert record.timing_info.timezone is None

    def test_1999_ascii_primary_secondary_stored(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1999, _DAT_1999_ASCII)
        record = ComtradeProvider().load(cfg_path)
        ch = record.analog_channels[0]
        assert ch.primary_ratio == pytest.approx(110.0)
        assert ch.secondary_ratio == pytest.approx(0.11)

    def test_multirate_sampling_info(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_MULTIRATE, _DAT_MULTIRATE_ASCII)
        record = ComtradeProvider().load(cfg_path)
        si = record.sampling_info
        assert si.sampling_rates == pytest.approx([5000.0, 500.0])
        assert si.samples_per_rate == [3, 2]

    def test_binary_returns_valid_record(self, tmp_path: Path) -> None:
        rows = [(1, 0, [50], [0]), (2, 1000, [100], [1]), (3, 2000, [75], [0])]
        binary_dat = _make_binary_rows(rows)
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        record = ComtradeProvider().load(cfg_path)
        assert record.validate() == []

    def test_binary_analog_scaling_applied(self, tmp_path: Path) -> None:
        # a=2.0, b=0.0 in _CFG_BINARY; raw=[50,100,75] → physical=[100,200,150]
        rows = [(1, 0, [50], [0]), (2, 1000, [100], [1]), (3, 2000, [75], [0])]
        binary_dat = _make_binary_rows(rows)
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        record = ComtradeProvider().load(cfg_path)
        np.testing.assert_allclose(
            record.waveform_data["VB"].to_numpy(), [100.0, 200.0, 150.0]
        )

    def test_binary_digital_channel_values(self, tmp_path: Path) -> None:
        rows = [(1, 0, [50], [0]), (2, 1000, [100], [1]), (3, 2000, [75], [0])]
        binary_dat = _make_binary_rows(rows)
        cfg_path = _write_cfg_dat(tmp_path, _CFG_BINARY, binary_dat)
        record = ComtradeProvider().load(cfg_path)
        np.testing.assert_array_equal(
            record.waveform_data["CB_B"].to_numpy(), [0, 1, 0]
        )

    def test_disturbance_info_is_none(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert record.disturbance_info is None

    def test_analog_channel_names_match_waveform_columns(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        for ch in record.analog_channels:
            assert ch.name in record.waveform_data.columns

    def test_digital_channel_names_match_waveform_columns(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        for ch in record.digital_channels:
            assert ch.name in record.waveform_data.columns

    def test_sample_count_matches_dat_rows(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, _DAT_1991_ASCII)
        record = ComtradeProvider().load(cfg_path)
        assert record.sample_count() == 4


# ─────────────────────────────────────────────────────────────────────────────
# TestErrorHandling
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_missing_cfg_raises_provider_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderLoadError):
            ComtradeProvider().load(tmp_path / "missing.cfg")

    def test_missing_dat_raises_provider_load_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "no_dat.cfg"
        cfg_path.write_text(_CFG_1991, encoding="latin-1")
        with pytest.raises(ProviderLoadError):
            ComtradeProvider().load(cfg_path)

    def test_empty_dat_raises_provider_load_error(self, tmp_path: Path) -> None:
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, "")
        with pytest.raises(ProviderLoadError):
            ComtradeProvider().load(cfg_path)

    def test_binary32_raises_provider_load_error(self, tmp_path: Path) -> None:
        binary32_cfg = _CFG_BINARY.replace("BINARY\n", "BINARY32\n")
        cfg_path = _write_cfg_dat(tmp_path, binary32_cfg, b"\x00" * 16)
        with pytest.raises(ProviderLoadError, match="BINARY32"):
            ComtradeProvider().load(cfg_path)

    def test_sample_count_mismatch_warns_not_raises(self, tmp_path: Path) -> None:
        # CFG declares 4 samples but DAT has only 3
        short_dat = "1,0,100,0\n2,1000,200,1\n3,2000,300,1\n"
        cfg_path = _write_cfg_dat(tmp_path, _CFG_1991, short_dat)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = ComtradeProvider().load(cfg_path)
        assert record.sample_count() == 3
        assert any("mismatch" in str(w.message).lower() for w in caught)

    def test_malformed_cfg_raises_provider_load_error(self, tmp_path: Path) -> None:
        truncated = "SUBST,IED\n2,1A,1D\n"  # missing channel lines
        cfg_path = _write_cfg_dat(tmp_path, truncated, _DAT_1991_ASCII)
        with pytest.raises(ProviderLoadError):
            ComtradeProvider().load(cfg_path)

    def test_provider_load_error_has_cause_for_io_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderLoadError) as exc_info:
            ComtradeProvider().load(tmp_path / "gone.cfg")
        assert exc_info.value.__cause__ is not None
