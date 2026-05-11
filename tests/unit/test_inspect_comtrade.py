"""Unit tests for tools/inspect_comtrade.py.

Uses minimal synthetic CFG fixtures written to temp files — no real recordings needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_comtrade import (
    ComtradeMetadata,
    _locate_cfg,
    format_json_summary,
    format_text_summary,
    parse_cfg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_MINIMAL_CFG = """\
TESTSTATION,REC01,1999
9,6A,3D
1,VA,A,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
2,VB,B,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
3,VC,C,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
4,IA,A,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
5,IB,B,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
6,IC,C,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
1,TRIP,,,0
2,START,,,0
3,ALARM,,,0
50
1
5000,32000
06/03/2026,18:04:08.817733
06/03/2026,18:04:09.317733
BINARY
1
"""

_DIGITAL_ONLY_CFG = """\
SIMPLESTATION,R2,1999
3,0A,3D
1,CB1,,,0
2,CB2,,,0
3,FAULT,,,0
50
1
100,500
01/01/2024,00:00:00.000000
01/01/2024,00:00:01.000000
ASCII
1
"""

_NO_REV_YR_CFG = """\
OLDSTATION,R3
3,2A,1D
1,VA,A,,kV,1.0,0.0,0,-32767,32767
2,IA,A,,A,1.0,0.0,0,-32767,32767
1,TRIP,,,0
50
1
3000,15000
15/06/2023,10:30:00
15/06/2023,10:30:00.500000
ASCII
1
"""


def _write_cfg(tmp_path: Path, content: str, stem: str = "event") -> Path:
    cfg = tmp_path / f"{stem}.cfg"
    cfg.write_text(content, encoding="latin-1")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# TestLocateCfg
# ─────────────────────────────────────────────────────────────────────────────


class TestLocateCfg:
    def test_accepts_cfg_file_directly(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        assert _locate_cfg(cfg) == cfg

    def test_finds_cfg_in_directory(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        assert _locate_cfg(tmp_path) == cfg

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _locate_cfg(tmp_path / "missing")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _locate_cfg(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# TestParseCfg — basic metadata extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCfg:
    def test_station_name(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.station_name == "TESTSTATION"

    def test_rec_dev_id(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.rec_dev_id == "REC01"

    def test_rev_yr(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.rev_yr == "1999"

    def test_n_analog(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.n_analog == 6

    def test_n_digital(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.n_digital == 3

    def test_analog_channel_names(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.analog_channel_names == ["VA", "VB", "VC", "IA", "IB", "IC"]

    def test_digital_channel_names(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.digital_channel_names == ["TRIP", "START", "ALARM"]

    def test_sampling_rate(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.sampling_rates_hz == [5000.0]

    def test_total_samples(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.total_samples == 32000

    def test_nominal_frequency(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.nominal_frequency_hz == 50.0

    def test_start_time_iso(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.start_time.startswith("2026-03-06T18:04:08")

    def test_trigger_time_iso(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.trigger_time.startswith("2026-03-06T18:04:09")

    def test_dat_format(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.dat_format == "BINARY"

    def test_dat_path_none_when_no_dat_file(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert meta.dat_path is None
        assert meta.dat_size_bytes is None

    def test_dat_detected_when_present(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        dat = tmp_path / "event.dat"
        dat.write_bytes(b"\x00" * 256)
        meta = parse_cfg(cfg)
        assert meta.dat_path is not None
        assert meta.dat_size_bytes == 256

    def test_duration_computed_from_samples(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        # 32000 samples at 5000 Hz = 6.4 s
        assert meta.duration_seconds == pytest.approx(6.4, abs=0.01)

    def test_digital_only_record(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _DIGITAL_ONLY_CFG)
        meta = parse_cfg(cfg)
        assert meta.n_analog == 0
        assert meta.n_digital == 3
        assert meta.digital_channel_names == ["CB1", "CB2", "FAULT"]
        assert meta.analog_channel_names == []

    def test_no_rev_yr_defaults_to_1999(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _NO_REV_YR_CFG)
        meta = parse_cfg(cfg)
        assert meta.rev_yr == "1999"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_cfg(tmp_path / "ghost.cfg")

    def test_cfg_path_stored_as_string(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        assert isinstance(meta.cfg_path, str)


# ─────────────────────────────────────────────────────────────────────────────
# TestFormatters
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatters:
    def test_text_summary_contains_station(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        text = format_text_summary(meta)
        assert "TESTSTATION" in text

    def test_text_summary_contains_channel_names(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        text = format_text_summary(meta)
        assert "VA" in text
        assert "TRIP" in text

    def test_text_summary_contains_sampling_rate(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        text = format_text_summary(meta)
        assert "5000" in text

    def test_json_summary_is_valid_json(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        j = format_json_summary(meta)
        parsed = json.loads(j)
        assert parsed["station_name"] == "TESTSTATION"

    def test_json_summary_contains_channels(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, _MINIMAL_CFG)
        meta = parse_cfg(cfg)
        j = format_json_summary(meta)
        parsed = json.loads(j)
        assert "VA" in parsed["analog_channel_names"]
        assert "TRIP" in parsed["digital_channel_names"]
