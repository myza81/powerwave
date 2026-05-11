"""Unit tests for tools/inspect_csv_timeseries.py.

Uses minimal in-memory CSV fixtures written to temp files.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from inspect_csv_timeseries import (
    CsvMetadata,
    _check_single_ambiguity,
    _find_timestamp_column,
    _infer_timestamp_format,
    format_json_summary,
    format_text_summary,
    inspect_file,
)
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CSV fixtures
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_CSV = """\
time,MW,MVar,Frequency
0.0,100.0,50.0,49.98
60.0,105.0,48.0,50.01
120.0,110.0,52.0,50.00
180.0,98.0,49.0,49.99
"""

_ISO_CSV = """\
Timestamp,MW,MVar
2026-03-06 17:25:00,100.0,50.0
2026-03-06 17:26:00,105.0,48.0
2026-03-06 17:27:00,110.0,52.0
"""

_US_AMBIGUOUS_CSV = """\
Timestamp,MW,Frequency
3/6/2026 6:00:00 AM,100.0,49.98
3/6/2026 6:01:00 AM,105.0,50.01
3/6/2026 6:02:00 AM,110.0,50.00
"""

_EU_UNAMBIGUOUS_CSV = """\
Timestamp,MW,Frequency
13/6/2026 6:00:00,100.0,49.98
13/6/2026 6:01:00,105.0,50.01
13/6/2026 6:02:00,110.0,50.00
"""

_NO_TIMESTAMP_CSV = """\
MW,MVar,Frequency
100.0,50.0,49.98
105.0,48.0,50.01
110.0,52.0,50.00
"""


def _write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TestFindTimestampColumn
# ─────────────────────────────────────────────────────────────────────────────


class TestFindTimestampColumn:
    def test_finds_time_column(self) -> None:
        df = pd.read_csv(io.StringIO(_NUMERIC_CSV))
        assert _find_timestamp_column(df) == "time"

    def test_finds_timestamp_column(self) -> None:
        df = pd.read_csv(io.StringIO(_ISO_CSV))
        assert _find_timestamp_column(df) == "Timestamp"

    def test_no_timestamp_returns_none(self) -> None:
        df = pd.read_csv(io.StringIO(_NO_TIMESTAMP_CSV))
        assert _find_timestamp_column(df) is None


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckSingleAmbiguity
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckSingleAmbiguity:
    def test_ambiguous_date(self) -> None:
        result = _check_single_ambiguity("3/6/2026 6:00:00 AM")
        assert result.is_ambiguous is True
        assert result.us_interpretation != result.eu_interpretation

    def test_unambiguous_day_gt_12(self) -> None:
        result = _check_single_ambiguity("13/6/2026 6:00:00")
        assert result.is_ambiguous is False

    def test_unambiguous_month_gt_12(self) -> None:
        result = _check_single_ambiguity("6/13/2026")
        assert result.is_ambiguous is False

    def test_same_result_not_reported_ambiguous(self) -> None:
        # 1/1/2026 → both M/D and D/M give Jan 1
        result = _check_single_ambiguity("1/1/2026")
        assert result.is_ambiguous is False  # same result, not ambiguous


# ─────────────────────────────────────────────────────────────────────────────
# TestInspectFile
# ─────────────────────────────────────────────────────────────────────────────


class TestInspectFile:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_file(tmp_path / "ghost.csv")

    def test_numeric_time_detected(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_column == "time"
        assert meta.timestamp_info is not None
        assert "numeric" in meta.timestamp_info.detected_type

    def test_iso_timestamp_detected(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _ISO_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_column == "Timestamp"
        assert meta.timestamp_info is not None
        assert not meta.timestamp_info.ambiguity_detected

    def test_ambiguous_format_detected(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _US_AMBIGUOUS_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_info is not None
        assert meta.timestamp_info.ambiguity_detected is True

    def test_unambiguous_eu_not_flagged(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _EU_UNAMBIGUOUS_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_info is not None
        assert meta.timestamp_info.ambiguity_detected is False

    def test_no_timestamp_column_gives_warning(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NO_TIMESTAMP_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_column is None
        assert any("No timestamp" in w for w in meta.warnings)

    def test_data_columns_exclude_timestamp(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        assert "time" not in meta.data_columns
        assert "MW" in meta.data_columns

    def test_interval_computed_for_numeric(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_info is not None
        assert meta.timestamp_info.nominal_interval_seconds == pytest.approx(60.0, rel=0.01)

    def test_interval_computed_for_iso(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _ISO_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_info is not None
        assert meta.timestamp_info.nominal_interval_seconds == pytest.approx(60.0, rel=0.01)

    def test_start_time_present_for_iso(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _ISO_CSV)
        meta = inspect_file(p)
        assert meta.timestamp_info is not None
        assert meta.timestamp_info.start_time is not None
        assert "2026-03-06" in meta.timestamp_info.start_time

    def test_file_path_stored(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        assert str(p.resolve()) == meta.file_path

    def test_total_columns_correct(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        assert meta.total_columns == 4  # time, MW, MVar, Frequency


# ─────────────────────────────────────────────────────────────────────────────
# TestFormatters
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatters:
    def test_text_summary_contains_column_names(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _NUMERIC_CSV)
        meta = inspect_file(p)
        text = format_text_summary(meta)
        assert "MW" in text
        assert "MVar" in text

    def test_text_summary_flags_ambiguity(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _US_AMBIGUOUS_CSV)
        meta = inspect_file(p)
        text = format_text_summary(meta)
        assert "YES" in text or "ambiguous" in text.lower() or "AMBIGUOUS" in text

    def test_json_summary_is_valid(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _ISO_CSV)
        meta = inspect_file(p)
        j = format_json_summary(meta)
        parsed = json.loads(j)
        assert "timestamp_column" in parsed
        assert parsed["timestamp_column"] == "Timestamp"

    def test_json_includes_ambiguity_flag(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, _US_AMBIGUOUS_CSV)
        meta = inspect_file(p)
        j = format_json_summary(meta)
        parsed = json.loads(j)
        assert parsed["timestamp_info"]["ambiguity_detected"] is True
