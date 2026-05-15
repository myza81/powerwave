"""Unit tests for CsvProvider.

Covers: can_load(), load() with numeric time, datetime timestamps, no time column,
analog/digital inference, unit inference, sampling rate estimation, error paths,
and DisturbanceRecord validation.

All tests use tmp_path (pytest fixture) for file-based testing.
"""
from __future__ import annotations

import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models import DisturbanceRecord
from app.providers.base.exceptions import ProviderLoadError
from app.providers.csv.csv_provider import (
    CsvProvider,
    _detect_time_column,
    _estimate_rate,
    _infer_unit,
    _is_digital_column,
    _is_time_like_column,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# can_load()
# ---------------------------------------------------------------------------


class TestCanLoad:
    def test_accepts_csv(self) -> None:
        assert CsvProvider().can_load(Path("data.csv")) is True

    def test_accepts_csv_uppercase(self) -> None:
        assert CsvProvider().can_load(Path("DATA.CSV")) is True

    def test_rejects_cfg(self) -> None:
        assert CsvProvider().can_load(Path("fault.cfg")) is False

    def test_rejects_xlsx(self) -> None:
        assert CsvProvider().can_load(Path("data.xlsx")) is False

    def test_rejects_dat(self) -> None:
        assert CsvProvider().can_load(Path("data.dat")) is False

    def test_rejects_no_extension(self) -> None:
        assert CsvProvider().can_load(Path("datafile")) is False

    def test_provider_name(self) -> None:
        assert CsvProvider.provider_name == "csv"


# ---------------------------------------------------------------------------
# _detect_time_column()
# ---------------------------------------------------------------------------


class TestDetectTimeColumn:
    def test_detects_time(self) -> None:
        assert _detect_time_column(["time", "VA", "VB"]) == "time"

    def test_detects_t(self) -> None:
        assert _detect_time_column(["t", "IA"]) == "t"

    def test_detects_seconds(self) -> None:
        assert _detect_time_column(["seconds", "VA"]) == "seconds"

    def test_detects_sec(self) -> None:
        assert _detect_time_column(["sec", "VA"]) == "sec"

    def test_detects_timestamp(self) -> None:
        assert _detect_time_column(["timestamp", "VA"]) == "timestamp"

    def test_detects_datetime(self) -> None:
        assert _detect_time_column(["datetime", "VA"]) == "datetime"

    def test_case_insensitive(self) -> None:
        assert _detect_time_column(["Time", "VA"]) == "Time"

    def test_returns_none_when_absent(self) -> None:
        assert _detect_time_column(["VA", "VB", "VC"]) is None

    def test_returns_first_match(self) -> None:
        assert _detect_time_column(["t", "time", "VA"]) == "t"

    def test_detects_pandas_duplicate_time_header(self) -> None:
        assert _detect_time_column(["Time.1", "MW"]) == "Time.1"


class TestIsTimeLikeColumn:
    def test_plain_time_name(self) -> None:
        assert _is_time_like_column("Time") is True

    def test_pandas_duplicate_time_name(self) -> None:
        assert _is_time_like_column("Time.1") is True

    def test_non_numeric_suffix_not_duplicate_time_name(self) -> None:
        assert _is_time_like_column("Time.A") is False

    def test_non_time_column(self) -> None:
        assert _is_time_like_column("Tie-Line") is False


# ---------------------------------------------------------------------------
# _infer_unit()
# ---------------------------------------------------------------------------


class TestInferUnit:
    def test_voltage_va(self) -> None:
        assert _infer_unit("VA") == "kV"

    def test_voltage_vb(self) -> None:
        assert _infer_unit("VB") == "kV"

    def test_voltage_keyword_volt(self) -> None:
        assert _infer_unit("voltage_a") == "kV"

    def test_voltage_kv_keyword(self) -> None:
        assert _infer_unit("kv_phase") == "kV"

    def test_current_ia(self) -> None:
        assert _infer_unit("IA") == "A"

    def test_current_ic(self) -> None:
        assert _infer_unit("IC") == "A"

    def test_current_keyword(self) -> None:
        assert _infer_unit("current_a") == "A"

    def test_frequency(self) -> None:
        assert _infer_unit("FREQ") == "Hz"

    def test_frequency_hz(self) -> None:
        assert _infer_unit("hz_measurement") == "Hz"

    def test_mvar(self) -> None:
        assert _infer_unit("MVAR") == "MVar"

    def test_mw(self) -> None:
        assert _infer_unit("MW") == "MW"

    def test_unknown(self) -> None:
        assert _infer_unit("RPM") == "unknown"

    def test_unknown_generic(self) -> None:
        assert _infer_unit("CHANNEL_X") == "unknown"


# ---------------------------------------------------------------------------
# _is_digital_column()
# ---------------------------------------------------------------------------


class TestIsDigitalColumn:
    def test_bool_dtype_is_digital(self) -> None:
        s = pd.Series([True, False, True])
        assert _is_digital_column(s, "trip") is True

    def test_binary_with_status_name(self) -> None:
        s = pd.Series([0, 1, 0, 1])
        assert _is_digital_column(s, "breaker_status") is True

    def test_binary_without_status_name_is_analog(self) -> None:
        s = pd.Series([0, 1, 0, 1])
        assert _is_digital_column(s, "VA") is False

    def test_non_binary_is_analog(self) -> None:
        s = pd.Series([0.0, 1.5, 2.0])
        assert _is_digital_column(s, "trip") is False

    def test_trip_keyword(self) -> None:
        s = pd.Series([0, 1])
        assert _is_digital_column(s, "trip_output") is True

    def test_pickup_keyword(self) -> None:
        s = pd.Series([0, 0, 1])
        assert _is_digital_column(s, "pickup_a") is True

    def test_relay_keyword(self) -> None:
        s = pd.Series([1, 1, 0])
        assert _is_digital_column(s, "relay_op") is True

    def test_empty_series_not_digital(self) -> None:
        s = pd.Series([], dtype=float)
        assert _is_digital_column(s, "trip") is False

    def test_all_nan_not_digital(self) -> None:
        s = pd.Series([float("nan"), float("nan")])
        assert _is_digital_column(s, "trip") is False


# ---------------------------------------------------------------------------
# _estimate_rate()
# ---------------------------------------------------------------------------


class TestEstimateRate:
    def test_uniform_1000hz(self) -> None:
        t = np.arange(0, 1.0, 0.001)
        rate = _estimate_rate(t)
        assert abs(rate - 1000.0) < 0.1

    def test_uniform_50hz(self) -> None:
        t = np.arange(0, 1.0, 0.02)
        rate = _estimate_rate(t)
        assert abs(rate - 50.0) < 0.1

    def test_single_sample_returns_zero(self) -> None:
        assert _estimate_rate(np.array([0.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert _estimate_rate(np.array([])) == 0.0

    def test_zero_interval_returns_zero(self) -> None:
        assert _estimate_rate(np.array([0.0, 0.0, 0.0])) == 0.0


# ---------------------------------------------------------------------------
# load() — numeric time column
# ---------------------------------------------------------------------------


class TestLoadNumericTime:
    def test_returns_disturbance_record(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,VA,VB
            0.0,275.0,274.0
            0.001,276.0,273.0
        """)
        record = CsvProvider().load(p)
        assert isinstance(record, DisturbanceRecord)

    def test_time_column_preserved(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,VA
            0.0,100.0
            0.01,101.0
        """)
        record = CsvProvider().load(p)
        assert "time" in record.waveform_data.columns

    def test_analog_channels_created(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,VA,VB,VC
            0.0,275.0,274.0,273.0
            0.001,276.0,273.0,272.0
        """)
        record = CsvProvider().load(p)
        assert len(record.analog_channels) == 3
        assert [ch.name for ch in record.analog_channels] == ["VA", "VB", "VC"]

    def test_sampling_rate_estimated(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{i*0.001:.3f},100.0" for i in range(100))
        p = _write_csv(tmp_path, "test.csv", f"time,VA\n{rows}\n")
        record = CsvProvider().load(p)
        rate = record.sampling_info.sampling_rates[0]
        assert abs(rate - 1000.0) < 1.0

    def test_sample_count_correct(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,VA
            0.0,1.0
            0.001,2.0
            0.002,3.0
        """)
        record = CsvProvider().load(p)
        assert record.sample_count() == 3
        assert record.sampling_info.samples_per_rate == [3]

    def test_waveform_data_has_channel_columns(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,IA,IB
            0.0,1.0,2.0
            0.001,1.1,2.1
        """)
        record = CsvProvider().load(p)
        assert "IA" in record.waveform_data.columns
        assert "IB" in record.waveform_data.columns

    def test_metadata_provider_type(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "time,VA\n0.0,1.0\n")
        record = CsvProvider().load(p)
        assert record.metadata.provider_type == "csv"

    def test_metadata_source_file(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "time,VA\n0.0,1.0\n")
        record = CsvProvider().load(p)
        assert str(p) == record.metadata.source_file

    def test_metadata_station_name(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "time,VA\n0.0,1.0\n")
        record = CsvProvider().load(p)
        assert record.metadata.station_name == "Unknown"

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            time,VA,VB
            0.0,100.0,101.0
            0.001,102.0,103.0
        """)
        record = CsvProvider().load(p)
        assert record.validate() == []

    def test_duplicate_time_column_is_not_loaded_or_warned(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            Time,Time.1,MW,Frequency
            0.0,17:25,100.0,50.0
            60.0,17:26,101.0,49.9
        """)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = CsvProvider().load(p)

        warning_text = "\n".join(str(w.message) for w in caught)
        assert "Time.1" not in warning_text
        assert "Time.1" not in record.waveform_data.columns
        assert [ch.name for ch in record.analog_channels] == ["MW", "Frequency"]


# ---------------------------------------------------------------------------
# load() — datetime/timestamp column
# ---------------------------------------------------------------------------


class TestLoadTimestampColumn:
    def test_iso_timestamp_column(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            timestamp,VA
            2024-01-01 12:00:00.000,100.0
            2024-01-01 12:00:00.001,101.0
            2024-01-01 12:00:00.002,102.0
        """)
        record = CsvProvider().load(p)
        assert isinstance(record, DisturbanceRecord)
        assert record.sample_count() == 3

    def test_start_time_populated_from_timestamp(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            timestamp,VA
            2024-06-15 08:30:00.000,275.0
            2024-06-15 08:30:00.001,276.0
        """)
        record = CsvProvider().load(p)
        assert record.timing_info.start_time == datetime(2024, 6, 15, 8, 30, 0)

    def test_time_array_starts_at_zero(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            timestamp,VA
            2024-01-01 00:00:00.000,1.0
            2024-01-01 00:00:00.010,2.0
            2024-01-01 00:00:00.020,3.0
        """)
        record = CsvProvider().load(p)
        first_t = float(record.waveform_data["time"].iloc[0])
        assert abs(first_t) < 1e-9

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            datetime,IA
            2024-03-10 10:00:00,1.0
            2024-03-10 10:00:01,1.1
        """)
        record = CsvProvider().load(p)
        assert record.validate() == []


# ---------------------------------------------------------------------------
# load() — no time column
# ---------------------------------------------------------------------------


class TestLoadNoTimeColumn:
    def test_no_time_column_creates_integer_index(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", """\
            VA,VB,VC
            275.0,274.0,273.0
            276.0,275.0,274.0
            277.0,276.0,275.0
        """)
        record = CsvProvider().load(p)
        assert "time" in record.waveform_data.columns
        times = record.waveform_data["time"].to_numpy()
        np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])

    def test_no_time_fallback_start_time(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "VA\n100.0\n101.0\n")
        record = CsvProvider().load(p)
        assert record.timing_info.start_time == datetime(2000, 1, 1)

    def test_sample_rate_zero_when_no_time(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "VA,VB\n1.0,2.0\n3.0,4.0\n")
        record = CsvProvider().load(p)
        assert record.sampling_info.sampling_rates[0] == 0.0

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "test.csv", "VA,VB\n1.0,2.0\n3.0,4.0\n")
        record = CsvProvider().load(p)
        assert record.validate() == []


# ---------------------------------------------------------------------------
# Unit inference via load()
# ---------------------------------------------------------------------------


class TestUnitInference:
    def _load(self, tmp_path: Path, header: str, row: str) -> DisturbanceRecord:
        p = _write_csv(tmp_path, "t.csv", f"{header}\n{row}\n")
        return CsvProvider().load(p)

    def test_va_unit_kv(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,VA", "0.0,275.0")
        ch = next(c for c in r.analog_channels if c.name == "VA")
        assert ch.unit == "kV"

    def test_ia_unit_a(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,IA", "0.0,100.0")
        ch = next(c for c in r.analog_channels if c.name == "IA")
        assert ch.unit == "A"

    def test_freq_unit_hz(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,FREQ", "0.0,50.0")
        ch = next(c for c in r.analog_channels if c.name == "FREQ")
        assert ch.unit == "Hz"

    def test_mw_unit(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,MW", "0.0,100.0")
        ch = next(c for c in r.analog_channels if c.name == "MW")
        assert ch.unit == "MW"

    def test_mvar_unit(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,MVAR", "0.0,50.0")
        ch = next(c for c in r.analog_channels if c.name == "MVAR")
        assert ch.unit == "MVAr"

    def test_unknown_unit(self, tmp_path: Path) -> None:
        r = self._load(tmp_path, "time,RPM", "0.0,1500.0")
        ch = next(c for c in r.analog_channels if c.name == "RPM")
        assert ch.unit == "unknown"


# ---------------------------------------------------------------------------
# Digital channel inference
# ---------------------------------------------------------------------------


class TestDigitalInference:
    def test_binary_trip_column_is_digital(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA,trip\n0.0,275.0,0\n0.001,276.0,1\n")
        record = CsvProvider().load(p)
        assert len(record.digital_channels) == 1
        assert record.digital_channels[0].name == "trip"

    def test_binary_trip_not_in_analog(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA,trip\n0.0,275.0,0\n0.001,276.0,1\n")
        record = CsvProvider().load(p)
        analog_names = [ch.name for ch in record.analog_channels]
        assert "trip" not in analog_names

    def test_digital_values_stored_as_int8(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,breaker\n0.0,0\n0.001,1\n")
        record = CsvProvider().load(p)
        vals = record.waveform_data["breaker"].to_numpy()
        assert vals.dtype == np.int8

    def test_non_binary_column_not_digital(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA\n0.0,275.0\n0.001,100.0\n")
        record = CsvProvider().load(p)
        assert len(record.digital_channels) == 0

    def test_binary_without_status_name_stays_analog(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA\n0.0,0.0\n0.001,1.0\n")
        record = CsvProvider().load(p)
        assert len(record.digital_channels) == 0
        assert len(record.analog_channels) == 1

    def test_digital_normal_state_zero(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,trip\n0.0,0\n0.001,1\n")
        record = CsvProvider().load(p)
        assert record.digital_channels[0].normal_state == 0

    def test_validate_passes_with_digital(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA,trip\n0.0,275.0,0\n0.001,276.0,1\n")
        assert CsvProvider().load(p).validate() == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(ProviderLoadError):
            CsvProvider().load(Path("nonexistent_file.csv"))

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ProviderLoadError):
            CsvProvider().load(p)

    def test_header_only_raises(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA,VB\n")
        with pytest.raises(ProviderLoadError):
            CsvProvider().load(p)

    def test_no_usable_columns_raises(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,notes\n0.0,hello\n0.001,world\n")
        with pytest.raises(ProviderLoadError):
            CsvProvider().load(p)

    def test_error_is_provider_load_error(self) -> None:
        from app.providers.base.exceptions import ProviderError
        with pytest.raises(ProviderError):
            CsvProvider().load(Path("nonexistent.csv"))

    def test_missing_file_error_message_contains_path(self) -> None:
        path = Path("specific_missing.csv")
        with pytest.raises(ProviderLoadError) as exc_info:
            CsvProvider().load(path)
        assert "specific_missing.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration with ProviderManager
# ---------------------------------------------------------------------------


class TestCsvProviderStubContract:
    """Confirm can_load / provider_name contract is unchanged after full implementation."""

    def test_can_load_csv(self) -> None:
        assert CsvProvider().can_load(Path("data.csv")) is True

    def test_cannot_load_cfg(self) -> None:
        assert CsvProvider().can_load(Path("data.cfg")) is False

    def test_load_returns_disturbance_record(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA\n0.0,275.0\n0.001,276.0\n")
        record = CsvProvider().load(p)
        assert isinstance(record, DisturbanceRecord)

    def test_provider_name(self) -> None:
        assert CsvProvider.provider_name == "csv"


# ---------------------------------------------------------------------------
# Analog channel index ordering
# ---------------------------------------------------------------------------


class TestChannelIndexing:
    def test_analog_indices_sequential(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path, "t.csv", "time,VA,VB,VC\n0.0,1.0,2.0,3.0\n")
        record = CsvProvider().load(p)
        indices = [ch.index for ch in record.analog_channels]
        assert indices == [0, 1, 2]

    def test_digital_indices_sequential(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path, "t.csv",
            "time,trip,pickup\n0.0,0,1\n0.001,1,0\n"
        )
        record = CsvProvider().load(p)
        indices = [ch.index for ch in record.digital_channels]
        assert indices == [0, 1]

    def test_mixed_channels_correct_counts(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path, "t.csv",
            "time,VA,VB,trip\n0.0,275.0,274.0,0\n0.001,276.0,275.0,1\n"
        )
        record = CsvProvider().load(p)
        assert len(record.analog_channels) == 2
        assert len(record.digital_channels) == 1
