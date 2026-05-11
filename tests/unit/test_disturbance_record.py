"""Unit tests for DisturbanceRecord and its supporting models.

Covers: construction, channel accessors, sample metrics, duration,
and the validate() consistency checker.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.models.channels import AnalogChannel, DigitalChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import DisturbanceInformation, SamplingInformation, TimingInformation


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------


def _make_record(
    analog_channels: list[AnalogChannel] | None = None,
    digital_channels: list[DigitalChannel] | None = None,
    df: pd.DataFrame | None = None,
    empty_df: bool = False,
) -> DisturbanceRecord:
    """Build a minimal valid DisturbanceRecord for testing."""
    if analog_channels is None:
        analog_channels = [
            AnalogChannel(name="VA", unit="kV", index=0, phase="A"),
            AnalogChannel(name="IA", unit="A", index=1, phase="A"),
        ]
    if digital_channels is None:
        digital_channels = [DigitalChannel(name="TRIP", index=0)]

    if df is None:
        if empty_df:
            df = pd.DataFrame(columns=["time", "VA", "IA", "TRIP"])
        else:
            df = pd.DataFrame(
                {
                    "time": [0.0, 0.01, 0.02],
                    "VA": [100.0, 101.0, 99.0],
                    "IA": [5.0, 5.1, 4.9],
                    "TRIP": [0, 0, 1],
                }
            )

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="TEST_STATION",
            recorder_name="TEST_IED",
            source_file="test.cfg",
            provider_type="COMTRADE",
            nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=SamplingInformation(
            sampling_rates=[100.0],
            samples_per_rate=[3],
        ),
        timing_info=TimingInformation(
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            trigger_time=datetime(2024, 1, 1, 12, 0, 1),
        ),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDisturbanceRecordConstruction:
    def test_creates_valid_record(self) -> None:
        record = _make_record()
        assert record is not None
        assert record.metadata.station_name == "TEST_STATION"
        assert record.metadata.nominal_frequency == 50.0
        assert record.metadata.provider_type == "COMTRADE"

    def test_waveform_data_stored_by_reference(self) -> None:
        df = pd.DataFrame({"time": [0.0], "VA": [1.0], "IA": [0.1], "TRIP": [0]})
        record = _make_record(df=df)
        assert record.waveform_data is df

    def test_with_optional_disturbance_info(self) -> None:
        record = _make_record()
        assert record.disturbance_info is None

        record.disturbance_info = DisturbanceInformation(
            event_type="FAULT", notes="Phase A to ground", tags=["protection", "line"]
        )
        assert record.disturbance_info.event_type == "FAULT"
        assert "protection" in record.disturbance_info.tags

    def test_metadata_optional_fields_default_none(self) -> None:
        record = _make_record()
        assert record.metadata.device_id is None
        assert record.metadata.location is None
        assert record.metadata.timezone is None
        assert record.metadata.comments is None


# ---------------------------------------------------------------------------
# Channel accessors
# ---------------------------------------------------------------------------


class TestChannelAccessors:
    def test_analog_channel_names(self) -> None:
        record = _make_record()
        assert record.analog_channel_names() == ["VA", "IA"]

    def test_digital_channel_names(self) -> None:
        record = _make_record()
        assert record.digital_channel_names() == ["TRIP"]

    def test_channel_names_combined_order(self) -> None:
        record = _make_record()
        assert record.channel_names() == ["VA", "IA", "TRIP"]

    def test_has_channel_analog(self) -> None:
        record = _make_record()
        assert record.has_channel("VA") is True
        assert record.has_channel("IA") is True

    def test_has_channel_digital(self) -> None:
        record = _make_record()
        assert record.has_channel("TRIP") is True

    def test_has_channel_missing(self) -> None:
        record = _make_record()
        assert record.has_channel("VB") is False
        assert record.has_channel("CBOPEN") is False
        assert record.has_channel("") is False

    def test_channel_names_preserves_declaration_order(self) -> None:
        analog = [
            AnalogChannel(name="VC", unit="kV", index=0),
            AnalogChannel(name="VA", unit="kV", index=1),
            AnalogChannel(name="VB", unit="kV", index=2),
        ]
        digital = [
            DigitalChannel(name="CB2", index=0),
            DigitalChannel(name="CB1", index=1),
        ]
        df = pd.DataFrame({"time": [0.0], "VC": [1.0], "VA": [1.0], "VB": [1.0], "CB2": [0], "CB1": [0]})
        record = _make_record(analog_channels=analog, digital_channels=digital, df=df)
        assert record.analog_channel_names() == ["VC", "VA", "VB"]
        assert record.digital_channel_names() == ["CB2", "CB1"]


# ---------------------------------------------------------------------------
# Sample count and duration
# ---------------------------------------------------------------------------


class TestSampleMetrics:
    def test_sample_count(self) -> None:
        assert _make_record().sample_count() == 3

    def test_sample_count_empty(self) -> None:
        assert _make_record(empty_df=True).sample_count() == 0

    def test_duration_from_time_column(self) -> None:
        record = _make_record()
        assert record.duration_seconds() == pytest.approx(0.02)

    def test_duration_from_sampling_info_fallback(self) -> None:
        # DataFrame with no 'time' column — must fall back to sampling_info
        df = pd.DataFrame(
            {
                "VA": [100.0, 101.0, 99.0],
                "IA": [5.0, 5.1, 4.9],
                "TRIP": [0, 0, 1],
            }
        )
        record = _make_record(df=df)
        # 3 samples at 100 Hz → 3 / 100 = 0.03 s
        assert record.duration_seconds() == pytest.approx(0.03)

    def test_duration_multirate_sampling_info_fallback(self) -> None:
        df = pd.DataFrame({"VA": list(range(10)), "IA": list(range(10)), "TRIP": [0] * 10})
        record = _make_record(df=df)
        record.sampling_info.sampling_rates = [1200.0, 600.0]
        record.sampling_info.samples_per_rate = [6, 4]
        # 6/1200 + 4/600 = 0.005 + 0.00667 = 0.01167 s
        assert record.duration_seconds() == pytest.approx(6 / 1200 + 4 / 600)

    def test_duration_empty_dataframe(self) -> None:
        assert _make_record(empty_df=True).duration_seconds() == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_passes_for_valid_record(self) -> None:
        errors = _make_record().validate()
        assert errors == []

    def test_validate_fails_for_empty_dataframe(self) -> None:
        errors = _make_record(empty_df=True).validate()
        assert any("empty" in e for e in errors)

    def test_validate_fails_for_missing_analog_channel(self) -> None:
        analog = [AnalogChannel(name="VB", unit="kV", index=0)]  # VB not in default df
        errors = _make_record(analog_channels=analog).validate()
        assert any("VB" in e for e in errors)

    def test_validate_fails_for_missing_digital_channel(self) -> None:
        digital = [DigitalChannel(name="CBOPEN", index=0)]  # CBOPEN not in default df
        errors = _make_record(digital_channels=digital).validate()
        assert any("CBOPEN" in e for e in errors)

    def test_validate_fails_for_sampling_rate_length_mismatch(self) -> None:
        record = _make_record()
        record.sampling_info.sampling_rates.append(600.0)  # now len 2 vs len 1
        errors = record.validate()
        assert any("equal length" in e for e in errors)

    def test_validate_fails_for_empty_sampling_rates(self) -> None:
        record = _make_record()
        record.sampling_info.sampling_rates.clear()
        record.sampling_info.samples_per_rate.clear()
        errors = record.validate()
        assert any("must not be empty" in e for e in errors)

    def test_validate_fails_for_trigger_before_start(self) -> None:
        record = _make_record()
        record.timing_info.trigger_time = datetime(2023, 12, 31, 0, 0, 0)
        errors = record.validate()
        assert any("trigger_time" in e for e in errors)

    def test_validate_returns_multiple_errors(self) -> None:
        # Empty df + mismatched sampling → at least two errors
        record = _make_record(empty_df=True)
        record.sampling_info.sampling_rates.append(600.0)
        errors = record.validate()
        assert len(errors) >= 2

    def test_validate_non_dataframe_stops_early(self) -> None:
        record = _make_record()
        object.__setattr__(record, "waveform_data", "not-a-dataframe")  # bypass slots
        errors = record.validate()
        assert any("DataFrame" in e for e in errors)
        # Should return immediately without crashing on further checks
        assert len(errors) == 1
