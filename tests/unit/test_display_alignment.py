"""Unit tests for app.data.display_alignment."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.data.display_alignment import (
    build_aligned_display_time,
    compute_relative_offsets,
    determine_reference_start,
)
from app.data.multi_source_session import SourceRecord
from app.data.synthetic import make_high_rate_record, make_low_rate_record


def _make_source(
    source_id: str = "src",
    start_time: datetime | float | None = None,
) -> SourceRecord:
    result = make_high_rate_record()
    if start_time is None:
        start_time = result.record.timing_info.start_time
    return SourceRecord(
        source_id=source_id,
        provider_type="comtrade",
        record=result.record,
        signal_metadata=result.signal_metadata,
        original_start_time=start_time,
        sampling_rates=list(result.record.sampling_info.sampling_rates),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestDetermineReferenceStart
# ─────────────────────────────────────────────────────────────────────────────


class TestDetermineReferenceStart:
    def test_empty_list_returns_none(self) -> None:
        assert determine_reference_start([]) is None

    def test_single_source_returns_its_start(self) -> None:
        dt = datetime(2024, 6, 1, 0, 0, 0)
        src = _make_source(start_time=dt)
        result = determine_reference_start([src])
        assert result == dt

    def test_returns_earliest_of_two_datetimes(self) -> None:
        t1 = datetime(2024, 6, 1, 12, 0, 0)
        t2 = datetime(2024, 6, 1, 11, 0, 0)  # earlier
        src1 = _make_source("a", start_time=t1)
        src2 = _make_source("b", start_time=t2)
        assert determine_reference_start([src1, src2]) == t2

    def test_returns_minimum_float(self) -> None:
        src1 = _make_source("a", start_time=5.0)
        src2 = _make_source("b", start_time=2.0)
        result = determine_reference_start([src1, src2])
        assert result == pytest.approx(2.0)

    def test_none_start_time_falls_back_to_timing_info(self) -> None:
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="x",
            provider_type="comtrade",
            record=result.record,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[6400.0],
        )
        ref = determine_reference_start([src])
        # Falls back to timing_info.start_time
        assert ref == result.record.timing_info.start_time

    def test_all_none_with_no_timing_info_fallback_returns_none(self) -> None:
        # Patch the record so timing_info raises — simplest: use two None sources
        result = make_high_rate_record()

        class _BadRecord:
            class timing_info:
                @property
                def start_time(self):
                    raise AttributeError("no timing")

            waveform_data = result.record.waveform_data

        src = SourceRecord(
            source_id="x",
            provider_type="csv",
            record=_BadRecord(),  # type: ignore[arg-type]
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[],
        )
        assert determine_reference_start([src]) is None

    def test_datetime_preferred_over_float(self) -> None:
        dt = datetime(2024, 1, 1)
        src_dt = _make_source("a", start_time=dt)
        src_fl = _make_source("b", start_time=1.0)
        # Mixed types — datetime branch wins (all datetime values collected first)
        result = determine_reference_start([src_dt, src_fl])
        assert isinstance(result, datetime)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeRelativeOffsets
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRelativeOffsets:
    def test_empty_sources_returns_empty_dict(self) -> None:
        assert compute_relative_offsets([], None) == {}

    def test_none_reference_gives_zero_offsets(self) -> None:
        src = _make_source("a")
        offsets = compute_relative_offsets([src], None)
        assert offsets["a"] == pytest.approx(0.0)

    def test_source_with_none_start_time_gives_zero(self) -> None:
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="x",
            provider_type="csv",
            record=result.record,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[],
        )
        ref = datetime(2024, 1, 1)
        offsets = compute_relative_offsets([src], ref)
        assert offsets["x"] == pytest.approx(0.0)

    def test_datetime_offset_computed_correctly(self) -> None:
        ref = datetime(2024, 1, 1, 0, 0, 0)
        t = datetime(2024, 1, 1, 0, 0, 10)  # 10 s later
        src = _make_source("a", start_time=t)
        offsets = compute_relative_offsets([src], ref)
        assert offsets["a"] == pytest.approx(10.0)

    def test_datetime_reference_source_is_at_zero(self) -> None:
        ref = datetime(2024, 6, 1)
        src = _make_source("a", start_time=ref)
        offsets = compute_relative_offsets([src], ref)
        assert offsets["a"] == pytest.approx(0.0)

    def test_float_offset_computed_correctly(self) -> None:
        src = _make_source("a", start_time=3.5)
        offsets = compute_relative_offsets([src], 1.0)
        assert offsets["a"] == pytest.approx(2.5)

    def test_type_mismatch_gives_zero(self) -> None:
        src = _make_source("a", start_time=datetime(2024, 1, 1))
        offsets = compute_relative_offsets([src], 5.0)
        assert offsets["a"] == pytest.approx(0.0)

    def test_multiple_sources_all_present(self) -> None:
        ref = datetime(2024, 1, 1, 0, 0, 0)
        src1 = _make_source("a", start_time=datetime(2024, 1, 1, 0, 0, 0))
        src2 = _make_source("b", start_time=datetime(2024, 1, 1, 0, 0, 5))
        offsets = compute_relative_offsets([src1, src2], ref)
        assert offsets["a"] == pytest.approx(0.0)
        assert offsets["b"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildAlignedDisplayTime
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildAlignedDisplayTime:
    def test_zero_offset_matches_original_time_column(self) -> None:
        result = make_high_rate_record()
        ref = result.record.timing_info.start_time
        src = SourceRecord(
            source_id="a",
            provider_type="comtrade",
            record=result.record,
            signal_metadata={},
            original_start_time=ref,
            sampling_rates=[6400.0],
        )
        aligned = build_aligned_display_time(src, ref)
        original = result.record.waveform_data["time"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(aligned, original)

    def test_offset_applied(self) -> None:
        result = make_high_rate_record()
        ref = datetime(2024, 1, 1, 0, 0, 0)
        src_start = datetime(2024, 1, 1, 0, 0, 2)  # 2 s after ref
        src = SourceRecord(
            source_id="a",
            provider_type="comtrade",
            record=result.record,
            signal_metadata={},
            original_start_time=src_start,
            sampling_rates=[6400.0],
        )
        aligned = build_aligned_display_time(src, ref)
        original = result.record.waveform_data["time"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(aligned, original + 2.0)

    def test_returns_float64(self) -> None:
        src = _make_source()
        result = build_aligned_display_time(src, None)
        assert result.dtype == np.float64

    def test_empty_waveform_returns_empty(self) -> None:
        import pandas as pd
        from app.models import DisturbanceRecord
        from app.models.channels import AnalogChannel
        from app.models.metadata import RecordingMetadata
        from app.models.timing import SamplingInformation, TimingInformation

        empty_df = pd.DataFrame({"time": pd.Series([], dtype=float)})
        rec = DisturbanceRecord(
            metadata=RecordingMetadata(
                station_name="", recorder_name="", source_file="",
                provider_type="", nominal_frequency=50.0,
            ),
            waveform_data=empty_df,
            analog_channels=[],
            digital_channels=[],
            sampling_info=SamplingInformation(
                sampling_rates=[100.0], samples_per_rate=[0]
            ),
            timing_info=TimingInformation(
                start_time=datetime(2024, 1, 1),
                trigger_time=datetime(2024, 1, 1),
            ),
        )
        src = SourceRecord(
            source_id="x",
            provider_type="csv",
            record=rec,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[],
        )
        result = build_aligned_display_time(src, None)
        assert result.shape == (0,)

    def test_none_reference_uses_no_offset(self) -> None:
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="a",
            provider_type="comtrade",
            record=result.record,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[6400.0],
        )
        aligned = build_aligned_display_time(src, None)
        original = result.record.waveform_data["time"].to_numpy(dtype=np.float64)
        np.testing.assert_array_almost_equal(aligned, original)
