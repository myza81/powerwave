"""Unit tests for app.data.multi_source_session (SourceRecord + MultiSourceSession)."""
from __future__ import annotations

from datetime import datetime

import pytest

from app.data.multi_source_session import MultiSourceSession, SourceRecord
from app.data.signal_metadata import SignalMetadata
from app.data.synthetic import make_high_rate_record, make_low_rate_record


def _make_source(source_id: str = "src_a") -> SourceRecord:
    result = make_high_rate_record()
    return SourceRecord(
        source_id=source_id,
        provider_type="comtrade",
        record=result.record,
        signal_metadata=result.signal_metadata,
        original_start_time=result.record.timing_info.start_time,
        sampling_rates=list(result.record.sampling_info.sampling_rates),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestSourceRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestSourceRecord:
    def test_creation_stores_source_id(self) -> None:
        src = _make_source("comtrade_1")
        assert src.source_id == "comtrade_1"

    def test_creation_stores_provider_type(self) -> None:
        src = _make_source()
        assert src.provider_type == "comtrade"

    def test_signal_metadata_is_dict(self) -> None:
        src = _make_source()
        assert isinstance(src.signal_metadata, dict)

    def test_original_start_time_can_be_none(self) -> None:
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="x",
            provider_type="csv",
            record=result.record,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[100.0],
        )
        assert src.original_start_time is None

    def test_original_start_time_can_be_datetime(self) -> None:
        dt = datetime(2024, 3, 15, 12, 0, 0)
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="x",
            provider_type="csv",
            record=result.record,
            signal_metadata={},
            original_start_time=dt,
            sampling_rates=[100.0],
        )
        assert src.original_start_time == dt

    def test_sampling_rates_stored(self) -> None:
        src = _make_source()
        assert len(src.sampling_rates) > 0
        assert all(isinstance(r, float) for r in src.sampling_rates)

    def test_record_identity_preserved(self) -> None:
        result = make_high_rate_record()
        src = SourceRecord(
            source_id="y",
            provider_type="csv",
            record=result.record,
            signal_metadata={},
            original_start_time=None,
            sampling_rates=[6400.0],
        )
        assert src.record is result.record


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiSourceSession
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSourceSession:
    def test_empty_on_creation(self) -> None:
        session = MultiSourceSession()
        assert session.sources == []

    def test_is_empty_true_when_no_sources(self) -> None:
        assert MultiSourceSession().is_empty()

    def test_is_empty_false_after_add(self) -> None:
        session = MultiSourceSession()
        session.add_source(_make_source())
        assert not session.is_empty()

    def test_add_source_appends(self) -> None:
        session = MultiSourceSession()
        src = _make_source("a")
        session.add_source(src)
        assert len(session.sources) == 1
        assert session.sources[0] is src

    def test_source_count(self) -> None:
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        session.add_source(_make_source("b"))
        assert session.source_count() == 2

    def test_source_ids(self) -> None:
        session = MultiSourceSession()
        session.add_source(_make_source("alpha"))
        session.add_source(_make_source("beta"))
        assert session.source_ids() == ["alpha", "beta"]

    def test_get_source_found(self) -> None:
        session = MultiSourceSession()
        src = _make_source("target")
        session.add_source(src)
        assert session.get_source("target") is src

    def test_get_source_not_found_returns_none(self) -> None:
        session = MultiSourceSession()
        session.add_source(_make_source("a"))
        assert session.get_source("missing") is None

    def test_multiple_sources_independent(self) -> None:
        session = MultiSourceSession()
        r1 = make_high_rate_record()
        r2 = make_low_rate_record()
        src1 = SourceRecord(
            source_id="hi",
            provider_type="comtrade",
            record=r1.record,
            signal_metadata=r1.signal_metadata,
            original_start_time=r1.record.timing_info.start_time,
            sampling_rates=[6400.0],
        )
        src2 = SourceRecord(
            source_id="lo",
            provider_type="csv",
            record=r2.record,
            signal_metadata=r2.signal_metadata,
            original_start_time=r2.record.timing_info.start_time,
            sampling_rates=[100.0],
        )
        session.add_source(src1)
        session.add_source(src2)
        assert session.source_count() == 2
        assert session.get_source("hi").record is r1.record  # type: ignore[union-attr]
        assert session.get_source("lo").record is r2.record  # type: ignore[union-attr]

    def test_sources_list_order_preserved(self) -> None:
        session = MultiSourceSession()
        ids = ["z", "a", "m"]
        for sid in ids:
            session.add_source(_make_source(sid))
        assert session.source_ids() == ids
