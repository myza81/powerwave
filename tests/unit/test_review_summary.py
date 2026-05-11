"""Unit tests for app/data/review_summary.py

Tests cover:
  - EventReviewSummary / SourceReviewSummary / ColumnReviewRow / TimestampReviewSummary
  - build_event_review_summary() with synthetic session
  - Manifest data integration: event_id, reference_start, offsets, notes
  - Column filtering logic (COMTRADE suppression, CSV inclusion)
  - Timestamp classification (ambiguous, confirmed, provider-parsed)
  - EventReviewSummary helper methods: has_unconfirmed_columns, unconfirmed_count, etc.

No Qt dependency — purely data-layer tests.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.data.multi_source_session import MultiSourceSession, SourceRecord
from app.data.review_summary import (
    ColumnReviewRow,
    EventReviewSummary,
    SourceReviewSummary,
    build_event_review_summary,
)
from app.data.signal_metadata import SignalMetadata
from app.models import (
    AnalogChannel,
    DisturbanceRecord,
    RecordingMetadata,
    SamplingInformation,
    TimingInformation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_disturbance_record(
    n_analog: int = 2,
    n_digital: int = 1,
    station: str = "TEST",
    source_file: str = "test.cfg",
) -> DisturbanceRecord:
    start = datetime(2026, 3, 6, 18, 0, 0)
    rows = 100
    t = np.linspace(0, 1, rows)
    df = pd.DataFrame({"time": t})

    analog_channels = []
    for i in range(n_analog):
        name = f"VA{i}"
        df[name] = np.sin(2 * np.pi * 50 * t)
        analog_channels.append(AnalogChannel(name=name, unit="kV", index=i))

    from app.models import DigitalChannel
    digital_channels = []
    for i in range(n_digital):
        name = f"DIG{i}"
        df[name] = 0
        digital_channels.append(DigitalChannel(name=name, index=n_analog + i))

    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name=station,
            recorder_name="TEST_IED",
            source_file=source_file,
            provider_type="comtrade",
            nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        sampling_info=SamplingInformation(sampling_rates=[5000.0], samples_per_rate=[rows]),
        timing_info=TimingInformation(start_time=start, trigger_time=start),
    )


def _make_source_record(
    source_id: str,
    provider_type: str = "comtrade",
    signal_metadata: dict[str, SignalMetadata] | None = None,
    n_analog: int = 2,
    n_digital: int = 1,
    start_time: datetime | None = None,
    source_file: str = "test.cfg",
) -> SourceRecord:
    record = _make_disturbance_record(
        n_analog=n_analog, n_digital=n_digital, source_file=source_file
    )
    if signal_metadata is None:
        signal_metadata = {
            ch.name: SignalMetadata(name=ch.name, unit=ch.unit, source=source_id)
            for ch in record.analog_channels
        }
    return SourceRecord(
        source_id=source_id,
        provider_type=provider_type,
        record=record,
        signal_metadata=signal_metadata,
        original_start_time=start_time or record.timing_info.start_time,
        sampling_rates=[5000.0],
    )


_SIMPLE_MANIFEST = {
    "event_id": "test_event_001",
    "sources": [
        {
            "source_id": "comtrade_main",
            "type": "comtrade",
            "start_time": "2026-03-06T18:00:00",
        },
        {
            "source_id": "csv_ops",
            "type": "csv",
            "start_time": "2026-03-06T17:00:00",
            "notes": [
                "Date format ambiguous. Verify M/D/YYYY vs D/M/YYYY.",
                "WARNING: 5 sample(s) produce different dates.",
            ],
            "columns": [
                {
                    "name": "Frequency",
                    "signal_type": "frequency",
                    "unit": "Hz",
                    "display_group": "frequency",
                    "confidence": 0.95,
                    "inferred_from": "name_exact",
                    "requires_user_confirmation": False,
                },
                {
                    "name": "Tie-Line",
                    "signal_type": "active_power",
                    "unit": "MW",
                    "display_group": "power",
                    "confidence": 0.70,
                    "inferred_from": "name_keyword",
                    "requires_user_confirmation": True,
                },
            ],
        },
    ],
    "alignment": {
        "reference_source": "csv_ops",
        "reference_start": "2026-03-06T17:00:00",
        "offsets_seconds": {
            "comtrade_main": 3600.0,
            "csv_ops": 0.0,
        },
    },
}


def _make_csv_signal_metadata() -> dict[str, SignalMetadata]:
    return {
        "Frequency": SignalMetadata(
            name="Frequency",
            unit="Hz",
            source="csv",
            signal_type="frequency",
            display_group="frequency",
            confidence=0.95,
            inferred_from="name_exact",
            requires_user_confirmation=False,
        ),
        "Tie-Line": SignalMetadata(
            name="Tie-Line",
            unit="MW",
            source="csv",
            signal_type="active_power",
            display_group="power",
            confidence=0.70,
            inferred_from="name_keyword",
            requires_user_confirmation=True,
        ),
    }


def _make_two_source_session() -> tuple[MultiSourceSession, dict]:
    comtrade_src = _make_source_record(
        "comtrade_main",
        provider_type="comtrade",
        start_time=datetime(2026, 3, 6, 18, 0, 0),
    )
    csv_src = _make_source_record(
        "csv_ops",
        provider_type="csv",
        signal_metadata=_make_csv_signal_metadata(),
        start_time=datetime(2026, 3, 6, 17, 0, 0),
        source_file="pulu.csv",
    )
    session = MultiSourceSession()
    session.add_source(comtrade_src)
    session.add_source(csv_src)
    return session, _SIMPLE_MANIFEST


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildEventReviewSummary — basic structure
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildEventReviewSummaryBasic:
    def test_returns_event_review_summary(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert isinstance(result, EventReviewSummary)

    def test_event_id_from_manifest(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.event_id == "test_event_001"

    def test_event_id_defaults_to_unknown_when_no_manifest(self) -> None:
        session, _ = _make_two_source_session()
        result = build_event_review_summary(session)
        assert result.event_id == "unknown"

    def test_source_count_matches_session(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert len(result.sources) == 2

    def test_source_ids_preserved(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        ids = [s.source_id for s in result.sources]
        assert "comtrade_main" in ids
        assert "csv_ops" in ids

    def test_source_provider_types(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        assert by_id["comtrade_main"].provider_type == "comtrade"
        assert by_id["csv_ops"].provider_type == "csv"

    def test_reference_start_from_manifest(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.reference_start == datetime(2026, 3, 6, 17, 0, 0)

    def test_reference_start_computed_when_no_manifest(self) -> None:
        session, _ = _make_two_source_session()
        result = build_event_review_summary(session)
        # Should be the earliest start_time across sources
        assert result.reference_start == datetime(2026, 3, 6, 17, 0, 0)

    def test_manifest_notes_empty_by_default(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.manifest_notes == []


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildEventReviewSummary — offsets
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayOffsets:
    def test_manifest_offsets_used_when_present(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        assert by_id["comtrade_main"].display_offset_seconds == pytest.approx(3600.0)
        assert by_id["csv_ops"].display_offset_seconds == pytest.approx(0.0)

    def test_computed_offsets_without_manifest(self) -> None:
        session, _ = _make_two_source_session()
        result = build_event_review_summary(session)
        by_id = {s.source_id: s for s in result.sources}
        # csv_ops starts at 17:00, comtrade_main at 18:00 — offset = 3600s
        assert by_id["comtrade_main"].display_offset_seconds == pytest.approx(3600.0)
        assert by_id["csv_ops"].display_offset_seconds == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildEventReviewSummary — source metadata fields
# ─────────────────────────────────────────────────────────────────────────────


class TestSourceMetadataFields:
    def test_sample_count_populated(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        for src in result.sources:
            assert src.sample_count == 100

    def test_analog_channel_count(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        for src in result.sources:
            assert src.analog_channel_count == 2

    def test_digital_channel_count(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        for src in result.sources:
            assert src.digital_channel_count == 1

    def test_sampling_rates_populated(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        for src in result.sources:
            assert src.sampling_rates == [5000.0]

    def test_file_path_from_record_metadata(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        assert by_id["csv_ops"].file_path == "pulu.csv"

    def test_start_time_from_manifest_overrides_record(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        # Manifest says 2026-03-06T17:00:00 for csv_ops
        assert by_id["csv_ops"].start_time == datetime(2026, 3, 6, 17, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestColumnRows — CSV source classification
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnRows:
    def _csv_src(self, session_result: EventReviewSummary) -> SourceReviewSummary:
        by_id = {s.source_id: s for s in session_result.sources}
        return by_id["csv_ops"]

    def _comtrade_src(self, session_result: EventReviewSummary) -> SourceReviewSummary:
        by_id = {s.source_id: s for s in session_result.sources}
        return by_id["comtrade_main"]

    def test_csv_columns_included(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        csv_src = self._csv_src(result)
        names = [r.column_name for r in csv_src.column_rows]
        assert "Frequency" in names
        assert "Tie-Line" in names

    def test_csv_column_count(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert len(self._csv_src(result).column_rows) == 2

    def test_csv_column_confidence(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        rows = {r.column_name: r for r in self._csv_src(result).column_rows}
        assert rows["Frequency"].confidence == pytest.approx(0.95)
        assert rows["Tie-Line"].confidence == pytest.approx(0.70)

    def test_csv_requires_user_confirmation_propagated(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        rows = {r.column_name: r for r in self._csv_src(result).column_rows}
        assert rows["Frequency"].requires_user_confirmation is False
        assert rows["Tie-Line"].requires_user_confirmation is True

    def test_csv_signal_type_propagated(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        rows = {r.column_name: r for r in self._csv_src(result).column_rows}
        assert rows["Frequency"].signal_type == "frequency"
        assert rows["Tie-Line"].signal_type == "active_power"

    def test_comtrade_auto_inferred_channels_suppressed(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        # COMTRADE channels without explicit confidence are suppressed
        assert len(self._comtrade_src(result).column_rows) == 0

    def test_comtrade_explicit_columns_included_when_flagged(self) -> None:
        flagged_meta = {
            "VR": SignalMetadata(
                name="VR", unit="kV", source="comtrade",
                confidence=0.50, inferred_from="uncertain",
                requires_user_confirmation=True,
            )
        }
        src = _make_source_record(
            "ct", provider_type="comtrade", signal_metadata=flagged_meta, n_analog=1
        )
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        ct = result.sources[0]
        assert len(ct.column_rows) == 1
        assert ct.column_rows[0].column_name == "VR"
        assert ct.column_rows[0].requires_user_confirmation is True


# ─────────────────────────────────────────────────────────────────────────────
# TestTimestampSummary
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestampSummary:
    def test_comtrade_timestamp_is_iso8601(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        ts = by_id["comtrade_main"].timestamp_summary
        assert ts is not None
        assert ts.confirmed_format == "ISO8601"
        assert ts.confidence == pytest.approx(1.0)
        assert ts.warnings == []

    def test_csv_timestamp_ambiguous_when_notes_say_so(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        by_id = {s.source_id: s for s in result.sources}
        ts = by_id["csv_ops"].timestamp_summary
        assert ts is not None
        assert ts.raw_format == "ambiguous"
        assert ts.confidence == pytest.approx(0.5)
        assert len(ts.warnings) == 2

    def test_csv_timestamp_no_warnings_when_no_notes(self) -> None:
        src = _make_source_record("csv_clean", provider_type="csv")
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        ts = result.sources[0].timestamp_summary
        assert ts is not None
        assert ts.warnings == []
        assert ts.confidence == pytest.approx(0.9)

    def test_timestamp_inferred_from_provider_parsed_for_comtrade(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        ct = next(s for s in result.sources if s.source_id == "comtrade_main")
        assert ct.timestamp_summary is not None
        assert ct.timestamp_summary.inferred_from == "provider_parsed"


# ─────────────────────────────────────────────────────────────────────────────
# TestEventReviewSummaryHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestEventReviewSummaryHelpers:
    def test_has_unconfirmed_columns_true_when_tie_line_present(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.has_unconfirmed_columns() is True

    def test_unconfirmed_count_is_one_for_tie_line(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.unconfirmed_count() == 1

    def test_has_unconfirmed_columns_false_when_all_confirmed(self) -> None:
        confirmed_meta = {
            "Frequency": SignalMetadata(
                name="Frequency", unit="Hz", source="csv",
                confidence=0.95, inferred_from="name_exact",
                requires_user_confirmation=False,
            )
        }
        src = _make_source_record("csv_ok", provider_type="csv", signal_metadata=confirmed_meta)
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.has_unconfirmed_columns() is False

    def test_unconfirmed_count_zero_when_all_confirmed(self) -> None:
        confirmed_meta = {
            "Frequency": SignalMetadata(
                name="Frequency", unit="Hz", source="csv",
                confidence=0.95, inferred_from="name_exact",
                requires_user_confirmation=False,
            )
        }
        src = _make_source_record("csv_ok", provider_type="csv", signal_metadata=confirmed_meta)
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.unconfirmed_count() == 0

    def test_all_sources_have_timestamps_true(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.all_sources_have_timestamps() is True

    def test_has_timestamp_warnings_true_when_csv_ambiguous(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        assert result.has_timestamp_warnings() is True

    def test_has_timestamp_warnings_false_when_comtrade_only(self) -> None:
        src = _make_source_record("ct", provider_type="comtrade")
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.has_timestamp_warnings() is False


# ─────────────────────────────────────────────────────────────────────────────
# TestSourceReviewSummaryHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestSourceReviewSummaryHelpers:
    def test_unconfirmed_columns_filters_correctly(self) -> None:
        session, manifest = _make_two_source_session()
        result = build_event_review_summary(session, manifest_data=manifest)
        csv_src = next(s for s in result.sources if s.source_id == "csv_ops")
        unconf = csv_src.unconfirmed_columns()
        assert len(unconf) == 1
        assert unconf[0].column_name == "Tie-Line"

    def test_unconfirmed_columns_empty_when_all_ok(self) -> None:
        confirmed_meta = {
            "Hz": SignalMetadata(
                name="Hz", unit="Hz", source="csv",
                confidence=0.95, inferred_from="name_exact",
                requires_user_confirmation=False,
            )
        }
        src = _make_source_record("csv_clean", provider_type="csv", signal_metadata=confirmed_meta)
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.sources[0].unconfirmed_columns() == []


# ─────────────────────────────────────────────────────────────────────────────
# TestColumnReviewRow — dataclass fields
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnReviewRow:
    def test_basic_construction(self) -> None:
        row = ColumnReviewRow(
            column_name="MW",
            signal_type="active_power",
            unit="MW",
            display_group="power",
            confidence=0.95,
            inferred_from="name_exact",
            requires_user_confirmation=False,
        )
        assert row.column_name == "MW"
        assert row.confidence == pytest.approx(0.95)
        assert not row.requires_user_confirmation

    def test_notes_defaults_to_none(self) -> None:
        row = ColumnReviewRow(
            column_name="X",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=0.0,
            inferred_from="unknown",
            requires_user_confirmation=True,
        )
        assert row.notes is None

    def test_low_confidence_row(self) -> None:
        row = ColumnReviewRow(
            column_name="col_7",
            signal_type=None,
            unit=None,
            display_group="other",
            confidence=0.32,
            inferred_from="unknown",
            requires_user_confirmation=True,
        )
        assert row.confidence < 0.50
        assert row.requires_user_confirmation is True


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildEventReviewSummaryEdgeCases
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildEventReviewSummaryEdgeCases:
    def test_empty_session_produces_empty_sources(self) -> None:
        session = MultiSourceSession()
        result = build_event_review_summary(session)
        assert result.sources == []

    def test_session_without_manifest_data(self) -> None:
        src = _make_source_record("ct", provider_type="comtrade")
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.event_id == "unknown"
        assert len(result.sources) == 1

    def test_manifest_missing_alignment_block(self) -> None:
        data = {"event_id": "no_align", "sources": []}
        session = MultiSourceSession()
        result = build_event_review_summary(session, manifest_data=data)
        assert result.event_id == "no_align"
        assert result.reference_start is None

    def test_manifest_with_global_notes(self) -> None:
        data = {
            "event_id": "ev",
            "notes": ["Global note A", "Global note B"],
            "sources": [],
        }
        session = MultiSourceSession()
        result = build_event_review_summary(session, manifest_data=data)
        assert "Global note A" in result.manifest_notes

    def test_single_comtrade_source_no_column_rows(self) -> None:
        src = _make_source_record("ct", provider_type="comtrade")
        session = MultiSourceSession()
        session.add_source(src)
        result = build_event_review_summary(session)
        assert result.sources[0].column_rows == []
        assert result.has_unconfirmed_columns() is False
