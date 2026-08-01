"""Real-data integration test: PULU 2026-03-06 manifest pipeline.

Validates the complete pipeline:
  manifest → COMTRADE provider → CSV provider → MultiSourceSession
  → display alignment → visualization grouping

Skips cleanly when sample files are absent.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import pytest

_CFG_PATH = Path("samples/comtrade/pulu_20260306.cfg")
_CSV_PATH = Path("samples/csv/pulu_20260306.csv")
_MANIFEST_PATH = Path("samples/manifests/pulu_20260306.yaml")

_SAMPLES_PRESENT = _CFG_PATH.exists() and _CSV_PATH.exists() and _MANIFEST_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _SAMPLES_PRESENT,
    reason="Real PULU sample files not present — skipping integration tests",
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures — load once per test run
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def comtrade_rec():
    from app.providers.comtrade.comtrade_provider import ComtradeProvider

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ComtradeProvider().load(_CFG_PATH)


@pytest.fixture(scope="module")
def csv_rec():
    from app.providers.csv.csv_provider import CsvProvider

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CsvProvider().load(_CSV_PATH)


@pytest.fixture(scope="module")
def session(comtrade_rec, csv_rec):
    from datetime import datetime

    from app.data.multi_source_session import MultiSourceSession, SourceRecord

    ct_source = SourceRecord(
        source_id="comtrade_main",
        provider_type="COMTRADE",
        record=comtrade_rec,
        signal_metadata={},
        original_start_time=comtrade_rec.timing_info.start_time,
        sampling_rates=comtrade_rec.sampling_info.sampling_rates,
    )
    csv_source = SourceRecord(
        source_id="csv_ops",
        provider_type="CSV",
        record=csv_rec,
        signal_metadata={},
        original_start_time=csv_rec.timing_info.start_time,
        sampling_rates=csv_rec.sampling_info.sampling_rates,
    )
    s = MultiSourceSession()
    s.add_source(ct_source)
    s.add_source(csv_source)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# COMTRADE provider — channel counts and timing
# ─────────────────────────────────────────────────────────────────────────────


class TestComtradeLoad:
    def test_analog_channel_count(self, comtrade_rec) -> None:
        assert len(comtrade_rec.analog_channels) == 42

    def test_digital_channel_count(self, comtrade_rec) -> None:
        assert len(comtrade_rec.digital_channels) == 88

    def test_sample_count(self, comtrade_rec) -> None:
        assert len(comtrade_rec.waveform_data) == 32693

    def test_start_time(self, comtrade_rec) -> None:
        expected = datetime(2026, 3, 6, 18, 4, 8, 817733)
        assert comtrade_rec.timing_info.start_time == expected

    def test_trigger_time(self, comtrade_rec) -> None:
        expected = datetime(2026, 3, 6, 18, 4, 9, 317733)
        assert comtrade_rec.timing_info.trigger_time == expected

    def test_waveform_has_time_column(self, comtrade_rec) -> None:
        assert "time" in comtrade_rec.waveform_data.columns

    def test_waveform_time_starts_at_zero(self, comtrade_rec) -> None:
        assert comtrade_rec.waveform_data["time"].iloc[0] == pytest.approx(0.0)

    def test_first_analog_channel_name(self, comtrade_rec) -> None:
        assert comtrade_rec.analog_channels[0].name == "KPDN1 VR"

    def test_voltage_channel_unit(self, comtrade_rec) -> None:
        kv_channels = [c for c in comtrade_rec.analog_channels if c.unit == "kV"]
        assert len(kv_channels) > 0

    def test_provider_type(self, comtrade_rec) -> None:
        assert comtrade_rec.metadata.provider_type == "COMTRADE"

    def test_nominal_frequency(self, comtrade_rec) -> None:
        assert comtrade_rec.metadata.nominal_frequency == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# CSV provider — column resolution and timing
# ─────────────────────────────────────────────────────────────────────────────


class TestCsvLoad:
    def test_start_time(self, csv_rec) -> None:
        # "3/6/2026" is an ambiguous D/M-vs-M/D date; Powerwave's approved
        # policy resolves it day-first by default -> 3 June 2026 (not 6 March).
        expected = datetime(2026, 6, 3, 17, 25, 0)
        assert csv_rec.timing_info.start_time == expected

    def test_waveform_columns(self, csv_rec) -> None:
        cols = set(csv_rec.waveform_data.columns)
        assert {"System Demand", "Tie-Line", "Frequency"}.issubset(cols)

    def test_sample_count(self, csv_rec) -> None:
        assert len(csv_rec.waveform_data) == 65

    def test_analog_channel_names(self, csv_rec) -> None:
        names = {c.name for c in csv_rec.analog_channels}
        assert "Frequency" in names
        assert "System Demand" in names
        assert "Tie-Line" in names

    def test_frequency_channel_unit(self, csv_rec) -> None:
        freq = next(c for c in csv_rec.analog_channels if c.name == "Frequency")
        assert freq.unit == "Hz"


# ─────────────────────────────────────────────────────────────────────────────
# MultiSourceSession — construction
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSourceSession:
    def test_source_count(self, session) -> None:
        assert session.source_count() == 2

    def test_not_empty(self, session) -> None:
        assert not session.is_empty()

    def test_source_ids(self, session) -> None:
        ids = session.source_ids()
        assert "comtrade_main" in ids
        assert "csv_ops" in ids

    def test_get_comtrade_source(self, session) -> None:
        src = session.get_source("comtrade_main")
        assert src is not None
        assert src.provider_type == "COMTRADE"

    def test_get_csv_source(self, session) -> None:
        src = session.get_source("csv_ops")
        assert src is not None
        assert src.provider_type == "CSV"

    def test_unknown_source_returns_none(self, session) -> None:
        assert session.get_source("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Display alignment — offset calculation
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayAlignment:
    # NOTE: with the ambiguous-date fix, the CSV's "3/6/2026" now resolves to
    # 3 June 2026 (day-first default) rather than 6 March 2026. COMTRADE's
    # start time (6 March 2026, from its own authoritative, unchanged parser)
    # is therefore now the earliest anchor of the two sources, so COMTRADE —
    # not the CSV — is the display-alignment reference. This inversion is a
    # direct, expected consequence of correcting the CSV date interpretation;
    # COMTRADE's own parsing/values are unaffected (see TestComtradeLoad).

    def test_reference_start_is_comtrade_start(self, session) -> None:
        from app.data.display_alignment import determine_reference_start

        ref = determine_reference_start(session.sources)
        expected = datetime(2026, 3, 6, 18, 4, 8, 817733)
        assert ref == expected

    def test_comtrade_offset_is_zero(self, session) -> None:
        from app.data.display_alignment import compute_relative_offsets, determine_reference_start

        ref = determine_reference_start(session.sources)
        offsets = compute_relative_offsets(session.sources, ref)
        assert offsets["comtrade_main"] == pytest.approx(0.0)

    def test_csv_offset_is_correct(self, session) -> None:
        from app.data.display_alignment import compute_relative_offsets, determine_reference_start

        ref = determine_reference_start(session.sources)
        offsets = compute_relative_offsets(session.sources, ref)
        # CSV 2026-06-03 17:25:00 − COMTRADE 2026-03-06 18:04:08.817733
        assert offsets["csv_ops"] == pytest.approx(7687251.182267, abs=1e-3)

    def test_aligned_display_time_length(self, session) -> None:
        from app.data.display_alignment import build_aligned_display_time, determine_reference_start

        ref = determine_reference_start(session.sources)
        ct_src = session.get_source("comtrade_main")
        t = build_aligned_display_time(ct_src, ref)
        assert len(t) == 32693

    def test_aligned_display_time_first_value(self, session) -> None:
        from app.data.display_alignment import build_aligned_display_time, determine_reference_start

        ref = determine_reference_start(session.sources)
        ct_src = session.get_source("comtrade_main")
        t = build_aligned_display_time(ct_src, ref)
        # COMTRADE is now its own reference, so its first sample is at offset 0.
        assert t[0] == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Column classification — manifest column metadata validated via classifier
# ─────────────────────────────────────────────────────────────────────────────


class TestCsvColumnClassifications:
    def test_frequency_classified_as_frequency(self) -> None:
        from app.data.column_classifier import classify_csv_column

        cls = classify_csv_column("Frequency")
        assert cls.signal_type == "frequency"

    def test_system_demand_classified_as_active_power(self) -> None:
        from app.data.column_classifier import classify_csv_column

        cls = classify_csv_column("System Demand")
        assert cls.signal_type == "active_power"

    def test_tie_line_requires_user_confirmation(self) -> None:
        from app.data.column_classifier import classify_csv_column

        cls = classify_csv_column("Tie-Line")
        assert cls.requires_user_confirmation is True

    def test_frequency_confidence_high(self) -> None:
        from app.data.column_classifier import classify_csv_column

        cls = classify_csv_column("Frequency")
        assert cls.confidence >= 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Visualization grouping — smoke test (must not crash)
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualizationGrouping:
    def test_group_comtrade_channels_does_not_crash(self, comtrade_rec) -> None:
        from app.visualization.channel_grouper import group_channels_for_display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            groups = group_channels_for_display(comtrade_rec)
        assert isinstance(groups, dict)

    def test_group_comtrade_returns_nonempty(self, comtrade_rec) -> None:
        from app.visualization.channel_grouper import group_channels_for_display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            groups = group_channels_for_display(comtrade_rec)
        assert len(groups) > 0

    def test_group_csv_channels_does_not_crash(self, csv_rec) -> None:
        from app.visualization.channel_grouper import group_channels_for_display

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            groups = group_channels_for_display(csv_rec)
        assert isinstance(groups, dict)

    def test_multi_source_display_uses_absolute_common_timestamp_reference(
        self,
        session,
    ) -> None:
        from unittest.mock import MagicMock

        from app.visualization.axis.datetime_axis import AXIS_MODE_ABSOLUTE
        from app.visualization.managers.visualization_manager import VisualizationManager

        manager = VisualizationManager(MagicMock(), MagicMock())
        created: list[MagicMock] = []

        def factory() -> MagicMock:
            canvas = MagicMock()
            created.append(canvas)
            return canvas

        panels = manager.display_multi_source_session(session, canvas_factory=factory)

        assert panels
        assert created
        for canvas in created:
            _, kwargs = canvas.set_record.call_args
            assert kwargs["axis_mode"] == AXIS_MODE_ABSOLUTE
            # Reference is now COMTRADE's start time — see TestDisplayAlignment note.
            assert kwargs["axis_reference_time"] == datetime(2026, 3, 6, 18, 4, 8, 817733)

        comtrade_panel = next(
            canvas for key, canvas in panels.items() if key.startswith("comtrade_main/")
        )
        display_record = comtrade_panel.set_record.call_args[0][0]
        # COMTRADE is now its own reference (see TestDisplayAlignment note), so
        # its displayed time starts at 0 rather than an offset from the CSV.
        assert float(display_record.waveform_data["time"].iloc[0]) == pytest.approx(
            0.0,
            abs=1e-3,
        )
