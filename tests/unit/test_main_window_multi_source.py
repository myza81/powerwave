"""Unit tests for Phase D3 additions to PowerwaveMainWindow.

Tests target module-level helpers and handler logic — no QApplication needed.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.data.multi_source_session import MultiSourceSession, SourceRecord
from app.data.synthetic import make_high_rate_record, make_low_rate_record
from app.ui.main_window.main_window import _make_source_record


# ─────────────────────────────────────────────────────────────────────────────
# TestMakeSourceRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeSourceRecord:
    def test_source_id_assigned(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("comtrade_1", result.record, "cfg")
        assert src.source_id == "comtrade_1"

    def test_provider_type_assigned(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record, "xlsx")
        assert src.provider_type == "xlsx"

    def test_default_provider_type(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        assert src.provider_type == "unknown"

    def test_record_identity_preserved(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        assert src.record is result.record

    def test_signal_metadata_covers_all_analog_channels(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        for ch in result.record.analog_channels:
            assert ch.name in src.signal_metadata

    def test_signal_metadata_source_field_is_source_id(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("my_source", result.record)
        for sm in src.signal_metadata.values():
            assert sm.source == "my_source"

    def test_original_start_time_taken_from_timing_info(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        assert src.original_start_time == result.record.timing_info.start_time

    def test_sampling_rates_copied(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        assert src.sampling_rates == list(result.record.sampling_info.sampling_rates)

    def test_sampling_rates_is_independent_copy(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("x", result.record)
        src.sampling_rates.append(999.0)
        assert 999.0 not in result.record.sampling_info.sampling_rates


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiSourceMenuAction
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSourceMenuAction:
    def test_on_multi_source_loaded_method_exists(self) -> None:
        from app.ui.main_window.main_window import PowerwaveMainWindow
        assert callable(getattr(PowerwaveMainWindow, "_on_multi_source_loaded", None))


# ─────────────────────────────────────────────────────────────────────────────
# TestMakeSourceRecordIntegration
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeSourceRecordIntegration:
    def test_make_source_record_for_high_rate_valid(self) -> None:
        result = make_high_rate_record()
        src = _make_source_record("hi_rate", result.record, "cfg")
        assert isinstance(src, SourceRecord)
        assert src.source_id == "hi_rate"
        assert src.sampling_rates == list(result.record.sampling_info.sampling_rates)

    def test_make_source_record_for_low_rate_valid(self) -> None:
        result = make_low_rate_record()
        src = _make_source_record("lo_rate", result.record, "csv")
        assert isinstance(src, SourceRecord)
        assert src.source_id == "lo_rate"

    def test_two_sources_can_form_session(self) -> None:
        hi = make_high_rate_record()
        lo = make_low_rate_record()
        session = MultiSourceSession()
        session.add_source(_make_source_record("hi", hi.record, "cfg"))
        session.add_source(_make_source_record("lo", lo.record, "csv"))
        assert session.source_count() == 2
        assert session.source_ids() == ["hi", "lo"]
