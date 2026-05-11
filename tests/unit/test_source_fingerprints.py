"""Unit tests for app.data.intelligence.fingerprints."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.data.intelligence.fingerprints import (
    _column_signature,
    build_fingerprint_from_columns,
    build_fingerprint_from_record,
    fingerprints_match,
)
from app.data.intelligence.models import SourceFingerprint


# ─────────────────────────────────────────────────────────────────────────────
# _column_signature
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnSignature:
    def test_deterministic_same_columns(self) -> None:
        a = _column_signature(["Frequency", "System Demand", "Tie-Line"])
        b = _column_signature(["Frequency", "System Demand", "Tie-Line"])
        assert a == b

    def test_order_independent(self) -> None:
        a = _column_signature(["Frequency", "System Demand", "Tie-Line"])
        b = _column_signature(["Tie-Line", "Frequency", "System Demand"])
        assert a == b

    def test_case_insensitive(self) -> None:
        a = _column_signature(["FREQUENCY"])
        b = _column_signature(["frequency"])
        assert a == b

    def test_different_columns_different_signature(self) -> None:
        a = _column_signature(["Frequency"])
        b = _column_signature(["MW"])
        assert a != b

    def test_signature_is_16_hex_chars(self) -> None:
        sig = _column_signature(["Frequency"])
        assert len(sig) == 16
        assert all(c in "0123456789abcdef" for c in sig)

    def test_whitespace_stripped(self) -> None:
        a = _column_signature(["  Frequency  "])
        b = _column_signature(["Frequency"])
        assert a == b

    def test_empty_names_ignored(self) -> None:
        a = _column_signature(["Frequency", "", "  "])
        b = _column_signature(["Frequency"])
        assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# build_fingerprint_from_columns
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFingerprintFromColumns:
    def test_column_signature_set(self) -> None:
        fp = build_fingerprint_from_columns(["Frequency", "MW"])
        assert fp.column_signature is not None
        assert len(fp.column_signature) == 16

    def test_empty_columns_no_signature(self) -> None:
        fp = build_fingerprint_from_columns([])
        assert fp.column_signature is None

    def test_source_type_preserved(self) -> None:
        fp = build_fingerprint_from_columns(["MW"], source_type="csv")
        assert fp.export_type == "csv"

    def test_station_preserved(self) -> None:
        fp = build_fingerprint_from_columns(["MW"], station="PULU")
        assert fp.station == "PULU"

    def test_source_kind_preserved(self) -> None:
        fp = build_fingerprint_from_columns(["MW"], source_kind="scada_trend")
        assert fp.source_kind == "scada_trend"

    def test_vendor_preserved(self) -> None:
        fp = build_fingerprint_from_columns(["MW"], vendor="ACME")
        assert fp.vendor == "ACME"

    def test_defaults_are_none(self) -> None:
        fp = build_fingerprint_from_columns(["MW"])
        assert fp.vendor is None
        assert fp.station is None
        assert fp.source_kind is None

    def test_same_columns_same_fingerprint(self) -> None:
        fp1 = build_fingerprint_from_columns(["Frequency", "MW"], source_type="csv")
        fp2 = build_fingerprint_from_columns(["MW", "Frequency"], source_type="csv")
        assert fp1 == fp2


# ─────────────────────────────────────────────────────────────────────────────
# build_fingerprint_from_record
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFingerprintFromRecord:
    def _make_record(self, channel_names: list[str], station: str = "TEST"):
        record = MagicMock()
        channels = [MagicMock(name=n) for n in channel_names]
        # MagicMock's .name attribute is special — set it via configure_mock
        for ch, n in zip(channels, channel_names):
            ch.name = n
        record.analog_channels = channels
        record.metadata.station_name = station
        return record

    def test_signature_from_channel_names(self) -> None:
        record = self._make_record(["KPDN1 VR", "KPDN1 IB"])
        fp = build_fingerprint_from_record(record, source_type="comtrade")
        expected = build_fingerprint_from_columns(
            ["KPDN1 VR", "KPDN1 IB"], source_type="comtrade", station="TEST"
        )
        assert fp == expected

    def test_station_name_from_metadata(self) -> None:
        record = self._make_record(["VR"], station="PULU")
        fp = build_fingerprint_from_record(record)
        assert fp.station == "PULU"

    def test_source_type_passed_through(self) -> None:
        record = self._make_record(["VR"])
        fp = build_fingerprint_from_record(record, source_type="comtrade")
        assert fp.export_type == "comtrade"


# ─────────────────────────────────────────────────────────────────────────────
# fingerprints_match
# ─────────────────────────────────────────────────────────────────────────────


class TestFingerprintsMatch:
    def test_identical_fingerprints_match(self) -> None:
        fp = SourceFingerprint(export_type="csv", station="PULU")
        assert fingerprints_match(fp, fp)

    def test_all_none_fingerprint_matches_anything(self) -> None:
        wildcard = SourceFingerprint()
        specific = SourceFingerprint(export_type="csv", station="PULU", column_signature="abc123")
        assert fingerprints_match(wildcard, specific)
        assert fingerprints_match(specific, wildcard)

    def test_matching_non_none_fields(self) -> None:
        a = SourceFingerprint(export_type="csv", station="PULU")
        b = SourceFingerprint(export_type="csv", station="PULU", source_kind="scada_trend")
        assert fingerprints_match(a, b)

    def test_conflicting_field_no_match(self) -> None:
        a = SourceFingerprint(export_type="csv")
        b = SourceFingerprint(export_type="comtrade")
        assert not fingerprints_match(a, b)

    def test_conflicting_station_no_match(self) -> None:
        a = SourceFingerprint(station="PULU")
        b = SourceFingerprint(station="KPDN")
        assert not fingerprints_match(a, b)

    def test_conflicting_signature_no_match(self) -> None:
        a = SourceFingerprint(column_signature="aabbccdd11223344")
        b = SourceFingerprint(column_signature="1122334455667788")
        assert not fingerprints_match(a, b)

    def test_same_signature_matches(self) -> None:
        sig = "aabbccdd11223344"
        a = SourceFingerprint(column_signature=sig)
        b = SourceFingerprint(column_signature=sig)
        assert fingerprints_match(a, b)
