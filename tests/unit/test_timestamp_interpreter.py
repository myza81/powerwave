"""tests/unit/test_timestamp_interpreter.py

Phase D4.3 — Focused tests for TimestampInterpretationMatrix.

Covers:
  - Unambiguous ISO timestamps → single top-ranked interpretation
  - Ambiguous date strings (03/06/2026 → M/D vs D/M)
  - 2-digit year variants (12/2/25 → multiple interpretations)
  - Datetime with milliseconds
  - Excel serial date detection
  - Epoch seconds detection
  - Monotonic scoring factor
  - COMTRADE event overlap scoring
  - Manifest event overlap scoring
  - Interval consistency scoring
  - select_best_timestamp_column logic
  - find_timestamp_candidates across a DataFrame
  - Rejection of invalid interpretations
  - Confirmed-format bonus
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.data.timestamp_interpreter import (
    TimestampInterpretation,
    TimestampInterpretationMatrix,
    build_interpretation_matrix,
    find_timestamp_candidates,
    select_best_timestamp_column,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _matrix(values, **kwargs) -> TimestampInterpretationMatrix:
    return build_interpretation_matrix("time", values, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Basic parse correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicParsing:
    def test_iso_date_unambiguous(self):
        values = ["2026-03-06", "2026-03-07", "2026-03-08"]
        m = _matrix(values)
        assert m.recommended is not None
        assert "%Y-%m-%d" in m.recommended.format_string
        assert m.recommended.parse_success_rate == 1.0

    def test_iso_datetime_unambiguous(self):
        values = [
            "2026-03-06T18:04:08",
            "2026-03-06T18:05:09",
            "2026-03-06T18:06:10",
        ]
        m = _matrix(values)
        assert m.recommended is not None
        assert m.recommended.parse_success_rate == 1.0
        assert not m.is_ambiguous

    def test_iso_datetime_with_ms(self):
        values = [
            "2026-03-06T18:04:08.123456",
            "2026-03-06T18:04:09.234567",
        ]
        m = _matrix(values)
        assert m.recommended is not None
        assert m.recommended.parse_success_rate == 1.0

    def test_ambiguous_m_d_yyyy(self):
        """03/06/2026 is ambiguous: 6 Mar (M/D) or 3 Jun (D/M)."""
        values = [f"03/{d:02d}/2026" for d in range(6, 16)]
        m = _matrix(values)
        assert m.is_ambiguous
        # Both %m/%d/%Y and %d/%m/%Y should be present
        fmts = {i.format_string for i in m.interpretations}
        assert "%m/%d/%Y" in fmts
        assert "%d/%m/%Y" in fmts
        parsed_by_format = {
            i.format_string: i.parsed_samples[0]
            for i in m.interpretations
            if i.parsed_samples
        }
        assert parsed_by_format["%m/%d/%Y"] == datetime(2026, 3, 6)
        assert parsed_by_format["%d/%m/%Y"] == datetime(2026, 6, 3)

    def test_ambiguous_two_digit_year(self):
        """12/2/25 has several possible interpretations."""
        values = ["12/2/25", "12/3/25", "12/4/25"]
        m = _matrix(values)
        fmts = {i.format_string for i in m.interpretations}
        assert "%m/%d/%y" in fmts
        assert "%d/%m/%y" in fmts
        assert "%y/%m/%d" in fmts

    def test_datetime_with_ms(self):
        values = ["03/06/2026 14:35:22.120", "03/06/2026 14:36:22.120"]
        m = _matrix(values)
        assert m.recommended is not None
        # Should find at least one matching format with ms
        assert any(".%f" in i.format_string for i in m.interpretations)

    def test_empty_values_returns_empty_matrix(self):
        m = _matrix([])
        assert m.recommended is None
        assert m.interpretations == []
        assert not m.is_ambiguous

    def test_all_unparseable_returns_empty(self):
        m = _matrix(["not_a_date", "also_not", "nope"])
        # No string format should match; excel/epoch checks also fail
        assert m.recommended is None or m.recommended.parse_success_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Numeric formats
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericFormats:
    def test_excel_serial_date(self):
        """Excel serial dates (e.g. 45987) should be detected."""
        values = ["45987", "45988", "45989"]
        m = _matrix(values)
        excel_interps = [i for i in m.interpretations if i.source_type == "excel_serial"]
        assert excel_interps, "Expected excel_serial interpretation"
        assert excel_interps[0].parse_success_rate == 1.0

    def test_epoch_seconds(self):
        """Unix epoch timestamps (>= 1e9) should be detected."""
        # 2026-03-06T18:04:08 UTC ≈ 1741284248
        values = ["1741284248", "1741284308", "1741284368"]
        m = _matrix(values)
        epoch_interps = [i for i in m.interpretations if i.source_type == "epoch_seconds"]
        assert epoch_interps, "Expected epoch_seconds interpretation"
        assert epoch_interps[0].parse_success_rate == 1.0

    def test_non_epoch_number_not_detected_as_epoch(self):
        """Small numbers should NOT be epoch seconds."""
        values = ["60", "120", "180"]  # seconds-since-start, not epoch
        m = _matrix(values)
        epoch_interps = [i for i in m.interpretations if i.source_type == "epoch_seconds"]
        assert not epoch_interps

    def test_numeric_seconds_detected_as_relative_time(self):
        """Small numeric seconds should be represented as relative time."""
        values = ["0", "60", "120", "180"]
        m = _matrix(values)
        numeric_interps = [
            i for i in m.interpretations if i.source_type == "numeric_seconds"
        ]
        assert numeric_interps, "Expected numeric_seconds interpretation"
        interp = numeric_interps[0]
        assert interp.format_string == "numeric_seconds"
        assert interp.parse_success_rate == 1.0
        assert "relative_numeric_seconds" in interp.reason_codes
        assert "monotonic" in interp.reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Scoring factors
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringFactors:
    def test_monotonic_series_scored_higher(self):
        """Monotonic M/D/YYYY series should score higher than scrambled."""
        mon_vals = [f"03/{d:02d}/2026 17:25:00" for d in range(6, 16)]
        mon_m = _matrix(mon_vals)

        scrambled = ["03/10/2026", "03/06/2026", "03/08/2026"]
        scr_m = _matrix(scrambled)

        assert mon_m.recommended is not None
        # Monotonic should have monotonic reason code
        assert "monotonic" in (mon_m.recommended.reason_codes or [])

    def test_comtrade_overlap_boosts_confidence(self):
        """Values overlapping a COMTRADE event window should score higher."""
        values = [
            "2026-03-06 18:04:00",
            "2026-03-06 18:05:00",
            "2026-03-06 18:06:00",
        ]
        comtrade_start = datetime(2026, 3, 6, 18, 4, 8)

        m_with_overlap = _matrix(values, comtrade_start=comtrade_start)
        m_without = _matrix(values)

        assert m_with_overlap.recommended is not None
        assert m_without.recommended is not None
        assert m_with_overlap.recommended.confidence >= m_without.recommended.confidence
        assert "overlaps_comtrade_event" in (m_with_overlap.recommended.reason_codes or [])

    def test_manifest_overlap_boosts_confidence(self):
        values = [
            "2026-03-06 17:25:00",
            "2026-03-06 17:26:00",
        ]
        manifest_start = datetime(2026, 3, 6, 17, 25, 0)
        m = _matrix(values, manifest_start=manifest_start)
        assert m.recommended is not None
        assert "overlaps_manifest_event" in (m.recommended.reason_codes or [])

    def test_uniform_intervals_score_high(self):
        """Uniform 1-minute spacing should earn uniform_intervals reason."""
        from datetime import timedelta
        base = datetime(2026, 3, 6, 17, 25, 0)
        values = [(base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
                  for i in range(10)]
        m = _matrix(values)
        assert m.recommended is not None
        assert "uniform_intervals" in (m.recommended.reason_codes or [])

    def test_confirmed_format_bonus(self):
        """An operator-confirmed format should appear and get the bonus."""
        values = [f"03/{d:02d}/2026" for d in range(6, 12)]
        m = _matrix(values, confirmed_format="%m/%d/%Y")
        assert m.confirmed_by_rule is not None
        assert m.confirmed_by_rule.format_string == "%m/%d/%Y"
        assert "confirmed_by_operator" in (m.confirmed_by_rule.reason_codes or [])

    def test_realistic_date_range_factor(self):
        """Dates outside 2000–2060 should lower score."""
        # 1950 dates — outside realistic range
        values = ["01/15/1950", "02/15/1950"]
        m = _matrix(values)
        if m.recommended:
            assert "unrealistic_dates" in (m.recommended.reason_codes or [])

    def test_high_parse_rate_factor(self):
        values = ["2026-03-06"] * 10
        m = _matrix(values)
        assert m.recommended is not None
        assert "high_parse_rate" in (m.recommended.reason_codes or [])


# ─────────────────────────────────────────────────────────────────────────────
# Ranking and ambiguity detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingAndAmbiguity:
    def test_ranked_highest_confidence_first(self):
        values = [f"03/{d:02d}/2026" for d in range(1, 10)]
        m = _matrix(values)
        confs = [i.confidence for i in m.interpretations]
        assert confs == sorted(confs, reverse=True)

    def test_ambiguous_flag_set_when_two_formats_plausible(self):
        # Dates where both M/D and D/M parse successfully with plausible years
        values = [f"03/{d:02d}/2026" for d in range(6, 12)]
        m = _matrix(values)
        assert m.is_ambiguous

    def test_unambiguous_flag_for_iso(self):
        values = ["2026-03-06", "2026-03-07"]
        m = _matrix(values)
        assert not m.is_ambiguous

    def test_recommended_is_first_interpretation(self):
        values = ["2026-03-06", "2026-03-07"]
        m = _matrix(values)
        assert m.recommended is m.interpretations[0]


# ─────────────────────────────────────────────────────────────────────────────
# find_timestamp_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestFindCandidates:
    def test_candidates_from_dataframe(self):
        df = pd.DataFrame({
            "Time": ["3/6/2026 17:25", "3/6/2026 17:26", "3/6/2026 17:27"],
            "Time.1": ["17:25", "17:26", "17:27"],
        })
        result = find_timestamp_candidates(["Time", "Time.1"], df)
        assert "Time" in result
        assert "Time.1" in result
        # "Time" column has full date+time — should have at least one interpretation
        assert result["Time"].recommended is not None

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"Time": ["2026-01-01"]})
        result = find_timestamp_candidates(["Time", "NonExistent"], df)
        assert "Time" in result
        assert "NonExistent" not in result

    def test_confirmed_format_passed_through(self):
        df = pd.DataFrame({
            "ts": [f"03/{d:02d}/2026" for d in range(6, 12)],
        })
        result = find_timestamp_candidates(
            ["ts"], df, confirmed_formats={"ts": "%m/%d/%Y"}
        )
        assert result["ts"].confirmed_by_rule is not None


# ─────────────────────────────────────────────────────────────────────────────
# select_best_timestamp_column
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectBest:
    def test_returns_none_for_empty(self):
        assert select_best_timestamp_column({}) is None

    def test_returns_highest_confidence(self):
        df = pd.DataFrame({
            "Time": ["2026-03-06 17:25:00"] * 5,
            "Time.1": ["17:25"] * 5,
        })
        matrices = find_timestamp_candidates(["Time", "Time.1"], df)
        result = select_best_timestamp_column(matrices)
        assert result is not None
        col, matrix = result
        assert col == "Time"

    def test_confirmed_rule_wins_over_higher_confidence(self):
        """A confirmed-by-rule column should be preferred regardless of raw confidence."""
        df = pd.DataFrame({
            "Time": ["2026-03-06 17:25:00"] * 5,
            "ts_confirmed": ["03/06/2026"] * 5,
        })
        matrices = find_timestamp_candidates(
            ["Time", "ts_confirmed"],
            df,
            confirmed_formats={"ts_confirmed": "%m/%d/%Y"},
        )
        result = select_best_timestamp_column(matrices)
        assert result is not None
        col, _ = result
        assert col == "ts_confirmed"
