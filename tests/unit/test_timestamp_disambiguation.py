"""Tests for the shared CSV/Excel ambiguous-date-order policy.

app.data.timestamp_disambiguation implements Powerwave's approved policy:
ambiguous D/M-vs-M/D dates default to day-first; unambiguous values are
preserved regardless; an explicit user-selected order always overrides the
default.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.timestamp_disambiguation import (
    AmbiguousDateDiagnostic,
    DateOrderResult,
    detect_date_order_ambiguity,
    parse_ambiguous_dates,
    resolve_component_order,
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_component_order — automatic default
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveComponentOrderAutomatic:
    def test_ambiguous_pair_defaults_day_first(self):
        result = resolve_component_order(3, 6)
        assert result == DateOrderResult(True, "day_first", "automatic")

    def test_ambiguous_pair_defaults_day_first_reversed_values(self):
        result = resolve_component_order(12, 11)
        assert result == DateOrderResult(True, "day_first", "automatic")

    def test_first_over_twelve_is_unambiguous_day_first(self):
        # 13 cannot be a month -> first is forced to be the day.
        result = resolve_component_order(13, 6)
        assert result == DateOrderResult(False, "day_first", "automatic")

    def test_second_over_twelve_is_unambiguous_month_first(self):
        # 13 cannot be a month -> second is forced to be the day, so the
        # first component must be read as the month.
        result = resolve_component_order(6, 13)
        assert result == DateOrderResult(False, "month_first", "automatic")

    def test_equal_components_not_flagged_ambiguous(self):
        # 12/12 -> both readings agree; not practically ambiguous.
        result = resolve_component_order(12, 12)
        assert result.is_ambiguous is False
        assert result.applied_order == "day_first"


# ─────────────────────────────────────────────────────────────────────────────
# resolve_component_order — explicit user override
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveComponentOrderUserOverride:
    def test_user_dayfirst_true_wins_for_ambiguous_input(self):
        result = resolve_component_order(3, 6, user_dayfirst=True)
        assert result.applied_order == "day_first"
        assert result.source == "user_override"

    def test_user_dayfirst_false_overrides_default_for_ambiguous_input(self):
        result = resolve_component_order(3, 6, user_dayfirst=False)
        assert result.applied_order == "month_first"
        assert result.source == "user_override"

    def test_user_override_is_recorded_even_for_unambiguous_input(self):
        # User choice is authoritative and never second-guessed, even where
        # automatic detection would have forced a specific order anyway.
        result = resolve_component_order(13, 6, user_dayfirst=False)
        assert result.applied_order == "month_first"
        assert result.source == "user_override"


# ─────────────────────────────────────────────────────────────────────────────
# detect_date_order_ambiguity
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectDateOrderAmbiguity:
    def test_ambiguous_values_flagged(self):
        diag = detect_date_order_ambiguity(["3/6/2026 17:25", "3/6/2026 17:26"])
        assert diag.is_ambiguous is True
        assert diag.applied_order == "day_first"
        assert diag.source == "automatic"
        assert diag.sample_value == "3/6/2026 17:25"

    def test_unambiguous_values_not_flagged(self):
        diag = detect_date_order_ambiguity(["13/6/2026 17:25", "13/6/2026 17:26"])
        assert diag.is_ambiguous is False

    def test_iso_dates_not_flagged(self):
        diag = detect_date_order_ambiguity(["2026-01-15 13:30:00", "2026-01-16 13:30:00"])
        assert diag.is_ambiguous is False
        assert diag.applied_order == "not_applicable"

    def test_no_matching_values_returns_not_applicable(self):
        diag = detect_date_order_ambiguity(["not a date", ""])
        assert diag == AmbiguousDateDiagnostic(False, "not_applicable", "automatic")

    def test_user_override_source_recorded(self):
        diag = detect_date_order_ambiguity(["3/6/2026"], user_dayfirst=False)
        assert diag.source == "user_override"
        assert diag.applied_order == "month_first"


# ─────────────────────────────────────────────────────────────────────────────
# parse_ambiguous_dates — Series-level parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParseAmbiguousDates:
    def test_automatic_default_resolves_ambiguous_date_day_first(self):
        s = pd.Series(["3/6/2026 17:25"])
        parsed, diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-03 17:25:00")
        assert diag.is_ambiguous is True

    def test_automatic_default_month_component_over_twelve(self):
        s = pd.Series(["12/11/2026"])
        parsed, _diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-11-12")

    def test_unambiguous_day_over_twelve_preserved(self):
        s = pd.Series(["13/6/2026"])
        parsed, diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-13")
        assert diag.is_ambiguous is False

    def test_unambiguous_month_first_input_preserved(self):
        s = pd.Series(["6/13/2026"])
        parsed, diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-13")
        assert diag.is_ambiguous is False

    def test_noon_am_pm_unaffected(self):
        s = pd.Series(["3/6/2026 12:00 PM"])
        parsed, _diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-03 12:00:00")

    def test_midnight_am_pm_unaffected(self):
        s = pd.Series(["3/6/2026 12:00 AM"])
        parsed, _diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-03 00:00:00")

    def test_24_hour_time_unaffected(self):
        s = pd.Series(["3/6/2026 23:59:00"])
        parsed, _diag = parse_ambiguous_dates(s)
        assert parsed.iloc[0] == pd.Timestamp("2026-06-03 23:59:00")

    def test_explicit_user_month_first_override(self):
        s = pd.Series(["3/6/2026 17:25"])
        parsed, diag = parse_ambiguous_dates(s, user_dayfirst=False)
        assert parsed.iloc[0] == pd.Timestamp("2026-03-06 17:25:00")
        assert diag.source == "user_override"

    def test_unparseable_values_become_nat(self):
        s = pd.Series(["not a date"])
        parsed, _diag = parse_ambiguous_dates(s)
        assert pd.isna(parsed.iloc[0])
