"""tests/unit/test_persistent_timestamp_rules_d43.py

Phase D4.3 — Persistent timestamp rules: extended fields and column-scoped matching.

Covers:
  - save_timestamp_rule_for_column stores timestamp_column, timezone, column_fingerprint
  - load_timestamp_rules correctly round-trips new fields
  - find_matching_timestamp_rule_for_column: column-scoped priority
  - find_matching_timestamp_rule_for_column: fallback to source-level rule
  - IntelligenceManager.resolve_timestamp_format with column_name
  - IntelligenceManager.save_timestamp_rule_for_column returns TimestampRule
  - Reuse of confirmed rule on future matching sources
  - Multiple column rules coexist under same source_pattern
  - Timezone field preserved
  - column_fingerprint field preserved
"""
from __future__ import annotations

import pytest

from app.data.intelligence import IntelligenceManager
from app.data.intelligence.models import TimestampRule
from app.data.intelligence.timestamp_rules import (
    find_matching_timestamp_rule,
    find_matching_timestamp_rule_for_column,
    load_timestamp_rules,
    save_timestamp_rules,
)


class TestTimestampRuleRoundTrip:
    def test_new_fields_saved_and_loaded(self, tmp_path):
        rule = TimestampRule(
            source_pattern="pulu_csv",
            date_format="%m/%d/%Y %H:%M",
            ambiguous_resolution="%m/%d/%Y %H:%M",
            confirmed_by_operator=True,
            timestamp_column="Time",
            timezone="Asia/Kuala_Lumpur",
            column_fingerprint="abc123def456",
            notes="Confirmed by operator 2026-05-11",
        )
        path = tmp_path / "ts_rules.yaml"
        save_timestamp_rules([rule], path)
        loaded = load_timestamp_rules(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.timestamp_column == "Time"
        assert r.timezone == "Asia/Kuala_Lumpur"
        assert r.column_fingerprint == "abc123def456"
        assert r.confirmed_by_operator is True
        assert r.notes == "Confirmed by operator 2026-05-11"

    def test_none_fields_not_written(self, tmp_path):
        """Optional fields that are None should not appear in the YAML."""
        rule = TimestampRule(
            source_pattern="generic_src",
            date_format="%Y-%m-%d",
            ambiguous_resolution="%Y-%m-%d",
            confirmed_by_operator=False,
        )
        path = tmp_path / "ts_rules.yaml"
        save_timestamp_rules([rule], path)
        text = path.read_text(encoding="utf-8")
        assert "timestamp_column" not in text
        assert "timezone" not in text
        assert "column_fingerprint" not in text
        loaded = load_timestamp_rules(path)
        assert loaded[0].timestamp_column is None
        assert loaded[0].timezone is None

    def test_multiple_rules_round_trip(self, tmp_path):
        rules = [
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%m/%d/%Y %H:%M",
                ambiguous_resolution="%m/%d/%Y %H:%M",
                confirmed_by_operator=True,
                timestamp_column="Time",
            ),
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%H:%M",
                ambiguous_resolution="%H:%M",
                confirmed_by_operator=False,
                timestamp_column="Time.1",
            ),
        ]
        path = tmp_path / "multi.yaml"
        save_timestamp_rules(rules, path)
        loaded = load_timestamp_rules(path)
        assert len(loaded) == 2
        cols = {r.timestamp_column for r in loaded}
        assert "Time" in cols
        assert "Time.1" in cols


class TestFindMatchingTimestampRuleForColumn:
    def _make_rules(self):
        return [
            # Source-level rule (no column)
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%m/%d/%Y",
                ambiguous_resolution="%m/%d/%Y",
                confirmed_by_operator=False,
            ),
            # Column-scoped rule for "Time"
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%m/%d/%Y %H:%M",
                ambiguous_resolution="%m/%d/%Y %H:%M",
                confirmed_by_operator=True,
                timestamp_column="Time",
            ),
            # Column-scoped rule for "Time.1"
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%H:%M",
                ambiguous_resolution="%H:%M",
                confirmed_by_operator=False,
                timestamp_column="Time.1",
            ),
        ]

    def test_column_scoped_rule_takes_priority(self):
        rules = self._make_rules()
        result = find_matching_timestamp_rule_for_column("pulu_csv", "Time", rules)
        assert result is not None
        assert result.date_format == "%m/%d/%Y %H:%M"
        assert result.confirmed_by_operator is True

    def test_source_level_rule_fallback(self):
        """When no column-specific rule exists, fall back to source-level."""
        rules = self._make_rules()
        result = find_matching_timestamp_rule_for_column("pulu_csv", "OtherCol", rules)
        assert result is not None
        assert result.date_format == "%m/%d/%Y"
        assert result.timestamp_column is None

    def test_no_match_returns_none(self):
        rules = self._make_rules()
        result = find_matching_timestamp_rule_for_column("unknown_src", "Time", rules)
        assert result is None

    def test_case_insensitive_source_pattern(self):
        rules = self._make_rules()
        result = find_matching_timestamp_rule_for_column("PULU_CSV", "Time", rules)
        assert result is not None

    def test_case_insensitive_column_name(self):
        rules = self._make_rules()
        result = find_matching_timestamp_rule_for_column("pulu_csv", "TIME", rules)
        assert result is not None
        assert result.date_format == "%m/%d/%Y %H:%M"


class TestIntelligenceManagerTimestampRules:
    def test_resolve_with_column_name(self, tmp_path):
        mgr = IntelligenceManager(timestamp_rules_path=tmp_path / "ts.yaml")
        mgr.save_timestamp_rule_for_column(
            source_pattern="pulu_csv",
            column_name="Time",
            date_format="%m/%d/%Y %H:%M",
            timezone="Asia/Kuala_Lumpur",
            confirmed_by_operator=True,
            path=tmp_path / "ts.yaml",
        )
        rule = mgr.resolve_timestamp_format("pulu_csv", column_name="Time")
        assert rule is not None
        assert rule.date_format == "%m/%d/%Y %H:%M"
        assert rule.timezone == "Asia/Kuala_Lumpur"

    def test_resolve_without_column_fallback(self, tmp_path):
        mgr = IntelligenceManager(timestamp_rules_path=tmp_path / "ts.yaml")
        # Save source-level rule only
        mgr.save_timestamp_rule(
            TimestampRule(
                source_pattern="pulu_csv",
                date_format="%m/%d/%Y",
                ambiguous_resolution="%m/%d/%Y",
                confirmed_by_operator=True,
            ),
            path=tmp_path / "ts.yaml",
        )
        rule = mgr.resolve_timestamp_format("pulu_csv")
        assert rule is not None
        assert rule.date_format == "%m/%d/%Y"

    def test_save_timestamp_rule_for_column_returns_rule(self, tmp_path):
        mgr = IntelligenceManager(timestamp_rules_path=tmp_path / "ts.yaml")
        rule = mgr.save_timestamp_rule_for_column(
            source_pattern="src1",
            column_name="ts_col",
            date_format="%d/%m/%Y",
            path=tmp_path / "ts.yaml",
        )
        assert isinstance(rule, TimestampRule)
        assert rule.timestamp_column == "ts_col"

    def test_multiple_column_rules_coexist(self, tmp_path):
        """Two column-scoped rules under same source_pattern both persist."""
        path = tmp_path / "ts.yaml"
        mgr = IntelligenceManager(timestamp_rules_path=path)
        mgr.save_timestamp_rule_for_column(
            "pulu", "Time", "%m/%d/%Y %H:%M", path=path
        )
        mgr.save_timestamp_rule_for_column(
            "pulu", "Time.1", "%H:%M", path=path
        )
        loaded = load_timestamp_rules(path)
        assert len(loaded) == 2
        formats = {r.date_format for r in loaded}
        assert "%m/%d/%Y %H:%M" in formats
        assert "%H:%M" in formats

    def test_rule_reused_for_future_source(self, tmp_path):
        """Confirmed rule is found and applied for a new IntelligenceManager instance."""
        path = tmp_path / "ts.yaml"
        mgr1 = IntelligenceManager(timestamp_rules_path=path)
        mgr1.save_timestamp_rule_for_column(
            "pulu_csv", "Time", "%m/%d/%Y %H:%M",
            confirmed_by_operator=True, path=path,
        )
        # New manager instance (simulating a new session loading persisted rules)
        mgr2 = IntelligenceManager(timestamp_rules_path=path)
        rule = mgr2.resolve_timestamp_format("pulu_csv", column_name="Time")
        assert rule is not None
        assert rule.confirmed_by_operator is True
        assert rule.date_format == "%m/%d/%Y %H:%M"

    def test_timezone_stored(self, tmp_path):
        path = tmp_path / "ts.yaml"
        mgr = IntelligenceManager(timestamp_rules_path=path)
        mgr.save_timestamp_rule_for_column(
            "pulu_csv", "Time", "%m/%d/%Y %H:%M",
            timezone="Asia/Kuala_Lumpur", path=path,
        )
        rule = mgr.resolve_timestamp_format("pulu_csv", column_name="Time")
        assert rule.timezone == "Asia/Kuala_Lumpur"
