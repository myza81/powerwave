"""Unit tests for app.data.intelligence.timestamp_rules."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.intelligence.models import TimestampRule
from app.data.intelligence.timestamp_rules import (
    find_matching_timestamp_rule,
    load_timestamp_rules,
    save_timestamp_rules,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadTimestampRules
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadTimestampRules:
    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        rules = load_timestamp_rules(tmp_path / "nonexistent.yaml")
        assert rules == []

    def test_empty_rules_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text("rules: []", encoding="utf-8")
        assert load_timestamp_rules(f) == []

    def test_loads_single_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text(
            "rules:\n"
            "  - source_pattern: pulu_scada_export\n"
            "    date_format: 'M/D/YYYY'\n"
            "    ambiguous_resolution: 'M/D/YYYY'\n"
            "    confirmed_by_operator: true\n"
            "    notes: 'Confirmed manually'\n",
            encoding="utf-8",
        )
        rules = load_timestamp_rules(f)
        assert len(rules) == 1
        r = rules[0]
        assert r.source_pattern == "pulu_scada_export"
        assert r.date_format == "M/D/YYYY"
        assert r.ambiguous_resolution == "M/D/YYYY"
        assert r.confirmed_by_operator is True
        assert r.notes == "Confirmed manually"

    def test_ambiguous_resolution_defaults_to_date_format(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text(
            "rules:\n"
            "  - source_pattern: src\n"
            "    date_format: 'D/M/YYYY'\n",
            encoding="utf-8",
        )
        rules = load_timestamp_rules(f)
        assert rules[0].ambiguous_resolution == "D/M/YYYY"

    def test_confirmed_by_operator_defaults_to_false(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text(
            "rules:\n  - source_pattern: src\n    date_format: 'M/D/YYYY'\n",
            encoding="utf-8",
        )
        rules = load_timestamp_rules(f)
        assert rules[0].confirmed_by_operator is False

    def test_missing_source_pattern_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text("rules:\n  - date_format: 'M/D/YYYY'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="source_pattern"):
            load_timestamp_rules(f)

    def test_missing_date_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text("rules:\n  - source_pattern: src\n", encoding="utf-8")
        with pytest.raises(ValueError, match="date_format"):
            load_timestamp_rules(f)

    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "ts.yaml"
        f.write_text("rules: [not: valid: yaml:\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_timestamp_rules(f)


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveTimestampRules
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveTimestampRules:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "ts.yaml"
        save_timestamp_rules([], p)
        assert p.exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        rule = TimestampRule(
            source_pattern="pulu_scada_export",
            date_format="M/D/YYYY",
            ambiguous_resolution="M/D/YYYY",
            confirmed_by_operator=True,
            notes="Confirmed",
        )
        p = tmp_path / "ts.yaml"
        save_timestamp_rules([rule], p)
        reloaded = load_timestamp_rules(p)
        assert len(reloaded) == 1
        r = reloaded[0]
        assert r.source_pattern == "pulu_scada_export"
        assert r.date_format == "M/D/YYYY"
        assert r.confirmed_by_operator is True
        assert r.notes == "Confirmed"

    def test_round_trip_without_notes(self, tmp_path: Path) -> None:
        rule = TimestampRule(
            source_pattern="src",
            date_format="D/M/YYYY",
            ambiguous_resolution="D/M/YYYY",
            confirmed_by_operator=False,
        )
        p = tmp_path / "ts.yaml"
        save_timestamp_rules([rule], p)
        reloaded = load_timestamp_rules(p)
        assert reloaded[0].notes is None


# ─────────────────────────────────────────────────────────────────────────────
# TestFindMatchingTimestampRule
# ─────────────────────────────────────────────────────────────────────────────


class TestFindMatchingTimestampRule:
    def test_exact_match(self) -> None:
        rule = TimestampRule(
            source_pattern="pulu_scada_export",
            date_format="M/D/YYYY",
            ambiguous_resolution="M/D/YYYY",
            confirmed_by_operator=True,
        )
        result = find_matching_timestamp_rule("pulu_scada_export", [rule])
        assert result is rule

    def test_case_insensitive_match(self) -> None:
        rule = TimestampRule(
            source_pattern="pulu_scada_export",
            date_format="M/D/YYYY",
            ambiguous_resolution="M/D/YYYY",
            confirmed_by_operator=True,
        )
        result = find_matching_timestamp_rule("PULU_SCADA_EXPORT", [rule])
        assert result is rule

    def test_no_match_returns_none(self) -> None:
        rule = TimestampRule(
            source_pattern="pulu_scada_export",
            date_format="M/D/YYYY",
            ambiguous_resolution="M/D/YYYY",
            confirmed_by_operator=True,
        )
        assert find_matching_timestamp_rule("other_source", [rule]) is None

    def test_first_match_wins(self) -> None:
        r1 = TimestampRule(
            source_pattern="src",
            date_format="M/D/YYYY",
            ambiguous_resolution="M/D/YYYY",
            confirmed_by_operator=True,
        )
        r2 = TimestampRule(
            source_pattern="src",
            date_format="D/M/YYYY",
            ambiguous_resolution="D/M/YYYY",
            confirmed_by_operator=False,
        )
        result = find_matching_timestamp_rule("src", [r1, r2])
        assert result is r1

    def test_empty_rules_returns_none(self) -> None:
        assert find_matching_timestamp_rule("anything", []) is None
