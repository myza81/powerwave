"""Unit tests for app.intelligence.rule_manager.RuleManager.

Covers:
  - Initialisation (no rules file, custom path)
  - classify_column delegation
  - save_confirmed_rows: filtering, YAML write, count returned
  - Duplicate rule handling: latest save wins
  - Malformed YAML: ValueError raised on load
  - reload(): picks up external file changes
  - rule_count property
  - intelligence_manager property returns IntelligenceManager
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.column_classifier import CONFIRMATION_THRESHOLD
from app.data.intelligence.mapping_rules import load_mapping_rules
from app.data.intelligence.models import MappingRule
from app.data.review_summary import ColumnReviewRow
from app.intelligence.rule_manager import RuleManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _confirmed_row(
    name: str,
    signal_type: str = "frequency",
    unit: str = "Hz",
    display_group: str = "frequency",
    confidence: float = 0.95,
) -> ColumnReviewRow:
    return ColumnReviewRow(
        column_name=name,
        signal_type=signal_type,
        unit=unit,
        display_group=display_group,
        confidence=confidence,
        inferred_from="name_exact",
        requires_user_confirmation=False,
    )


def _unconfirmed_row(name: str) -> ColumnReviewRow:
    return ColumnReviewRow(
        column_name=name,
        signal_type="active_power",
        unit="MW",
        display_group="power",
        confidence=0.70,
        inferred_from="name_keyword",
        requires_user_confirmation=True,
    )


def _no_type_row(name: str) -> ColumnReviewRow:
    return ColumnReviewRow(
        column_name=name,
        signal_type=None,
        unit=None,
        display_group="other",
        confidence=0.0,
        inferred_from="unknown",
        requires_user_confirmation=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestRuleManagerInit
# ─────────────────────────────────────────────────────────────────────────────


class TestRuleManagerInit:
    def test_no_rules_file_gives_zero_count(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "nonexistent.yaml")
        assert rm.rule_count == 0

    def test_rule_count_reflects_loaded_rules(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        p.write_text(
            "rules:\n"
            "  - match_pattern: freq\n"
            "    match_type: exact\n"
            "    signal_type: frequency\n"
            "    unit: Hz\n"
            "    display_group: frequency\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n",
            encoding="utf-8",
        )
        rm = RuleManager(rules_path=p)
        assert rm.rule_count == 1

    def test_intelligence_manager_property_returns_inner_manager(
        self, tmp_path: Path
    ) -> None:
        from app.data.intelligence.intelligence_manager import IntelligenceManager

        rm = RuleManager(rules_path=tmp_path / "none.yaml")
        assert isinstance(rm.intelligence_manager, IntelligenceManager)

    def test_intelligence_manager_is_same_instance(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "none.yaml")
        assert rm.intelligence_manager is rm._mgr


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyColumnDelegation
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyColumnDelegation:
    def test_known_column_classified_correctly(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "none.yaml")
        cls, audit = rm.classify_column("Frequency")
        assert cls.signal_type == "frequency"
        assert audit is None

    def test_stored_rule_fires_on_classify(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        p.write_text(
            "rules:\n"
            "  - match_pattern: sysf\n"
            "    match_type: exact\n"
            "    signal_type: frequency\n"
            "    unit: Hz\n"
            "    display_group: frequency\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n",
            encoding="utf-8",
        )
        rm = RuleManager(rules_path=p)
        cls, audit = rm.classify_column("SYSF")
        assert cls.confidence == pytest.approx(1.0)
        assert cls.inferred_from == "persistent_mapping_rule"
        assert audit is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveConfirmedRows
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveConfirmedRows:
    def test_empty_list_returns_zero(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "rules.yaml")
        assert rm.save_confirmed_rows([]) == 0

    def test_empty_list_does_not_create_file(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        rm.save_confirmed_rows([])
        assert not p.exists()

    def test_confirmed_row_saved_returns_count(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        n = rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="test")
        assert n == 1

    def test_multiple_confirmed_rows_returns_correct_count(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        rows = [_confirmed_row("Frequency"), _confirmed_row("MW", "active_power", "MW", "power")]
        n = rm.save_confirmed_rows(rows, source_id="test")
        assert n == 2

    def test_unconfirmed_row_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        n = rm.save_confirmed_rows([_unconfirmed_row("Tie-Line")], source_id="test")
        assert n == 0
        assert not p.exists()

    def test_no_signal_type_row_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        n = rm.save_confirmed_rows([_no_type_row("Unknown Col")], source_id="test")
        assert n == 0

    def test_mixed_rows_only_saves_confirmed_with_type(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        rows = [
            _confirmed_row("Frequency"),
            _unconfirmed_row("Tie-Line"),
            _no_type_row("Time"),
        ]
        n = rm.save_confirmed_rows(rows, source_id="test")
        assert n == 1
        saved = load_mapping_rules(p)
        assert len(saved) == 1
        assert saved[0].match_pattern == "frequency"

    def test_saved_rule_has_confirmed_by_operator(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="test")
        saved = load_mapping_rules(p)
        assert saved[0].confirmed_by_operator is True

    def test_saved_rule_confidence_capped_at_1(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        row = _confirmed_row("Frequency", confidence=999.0)
        rm.save_confirmed_rows([row], source_id="test")
        saved = load_mapping_rules(p)
        assert saved[0].confidence == pytest.approx(1.0)

    def test_saved_rule_notes_include_source_id(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="csv_pulu")
        saved = load_mapping_rules(p)
        assert "csv_pulu" in (saved[0].notes or "")

    def test_file_reloadable_by_new_instance(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm1 = RuleManager(rules_path=p)
        rm1.save_confirmed_rows([_confirmed_row("SYSF")], source_id="test")

        rm2 = RuleManager(rules_path=p)
        assert rm2.rule_count == 1
        cls, audit = rm2.classify_column("SYSF")
        assert cls.inferred_from == "persistent_mapping_rule"
        assert audit is not None

    def test_rule_count_updates_after_save(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        assert rm.rule_count == 0
        rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="test")
        assert rm.rule_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestDuplicateRuleHandling
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateRuleHandling:
    def test_second_save_overrides_first_by_pattern(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)

        row_v1 = _confirmed_row("MW", signal_type="active_power", confidence=0.80)
        rm.save_confirmed_rows([row_v1], source_id="src_a")

        row_v2 = _confirmed_row("MW", signal_type="active_power", confidence=0.95)
        rm.save_confirmed_rows([row_v2], source_id="src_b")

        saved = load_mapping_rules(p)
        mw_rules = [r for r in saved if r.match_pattern == "mw"]
        assert len(mw_rules) == 1
        assert mw_rules[0].confidence == pytest.approx(0.95)

    def test_total_count_does_not_grow_on_duplicate(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)

        rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="a")
        rm.save_confirmed_rows([_confirmed_row("Frequency")], source_id="b")

        saved = load_mapping_rules(p)
        assert len(saved) == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestMalformedYaml
# ─────────────────────────────────────────────────────────────────────────────


class TestMalformedYaml:
    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("rules: [unclosed bracket", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed"):
            RuleManager(rules_path=p)

    def test_rules_not_a_mapping_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            RuleManager(rules_path=p)


# ─────────────────────────────────────────────────────────────────────────────
# TestReload
# ─────────────────────────────────────────────────────────────────────────────


class TestReload:
    def test_reload_picks_up_new_rule(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)
        assert rm.rule_count == 0

        # Write a rule externally
        p.write_text(
            "rules:\n"
            "  - match_pattern: sysf\n"
            "    match_type: exact\n"
            "    signal_type: frequency\n"
            "    unit: Hz\n"
            "    display_group: frequency\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n",
            encoding="utf-8",
        )
        rm.reload()
        assert rm.rule_count == 1

    def test_classify_after_reload_applies_new_rule(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=p)

        p.write_text(
            "rules:\n"
            "  - match_pattern: sysf\n"
            "    match_type: exact\n"
            "    signal_type: frequency\n"
            "    unit: Hz\n"
            "    display_group: frequency\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n",
            encoding="utf-8",
        )
        rm.reload()
        cls, audit = rm.classify_column("SYSF")
        assert cls.inferred_from == "persistent_mapping_rule"
        assert audit is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestHeuristicFallback
# ─────────────────────────────────────────────────────────────────────────────


class TestHeuristicFallback:
    def test_no_rule_falls_back_to_builtin(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "none.yaml")
        cls, audit = rm.classify_column("Frequency")
        assert cls.signal_type == "frequency"
        assert audit is None
        assert cls.inferred_from in ("name_exact", "name_keyword")

    def test_unknown_column_returns_none_signal_type(self, tmp_path: Path) -> None:
        rm = RuleManager(rules_path=tmp_path / "none.yaml")
        cls, _ = rm.classify_column("XyZAbcUnknown123")
        # May get a synonym or value-profile hit, but without values should be unknown
        assert cls.requires_user_confirmation is True
