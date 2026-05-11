"""Unit tests for app.data.intelligence.mapping_rules."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.column_classifier import ColumnClassification
from app.data.intelligence.mapping_rules import (
    apply_rule_to_classification,
    find_matching_rule,
    load_mapping_rules,
    save_mapping_rules,
)
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _exact_rule(
    pattern: str,
    signal_type: str = "frequency",
    unit: str = "Hz",
    confirmed: bool = True,
    confidence: float = 1.0,
    fingerprint: SourceFingerprint | None = None,
) -> MappingRule:
    return MappingRule(
        match_pattern=pattern,
        match_type="exact",
        signal_type=signal_type,
        unit=unit,
        display_group="frequency" if signal_type == "frequency" else "power",
        confidence=confidence,
        confirmed_by_operator=confirmed,
        source_fingerprint=fingerprint,
    )


def _keyword_rule(pattern: str, signal_type: str = "active_power") -> MappingRule:
    return MappingRule(
        match_pattern=pattern,
        match_type="keyword",
        signal_type=signal_type,
        unit="MW",
        display_group="power",
        confidence=0.95,
        confirmed_by_operator=True,
    )


def _base_cls(column_name: str = "X", signal_type: str | None = None) -> ColumnClassification:
    return ColumnClassification(
        column_name=column_name,
        signal_type=signal_type,
        unit=None,
        display_group="other",
        confidence=0.0,
        inferred_from="unknown",
        requires_user_confirmation=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadMappingRules
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadMappingRules:
    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        rules = load_mapping_rules(tmp_path / "nonexistent.yaml")
        assert rules == []

    def test_empty_rules_list_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text("rules: []", encoding="utf-8")
        assert load_mapping_rules(f) == []

    def test_loads_exact_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text(
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
        rules = load_mapping_rules(f)
        assert len(rules) == 1
        r = rules[0]
        assert r.match_pattern == "sysf"
        assert r.match_type == "exact"
        assert r.signal_type == "frequency"
        assert r.unit == "Hz"
        assert r.confidence == pytest.approx(1.0)
        assert r.confirmed_by_operator is True

    def test_match_pattern_normalised_on_load(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text(
            "rules:\n  - match_pattern: '  SYSF  '\n    signal_type: frequency\n",
            encoding="utf-8",
        )
        rules = load_mapping_rules(f)
        assert rules[0].match_pattern == "sysf"

    def test_loads_fingerprint_scoped_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text(
            "rules:\n"
            "  - match_pattern: freq\n"
            "    signal_type: frequency\n"
            "    source_fingerprint:\n"
            "      export_type: csv\n"
            "      station: PULU\n",
            encoding="utf-8",
        )
        rules = load_mapping_rules(f)
        assert rules[0].source_fingerprint is not None
        assert rules[0].source_fingerprint.export_type == "csv"
        assert rules[0].source_fingerprint.station == "PULU"

    def test_missing_match_pattern_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text("rules:\n  - signal_type: frequency\n", encoding="utf-8")
        with pytest.raises(ValueError, match="match_pattern"):
            load_mapping_rules(f)

    def test_missing_signal_type_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text("rules:\n  - match_pattern: sysf\n", encoding="utf-8")
        with pytest.raises(ValueError, match="signal_type"):
            load_mapping_rules(f)

    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "rules.yaml"
        f.write_text("rules: [unclosed bracket\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_mapping_rules(f)


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveMappingRules
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveMappingRules:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        save_mapping_rules([], p)
        assert p.exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "rules.yaml"
        save_mapping_rules([], p)
        assert p.exists()

    def test_round_trip_exact_rule(self, tmp_path: Path) -> None:
        rule = _exact_rule("sysf", signal_type="frequency", unit="Hz", confirmed=True)
        p = tmp_path / "rules.yaml"
        save_mapping_rules([rule], p)
        reloaded = load_mapping_rules(p)
        assert len(reloaded) == 1
        r = reloaded[0]
        assert r.match_pattern == "sysf"
        assert r.signal_type == "frequency"
        assert r.unit == "Hz"
        assert r.confirmed_by_operator is True

    def test_round_trip_rule_with_fingerprint(self, tmp_path: Path) -> None:
        fp = SourceFingerprint(export_type="csv", station="PULU")
        rule = _exact_rule("freq", fingerprint=fp)
        p = tmp_path / "rules.yaml"
        save_mapping_rules([rule], p)
        reloaded = load_mapping_rules(p)
        assert reloaded[0].source_fingerprint is not None
        assert reloaded[0].source_fingerprint.export_type == "csv"

    def test_round_trip_rule_without_unit(self, tmp_path: Path) -> None:
        rule = MappingRule(
            match_pattern="x",
            match_type="exact",
            signal_type="frequency",
            unit=None,
            display_group="frequency",
            confidence=1.0,
            confirmed_by_operator=True,
        )
        p = tmp_path / "rules.yaml"
        save_mapping_rules([rule], p)
        reloaded = load_mapping_rules(p)
        assert reloaded[0].unit is None

    def test_empty_rules_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.yaml"
        save_mapping_rules([], p)
        assert load_mapping_rules(p) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestFindMatchingRule
# ─────────────────────────────────────────────────────────────────────────────


class TestFindMatchingRule:
    def test_exact_match(self) -> None:
        rules = [_exact_rule("sysf")]
        assert find_matching_rule("sysf", rules) is rules[0]

    def test_exact_match_case_insensitive(self) -> None:
        rules = [_exact_rule("sysf")]
        assert find_matching_rule("SYSF", rules) is rules[0]

    def test_exact_match_strips_whitespace(self) -> None:
        rules = [_exact_rule("sysf")]
        assert find_matching_rule("  SYSF  ", rules) is rules[0]

    def test_no_match_returns_none(self) -> None:
        rules = [_exact_rule("sysf")]
        assert find_matching_rule("frequency", rules) is None

    def test_keyword_match(self) -> None:
        rules = [_keyword_rule("demand")]
        assert find_matching_rule("System Demand", rules) is rules[0]

    def test_keyword_does_not_match_unrelated_name(self) -> None:
        rules = [_keyword_rule("demand")]
        assert find_matching_rule("Frequency", rules) is None

    def test_fingerprint_specific_rule_takes_priority(self) -> None:
        global_rule = _exact_rule("freq", signal_type="frequency", confidence=0.85)
        fp = SourceFingerprint(export_type="csv")
        scoped_rule = _exact_rule("freq", signal_type="rocof", confidence=1.0, fingerprint=fp)
        rules = [global_rule, scoped_rule]  # global first

        caller_fp = SourceFingerprint(export_type="csv")
        matched = find_matching_rule("freq", rules, fingerprint=caller_fp)
        assert matched is scoped_rule

    def test_global_rule_used_when_no_fingerprint(self) -> None:
        global_rule = _exact_rule("freq")
        fp = SourceFingerprint(export_type="csv")
        scoped_rule = _exact_rule("freq", fingerprint=fp)
        rules = [scoped_rule, global_rule]

        matched = find_matching_rule("freq", rules, fingerprint=None)
        assert matched is global_rule

    def test_scoped_rule_not_matched_when_fingerprint_differs(self) -> None:
        fp_csv = SourceFingerprint(export_type="csv")
        scoped = _exact_rule("freq", fingerprint=fp_csv)
        fp_comtrade = SourceFingerprint(export_type="comtrade")
        matched = find_matching_rule("freq", [scoped], fingerprint=fp_comtrade)
        assert matched is None

    def test_empty_rules_returns_none(self) -> None:
        assert find_matching_rule("anything", []) is None


# ─────────────────────────────────────────────────────────────────────────────
# TestApplyRuleToClassification
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyRuleToClassification:
    def test_signal_type_updated(self) -> None:
        rule = _exact_rule("sysf", signal_type="frequency")
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.signal_type == "frequency"

    def test_unit_from_rule(self) -> None:
        rule = _exact_rule("sysf", unit="Hz")
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.unit == "Hz"

    def test_null_unit_preserves_original(self) -> None:
        rule = MappingRule(
            match_pattern="x", match_type="exact",
            signal_type="frequency", unit=None,
            display_group="frequency", confidence=1.0, confirmed_by_operator=True,
        )
        cls = ColumnClassification(
            column_name="x", signal_type=None, unit="kHz",
            display_group="other", confidence=0.0,
            inferred_from="unknown", requires_user_confirmation=True,
        )
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.unit == "kHz"

    def test_confidence_promoted(self) -> None:
        rule = _exact_rule("sysf", confidence=1.0)
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.confidence == pytest.approx(1.0)

    def test_inferred_from_updated(self) -> None:
        rule = _exact_rule("sysf")
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.inferred_from == "persistent_mapping_rule"

    def test_confirmed_by_operator_clears_confirmation_flag(self) -> None:
        rule = _exact_rule("sysf", confirmed=True)
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.requires_user_confirmation is False

    def test_not_confirmed_keeps_confirmation_flag(self) -> None:
        rule = _exact_rule("sysf", confirmed=False)
        cls = _base_cls("sysf")
        updated, _ = apply_rule_to_classification(cls, rule)
        assert updated.requires_user_confirmation is True

    def test_confidence_promotion_audit_original_preserved(self) -> None:
        rule = _exact_rule("sysf", confidence=1.0)
        cls = ColumnClassification(
            column_name="sysf", signal_type=None, unit=None,
            display_group="other", confidence=0.45,
            inferred_from="value_profile", requires_user_confirmation=True,
        )
        _, audit = apply_rule_to_classification(cls, rule)
        assert isinstance(audit, ConfidencePromotion)
        assert audit.original_confidence == pytest.approx(0.45)
        assert audit.promoted_confidence == pytest.approx(1.0)
        assert audit.original_inferred_from == "value_profile"
        assert audit.promoted_inferred_from == "persistent_mapping_rule"
        assert audit.rule_match_pattern == "sysf"

    def test_original_classification_not_mutated(self) -> None:
        rule = _exact_rule("sysf")
        cls = _base_cls("sysf")
        apply_rule_to_classification(cls, rule)
        assert cls.signal_type is None   # frozen dataclass — cannot mutate
        assert cls.confidence == pytest.approx(0.0)
