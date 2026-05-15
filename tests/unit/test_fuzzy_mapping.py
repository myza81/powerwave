"""tests/unit/test_fuzzy_mapping.py

Phase D4.3 — Fuzzy/variant column mapping tests.

Covers:
  - Synonym map: "sys demand", "system demand", "load mw" → active_power
  - Synonym map: "freq", "sys freq" → frequency
  - Synonym map: "tie-line", "tie line", "tieline" → active_power
  - classify_by_synonym exact and substring behaviour
  - IntelligenceManager.classify_column with synonym fallback
  - MappingRule.match_type == "fuzzy" pattern
  - Confirmed mapping rule takes priority over synonym
  - No promotion audit returned for synonym-only match
  - system demand classified/plotted correctly (was a known bug)
"""
from __future__ import annotations

import pytest

from app.data.intelligence import IntelligenceManager
from app.data.intelligence.mapping_rules import classify_by_synonym, _SYNONYM_MAP
from app.data.intelligence.models import MappingRule


class TestSynonymMap:
    """Direct synonym map tests."""

    def test_frequency_exact(self):
        result = classify_by_synonym("frequency")
        assert result is not None
        st, unit, dg, conf = result
        assert st == "frequency"
        assert unit == "Hz"

    def test_freq_exact(self):
        result = classify_by_synonym("freq")
        assert result is not None
        st, _, _, _ = result
        assert st == "frequency"

    def test_sys_freq_exact(self):
        result = classify_by_synonym("sys freq")
        assert result is not None
        st, _, _, _ = result
        assert st == "frequency"

    def test_system_demand_exact(self):
        result = classify_by_synonym("system demand")
        assert result is not None
        st, unit, dg, conf = result
        assert st == "active_power"
        assert unit == "MW"
        assert dg == "power"
        assert conf >= 0.85

    def test_sys_demand_variant(self):
        result = classify_by_synonym("sys demand")
        assert result is not None
        assert result[0] == "active_power"

    def test_load_mw(self):
        result = classify_by_synonym("load mw")
        assert result is not None
        assert result[0] == "active_power"

    def test_tie_line_exact(self):
        result = classify_by_synonym("tie-line")
        assert result is not None
        assert result[0] == "active_power"

    def test_tieline_variant(self):
        result = classify_by_synonym("tieline")
        assert result is not None
        assert result[0] == "active_power"

    def test_tie_line_space(self):
        result = classify_by_synonym("tie line")
        assert result is not None
        assert result[0] == "active_power"

    def test_mvar_exact(self):
        result = classify_by_synonym("mvar")
        assert result is not None
        assert result[0] == "reactive_power"

    def test_mw_exact(self):
        result = classify_by_synonym("mw")
        assert result is not None
        assert result[0] == "active_power"

    def test_rocof_exact(self):
        result = classify_by_synonym("rocof")
        assert result is not None
        assert result[0] == "rocof"

    def test_unknown_returns_none(self):
        assert classify_by_synonym("xyzzy") is None

    def test_case_insensitive_via_normalisation(self):
        """classify_by_synonym normalises to lowercase before lookup."""
        result = classify_by_synonym("System Demand")
        assert result is not None
        assert result[0] == "active_power"

    def test_substring_match_longest_key_wins(self):
        """'Total System Demand' contains 'system demand' → should match."""
        result = classify_by_synonym("Total System Demand")
        assert result is not None
        assert result[0] == "active_power"


class TestFuzzyMatchType:
    """MappingRule.match_type == 'fuzzy' pattern tests."""

    def test_fuzzy_match_substring(self):
        from app.data.intelligence.mapping_rules import _pattern_matches
        rule = MappingRule(
            match_pattern="demand",
            match_type="fuzzy",
            signal_type="active_power",
            unit="MW",
            display_group="power",
            confidence=0.85,
            confirmed_by_operator=False,
        )
        assert _pattern_matches("system demand", rule)
        assert _pattern_matches("demand", rule)
        assert not _pattern_matches("frequency", rule)

    def test_fuzzy_bidirectional(self):
        """Fuzzy matches when pattern is longer than name (norm in pattern)."""
        from app.data.intelligence.mapping_rules import _pattern_matches
        rule = MappingRule(
            match_pattern="system demand",
            match_type="fuzzy",
            signal_type="active_power",
            unit="MW",
            display_group="power",
            confidence=0.88,
            confirmed_by_operator=True,
        )
        # exact
        assert _pattern_matches("system demand", rule)
        # shorter name contained in pattern — bidirectional fuzzy
        assert _pattern_matches("demand", rule)


class TestIntelligenceManagerSynonymFallback:
    """IntelligenceManager.classify_column synonym integration tests."""

    def setup_method(self):
        self.mgr = IntelligenceManager()

    def test_system_demand_classified(self):
        """'System Demand' must be classified as active_power via synonym."""
        cls, audit = self.mgr.classify_column("System Demand")
        assert cls.signal_type == "active_power"
        assert cls.unit == "MW"
        assert audit is None  # synonym, not persistent rule → no audit

    def test_sys_demand_classified(self):
        cls, _ = self.mgr.classify_column("sys demand")
        assert cls.signal_type == "active_power"

    def test_tie_line_classified(self):
        cls, _ = self.mgr.classify_column("Tie-Line")
        assert cls.signal_type == "active_power"

    def test_frequency_classified(self):
        cls, _ = self.mgr.classify_column("Frequency")
        assert cls.signal_type == "frequency"
        assert cls.inferred_from == "name_exact"  # exact match takes priority

    def test_freq_classified(self):
        cls, _ = self.mgr.classify_column("freq")
        assert cls.signal_type == "frequency"

    def test_synonym_inferred_from_label(self):
        """Columns not in the base classifier keyword/exact table reach synonym fallback."""
        # "sys demand" is in _KEYWORD ("demand") → base wins with name_keyword
        cls_demand, audit = self.mgr.classify_column("sys demand")
        assert cls_demand.signal_type == "active_power"
        assert audit is None  # synonym fallback never returns an audit

        # "sysdem" is in the synonym map but NOT in _EXACT or _KEYWORD → pure synonym path
        cls_sysdem, _ = self.mgr.classify_column("sysdem")
        assert cls_sysdem.signal_type == "active_power"
        assert cls_sysdem.inferred_from == "synonym_match"

    def test_confirmed_rule_takes_priority_over_synonym(self, tmp_path):
        """A confirmed persistent rule should override the synonym result."""
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "rules.yaml")
        # Save a rule that maps "sys demand" to reactive_power (deliberately wrong, for test)
        from app.data.intelligence.mapping_rules import save_mapping_rules
        rule = MappingRule(
            match_pattern="sys demand",
            match_type="exact",
            signal_type="reactive_power",
            unit="MVAr",
            display_group="power",
            confidence=0.99,
            confirmed_by_operator=True,
        )
        save_mapping_rules([rule], tmp_path / "rules.yaml")
        mgr2 = IntelligenceManager(mapping_rules_path=tmp_path / "rules.yaml")
        cls, audit = mgr2.classify_column("sys demand")
        assert cls.signal_type == "reactive_power"  # rule wins
        assert audit is not None  # persistent rule → audit returned

    def test_base_classifier_result_preserved_when_no_synonym(self):
        """Unknown column with no synonym should return base classifier result."""
        cls, audit = self.mgr.classify_column("xyzzy_unknown_signal")
        assert cls.signal_type is None
        assert audit is None

    def test_high_base_confidence_not_overridden_by_synonym(self):
        """If base classifier already scored ≥0.80, synonym should not demote."""
        # "frequency" is exact in _EXACT → confidence=0.95, synonym also=0.95
        cls, _ = self.mgr.classify_column("frequency")
        # Should come from name_exact (higher or equal confidence)
        assert cls.inferred_from == "name_exact"
        assert cls.confidence >= 0.90
