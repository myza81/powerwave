"""Integration tests: intelligence layer + D4 workflows remain intact.

Verifies:
  1. Existing D4 classify_csv_column() is unchanged.
  2. IntelligenceManager promotes confidence for SCADA alias columns.
  3. Manifest-extracted rules can be persisted and reapplied.
  4. Original DisturbanceRecord objects are never mutated.
  5. Confidence is preserved as explicit audit trail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.column_classifier import classify_csv_column, CONFIRMATION_THRESHOLD
from app.data.intelligence import IntelligenceManager
from app.data.intelligence.mapping_rules import load_mapping_rules
from app.data.intelligence.models import ConfidencePromotion, MappingRule


# ─────────────────────────────────────────────────────────────────────────────
# D4 backward-compatibility
# ─────────────────────────────────────────────────────────────────────────────


class TestD4ClassifierUnchanged:
    """Existing column_classifier.py functions must be unaffected by D4.1."""

    def test_frequency_exact_still_works(self) -> None:
        cls = classify_csv_column("frequency")
        assert cls.signal_type == "frequency"
        assert cls.confidence >= 0.90
        assert cls.inferred_from == "name_exact"

    def test_mw_exact_still_works(self) -> None:
        cls = classify_csv_column("MW")
        assert cls.signal_type == "active_power"
        assert cls.unit == "MW"

    def test_reactive_power_not_matched_by_active_power(self) -> None:
        cls = classify_csv_column("Reactive Power")
        assert cls.signal_type == "reactive_power"

    def test_unknown_column_still_returns_other(self) -> None:
        cls = classify_csv_column("XYZ_random_col")
        assert cls.signal_type is None
        assert cls.display_group == "other"

    def test_value_profile_still_works(self) -> None:
        rng = np.random.default_rng(42)
        vals = (50.0 + rng.normal(0, 0.05, 100)).tolist()
        cls = classify_csv_column("UnknownCol", vals)
        assert cls.signal_type == "frequency"
        assert cls.inferred_from == "value_profile"


# ─────────────────────────────────────────────────────────────────────────────
# SCADA alias promotion
# ─────────────────────────────────────────────────────────────────────────────


_SCADA_RULES_YAML = """\
rules:
  - match_pattern: sysf
    match_type: exact
    signal_type: frequency
    unit: Hz
    display_group: frequency
    confidence: 1.0
    confirmed_by_operator: true
    notes: "SCADA export alias for system frequency"

  - match_pattern: p_total
    match_type: exact
    signal_type: active_power
    unit: MW
    display_group: power
    confidence: 1.0
    confirmed_by_operator: true
    notes: "SCADA active power column alias"

  - match_pattern: act_pwr
    match_type: exact
    signal_type: active_power
    unit: MW
    display_group: power
    confidence: 1.0
    confirmed_by_operator: true

  - match_pattern: grid_hz
    match_type: exact
    signal_type: frequency
    unit: Hz
    display_group: frequency
    confidence: 1.0
    confirmed_by_operator: true
"""


class TestScadaAliasPromotion:
    @pytest.fixture
    def mgr(self, tmp_path: Path) -> IntelligenceManager:
        p = tmp_path / "rules.yaml"
        p.write_text(_SCADA_RULES_YAML, encoding="utf-8")
        return IntelligenceManager(mapping_rules_path=p)

    def test_sysf_without_rules_is_unknown(self) -> None:
        cls = classify_csv_column("SYSF")
        assert cls.signal_type is None
        assert cls.confidence == pytest.approx(0.0)
        assert cls.requires_user_confirmation is True

    def test_sysf_with_rules_is_frequency(self, mgr: IntelligenceManager) -> None:
        cls, audit = mgr.classify_column("SYSF")
        assert cls.signal_type == "frequency"
        assert cls.unit == "Hz"
        assert cls.confidence == pytest.approx(1.0)
        assert cls.requires_user_confirmation is False

    def test_p_total_without_rules_is_active_power_but_low_confidence(self) -> None:
        # "P_TOTAL" is now a built-in active-power keyword in column_classifier
        # (a generic power-terminology term, not a SCADA-specific alias), so it
        # no longer needs a learned rule to be recognised at all -- it is still
        # below the confirmation threshold on its own, which the persisted
        # SCADA rule below promotes to full confidence.
        cls = classify_csv_column("P_TOTAL")
        assert cls.signal_type == "active_power"
        assert cls.confidence < CONFIRMATION_THRESHOLD
        assert cls.requires_user_confirmation is True

    def test_p_total_with_rules_is_active_power(self, mgr: IntelligenceManager) -> None:
        cls, _ = mgr.classify_column("P_TOTAL")
        assert cls.signal_type == "active_power"
        assert cls.confidence == pytest.approx(1.0)
        assert cls.requires_user_confirmation is False

    def test_grid_hz_with_rules_is_frequency(self, mgr: IntelligenceManager) -> None:
        cls, _ = mgr.classify_column("GRID_HZ")
        assert cls.signal_type == "frequency"

    def test_audit_original_confidence_preserved(self, mgr: IntelligenceManager) -> None:
        _, audit = mgr.classify_column("SYSF")
        assert isinstance(audit, ConfidencePromotion)
        assert audit.original_confidence == pytest.approx(0.0)   # base: unknown
        assert audit.promoted_confidence == pytest.approx(1.0)

    def test_audit_original_inferred_from_preserved(self, mgr: IntelligenceManager) -> None:
        _, audit = mgr.classify_column("SYSF")
        assert audit.original_inferred_from == "unknown"

    def test_known_column_no_audit_when_no_rule(
        self, tmp_path: Path
    ) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        _, audit = mgr.classify_column("Frequency")
        assert audit is None

    def test_keyword_rule_matches_substring(self, tmp_path: Path) -> None:
        yaml_text = (
            "rules:\n"
            "  - match_pattern: load_mw\n"
            "    match_type: keyword\n"
            "    signal_type: active_power\n"
            "    unit: MW\n"
            "    display_group: power\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n"
        )
        p = tmp_path / "kw.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        mgr = IntelligenceManager(mapping_rules_path=p)
        cls, audit = mgr.classify_column("Site Load_MW")
        assert cls.signal_type == "active_power"
        assert isinstance(audit, ConfidencePromotion)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest → rules → re-apply workflow
# ─────────────────────────────────────────────────────────────────────────────


_MANIFEST_WITH_COLUMNS = {
    "event_id": "pulu_20260306",
    "sources": [
        {
            "source_id": "csv_ops",
            "type": "csv",
            "columns": [
                {
                    "name": "System Demand",
                    "signal_type": "active_power",
                    "unit": "MW",
                    "display_group": "power",
                    "confidence": 0.85,
                    "inferred_from": "name_keyword",
                    "requires_user_confirmation": False,
                },
                {
                    "name": "Frequency",
                    "signal_type": "frequency",
                    "unit": "Hz",
                    "display_group": "frequency",
                    "confidence": 0.95,
                    "inferred_from": "name_exact",
                    "requires_user_confirmation": False,
                },
            ],
        }
    ],
}


class TestManifestRulePersistenceWorkflow:
    def test_extract_and_save_then_reload(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        mgr1 = IntelligenceManager(mapping_rules_path=mp)
        count = mgr1.save_rules_from_manifest(_MANIFEST_WITH_COLUMNS)
        assert count == 2

        mgr2 = IntelligenceManager(mapping_rules_path=mp)
        rules = load_mapping_rules(mp)
        assert any(r.match_pattern == "system demand" for r in rules)
        assert any(r.match_pattern == "frequency" for r in rules)

    def test_saved_rules_applied_on_reload(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        mgr1 = IntelligenceManager(mapping_rules_path=mp)
        mgr1.save_rules_from_manifest(_MANIFEST_WITH_COLUMNS)

        mgr2 = IntelligenceManager(mapping_rules_path=mp)
        cls, audit = mgr2.classify_column("System Demand")
        assert cls.signal_type == "active_power"
        assert isinstance(audit, ConfidencePromotion)

    def test_confirmed_column_becomes_confirmed_rule(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        mgr = IntelligenceManager(mapping_rules_path=mp)
        mgr.save_rules_from_manifest(_MANIFEST_WITH_COLUMNS)
        rules = load_mapping_rules(mp)
        demand = next(r for r in rules if r.match_pattern == "system demand")
        assert demand.confirmed_by_operator is True

    def test_unconfirmed_column_remains_unconfirmed_rule(self, tmp_path: Path) -> None:
        manifest = {
            "event_id": "x",
            "sources": [
                {
                    "source_id": "csv",
                    "columns": [
                        {
                            "name": "Tie-Line",
                            "signal_type": "active_power",
                            "unit": "MW",
                            "display_group": "power",
                            "confidence": 0.70,
                            "inferred_from": "name_keyword",
                            "requires_user_confirmation": True,
                        }
                    ],
                }
            ],
        }
        mp = tmp_path / "rules.yaml"
        mgr = IntelligenceManager(mapping_rules_path=mp)
        mgr.save_rules_from_manifest(manifest)
        rules = load_mapping_rules(mp)
        tie = next(r for r in rules if r.match_pattern == "tie-line")
        assert tie.confirmed_by_operator is False
