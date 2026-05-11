"""Unit tests for app.data.intelligence.intelligence_manager.IntelligenceManager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.data.intelligence import IntelligenceManager
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
    TimestampRule,
)
from app.data.intelligence.mapping_rules import load_mapping_rules
from app.data.intelligence.timestamp_rules import load_timestamp_rules


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_mapping_rules(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "column_mapping_rules.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def _write_timestamp_rules(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "timestamp_rules.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p


_SYSF_RULE_YAML = (
    "rules:\n"
    "  - match_pattern: sysf\n"
    "    match_type: exact\n"
    "    signal_type: frequency\n"
    "    unit: Hz\n"
    "    display_group: frequency\n"
    "    confidence: 1.0\n"
    "    confirmed_by_operator: true\n"
    "    notes: SCADA alias\n"
)

_PULU_TS_RULE_YAML = (
    "rules:\n"
    "  - source_pattern: pulu_scada_export\n"
    "    date_format: 'M/D/YYYY'\n"
    "    ambiguous_resolution: 'M/D/YYYY'\n"
    "    confirmed_by_operator: true\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# TestIntelligenceManagerInit
# ─────────────────────────────────────────────────────────────────────────────


class TestIntelligenceManagerInit:
    def test_no_config_files_initialises_empty(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(
            mapping_rules_path=tmp_path / "missing.yaml",
            timestamp_rules_path=tmp_path / "missing_ts.yaml",
        )
        cls, audit = mgr.classify_column("Frequency")
        # Falls through to base classifier — no rules, no audit
        assert audit is None
        assert cls.signal_type == "frequency"    # base classifier handles this


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyColumn
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyColumn:
    def test_no_rules_falls_through_to_base_classifier(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(
            mapping_rules_path=tmp_path / "none.yaml",
        )
        cls, audit = mgr.classify_column("Frequency")
        assert cls.signal_type == "frequency"
        assert audit is None

    def test_matching_rule_promotes_confidence(self, tmp_path: Path) -> None:
        p = _write_mapping_rules(tmp_path, _SYSF_RULE_YAML)
        mgr = IntelligenceManager(mapping_rules_path=p)
        cls, audit = mgr.classify_column("SYSF")
        assert cls.signal_type == "frequency"
        assert cls.confidence == pytest.approx(1.0)
        assert cls.inferred_from == "persistent_mapping_rule"

    def test_matching_rule_clears_confirmation_flag(self, tmp_path: Path) -> None:
        p = _write_mapping_rules(tmp_path, _SYSF_RULE_YAML)
        mgr = IntelligenceManager(mapping_rules_path=p)
        cls, _ = mgr.classify_column("SYSF")
        assert cls.requires_user_confirmation is False

    def test_audit_returned_when_rule_applies(self, tmp_path: Path) -> None:
        p = _write_mapping_rules(tmp_path, _SYSF_RULE_YAML)
        mgr = IntelligenceManager(mapping_rules_path=p)
        _, audit = mgr.classify_column("SYSF")
        assert isinstance(audit, ConfidencePromotion)
        assert audit.promoted_inferred_from == "persistent_mapping_rule"
        assert audit.rule_match_pattern == "sysf"

    def test_no_audit_when_no_rule_matches(self, tmp_path: Path) -> None:
        p = _write_mapping_rules(tmp_path, _SYSF_RULE_YAML)
        mgr = IntelligenceManager(mapping_rules_path=p)
        _, audit = mgr.classify_column("Totally Unknown Column XYZ")
        assert audit is None

    def test_fingerprint_scoped_rule_takes_priority(self, tmp_path: Path) -> None:
        yaml_text = (
            "rules:\n"
            "  - match_pattern: freq\n"
            "    signal_type: rocof\n"
            "    unit: Hz/s\n"
            "    display_group: rocof\n"
            "    confidence: 1.0\n"
            "    confirmed_by_operator: true\n"
            "    source_fingerprint:\n"
            "      export_type: csv\n"
            "  - match_pattern: freq\n"
            "    signal_type: frequency\n"
            "    unit: Hz\n"
            "    display_group: frequency\n"
            "    confidence: 0.90\n"
            "    confirmed_by_operator: true\n"
        )
        p = _write_mapping_rules(tmp_path, yaml_text)
        mgr = IntelligenceManager(mapping_rules_path=p)
        fp = SourceFingerprint(export_type="csv")
        cls, _ = mgr.classify_column("freq", fingerprint=fp)
        assert cls.signal_type == "rocof"

    def test_values_passed_to_base_classifier(self, tmp_path: Path) -> None:
        import numpy as np

        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rng = np.random.default_rng(42)
        values = (50.0 + rng.normal(0, 0.05, 100)).tolist()
        cls, _ = mgr.classify_column("UnknownCol", values=values)
        # Base classifier should infer frequency from values
        assert cls.signal_type == "frequency"
        assert cls.inferred_from == "value_profile"


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyColumns (DataFrame API)
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyColumns:
    def test_classifies_all_non_timestamp_columns(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        df = pd.DataFrame({"Time": [1, 2], "Frequency": [50.0, 50.1], "MW": [100.0, 110.0]})
        result = mgr.classify_columns(df, timestamp_column="Time")
        assert "Frequency" in result
        assert "MW" in result
        assert "Time" not in result

    def test_returns_tuple_per_column(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        df = pd.DataFrame({"MW": [100.0]})
        result = mgr.classify_columns(df)
        cls, audit = result["MW"]
        assert cls.signal_type == "active_power"
        assert audit is None

    def test_rule_applied_in_batch(self, tmp_path: Path) -> None:
        p = _write_mapping_rules(tmp_path, _SYSF_RULE_YAML)
        mgr = IntelligenceManager(mapping_rules_path=p)
        df = pd.DataFrame({"SYSF": [50.0, 50.1], "MW": [100.0, 110.0]})
        result = mgr.classify_columns(df)
        sysf_cls, sysf_audit = result["SYSF"]
        assert sysf_cls.signal_type == "frequency"
        assert isinstance(sysf_audit, ConfidencePromotion)
        mw_cls, mw_audit = result["MW"]
        assert mw_cls.signal_type == "active_power"
        assert mw_audit is None


# ─────────────────────────────────────────────────────────────────────────────
# TestResolveTimestampFormat
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveTimestampFormat:
    def test_matching_rule_returned(self, tmp_path: Path) -> None:
        p = _write_timestamp_rules(tmp_path, _PULU_TS_RULE_YAML)
        mgr = IntelligenceManager(timestamp_rules_path=p)
        rule = mgr.resolve_timestamp_format("pulu_scada_export")
        assert isinstance(rule, TimestampRule)
        assert rule.date_format == "M/D/YYYY"
        assert rule.confirmed_by_operator is True

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        p = _write_timestamp_rules(tmp_path, _PULU_TS_RULE_YAML)
        mgr = IntelligenceManager(timestamp_rules_path=p)
        assert mgr.resolve_timestamp_format("unknown_source") is None

    def test_no_rules_file_returns_none(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(timestamp_rules_path=tmp_path / "none.yaml")
        assert mgr.resolve_timestamp_format("any_source") is None


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildFingerprint
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFingerprint:
    def test_deterministic(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        fp1 = mgr.build_fingerprint(["Frequency", "MW"], source_type="csv")
        fp2 = mgr.build_fingerprint(["MW", "Frequency"], source_type="csv")
        assert fp1 == fp2

    def test_station_included(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        fp = mgr.build_fingerprint(["Frequency"], station="PULU")
        assert fp.station == "PULU"


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractRulesFromManifest
# ─────────────────────────────────────────────────────────────────────────────


_PULU_MANIFEST = {
    "event_id": "pulu_20260306",
    "sources": [
        {
            "source_id": "csv_ops",
            "type": "csv",
            "columns": [
                {
                    "name": "Time.1",
                    "display_group": "other",
                    "confidence": 0.0,
                    "inferred_from": "unknown",
                    "requires_user_confirmation": True,
                },
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
                    "name": "Tie-Line",
                    "signal_type": "active_power",
                    "unit": "MW",
                    "display_group": "power",
                    "confidence": 0.70,
                    "inferred_from": "name_keyword",
                    "requires_user_confirmation": True,
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


class TestExtractRulesFromManifest:
    def test_skips_unknown_inferred_from(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest(_PULU_MANIFEST)
        patterns = [r.match_pattern for r in rules]
        assert "time.1" not in patterns

    def test_includes_confirmed_column(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest(_PULU_MANIFEST)
        demand_rule = next(r for r in rules if r.match_pattern == "system demand")
        assert demand_rule.confirmed_by_operator is True
        assert demand_rule.signal_type == "active_power"

    def test_includes_unconfirmed_column(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest(_PULU_MANIFEST)
        tie_rule = next(r for r in rules if r.match_pattern == "tie-line")
        assert tie_rule.confirmed_by_operator is False
        assert tie_rule.signal_type == "active_power"

    def test_confidence_preserved(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest(_PULU_MANIFEST)
        freq_rule = next(r for r in rules if r.match_pattern == "frequency")
        assert freq_rule.confidence == pytest.approx(0.95)

    def test_three_rules_extracted(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest(_PULU_MANIFEST)
        assert len(rules) == 3  # Time.1 excluded (unknown)

    def test_empty_manifest_returns_empty(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "none.yaml")
        rules = mgr.extract_rules_from_manifest({"event_id": "x", "sources": []})
        assert rules == []


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveRulesFromManifest
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveRulesFromManifest:
    def test_returns_count_of_extracted_rules(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        mgr = IntelligenceManager(mapping_rules_path=mp)
        n = mgr.save_rules_from_manifest(_PULU_MANIFEST)
        assert n == 3

    def test_rules_written_to_file(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        mgr = IntelligenceManager(mapping_rules_path=mp)
        mgr.save_rules_from_manifest(_PULU_MANIFEST)
        assert mp.exists()
        reloaded = load_mapping_rules(mp)
        patterns = {r.match_pattern for r in reloaded}
        assert "system demand" in patterns
        assert "frequency" in patterns

    def test_merge_new_rules_override_existing(self, tmp_path: Path) -> None:
        mp = tmp_path / "rules.yaml"
        # Pre-existing rule for "frequency" with low confidence
        from app.data.intelligence.mapping_rules import save_mapping_rules
        existing = MappingRule(
            match_pattern="frequency",
            match_type="exact",
            signal_type="frequency",
            unit=None,
            display_group="frequency",
            confidence=0.50,
            confirmed_by_operator=False,
        )
        save_mapping_rules([existing], mp)

        mgr = IntelligenceManager(mapping_rules_path=mp)
        mgr.save_rules_from_manifest(_PULU_MANIFEST)

        reloaded = load_mapping_rules(mp)
        freq_rule = next(r for r in reloaded if r.match_pattern == "frequency")
        # Manifest has confidence=0.95 — should override
        assert freq_rule.confidence == pytest.approx(0.95)

    def test_custom_path_used(self, tmp_path: Path) -> None:
        mgr = IntelligenceManager(mapping_rules_path=tmp_path / "default.yaml")
        custom = tmp_path / "custom.yaml"
        mgr.save_rules_from_manifest(_PULU_MANIFEST, path=custom)
        assert custom.exists()
        assert not (tmp_path / "default.yaml").exists()
