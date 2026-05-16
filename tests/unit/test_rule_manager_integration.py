"""Integration tests for Phase D4.4.3C — Persistent Column Mapping Rules.

These tests verify the closed-loop workflow:
  1. First session: column classified by heuristic, operator confirms it
  2. Confirmed mapping saved to YAML
  3. New session (simulates app restart): same column re-classified
  4. Persisted rule fires — confidence improves, confirmed_by_operator=True

Also tests the dialog → rule_manager pipeline using pure-Python mocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.column_classifier import CONFIRMATION_THRESHOLD
from app.data.intelligence.mapping_rules import load_mapping_rules
from app.data.review_summary import ColumnReviewRow
from app.intelligence.rule_manager import RuleManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _confirmed(name: str, signal_type: str, unit: str, group: str, conf: float) -> ColumnReviewRow:
    return ColumnReviewRow(
        column_name=name,
        signal_type=signal_type,
        unit=unit,
        display_group=group,
        confidence=conf,
        inferred_from="name_keyword",
        requires_user_confirmation=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestConfirmThenReclassify
# Core integration: confirm → restart → verify improvement
# ─────────────────────────────────────────────────────────────────────────────


class TestConfirmThenReclassify:
    def test_frequency_column_improves_on_second_session(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "column_mapping_rules.yaml"

        # Session 1: heuristic classifies "SYSF" (not in exact table, unknown)
        rm1 = RuleManager(rules_path=rules_path)
        cls1, audit1 = rm1.classify_column("SYSF")
        # May match synonym "sysfreq"-like or be unknown; either way audit should be None
        assert audit1 is None

        # Operator confirms: SYSF → frequency, Hz, conf=1.0
        row = _confirmed("SYSF", "frequency", "Hz", "frequency", 0.95)
        n = rm1.save_confirmed_rows([row], source_id="csv_scada")
        assert n == 1

        # Session 2 (simulates app restart): load same rules file
        rm2 = RuleManager(rules_path=rules_path)
        cls2, audit2 = rm2.classify_column("SYSF")

        # The persisted rule should now fire
        assert cls2.signal_type == "frequency"
        assert cls2.confidence == pytest.approx(0.95)
        assert cls2.inferred_from == "persistent_mapping_rule"
        assert cls2.requires_user_confirmation is False
        assert audit2 is not None
        assert audit2.promoted_inferred_from == "persistent_mapping_rule"

    def test_active_power_column_improves_on_second_session(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "column_mapping_rules.yaml"

        rm1 = RuleManager(rules_path=rules_path)
        # "SysDem" — not in exact table; synonym may catch it with moderate confidence
        cls1, _ = rm1.classify_column("SysDem")
        original_inferred = cls1.inferred_from

        # Operator confirms: SysDem → active_power, MW
        row = _confirmed("SysDem", "active_power", "MW", "power", 1.0)
        rm1.save_confirmed_rows([row], source_id="csv_scada")

        rm2 = RuleManager(rules_path=rules_path)
        cls2, audit2 = rm2.classify_column("SysDem")

        assert cls2.signal_type == "active_power"
        assert cls2.confidence == pytest.approx(1.0)
        assert cls2.inferred_from == "persistent_mapping_rule"
        assert cls2.requires_user_confirmation is False

    def test_confirmed_rule_clears_user_confirmation_flag(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "column_mapping_rules.yaml"

        rm1 = RuleManager(rules_path=rules_path)
        # Verify baseline: an unknown column requires confirmation
        cls_base, _ = rm1.classify_column("MySpecialCol")
        assert cls_base.requires_user_confirmation is True

        # Confirm it
        row = _confirmed("MySpecialCol", "active_power", "MW", "power", 0.95)
        rm1.save_confirmed_rows([row], source_id="test")

        rm2 = RuleManager(rules_path=rules_path)
        cls2, _ = rm2.classify_column("MySpecialCol")
        assert cls2.requires_user_confirmation is False

    def test_rule_persists_across_three_sessions(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "column_mapping_rules.yaml"

        # Session 1: save
        rm1 = RuleManager(rules_path=rules_path)
        rm1.save_confirmed_rows(
            [_confirmed("FreqCol", "frequency", "Hz", "frequency", 0.95)],
            source_id="src",
        )

        # Session 2: verify and save another column
        rm2 = RuleManager(rules_path=rules_path)
        cls2, _ = rm2.classify_column("FreqCol")
        assert cls2.inferred_from == "persistent_mapping_rule"
        rm2.save_confirmed_rows(
            [_confirmed("MWCol", "active_power", "MW", "power", 0.95)],
            source_id="src",
        )

        # Session 3: both rules should fire
        rm3 = RuleManager(rules_path=rules_path)
        cls3a, a3a = rm3.classify_column("FreqCol")
        cls3b, a3b = rm3.classify_column("MWCol")
        assert cls3a.inferred_from == "persistent_mapping_rule"
        assert cls3b.inferred_from == "persistent_mapping_rule"
        assert rm3.rule_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestDialogConfirmedRowsPipeline
# Simulates what happens when DataReviewDialog populates confirmed_column_rows
# ─────────────────────────────────────────────────────────────────────────────


class TestDialogConfirmedRowsPipeline:
    def test_dialog_confirmed_rows_structure_is_saveable(self, tmp_path: Path) -> None:
        """Verify the dict[source_id → list[ColumnReviewRow]] contract works end-to-end."""
        rules_path = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=rules_path)

        # Simulate what _harvest_confirmed_rows() produces
        confirmed_column_rows: dict[str, list[ColumnReviewRow]] = {
            "csv_scada": [
                _confirmed("Frequency",    "frequency",    "Hz",   "frequency", 0.95),
                _confirmed("System Demand","active_power", "MW",   "power",     0.88),
            ],
            "comtrade_src": [],   # COMTRADE sources have no CSV rows to save
        }

        total_saved = 0
        for src_id, rows in confirmed_column_rows.items():
            total_saved += rm.save_confirmed_rows(rows, source_id=src_id)

        assert total_saved == 2
        assert rm.rule_count == 2

    def test_empty_sources_do_not_error(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "rules.yaml"
        rm = RuleManager(rules_path=rules_path)

        # All sources have no confirmed rows (e.g. all COMTRADE)
        confirmed_column_rows: dict[str, list[ColumnReviewRow]] = {
            "comtrade_src": [],
        }
        for src_id, rows in confirmed_column_rows.items():
            rm.save_confirmed_rows(rows, source_id=src_id)

        assert rm.rule_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestExactMatchOverride
# A stored rule (exact) beats the built-in heuristic
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatchOverride:
    def test_exact_rule_overrides_heuristic(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "rules.yaml"

        # "MW" is in the built-in _EXACT table as active_power; conf=0.95
        rm1 = RuleManager(rules_path=rules_path)
        cls_heuristic, _ = rm1.classify_column("MW")
        assert cls_heuristic.signal_type == "active_power"
        assert cls_heuristic.inferred_from == "name_exact"

        # Operator re-labels "MW" as reactive_power (unusual, but operator's call)
        row = _confirmed("MW", "reactive_power", "MVAr", "power", 1.0)
        rm1.save_confirmed_rows([row], source_id="unusual_src")

        rm2 = RuleManager(rules_path=rules_path)
        cls_rule, audit = rm2.classify_column("MW")
        assert cls_rule.signal_type == "reactive_power"
        assert cls_rule.inferred_from == "persistent_mapping_rule"
        assert audit is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestConfidenceBoosting
# Saving a rule with higher confidence should be visible to next session
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceBoosting:
    def test_below_threshold_column_boosted_to_confirmed(self, tmp_path: Path) -> None:
        rules_path = tmp_path / "rules.yaml"

        rm1 = RuleManager(rules_path=rules_path)
        # "Tie-Line" matches tie-line keyword at 0.70 < CONFIRMATION_THRESHOLD
        cls1, _ = rm1.classify_column("Tie-Line")
        assert cls1.confidence < CONFIRMATION_THRESHOLD
        assert cls1.requires_user_confirmation is True

        # Operator confirms it at full confidence
        row = _confirmed("Tie-Line", "active_power", "MW", "power", 1.0)
        rm1.save_confirmed_rows([row], source_id="grid_ops")

        rm2 = RuleManager(rules_path=rules_path)
        cls2, _ = rm2.classify_column("Tie-Line")
        assert cls2.confidence == pytest.approx(1.0)
        assert cls2.requires_user_confirmation is False
