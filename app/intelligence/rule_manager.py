"""Application-layer service for persistent column mapping rules.

RuleManager is the single interface between the UI layer and the intelligence
system.  It wraps IntelligenceManager (app/data/intelligence/) and exposes a
clean, minimal API for:
  - classifying columns          (delegates to IntelligenceManager)
  - saving operator-confirmed mappings back to config/column_mapping_rules.yaml

Must NOT contain Qt or UI logic.

Matching priority (applied in IntelligenceManager.classify_column):
  1. Fingerprint-scoped persistent rule (confirmed_by_operator=True first)
  2. Global persistent rule
  3. Built-in classifier (exact keyword)
  4. Synonym/variant library
  5. Value-profile inference
  6. Unknown — requires operator review

Usage::

    rule_manager = RuleManager()
    cls, audit = rule_manager.classify_column("SYSF")
    # after dialog accept:
    n = rule_manager.save_confirmed_rows(confirmed_rows, source_id="csv_pulu")
"""
from __future__ import annotations

from pathlib import Path

from app.data.column_classifier import ColumnClassification
from app.data.intelligence.intelligence_manager import IntelligenceManager
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
)


class RuleManager:
    """Application-level coordinator for column mapping rules.

    Wraps IntelligenceManager to present a clean, UI-facing API.
    Instantiate once per application session and hold in the main window.

    The inner IntelligenceManager is accessible via the *intelligence_manager*
    property so existing worker code that expects an IntelligenceManager
    continues to work without modification.

    Example::

        rule_manager = RuleManager()
        cls, audit = rule_manager.classify_column("SYSF")
        rule_manager.save_confirmed_rows(confirmed_rows, source_id="csv_pulu")
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        self._mgr = IntelligenceManager(mapping_rules_path=rules_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def intelligence_manager(self) -> IntelligenceManager:
        """The underlying IntelligenceManager for injection into worker threads."""
        return self._mgr

    @property
    def rule_count(self) -> int:
        """Number of currently loaded mapping rules."""
        return len(self._mgr._mapping_rules)

    # ─────────────────────────────────────────────────────────────────────────
    # Classification (delegates to IntelligenceManager)
    # ─────────────────────────────────────────────────────────────────────────

    def classify_column(
        self,
        column_name: str,
        values=None,
        fingerprint: SourceFingerprint | None = None,
    ) -> tuple[ColumnClassification, ConfidencePromotion | None]:
        """Classify a single column, applying stored rules when they match."""
        return self._mgr.classify_column(column_name, values, fingerprint)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_confirmed_rows(
        self,
        confirmed_rows: list,           # list[ColumnReviewRow]
        source_id: str = "",
        fingerprint: SourceFingerprint | None = None,
    ) -> int:
        """Persist operator-confirmed column classifications as MappingRules.

        Converts ColumnReviewRow objects where requires_user_confirmation=False
        and signal_type is set into confirmed MappingRule entries, then merges
        them into config/column_mapping_rules.yaml.

        Rows with requires_user_confirmation=True or signal_type=None are
        silently skipped — the operator did not confirm those.

        Returns the number of rules persisted (0 if all rows skipped).
        """
        rules = [
            MappingRule(
                match_pattern=row.column_name.strip().lower(),
                match_type="exact",
                signal_type=row.signal_type,
                unit=row.unit,
                display_group=row.display_group,
                confidence=min(1.0, max(0.0, row.confidence)),
                confirmed_by_operator=True,
                source_fingerprint=fingerprint,
                notes=(
                    f"Confirmed by operator from source '{source_id}'"
                    if source_id else None
                ),
            )
            for row in confirmed_rows
            if row.signal_type is not None and not row.requires_user_confirmation
        ]
        if not rules:
            return 0
        return self._mgr.save_confirmed_rules(rules)

    def reload(self) -> None:
        """Reload rules from disk (useful after external YAML edits)."""
        from app.data.intelligence.mapping_rules import load_mapping_rules
        self._mgr._mapping_rules = load_mapping_rules(self._mgr._mapping_rules_path)
