"""Data intelligence layer for Powerwave.

Provides persistent mapping rules, source fingerprinting, timestamp
interpretation rules, and confidence promotion for the classification system.

Primary entry point::

    from app.data.intelligence import IntelligenceManager

    mgr = IntelligenceManager()
    classification, audit = mgr.classify_column("SYSF")
"""
from app.data.intelligence.intelligence_manager import IntelligenceManager
from app.data.intelligence.models import (
    ConfidencePromotion,
    MappingRule,
    SourceFingerprint,
    TimestampRule,
)
from app.data.intelligence.fingerprints import (
    build_fingerprint_from_columns,
    build_fingerprint_from_record,
    fingerprints_match,
)
from app.data.intelligence.mapping_rules import (
    apply_rule_to_classification,
    find_matching_rule,
    load_mapping_rules,
    save_mapping_rules,
)
from app.data.intelligence.timestamp_rules import (
    find_matching_timestamp_rule,
    load_timestamp_rules,
    save_timestamp_rules,
)

__all__ = [
    "IntelligenceManager",
    "ConfidencePromotion",
    "MappingRule",
    "SourceFingerprint",
    "TimestampRule",
    "build_fingerprint_from_columns",
    "build_fingerprint_from_record",
    "fingerprints_match",
    "apply_rule_to_classification",
    "find_matching_rule",
    "load_mapping_rules",
    "save_mapping_rules",
    "find_matching_timestamp_rule",
    "load_timestamp_rules",
    "save_timestamp_rules",
]
