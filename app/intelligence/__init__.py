"""app/intelligence — Application-layer intelligence service.

Provides RuleManager: the UI-facing service for persistent column mapping rules.
This is the interface the UI layer talks to; low-level rule I/O lives in
app/data/intelligence/.
"""
from app.intelligence.rule_manager import RuleManager

__all__ = ["RuleManager"]
