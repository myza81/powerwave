"""Shared, UI-independent helpers for boundary-aware channel-name matching.

Used by the shared semantic classifier (column_classifier.py), the Import
Wizard's own name rules (column_detector.py), and the CSV/Excel providers'
name-based unit fallback (_infer_unit) to avoid the same class of defect in
three places: a measurement word matched as a raw substring of an unrelated
word (e.g. "curr" inside "Occurrence"), or a genuine measurement word
matched inside a status/control-qualified name (e.g. "voltage" inside
"Voltage Status") that should not become an authoritative analog
measurement.

This module intentionally does not classify anything itself -- it has no
opinion about voltage vs. current vs. power, and no confidence model. It
only answers narrow, reusable questions about tokens: what are they, and
does a candidate token/prefix/phrase/qualifier appear at a real token
boundary rather than as an arbitrary substring.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

_DELIMITER_RE = re.compile(r"[^A-Za-z0-9]+")

# Deliberately narrow: only a lowercase-letter-immediately-followed-by-
# uppercase-letter transition is treated as a word boundary (classic
# camelCase, e.g. "VoltageStatus" -> "Voltage"+"Status"). An
# uppercase-run-followed-by-capitalized-word transition (e.g. "MWStatus")
# is deliberately NOT split, because that same shape is used both for a
# genuine two-word compound ("MW" + "Status") and for a single,
# conventional electrical abbreviation ("MVar", "MW" itself, "ROCOF") that
# must not be broken apart -- there is no purely structural way to tell
# these apart. An under-split ("mwstatus" stays one token) is deliberately
# preferred over an over-split that could break "MVar" into "M"+"Var":
# an under-split still correctly fails to match any known measurement
# token, which is the safe outcome either way.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Conservative, explicit qualifier vocabulary: words that indicate a name
# refers to a status/control/mode signal rather than the analog measurement
# it happens to mention. Checked against every measurement name already
# recognised by column_classifier.py's _EXACT/_KEYWORD tables and
# column_detector.py's _NAME_RULES -- none of them contain any of these
# words, so this list does not suppress any currently-valid measurement
# name. Domain-ambiguous words ("load", "output", "demand") are
# deliberately excluded even though they appear in the wild, because they
# are also legitimate parts of real measurement names (e.g. "Load Demand",
# "Plant Output") in this repository's own vocabulary.
_QUALIFIER_TOKENS: frozenset[str] = frozenset({
    "status", "state", "alarm", "trip", "enable", "enabled", "disable",
    "disabled", "control", "command", "mode", "limit", "healthy", "health",
    "failure", "failed", "fail", "running", "run", "open", "closed",
    "close", "blocked", "block", "available", "availability",
})


def tokenize_channel_name(name: str) -> tuple[str, ...]:
    """Split a channel name into lowercase alphanumeric tokens.

    Splits on whitespace, underscore, hyphen, slash, brackets, and other
    punctuation, and on camelCase word boundaries (see _CAMEL_BOUNDARY_RE).
    Does not split letter/digit boundaries, so relay-style compact tokens
    such as "Vab", "I0", "V1", "MW", and "MVar" remain single tokens
    instead of being broken into meaningless fragments.
    """
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", name)
    return tuple(t.lower() for t in _DELIMITER_RE.split(spaced) if t)


def _coerce_tokens(name_or_tokens: "str | Sequence[str]") -> tuple[str, ...]:
    if isinstance(name_or_tokens, str):
        return tokenize_channel_name(name_or_tokens)
    return tuple(name_or_tokens)


def has_exact_token(name_or_tokens: "str | Sequence[str]", candidates: Iterable[str]) -> bool:
    """True if any of *candidates* appears as a whole token, not a substring."""
    tokens = _coerce_tokens(name_or_tokens)
    wanted = {c.lower() for c in candidates}
    return any(t in wanted for t in tokens)


def has_token_prefix(name_or_tokens: "str | Sequence[str]", prefixes: Iterable[str]) -> bool:
    """True if any token starts with one of *prefixes* (leading characters of
    the token, not an arbitrary mid-word occurrence).
    """
    tokens = _coerce_tokens(name_or_tokens)
    prefix_tuple = tuple(p.lower() for p in prefixes)
    if not prefix_tuple:
        return False
    return any(t.startswith(prefix_tuple) for t in tokens)


def has_token_phrase(name_or_tokens: "str | Sequence[str]", phrase: str) -> bool:
    """True if *phrase* (one or more words) appears as a contiguous run of
    tokens in the name -- the boundary-safe equivalent of substring
    containment for multi-word keyword phrases (e.g. "bus voltage").
    """
    tokens = _coerce_tokens(name_or_tokens)
    phrase_tokens = tokenize_channel_name(phrase)
    n = len(phrase_tokens)
    if n == 0:
        return False
    if n == 1:
        return phrase_tokens[0] in tokens
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == phrase_tokens:
            return True
    return False


def has_status_qualifier(name_or_tokens: "str | Sequence[str]") -> bool:
    """True if the name contains a status/control qualifier token (see
    _QUALIFIER_TOKENS) -- a signal that a name mentioning a measurement
    word is describing a status/control/mode signal about that
    measurement, not the measurement itself.
    """
    tokens = _coerce_tokens(name_or_tokens)
    return any(t in _QUALIFIER_TOKENS for t in tokens)
