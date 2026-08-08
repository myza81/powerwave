"""Shared ambiguous-date-order policy for CSV/Excel timestamp interpretation.

Powerwave's approved policy for numeric slash/dash-separated dates where both
components could validly be a day or a month (e.g. "3/6/2026") is: assume
DD/MM/YYYY by default. Values that are not actually ambiguous (one component
is out of the 1-12 month range) are parsed correctly regardless of the
default. An explicit user- or detector-selected date order always overrides
the default and is never second-guessed.

This module is stateless, Qt-independent, and does not replace either
ingestion pipeline's own timestamp-parsing machinery. It supplies:

* the disambiguation policy itself (``resolve_component_order``), and
* an ambiguity signal usable for diagnostics (``detect_date_order_ambiguity``,
  ``parse_ambiguous_dates``).

Both the direct CSV/Excel providers (``app.providers.csv.csv_provider``,
``app.providers.excel.excel_provider``) and the Import Wizard's timestamp
repair executors use this module so the two independent pipelines apply the
same default for the same ambiguous input. COMTRADE is unaffected — it has
its own authoritative, format-native timestamp parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

_SLASH_DATE_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}")

_SOURCE_AUTOMATIC = "automatic"
_SOURCE_USER_OVERRIDE = "user_override"

_ORDER_DAY_FIRST = "day_first"
_ORDER_MONTH_FIRST = "month_first"
_ORDER_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class DateOrderResult:
    """Outcome of resolving a single numeric ``first/second`` date-component pair."""

    is_ambiguous: bool
    applied_order: str  # "day_first" | "month_first"
    source: str  # "automatic" | "user_override"


@dataclass(frozen=True)
class AmbiguousDateDiagnostic:
    """Aggregate ambiguity signal for a sample of raw date strings.

    Intended for surfacing an import diagnostic — not for driving parsing
    decisions beyond what ``parse_ambiguous_dates``/``resolve_component_order``
    already apply.
    """

    is_ambiguous: bool
    applied_order: str  # "day_first" | "month_first" | "not_applicable"
    source: str  # "automatic" | "user_override"
    sample_value: str | None = None


def resolve_component_order(
    first: int,
    second: int,
    *,
    user_dayfirst: bool | None = None,
) -> DateOrderResult:
    """Decide whether *first* is the day or the month for a ``first/second/year``
    numeric date, applying Powerwave's day-first-default policy.

    An explicit *user_dayfirst* always wins and is recorded as user-sourced,
    even when the input would have been unambiguous either way.
    """
    first_valid_month = 1 <= first <= 12
    second_valid_month = 1 <= second <= 12
    is_ambiguous = first_valid_month and second_valid_month and first != second

    if user_dayfirst is not None:
        return DateOrderResult(
            is_ambiguous=is_ambiguous,
            applied_order=_ORDER_DAY_FIRST if user_dayfirst else _ORDER_MONTH_FIRST,
            source=_SOURCE_USER_OVERRIDE,
        )

    if not first_valid_month:
        # first > 12: cannot be a month, so first is the day (forced, unambiguous).
        return DateOrderResult(False, _ORDER_DAY_FIRST, _SOURCE_AUTOMATIC)
    if not second_valid_month:
        # second > 12: cannot be a month, so first is the month (forced, unambiguous).
        return DateOrderResult(False, _ORDER_MONTH_FIRST, _SOURCE_AUTOMATIC)

    # Both components could plausibly be a month: genuinely ambiguous.
    # Policy default: day-first.
    return DateOrderResult(is_ambiguous, _ORDER_DAY_FIRST, _SOURCE_AUTOMATIC)


def detect_date_order_ambiguity(
    values: Iterable[object],
    *,
    user_dayfirst: bool | None = None,
) -> AmbiguousDateDiagnostic:
    """Scan sample date strings for day/month ambiguity in numeric
    slash/dash-separated dates (e.g. "3/6/2026"). Non-matching values (ISO
    dates, already-parsed datetimes, etc.) are skipped — they carry no
    day/month ambiguity risk.

    Returns a single diagnostic summarizing whether any sampled value was
    ambiguous and which date order applies.
    """
    any_ambiguous = False
    applied_order = _ORDER_NOT_APPLICABLE
    sample_value: str | None = None
    saw_any = False

    for raw in values:
        if raw is None:
            continue
        text = str(raw)
        match = _SLASH_DATE_RE.match(text)
        if not match:
            continue
        saw_any = True
        first, second = int(match.group(1)), int(match.group(2))
        result = resolve_component_order(first, second, user_dayfirst=user_dayfirst)
        applied_order = result.applied_order
        if result.is_ambiguous:
            any_ambiguous = True
            sample_value = text
            break

    source = _SOURCE_USER_OVERRIDE if user_dayfirst is not None else _SOURCE_AUTOMATIC
    if not saw_any:
        return AmbiguousDateDiagnostic(False, _ORDER_NOT_APPLICABLE, source)

    return AmbiguousDateDiagnostic(
        is_ambiguous=any_ambiguous,
        applied_order=applied_order,
        source=source,
        sample_value=sample_value,
    )


def format_ambiguous_date_example(sample_value: str) -> str | None:
    """Render a human-readable day-first interpretation of *sample_value*
    for UI display, e.g. ``"3/6/2026" -> "3/6/2026 -> 3 June 2026"``.

    Purely a diagnostic rendering of the day-first default already applied
    by ``parse_ambiguous_dates``/``resolve_component_order`` -- it never
    drives a parsing decision. Returns None if *sample_value* cannot be
    parsed (should not normally happen for a sample already identified as
    an ambiguous date by ``detect_date_order_ambiguity``).
    """
    parsed = pd.to_datetime(sample_value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return f"{sample_value} → {parsed.day} {parsed.strftime('%B %Y')}"


def parse_ambiguous_dates(
    series: pd.Series,
    *,
    user_dayfirst: bool | None = None,
) -> tuple[pd.Series, AmbiguousDateDiagnostic]:
    """Parse a Series of raw date/datetime strings applying Powerwave's
    day-first-default ambiguous-date policy.

    Returns the parsed ``datetime64[ns]`` Series plus a diagnostic describing
    whether ambiguity was found in the sampled values and which order was
    applied. An explicit *user_dayfirst* always overrides the default and is
    recorded as the diagnostic source; unparseable values become ``NaT`` as
    with any ``pandas.to_datetime(..., errors="coerce")`` call.
    """
    dayfirst = True if user_dayfirst is None else user_dayfirst
    sample = series.dropna().astype(str).head(50).tolist()
    diagnostic = detect_date_order_ambiguity(sample, user_dayfirst=user_dayfirst)
    parsed = pd.to_datetime(series, format="mixed", dayfirst=dayfirst, errors="coerce")
    return parsed, diagnostic
