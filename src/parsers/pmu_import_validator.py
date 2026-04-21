"""
src/parsers/pmu_import_validator.py

Pure-logic validator for PMU CSV parse results.

Classifies anomalies detected during PMU CSV parsing into severity tiers and
produces a ParseInspectionReport that drives PmuImportDialog in the UI layer.

Severity tiers
--------------
BLOCKER — dialog required; file cannot be correctly placed on the time axis
          without user input (e.g. broken timestamp, ambiguous date).
INFO    — auto-resolved; shown in the dialog for transparency but no input
          needed from the user.

Architecture: Data layer (parsers/) — no UI imports (LAW 1).
              Independently unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# ── Enumerations ──────────────────────────────────────────────────────────────

class IssueKind(Enum):
    """Categories of detected anomaly."""
    TIMESTAMP_BROKEN = "timestamp_broken"
    DATE_AMBIGUOUS   = "date_ambiguous"


class IssueSeverity(Enum):
    """Whether user input is required to resolve the issue."""
    BLOCKER = "blocker"
    INFO    = "info"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Issue:
    """One detected anomaly."""
    kind:        IssueKind
    severity:    IssueSeverity
    description: str


@dataclass
class AutoResolved:
    """One item the parser handled automatically (informational)."""
    description: str


@dataclass
class ParseInspectionReport:
    """Full classification result for one PMU CSV file parse attempt.

    Attributes:
        station_name:   Station name extracted from metadata row.
        pmu_id:         PMU device ID extracted from metadata row.
        filepath:       Absolute path to the source file.
        gps_quality:    'OK' | 'LOW (GPS fault)' | 'UNKNOWN …'
        first_date_str: Raw date string from the first data row.
        first_time_str: Raw time string from the first data row.
        issues:         Detected anomalies ordered by severity.
        auto_resolved:  Items handled automatically (for display only).
    """
    station_name:   str
    pmu_id:         int
    filepath:       Path
    gps_quality:    str
    first_date_str: str
    first_time_str: str
    issues:         list[Issue]        = field(default_factory=list)
    auto_resolved:  list[AutoResolved] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        """True when at least one BLOCKER-severity issue is present."""
        return any(i.severity == IssueSeverity.BLOCKER for i in self.issues)

    @property
    def blocker_issues(self) -> list[Issue]:
        """Subset of issues with BLOCKER severity."""
        return [i for i in self.issues if i.severity == IssueSeverity.BLOCKER]


# ── Public factory ────────────────────────────────────────────────────────────

def build_report(
    station_name:      str,
    pmu_id:            int,
    filepath:          Path,
    gps_quality:       str,
    first_date_str:    str,
    first_time_str:    str,
    stripped_prefixes: list[str],
    voltage_scaled:    bool,
) -> ParseInspectionReport:
    """Classify PMU CSV parse anomalies and return a ParseInspectionReport.

    Args:
        station_name:      Station name from the metadata row.
        pmu_id:            Numeric PMU ID from the metadata row.
        filepath:          Source file path.
        gps_quality:       GPS quality string returned by the parser.
        first_date_str:    Raw date string from the first data row.
        first_time_str:    Raw time string from the first data row.
        stripped_prefixes: List of bay/unit prefixes that were stripped from
                           column names (e.g. ['KAWA1_', 'UNIT2*']).
        voltage_scaled:    True when magnitude columns were divided by 1000.

    Returns:
        ParseInspectionReport with issues and auto_resolved lists populated.
    """
    report = ParseInspectionReport(
        station_name=station_name,
        pmu_id=pmu_id,
        filepath=filepath,
        gps_quality=gps_quality,
        first_date_str=first_date_str,
        first_time_str=first_time_str,
    )

    # ── Timestamp issues ──────────────────────────────────────────────────────
    if not gps_quality.startswith('OK'):
        colon_count = first_time_str.count(':') if first_time_str else 0
        if colon_count == 1:
            desc = (
                f'Hour field missing — timestamp reads "{first_time_str}" '
                f'(MM:SS.f format, expected HH:MM:SS.mmm). '
                f'The actual recording start time is required to place this '
                f'file on the shared time axis with other files.'
            )
        else:
            desc = (
                f'Timestamp unparseable ("{first_time_str}"). '
                f'The actual recording start time is required.'
            )
        report.issues.append(Issue(
            kind=IssueKind.TIMESTAMP_BROKEN,
            severity=IssueSeverity.BLOCKER,
            description=desc,
        ))

    # ── Date ambiguity (MM/DD vs DD/MM) ──────────────────────────────────────
    if first_date_str:
        parts = first_date_str.replace('-', '/').split('/')
        if len(parts) == 3:
            try:
                a, b = int(parts[0]), int(parts[1])
                if a <= 12 and b <= 12 and a != b:
                    report.issues.append(Issue(
                        kind=IssueKind.DATE_AMBIGUOUS,
                        severity=IssueSeverity.BLOCKER,
                        description=(
                            f'Date "{first_date_str}" is ambiguous — could be '
                            f'MM/DD ({parts[0]}/{parts[1]}) or '
                            f'DD/MM ({parts[1]}/{parts[0]}). '
                            f'Verify the correct interpretation.'
                        ),
                    ))
            except ValueError:
                pass

    # ── Auto-resolved items (informational) ──────────────────────────────────
    if stripped_prefixes:
        unique = sorted(set(stripped_prefixes))
        report.auto_resolved.append(AutoResolved(
            description=f'Column prefix stripped: {", ".join(unique)}'
        ))
    if voltage_scaled:
        report.auto_resolved.append(AutoResolved(
            description='Voltage/current magnitude: raw V/A ÷ 1000 → kV/kA'
        ))

    return report
