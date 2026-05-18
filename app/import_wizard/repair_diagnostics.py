"""Structured diagnostics produced by the timestamp repair pipeline.

RepairDiagnostics is a plain dataclass — no Qt, no pandas, no numpy arrays.
It is safe to serialise, copy, and log.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepairDiagnostics:
    """All quantitative outputs from one timestamp repair run.

    Fields
    ------
    strategy_used               String label of the TimestampRepairStrategy applied.
    total_rows                  Number of rows in the raw input.
    valid_rows                  Rows with a successfully parsed timestamp.
    repaired_rows               Rows where repair logic produced a value that was
                                absent or unparseable in the raw data.
    dropped_rows                Rows discarded because they could not be recovered.
    duplicate_rows              Rows with a timestamp identical to a previous row.
    non_monotonic_rows          Rows where the timestamp decreased or did not advance.
    missing_rows                Rows estimated to be missing (gaps > 1.5× interval).
    nat_rows                    Rows containing NaT after parsing.
    inferred_interval_seconds   Dominant sampling interval, or None.
    interval_jitter_seconds     Median absolute deviation of positive diffs.
    interval_jitter_fraction    jitter / dominant_interval, or None.
    is_irregular                True when jitter_fraction > 5%.
    is_monotonic                True when no backward steps remain after repair.
    timezone_detected           IANA/offset string seen in the source data, or None.
    target_timezone             Target timezone applied (e.g. "UTC"), or None.
    normalization_warnings      Human-readable summary strings for the user.
    """

    strategy_used: str = "unknown"
    total_rows: int = 0
    valid_rows: int = 0
    repaired_rows: int = 0
    dropped_rows: int = 0
    duplicate_rows: int = 0
    non_monotonic_rows: int = 0
    missing_rows: int = 0
    nat_rows: int = 0
    inferred_interval_seconds: float | None = None
    interval_jitter_seconds: float | None = None
    interval_jitter_fraction: float | None = None
    is_irregular: bool = False
    is_monotonic: bool = True
    timezone_detected: str | None = None
    target_timezone: str | None = None
    normalization_warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def add_warning(self, msg: str) -> None:
        self.normalization_warnings.append(msg)

    @property
    def has_quality_issues(self) -> bool:
        """True when any non-trivial anomaly was detected."""
        return bool(
            self.duplicate_rows
            or self.non_monotonic_rows
            or self.missing_rows
            or self.nat_rows
            or self.dropped_rows
            or self.is_irregular
        )
