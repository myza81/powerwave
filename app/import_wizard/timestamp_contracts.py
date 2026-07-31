"""Timestamp repair strategy contract.

Defines *what* kind of repair is needed and the parameters that repair engine
will require.  Contains NO repair logic — that is deferred to a later phase.

Strategy selection guide
------------------------
NO_REPAIR                  Timestamps already valid; parse as-is.
PARSE_DETECTED_FORMAT      Auto-detected strptime/ISO format; use detected_format.
PARSE_USER_FORMAT          User supplied a custom strptime format string; use user_format.
INTERPOLATE_MISSING        Some timestamps are NaT/blank; fill via linear interpolation.
RECONSTRUCT_FROM_INTERVAL  No usable timestamp column; rebuild from sampling_interval_seconds.
COMBINE_DATE_TIME_COLUMNS  Date and time are in separate columns; combine via date_column + time_column.
EXCEL_SERIAL_CONVERSION    Column contains Excel serial date numbers (days since 1899-12-30).
TIMEZONE_ALIGNMENT         Timestamps are in a non-UTC timezone; convert to target_timezone.
RECONSTRUCT_HYBRID         Low-resolution timestamps used as 100 ms anchors; sub-interval
                           times are interpolated within each anchor window using a fixed dt.
                           Parameters: sampling_interval_seconds (or override_sample_rate_hz),
                           override_start_datetime (optional), date_column + time_column
                           (optional, when date/time are split across two columns).
PARSE_ELAPSED_TIME         Column contains relative elapsed time; convert to seconds.
GENERATE_SYNTHETIC_ELAPSED No usable column; generate elapsed seconds from interval.
GENERATE_SAMPLE_INDEX      No timing metadata; use row order as sample index.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class TimestampRepairStrategy(enum.Enum):
    NO_REPAIR                  = "no_repair"
    PARSE_DETECTED_FORMAT      = "parse_detected_format"
    PARSE_USER_FORMAT          = "parse_user_format"
    INTERPOLATE_MISSING        = "interpolate_missing"
    RECONSTRUCT_FROM_INTERVAL  = "reconstruct_from_interval"
    COMBINE_DATE_TIME_COLUMNS  = "combine_date_time_columns"
    EXCEL_SERIAL_CONVERSION    = "excel_serial_conversion"
    TIMEZONE_ALIGNMENT         = "timezone_alignment"
    RECONSTRUCT_HYBRID         = "reconstruct_hybrid"
    PARSE_ELAPSED_TIME         = "parse_elapsed_time"
    GENERATE_SYNTHETIC_ELAPSED = "generate_synthetic_elapsed"
    GENERATE_SAMPLE_INDEX      = "generate_sample_index"


@dataclass(frozen=True, slots=True)
class TimestampRepairPlan:
    """All parameters needed by the (future) timestamp repair engine.

    Immutable once created — the wizard builds a new instance when the user
    confirms or changes any timestamp setting.
    """

    strategy: TimestampRepairStrategy

    # Format strings used by PARSE_DETECTED_FORMAT / PARSE_USER_FORMAT
    detected_format: str | None = None
    user_format: str | None = None

    # Used by RECONSTRUCT_FROM_INTERVAL and RECONSTRUCT_HYBRID
    sampling_interval_seconds: float | None = None

    # Used by COMBINE_DATE_TIME_COLUMNS and RECONSTRUCT_HYBRID (split date+time)
    # Used by GENERATE_SYNTHETIC_ELAPSED when the user supplies sample rate.
    sample_rate_hz: float | None = None

    # Used by PARSE_ELAPSED_TIME
    elapsed_time_unit: str | None = None

    # Used by COMBINE_DATE_TIME_COLUMNS and RECONSTRUCT_HYBRID (split date+time)
    date_column: str | None = None
    time_column: str | None = None

    # Used by TIMEZONE_ALIGNMENT
    source_timezone: str | None = None
    target_timezone: str | None = None

    # Used by RECONSTRUCT_HYBRID — user-assisted overrides
    override_start_datetime: str | None = None     # ISO string; replaces first anchor
    override_sample_rate_hz: float | None = None   # explicit Fs; overrides inferred

    # Set to True once the wizard has validated this plan is internally consistent
    repair_validated: bool = False

    # Human-readable notes surfaced to the user during review
    repair_notes: str | None = None
