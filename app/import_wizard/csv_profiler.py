"""CSV file profiling — delimiter detection, header detection, preview extraction.

All functions are pure (no side effects beyond reading the file).
No pandas, Qt, or numpy imports — standard library only.
"""
from __future__ import annotations

import csv
import io
from typing import Any

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.models import RawPreviewModel
from app.import_wizard.preview_sampler import (
    estimate_csv_row_count,
    read_text_sample,
)

_CANDIDATE_DELIMITERS = (",", ";", "\t", "|")


def detect_delimiter(path: str, *, sample_bytes: int = 16_384) -> str:
    """Return the most likely column delimiter for a CSV file.

    Strategy
    --------
    1. Try ``csv.Sniffer`` on the first *sample_bytes* bytes (fast path).
    2. Fall back to counting occurrences of each candidate delimiter per line
       and choosing the one with the most consistent non-zero count.
    3. Default to "," when nothing is conclusive.
    """
    from pathlib import Path

    path_obj = Path(path)
    try:
        with path_obj.open("rb") as fh:
            raw = fh.read(sample_bytes)
    except OSError:
        return ","

    # Attempt 1: Sniffer
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="replace")
            dialect = csv.Sniffer().sniff(text, delimiters="".join(_CANDIDATE_DELIMITERS))
            return dialect.delimiter
        except csv.Error:
            continue

    # Attempt 2: Frequency counting
    try:
        text = raw.decode("latin-1", errors="replace")
    except Exception:
        return ","

    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if not lines:
        return ","

    best_delim = ","
    best_score = -1
    for delim in _CANDIDATE_DELIMITERS:
        counts = [line.count(delim) for line in lines]
        nonzero = [c for c in counts if c > 0]
        if not nonzero:
            continue
        consistency = len(nonzero) / len(counts)
        avg_count = sum(nonzero) / len(nonzero)
        score = consistency * avg_count
        if score > best_score:
            best_score = score
            best_delim = delim

    return best_delim


def _is_mostly_numeric(cells: list[str], threshold: float = 0.5) -> bool:
    """Return True when ≥ *threshold* fraction of non-empty cells look numeric."""
    non_empty = [c for c in cells if c.strip()]
    if not non_empty:
        return False
    numeric_count = 0
    for cell in non_empty:
        stripped = cell.strip().lstrip("+-")
        # Handle scientific notation and decimal points
        stripped = stripped.replace(".", "", 1).replace("e", "", 1).replace("E", "", 1).replace("+", "", 1).replace("-", "", 1)
        if stripped.isdigit():
            numeric_count += 1
    return (numeric_count / len(non_empty)) >= threshold


def _alpha_score(cells: list[str]) -> float:
    """Fraction of non-empty cells whose content is predominantly alphabetic."""
    non_empty = [c.strip() for c in cells if c.strip()]
    if not non_empty:
        return 0.0
    alpha_count = sum(1 for c in non_empty if sum(ch.isalpha() for ch in c) / max(len(c), 1) >= 0.5)
    return alpha_count / len(non_empty)


def _find_header_row_index(rows: list[list[str]]) -> int:
    """Return the zero-based index of the header row.

    Algorithm (lookahead)
    ---------------------
    A row is considered the header when ALL of:
    - It has ≥ 2 non-empty cells.
    - Its alpha score is ≥ 0.60 (most cells contain text).
    - The next non-blank row (if any) has ≥ 50% numeric cells.

    Falls back to the first non-blank row when the lookahead criterion
    cannot be satisfied (e.g. too few rows to look ahead).
    """
    for i, row in enumerate(rows):
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            continue
        if _alpha_score(row) < 0.60:
            continue
        # Lookahead: find next non-blank row
        for j in range(i + 1, len(rows)):
            next_row = rows[j]
            if not any(c.strip() for c in next_row):
                continue  # skip blank rows
            if _is_mostly_numeric(next_row):
                return i
            break  # next non-blank row is not numeric → not a valid header

    # Fallback: first non-blank row
    for i, row in enumerate(rows):
        if any(c.strip() for c in row):
            return i
    return 0


def profile_csv(
    path: str,
    *,
    delimiter: str | None = None,
    max_preview_rows: int = 50,
    max_scan_rows: int = 200,
) -> tuple[RawPreviewModel, list[ValidationMessage]]:
    """Build a RawPreviewModel for a CSV file without loading it fully.

    Parameters
    ----------
    path:               Absolute path to the CSV file.
    delimiter:          Force a specific delimiter; auto-detect when None.
    max_preview_rows:   Maximum number of data rows to include in preview_rows.
    max_scan_rows:      Maximum lines to read when searching for the header.

    Returns
    -------
    (model, warnings) — warnings is a list of ValidationMessage (never errors).
    """
    warnings: list[ValidationMessage] = []

    lines, encoding = read_text_sample(path, max_lines=max_scan_rows)
    if not lines:
        return (
            RawPreviewModel(column_names=[], preview_rows=[]),
            [
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="CSV_EMPTY",
                    message="File is empty or could not be read.",
                )
            ],
        )

    effective_delimiter = delimiter or detect_delimiter(path)

    # Parse all sampled lines into raw string rows
    text_block = "\n".join(lines)
    reader = csv.reader(io.StringIO(text_block), delimiter=effective_delimiter)
    raw_rows: list[list[str]] = [row for row in reader]

    if not raw_rows:
        return (
            RawPreviewModel(column_names=[], preview_rows=[]),
            [
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="CSV_PARSE_FAILED",
                    message="Could not parse any rows from the file.",
                )
            ],
        )

    header_idx = _find_header_row_index(raw_rows)
    skipped_count = header_idx

    if header_idx >= len(raw_rows):
        return (
            RawPreviewModel(column_names=[], preview_rows=[]),
            [
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="CSV_NO_HEADER",
                    message="Could not locate a header row.",
                )
            ],
        )

    header_row = raw_rows[header_idx]
    column_names = [cell.strip() for cell in header_row]

    # Deduplicate blank/duplicate column names
    seen: dict[str, int] = {}
    clean_names: list[str] = []
    for name in column_names:
        key = name or "Column"
        if key in seen:
            seen[key] += 1
            clean_names.append(f"{key}_{seen[key]}")
        else:
            seen[key] = 0
            clean_names.append(key)
    column_names = clean_names

    if not any(column_names):
        warnings.append(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="CSV_BLANK_HEADERS",
                message="All column header cells are blank.",
            )
        )

    # Build preview rows (data only, after header)
    data_rows: list[list[Any]] = []
    expected_cols = len(column_names)
    for row in raw_rows[header_idx + 1 : header_idx + 1 + max_preview_rows]:
        if not any(cell.strip() for cell in row):
            continue  # skip blank lines
        # Pad or trim to match header width
        if len(row) < expected_cols:
            row = row + [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        data_rows.append(list(row))

    row_count = estimate_csv_row_count(path, header_rows=header_idx + 1)

    return (
        RawPreviewModel(
            column_names=column_names,
            preview_rows=data_rows,
            header_row_index=header_idx,
            skipped_row_count=skipped_count,
            row_count_estimate=row_count,
            sheet_name=None,
            parse_warnings=[w.message for w in warnings],
        ),
        warnings,
    )
