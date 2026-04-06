"""
src/parsers/csv_parser.py

CSV parser — loads any delimited text file into a DisturbanceRecord.

Supports:
  Separators   : comma (default), semicolon (European), tab — auto-detected
  Time columns  : datetime strings or numeric seconds-from-start
                  detected by header keyword scan
  Unit inference: "Column Name (kV)" / "Column Name [kV]" bracket syntax
  Roles         : delegated to signal_role_detector after channel creation
  Fallback      : NeedsMappingDialog raised when roles remain undetectable

Architecture: Data layer (parsers/) — imports models/ only.
              Never import from ui/ or engine/ here (LAW 1).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.channel import AnalogueChannel, RoleConfidence, SignalRole
from models.disturbance_record import DisturbanceRecord, SourceFormat
from parsers.parser_exceptions import NeedsMappingDialog
from parsers.signal_role_detector import detect_signal_roles

# ── Module-level constants ────────────────────────────────────────────────────

# Bytes to read for separator sniffing
_SNIFFER_SAMPLE: int = 2048

# Fallback separator when sniffing fails
_DEFAULT_SEPARATOR: str = ','

# Supported separators (tried when sniffer is inconclusive)
_CANDIDATE_SEPARATORS: tuple[str, ...] = (',', ';', '\t')

# Header keywords that identify the time column (case-insensitive substring)
_TIME_KEYWORDS: frozenset[str] = frozenset({
    'time', 'timestamp', 'datetime', 'date_time', 't_stamp',
    'utc', 'date', 'seconds', 'second', 'milliseconds',
})

# Regex to extract unit from bracket-style header: "Va (kV)" or "Ia [A]"
_UNIT_RE = re.compile(r'[\(\[]\s*([^\)\]]+?)\s*[\)\]]$')

# Default values when metadata is absent in CSV files
_DEFAULT_NOMINAL_FREQUENCY: float = 50.0
_DEFAULT_SAMPLE_RATE: float = 50.0
_FALLBACK_EPOCH: datetime = datetime(1970, 1, 1, 0, 0, 0)

# Minimum samples needed to compute a reliable sample rate
_MIN_SAMPLES_FOR_RATE: int = 2

# Fraction of channels that must have a non-ANALOGUE role to avoid the dialog
# i.e. if ALL channels are ANALOGUE/LOW the dialog is triggered
_ANALOGUE_ONLY_THRESHOLD: float = 1.0


# ── CsvParser ─────────────────────────────────────────────────────────────────

class CsvParser:
    """Parser for delimited text (CSV/TSV) disturbance record files.

    Produces a single DisturbanceRecord (LAW 5).  Signal roles are
    auto-detected via signal_role_detector; when detection fails for
    all channels, NeedsMappingDialog is raised so the UI can collect
    a column_map from the user.

    Usage::

        record = CsvParser().load(Path('data/fault.csv'))

        # With explicit mapping after dialog:
        record = CsvParser().load(Path('data/fault.csv'), column_map={
            'time_column': 'Timestamp',
            'channels': {
                'Va (kV)': {'role': 'V_PHASE', 'phase': 'A', 'unit': 'kV'},
            }
        })
    """

    def load(
        self,
        filepath: Path,
        column_map: Optional[dict] = None,
    ) -> DisturbanceRecord:
        """Load a CSV file and return a populated DisturbanceRecord.

        Args:
            filepath:   Path to the CSV file (any delimiter).
            column_map: Optional dict controlling column assignment.
                        Schema::

                            {
                              'time_column':       str,    # which col is time
                              'station_name':      str,    # optional override
                              'nominal_frequency': float,  # default 50.0
                              'channels': {
                                  col_name: {
                                      'role':  str,   # SignalRole constant
                                      'phase': str,
                                      'unit':  str,
                                  }, ...
                              }
                            }

                        When None, auto-detection is attempted.  If all
                        analogue channels remain unidentified after running
                        signal_role_detector, NeedsMappingDialog is raised.

        Returns:
            DisturbanceRecord with source_format='CSV'.

        Raises:
            NeedsMappingDialog: When column_map is None and no channels
                                could be assigned a meaningful signal role.
            ValueError: On unreadable or empty files.
        """
        filepath = Path(filepath)
        sep = self._detect_separator(filepath)
        df = pd.read_csv(filepath, sep=sep, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        return self._parse_dataframe(df, filepath, column_map)

    # ── Shared DataFrame → DisturbanceRecord logic ────────────────────────────

    def _parse_dataframe(
        self,
        df: pd.DataFrame,
        filepath: Path,
        column_map: Optional[dict],
    ) -> DisturbanceRecord:
        """Convert a DataFrame (from CSV or Excel sheet) into a DisturbanceRecord.

        This method is the shared core used by both CsvParser.load() and
        ExcelParser (which delegates here after sheet selection).

        Args:
            df:          DataFrame with all values as strings or mixed types.
            filepath:    Source file path (used for station_name fallback).
            column_map:  Optional channel mapping dict (see load() docstring).

        Returns:
            Populated DisturbanceRecord.

        Raises:
            NeedsMappingDialog: When auto-detection fails for all channels.
        """
        # ── Apply column_map overrides to column labels ───────────────────────
        # The time_column key in column_map may reference the original header
        # even if the user has renamed columns.  We do not rename in-place;
        # we just honour time_column when deciding which column is time.

        # ── Identify time column ──────────────────────────────────────────────
        time_col, time_type = self._detect_time_column(df, column_map)

        # ── Build time array ──────────────────────────────────────────────────
        time_array, start_time = self._build_time_array(df, time_col, time_type)

        # ── Compute sample rate ───────────────────────────────────────────────
        sample_rate = self._compute_sample_rate(time_array)

        # ── Collect analogue data columns ─────────────────────────────────────
        data_cols = [c for c in df.columns if c != time_col]
        analogue_channels = self._build_analogue_channels(df, data_cols)

        # ── Run signal role auto-detection (detector sees inferred units) ────
        detect_signal_roles(analogue_channels)  # type: ignore[arg-type]

        # ── Raise NeedsMappingDialog when all roles remain undetected ─────────
        if column_map is None:
            all_analogue = all(
                ch.signal_role == SignalRole.ANALOGUE
                for ch in analogue_channels
            )
            if all_analogue and analogue_channels:
                raise NeedsMappingDialog([c for c in data_cols])

        # ── Apply column_map channel overrides AFTER detection so user wins ──
        if column_map and 'channels' in column_map:
            self._apply_channel_map(analogue_channels, column_map['channels'], data_cols)

        # ── Derive record metadata ────────────────────────────────────────────
        n_samples = len(time_array)
        trigger_sample = n_samples // 2

        station_name = (
            column_map.get('station_name', '') if column_map else ''
        ) or filepath.stem

        nominal_frequency = float(
            (column_map or {}).get('nominal_frequency', _DEFAULT_NOMINAL_FREQUENCY)
        )

        trigger_time = start_time
        if n_samples > 0 and len(time_array) > 0:
            trigger_offset = float(time_array[trigger_sample]) if trigger_sample < n_samples else 0.0
            # trigger_time is start_time + offset; for epoch-based start the offset is exact
            from datetime import timedelta
            trigger_time = start_time + timedelta(seconds=trigger_offset)

        return DisturbanceRecord(
            station_name=station_name,
            device_id='',
            start_time=start_time,
            trigger_time=trigger_time,
            trigger_sample=trigger_sample,
            sample_rate=sample_rate,
            nominal_frequency=nominal_frequency,
            source_format=SourceFormat.CSV,
            file_path=filepath,
            analogue_channels=analogue_channels,
            digital_channels=[],
            time_array=time_array,
            header_text='',
        )

    # ── Separator detection ───────────────────────────────────────────────────

    def _detect_separator(self, filepath: Path) -> str:
        """Detect the field separator by sniffing the first bytes of the file.

        Tries csv.Sniffer first; falls back to counting occurrences of
        candidate separators in the header line.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Single-character separator string.
        """
        try:
            raw = filepath.read_bytes()[:_SNIFFER_SAMPLE].decode('utf-8', errors='replace')
        except OSError:
            return _DEFAULT_SEPARATOR

        # Try the standard library sniffer
        try:
            dialect = csv.Sniffer().sniff(raw, delimiters=',;\t')
            if dialect.delimiter in _CANDIDATE_SEPARATORS:
                return dialect.delimiter
        except csv.Error:
            pass

        # Fallback: count delimiter occurrences in the first line
        first_line = raw.splitlines()[0] if raw.strip() else ''
        counts = {sep: first_line.count(sep) for sep in _CANDIDATE_SEPARATORS}
        best = max(counts, key=lambda s: counts[s])
        return best if counts[best] > 0 else _DEFAULT_SEPARATOR

    # ── Time column detection ─────────────────────────────────────────────────

    def _detect_time_column(
        self,
        df: pd.DataFrame,
        column_map: Optional[dict],
    ) -> tuple[Optional[str], str]:
        """Identify the time column name and its type ('datetime' or 'seconds').

        Resolution order:
          1. ``column_map['time_column']`` if provided
          2. First column whose header contains a _TIME_KEYWORDS substring
          3. First column of the DataFrame (treated as seconds-from-start)

        Args:
            df:         Source DataFrame.
            column_map: Optional caller-supplied mapping.

        Returns:
            Tuple of (column_name_or_None, time_type) where time_type is
            'datetime' or 'seconds'.
        """
        # Explicit override
        if column_map and 'time_column' in column_map:
            col = column_map['time_column']
            if col in df.columns:
                if 'time_type' in column_map:
                    time_type = column_map['time_type']
                else:
                    # Auto-detect from actual values rather than defaulting to 'datetime'
                    time_type = self._guess_time_type(df[col])
                return (col, time_type)

        # Keyword scan (case-insensitive)
        for col in df.columns:
            if any(kw in col.lower() for kw in _TIME_KEYWORDS):
                time_type = self._guess_time_type(df[col])
                return (col, time_type)

        # Fallback: first column as numeric seconds
        if len(df.columns) > 0:
            return (df.columns[0], 'seconds')

        return (None, 'seconds')

    def _guess_time_type(self, series: pd.Series) -> str:
        """Heuristic: if first non-null value parses as a float → 'seconds'.

        Args:
            series: The candidate time column as a Pandas Series.

        Returns:
            'seconds' or 'datetime'.
        """
        first = series.dropna().iloc[0] if not series.dropna().empty else None
        if first is None:
            return 'seconds'
        try:
            float(str(first))
            return 'seconds'
        except (ValueError, TypeError):
            return 'datetime'

    # ── Time array builder ────────────────────────────────────────────────────

    def _build_time_array(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        time_type: str,
    ) -> tuple[np.ndarray, datetime]:
        """Build a float64 seconds-from-start array and extract start_time.

        Args:
            df:        Source DataFrame.
            time_col:  Column name for time, or None.
            time_type: 'datetime' or 'seconds'.

        Returns:
            (time_array, start_time) — float64 ndarray and datetime.
        """
        if time_col is None or time_col not in df.columns:
            n = len(df)
            return (np.arange(n, dtype=np.float64) / _DEFAULT_SAMPLE_RATE, _FALLBACK_EPOCH)

        raw = df[time_col]

        if time_type == 'seconds':
            try:
                numeric = pd.to_numeric(raw, errors='coerce').dropna()
                arr = numeric.to_numpy(dtype=np.float64)
                # Normalise to start at 0.0
                if len(arr) > 0:
                    arr = arr - arr[0]
                return (arr, _FALLBACK_EPOCH)
            except Exception:
                n = len(df)
                return (np.arange(n, dtype=np.float64) / _DEFAULT_SAMPLE_RATE, _FALLBACK_EPOCH)

        # datetime parsing
        try:
            parsed = pd.to_datetime(raw, errors='coerce')
            valid = parsed.dropna()
            if valid.empty:
                n = len(df)
                return (np.arange(n, dtype=np.float64) / _DEFAULT_SAMPLE_RATE, _FALLBACK_EPOCH)

            start_ts = valid.iloc[0]
            start_time = start_ts.to_pydatetime().replace(tzinfo=None)
            delta_s = (parsed - start_ts).dt.total_seconds()
            arr = delta_s.ffill().fillna(0.0).to_numpy(dtype=np.float64)
            return (arr, start_time)
        except Exception:
            n = len(df)
            return (np.arange(n, dtype=np.float64) / _DEFAULT_SAMPLE_RATE, _FALLBACK_EPOCH)

    # ── Sample rate computation ───────────────────────────────────────────────

    def _compute_sample_rate(self, time_array: np.ndarray) -> float:
        """Derive sample rate (Hz) from median time delta between samples.

        Args:
            time_array: float64 array of seconds from record start.

        Returns:
            Sample rate in Hz.  Falls back to _DEFAULT_SAMPLE_RATE if the
            array has fewer than two samples or all deltas are zero.
        """
        if len(time_array) < _MIN_SAMPLES_FOR_RATE:
            return _DEFAULT_SAMPLE_RATE
        diffs = np.diff(time_array)
        positive_diffs = diffs[diffs > 0.0]
        if len(positive_diffs) == 0:
            return _DEFAULT_SAMPLE_RATE
        median_dt = float(np.median(positive_diffs))
        if median_dt <= 0.0:
            return _DEFAULT_SAMPLE_RATE
        return 1.0 / median_dt

    # ── Analogue channel builder ──────────────────────────────────────────────

    def _build_analogue_channels(
        self,
        df: pd.DataFrame,
        data_cols: list[str],
    ) -> list[AnalogueChannel]:
        """Create AnalogueChannel objects from the non-time DataFrame columns.

        Only numeric columns (or columns that parse as numeric) are included.
        Unit is inferred from bracket syntax in the column header.

        Args:
            df:        Source DataFrame.
            data_cols: Column names to convert (time column already excluded).

        Returns:
            List of AnalogueChannel objects with raw_data populated (float32).
        """
        channels: list[AnalogueChannel] = []
        ch_id = 1
        for col in data_cols:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.isna().all():
                # Skip entirely non-numeric columns (e.g. status strings)
                continue
            raw = numeric.fillna(0.0).to_numpy(dtype=np.float32)
            unit = self._infer_unit(col)
            ch = AnalogueChannel(
                channel_id=ch_id,
                name=col,
                phase='',
                unit=unit,
                multiplier=1.0,
                offset=0.0,
                raw_data=raw,
            )
            channels.append(ch)
            ch_id += 1
        return channels

    # ── Unit inference ────────────────────────────────────────────────────────

    def _infer_unit(self, col_name: str) -> str:
        """Extract unit from a bracket-style column header.

        Examples::

            "Va (kV)"   → "kV"
            "Ia [A]"    → "A"
            "P_MW (MW)" → "MW"
            "Channel1"  → ""

        Args:
            col_name: Raw column header string.

        Returns:
            Unit string, or '' when no bracket syntax is found.
        """
        m = _UNIT_RE.search(col_name.strip())
        return m.group(1).strip() if m else ''

    # ── Column map application ────────────────────────────────────────────────

    def _apply_channel_map(
        self,
        channels: list[AnalogueChannel],
        ch_map: dict,
        data_cols: list[str],
    ) -> None:
        """Apply the 'channels' section of a column_map to AnalogueChannel objects.

        Matches channels by their name (original column header).  Overrides
        signal_role, phase, and unit in-place.  Sets role_confirmed=True so
        the UI knows these were explicitly assigned.

        Args:
            channels:  List of AnalogueChannel objects to mutate.
            ch_map:    Dict mapping column name → {role, phase, unit}.
            data_cols: Ordered list of data column names (for index lookup).
        """
        name_to_ch = {ch.name: ch for ch in channels}
        for col_name, overrides in ch_map.items():
            ch = name_to_ch.get(col_name)
            if ch is None:
                continue
            if 'role' in overrides:
                ch.signal_role = overrides['role']
            if 'phase' in overrides:
                ch.phase = overrides['phase']
            if 'unit' in overrides:
                ch.unit = overrides['unit']
            ch.role_confidence = RoleConfidence.HIGH
            ch.role_confirmed = True
