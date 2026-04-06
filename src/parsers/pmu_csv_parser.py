"""
src/parsers/pmu_csv_parser.py

PMU CSV parser — loads Malaysian grid PMU CSV files into a DisturbanceRecord.

File format (confirmed from real files):
  Row 1: metadata — "ID: NNN, Station Name: XXXXX,,,,,"
  Row 2: column headers — always starts with "Date"
  Row 3+: data rows at 50 fps (0.020 s interval)

Time handling:
  Time column: header contains "Time" (case-insensitive)
  Valid timestamp: "HH:MM:SS.mmm" (two colons) — combined with Date, converted
                   from SGT (UTC+8) to UTC, seconds computed from first sample.
  Broken timestamp: "MM:SS.s" (one colon) — PMU GPS fault detected; synthetic
                    time array is built from sample index at PMU_SAMPLE_RATE.

Column prefix stripping:
  "KAWA1_V1 Magnitude" → "V1 Magnitude"
  "UNIT2*V1 Magnitude" → "V1 Magnitude"
  Prefix pattern: word characters followed by '_' or '*'.

Voltage/current scaling:
  Raw values in Volts → ÷ 1000 → kV for display
  Raw values in Amps → ÷ 1000 → kA for display
  Frequency and angle channels: no scaling

Architecture: Data layer (parsers/) — imports models/ only.
              Never import from ui/ or engine/ here (LAW 1).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.channel import (
    AnalogueChannel,
    DigitalChannel,
    RoleConfidence,
    SignalRole,
    default_colour_for,
)
from models.disturbance_record import DisturbanceRecord, SourceFormat

# ── Module constants ──────────────────────────────────────────────────────────

PMU_SAMPLE_RATE: float = 50.0        # Hz — confirmed 0.020 s between rows
PMU_SGT_OFFSET_H: int  = 8           # SGT is UTC+8 — subtract to get UTC
PMU_NOMINAL_FREQ: float = 50.0       # Malaysian grid
PMU_V_SCALE: float  = 1e-3           # raw Volts → kV
PMU_I_SCALE: float  = 1e-3           # raw Amps  → kA

# Regex to strip bay/unit prefix: "KAWA1_..." → strip "KAWA1_"
#                                  "UNIT2*..." → strip "UNIT2*"
_PREFIX_RE = re.compile(r'^[A-Za-z0-9]+[_*](.+)$')

# Regex to extract metadata from row 1
_ID_RE:      re.Pattern[str] = re.compile(r'ID:\s*(\d+)',            re.IGNORECASE)
_STATION_RE: re.Pattern[str] = re.compile(r'Station Name:\s*([^,\n]+)', re.IGNORECASE)

# Regex to extract nominal kV from station name prefix digits
_NOMINAL_KV_RE: re.Pattern[str] = re.compile(r'^(\d+)')

# Fallback epoch when no valid timestamp can be parsed
_FALLBACK_EPOCH: datetime = datetime(1970, 1, 1, 0, 0, 0)

# ── Core column definitions ───────────────────────────────────────────────────
# Key: lower-cased core column name (after prefix strip)
# Value: (display_name, signal_role, phase, unit, scale_factor)

_CORE_COLUMNS: dict[str, tuple[str, str, str, str, float]] = {
    'v1 magnitude': ('V1 Magnitude', SignalRole.V1_PMU,  'Pos-seq', 'kV',   PMU_V_SCALE),
    'v1 angle':     ('V1 Angle',     SignalRole.V1_PMU,  'Pos-seq', '°',    1.0),
    'i1 magnitude': ('I1 Magnitude', SignalRole.I1_PMU,  'Pos-seq', 'kA',   PMU_I_SCALE),
    'i1 angle':     ('I1 Angle',     SignalRole.I1_PMU,  'Pos-seq', '°',    1.0),
    'va magnitude': ('VA Magnitude', SignalRole.V_PHASE, 'A',       'kV',   PMU_V_SCALE),
    'va angle':     ('VA Angle',     SignalRole.V_PHASE, 'A',       '°',    1.0),
    'vb magnitude': ('VB Magnitude', SignalRole.V_PHASE, 'B',       'kV',   PMU_V_SCALE),
    'vb angle':     ('VB Angle',     SignalRole.V_PHASE, 'B',       '°',    1.0),
    'vc magnitude': ('VC Magnitude', SignalRole.V_PHASE, 'C',       'kV',   PMU_V_SCALE),
    'vc angle':     ('VC Angle',     SignalRole.V_PHASE, 'C',       '°',    1.0),
    'frequency':    ('Frequency',    SignalRole.FREQ,    '',        'Hz',   1.0),
    'df/dt':        ('df/dt',        SignalRole.ROCOF,   '',        'Hz/s', 1.0),
}

# Columns to skip (time axis and status — not analogue signal channels)
_SKIP_KEYWORDS: frozenset[str] = frozenset({'date', 'time', 'status', 'stat'})


# ── Public detection helper ───────────────────────────────────────────────────

def is_pmu_csv(filepath: Path) -> bool:
    """Return True when filepath is a PMU CSV file (starts with metadata row).

    Checks whether the first non-empty line starts with "ID:" or contains
    "Station Name:", which is the PMU CSV header format used by Malaysian
    grid PMU data loggers.

    Args:
        filepath: Path to the file to probe.

    Returns:
        True if the file is a PMU CSV; False otherwise.
    """
    try:
        raw = filepath.read_bytes()[:512].decode('utf-8-sig', errors='replace')
        first_line = raw.splitlines()[0].strip() if raw.strip() else ''
        return first_line.upper().startswith('ID:') or 'STATION NAME:' in first_line.upper()
    except OSError:
        return False


# ── PmuCsvParser ──────────────────────────────────────────────────────────────

class PmuCsvParser:
    """Parser for Malaysian grid PMU CSV files.

    Produces a single DisturbanceRecord (LAW 5) with source_format='PMU_CSV'.
    PMU data is always 50 fps → TREND display mode.

    Usage::

        from parsers.pmu_csv_parser import is_pmu_csv, PmuCsvParser
        from pathlib import Path

        path = Path('data/500JMJG_U5.csv')
        if is_pmu_csv(path):
            record = PmuCsvParser().load(path)
    """

    def load(self, filepath: Path) -> DisturbanceRecord:
        """Load a PMU CSV file and return a populated DisturbanceRecord.

        Args:
            filepath: Path to the PMU CSV file.

        Returns:
            DisturbanceRecord with source_format='PMU_CSV' and
            display_mode='TREND' (50 Hz sample rate).

        Raises:
            ValueError: When the file is unreadable or the header format
                        is not recognised as a PMU CSV file.
        """
        filepath = Path(filepath)

        # ── Read metadata row and column headers ──────────────────────────
        raw_first_line = self._read_first_line(filepath)
        pmu_id, station_name = self._parse_metadata(raw_first_line, filepath)

        # ── Read data (skip metadata row → header is row index 1) ─────────
        df = pd.read_csv(
            filepath,
            skiprows=1,
            dtype=str,
            encoding='utf-8-sig',
        )
        df.columns = [str(c).strip() for c in df.columns]

        # ── Identify Date and Time columns ────────────────────────────────
        date_col = self._find_date_column(df)
        time_col = self._find_time_column(df)

        # ── Build time array ──────────────────────────────────────────────
        n_rows = len(df)
        time_array, start_time, gps_quality = self._build_time_array(
            df, date_col, time_col, n_rows
        )

        # ── Build analogue channels ───────────────────────────────────────
        skip_cols: set[str] = set()
        if date_col:
            skip_cols.add(date_col)
        if time_col:
            skip_cols.add(time_col)

        analogue_channels = self._build_analogue_channels(df, skip_cols, pmu_id)

        # ── Build status digital channel ──────────────────────────────────
        digital_channels = self._build_status_channel(df, skip_cols)

        # ── Record metadata ───────────────────────────────────────────────
        n_samples     = len(time_array)
        trigger_sample = 0
        trigger_time   = start_time

        nominal_kv = self._extract_nominal_kv(station_name)
        header_text = (
            f"PMU CSV  |  ID: {pmu_id}  |  Station: {station_name}"
            f"  |  {nominal_kv} kV  |  GPS: {gps_quality}"
        )

        return DisturbanceRecord(
            station_name=station_name,
            device_id=str(pmu_id),
            start_time=start_time,
            trigger_time=trigger_time,
            trigger_sample=trigger_sample,
            sample_rate=PMU_SAMPLE_RATE,
            nominal_frequency=PMU_NOMINAL_FREQ,
            source_format=SourceFormat.PMU_CSV,
            file_path=filepath,
            analogue_channels=analogue_channels,
            digital_channels=digital_channels,
            time_array=time_array,
            header_text=header_text,
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _read_first_line(self, filepath: Path) -> str:
        """Read the metadata row (row 0) from the file.

        Args:
            filepath: Path to the PMU CSV file.

        Returns:
            Raw first-line string (BOM stripped, whitespace stripped).

        Raises:
            ValueError: On unreadable file.
        """
        try:
            raw = filepath.read_bytes()[:1024].decode('utf-8-sig', errors='replace')
            lines = raw.splitlines()
            return lines[0].strip() if lines else ''
        except OSError as exc:
            raise ValueError(f"Cannot read {filepath}: {exc}") from exc

    def _parse_metadata(
        self, first_line: str, filepath: Path
    ) -> tuple[int, str]:
        """Extract PMU ID and station name from the metadata row.

        Args:
            first_line: Raw first line of the PMU CSV file.
            filepath:   Source path (used as fallback station name).

        Returns:
            (pmu_id, station_name) — int and str.
        """
        pmu_id = 0
        id_m = _ID_RE.search(first_line)
        if id_m:
            pmu_id = int(id_m.group(1))

        station_name = filepath.stem
        st_m = _STATION_RE.search(first_line)
        if st_m:
            station_name = st_m.group(1).strip().strip(',')

        return pmu_id, station_name

    # ── Column discovery ──────────────────────────────────────────────────────

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the column name whose header equals 'Date' (case-insensitive).

        Args:
            df: DataFrame with actual data columns as column headers.

        Returns:
            Column name string, or None when not found.
        """
        for col in df.columns:
            if col.strip().lower() == 'date':
                return col
        return None

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the column name whose header contains 'Time' (case-insensitive).

        Prefers a column whose header starts with 'Time' to avoid false
        matches on columns whose names contain 'time' mid-word.

        Args:
            df: DataFrame with actual data columns as column headers.

        Returns:
            Column name string, or None when not found.
        """
        # Prefer exact prefix match first: "Time", "Time (Asia/Singapore)", etc.
        for col in df.columns:
            if col.strip().lower().startswith('time'):
                return col
        # Fallback: any column with 'time' anywhere
        for col in df.columns:
            if 'time' in col.strip().lower():
                return col
        return None

    # ── Time array ────────────────────────────────────────────────────────────

    def _build_time_array(
        self,
        df: pd.DataFrame,
        date_col: Optional[str],
        time_col: Optional[str],
        n_rows: int,
    ) -> tuple[np.ndarray, datetime, str]:
        """Build a float64 seconds-from-start time array.

        Attempts to parse combined Date + Time columns into UTC datetimes.
        When timestamps are broken (PMU GPS fault — single colon in time string)
        or unparseable, a synthetic array at PMU_SAMPLE_RATE is returned.

        Args:
            df:       Source DataFrame.
            date_col: Column name for Date values, or None.
            time_col: Column name for Time values, or None.
            n_rows:   Number of data rows.

        Returns:
            (time_array, start_time, gps_quality)
            time_array   — float64 ndarray, seconds from record start
            start_time   — UTC datetime of first sample (or fallback epoch)
            gps_quality  — 'OK' or 'LOW (GPS fault)'
        """
        # Synthetic fallback array
        synthetic = np.arange(n_rows, dtype=np.float64) / PMU_SAMPLE_RATE

        if date_col is None or time_col is None or df.empty:
            return synthetic, _FALLBACK_EPOCH, 'UNKNOWN (no time column)'

        # Probe the first non-null time value for broken-timestamp detection
        time_series = df[time_col].dropna()
        if time_series.empty:
            return synthetic, _FALLBACK_EPOCH, 'UNKNOWN (empty time column)'

        first_time_str = str(time_series.iloc[0]).strip()
        broken = self._is_broken_timestamp(first_time_str)

        if broken:
            return synthetic, _FALLBACK_EPOCH, 'LOW (GPS fault)'

        # Combine Date + Time into a single datetime string column
        date_series = df[date_col].fillna('')
        time_series_full = df[time_col].fillna('')

        combined = (date_series + ' ' + time_series_full).str.strip()

        try:
            parsed = pd.to_datetime(combined, errors='coerce', format='mixed', dayfirst=False)
            valid = parsed.dropna()
            if valid.empty:
                return synthetic, _FALLBACK_EPOCH, 'LOW (parse failure)'

            # Convert SGT (UTC+8) → UTC
            start_ts_sgt = valid.iloc[0]
            start_ts_utc = start_ts_sgt - timedelta(hours=PMU_SGT_OFFSET_H)
            start_time = start_ts_utc.to_pydatetime().replace(tzinfo=None)

            # Compute seconds from first sample (SGT offsets cancel — raw delta OK)
            delta_s = (parsed - start_ts_sgt).dt.total_seconds()
            arr = delta_s.ffill().fillna(0.0).to_numpy(dtype=np.float64)

            return arr, start_time, 'OK'

        except Exception:
            return synthetic, _FALLBACK_EPOCH, 'LOW (parse exception)'

    def _is_broken_timestamp(self, time_str: str) -> bool:
        """Return True when time_str is a broken PMU timestamp.

        Broken format: "MM:SS.s" — only one colon (missing hours).
        Valid format:  "HH:MM:SS.mmm" — two colons.

        Args:
            time_str: Raw time string value from the time column.

        Returns:
            True if broken (GPS fault), False if valid.
        """
        return time_str.count(':') == 1

    # ── Analogue channels ─────────────────────────────────────────────────────

    def _build_analogue_channels(
        self,
        df: pd.DataFrame,
        skip_cols: set[str],
        pmu_id: int,
    ) -> list[AnalogueChannel]:
        """Create AnalogueChannel objects from data columns.

        Each column is matched against the known core PMU column set after
        stripping any bay/unit prefix.  Unrecognised numeric columns are
        included as ANALOGUE role with no scaling.  Entirely non-numeric
        columns are skipped.

        Args:
            df:        Source DataFrame.
            skip_cols: Column names to exclude (date, time).
            pmu_id:    PMU device ID (used as bay_name for all channels).

        Returns:
            List of AnalogueChannel objects, scaled and typed.
        """
        channels: list[AnalogueChannel] = []
        ch_id = 1

        for col in df.columns:
            if col in skip_cols:
                continue

            # Skip status/stat columns
            col_lower = col.strip().lower()
            if any(kw in col_lower for kw in _SKIP_KEYWORDS):
                continue

            # Parse numeric data — skip wholly non-numeric columns
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.isna().all():
                continue

            # Determine core column name (strip prefix if present)
            core_name = self._strip_prefix(col)
            core_lower = core_name.lower()

            if core_lower in _CORE_COLUMNS:
                display_name, role, phase, unit, scale = _CORE_COLUMNS[core_lower]
                raw = (numeric.fillna(0.0).to_numpy(dtype=np.float32) * scale)
                # Unwrap angle channels — phasor angles rotate continuously
                # and wrap at ±180°, producing a sawtooth that decimation
                # renders as a zebra pattern.  np.unwrap() removes the jumps.
                if unit == '°':
                    unwrapped = np.degrees(
                        np.unwrap(np.radians(raw.astype(np.float64)))
                    )
                    raw = unwrapped.astype(np.float32)
                    unit = 'deg (unwrapped)'
                confidence = RoleConfidence.HIGH
            else:
                # Unrecognised column — keep as generic analogue, no scaling
                display_name = core_name
                role = SignalRole.ANALOGUE
                phase = ''
                unit = ''
                scale = 1.0
                raw = numeric.fillna(0.0).to_numpy(dtype=np.float32)
                confidence = RoleConfidence.LOW

            ch = AnalogueChannel(
                channel_id=ch_id,
                name=display_name,
                phase=phase,
                unit=unit,
                multiplier=1.0,
                offset=0.0,
                raw_data=raw,
                signal_role=role,
                role_confidence=confidence,
                role_confirmed=(confidence == RoleConfidence.HIGH),
                bay_name='',
            )
            channels.append(ch)
            ch_id += 1

        return channels

    def _build_status_channel(
        self,
        df: pd.DataFrame,
        skip_cols: set[str],
    ) -> list[DigitalChannel]:
        """Create a DigitalChannel from the Status column if present.

        Status "00 00" = healthy (0); any non-zero value = flagged (1).

        Args:
            df:        Source DataFrame.
            skip_cols: Already-used column names set (not modified).

        Returns:
            List of zero or one DigitalChannel.
        """
        for col in df.columns:
            if col in skip_cols:
                continue
            if col.strip().lower() in ('status', 'stat'):
                # Map "00 00" → 0, anything else → 1
                data = df[col].fillna('00 00').apply(
                    lambda v: 0 if str(v).strip() in ('00 00', '0', '') else 1
                ).to_numpy(dtype=np.uint8)
                dig = DigitalChannel(
                    channel_id=1,
                    name='Status',
                    signal_role=SignalRole.DIG_GENERIC,
                    data=data,
                )
                return [dig]
        return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _strip_prefix(self, col: str) -> str:
        """Strip a leading bay/unit prefix from a column name.

        Examples::

            "KAWA1_V1 Magnitude" → "V1 Magnitude"
            "UNIT2*V1 Angle"     → "V1 Angle"
            "Frequency"          → "Frequency"

        Args:
            col: Raw column header string.

        Returns:
            Core column name with prefix removed (or original if no prefix).
        """
        m = _PREFIX_RE.match(col.strip())
        return m.group(1).strip() if m else col.strip()

    def _extract_nominal_kv(self, station_name: str) -> str:
        """Extract nominal voltage level from leading digits of station name.

        Examples::

            "500JMJG-U5"   → "500"
            "275BAHS-KAWA1" → "275"
            "UNKNOWN"       → "?"

        Args:
            station_name: Raw station name string from metadata row.

        Returns:
            Voltage level string (e.g. "500"), or "?" if not detectable.
        """
        m = _NOMINAL_KV_RE.match(station_name.strip())
        return m.group(1) if m else '?'
