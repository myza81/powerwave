"""
src/parsers/comtrade_parser.py

COMTRADE (IEEE/IEC 60255-24) parser — CFG + DAT loader.
Produces a single DisturbanceRecord from any conforming .cfg/.dat pair.

Supports:
  Revision years : 1991, 1999, 2013 (non-standard calendar year → '1999')
  Analogue lines : 10-field (NARI short) and 13-field (BEN32 / ABB / Siemens)
  Digital lines  : 3-field (NARI short) and 5-field (BEN32 / ABB / Siemens)
  DAT formats    : ASCII (Step 1 complete), BINARY / BINARY32 / FLOAT32 (Step 2)
  Multi-rate     : one or more rate sections; NON-UNIFORM time array built from CFG
  Encoding       : UTF-8 with Latin-1 fallback
  Line endings   : LF and CRLF

Architecture: Data layer (parsers/) — imports models/ only.
              Never import from ui/ or engine/ here (LAW 1).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from src.models.channel import AnalogueChannel, DigitalChannel
from src.models.disturbance_record import DisturbanceRecord

# ── Module-level constants ────────────────────────────────────────────────────

VALID_REV_YEARS: frozenset[str] = frozenset({'1991', '1999', '2013'})
DEFAULT_REV_YEAR: str = '1999'

# When CFG rate section is 0 (COMTRADE variable-rate), use this placeholder.
# Step 2 will override sample_rate from DAT timestamps.
DEFAULT_VARIABLE_RATE: float = 50.0  # Hz

# BEN32 suffix signal codes: the last token of "BAYNAME SIGNALCODE"
_SUFFIX_SIGNAL_CODES: frozenset[str] = frozenset({
    'VR', 'VY', 'VB', 'VN',
    'IR', 'IY', 'IB', 'IN',
    'VA', 'VC',
    'IA', 'IC',
    'UA', 'UB', 'UC',
    'Ua', 'Ub', 'Uc',
    'Ia', 'Ib', 'Ic',
})

# BEN32 name-first signal-type keywords: "SIGNAL BAYNAME ..."
_NAME_FIRST_TYPES: frozenset[str] = frozenset({
    'FREQ', 'POWER', 'R.POWER', 'FIELD', 'MECH', 'GOVERNOR',
    'O', 'ZERO', 'NEG', 'POS',
})

# Pattern for BEN32 phase codes that may appear between signal type and bay name.
# Examples: UR, UY, UB, IR, IY, IB (2-char: [UIV][RYBNAC])
_PHASE_CODE_RE = re.compile(r'^[UIVuiv][RYBNACrybnaс]$')


# ── Helper functions ──────────────────────────────────────────────────────────

def parse_rev_year(raw: str) -> str:
    """Return a valid COMTRADE revision year string.

    Rules:
    - '1991', '1999', '2013' → returned as-is
    - Missing (NARI omits the field) → '1999'
    - Non-standard calendar year (BEN32 writes e.g. '2005', '2025') → '1999'
    """
    raw = raw.strip()
    if raw in VALID_REV_YEARS:
        return raw
    if not raw:
        return DEFAULT_REV_YEAR
    try:
        if int(raw) > 2013:
            return DEFAULT_REV_YEAR
    except ValueError:
        pass
    return DEFAULT_REV_YEAR


def extract_bay_from_analogue_name(name: str) -> tuple[str, str]:
    """Extract (bay_name, signal_code) from a BEN32 analogue channel name.

    Two BEN32 naming patterns:

    1. Suffix format  — "BAYNAME SIGNALCODE" (most channels)
       "SJTC2 VR"   → ("SJTC2", "VR")
       "LGNG1 IB"   → ("LGNG1", "IB")

    2. Name-first format — "SIGNALTYPE [PHASECODE] BAYNAME ..."
       "FREQ PGPS 2"      → ("PGPS 2",  "FREQ")
       "FREQ UY PLTG 1"   → ("PLTG 1",  "FREQ UY")   ← phase code stripped
       "POWER PGPS 1"     → ("PGPS 1",  "POWER")

    Returns ('', name) when no BEN32 structure is detected.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return ('', name.strip())

    # ── Suffix format: last token is a recognised signal code ────────────────
    last = parts[-1]
    if last in _SUFFIX_SIGNAL_CODES or last.upper() in _SUFFIX_SIGNAL_CODES:
        bay = ' '.join(parts[:-1])
        return (bay, last)

    # ── Name-first format: first token is a signal-type keyword ──────────────
    if parts[0].upper() in _NAME_FIRST_TYPES:
        signal = parts[0]
        remaining = parts[1:]
        # Optional 2-char phase code directly after signal type (e.g. UY, IR)
        if remaining and _PHASE_CODE_RE.match(remaining[0]):
            signal = f"{parts[0]} {remaining[0]}"
            remaining = remaining[1:]
        bay = ' '.join(remaining)
        return (bay, signal)

    return ('', name.strip())


def extract_bay_from_digital_name(name: str, known_bays: set[str]) -> str:
    """Find the bay name within a BEN32 digital channel name.

    BEN32 digital names embed the bay name ANYWHERE, e.g.:
      "OVER UR SJTC NO.2" → bay matches "SJTC2" (fuzzy)
      "SJTC2 87L/1"       → bay matches "SJTC2" (prefix)

    Matches against ``known_bays`` populated from analogue channel parsing.
    Returns '' when no match is found.
    """
    name_upper = name.upper()
    for bay in known_bays:
        if bay.upper() in name_upper:
            return bay
    return ''


def build_time_array(rate_sections: list[dict]) -> np.ndarray:
    """Build a float64 time array (seconds from record start) from CFG rate sections.

    Args:
        rate_sections: list of {'rate': float, 'end_sample': int}
            ``rate`` is in Hz; ``end_sample`` is the cumulative sample index
            at the end of that section (as written in the CFG).

    Returns:
        float64 ndarray.  For variable-rate files (any section has rate == 0)
        the array is left empty — timestamps must be built from DAT in Step 2.

    The returned array is NON-UNIFORM when multiple sections have different rates
    (NARI multi-rate, COMTRADE 2013).
    """
    if any(s['rate'] == 0.0 for s in rate_sections):
        return np.array([], dtype=np.float64)

    segments: list[np.ndarray] = []
    t_current = 0.0
    prev_end = 0

    for section in rate_sections:
        n = section['end_sample'] - prev_end
        if n <= 0:
            continue
        dt = 1.0 / section['rate']
        t = t_current + np.arange(n, dtype=np.float64) * dt
        segments.append(t)
        t_current = float(t[-1]) + dt
        prev_end = section['end_sample']

    return np.concatenate(segments) if segments else np.array([], dtype=np.float64)


def _parse_timestamp(date_str: str, time_str: str) -> datetime:
    """Parse a COMTRADE date + time string pair into a datetime.

    Handles:
    - Standard COMTRADE date format  DD/MM/YYYY
    - Non-standard NARI date format  MM/DD/YY   (auto-detected when middle
      component > 12, since it cannot be a valid month)
    - Microseconds with 1–6 fractional digits (zero-padded / truncated to 6)
    """
    d_parts = date_str.strip().split('/')
    p1, p2, p3 = int(d_parts[0]), int(d_parts[1]), int(d_parts[2])

    # If the middle component exceeds 12 it cannot be a month → MM/DD/YY
    if p2 > 12:
        month, day, year = p1, p2, p3
    else:
        day, month, year = p1, p2, p3

    # Two-digit year: 00–69 → 2000s, 70–99 → 1900s
    if year < 100:
        year += 2000 if year < 70 else 1900

    t_parts = time_str.strip().split(':')
    hour = int(t_parts[0])
    minute = int(t_parts[1])
    sec_frac = t_parts[2].split('.')
    second = int(sec_frac[0])
    microsecond = 0
    if len(sec_frac) > 1:
        frac = sec_frac[1].ljust(6, '0')[:6]
        microsecond = int(frac)

    return datetime(year, month, day, hour, minute, second, microsecond)


# ── ComtradeParser ────────────────────────────────────────────────────────────

class ComtradeParser:
    """Parser for COMTRADE (.cfg + .dat) disturbance record files.

    Step 1: CFG metadata and channel skeleton (raw_data / data arrays are empty).
    Step 2 (TODO): DAT sample reading for ASCII and binary formats.

    Usage::

        record = ComtradeParser().load(Path('tests/test_data/JMHE_500kV.cfg'))
    """

    def load(self, filepath: Path) -> DisturbanceRecord:
        """Load a COMTRADE record from ``filepath``.

        ``filepath`` may point to the .cfg, .CFG, .dat, or .DAT file —
        the sibling CFG file is located automatically.

        Returns:
            DisturbanceRecord with all metadata populated.  Channel
            ``raw_data`` / ``data`` arrays are empty (Step 1 — CFG only).
            They will be populated in Step 2 when DAT reading is added.
        """
        cfg_path = self._find_cfg(filepath)
        cfg = self._parse_cfg(cfg_path)

        analogue_channels = self._build_analogue_channels(cfg)
        digital_channels = self._build_digital_channels(cfg)

        # ── Populate bay_names from analogue channels (BEN32 multi-bay) ──────
        bay_names: list[str] = []
        seen_bays: set[str] = set()
        for ch in analogue_channels:
            if ch.bay_name and ch.bay_name not in seen_bays:
                bay_names.append(ch.bay_name)
                seen_bays.add(ch.bay_name)

        # ── Apply bay names to digital channels ───────────────────────────────
        known_bays: set[str] = set(bay_names)
        for ch in digital_channels:
            ch.bay_name = extract_bay_from_digital_name(ch.name, known_bays)

        # ── Sample rate: use first section; default when rate == 0 ────────────
        rate_sections: list[dict] = cfg['rate_sections']
        primary_rate = rate_sections[0]['rate'] if rate_sections else DEFAULT_VARIABLE_RATE
        if primary_rate == 0.0:
            primary_rate = DEFAULT_VARIABLE_RATE

        # ── Time array from CFG (empty for variable-rate files) ───────────────
        time_array = build_time_array(rate_sections)

        # ── trigger_sample: derived from timestamps where rate is known ───────
        start_time: datetime = cfg['start_time']
        trigger_time: datetime = cfg['trigger_time']
        delta = (trigger_time - start_time).total_seconds()
        trigger_sample = max(0, round(delta * primary_rate))

        source_format = f"COMTRADE_{cfg['rev_year']}"

        return DisturbanceRecord(
            station_name=cfg['station_name'],
            device_id=cfg['rec_dev_id'],
            start_time=start_time,
            trigger_time=trigger_time,
            trigger_sample=trigger_sample,
            sample_rate=primary_rate,
            nominal_frequency=cfg['nominal_frequency'],
            source_format=source_format,
            file_path=cfg_path,
            analogue_channels=analogue_channels,
            digital_channels=digital_channels,
            time_array=time_array,
            header_text=cfg.get('header_text', ''),
            bay_names=bay_names,
        )

    # ── File discovery ────────────────────────────────────────────────────────

    def _find_cfg(self, filepath: Path) -> Path:
        """Return the .cfg path for ``filepath``, handling case variants."""
        p = Path(filepath)
        if p.suffix.lower() == '.cfg' and p.exists():
            return p
        for suffix in ('.cfg', '.CFG'):
            candidate = p.with_suffix(suffix)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No .cfg file found alongside {filepath}")

    def _find_dat(self, filepath: Path) -> Path:
        """Return the .dat path for ``filepath``, handling case variants."""
        p = Path(filepath)
        for suffix in ('.dat', '.DAT'):
            candidate = p.with_suffix(suffix)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No .dat file found alongside {filepath}")

    # ── CFG reading ───────────────────────────────────────────────────────────

    def _read_lines(self, path: Path) -> list[str]:
        """Read a text file; normalise CRLF → LF; try UTF-8 then Latin-1."""
        for encoding in ('utf-8', 'latin-1'):
            try:
                text = path.read_text(encoding=encoding)
                return text.replace('\r\n', '\n').replace('\r', '\n').splitlines()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode {path} as UTF-8 or Latin-1")

    def _parse_cfg(self, cfg_path: Path) -> dict:
        """Parse the CFG file; return a metadata dict consumed by ``load``."""
        lines = self._read_lines(cfg_path)

        # Drop blank trailing lines
        while lines and not lines[-1].strip():
            lines.pop()

        idx = 0

        # ── Line 1: station_name, rec_dev_id [, rev_year] ─────────────────────
        parts = lines[idx].split(',')
        station_name = parts[0].strip()
        rec_dev_id = parts[1].strip() if len(parts) > 1 else ''
        rev_year = parse_rev_year(parts[2] if len(parts) > 2 else '')
        idx += 1

        # ── Line 2: TT, ##A, ##D ──────────────────────────────────────────────
        ch_parts = lines[idx].split(',')
        n_analogue = int(re.sub(r'[Aa]', '', ch_parts[1].strip()))
        n_digital = int(re.sub(r'[Dd]', '', ch_parts[2].strip()))
        idx += 1

        # ── Analogue channel definitions ──────────────────────────────────────
        analogue_defs: list[dict] = []
        for _ in range(n_analogue):
            analogue_defs.append(self._parse_analogue_line(lines[idx]))
            idx += 1

        # ── Digital channel definitions ───────────────────────────────────────
        digital_defs: list[dict] = []
        for _ in range(n_digital):
            digital_defs.append(self._parse_digital_line(lines[idx]))
            idx += 1

        # ── Nominal frequency (50 or 60 Hz) ───────────────────────────────────
        nominal_frequency = float(lines[idx].strip())
        idx += 1

        # ── Number of sampling-rate sections ──────────────────────────────────
        # nrates == 0 means variable rate; one "0,endsamp" line still follows.
        nrates = int(lines[idx].strip())
        idx += 1

        n_sections = max(nrates, 1)
        rate_sections: list[dict] = []
        for _ in range(n_sections):
            s_parts = lines[idx].split(',')
            rate = float(s_parts[0].strip())
            end_sample = int(s_parts[1].strip())
            rate_sections.append({'rate': rate, 'end_sample': end_sample})
            idx += 1

        # ── Start timestamp ───────────────────────────────────────────────────
        ts_parts = lines[idx].split(',', 1)
        start_time = _parse_timestamp(
            ts_parts[0],
            ts_parts[1] if len(ts_parts) > 1 else '00:00:00.000000',
        )
        idx += 1

        # ── Trigger timestamp ─────────────────────────────────────────────────
        ts_parts = lines[idx].split(',', 1)
        trigger_time = _parse_timestamp(
            ts_parts[0],
            ts_parts[1] if len(ts_parts) > 1 else '00:00:00.000000',
        )
        idx += 1

        # ── DAT format identifier ─────────────────────────────────────────────
        dat_format = lines[idx].strip().upper() if idx < len(lines) else 'ASCII'

        return {
            'station_name': station_name,
            'rec_dev_id': rec_dev_id,
            'rev_year': rev_year,
            'n_analogue': n_analogue,
            'n_digital': n_digital,
            'analogue_defs': analogue_defs,
            'digital_defs': digital_defs,
            'nominal_frequency': nominal_frequency,
            'nrates': nrates,
            'rate_sections': rate_sections,
            'start_time': start_time,
            'trigger_time': trigger_time,
            'dat_format': dat_format,
            'header_text': '',
        }

    def _parse_analogue_line(self, line: str) -> dict:
        """Safe parse of one analogue channel CFG line (10 or 13 fields).

        Field layout::

            13-field: ch_num, name, phase, ccbm, unit, mult, offset, skew,
                      min, max, primary, secondary, PS
            10-field: ch_num, name, phase, ccbm, unit, mult, offset, skew,
                      min, max
                      (NARI — primary / secondary / PS absent → default None / 'S')
        """
        parts = [p.strip() for p in line.split(',')]

        def _safe_float(parts: list[str], i: int, default: float) -> float:
            if i < len(parts) and parts[i]:
                return float(parts[i])
            return default

        def _safe_optional_float(parts: list[str], i: int):
            if i < len(parts) and parts[i]:
                return float(parts[i])
            return None

        return {
            'ch_num':     int(parts[0]),
            'name':       parts[1] if len(parts) > 1 else '',
            'phase':      parts[2] if len(parts) > 2 else '',
            'ccbm':       parts[3] if len(parts) > 3 else '',
            'unit':       parts[4] if len(parts) > 4 else '',
            'multiplier': _safe_float(parts, 5, 1.0),
            'offset':     _safe_float(parts, 6, 0.0),
            'skew':       _safe_float(parts, 7, 0.0),
            'min_val':    _safe_float(parts, 8, -32768.0),
            'max_val':    _safe_float(parts, 9,  32767.0),
            'primary':    _safe_optional_float(parts, 10),
            'secondary':  _safe_optional_float(parts, 11),
            'ps_flag':    parts[12] if len(parts) > 12 and parts[12] else 'S',
        }

    def _parse_digital_line(self, line: str) -> dict:
        """Safe parse of one digital channel CFG line (3 or 5 fields).

        Field layout::

            5-field: ch_num, name, phase, ccbm, normal_state  (BEN32 / ABB)
            3-field: ch_num, name, normal_state               (NARI short)
        """
        parts = [p.strip() for p in line.split(',')]
        ch_num = int(parts[0])
        name = parts[1] if len(parts) > 1 else ''

        if len(parts) == 3:
            # NARI short format
            phase = ''
            ccbm = ''
            normal_state = int(parts[2]) if parts[2] else 0
        elif len(parts) >= 5:
            # Full format
            phase = parts[2]
            ccbm = parts[3]
            normal_state = int(parts[4]) if parts[4] else 0
        else:
            phase = ''
            ccbm = ''
            normal_state = 0

        return {
            'ch_num':       ch_num,
            'name':         name,
            'phase':        phase,
            'ccbm':         ccbm,
            'normal_state': normal_state,
        }

    # ── Channel object builders ───────────────────────────────────────────────

    def _build_analogue_channels(self, cfg: dict) -> list[AnalogueChannel]:
        """Create AnalogueChannel instances from parsed CFG analogue definitions."""
        channels: list[AnalogueChannel] = []
        for d in cfg['analogue_defs']:
            bay_name, _ = extract_bay_from_analogue_name(d['name'])
            ch = AnalogueChannel(
                channel_id=d['ch_num'],
                name=d['name'],
                phase=d['phase'],
                unit=d['unit'],
                multiplier=d['multiplier'],
                offset=d['offset'],
                primary=d['primary'],
                secondary=d['secondary'],
                ps_flag=d['ps_flag'],
                bay_name=bay_name,
            )
            channels.append(ch)
        return channels

    def _build_digital_channels(self, cfg: dict) -> list[DigitalChannel]:
        """Create DigitalChannel instances from parsed CFG digital definitions."""
        channels: list[DigitalChannel] = []
        for d in cfg['digital_defs']:
            ch = DigitalChannel(
                channel_id=d['ch_num'],
                name=d['name'],
                phase=d['phase'],
                ccbm=d['ccbm'],
                normal_state=d['normal_state'],
            )
            channels.append(ch)
        return channels
