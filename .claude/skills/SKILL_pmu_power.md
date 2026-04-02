# SKILL: PMU Data & Power Calculations
# Load when: pmu_csv_parser.py, power_calculator.py, pmu_record.py, power_canvas.py

## ── PMU FILE STRUCTURE (Malaysian grid — confirmed real files) ───────────────

```
Row 0:  ID: 241, Station Name: 500JMJG-U5,,,,,,,    ← METADATA (not data)
Row 1:  Date,Time (Asia/Singapore),Status,...         ← ACTUAL COLUMN HEADERS
Row 2+: 10/15/25,12:00.0,00 00,49.948,...            ← DATA ROWS
```

The parser MUST find the header row dynamically — never assume row index.

---

## ── METADATA EXTRACTION ─────────────────────────────────────────────────────

```python
def extract_pmu_metadata(filepath: Path) -> dict:
    meta = {'pmu_id': '', 'station_name': '', 'header_row_index': 1,
            'nominal_voltage_kv': None}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.upper().startswith('ID:') or 'STATION NAME' in line.upper():
                for part in line.split(','):
                    p = part.strip()
                    if p.upper().startswith('ID:'):
                        meta['pmu_id'] = p.split(':',1)[1].strip()
                    elif 'STATION NAME' in p.upper():
                        meta['station_name'] = p.split(':',1)[1].strip()
            elif line.upper().startswith('DATE'):
                meta['header_row_index'] = i
                break

    # Extract nominal voltage from station name prefix digits
    # "500JMJG-U5" → 500kV,  "275TMGR-U2" → 275kV
    import re
    m = re.match(r'^(\d+)', meta['station_name'])
    if m:
        meta['nominal_voltage_kv'] = int(m.group(1))

    return meta
```

---

## ── COLUMN DETECTION WITH PREFIX STRIPPING ──────────────────────────────────

Three PMU variants confirmed — all use same core names, different prefixes:
```
PMU 241:  "V1 Magnitude"           → no prefix
PMU 254:  "UNIT2_V1 Magnitude"     → prefix "UNIT2_"
PMU 571:  "KAWA1_V1 Magnitude"     → prefix "KAWA1_"
```

```python
import re

CORE_SIGNAL_MAP = {
    'v_mag':     ['V1 MAGNITUDE','V1MAG','VMAG','V_MAG','VOLTAGE MAG','V1_MAG'],
    'v_ang':     ['V1 ANGLE','V1ANG','VANG','V_ANG','VOLTAGE ANG','V1_ANG'],
    'i_mag':     ['I1 MAGNITUDE','I1MAG','IMAG','I_MAG','CURRENT MAG','I1_MAG'],
    'i_ang':     ['I1 ANGLE','I1ANG','IANG','I_ANG','CURRENT ANG','I1_ANG'],
    'frequency': ['FREQUENCY','FREQ','HZ','F_HZ','SYS_FREQ'],
    'rocof':     ['DF/DT','ROCOF','DFDT','RATE OF CHANGE','D_FREQ'],
    'status':    ['STATUS','STAT','FLAGS','QUALITY','DATA VALID'],
    'timestamp': ['TIME','TIMESTAMP','DATETIME','TIME (ASIA/SINGAPORE)'],
    'date':      ['DATE'],
}

def detect_pmu_columns(columns: list[str]) -> dict | None:
    """
    Returns column mapping dict or None if cannot auto-detect.
    Strips WORD_ prefix before matching core signal names.
    """
    mapping = {}
    PREFIX_RE = re.compile(r'^[A-Z0-9]+_', re.IGNORECASE)

    for col in columns:
        col_upper = col.upper().strip()
        stripped  = PREFIX_RE.sub('', col_upper)   # remove e.g. "UNIT2_"

        for role, keywords in CORE_SIGNAL_MAP.items():
            if any(kw in stripped or kw in col_upper for kw in keywords):
                if role not in mapping:
                    mapping[role] = col   # store ORIGINAL column name
                break

    # Minimum required: date, timestamp, v_mag, v_ang
    required = {'date', 'timestamp', 'v_mag', 'v_ang'}
    if not required.issubset(mapping.keys()):
        return None   # → show PmuColumnMappingDialog

    return mapping
```

---

## ── TIMESTAMP PARSING ────────────────────────────────────────────────────────

```python
import pandas as pd
import numpy as np

TIMEZONE_SGT = 'Asia/Singapore'   # UTC+8 — ALL Malaysian PMU timestamps

TIME_FORMATS = [
    '%m/%d/%y %H:%M:%S.%f',    # PMU 254/571: "10/15/25 16:11:10.000"  CORRECT
    '%m/%d/%y %H:%M:%S',        # without milliseconds
    '%m/%d/%Y %H:%M:%S.%f',    # 4-digit year
    '%d/%m/%y %H:%M:%S.%f',    # day-first variant
    '%m/%d/%y %H:%M.%f',       # PMU 241 BROKEN: "10/15/25 12:00.0" — MM:SS.f
]

def parse_pmu_timestamps(date_col: pd.Series,
                          time_col: pd.Series) -> tuple[pd.DatetimeIndex, str]:
    """
    Returns (utc_timestamps, quality_flag).
    quality_flag: 'HIGH' if parsed correctly, 'LOW' if fallback used.
    """
    combined = (date_col.astype(str) + ' ' + time_col.astype(str)).str.strip()

    for i, fmt in enumerate(TIME_FORMATS):
        try:
            parsed = pd.to_datetime(combined, format=fmt, utc=False)
            parsed = parsed.dt.tz_localize(TIMEZONE_SGT, ambiguous='NaT')
            parsed = parsed.dt.tz_convert('UTC')
            quality = 'LOW' if i >= 4 else 'HIGH'   # broken format = LOW
            return parsed, quality
        except Exception:
            continue

    # Last resort — pandas infer
    parsed = pd.to_datetime(combined, infer_datetime_format=True, utc=False)
    parsed = parsed.dt.tz_localize(TIMEZONE_SGT, ambiguous='NaT').dt.tz_convert('UTC')
    return parsed, 'LOW'
```

---

## ── VOLTAGE UNIT AUTO-SCALING ───────────────────────────────────────────────

All Malaysian PMU files store voltage in RAW VOLTS, not kV.
Station name prefix gives nominal voltage for validation.

```python
def auto_scale_voltage(v_mag: np.ndarray,
                       nominal_kv: int | None) -> tuple[np.ndarray, str]:
    max_val = float(np.nanmax(np.abs(v_mag)))

    if nominal_kv is not None:
        # Check if values match nominal in Volts (within ±20%)
        if 0.80 * nominal_kv * 1000 < max_val < 1.20 * nominal_kv * 1000:
            return v_mag / 1000.0, 'kV'
        # Check if already in kV
        if 0.80 * nominal_kv < max_val < 1.20 * nominal_kv:
            return v_mag, 'kV'

    # Heuristic fallback
    if max_val > 1000:
        return v_mag / 1000.0, 'kV'
    if max_val > 1.5:
        return v_mag, 'kV'
    return v_mag, 'pu'

# Real examples:
# PMU 241: max=505453V, nominal=500kV → 505453/1000 = 505.5kV ✓
# PMU 254: max=283861V, nominal=275kV → 283861/1000 = 283.9kV ✓
# PMU 571: max=280379V, nominal=275kV → 280379/1000 = 280.4kV ✓
```

---

## ── GAP DETECTION ────────────────────────────────────────────────────────────

```python
def detect_gaps(timestamps_utc: pd.DatetimeIndex,
                expected_rate_hz: float = 50.0,
                tolerance_factor: float = 2.0) -> list[tuple[int, int]]:
    """
    Returns list of (start_idx, end_idx) for data gaps.
    A gap = time delta > tolerance_factor × expected_period.
    """
    expected_period_s = 1.0 / expected_rate_hz
    ts_s = timestamps_utc.astype(np.int64) / 1e9   # nanoseconds → seconds
    deltas = np.diff(ts_s)
    gap_mask = deltas > (expected_period_s * tolerance_factor)
    return [(int(i), int(i+1)) for i in np.where(gap_mask)[0]]
```

---

## ── POWER CALCULATIONS ───────────────────────────────────────────────────────

All functions fully vectorised — no Python loops. Work on np.ndarray inputs.

```python
import numpy as np

def calc_active_power(v_mag, v_ang_deg, i_mag, i_ang_deg) -> np.ndarray:
    """P = |V| * |I| * cos(θV - θI)  [MW when inputs in kV and kA]"""
    return v_mag * i_mag * np.cos(np.radians(v_ang_deg - i_ang_deg))

def calc_reactive_power(v_mag, v_ang_deg, i_mag, i_ang_deg) -> np.ndarray:
    """Q = |V| * |I| * sin(θV - θI)  [MVAR]  +ve=inductive/lagging"""
    return v_mag * i_mag * np.sin(np.radians(v_ang_deg - i_ang_deg))

def calc_apparent_power(v_mag, i_mag) -> np.ndarray:
    """S = |V| * |I|  [MVA]"""
    return v_mag * i_mag

def calc_power_factor(P, S) -> np.ndarray:
    """PF = P/S  range [-1,1],  NaN where S=0"""
    with np.errstate(invalid='ignore', divide='ignore'):
        pf = np.where(S != 0, P / S, np.nan)
    return np.clip(pf, -1.0, 1.0)

def calc_three_phase_power_from_pmu(v1_mag, v1_ang_deg,
                                     i1_mag, i1_ang_deg
                                     ) -> tuple[np.ndarray,...]:
    """
    Three-phase power from PMU positive sequence phasors.
    P_3ph = 3 × |V1| × |I1| × cos(θV1 - θI1)
    Returns (P_MW, Q_MVAR, S_MVA, PF)
    """
    P  = 3.0 * calc_active_power(v1_mag, v1_ang_deg, i1_mag, i1_ang_deg)
    Q  = 3.0 * calc_reactive_power(v1_mag, v1_ang_deg, i1_mag, i1_ang_deg)
    S  = 3.0 * calc_apparent_power(v1_mag, i1_mag)
    PF = calc_power_factor(P, S)
    return P, Q, S, PF
```

### Validation tests
```python
# Unity PF (angle diff=0): P=S, Q=0
v = np.array([132.0]); i = np.array([0.5])
P = calc_active_power(v, np.array([0.0]), i, np.array([0.0]))
Q = calc_reactive_power(v, np.array([0.0]), i, np.array([0.0]))
assert abs(P[0] - 66.0) < 0.01    # 132 × 0.5 × cos(0) = 66 MW
assert abs(Q[0]) < 0.001           # sin(0) = 0

# 0.85 PF lagging: angle = arccos(0.85) ≈ 31.79°
import math
ang = math.degrees(math.acos(0.85))
P2 = calc_active_power(v, np.array([0.0]), i, np.array([ang]))
S2 = calc_apparent_power(v, i)
PF = calc_power_factor(P2, S2)
assert abs(PF[0] - 0.85) < 0.001

# Pre-calculated channels (is_derived=True) — use directly, do NOT recalculate
# These come from BEN32 slow records: POWER UNIT UNIT NO.2, R.POWER UNIT NO.2
```

### CRITICAL engineering validation note
```
# Do NOT mark Phase 6 complete until power values verified against
# independent metering data for a known load condition.
# Claude Code tests verify math. Engineer verifies engineering correctness.
```

---

## ── PMU RENDERING RULES ──────────────────────────────────────────────────────

```python
# PMU data renders as scatter plot — visually distinct from COMTRADE waveforms
pmu_scatter = pg.ScatterPlotItem(
    x=time_seconds, y=values,
    size=4, pen=None, brush=pg.mkBrush(colour)
)

# Data gaps render as shaded bands — NEVER interpolate across gaps
for start_idx, end_idx in pmu_record.gap_indices:
    t0 = time_seconds[start_idx]
    t1 = time_seconds[end_idx]
    region = pg.LinearRegionItem(
        values=(t0, t1),
        brush=pg.mkBrush('#FFFFFF18'),   # very subtle grey
        movable=False
    )
    plot.addItem(region)

# LOW quality timestamp → show amber warning banner on canvas header
if pmu_record.timestamp_quality == 'LOW':
    canvas.show_quality_warning(f"PMU {pmu_record.pmu_id}: low-quality timestamp — GPS alignment unreliable")
```

---

## ── STATUS FIELD INTERPRETATION ─────────────────────────────────────────────

```python
STATUS_FLAGS = {
    '00 00': ('OK',       'HIGH'),    # all healthy
    '00 80': ('WARNING',  'MEDIUM'),  # GPS quality degraded or data flag set
}

def parse_pmu_status(status_str: str) -> tuple[str, str]:
    """Returns (status_label, quality_confidence)."""
    s = status_str.strip().upper()
    return STATUS_FLAGS.get(s, ('FLAGGED', 'LOW'))
```

---

## ── SAME-EVENT ALIGNMENT NOTE ────────────────────────────────────────────────

Confirmed from real files: PMU 571 (275BAHS, 16:12:00) and BEN32 slow
(JMHE U2, trigger 16:12:36) both record the same 15/10/2025 grid event.

For cross-source alignment:
1. PMU 571 has good GPS timestamps → use as time reference (alignment_quality=CONFIRMED)
2. BEN32 slow has GPS start time → compute absolute offset
3. BEN32 fast (if available) → same absolute offset approach
4. PMU 241 has broken timestamp → cross-correlate against PMU 571 frequency channel
