# SKILL: COMTRADE Parser
# Load when: implementing src/parsers/comtrade_parser.py or debugging .cfg/.dat issues

## ── CFG LINE-BY-LINE STRUCTURE ─────────────────────────────────────────────

```
Line 1:  station_name, rec_dev_id [, rev_year]   ← rev_year OPTIONAL (NARI omits it)
Line 2:  TT, ##A, ##D                            ← total, analogue count, digital count
Lines:   [analogue channel definitions]           ← ##A lines
Lines:   [digital channel definitions]            ← ##D lines
Line:    lf                                       ← nominal frequency (50 or 60)
Line:    nrates                                   ← number of sampling rate sections
Lines:   samp,endsamp                             ← one per rate section
Line:    dd/mm/yyyy,hh:mm:ss.ssssss              ← start timestamp
Line:    dd/mm/yyyy,hh:mm:ss.ssssss              ← trigger timestamp
Line:    ASCII|BINARY|BINARY32|FLOAT32            ← DAT format
[Line:   1]                                       ← optional timestamp quality flag
```

---

## ── ANALOGUE CHANNEL LINE FORMATS ──────────────────────────────────────────

### Full format — 13 fields (BEN32, ABB, Siemens, GE, SEL)
```
ch_num, name, phase, ccbm, unit, multiplier, offset, skew, min, max, primary, secondary, PS
```
Example (BEN32):
```
6,SJTC2 IR,A,,kA,0.0025896605,0.0000000000,0,-32767,+32767,1.200000,0.001000,P
```

### Short format — 10 fields (NARI — primary/secondary/PS missing)
```
ch_num, name, phase, ccbm, unit, multiplier, offset, skew, min, max
```
Example (NARI):
```
3, Ia,,,A,0.005413,0.,0.,-32768,32767
```

### Safe parsing — handle both:
```python
parts = [p.strip() for p in line.split(',')]
ch_num     = int(parts[0])
name       = parts[1].strip()
phase      = parts[2].strip() if len(parts) > 2 else ''
ccbm       = parts[3].strip() if len(parts) > 3 else ''
unit       = parts[4].strip() if len(parts) > 4 else ''
multiplier = float(parts[5])  if len(parts) > 5 else 1.0
offset     = float(parts[6])  if len(parts) > 6 else 0.0
skew       = float(parts[7])  if len(parts) > 7 else 0.0
min_val    = float(parts[8])  if len(parts) > 8 else -32768.0
max_val    = float(parts[9])  if len(parts) > 9 else 32767.0
primary    = float(parts[10]) if len(parts) > 10 else None
secondary  = float(parts[11]) if len(parts) > 11 else None
ps_flag    = parts[12].strip() if len(parts) > 12 else 'S'
```

---

## ── DIGITAL CHANNEL LINE FORMATS ────────────────────────────────────────────

### Full format — 5 fields (BEN32, ABB, Siemens)
```
ch_num, name, phase, ccbm, normal_state
```
Example (BEN32):
```
24,SJTC2 CB_R OPEN,,,0
```

### Short format — 3 fields (NARI)
```
ch_num, name, normal_state
```
Example (NARI):
```
6,Trip,0
```

### Safe parsing — handle both:
```python
parts = [p.strip() for p in line.split(',')]
ch_num       = int(parts[0])
name         = parts[1].strip()
if len(parts) == 3:
    normal_state = int(parts[2])      # NARI short format
elif len(parts) >= 5:
    normal_state = int(parts[4])      # full format
else:
    normal_state = 0                  # safe default
```

---

## ── PHYSICAL VALUE CONVERSION (ALWAYS apply both) ───────────────────────────

```python
physical_value = (raw_integer * multiplier) + offset
```

CRITICAL: offset is NON-ZERO for DC and mechanical channels.
Never skip offset. Apply uniformly to every channel.

Examples from real files:
```
BEN32 current:   physical = raw × 0.0025896605 + 0.0    (AC — zero offset)
BEN32 voltage:   physical = raw × 0.0151063530 + 0.0    (AC — zero offset)
BEN32 field I:   physical = raw × 0.2862979770 + (-1875.000)  (DC — NON-ZERO)
BEN32 field V:   physical = raw × 0.0687115118 + (-1350.000)  (DC — NON-ZERO)
BEN32 RPM:       physical = raw × 0.1526922435 + (-1000.000)  (mechanical)
BEN32 valve %:   physical = raw × 0.0038173061 + (-25.000)    (mechanical)
NARI current:    physical = raw × 0.005413     + 0.0
NARI voltage:    physical = raw × 0.019134     + 0.0
```

---

## ── DAT FILE FORMATS ────────────────────────────────────────────────────────

| Format   | Sample word | Timestamp word | Notes                       |
|----------|-------------|----------------|-----------------------------|
| ASCII    | comma-sep   | microseconds   | BEN32 typical, NARI typical |
| BINARY   | int16       | uint32 µs      | Most IED COMTRADE           |
| BINARY32 | int32       | uint32 µs      | Higher resolution           |
| FLOAT32  | float32     | uint32 µs      | Modern IEDs                 |

Binary record per sample:
```
[uint32 sample_number][uint32 timestamp_µs][int16 × n_analogue][uint16 × ceil(n_digital/16)]
```

Digital channel extraction from uint16 word:
```python
word_index = digital_ch_index // 16
bit_index  = digital_ch_index % 16
value = (digital_word[word_index] >> bit_index) & 1
```

---

## ── REVISION YEAR HANDLING ──────────────────────────────────────────────────

```python
VALID_REV_YEARS = {'1991', '1999', '2013'}

def parse_rev_year(raw: str) -> str:
    raw = raw.strip()
    if raw in VALID_REV_YEARS:
        return raw
    # NARI: field missing entirely → default 1999
    if not raw:
        return '1999'
    # BEN32 quirk: actual calendar year written instead of standard year
    # e.g. "2005", "2024", "2025" → treat as 1999 format
    try:
        year = int(raw)
        if year > 2013:
            return '1999'   # non-standard year → safest default
    except ValueError:
        pass
    return '1999'
```

---

## ── MULTI-RATE TIME ARRAY CONSTRUCTION ─────────────────────────────────────

NARI and COMTRADE 2013 use multiple sampling rate sections.
Time array is NON-UNIFORM across section boundaries.

```python
def build_time_array(sample_rate_sections: list[dict]) -> np.ndarray:
    """
    sample_rate_sections: [{'rate': 1200.0, 'end_sample': 192},
                           {'rate': 600.0,  'end_sample': 2992}]
    Returns float64 time array in seconds from record start.
    """
    segments = []
    t_current = 0.0
    prev_end  = 0

    for section in sample_rate_sections:
        n  = section['end_sample'] - prev_end
        dt = 1.0 / section['rate']
        t  = t_current + np.arange(n, dtype=np.float64) * dt
        segments.append(t)
        t_current = float(t[-1]) + dt
        prev_end  = section['end_sample']

    return np.concatenate(segments)
```

Real example from NARI file:
```
Section 1: rate=1200, end=192   → 192 samples at 1/1200 = 0.000833s each
Section 2: rate=600,  end=2992  → 2800 samples at 1/600 = 0.001667s each
Total duration ≈ 0.16 + 4.67 = 4.83 seconds
```

---

## ── BAY EXTRACTION FROM CHANNEL NAMES ──────────────────────────────────────

```python
def extract_bay_from_analogue_name(name: str) -> tuple[str, str]:
    """
    BEN32 format: "BAYNAME SIGNALCODE"
    Returns (bay_name, signal_code)
    Examples:
      "SJTC2 VR"       → ("SJTC2", "VR")
      "KPAR IY"        → ("KPAR", "IY")
      "IR UNIT NO.2"   → ("UNIT NO.2", "IR")   ← name-first variant
      "FREQ UR UNIT NO.2" → complex — detect by signal code position
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return ('', name)

    last = parts[-1].upper()
    # Standard BEN32: last token is the signal code
    SIGNAL_CODES = {'VR','VY','VB','VN','IR','IY','IB','IN',
                    'VA','VB','VC','IA','IB','IC',
                    'UA','UB','UC','Ua','Ub','Uc',
                    'Ia','Ib','Ic'}
    if last in SIGNAL_CODES:
        bay    = ' '.join(parts[:-1])
        signal = parts[-1]
        return (bay, signal)

    # Name-first variant: first token is signal type
    SIGNAL_TYPES = {'FREQ','POWER','R.POWER','FIELD','MECH','GOVERNOR',
                    'O','ZERO','NEG','POS'}
    if parts[0].upper() in SIGNAL_TYPES:
        return (' '.join(parts[1:]), parts[0])

    return ('', name)


def extract_bay_from_digital_name(name: str, known_bays: set[str]) -> str:
    """
    Digital channel names have bay name ANYWHERE in string.
    "OVER UR SJTC NO.2" → bay="SJTC2"
    "SJTC2 87L/1"       → bay="SJTC2"
    Match against known_bays populated from analogue channel parsing.
    """
    name_upper = name.upper()
    for bay in known_bays:
        if bay.upper() in name_upper:
            return bay
    return ''
```

---

## ── COMPLETE PARSER CLASS STRUCTURE ─────────────────────────────────────────

```python
class ComtradeParser:
    def load(self, filepath: Path) -> DisturbanceRecord:
        cfg_path = self._find_cfg(filepath)
        dat_path = self._find_dat(filepath)

        cfg = self._parse_cfg(cfg_path)
        raw = self._parse_dat(dat_path, cfg)

        analogue_channels = self._build_analogue_channels(cfg, raw)
        digital_channels  = self._build_digital_channels(cfg, raw)

        # Extract bay names from analogue channels first
        known_bays = {ch.bay_name for ch in analogue_channels if ch.bay_name}

        # Apply bay names to digital channels using known_bays
        for ch in digital_channels:
            ch.bay_name = extract_bay_from_digital_name(ch.name, known_bays)

        # Run auto-detection (signal_role_detector.py)
        from parsers.signal_role_detector import detect_signal_roles
        analogue_channels = detect_signal_roles(analogue_channels)
        digital_channels  = detect_signal_roles(digital_channels)

        # Determine display mode
        primary_rate = cfg['sample_rate_sections'][0]['rate']
        display_mode = 'WAVEFORM' if primary_rate >= 200 else 'TREND'

        return DisturbanceRecord(
            station_name      = cfg['station_name'],
            device_id         = cfg['rec_dev_id'],
            start_time        = cfg['start_time'],
            trigger_time      = cfg['trigger_time'],
            trigger_sample    = cfg['trigger_sample'],
            sample_rate       = primary_rate,
            nominal_frequency = cfg['nominal_frequency'],
            display_mode      = display_mode,
            analogue_channels = analogue_channels,
            digital_channels  = digital_channels,
            time_array        = build_time_array(cfg['sample_rate_sections']),
            source_format     = f"COMTRADE_{cfg['rev_year']}",
            file_path         = filepath,
            header_text       = self._read_hdr(cfg_path),
        )
```

---

## ── VALIDATION TESTS (one per real file seen) ──────────────────────────────

```python
# Test 1 — BEN32 fast (KULW_275kV)
r = ComtradeParser().load(Path('tests/test_data/KULW_275kV.cfg'))
assert r.station_name == 'KULW_275kV'
assert r.device_id == '1215'
assert len(r.analogue_channels) == 28
assert len(r.digital_channels) == 104
assert r.sample_rate == 5000.0
assert r.display_mode == 'WAVEFORM'
assert r.analogue_channels[0].bay_name == 'SJTC2'
assert r.analogue_channels[0].signal_role == 'V_PHASE'
assert r.analogue_channels[0].phase == 'C'        # VB = Blue = C
assert r.analogue_channels[0].primary == 275.0
assert r.analogue_channels[0].ps_flag == 'P'
# Physical value sanity: pre-fault voltage ≈ 275kV / sqrt(3) ≈ 158.8 kV
prefault = r.analogue_channels[0].raw_data[:100]
assert 130.0 < np.max(np.abs(prefault)) < 250.0

# Test 2 — BEN32 slow (JMHE U2)
r2 = ComtradeParser().load(Path('tests/test_data/JMHE_U2.cfg'))
assert r2.sample_rate == 50.0
assert r2.display_mode == 'TREND'
assert len(r2.analogue_channels) == 15
# Field current has non-zero offset — verify applied
field_ch = next(c for c in r2.analogue_channels if 'FIELD CURRENT' in c.name)
assert field_ch.offset == pytest.approx(-1875.0, abs=1.0)
assert field_ch.signal_role == 'DC_FIELD_I'
# MW channel detected as pre-calculated (is_derived=True)
mw_ch = next(c for c in r2.analogue_channels if c.signal_role == 'P_MW')
assert mw_ch.is_derived == True

# Test 3 — NARI relay
r3 = ComtradeParser().load(Path('tests/test_data/NARI_relay.cfg'))
assert r3.device_id == '0'
assert len(r3.analogue_channels) == 9
assert len(r3.digital_channels) == 32
# Multi-rate: time array should be non-uniform
dt = np.diff(r3.time_array)
assert not np.allclose(dt, dt[0], rtol=0.01)   # NOT uniform — rate changes
# Chinese naming detection
ia = next(c for c in r3.analogue_channels if c.name.strip() == 'Ia')
assert ia.signal_role == 'I_PHASE'
assert ia.phase == 'A'
zero_seq = next(c for c in r3.analogue_channels if '3I0' in c.name)
assert zero_seq.signal_role == 'I_EARTH'
# NARI: primary/secondary should be None (not in CFG)
assert ia.primary is None
```
