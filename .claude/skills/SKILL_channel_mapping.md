# SKILL: Channel Signal Mapping & Auto-Detection
# Load when: signal_role_detector.py, channel_mapping_dialog.py, or any signal role assignment

## ── COMPLETE SIGNAL ROLE TAXONOMY ───────────────────────────────────────────

```python
ANALOGUE_ROLES = {
    'V_PHASE':    ('Voltage — Phase-to-earth',  ['A','B','C'],       'kV'),
    'V_LINE':     ('Voltage — Phase-to-phase',  ['AB','BC','CA'],    'kV'),
    'V_RESIDUAL': ('Voltage — Residual/Zero',   ['N',''],            'kV'),
    'I_PHASE':    ('Current — Phase',           ['A','B','C'],       'A'),
    'I_EARTH':    ('Current — Earth/Neutral',   ['N',''],            'A'),
    'V1_PMU':     ('Voltage — Pos Seq (PMU)',   ['Pos-seq'],         'kV'),
    'I1_PMU':     ('Current — Pos Seq (PMU)',   ['Pos-seq'],         'A'),
    'P_MW':       ('Active Power',              ['3-phase',''],      'MW'),
    'Q_MVAR':     ('Reactive Power',            ['3-phase',''],      'MVAR'),
    'FREQ':       ('System Frequency',          [''],                'Hz'),
    'ROCOF':      ('Rate of Change of Freq',    [''],                'Hz/s'),
    'DC_FIELD_I': ('Field Current (DC)',        [''],                'A'),
    'DC_FIELD_V': ('Field Voltage (DC)',        [''],                'V'),
    'MECH_SPEED': ('Mechanical Shaft Speed',    [''],                'RPM'),
    'MECH_VALVE': ('Valve/Gate Position',       [''],                '%'),
    'SEQ_RMS':    ('Sequence Component RMS',    ['pos','neg','zero'],'kV'),
    'ANALOGUE':   ('Generic — Unknown',         [''],                ''),
}

DIGITAL_ROLES = {
    'DIG_TRIP':      'Protection Trip Output',
    'DIG_CB':        'Circuit Breaker / Switch Status',
    'DIG_PICKUP':    'Protection Pickup / Alarm Threshold',
    'DIG_AR':        'Auto-Reclose (attempt/block/lockout)',
    'DIG_INTERTRIP': 'Teleprotection / Breaker-Fail / Inter-trip',
    'DIG_TRIGGER':   'External Trigger Input',
    'DIG_GENERIC':   'Alarm / Supervision / Comms-fail / Enable',
}
```

---

## ── AUTO-DETECTION ENGINE ────────────────────────────────────────────────────

Detection applies rules in STRICT PRIORITY ORDER. First match wins.
For digital channels, ALARM EXCEPTION checked FIRST before any other rule.

### ANALOGUE DETECTION

```python
import re
import numpy as np

# ── Step 0: Strip bay prefix (BEN32/Malaysian naming) ──────────────────────
def strip_bay_and_get_signal(name: str) -> str:
    """Returns just the signal code portion, uppercased."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return parts[-1].upper()   # last token = signal code
    return name.upper()

# ── Direct signal code lookup table ────────────────────────────────────────
# Built from ALL real files seen: BEN32 R/Y/B, IEC A/B/C, Chinese Ia/Ua
SIGNAL_CODE_TO_ROLE = {
    # R/Y/B convention (Malaysian/British/TNB)
    'VR': ('V_PHASE','A'),    'VY': ('V_PHASE','B'),    'VB': ('V_PHASE','C'),
    'UR': ('V_PHASE','A'),    'UY': ('V_PHASE','B'),    'UB': ('V_PHASE','C'),
    'IR': ('I_PHASE','A'),    'IY': ('I_PHASE','B'),    'IB': ('I_PHASE','C'),
    'IN': ('I_EARTH','N'),    'VN': ('V_RESIDUAL','N'), 'UN': ('V_RESIDUAL','N'),
    # IEC A/B/C convention
    'VA': ('V_PHASE','A'),    'VC': ('V_PHASE','C'),
    'IA': ('I_PHASE','A'),    'IC': ('I_PHASE','C'),
    # Chinese IEC convention (NARI) — case-sensitive lookup
    'Ia': ('I_PHASE','A'),    'Ib': ('I_PHASE','B'),    'Ic': ('I_PHASE','C'),
    'Ua': ('V_PHASE','A'),    'Ub': ('V_PHASE','B'),    'Uc': ('V_PHASE','C'),
    'ia': ('I_PHASE','A'),    'ib': ('I_PHASE','B'),    'ic': ('I_PHASE','C'),
    'ua': ('V_PHASE','A'),    'ub': ('V_PHASE','B'),    'uc': ('V_PHASE','C'),
    # Zero/residual sequence — case-insensitive
    '3I0': ('I_EARTH','N'),   '3IO': ('I_EARTH','N'),
    '3U0': ('V_RESIDUAL','N'),'3UO': ('V_RESIDUAL','N'),
    'I0':  ('I_EARTH','N'),   'U0':  ('V_RESIDUAL','N'),
    'V0':  ('V_RESIDUAL','N'),'UX':  ('V_RESIDUAL','N'),
    'Ux':  ('V_RESIDUAL','N'),'ux':  ('V_RESIDUAL','N'),
}

def detect_analogue_role(ch) -> tuple[str, str, str]:
    """
    Returns (signal_role, phase, confidence).
    Applies 8 priority rules in order.
    """
    name     = ch.name.strip()
    name_up  = name.upper()
    unit_up  = (ch.unit or '').upper().strip()

    # ── PRIORITY 1: Unit field — most reliable ──────────────────────────────
    UNIT_MAP = {
        'KV': ('V_PHASE', ''),    'V': ('V_PHASE', ''),
        'KA': ('I_PHASE', ''),    'A': ('I_PHASE', ''),
        'HZ': ('FREQ', ''),       'MW': ('P_MW', ''),
        'MVAR': ('Q_MVAR', ''),   'MVAR': ('Q_MVAR', ''),
        'MVAr': ('Q_MVAR', ''),   'RPM': ('MECH_SPEED', ''),
    }
    if unit_up in UNIT_MAP:
        role, phase = UNIT_MAP[unit_up]
        # Refine V_PHASE and I_PHASE with phase from name
        if role in ('V_PHASE', 'I_PHASE', 'V_RESIDUAL', 'I_EARTH'):
            signal = strip_bay_and_get_signal(name)
            if signal in SIGNAL_CODE_TO_ROLE:
                role, phase = SIGNAL_CODE_TO_ROLE[signal]
            else:
                phase = _extract_phase(name)
            # Check for residual/earth within V/I
            if role == 'V_PHASE' and any(k in name_up for k in
               ('RESID','ZERO','3U0','U0','V0','UX','VN','UN','O SEQ')):
                role = 'V_RESIDUAL'; phase = 'N'
            if role == 'I_PHASE' and any(k in name_up for k in
               ('NEUTR','EARTH','3I0','I0','IN','RESID')):
                role = 'I_EARTH'; phase = 'N'
        # Handle % with name check
        if unit_up == '%' or (not unit_up and any(k in name_up for k in
           ('VALVE','GATE','GOVERNOR','GUIDE VANE','WICKET','POSITION'))):
            role = 'MECH_VALVE'
        return (role, phase, 'HIGH')

    # ── PRIORITY 2: Non-zero offset signature — DC/mechanical channels ──────
    if ch.offset != 0.0:
        if any(k in name_up for k in ('FIELD CURRENT','FIELD I','IF ')):
            return ('DC_FIELD_I', '', 'HIGH')
        if any(k in name_up for k in ('FIELD VOLTAGE','FIELD V','VF ')):
            return ('DC_FIELD_V', '', 'HIGH')
        if any(k in name_up for k in ('SPEED','RPM','MECH')):
            return ('MECH_SPEED', '', 'HIGH')
        if any(k in name_up for k in ('VALVE','GATE','GOVERNOR','POSITION')):
            return ('MECH_VALVE', '', 'HIGH')

    # ── PRIORITY 3: Power channel names ────────────────────────────────────
    if any(k in name_up for k in ('POWER UNIT','ACTIVE POWER','MW')):
        return ('P_MW', '3-phase', 'HIGH')
    if any(k in name_up for k in ('R.POWER','REACTIVE','MVAR','MVAr')):
        return ('Q_MVAR', '3-phase', 'HIGH')

    # ── PRIORITY 4: Frequency channel names ────────────────────────────────
    if any(k in name_up for k in ('FREQ ','FREQUENCY','DF/DT','ROCOF')):
        if 'DF/DT' in name_up or 'ROCOF' in name_up:
            return ('ROCOF', '', 'HIGH')
        return ('FREQ', '', 'HIGH')

    # ── PRIORITY 5: Sequence RMS channels ──────────────────────────────────
    if any(k in name_up for k in ('O SEQ RMS','ZERO SEQ','NEG SEQ',
                                   'POS SEQ','SEQ RMS','0SEQ')):
        return ('SEQ_RMS', 'zero', 'HIGH')

    # ── PRIORITY 6: Direct signal code lookup (case-sensitive first) ────────
    signal_raw = name.strip().split()[-1]        # last token, original case
    signal_up  = strip_bay_and_get_signal(name)  # last token, uppercased
    if signal_raw in SIGNAL_CODE_TO_ROLE:
        role, phase = SIGNAL_CODE_TO_ROLE[signal_raw]
        return (role, phase, 'HIGH')
    if signal_up in SIGNAL_CODE_TO_ROLE:
        role, phase = SIGNAL_CODE_TO_ROLE[signal_up]
        return (role, phase, 'HIGH')

    # ── PRIORITY 7: Pattern matching on full name ───────────────────────────
    CURRENT_PATTERNS = [r'^I[_\-]?[RYBABC]', r'^IL\d', r'CURR', r'AMPERE']
    VOLTAGE_PATTERNS = [r'^[VU][_\-]?[RYBABC]', r'^[VU]\d', r'VOLT']
    for pat in CURRENT_PATTERNS:
        if re.search(pat, name_up):
            return ('I_PHASE', _extract_phase(name), 'MEDIUM')
    for pat in VOLTAGE_PATTERNS:
        if re.search(pat, name_up):
            return ('V_PHASE', _extract_phase(name), 'MEDIUM')

    # ── PRIORITY 8: Magnitude heuristic (last resort) ───────────────────────
    if hasattr(ch, 'raw_data') and ch.raw_data is not None and len(ch.raw_data):
        maxabs = float(np.nanmax(np.abs(ch.raw_data)))
        if maxabs > 1000:
            return ('V_PHASE', '', 'LOW')
        if 0.1 < maxabs < 50:
            return ('I_PHASE', '', 'LOW')

    return ('ANALOGUE', '', 'LOW')


def _extract_phase(name: str) -> str:
    """Extract phase from name using R/Y/B and A/B/C conventions."""
    n = name.upper().strip()
    last = n.split()[-1] if n.split() else n

    # R/Y/B convention
    if last.endswith('R') or '_R' in n or 'PHASE R' in n or ' R ' in n:
        return 'A'
    if last.endswith('Y') or '_Y' in n or 'PHASE Y' in n or ' Y ' in n:
        return 'B'
    if last.endswith('B') and 'IB' not in last[-3:]:  # avoid false IB match
        return 'C'

    # A/B/C convention
    PHASE_ENDINGS = {'A':'A', 'B':'B', 'C':'C', 'a':'A', 'b':'B', 'c':'C'}
    if len(last) >= 2 and last[-1] in PHASE_ENDINGS:
        return PHASE_ENDINGS[last[-1]]

    # Numeric L1/L2/L3
    if last.endswith('1') or 'L1' in n: return 'A'
    if last.endswith('2') or 'L2' in n: return 'B'
    if last.endswith('3') or 'L3' in n: return 'C'

    # Neutral/earth
    if any(k in n for k in ('IN','UN','VN','NEUTRAL','EARTH','GROUND','N ')):
        return 'N'

    return ''
```

---

### DIGITAL DETECTION

```python
# ── ALARM EXCEPTION (check FIRST — overrides ALL other rules) ───────────────
ALARM_EXCEPTION_KEYWORDS = [
    'COMMFAIL', 'COMM FAIL', 'COMM_FAIL', '_FAIL', 'FAIL ',
    'ALARM', 'WARNING', 'MCB', 'VTS', 'BI_EN', 'BI_MCB',
    'EN_Z1', 'EN_Z', '_ENABLE', 'ENABLE_',
]

# ── DETECTION KEYWORD SETS ───────────────────────────────────────────────────

# DIG_CB keywords
CB_KEYWORDS = [
    'CB_R', 'CB_Y', 'CB_B', 'CB_R OPEN', 'CB_Y OPEN', 'CB_B OPEN',
    'CB OPEN', 'CB CLOSE', 'CB_OPEN', 'CB_CLOSE',
    'GCB OPEN', 'GCB CLOSED', 'GCB_OPEN', 'GCB_CLOSED',   # generator CB
    'FCB OPEN', 'FCB CLOSED', 'FCB_OPEN', 'FCB_CLOSED',   # field CB
    'BI_52B', 'BI_52A', '52B', '52A',
]

# DIG_TRIP keywords (protection function codes — all relay vendors)
TRIP_KEYWORDS = [
    # Differential protection
    '87L', '87T', '87B', '87G', '87N', '87M',
    # Distance protection
    '21ZBU', 'OP_Z1', 'OP_Z2', 'OP_Z3', 'OP_Z_REV',
    'M1 Z1', 'M2 Z1', 'M1 21ZBU', 'M2 21ZBU',
    'M1 GEN', 'M1 GENERAL', 'M2 GEN', 'GENERAL TRIP',
    'DELAY TRIP', 'M1 DELAY', 'M2 DELAY',
    # Overcurrent / ROC / OC
    'OP_ROC', 'OP_OC', 'OP_OVLD', 'TOL STAGE',
    'BACKUP TOL', 'THERMAL OL', 'OVLD_TRP',
    # Switch-onto-fault
    'SOTF', 'OP_Z_SOTF', 'OP_ROC_SOTF', 'M1/M2_SOTF', 'M1/M2 SOTF',
    # Teleprotection assisted
    'OP_Z_TELEP', 'OP_ROC_TELEP', 'OP_WE_POTT', 'OP_ROC_POTT', 'OP_Z_TeleP',
    # Generic trip labels
    'TRIP', 'OPERATED', 'GEN TRIP',
    # NARI prefix
    'OP_',
]

# DIG_PICKUP keywords
PICKUP_KEYWORDS = [
    'OVER ', 'UNDER ', 'OVERVOLTAGE', 'UNDERVOLTAGE', 'OVERCURRENT',
    'FD,', ' FD', '_FD', 'FAULT DET', 'FAULT DETECT',
    'PICKUP', 'PICK UP', 'START ', 'ELEMENT START',
    'AR_INPROGRESS', 'AR INPROG',
    'VEBI_DISTP', 'VEBI_ROC',
    # NARI binary input monitoring
    'BI_EXTRP',
]

# DIG_AR keywords
AR_KEYWORDS = [
    '79AR', '79 AR', 'AR_BLOCK', 'AR_ATTEMPTED', 'AR ATTEMPTED',
    'AR_UNSUCCESSFUL', 'AR UNSUCCESSFUL', '79AR_L/O', 'AR_L/O',
    'AR_LOCKOUT', 'AR LOCKOUT', 'RECLOSE', 'AR_INPROGRESS',
    'AUTORECLOSE', 'AUTO RECLOSE',
]

# DIG_INTERTRIP keywords
INTERTRIP_KEYWORDS = [
    '50BF', 'BF_SEND', 'BF_REC', 'BF_STG', 'BF_INTR',
    '50BF_STG1', '50BF_STG2', '50BF_INTR', 'BF STAGE',
    'SEND1', 'RECV1', 'SEND2', 'RECV2', 'SEND3', 'RECV3',
    'INTERTRIP', 'INTER-TRIP', 'DIRECT TRIP', 'REMOTE TRIP',
    'DIRECT TRIP FROM REMOTE', 'INTERTRIP RECEIVE',
    '_CS', '_CR', '_REC', 'TELEPROTECT',
    '87L/1/2_L1', '87L/1/2_L2', '87L/1/2_L3',
    'DEF M1/M2', 'POTT',
]

# DIG_TRIGGER keywords
TRIGGER_KEYWORDS = ['TRIGGER', 'TRIG ', 'EXT TRIG', 'EXTERNAL TRIG']


def detect_digital_role(ch) -> tuple[str, str]:
    """Returns (signal_role, confidence)."""
    name_up = ch.name.upper().strip()

    # ── ALARM EXCEPTION — must check first ──────────────────────────────────
    if any(k in name_up for k in ALARM_EXCEPTION_KEYWORDS):
        return ('DIG_GENERIC', 'HIGH')

    # ── CB detection ────────────────────────────────────────────────────────
    if any(k in name_up for k in CB_KEYWORDS):
        return ('DIG_CB', 'HIGH')

    # ── AR detection ────────────────────────────────────────────────────────
    if any(k in name_up for k in AR_KEYWORDS):
        return ('DIG_AR', 'HIGH')

    # ── Intertrip / BF detection ────────────────────────────────────────────
    if any(k in name_up for k in INTERTRIP_KEYWORDS):
        return ('DIG_INTERTRIP', 'HIGH')

    # ── Trigger detection ───────────────────────────────────────────────────
    if any(k in name_up for k in TRIGGER_KEYWORDS):
        return ('DIG_TRIGGER', 'HIGH')

    # ── Trip detection (must come AFTER specific checks above) ──────────────
    if any(k in name_up for k in TRIP_KEYWORDS):
        return ('DIG_TRIP', 'HIGH')

    # ── Pickup detection ────────────────────────────────────────────────────
    if any(k in name_up for k in PICKUP_KEYWORDS):
        return ('DIG_PICKUP', 'MEDIUM')

    # ── No match ────────────────────────────────────────────────────────────
    return ('DIG_GENERIC', 'LOW')
```

---

## ── COMPLEMENTARY DIGITAL PAIR DETECTION ───────────────────────────────────

```python
def detect_complementary_cb_pairs(digital_channels: list) -> list[tuple]:
    """
    Finds GCB OPEN + GCB CLOSED pairs (and similar).
    Returns list of (open_ch_id, closed_ch_id) tuples.
    These should be displayed as one state bar on event timeline.
    """
    pairs = []
    cb_channels = [c for c in digital_channels if c.signal_role == 'DIG_CB']

    for i, ch_open in enumerate(cb_channels):
        if 'OPEN' not in ch_open.name.upper():
            continue
        base = ch_open.name.upper().replace('OPEN','').strip()
        for ch_close in cb_channels[i+1:]:
            if 'CLOSED' in ch_close.name.upper() or 'CLOSE' in ch_close.name.upper():
                base2 = ch_close.name.upper().replace('CLOSED','').replace('CLOSE','').strip()
                if base == base2:
                    pairs.append((ch_open.channel_id, ch_close.channel_id))
    return pairs
```

---

## ── WHEN TO SHOW CHANNEL MAPPING DIALOG ─────────────────────────────────────

```python
def needs_mapping_dialog(record, source_type: str) -> bool:
    """
    Returns True if the channel mapping dialog should be shown automatically.
    SIGRA-equivalent manual fallback — shown when auto-detect is insufficient.
    """
    # Always show for generic CSV and Excel — no naming standard exists
    if source_type in ('CSV', 'EXCEL'):
        return True

    # Always show for PMU CSV when column detection was not HIGH confidence
    if source_type == 'PMU_CSV':
        return any(not hasattr(ch, 'role_confirmed') or not ch.role_confirmed
                   for ch in record.analogue_channels)

    # Show for COMTRADE when >20% of analogue channels are LOW confidence
    if source_type.startswith('COMTRADE'):
        analogue = record.analogue_channels
        if not analogue:
            return False
        low = sum(1 for c in analogue if c.role_confidence == 'LOW')
        return (low / len(analogue)) > 0.20

    return False

# Never show for:
# - Well-formed COMTRADE from known vendors (BEN32, NARI, ABB etc)
#   where all channels detected at HIGH confidence
# - Files matching a previously saved mapping profile
```

---

## ── MAPPING PROFILE STORAGE ──────────────────────────────────────────────────

```python
# Profile key: station_name + device_id (for COMTRADE)
#              filename stem pattern (for CSV/Excel)
# Storage:     JSON files in AppData/PowerWaveAnalyst/profiles/
# Format:
{
    "profile_name": "BEN32 KULW Standard",
    "key": "KULW_275kV_1215",
    "created": "2025-08-25",
    "mappings": [
        {"original_name": "SJTC2 VR", "signal_role": "V_PHASE",
         "phase": "A", "unit": "kV", "bay": "SJTC2"},
        {"original_name": "SJTC2 IR", "signal_role": "I_PHASE",
         "phase": "A", "unit": "kA", "bay": "SJTC2"},
    ]
}
# On load: match by key → auto-apply → show notification banner not dialog
# Profile manager: Edit → Channel Mapping Profiles → list/rename/delete
```

---

## ── DOWNSTREAM USAGE OF SIGNAL ROLE ────────────────────────────────────────

| Consumer | Uses signal_role for |
|---|---|
| Colour coding | V_PHASE → R(A)/Y(B)/B(C). I_PHASE same. DC → grey. FREQ → cyan |
| Display mode | DC_FIELD_I/V, MECH_SPEED/VALVE → always TREND regardless of sample_rate |
| RMS calculator | V_PHASE → full-cycle. I_PHASE → half-cycle. Others → excluded |
| Phasor calculator | Only V_PHASE and I_PHASE included (3-phase sets only) |
| Symmetrical components | Groups V_PHASE A+B+C. Groups I_PHASE A+B+C. Warns if incomplete |
| Power calculator | Matches V_PHASE+I_PHASE by phase. Uses pre-calc if is_derived=True |
| Frequency tracker | First FREQ channel. Falls back to first V_PHASE channel |
| Event timeline | DIG_TRIP=red, DIG_CB=orange, DIG_AR=blue, DIG_INTERTRIP=magenta |
| Complementary CB | GCB/FCB open+closed pairs → single state bar |
| Measurement panel | Unit label from role definition. Groups by signal type |
| PDF report | Role names used, not raw channel names. Sections auto-labelled |
| Bay grouping | bay_name used to group channels in channel panel and canvas |
