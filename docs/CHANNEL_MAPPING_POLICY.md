CHANNEL_MAPPING_POLICY.md — Powerwave Channel Naming and Signal Role Policy
PURPOSE

This document defines the authoritative policy for:

  - Channel naming normalization
  - Phase naming conventions across vendor formats
  - Analog vs digital classification rules
  - Signal role taxonomy (analog and digital)
  - Engineering signal aliases and lookup conventions
  - Canonical signal codes for common power system quantities
  - Unit normalization expectations
  - Downstream signal role usage expectations
  - Architectural boundary between ingestion and analytics

This document is the reference for the future signal role detection engine
(app/analytics/ or app/services/) and is NOT a parser-layer contract.

ARCHITECTURAL BOUNDARY

The parser layer (app/providers/) SHALL NOT assign signal roles.

Parsers are responsible for:
  - Preserving the raw channel name exactly as found in the source file
  - Storing the raw phase field (ph) exactly as found in COMTRADE CFG
  - Storing the unit field exactly as found in the source
  - Assigning a conservative analog/digital classification using structural heuristics

Signal role assignment (V_PHASE, I_EARTH, DIG_TRIP, etc.) is the responsibility of:
  - app/analytics/ — signal role detection engine (future)
  - app/services/ — channel mapping service (future)

Downstream systems (visualization, analytics, synchronization, reporting) consume
DisturbanceRecord channels after signal roles have been assigned by the analytics layer.
No visualization or analytics logic shall be performed inside providers.

───────────────────────────────────────────────────────────────────────────────
SECTION 1 — ANALOG SIGNAL ROLE TAXONOMY
───────────────────────────────────────────────────────────────────────────────

The following signal roles are defined for analog channels:

  Role            Description                        Valid Phases         Canonical Unit
  ─────────────────────────────────────────────────────────────────────────────────────
  V_PHASE         Phase-to-earth voltage             A, B, C              kV
  V_LINE          Phase-to-phase voltage             AB, BC, CA           kV
  V_RESIDUAL      Residual / zero-sequence voltage   N, (empty)           kV
  I_PHASE         Phase current                      A, B, C              A
  I_EARTH         Earth / neutral / residual current N, (empty)           A
  V1_PMU          Positive-sequence voltage (PMU)    Pos-seq              kV
  I1_PMU          Positive-sequence current (PMU)    Pos-seq              A
  P_MW            Active power                       3-phase, (empty)     MW
  Q_MVAR          Reactive power                     3-phase, (empty)     MVAR
  FREQ            System frequency                   (empty)              Hz
  ROCOF           Rate of change of frequency        (empty)              Hz/s
  DC_FIELD_I      Generator field current (DC)       (empty)              A
  DC_FIELD_V      Generator field voltage (DC)       (empty)              V
  MECH_SPEED      Mechanical shaft speed             (empty)              RPM
  MECH_VALVE      Valve / gate / governor position   (empty)              %
  SEQ_RMS         Sequence component RMS             pos, neg, zero       kV
  ANALOGUE        Generic — unknown or other         (empty)              (any)

ANALOG ROLE ASSIGNMENT RULES:

  Roles V_PHASE and I_PHASE require phase resolution (A/B/C).
  Roles V_RESIDUAL and I_EARTH use phase N or empty.
  Roles P_MW, Q_MVAR, FREQ, ROCOF, DC_*, MECH_* are single-channel (no phase set).
  Default when detection fails: ANALOGUE.

───────────────────────────────────────────────────────────────────────────────
SECTION 2 — DIGITAL SIGNAL ROLE TAXONOMY
───────────────────────────────────────────────────────────────────────────────

The following signal roles are defined for digital channels:

  Role              Description
  ─────────────────────────────────────────────────────────────────────────────
  DIG_TRIP          Protection trip output
  DIG_CB            Circuit breaker / switch status (GCB, FCB, phase CB)
  DIG_PICKUP        Protection pickup / alarm threshold / element start
  DIG_AR            Auto-reclose (attempt, block, lockout, in-progress)
  DIG_INTERTRIP     Teleprotection / breaker-fail / inter-trip
  DIG_TRIGGER       External trigger input
  DIG_GENERIC       Alarm, supervision, comms-fail, enable, MCB, VTS

  Default when no keywords match: DIG_GENERIC.

ALARM EXCEPTION:
  The alarm/generic exception must be evaluated FIRST before any other digital
  rule. Keywords that indicate supervision or comms-failure channels (COMMFAIL,
  ALARM, MCB, VTS, ENABLE, etc.) always resolve to DIG_GENERIC regardless of
  whether other trip/CB keywords also appear in the name.

───────────────────────────────────────────────────────────────────────────────
SECTION 3 — PHASE NAMING CONVENTIONS
───────────────────────────────────────────────────────────────────────────────

COMTRADE and other power system sources use multiple phase naming conventions.
The analytics layer SHALL normalise all phase labels to A/B/C/N internally.
The parser layer SHALL preserve the raw phase field without normalisation.

  Convention        Phase A   Phase B   Phase C   Neutral / Earth
  ───────────────────────────────────────────────────────────────
  IEC A/B/C         A         B         C         N
  British R/Y/B     R         Y         B         N
  Chinese a/b/c     a         b         c         N
  IEC L1/L2/L3      L1        L2        L3        N
  ABB (some)        L1        L2        L3        PE

Normalisation rules for the analytics layer:
  R → A,   Y → B,   B → C  (only when part of a signal code, e.g. VR/IR/VB)
  a → A,   b → B,   c → C  (Chinese IEC convention, e.g. Ia/Ua)
  L1 → A,  L2 → B,  L3 → C
  N → N    (neutral / earth — preserved as-is)

IMPORTANT: Phase 'B' in R/Y/B convention means phase C (not phase B).
The signal code context (VB, IB) is the primary discriminator.
When phase is part of a full signal code (VR, VY, VB, IR, IY, IB), use the
SIGNAL CODE LOOKUP TABLE (Section 4) rather than single-letter parsing.

───────────────────────────────────────────────────────────────────────────────
SECTION 4 — SIGNAL CODE LOOKUP TABLE
───────────────────────────────────────────────────────────────────────────────

The following lookup table maps signal codes (typically the last token of a
channel name) to their canonical role and phase. This covers all vendor
conventions confirmed from real power system disturbance files.

Case-sensitive entries (NARI Chinese IEC) take priority over uppercased lookup.

  Signal Code   Role          Phase   Convention
  ────────────────────────────────────────────────────────────────────────────
  VR            V_PHASE       A       British / Malaysian R/Y/B
  VY            V_PHASE       B       British / Malaysian R/Y/B
  VB            V_PHASE       C       British / Malaysian R/Y/B
  UR            V_PHASE       A       German/British U notation, R/Y/B
  UY            V_PHASE       B       German/British U notation, R/Y/B
  UB            V_PHASE       C       German/British U notation, R/Y/B
  IR            I_PHASE       A       British / Malaysian R/Y/B
  IY            I_PHASE       B       British / Malaysian R/Y/B
  IB            I_PHASE       C       British / Malaysian R/Y/B
  IN            I_EARTH       N       Neutral current
  VN            V_RESIDUAL    N       Neutral voltage
  UN            V_RESIDUAL    N       German/British U notation, neutral
  VA            V_PHASE       A       IEC A/B/C
  VC            V_PHASE       C       IEC A/B/C
  IA            I_PHASE       A       IEC A/B/C
  IC            I_PHASE       C       IEC A/B/C
  Ia            I_PHASE       A       Chinese IEC (NARI) — case-sensitive
  Ib            I_PHASE       B       Chinese IEC (NARI) — case-sensitive
  Ic            I_PHASE       C       Chinese IEC (NARI) — case-sensitive
  Ua            V_PHASE       A       Chinese IEC (NARI) — case-sensitive
  Ub            V_PHASE       B       Chinese IEC (NARI) — case-sensitive
  Uc            V_PHASE       C       Chinese IEC (NARI) — case-sensitive
  3I0 / 3IO     I_EARTH       N       Zero-sequence current sum
  3U0 / 3UO     V_RESIDUAL    N       Zero-sequence voltage sum
  I0            I_EARTH       N       Zero-sequence current (short form)
  U0 / V0       V_RESIDUAL    N       Zero-sequence voltage (short form)
  UX / Ux       V_RESIDUAL    N       Open-delta residual voltage

NOTE: IB (British/Malaysian) maps to C; Ib (Chinese) maps to B.
The case-sensitive lookup resolves this ambiguity.

───────────────────────────────────────────────────────────────────────────────
SECTION 5 — ANALOG CHANNEL DETECTION PRIORITY RULES
───────────────────────────────────────────────────────────────────────────────

The analytics signal role detector SHALL apply these rules in strict priority
order. First match wins.

PRIORITY 1 — Unit field (most reliable discriminator)
  If the channel unit field is present and recognisable:
    kV or V   → V_PHASE (refine phase with signal code or name)
    kA or A   → I_PHASE (refine phase with signal code or name)
    Hz        → FREQ
    MW        → P_MW (phase: 3-phase)
    MVAR/MVAr → Q_MVAR (phase: 3-phase)
    RPM       → MECH_SPEED
    %         → MECH_VALVE

  After role assignment from unit, refine V_PHASE and I_PHASE:
    - Look up the last name token in the Signal Code Lookup Table (Section 4)
    - Check for residual/earth keywords in the full name → reclassify to V_RESIDUAL / I_EARTH

  Confidence: HIGH

PRIORITY 2 — Non-zero offset signature (DC / mechanical channels)
  If channel offset (b field from COMTRADE CFG) is non-zero:
    Name contains FIELD CURRENT / FIELD I / IF → DC_FIELD_I
    Name contains FIELD VOLTAGE / FIELD V / VF → DC_FIELD_V
    Name contains SPEED / RPM / MECH           → MECH_SPEED
    Name contains VALVE / GATE / GOVERNOR / POSITION → MECH_VALVE
  Confidence: HIGH

  Non-zero offsets are characteristic of DC and mechanical channels.
  See COMTRADE_NORMALIZATION_POLICY.md Section 4.1 for the offset formula.
  Common real-file examples: field current offset ≈ −1875 A, RPM offset ≈ −1000.

PRIORITY 3 — Power channel name keywords
  Name contains POWER UNIT / ACTIVE POWER / MW → P_MW (phase: 3-phase)
  Name contains R.POWER / REACTIVE / MVAR / MVAr → Q_MVAR (phase: 3-phase)
  Confidence: HIGH

PRIORITY 4 — Frequency channel name keywords
  Name contains FREQ (with trailing space or end) / FREQUENCY → FREQ
  Name contains DF/DT / ROCOF → ROCOF
  Confidence: HIGH

PRIORITY 5 — Sequence RMS channel name keywords
  Name contains O SEQ RMS / ZERO SEQ / NEG SEQ / POS SEQ / SEQ RMS / 0SEQ → SEQ_RMS
  Confidence: HIGH

PRIORITY 6 — Direct signal code lookup
  Extract the last whitespace-delimited token from the channel name.
  Look up (case-sensitive first, then uppercase) in Signal Code Lookup Table (Section 4).
  Confidence: HIGH

PRIORITY 7 — Pattern matching on full channel name
  Regex patterns to catch voltage/current when code lookup failed:
    Current patterns: ^I[_-]?[RYBABC], ^IL\d, CURR, AMPERE
    Voltage patterns: ^[VU][_-]?[RYBABC], ^[VU]\d, VOLT
  Extract phase using Phase Naming Conventions (Section 3).
  Confidence: MEDIUM

PRIORITY 8 — Default
  Return ANALOGUE with empty phase.
  Confidence: LOW

  Note: A magnitude-based heuristic (using raw sample values) is explicitly
  deferred to the analytics layer and must not be used inside the ingestion
  provider. The analytics layer may optionally apply it as a last-resort fallback
  only after all name-based rules are exhausted.

───────────────────────────────────────────────────────────────────────────────
SECTION 6 — DIGITAL CHANNEL DETECTION RULES
───────────────────────────────────────────────────────────────────────────────

Digital detection applies keyword sets in STRICT PRIORITY ORDER.
The ALARM EXCEPTION is checked FIRST before all other rules.

ALARM EXCEPTION (checked first — overrides all other rules):
  Keywords: COMMFAIL, COMM FAIL, COMM_FAIL, _FAIL, FAIL , ALARM, WARNING,
            MCB, VTS, BI_EN, BI_MCB, EN_Z, _ENABLE, ENABLE_
  Match → DIG_GENERIC (HIGH confidence)

DIG_CB — Circuit breaker / switch status:
  Keywords: CB_R, CB_Y, CB_B, CB OPEN, CB CLOSE, CB_OPEN, CB_CLOSE,
            GCB OPEN, GCB CLOSED, GCB_OPEN, GCB_CLOSED,
            FCB OPEN, FCB CLOSED, FCB_OPEN, FCB_CLOSED,
            BI_52B, BI_52A, 52B, 52A
  Confidence: HIGH

DIG_AR — Auto-reclose:
  Keywords: 79AR, 79 AR, AR_BLOCK, AR_ATTEMPTED, AR_UNSUCCESSFUL,
            79AR_L/O, AR_LOCKOUT, RECLOSE, AR_INPROGRESS,
            AUTORECLOSE, AUTO RECLOSE
  Confidence: HIGH

DIG_INTERTRIP — Teleprotection / breaker-fail:
  Keywords: 50BF, BF_SEND, BF_REC, BF_STG, BF_INTR, 50BF_STG1, 50BF_STG2,
            BF STAGE, SEND1..SEND3, RECV1..RECV3, INTERTRIP, INTER-TRIP,
            DIRECT TRIP, REMOTE TRIP, _CS, _CR, _REC, TELEPROTECT,
            87L/1/2_L1..L3, DEF M1/M2, POTT
  Confidence: HIGH

DIG_TRIGGER — External trigger:
  Keywords: TRIGGER, TRIG , EXT TRIG, EXTERNAL TRIG
  Confidence: HIGH

DIG_TRIP — Protection trip output (checked AFTER specific categories above):
  Keywords: 87L, 87T, 87B, 87G, 87N, 87M (differential protection)
            21ZBU, OP_Z1, OP_Z2, OP_Z3, M1 Z1, M2 Z1 (distance)
            OP_ROC, OP_OC, OP_OVLD, THERMAL OL (overcurrent/OL)
            SOTF, OP_Z_SOTF, OP_ROC_SOTF (switch-onto-fault)
            OP_Z_TELEP, OP_ROC_TELEP, OP_WE_POTT (teleprotection-assisted)
            TRIP, OPERATED, GEN TRIP (generic)
            OP_ (NARI prefix)
  Confidence: HIGH

DIG_PICKUP — Element start / pickup:
  Keywords: OVER , UNDER , OVERVOLTAGE, UNDERVOLTAGE, OVERCURRENT,
            FD, (with comma), _FD, FAULT DET, PICKUP, PICK UP,
            START , ELEMENT START, VEBI_DISTP, VEBI_ROC, BI_EXTRP
  Confidence: MEDIUM

DEFAULT: DIG_GENERIC — Low confidence

NOTE: NARI relay prefixes: Op_ → DIG_TRIP, BI_ → DIG_CB, VEBI_ → DIG_PICKUP.
All keyword matching is case-insensitive on the full channel name.

───────────────────────────────────────────────────────────────────────────────
SECTION 7 — COMPLEMENTARY DIGITAL CHANNEL PAIRS
───────────────────────────────────────────────────────────────────────────────

Some digital channels represent a single physical state as two complementary
signals: one for the OPEN state and one for the CLOSED state.

Examples:
  GCB OPEN  + GCB CLOSED  → one generator circuit breaker status
  FCB OPEN  + FCB CLOSED  → one field circuit breaker status
  CB_R OPEN (alone)       → single phase-R circuit breaker status

Detection rule:
  Among all channels with role DIG_CB, find pairs where:
  - One channel name contains "OPEN"
  - The other contains "CLOSED" or "CLOSE"
  - The base name (with OPEN/CLOSED removed and stripped) matches

Display convention:
  Complementary pairs SHALL be displayed as a single state timeline bar,
  not as two separate binary channels.
  open=1 / closed=0 is the display convention (OPEN state as the active state).

───────────────────────────────────────────────────────────────────────────────
SECTION 8 — INGESTION-LAYER CHANNEL CLASSIFICATION (PARSER SCOPE)
───────────────────────────────────────────────────────────────────────────────

This section defines what the parser/provider layer MAY do for basic channel
classification, without entering analytics territory.

Parsers SHALL apply only STRUCTURAL classification:

  ANALOG (default):
    Any numeric column that is not identified as digital.
    Unit inference from column name keywords:
      Voltage-like names (VA, VB, VC, VR, VY, VB, "volt", "kv") → "kV"
      Current-like names (IA, IB, IC, IR, IY, IB, "curr", "amp") → "A"
      Frequency-like names ("freq", "hz") → "Hz"
      Power MW names ("mw") → "MW"
      Reactive power names ("mvar") → "MVar"
      All others → "unknown"

  DIGITAL (conservative):
    Only when ALL of the following are true:
      1. Column values contain only 0 and 1 (after dropping NaN)
      2. Column name contains a status keyword:
         trip, pickup, breaker, status, cb, relay, alarm, open, close, signal,
         flag, state
    OR: column is boolean dtype.

Parsers SHALL NOT:
  - Assign V_PHASE, I_PHASE, DIG_TRIP, DIG_CB or any other analytics role
  - Apply phase normalisation (R→A, Y→B, etc.)
  - Perform signal code lookup

These responsibilities belong to the analytics signal role detector (future
app/analytics/ or app/services/ module).

───────────────────────────────────────────────────────────────────────────────
SECTION 9 — DOWNSTREAM SIGNAL ROLE USAGE
───────────────────────────────────────────────────────────────────────────────

The following table documents how downstream systems consume signal roles.
This drives the engineering importance of accurate role assignment.

  Consumer            Uses signal_role for
  ────────────────────────────────────────────────────────────────────────────
  Colour coding       V_PHASE → R(A)/#FF4444, Y(B)/#FFCC00, B(C)/#4488FF
                      I_PHASE → same phase colours as V
                      DC_FIELD_I/V, MECH_* → #AAAAAA (grey)
                      FREQ → #00DDDD (cyan)
  Display mode        DC_FIELD_I/V, MECH_SPEED/VALVE → always TREND (scatter)
                      regardless of sample_rate (overrides LAW 9 sample_rate check)
  RMS calculator      V_PHASE, I_PHASE → full-cycle RMS. Others excluded.
  Phasor calculator   V_PHASE + I_PHASE only, 3-phase sets, phase A/B/C required
  Symmetrical comps   Groups V_PHASE A+B+C; groups I_PHASE A+B+C; warns if incomplete
  Power calculator    Matches V_PHASE+I_PHASE by phase; uses is_derived=True if available
  Frequency tracker   First FREQ channel; falls back to first V_PHASE channel
  Event timeline      DIG_TRIP=red, DIG_CB=orange, DIG_AR=blue, DIG_INTERTRIP=magenta
  Complementary CB    GCB/FCB open+closed pairs displayed as single state bar
  Measurement panel   Unit label from role definition; channels grouped by signal type
  PDF report          Role names used, not raw channel names; sections auto-labelled

───────────────────────────────────────────────────────────────────────────────
SECTION 10 — PROVIDER NORMALISATION CONTRACT
───────────────────────────────────────────────────────────────────────────────

What providers MUST deliver (parser-layer contract):

  RecordingMetadata.provider_type  — "COMTRADE" / "csv" / "excel" / etc.
  AnalogChannel.name               — raw channel name, stripped of whitespace only
  AnalogChannel.unit               — raw unit string from source (or inferred from name)
  AnalogChannel.index              — 0-based position within analog_channels list
  AnalogChannel.phase              — raw phase string from COMTRADE CFG (or None for CSV/Excel)
  AnalogChannel.scale, .offset     — from COMTRADE a/b fields (default 1.0 / 0.0 for CSV/Excel)
  DigitalChannel.name              — raw channel name, stripped of whitespace only
  DigitalChannel.index             — 0-based position within digital_channels list
  DigitalChannel.normal_state      — from COMTRADE y field (default 0 for CSV/Excel)

What providers SHALL NOT add:
  signal_role assignment (belongs in analytics layer)
  role_confidence (belongs in analytics layer)
  bay_name extraction (belongs in analytics layer)
  phase normalisation (R→A, Y→B) (belongs in analytics layer)

───────────────────────────────────────────────────────────────────────────────
SECTION 11 — FUTURE EXTENSIBILITY
───────────────────────────────────────────────────────────────────────────────

Future capabilities that SHALL NOT be pre-implemented in the parser layer:

  Signal role detector (app/analytics/ or app/services/):
    Implements the 8-priority analog detection rules and digital keyword sets.
    Operates on a fully populated DisturbanceRecord.
    Adds signal roles, confidence, and phase assignments post-ingestion.

  Channel mapping dialog (app/ui/):
    SIGRA-equivalent manual override for auto-detection failures.
    Always shown for CSV and Excel files (no naming standard).
    Shown for COMTRADE when >20% of analog channels are LOW confidence.
    Allows user to save named mapping profiles keyed by station + device ID.

  Mapping profile persistence:
    JSON-backed profiles stored in application data directory.
    Key: station_name + recorder_name (COMTRADE) or filename stem pattern (CSV/Excel).
    On load: match key → auto-apply → show notification banner (not full dialog).

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURAL ALIGNMENT
───────────────────────────────────────────────────────────────────────────────

This policy is aligned with:

  docs/ARCHITECTURE.md             — provider layer responsibilities, strict layer isolation
  docs/DATA_CONTRACT.md            — DisturbanceRecord structure and channel contracts
  docs/PROVIDER_PATTERN.md         — provider isolation rules
  docs/COMTRADE_NORMALIZATION_POLICY.md — parser boundary, what parsers preserve
  docs/LEGACY_CODEBASE_POLICY.md   — no src/ imports in app/ providers

The signal role taxonomy in this document supersedes and absorbs:
  .claude/skills/SKILL_channel_mapping.md (DELETED — consolidated 2026-05-10)

FINAL PRINCIPLE

Channel naming normalisation is an analytics responsibility, not a parser responsibility.

Parsers provide clean, faithful representations of source data.
The analytics layer transforms raw channel names and units into structured signal roles.
The UI layer provides manual override and profile recall when auto-detection is insufficient.

This separation preserves the strict layer isolation that makes Powerwave maintainable
and extensible as the platform grows toward full analytics capability.
