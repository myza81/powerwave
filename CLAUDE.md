# PowerWave Analyst — Claude Code Master Context

## ── CURRENT SESSION ────────────────────────────────────────────────────────

Phase: Phase 1
Milestone: Milestone 1B — COMTRADE Parser
Module: src/parsers/comtrade_parser.py
Status: IN PROGRESS — Step 2 (DAT) complete, Step 3 (signal_role_detector) next

## ── PROJECT IDENTITY ───────────────────────────────────────────────────────

Name: PowerWave Analyst
Type: Standalone Windows desktop application (.exe)
Domain: Power system disturbance record analysis
Users: Protection Engineers, Fault Analysts, System Operators
Offline: 100% — zero network calls, zero telemetry, ever

## ── TECHNOLOGY STACK ───────────────────────────────────────────────────────

Python: 3.11 (NOT 3.12 or 3.13)
UI: PyQt6
Rendering: PyQtGraph + OpenGL (useOpenGL=True always)
Numerics: NumPy 1.26+ (arrays only — no Pandas for time-series)
Signal proc: SciPy 1.12+
Acceleration: Numba JIT (@numba.jit nopython=True on hot loops)
Data I/O: Pandas (import/export only, not internal data model)
COMTRADE: comtrade lib + custom extensions in parsers/
PDF export: ReportLab
Packaging: PyInstaller (dev) → Nuitka (release)

## ── ARCHITECTURE LAWS (never violate) ─────────────────────────────────────

LAW 1 Four layers — strict upward-only imports:
Presentation (ui/) → Rendering (ui/canvas\*) → Service (engine/) → Data (parsers/, models/)
No layer imports from a layer above it. Ever.

LAW 2 UI thread is sacred — never block it.
All file I/O, parsing, heavy computation → QThreadPool via core/thread_manager.py
Emit signals when done. UI receives results via slots.

LAW 3 Decimation cap — never render more than 4000 points per channel per render call.
engine/decimator.py handles min/max envelope for all zoom levels.
PyQtGraph setData() only — never remove/re-add PlotDataItems.

LAW 4 Never resample for display — only for export.
Each source renders at its native sampling rate on the shared time axis.
Fast record (5000Hz) and slow record (50Hz) coexist on same canvas natively.

LAW 5 Single source of truth — DisturbanceRecord.
Every COMTRADE/CSV/Excel parser produces exactly one DisturbanceRecord.
PmuRecord for PMU CSV. MergedSession for multi-source sessions.

LAW 6 Signals and slots for all UI↔engine communication.
No direct method calls across layer boundaries.

LAW 7 Type hints on every function signature.
Docstrings on every class and every public method.

LAW 8 No magic numbers — define constants at top of each file.

LAW 9 Display mode is determined by sample rate, not file type.
sample_rate >= 200 Hz → WAVEFORM mode (continuous line + decimation)
sample_rate < 200 Hz → TREND mode (scatter plot, no interpolation)
This applies to ALL file formats including COMTRADE slow records.

LAW 10 Offset must always be applied to raw analogue values.
physical = (raw × multiplier) + offset
Non-zero offsets appear on DC quantities (field current/voltage),
mechanical quantities (RPM, valve %), and pre-calculated power channels.
Never skip offset even when it is zero — always apply uniformly.

## ── PROJECT STRUCTURE ──────────────────────────────────────────────────────

powerwave_analyst/
├── CLAUDE.md ← this file (update every session)
├── requirements.txt
├── build.spec
├── .claude/
│ └── skills/
│ ├── SKILL_INDEX.md
│ ├── SKILL_comtrade_parser.md
│ ├── SKILL_signal_processing.md
│ ├── SKILL_pyqt6_rendering.md
│ ├── SKILL_pmu_power.md
│ ├── SKILL_merging_timesync.md
│ └── SKILL_channel_mapping.md
├── src/
│ ├── main.py
│ ├── core/
│ │ ├── app_state.py
│ │ ├── settings.py
│ │ └── thread_manager.py
│ ├── models/
│ │ ├── channel.py
│ │ ├── disturbance_record.py
│ │ ├── pmu_record.py
│ │ └── merged_session.py
│ ├── parsers/
│ │ ├── comtrade_parser.py
│ │ ├── csv_parser.py
│ │ ├── excel_parser.py
│ │ ├── pmu_csv_parser.py
│ │ └── signal_role_detector.py
│ ├── engine/
│ │ ├── decimator.py
│ │ ├── rms_calculator.py
│ │ ├── phasor_calculator.py
│ │ ├── symmetrical_components.py
│ │ ├── fft_analyzer.py
│ │ ├── frequency_tracker.py
│ │ ├── signal_processor.py
│ │ ├── time_sync_engine.py
│ │ └── power_calculator.py
│ ├── ui/
│ │ ├── main_window.py
│ │ ├── channel_canvas.py
│ │ ├── waveform_panel.py
│ │ ├── measurement_panel.py
│ │ ├── phasor_canvas.py
│ │ ├── fft_canvas.py
│ │ ├── annotation_layer.py
│ │ ├── multi_bay_view.py
│ │ ├── merge_manager.py
│ │ ├── sync_panel.py
│ │ ├── channel_mapping_dialog.py
│ │ ├── pmu_channel_panel.py
│ │ ├── power_canvas.py
│ │ └── gap_indicator.py
│ ├── export/
│ │ ├── pdf_exporter.py
│ │ └── image_exporter.py
│ └── assets/
├── tests/
│ ├── test_parsers/
│ ├── test_engine/
│ └── test_data/
└── docs/

## ── KEY DATA MODELS ────────────────────────────────────────────────────────

DisturbanceRecord (src/models/disturbance_record.py):
station_name: str device_id: str
start_time: datetime trigger_time: datetime
trigger_sample: int sample_rate: float
nominal_frequency: float (50.0 or 60.0 only)
display_mode: str ('WAVEFORM' or 'TREND' — auto from sample_rate)
analogue_channels: list[AnalogueChannel]
digital_channels: list[DigitalChannel]
time_array: np.ndarray (float64, seconds from start, NON-UNIFORM if multi-rate)
source_format: str (COMTRADE_1991/1999/2013/CSV/EXCEL/PMU_CSV)
file_path: Path
header_text: str
\_rms_cache: dict
\_phasor_cache: dict

AnalogueChannel (src/models/channel.py):
channel_id: int name: str
phase: str unit: str
multiplier: float offset: float (ALWAYS apply — may be non-zero)
primary: float secondary: float (None if not in CFG)
ps_flag: str raw_data: np.ndarray (float32, AFTER scaling)
signal_role: str role_confidence: str
role_confirmed: bool colour: str
visible: bool bay_name: str (extracted from channel name)
is_derived: bool (True for pre-calculated MW, MVAr stored in file)

## ── SIGNAL ROLE TAXONOMY (complete) ────────────────────────────────────────

Analogue roles:
V_PHASE Phase-to-earth voltage kV phases: A,B,C
V_LINE Phase-to-phase voltage kV phases: AB,BC,CA
V_RESIDUAL Residual/zero seq voltage kV phase: N
I_PHASE Phase current A/kA phases: A,B,C
I_EARTH Earth/neutral/residual current A/kA phase: N
V1_PMU Positive seq voltage (PMU) kV phase: Pos-seq
I1_PMU Positive seq current (PMU) A/kA phase: Pos-seq
P_MW Active power (pre-calc or PMU) MW phase: 3-phase
Q_MVAR Reactive power MVAR phase: 3-phase
FREQ System frequency Hz
ROCOF Rate of change of frequency Hz/s
DC_FIELD_I Generator field current (DC) A (non-zero offset expected)
DC_FIELD_V Generator field voltage (DC) V (non-zero offset expected)
MECH_SPEED Mechanical shaft speed RPM (non-zero offset expected)
MECH_VALVE Valve/gate/governor position % (non-zero offset expected)
SEQ_RMS Sequence component RMS kV phases: pos,neg,zero
ANALOGUE Generic — unknown or other any

Digital roles:
DIG_TRIP Protection trip output
DIG_CB Circuit breaker / switch status (incl. GCB, FCB)
DIG_PICKUP Protection element pickup / alarm threshold
DIG_AR Auto-reclose (attempt, block, lockout, in-progress)
DIG_INTERTRIP Teleprotection / breaker-fail / inter-trip
DIG_TRIGGER External trigger input
DIG_GENERIC Alarm, supervision, comms-fail, enable, MCB

## ── DISPLAY MODE RULES ──────────────────────────────────────────────────────

WAVEFORM_THRESHOLD = 200 # Hz

sample_rate >= 200 Hz:

- Render as continuous line (PlotDataItem)
- Apply min/max decimation (engine/decimator.py)
- RMS overlay available
- Phasor calculation available
- FFT available

sample_rate < 200 Hz:

- Render as scatter plot (ScatterPlotItem, size=4)
- No decimation needed (few points)
- No RMS overlay (already cycle-averaged or slower)
- No FFT (insufficient rate for harmonic analysis)
- Trend zoom: X axis in minutes not seconds for long records
- Label canvas header: "TREND RECORD — {sample_rate} Hz"

Mixed session (fast + slow records merged):

- Each source renders in its own mode independently
- Shared time axis works across both modes
- Measurement cursors work on both modes

## ── BAY AUTO-GROUPING ───────────────────────────────────────────────────────

BEN32 multi-bay files contain channels from multiple bays in one file.
The parser MUST extract the bay name from each channel name and group channels.

Bay extraction rule (analogue): "BAYNAME SIGNAL" → bay = everything before last token
Example: "SJTC2 VR" → bay="SJTC2", signal="VR"
Example: "KLBG1 IY" → bay="KLBG1", signal="IY"
Exception: "FREQ UR UNIT NO.2" → bay="UNIT NO.2", signal="FREQ UR" (name-first)

Bay extraction rule (digital): bay name appears ANYWHERE in string
Example: "OVER UR SJTC NO.2" → bay="SJTC2" (fuzzy match against known bays)
Example: "SJTC2 87L/1" → bay="SJTC2" (prefix match)

Known bay names are populated from analogue channel parsing first,
then used to classify digital channels. Store in DisturbanceRecord.bay_names list.
Each bay becomes a named group in the channel panel and multi-bay canvas.

## ── VENDOR QUIRKS (confirmed from real files) ──────────────────────────────

BEN32 FAST RECORD (≥1200Hz — fault/transient):
rev_year: 1999 (valid) or may be actual year e.g.2005 → parse as 1999
Analogue naming: "BAYNAME SIGNAL" e.g. "SJTC2 VR", "KPAR IY"
Phase convention: R/Y/B → map to A/B/C. N = earth.
Also uses U for voltage in some channels: "UR UNIT NO.2" = phase A voltage
Units: kV for voltage, kA for current. PS flag always P (primary values stored).
Primary/secondary always populated — no CT/VT ratio needed separately.
Analogue CFG fields: 13 fields (full — includes primary, secondary, PS)
Digital CFG fields: 5 fields (ch, name, ph, ccbm, normal_state)
Digital naming: qualifier-first "OVER UR SJTC NO.2" — bay name AFTER signal
Single sampling rate. High channel count (28A, 104D typical).
Multi-bay: one file contains all substation bays simultaneously.

BEN32 SLOW RECORD (<200Hz — any sustained disturbance event):
Same format rules as fast record EXCEPT:
Sample rate: 50Hz typical (one sample per cycle).
Duration: 70+ seconds (much longer than fast record).
May contain non-electrical channels: MW, MVAr, Hz, RPM, %, DC quantities.
Non-zero offsets on DC/mechanical channels — MUST apply offset.
Pre-calculated P and Q stored as analogue channels → is_derived=True.
Blank unit field on RPM and % channels — detect by name keywords.
Display as TREND (scatter) not WAVEFORM.
May record same event as a fast record — both should be mergeable.

NARI RELAY:
Line 1: only 2 fields (station, device) — rev*year MISSING → default 1999
Analogue naming: Chinese IEC convention — Ia/Ib/Ic, Ua/Ub/Uc, 3I0, 3U0, Ux
Phase: lowercase a/b/c → map to A/B/C
Zero sequence: 3I0=I_EARTH, 3U0=V_RESIDUAL, Ux=V_RESIDUAL
Analogue CFG fields: 10 fields — primary, secondary, PS MISSING → default None/'S'
Digital CFG fields: 3 fields (ch, name, normal_state) — no ph or ccbm
Digital prefixes: Op*=protection operated (→DIG*TRIP), BI*=binary input,
VEBI\_=virtual binary input
Multi-rate common: section 1 high rate (1200Hz) for transient,
section 2 lower rate (600Hz) for post-fault
Time array is NON-UNIFORM across rate boundary — build section by section.

ABB:
Underscore suffix naming convention (e.g. IL1_ABB). Strip suffix for display.
Check multiplier carefully — sometimes non-standard scaling.

Siemens:
Microsecond timestamp precision in CFG.
High digital channel count common.

GE:
May use COMTRADE 1991 format.
Verify P/S flag — primary/secondary values must be confirmed.

SEL:
Non-standard channel labels.
Sometimes exports CSV alongside COMTRADE.

PMU CSV (Malaysian grid — confirmed from real files):
File starts with metadata row: "ID: NNN, Station Name: XXXXX,,,,,"
Actual column header row always starts with "Date"
Time column header: "Time (Asia/Singapore)" → timezone SGT (UTC+8)
ALWAYS convert SGT → UTC before any time alignment operation
Voltage magnitude in raw Volts (not kV) → divide by 1000 for kV display
Station name encodes nominal voltage: prefix digits = kV level
(e.g. "500JMJG-U5" → 500kV, "275TMGR-U2" → 275kV)
Column prefix pattern: signals may have unit/bay prefix before core name
e.g. "UNIT2*V1 Magnitude", "KAWA1_V1 Angle" → strip "WORD*" prefix
Core column names (after prefix strip): V1 Magnitude, V1 Angle,
I1 Magnitude, I1 Angle, Frequency, df/dt, Status, Date, Time
Report rate: 50 fps confirmed (0.020s between rows)
Status field: "00 00"=healthy, non-zero=flag for engineer review
PMU 241 broken timestamp "12:00.0" — detect and flag as LOW quality GPS
Data is positive-sequence phasors only — no individual phase reconstruction

## ── CHANNEL MAPPING DIALOG (Manual Fallback) ───────────────────────────────

The channel mapping dialog is the SIGRA-equivalent manual identifier.
It appears automatically when auto-detection confidence is low.
It is ALWAYS accessible manually via right-click on any channel.

Show dialog automatically when:

- Any CSV or Excel file is loaded (always — no standard header convention)
- COMTRADE where >20% of analogue channels are LOW confidence
- PMU CSV where auto-detection of column roles fails
- User right-clicks channel → "Reassign Signal Type"
- User clicks Edit → Channel Mapping

Do NOT show for:

- Well-formed COMTRADE from known vendors where all channels detected HIGH
- Subsequent loads of files matching a saved mapping profile

Dialog must provide:

- Scrollable channel table: one row per channel
- Per row: original name, data preview (5 values), signal role dropdown,
  phase dropdown, unit field, confidence badge
- Bulk assignment: Shift+click / Ctrl+click multi-select
- Auto-detection suggestions shown in italic (HIGH/MEDIUM/LOW badge)
- Save Profile: saves mapping keyed to station_name + device_id
- Load Profile: auto-apply on matching file, skip dialog
- Validation warnings (non-blocking): duplicate phase assignments,
  incomplete 3-phase sets, missing V+I pair for power calculation
- Confirm / Cancel buttons

## ── COMPLEMENTARY DIGITAL CHANNEL PAIRS ────────────────────────────────────

Some digital channels come in complementary pairs that represent one state:
GCB OPEN + GCB CLOSED → one GCB status (open=1/closed=0)
FCB OPEN + FCB CLOSED → one FCB status
CB_R OPEN (alone) → single CB status per phase

Detection rule: if two DIG_CB channels share the same device name and one
contains OPEN and the other CLOSED → mark as complementary pair.
Display as single state bar on event timeline (not two separate channels).

## ── COLOUR CONVENTION ──────────────────────────────────────────────────────

Phase A / Red: #FF4444 Phase B / Yellow: #FFCC00
Phase C / Blue: #4488FF Earth/Neutral: #44BB44
DIG_TRIP: #FF2222 DIG_CB: #FF8800
DIG_PICKUP: #FFAA00 DIG_AR: #44AAFF
DIG_INTERTRIP: #FF44FF DIG_GENERIC: #888888
FREQ channel: #00DDDD MW channel: #FFAA44
MVAr channel: #AA44FF DC quantities: #AAAAAA
Trend mode: use filled circles, colour from phase convention

## ── PERFORMANCE TARGETS ────────────────────────────────────────────────────

File load (10s, 100ch, 6kHz): < 3 seconds
Pre-computation (RMS + phasor): < 5 seconds post-load
Initial waveform render: < 500ms
Zoom/pan response: < 16ms (60fps)
Cursor → all measurements update: < 100ms
FFT at cursor: < 200ms
Slow-rate trend render (3500pt): < 100ms (no decimation needed)
Multi-source merge (5 sources): < 2 seconds for alignment computation

## ── COMPATIBILITY PHILOSOPHY ────────────────────────────────────────────────

PowerWave Analyst targets full COMTRADE standard compliance — not vendor-specific
implementations. Any CFG/DAT file conforming to COMTRADE 1991, 1999, or 2013
must load correctly regardless of manufacturer (ABB, Siemens, Reyrolle, Toshiba,
GE, SEL, NARI, BEN32, or any other vendor).

All vendor quirks fall into a finite set of categories already handled:
Structural: safe field-count parsing (10 vs 13 analogue, 3 vs 5 digital)
Rev year: missing or invalid → default to 1999
Naming: R/Y/B, A/B/C, a/b/c, I/U — all detected by signal_role_detector
Offset: always applied — non-zero for DC/mechanical channels
Multi-rate: section-by-section time array construction
Encoding: UTF-8, Latin-1, ASCII all handled

For anything else: channel mapping dialog + profile save/recall.
No vendor should ever cause a crash — only a mapping dialog at worst.

## ── COMPLETED MILESTONES ───────────────────────────────────────────────────

# ✓ Milestone 1A — Data Models: AnalogueChannel, DigitalChannel, DisturbanceRecord,

# SignalRole constants, 91 tests passing

# ✓ Milestone 1B Step 1 — CFG parsing: ComtradeParser.\_parse_cfg(),

# build_time_array(), extract_bay_from_analogue_name(), 185 tests passing

# (5 real files: BEN32 fast/slow, NARI multi-rate, variable-rate IED, MM/DD/YY quirk)

# ✓ Milestone 1B Step 2 — DAT reading: \_read_dat_ascii(), \_read_dat_binary(),

# \_read_dat_binary32(), \_read_dat_float32(). Variable-rate nrates=0 resolved from

# DAT timestamps. Windows \x1a EOF handled. 104 new tests, all formats passing.

## ── ARCHITECTURE DECISIONS & KNOWN ISSUES ──────────────────────────────────

[Add decisions and issues here as discovered during development]

# DECISION: reason

# ISSUE: description and workaround

## ── DEVELOPMENT ENVIRONMENT ────────────────────────────────────────────────

OS: macOS
Python: 3.11.9 via pyenv (set by .python-version file in project root)
Venv: .venv/ — MUST be active before running any command

ACTIVATION (run once per VS Code session):
source .venv/bin/activate
Prompt must show (.venv) prefix before proceeding.

ALWAYS use venv-prefixed commands — never bare python/pip/pytest:
Python: .venv/bin/python
Pip: .venv/bin/pip
Pytest: .venv/bin/pytest
App: .venv/bin/python src/main.py

VERIFY correct interpreter:
.venv/bin/python -c "import sys; print(sys.executable)"
Expected: /Volumes/externalDrive/code-gym/powerwave/.venv/bin/python

PACKAGE MANAGEMENT:
Install missing package: .venv/bin/pip install <package>
Never install to system Python or pyenv global.

## ── PLANNING DOCUMENTS ─────────────────────────────────────────────────────

docs/PowerWave_Doc1_Skills_Knowledge_Base_Rev2.docx
docs/PowerWave_Doc2_High_Level_Overview_Rev2.docx
docs/PowerWave_Doc3_Implementation_Plan_Rev2.docx
docs/PowerWave_Supplement_Merging_PMU_Rev2.docx
docs/PowerWave_FeatureNote_ChannelSignalMapping_Rev2.docx
