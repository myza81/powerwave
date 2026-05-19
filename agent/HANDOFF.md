HANDOFF.md — Powerwave Cross-Agent Handoff Protocol
PURPOSE

This document defines the standard handoff protocol between:

ChatGPT
Claude
Claude Code
Codex

The objective is to ensure:

continuity between implementation sessions
architecture consistency
reduced duplicated work
reduced AI confusion
traceable engineering progress

Every significant implementation session SHALL update this document.

HANDOFF RULES
RULE 1 — ALWAYS UPDATE AFTER IMPLEMENTATION

After completing meaningful work, the implementation agent SHALL update:

completed work
modified files
current architecture state
unresolved concerns
next recommended step

This prevents:

repeated work
architecture drift
context loss
RULE 2 — DO NOT REMOVE HISTORY

Do NOT delete prior handoff entries.

Append new entries chronologically.

Engineering history must remain traceable.

RULE 3 — KEEP ENTRIES TECHNICAL

Handoff entries must contain:

implementation details
architecture impact
known limitations
next engineering actions

Avoid:

conversational summaries
vague descriptions
non-technical commentary
HANDOFF ENTRY FORMAT

Every handoff SHALL use this structure:

## DATE / SESSION

### Agent
Claude / Codex / ChatGPT / Claude Code

### Task
Short task description

### Completed
- item
- item
- item

### Files Modified
- path/file.py
- path/file.py

### Architecture Impact
Description of affected systems

### Performance Impact
Rendering/runtime/memory considerations

### Risks / Concerns
Known issues or future concerns

### Next Recommended Step
Suggested continuation
CURRENT SYSTEM STATUS
Architecture Status

IN PROGRESS

Repository Status

INITIALIZATION

Rendering Engine

NOT STARTED

Data Provider System

NOT STARTED

COMTRADE Parser

NOT STARTED

Synchronization Engine

NOT STARTED

ACTIVE DEVELOPMENT PRIORITIES

Current development order:

Repository structure
DisturbanceRecord contract
Provider pattern
COMTRADE parser
FastWaveformWidget
Multi-pane synchronization
Master time cursor
Performance optimization
IMPLEMENTATION TRACKING
PHASE 1 — FOUNDATION

Status: NOT STARTED

Scope:

repository structure
core contracts
application bootstrap
PHASE 2 — DATA INGESTION

Status: NOT STARTED

Scope:

provider pattern
COMTRADE parser
CSV parser
Excel parser
PHASE 3 — VISUALIZATION ENGINE

Status: NOT STARTED

Scope:

FastWaveformWidget
OpenGL acceleration
synchronized rendering
multi-pane infrastructure
PHASE 4 — INTERACTION ENGINE

Status: NOT STARTED

Scope:

master time cursor
synchronized zoom/pan
waveform interaction
PHASE 5 — ANALYTICS FOUNDATION

Status: NOT STARTED

Scope:

RMS
ROCOF
frequency analysis
harmonic foundation
phasor hooks
KNOWN ARCHITECTURAL RULES
RULE — SINGLE DATA CONTRACT

All waveform data shall use:

DisturbanceRecord

No parser-specific structures shall leak outside parser modules.

RULE — PERFORMANCE FIRST

The system must support:

large COMTRADE files
high-frequency waveform rendering
synchronized multi-pane interaction
industrial-scale disturbance analysis

Preferred techniques:

vectorization
downsampling
clip-to-view
incremental redraw
GPU acceleration
RULE — MODULAR ISOLATION

UI modules SHALL NOT:

parse files
perform heavy analytics

Parser modules SHALL NOT:

know UI state

Analysis modules SHALL NOT:

depend on widgets
ENGINEERING DECISION LOG
Decision 001

Powerwave uses:

PyQt6
PyQtGraph
PyOpenGL

Reason:
Industrial-grade high-performance rendering requirements.

Decision 002

Unified waveform contract:

DisturbanceRecord

Reason:
Prevent parser leakage and maintain modularity.

Decision 003

Provider-pattern architecture for ingestion.

Reason:
Future parser extensibility without UI modification.

OPEN CONCERNS
Concern 001

Need benchmark methodology for:

100MB+ COMTRADE loading
rendering latency
memory footprint

Status:
Pending implementation phase.

Concern 002

Need synchronization strategy validation for:

multiple waveform panes
master cursor performance

Status:
Pending visualization phase.

NEXT IMMEDIATE TARGET

Current immediate target:

docs/SYSTEM_OVERVIEW.md

This document will define:

overall system purpose
application modules
architectural philosophy
major subsystem interaction

It becomes the primary technical overview document for the entire platform.

---

## 2026-05-10 — Session 002

### Agent
Claude Code

### Task
Repository bootstrap — initialize approved directory structure per directives/build_repository_structure.md

### Completed
- Created app/ package with all 20 subdirectory packages (each with __init__.py)
- Created app/main.py — minimal QApplication + PowerwaveMainWindow placeholder
- Created tests/unit/, tests/integration/, tests/benchmarks/ with __init__.py
- Created resources/ directory
- Created README.md with project description, stack, and setup instructions
- Verified app/main.py syntax with Python compiler

### Files Modified
- app/__init__.py (new)
- app/main.py (new)
- app/config/__init__.py (new)
- app/ui/__init__.py (new)
- app/ui/main_window/__init__.py (new)
- app/ui/widgets/__init__.py (new)
- app/ui/dialogs/__init__.py (new)
- app/ui/panels/__init__.py (new)
- app/visualization/__init__.py (new)
- app/visualization/widgets/__init__.py (new)
- app/visualization/rendering/__init__.py (new)
- app/visualization/managers/__init__.py (new)
- app/visualization/interaction/__init__.py (new)
- app/analytics/__init__.py (new)
- app/analytics/rms/__init__.py (new)
- app/analytics/frequency/__init__.py (new)
- app/analytics/rocof/__init__.py (new)
- app/analytics/harmonics/__init__.py (new)
- app/analytics/phasor/__init__.py (new)
- app/providers/__init__.py (new)
- app/providers/base/__init__.py (new)
- app/providers/comtrade/__init__.py (new)
- app/providers/csv/__init__.py (new)
- app/providers/excel/__init__.py (new)
- app/models/__init__.py (new)
- app/synchronization/__init__.py (new)
- app/synchronization/cursor/__init__.py (new)
- app/synchronization/viewport/__init__.py (new)
- app/synchronization/managers/__init__.py (new)
- app/services/__init__.py (new)
- app/utils/__init__.py (new)
- tests/unit/__init__.py (new)
- tests/integration/__init__.py (new)
- tests/benchmarks/__init__.py (new)
- resources/ (new empty directory)
- README.md (new)

### Architecture Impact
Foundation layer established. All subsystem boundaries defined by directory structure:
UI layer (app/ui/), visualization engine (app/visualization/), analytics engine (app/analytics/),
provider layer (app/providers/), data model layer (app/models/), synchronization engine (app/synchronization/),
services layer (app/services/), utilities (app/utils/).
No existing src/ code was modified.

### Performance Impact
None — bootstrap only. No rendering, computation, or data loading implemented.

### Risks / Concerns
- Existing src/ contains the PowerWave Analyst codebase (separate project). The new app/ is cleanly isolated.
- tests/ directory contains both the legacy test_engine/test_parsers/test_ui subdirectories and the new unit/integration/benchmarks layout. These coexist without conflict.
- requirements.txt retains pinned versions from existing project — acceptable for now.

### Next Recommended Step
Implement DisturbanceRecord contract per directives/implement_disturbance_record.md (to be created by ChatGPT).
Bootstrap is complete and implementation-ready.

---

## 2026-05-10 — Session 003

### Agent
Claude Code

### Task
Implement DisturbanceRecord — foundational internal waveform contract per directives/implement_disturbance_record.md

### Completed
- Created app/models/metadata.py — RecordingMetadata dataclass (slots=True)
- Created app/models/channels.py — AnalogChannel, DigitalChannel dataclasses (slots=True)
- Created app/models/timing.py — SamplingInformation, TimingInformation, DisturbanceInformation (slots=True)
- Created app/models/disturbance_record.py — DisturbanceRecord (slots=True) with:
  - validate() returning list[str] (lightweight, non-raising)
  - analog_channel_names(), digital_channel_names(), channel_names()
  - has_channel(name) — short-circuit generator scan, no list allocation
  - sample_count() — len(waveform_data)
  - duration_seconds() — time column first, sampling_info fallback, 0.0 for empty
  - waveform_data stored by direct reference (no copy)
- Updated app/models/__init__.py — re-exports all 7 public symbols
- Updated pyproject.toml — added "." to pytest pythonpath so app/ is importable
- Created pyrightconfig.json — fixes IDE import resolution (extraPaths: ["."])
- Created tests/unit/test_disturbance_record.py — 26 unit tests across 4 test classes

### Files Modified
- app/models/__init__.py (updated)
- app/models/metadata.py (new)
- app/models/channels.py (new)
- app/models/timing.py (new)
- app/models/disturbance_record.py (new)
- tests/unit/test_disturbance_record.py (new)
- pyproject.toml (updated: pythonpath)
- pyrightconfig.json (new)

### Architecture Impact
DisturbanceRecord is now the established internal contract. All downstream systems
(providers, analytics, visualization, synchronization) must consume DisturbanceRecord
exclusively. No app/ module imports from src/. No UI, parser, or analytics imports
present in app/models/. Contract is parser-independent and visualization-independent.

### Performance Impact
- slots=True on all dataclasses: lower per-instance memory overhead vs __dict__
- waveform_data stored by reference: zero copy cost on construction
- validate() uses no per-sample loops; all checks are O(channels) not O(samples)
- has_channel() uses short-circuit generator: no intermediate list allocation

### Risks / Concerns
- Pre-existing src/ test failures (14 failed, 35 errors) are unrelated to app/ — legacy
  test data and parser issues in the old codebase. Not caused by this session.
- Python version note: venv uses Python 3.14.4 at runtime (not 3.11 as stated in pyproject.toml
  requires-python). slots=True and | union syntax work on 3.14 as well.
- object.__setattr__ used in one test to bypass slots for a non-DataFrame waveform_data
  scenario — this is intentional test-only behavior.

### Next Recommended Step
Implement provider pattern per directives/implement_provider_system.md:
- BaseProvider abstract class in app/providers/base/
- ProviderManager in app/providers/base/
- Stub ComtradeProvider, CsvProvider, ExcelProvider shells

---

## 2026-05-10 — Session 004

### Agent
Claude Code

### Task
Implement provider system foundation per directives/implement_provider_system.md

### Completed
- Created app/providers/base/exceptions.py — ProviderError, ProviderNotFoundError,
  ProviderLoadError, DuplicateProviderError (flat 4-class hierarchy)
- Created app/providers/base/base_provider.py — BaseProvider ABC with can_load() and load()
- Created app/providers/base/provider_registry.py — ProviderRegistry (ordered dict-backed,
  O(1) uniqueness, O(n) discovery, insertion-order preserved)
- Created app/providers/base/provider_manager.py — ProviderManager (registration validation,
  find_provider, load with ProviderLoadError wrapping and cause chaining)
- Updated app/providers/base/__init__.py — exports all 7 public symbols
- Created app/providers/comtrade/comtrade_provider.py — ComtradeProvider stub (.cfg, .comtrade)
- Created app/providers/csv/csv_provider.py — CsvProvider stub (.csv)
- Created app/providers/excel/excel_provider.py — ExcelProvider stub (.xlsx, .xls)
- Updated app/providers/comtrade/__init__.py, csv/__init__.py, excel/__init__.py
- Updated app/providers/__init__.py — flat re-export of entire provider surface
- Created tests/unit/test_provider_manager.py — 40 tests across 6 test classes
- Updated pyproject.toml — added [tool.pyright] section for IDE import resolution

### Files Modified
- app/providers/base/exceptions.py (new)
- app/providers/base/base_provider.py (new)
- app/providers/base/provider_registry.py (new)
- app/providers/base/provider_manager.py (new)
- app/providers/base/__init__.py (updated)
- app/providers/comtrade/comtrade_provider.py (new)
- app/providers/comtrade/__init__.py (updated)
- app/providers/csv/csv_provider.py (new)
- app/providers/csv/__init__.py (updated)
- app/providers/excel/excel_provider.py (new)
- app/providers/excel/__init__.py (updated)
- app/providers/__init__.py (updated)
- tests/unit/test_provider_manager.py (new)
- pyproject.toml (updated: [tool.pyright] section added)

### Architecture Impact
Provider ingestion boundary is now established. All downstream systems must load files
through ProviderManager.load() → DisturbanceRecord. No provider imports UI, visualization,
analytics, or src/ code. ProviderManager is the sole public ingestion entry point.
BaseProvider ABC enforces the contract at the type level. Exception hierarchy gives callers
predictable, debuggable error handling. Provider discovery is deterministic (insertion order).

### Performance Impact
- ProviderRegistry uses dict keyed by provider_name: O(1) duplicate detection
- Discovery is O(n·m) where n=providers, m=can_load cost (suffix check: O(1)) → O(n) total
- No file I/O in can_load() for current stubs — only suffix comparison
- ProviderLoadError wraps with __cause__ chain: zero overhead on the happy path

### Risks / Concerns
- Stubs raise NotImplementedError; ProviderManager wraps in ProviderLoadError — callers
  must handle ProviderLoadError when loading real files until parsers are implemented.
- provider_name is a class attribute with default "base" — no enforcement that subclasses
  override it. Registry duplicate detection will catch two "base" providers, but a forgotten
  name goes undetected until load time.
- IDE (pyright) still shows import resolution errors due to src/ as inferred import root.
  pyrightconfig.json + [tool.pyright] in pyproject.toml should resolve after IDE restart.

### Next Recommended Step
Implement COMTRADE parser per directives/implement_comtrade_provider.md (to be issued by
ChatGPT). ComtradeProvider.load() will be the first real provider implementation.

---

## 2026-05-10 — Session 005

### Agent
Claude Code

### Task
Author docs/COMTRADE_NORMALIZATION_POLICY.md — authoritative pre-implementation normalization
reference for ComtradeProvider.load()

### Completed
- Created docs/COMTRADE_NORMALIZATION_POLICY.md — 14 sections covering:
  - CFG structure parsing policy (rev year detection, field count tolerance, metadata, nominal freq, TIMEMULT)
  - Timestamp normalization (start_time/trigger_time parsing, time array construction, no timezone conversion)
  - DAT format handling (ASCII, Binary 16-bit, Binary32/float32; format detection from CFG ft field)
  - Analog channel scaling (a/b formula, Binary32 pre-scaled exception, skew not applied, PS not applied)
  - Digital channel normalization (vectorized bit extraction, normal_state preserved not inverted)
  - Multi-rate sampling (SamplingInformation construction, non-uniform time array from DAT, no resampling)
  - Phase naming normalization (preserve ph field as-is; mapping to A/B/C belongs in analytics)
  - Engineering unit normalization (no unit conversion; a/b scaling already encodes physical units)
  - Sampling integrity preservation (no resampling, no interpolation, partial file warning not error)
  - Parser responsibilities vs analytics layer (explicit boundary table)
  - What must be preserved from raw COMTRADE (19 preserved fields enumerated)
  - Error handling philosophy (fail-safe defaults vs hard ProviderLoadError, no silent partial failure)
  - DisturbanceRecord construction checklist (exact field mapping from CFG/DAT to contract fields)
  - Performance requirements (vectorized ops mandated: loadtxt/frombuffer/broadcast, float64/int8 dtypes)
  - Normalization policy summary table (parser action vs downstream action for each concern)

### Files Modified
- docs/COMTRADE_NORMALIZATION_POLICY.md (new)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (COMTRADE Normalization Policy documentation task added and marked COMPLETED)
- agent/REPOSITORY_STATE.md (updated to reflect new document)

### Architecture Impact
COMTRADE_NORMALIZATION_POLICY.md establishes the definitive normalization contract
before any parser code is written. It defines exactly:
- Which processing belongs inside ComtradeProvider (a/b scaling, bit extraction, time array)
- Which processing belongs outside (phase normalization, PU, RMS, skew correction, tz conversion)
- How all three DAT formats are handled uniformly
- How DisturbanceRecord fields map from CFG/DAT source fields

This document prevents architectural drift during implementation: ComtradeProvider.load()
implementors have a precise boundary definition. The analytics, visualization, and
synchronization layers have confirmed expectations about what DisturbanceRecord contains.

### Performance Impact
The document mandates:
- numpy.frombuffer for Binary/Binary32 (no Python loops)
- numpy.loadtxt or vectorized split for ASCII (no row iteration)
- Broadcast operations for a/b scaling and digital bit extraction
- pd.DataFrame construction from pre-built arrays (no row-by-row append)
These mandates prevent the most common parser performance antipatterns.

### Risks / Concerns
- The policy defines timezone = None in TimingInformation; the application layer applies
  a tz offset setting. This is correct but must be enforced at implementation time to
  prevent a parser developer from introducing a tz assumption.
- Binary32 (COMTRADE 2013) is documented but will require a real 2013-format test file
  for integration testing. No such file is confirmed available yet.
- Skew correction is deferred to analytics — this is architecturally correct but means
  early waveform display will show uncorrected skew. Acceptable for Phase 2.

### Next Recommended Step
Issue directives/implement_comtrade_provider.md (ChatGPT) and implement
ComtradeProvider.load() per this normalization policy. The policy document is the
primary implementation reference — the directive should cite it explicitly.

---

## 2026-05-10 — Session 006

### Agent
Claude Code

### Task
Implement ComtradeProvider — first real COMTRADE parser per
directives/implement_comtrade_provider.md and docs/COMTRADE_NORMALIZATION_POLICY.md

### Completed
- Created directives/implement_comtrade_provider.md — implementation SOP citing
  COMTRADE_NORMALIZATION_POLICY.md as primary reference
- Replaced ComtradeProvider stub with full production implementation:
  - _parse_cfg(): positional CFG parser; 1991/1999/2013 rev_yr; 10/13-field analog;
    3/5-field digital; TIMEMULT; fail-safe defaults with warnings for benign defects
  - _parse_ascii_dat(): numpy.loadtxt vectorized; column count validation; TIMEMULT applied
  - _parse_binary_dat(): numpy.frombuffer + structured dtype; file size validation; LE layout
  - _apply_analog_scaling(): broadcast (n_samples,nA)*(nA,)+(nA,) — no Python loops
  - _extract_digital_channels(): fully vectorized fancy-index bit extraction
  - _build_dataframe(): dict→DataFrame; duplicate name detection with warning
  - _build_record(): sample count validation + DisturbanceRecord assembly
  - BINARY32: raises ProviderLoadError with clear "not yet supported" message
- Updated tests/unit/test_provider_manager.py: stub test updated to expect ProviderLoadError
- Created tests/unit/test_comtrade_provider.py — 86 new tests, 10 test classes:
  TestComtradeProviderCanLoad, TestCfgParsing, TestParseAnalogLine, TestParseDigitalLine,
  TestParseTimestamp, TestAsciiDatParsing, TestBinaryDatParsing, TestAnalogScaling,
  TestDigitalExtraction, TestFullLoad, TestErrorHandling

### Files Modified
- directives/implement_comtrade_provider.md (new)
- app/providers/comtrade/comtrade_provider.py (full implementation — replaces stub)
- tests/unit/test_comtrade_provider.py (new — 86 tests)
- tests/unit/test_provider_manager.py (1 test updated)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (COMTRADE Parser marked COMPLETED)
- agent/REPOSITORY_STATE.md (updated)

### Architecture Impact
ComtradeProvider is the first real ingestion provider in Powerwave. All COMTRADE-specific
structures (_AnalogDef, _DigitalDef, _CfgData) are module-private. Only DisturbanceRecord
crosses the boundary. DisturbanceRecord contract, ProviderManager, ProviderRegistry, and
BaseProvider are all unchanged. No src/ imports. No analytics, UI, or rendering logic.
TimingInformation.timezone = None (no timezone conversion — per policy).
DigitalChannel.normal_state preserved as raw; inversion is display-layer responsibility.
AnalogChannel.description encodes ps_flag and non-zero skew for downstream reference.

### Performance Impact
- ASCII: numpy.loadtxt single-pass vectorized parse
- Binary: numpy.frombuffer structured dtype — zero-copy from raw bytes
- Scaling + digital extraction: pure numpy broadcast operations, O(n) in C layer
- DataFrame: dict→constructor, single allocation
- 152 tests (86 new + 66 existing) run in 1.41s

### Risks / Concerns
- BINARY32 deliberately deferred; raises ProviderLoadError with explanation
- Duplicate channel name rename means AnalogChannel.name may diverge from DataFrame column
- np.loadtxt emits UserWarning on empty file before ProviderLoadError — benign
- Skew correction deferred to analytics layer (correct architectural decision)

### Next Recommended Step
Issue directive for FastWaveformWidget (Phase 3 visualization) or CSV/Excel parsers
(Phase 2 data ingestion completion). COMTRADE ingestion pipeline is functional end-to-end.

---

## 2026-05-10 — Session 007

### Agent
Claude Code

### Task
Implement CsvProvider — second real ingestion provider per
directives/implement_csv_provider.md

### Completed
- Replaced CsvProvider stub with full production implementation:
  - _detect_time_column(): case-insensitive match against {"time","t","seconds","sec","timestamp","datetime"}
  - _infer_unit(): keyword-based unit inference (kV/A/Hz/MW/MVar/unknown) from column name
  - _is_digital_column(): conservative binary classifier; requires 0/1-only values AND
    status keyword in name (trip, pickup, breaker, status, cb, relay, alarm, open, close,
    signal, flag, state); boolean dtype always digital
  - _estimate_rate(): median inter-sample interval → Hz; 0.0 when indeterminate
  - _build_time_array(): handles numeric seconds, datetime/timestamp strings (pd.to_datetime),
    and no-time-column fallback (integer index, _EPOCH_FALLBACK, rate=0.0)
  - load(): validates file exists; reads via pd.read_csv; classifies columns; assembles
    DisturbanceRecord with correct analog_channels, digital_channels, waveform_data, metadata,
    sampling_info (rate estimated or 0.0), timing_info
- Updated tests/unit/test_provider_manager.py: CsvProvider stub test updated to expect
  ProviderLoadError (file not found) instead of NotImplementedError
- Created tests/unit/test_csv_provider.py — 65 tests across 11 test classes:
  TestCanLoad, TestDetectTimeColumn, TestInferUnit, TestIsDigitalColumn,
  TestEstimateRate, TestLoadNumericTime, TestLoadTimestampColumn, TestLoadNoTimeColumn,
  TestUnitInference, TestDigitalInference, TestErrorHandling,
  TestCsvProviderStubContract, TestChannelIndexing

### Files Modified
- app/providers/csv/csv_provider.py (full implementation — replaces stub)
- tests/unit/test_csv_provider.py (new — 65 tests)
- tests/unit/test_provider_manager.py (1 test updated: NotImplementedError → ProviderLoadError)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (CSV Parser marked COMPLETED)
- agent/REPOSITORY_STATE.md (updated)

### Architecture Impact
CsvProvider is the second real ingestion provider. All CSV-specific logic is encapsulated
inside the provider module. Only DisturbanceRecord crosses the boundary. DisturbanceRecord
contract, ProviderManager, ProviderRegistry, and BaseProvider are all unchanged.
No src/ imports. No analytics, UI, or rendering logic present.
TimingInformation.timezone = None (no tz assumption for CSV files).
Digital channel normal_state = 0 (raw bit states, inversion is display-layer responsibility).
Sampling rate 0.0 when no reliable time column (validated by DisturbanceRecord.validate()).

### Performance Impact
- pd.read_csv: single-pass vectorised pandas parse — no per-row Python loops
- Column classification: per-column O(1) dtype checks and set membership
- Time array: numpy-backed .astype(float) and .to_numpy() — no Python iteration
- Digital conversion: series.fillna().astype().to_numpy() — fully vectorised
- Analog conversion: pd.to_numeric vectorised; NaN fill skips unparseable columns
- DataFrame: dict→constructor, single allocation, no row-by-row append
- 153 tests (65 new + 88 existing) run in 3.02s

### Risks / Concerns
- Sampling rate is estimated from median diff — non-uniform CSV time columns will
  produce an approximate rate, which is correct for engineering purposes
- Digital inference requires BOTH binary values AND status keyword — purely numeric
  0/1 channels with arbitrary names remain analog (intentionally conservative)
- No resampling or gap filling (correct — preserves measurement integrity)
- UserWarning emitted for non-numeric columns (one pre-existing warning in tests — benign)

### Next Recommended Step
Issue directive for Excel provider (directives/implement_excel_provider.md) to complete
Phase 2 data ingestion, OR proceed to Phase 3 (FastWaveformWidget visualization engine).

---

## 2026-05-10 — Session 008

### Agent
Claude Code

### Task
Audit and consolidate .claude/skills/SKILL_comtrade_parser.md — determine what is covered
vs. useful vs. legacy, migrate valuable knowledge, then delete the skill file.

### Completed
- Read all 9 required documents + skill file + comtrade_provider.py + test_comtrade_provider.py
- Performed section-by-section audit of SKILL_comtrade_parser.md (10 sections):
  - CFG line structure → fully covered in COMTRADE_NORMALIZATION_POLICY.md §1.1-1.5
  - Analog 13/10-field parsing → fully covered in policy §1.2 + _parse_analog_line()
  - Digital 5/3-field parsing → fully covered in policy §1.2 + _parse_digital_line()
  - Physical value conversion → fully covered in policy §4.1
  - DAT file formats table → covered §3.2-3.4; BUT: table had FLOAT32 alias not in policy
  - Revision year handling → covered; BUT: BEN32 calendar-year quirk not documented
  - Multi-rate time construction → CONFLICT: skill used computed ideal timestamps (WRONG);
    policy mandates DAT timestamp field as authoritative (CORRECT) → discarded
  - Bay extraction logic → analytics layer, not parser; violates policy §10.2 → discarded
  - Complete parser class → old src/ DisturbanceRecord + src/ imports → entirely discarded
  - Validation tests → old field names, legacy test data paths → discarded
- Migrated 2 pieces of engineering knowledge to docs/COMTRADE_NORMALIZATION_POLICY.md:
  - §1.1: Added BEN32 calendar-year quirk note (some BEN32 write "2005"/"2024" as rev_yr)
  - §3.1: Added FLOAT32 vendor alias note (functionally identical to BINARY32; future fix)
- Deleted .claude/skills/SKILL_comtrade_parser.md — fully absorbed

### Files Modified
- docs/COMTRADE_NORMALIZATION_POLICY.md (2 notes added: BEN32 year quirk + FLOAT32 alias)
- .claude/skills/SKILL_comtrade_parser.md (DELETED)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (skill consolidation entry added)
- agent/REPOSITORY_STATE.md (updated)

### Architecture Impact
No provider code modified. No DisturbanceRecord modified. No tests affected.
COMTRADE_NORMALIZATION_POLICY.md now documents two real-world vendor variants that
were previously only in the legacy skill file:
1. BEN32 non-standard rev_yr → explains existing default-to-1999 behavior
2. FLOAT32 alias → flagged as future compatibility fix (not implemented yet)

### Performance Impact
None — documentation-only change.

### Risks / Concerns
- FLOAT32 alias not yet implemented in ComtradeProvider; files using ft=FLOAT32 will
  raise ProviderLoadError. Documented in policy for future resolution.
- Remaining skill files (.claude/skills/) not yet evaluated — consolidation is ongoing.
  Recommend auditing SKILL_signal_processing.md and SKILL_channel_mapping.md next as
  they may contain src/-centric patterns.

### Next Recommended Step
Continue skill consolidation (remaining 5 skill files) OR issue next implementation
directive (Excel provider or FastWaveformWidget).
---

## Session 009 — ExcelProvider + Channel Mapping Policy Consolidation
Date: 2026-05-10
Agent: Claude Code

### Work Completed

**Part A — SKILL_channel_mapping.md Consolidation**
- Audited .claude/skills/SKILL_channel_mapping.md section-by-section
- Created docs/CHANNEL_MAPPING_POLICY.md — 11-section authoritative policy covering:
  - Signal role taxonomy (17 analog roles, 7 digital roles)
  - Phase naming conventions (R/Y/B → A/B/C, a/b/c → A/B/C, L1/L2/L3 → A/B/C)
  - Signal code lookup table (VR/VY/VB, Ia/Ib/Ic, 3I0/3U0, etc.)
  - 8-priority analog detection algorithm
  - Digital keyword sets with alarm-exception-first rule
  - Complementary CB pair detection concept
  - Ingestion-layer vs analytics-layer scope boundary
  - Downstream usage table and future extensibility notes
- All src/-centric code references stripped (old DisturbanceRecord field names, raw_data
  access patterns, legacy ComtradeParser imports — all discarded as analytics-layer concerns)
- Deleted .claude/skills/SKILL_channel_mapping.md — fully absorbed into policy

**Part B — ExcelProvider Implementation**
- Replaced ExcelProvider stub with full implementation (app/providers/excel/excel_provider.py)
- .xlsx: full support via pd.read_excel(engine="openpyxl")
- .xls: raises ProviderLoadError with clear message explaining xlrd dependency gap
- Sheet selection: _select_sheet() scores each sheet by (rows × numeric-like columns),
  selects most data-rich sheet; samples only first 200 rows per sheet for performance
- Same column classification heuristics as CsvProvider:
  _detect_time_column / _infer_unit / _is_digital_column / _estimate_rate / _build_time_array
  (duplicated deliberately — providers are self-contained per architecture policy)
- Fixed 3 Pyright diagnostics:
  - pd.to_numeric overload → pd.Series cast with # type: ignore[assignment]
  - xl.sheet_names list[int|str] → [str(s) for s in xl.sheet_names]
  - str(c) redundant cast → f"{c}".strip() for column name normalisation
- Updated test_provider_manager.py: ExcelProvider stub test updated to expect
  ProviderLoadError (file not found) instead of NotImplementedError

**Tests**
- Created tests/unit/test_excel_provider.py — 68 tests, all passing
  TestCanLoad (6), TestDetectTimeColumn (8), TestInferUnit (10), TestIsDigitalColumn (7),
  TestEstimateRate (4), TestScoreSheet (3), TestSelectSheet (3), TestLoadNumericTime (6),
  TestLoadNoTimeColumn (3), TestUnitInference (4), TestDigitalInference (3),
  TestMultiSheetSelection (2), TestErrorHandling (4), TestChannelIndexing (2), TestMetadata (3)
- Full suite: 307 tests passing (86 COMTRADE + 65 CSV + 68 Excel + 26 model + 40 provider + 22 other)

### Files Modified
- app/providers/excel/excel_provider.py (IMPLEMENTED — replaces stub)
- tests/unit/test_excel_provider.py (NEW — 68 tests)
- docs/CHANNEL_MAPPING_POLICY.md (NEW — 11-section signal role policy)
- .claude/skills/SKILL_channel_mapping.md (DELETED)
- tests/unit/test_provider_manager.py (1 test updated: stub → real error)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (Excel Parser → COMPLETED; skill consolidation entry added)
- agent/REPOSITORY_STATE.md (updated)

### Architecture Impact
- ExcelProvider is now production-capable for .xlsx waveform files
- .xls support deferred pending xlrd installation decision (clear error message provided)
- Sheet selection heuristic (most data-rich sheet) is extensible — can be exposed as
  user preference or provider config in a future directive
- CHANNEL_MAPPING_POLICY.md establishes the authoritative reference for any future
  signal_role_detector or channel mapping dialog implementation
- docs/ now contains: ARCHITECTURE.md, COMTRADE_NORMALIZATION_POLICY.md, DATA_CONTRACT.md,
  LEGACY_CODEBASE_POLICY.md, PERFORMANCE_REQUIREMENTS.md, PROVIDER_PATTERN.md,
  SYSTEM_OVERVIEW.md, VISUALIZATION_CONTRACT.md, CHANNEL_MAPPING_POLICY.md (9 total)

### Performance Impact
- Sheet scoring samples first 200 rows only (pd.ExcelFile.parse(nrows=200)) — negligible
  overhead even for large workbooks with many sheets
- Column classification is O(n_columns) — same as CsvProvider

### Risks / Concerns
- xlrd not installed: .xls files will raise ProviderLoadError. Decision needed on whether
  to add xlrd to requirements.txt or document it as optional (recommend optional — .xls is legacy)
- FLOAT32 COMTRADE alias still unimplemented (carried forward from Session 008)

### Next Recommended Step
Issue next directive: FastWaveformWidget (directives/implement_fast_waveform_widget.md)
or ExcelProvider xls support (install xlrd + add test coverage).
Phase 2 data ingestion is now complete with CSV + COMTRADE + Excel (.xlsx).

---

## Session 010 — PyQtGraph Rendering Skill Consolidation + Phase 3A Directive
Date: 2026-05-10
Agent: Claude Code

### Work Completed

**SKILL_pyqt6_rendering.md Consolidation**
- Read and audited SKILL_pyqt6_rendering.md section-by-section against:
  - docs/VISUALIZATION_CONTRACT.md (what the engine must do)
  - docs/PERFORMANCE_REQUIREMENTS.md (performance targets)
  - docs/ARCHITECTURE.md (subsystem architecture)
  - docs/LEGACY_CODEBASE_POLICY.md (src/ isolation rules)
  - docs/CHANNEL_MAPPING_POLICY.md (color assignment reference)

- Created docs/VIEWPORT_RENDERING_POLICY.md — 15-section authoritative rendering policy
  Sections: PyQtGraph global config | Widget architecture | Curve lifecycle law |
  Decimation policy (4000-pt, stride algorithm, no interpolation) | Clip-to-view |
  Cursor rendering rules | Trigger line rules | X-axis sync (setXLink) |
  Dark engineering theme (full color table) | UI thread protection (QRunnable pattern) |
  DisturbanceRecord access pattern (new vs old field names) | Zoom-to-trigger algorithm |
  Rendering anti-patterns (5 forbidden patterns) | Out-of-scope table | Document authority

- Updated docs/VISUALIZATION_CONTRACT.md: added IMPLEMENTATION REFERENCE footer
  pointing to VIEWPORT_RENDERING_POLICY.md as the HOW companion to the WHAT contract

- Deleted .claude/skills/SKILL_pyqt6_rendering.md — fully absorbed into policy

**Created directives/implement_fast_waveform_widget.md (Phase 3A)**
- Scope: app/visualization/rendering/downsampling.py + app/visualization/widgets/fast_waveform_widget.py
         + tests/unit/test_downsampling.py
- Full API specification: decimate_for_display() algorithm, FastWaveformWidget class
  with all required public methods (set_record, set_visible_channels, zoom_to_trigger,
  set_cursor_pos, clear) and private methods (_on_x_range_changed, _update_viewport,
  _add_trigger_line, _add_cursor, _on_cursor_moved, _channel_color)
- Explicit NOT IN SCOPE list: VisualizationManager, SynchronizationManager,
  DigitalSignalWidget, AppState, docking, phasor canvas, RMS/ROCOF overlays
- Success criteria checklist
- Repository tracking update requirements

### Rendering Knowledge Retained vs Discarded

RETAINED (migrated to VIEWPORT_RENDERING_POLICY.md):
  - pg.setConfigOptions(useOpenGL=True, antialias=False) — mandatory initialization
  - setData() curve lifecycle law (never remove/re-add)
  - 4000-point display decimation limit with stride algorithm
  - QRunnable/QThreadPool worker pattern for file loading
  - InfiniteLine cursor (movable=True, DashLine, yellow) with sigPositionChanged
  - Trigger InfiniteLine (movable=False, DotLine, red, label='T')
  - setXLink() for multi-pane X-axis sync (Phase 3B+)
  - Full dark engineering color palette (background, phase A/B/C, earth, cursor, trigger)
  - showGrid(x=True, y=True, alpha=0.2) grid pattern
  - All 4 original anti-patterns (extended to 5 in policy)
  - DisturbanceRecord access field name mapping (new vs old)
  - Zoom-to-trigger algorithm with correct field references

DISCARDED (legacy/outdated/wrong architecture):
  - ChannelCanvas(pg.GraphicsLayoutWidget) — wrong base class for FastWaveformWidget
    (contract mandates pg.PlotWidget; GraphicsLayoutWidget is Phase 3B scope)
  - record.analogue_channels / ch.raw_data / record.time_array / ch.channel_id
    / ch.visible / ch.colour — all old DisturbanceRecord field names, superseded
    by new data contract (analog_channels, waveform_data, timing_info, etc.)
  - PhasorCanvas (QPainter-based) — Phase 5+ scope, not relevant to Phase 3A
  - AppState singleton pattern — Phase 4+ scope (global state manager deferred)
  - Trigger: `src/ui/` reference in skill trigger line — discarded, app/ only

### Files Created / Modified
- docs/VIEWPORT_RENDERING_POLICY.md (NEW — 15 sections)
- directives/implement_fast_waveform_widget.md (NEW — Phase 3A directive)
- docs/VISUALIZATION_CONTRACT.md (IMPLEMENTATION REFERENCE footer added)
- .claude/skills/SKILL_pyqt6_rendering.md (DELETED)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (FastWaveformWidget updated; skill consolidation entry added)
- agent/REPOSITORY_STATE.md (updated)

### Architecture Alignment
- VIEWPORT_RENDERING_POLICY.md correctly uses new DisturbanceRecord field names
  throughout (waveform_data, timing_info, analog_channels, digital_channels)
- Directive correctly scopes Phase 3A to single-widget foundation only,
  deferring multi-pane manager to Phase 3B per phased development policy
- PyQtGraph global config correctly placed at app/main.py level, not widget
- pg.PlotWidget inheritance confirmed per VISUALIZATION_CONTRACT.md

### Risks / Concerns
- UI widget tests (test_fast_waveform_widget.py) require a QApplication and
  display. This is excluded from Phase 3A test scope; only test_downsampling.py
  (pure NumPy, display-free) is in scope. Widget integration testing deferred.
- Remaining skill files (.claude/skills/): SKILL_INDEX.md, SKILL_merging_timesync.md,
  SKILL_pmu_power.md, SKILL_signal_processing.md — not touched per directive scope.
  Recommend auditing in future sessions as Phase 5 (analytics) approaches.

### Next Recommended Step
Execute directives/implement_fast_waveform_widget.md — Phase 3A implementation.
Target: app/visualization/rendering/downsampling.py + app/visualization/widgets/fast_waveform_widget.py
        + tests/unit/test_downsampling.py

---

## Session 011 — N-Axis Single Canvas Architecture Lock + FlexiblePlotCanvas Directive
Date: 2026-05-10
Agent: Claude Code

### Work Completed

**Architecture revision: SIGRA-style N-Axis Single Canvas**

ChatGPT issued an architecture update mid-Session-010. The visualization
architecture was revised from:
  FastWaveformWidget(pg.PlotWidget) — single Y-axis, single parameter
to:
  FlexiblePlotCanvas(pg.GraphicsLayoutWidget) — N independent Y-axes (ViewBox per
  analog parameter), shared X-axis, SIGRA-style multi-parameter canvas.

Digital signals are separated into a distinct DigitalEventTimeline component (Phase 3B).

**VIEWPORT_RENDERING_POLICY.md updates (docs/)**
- §2 Widget Architecture: Revised to FlexiblePlotCanvas(pg.GraphicsLayoutWidget) + MultiAxisManager
- §8 X-Axis Synchronization: Updated for secondary ViewBox setXLink pattern
- §16 N-Axis ViewBox Multi-Parameter Architecture: NEW 9-subsection chapter covering:
  - Architecture overview (one ViewBox per param, shared X)
  - Primary PlotItem setup
  - Secondary ViewBox creation and scene registration
  - sigResized geometry synchronization (MANDATORY — §16.4)
  - Axis positioning strategy (right-stacking for Phase 3A)
  - Independent Y-axis scaling behavior
  - Cursor and trigger line placement in N-Axis canvas
  - setXLink() X-axis sync behavior
  - Performance note for N-Axis (N × 4000 pts per viewport update)
- §17 Digital Event Timeline: NEW section defining the separate component,
  its architecture separation rationale, rendering model, and Phase 3B scope
- §15 Document Authority: Updated reference to implement_flexible_plot_canvas.md

**VISUALIZATION_CONTRACT.md updates (docs/)**
- Architecture diagram: updated to show FlexiblePlotCanvas + DigitalEventTimeline
- PRIMARY VISUALIZATION COMPONENTS: replaced FastWaveformWidget with FlexiblePlotCanvas + MultiAxisManager
- N-AXIS SINGLE CANVAS ARCHITECTURE: NEW section (mandatory architecture mandates)
- DIGITAL EVENT TIMELINE: NEW section (separate component, Phase 3B scope)
- FLEXIBLEPLOTCANVAS RESPONSIBILITIES: replaced FASTWAVEFORMWIDGET RESPONSIBILITIES
- VISUALIZATION DIRECTORY STRUCTURE: updated with correct filenames and phase labels
- IMPLEMENTATION REFERENCE footer: references VIEWPORT_RENDERING_POLICY.md

**implement_fast_waveform_widget.md (directives/) — SUPERSEDED**
- Added SUPERSEDED banner at top explaining architectural revision
- File retained as archived reference

**directives/implement_flexible_plot_canvas.md — NEW (Phase 3A)**
- Full implementation directive for FlexiblePlotCanvas + MultiAxisManager + downsampling
- 4 target files: downsampling.py, flexible_plot_canvas.py, multi_axis_manager.py, test_downsampling.py
- Complete API specification for all public and private methods
- MultiAxisManager detailed class spec (add_axis, register, remove_axis, _sync_geometries)
- FlexiblePlotCanvas constructor, set_record, add_parameter, remove_parameter,
  set_visible_channels, zoom_to_trigger, set_cursor_pos, clear, all private methods
- test_downsampling.py: 26 tests across 6 test classes
- Implementation constraints (12 items)
- Phase 3A success criteria checklist

**directives/generate_fault_test_record.md — NEW (placeholder)**
- Codex directive placeholder for synthetic fault dataset generation
- Dataset spec: VA/VB/VC/IA/IB/IC/FREQ/ROCOF + digital CB_status/TRIP_A/PICKUP_A
- Waveform characteristics (pre-fault/fault/post-fault amplitudes)
- Activation condition: FlexiblePlotCanvas must be complete first

### Architecture Alignment Notes

- FlexiblePlotCanvas correctly uses GraphicsLayoutWidget as base per N-Axis pattern
- Secondary ViewBoxes MUST be geometry-synced via sigResized (§16.4) — critical rule
- cursor and trigger line go on primary_plot only (InfiniteLines span full canvas height)
- Digital channels explicitly excluded from FlexiblePlotCanvas per §17 mandate
- implement_fast_waveform_widget.md is superseded but retained for audit trail

### Files Created / Modified
- docs/VIEWPORT_RENDERING_POLICY.md (§2, §8, §15 revised; §16, §17 ADDED)
- docs/VISUALIZATION_CONTRACT.md (N-Axis + Digital Event Timeline architecture + directory structure)
- directives/implement_fast_waveform_widget.md (SUPERSEDED banner added)
- directives/implement_flexible_plot_canvas.md (NEW — Phase 3A directive)
- directives/generate_fault_test_record.md (NEW — Codex placeholder)
- agent/HANDOFF.md (this entry appended)
- agent/TASK.md (updated)
- agent/REPOSITORY_STATE.md (updated)

### Risks / Concerns
- PyQtGraph sigResized geometry synchronization (§16.4) is the most likely implementation
  pitfall — secondary ViewBoxes will appear empty if geometry sync is not wired correctly
- Right-only axis stacking (Phase 3A simplification) limits readability at 5+ axes;
  left/right alternation should be addressed in Phase 3B
- Widget UI tests not in Phase 3A scope; test_downsampling.py covers the critical
  performance-path logic independently of display availability

### Next Recommended Step
Execute directives/implement_flexible_plot_canvas.md — Phase 3A implementation.

---

## Session 012 — Phase 3A: FlexiblePlotCanvas Implementation
**Date:** 2026-05-10
**Agent:** Claude Code (claude-sonnet-4-6)
**Directive:** directives/implement_flexible_plot_canvas.md

### Scope Executed
Implemented the complete Phase 3A visualization core:
- `app/visualization/rendering/downsampling.py` — pure-NumPy `decimate_for_display()`
- `app/visualization/managers/multi_axis_manager.py` — `MultiAxisManager` + `_AxisEntry`
- `app/visualization/widgets/flexible_plot_canvas.py` — `FlexiblePlotCanvas`
- `tests/unit/test_downsampling.py` — 28 tests (all non-GUI, NumPy-only)

### Implementation Details

**downsampling.py**
- Validation: rejects non-1-D arrays ("1-D" match), rejects length mismatch ("length mismatch")
- Clipping: boolean mask `(time >= t_start) & (time <= t_end)`
- t_start > t_end: silently swapped
- Decimation: ceiling-division stride `(n + max_points - 1) // max_points` — guarantees output ≤ max_points
- Bug fixed during testing: floor `//` stride could produce 4167 points for max_points=4000 with 50000 inputs; ceiling stride fixed this
- All outputs cast to float64 regardless of input dtype

**multi_axis_manager.py**
- First parameter: reuses primary PlotItem left axis (no new ViewBox)
- Secondary parameters: bare `pg.ViewBox()` added to scene, linked via `setXLink(primary_plot)`, independent right-side `pg.AxisItem`
- `_pending_axis` dict pattern: `add_axis()` stages the AxisItem internally; `register()` retrieves it — avoids broken caller-passes-axis_item API
- `_sync_geometries()`: connected to `primary_vb.sigResized`; sets geometry + calls `linkedViewChanged` for all secondary ViewBoxes
- `clear()`: removes secondary ViewBoxes and AxisItems from scene; leaves primary untouched

**flexible_plot_canvas.py**
- Inherits `pg.GraphicsLayoutWidget` (not PlotWidget)
- `cursor_moved = pyqtSignal(float)` for Phase 3B sync
- `_channel_color()`: module-level function; phase-detection heuristic on `ch.name.lower()`
- `set_record()`: caches numpy arrays once from DataFrame (`_time_cache`, `_data_cache`); iterates `record.analog_channels` only (digital excluded)
- `clear()`: disconnects old `_axis_manager._sync_geometries` from `sigResized` before rebuilding manager (prevents double-connection leak)
- `_update_viewport()`: hot path — only `decimate_for_display()` + `setData()`; no DataFrame ops
- `set_cursor_pos()`: uses `blockSignals(True/False)` to prevent cursor_moved re-emission in sync loops
- `add_parameter()`: supports Phase 5 analytics overlays (name → data injected into `_data_cache`)
- `zoom_to_trigger()`: centres viewport on `trigger_time - start_time` ± window_s

### Test Results
```
335 passed, 4 warnings in 2.61s
```
- 307 pre-existing tests: all still passing (no regressions)
- 28 new test_downsampling.py tests: all passing after stride fix

### Files Created
- `app/visualization/rendering/downsampling.py`
- `app/visualization/managers/multi_axis_manager.py`
- `app/visualization/widgets/flexible_plot_canvas.py`
- `tests/unit/test_downsampling.py`

### Files Modified
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (FlexiblePlotCanvas → COMPLETED; Phase 3B set as next target)
- `agent/REPOSITORY_STATE.md` (updated test count, implemented systems, next action)

### Architecture Notes
- Digital channels: explicitly excluded from FlexiblePlotCanvas (Phase 3B DigitalEventTimeline)
- `pg.setConfigOptions(useOpenGL=True, antialias=False)` NOT called in widget — belongs in `app/main.py`
- No GUI tests: PyQtGraph widget tests require display; downsampling is the critical hot-path and tested independently

### Next Recommended Step
Phase 3B: DigitalEventTimeline + VisualizationManager + multi-canvas cursor synchronization.

---

## Session 013 — Phase 3B: DigitalEventTimeline Implementation
**Date:** 2026-05-10
**Agent:** Claude Code (claude-sonnet-4-6)
**Directive:** directives/implement_digital_event_timeline.md (created and executed in this session)

### Scope Executed
Implemented the complete Phase 3B digital channel visualization core:
- `directives/implement_digital_event_timeline.md` — directive authored and executed
- `app/visualization/rendering/digital_transforms.py` — pure-NumPy digital processing (4 functions)
- `app/visualization/widgets/digital_event_timeline.py` — DigitalEventTimeline widget
- `tests/unit/test_digital_transforms.py` — 39 tests (all non-GUI, NumPy-only)

### Implementation Details

**digital_transforms.py (4 public functions):**
- `digital_role_color(name)`: keyword heuristic, alarm-exception checked first per CHANNEL_MAPPING_POLICY §2
  Colors: DIG_CB=#FF8800, DIG_AR=#4488FF, DIG_INTERTRIP=#FF44FF, DIG_TRIP=#FF2222, DIG_PICKUP=#FFAA00, DIG_GENERIC=#AAAAAA
- `extract_transitions(time, data)`: reduces N samples to M transition points; binary coercion; sentinel appended for step function completeness; validates 1-D and length
- `clip_digital_to_viewport(t_trans, d_trans, t_start, t_end)`: viewport clip with carry-state at left edge; handles all-before-viewport, all-after-viewport, and mid-recording viewport cases; searchsorted-based O(log M) state lookup
- `build_step_series(t, d, y_offset, track_height)`: expands transition data to explicit step-function segments for correct fill behavior; no stepMode dependency

**digital_event_timeline.py:**
- Inherits `pg.PlotWidget` (single PlotItem) — all digital channels as curves with vertical offsets
- Each channel i: baseline y = i × 1.5; HIGH state filled up to y_offset + 1.0
- `cursor_moved = pyqtSignal(float)` for Phase 3B+ cursor sync
- `set_record()`: builds tracks from `record.digital_channels`, caches `_time_cache`, extracts transitions once per channel
- `_add_track()`: creates `pg.PlotDataItem` with `fillLevel=y_offset`, `brush=color+'55'` for semi-transparent HIGH fill
- `_update_y_axis()`: sets Y-range and custom left-axis tick labels (channel names at track midpoints)
- `link_x_to(view_or_plot)`: X-links timeline to FlexiblePlotCanvas primary_plot for synchronized navigation
- `set_cursor_pos(t)`: blockSignals pattern prevents cursor_moved re-emission
- `_update_viewport()`: hot path — only clip_digital_to_viewport + build_step_series + setData()
- Trigger line + movable cursor: same patterns as FlexiblePlotCanvas (VIEWPORT_RENDERING_POLICY §6, §7)
- `clear()` + `_restore_plot_config()`: clean teardown, safe for re-use

### Test Results
```
374 passed, 4 warnings in 3.18s
```
- 335 pre-existing tests: all still passing (no regressions)
- 39 new test_digital_transforms.py tests: all passing first run

### Files Created
- `directives/implement_digital_event_timeline.md`
- `app/visualization/rendering/digital_transforms.py`
- `app/visualization/widgets/digital_event_timeline.py`
- `tests/unit/test_digital_transforms.py`

### Files Modified
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (DigitalEventTimeline → COMPLETED; Phase 3B+ set as next)
- `agent/REPOSITORY_STATE.md` (updated state, test count)

### Architecture Notes
- `digital_transforms.py` is Qt-free, mirroring `downsampling.py` pattern — testable independently
- Transition extraction is O(N) once on record load; viewport clip is O(log M + viewport_transitions)
- No VisualizationManager or SynchronizationManager implemented (later Phase 3B+)
- `link_x_to()` provides the integration point with FlexiblePlotCanvas without requiring a coordinator

### Risks / Concerns
- `pg.PlotDataItem` fill with explicit step series: correct and PyQtGraph-version-independent
- Y-axis tick labels truncate for long channel names — future UX concern, not a Phase 3B blocker
- No GUI tests: PyQtGraph widget tests require display; digital_transforms is the critical path and tested independently
- `link_x_to()` must be called after both widgets are shown (PyQtGraph scene requirement for setXLink)

### Next Recommended Step
Phase 3B+: VisualizationManager + SynchronizationManager — wires FlexiblePlotCanvas and DigitalEventTimeline together with shared cursor sync and coordinated record loading.

---

## Session 014 — Phase 3C: VisualizationManager Implementation
**Date:** 2026-05-10
**Agent:** Claude Code (claude-sonnet-4-6)
**Directive:** directives/implement_visualization_manager.md (created and executed in this session)

### Scope Executed
Implemented the Phase 3C coordination layer:
- `directives/implement_visualization_manager.md` — directive authored and executed
- `app/visualization/managers/visualization_manager.py` — VisualizationManager class
- `tests/unit/test_visualization_manager.py` — 32 tests (all mock-based, no display required)

### Implementation Details

**visualization_manager.py — design decisions:**
- Plain Python class (NOT QObject): matches VISUALIZATION_CONTRACT.md spec (`class VisualizationManager: pass`), avoids QApplication dependency in tests, simpler lifecycle
- Lifetime contract: caller MUST keep the manager alive; pyqtSignal stores weak references to bound methods — GC of manager silently drops cursor_moved connections
- `_canvas._primary_plot` accessed by design: the DigitalEventTimeline.link_x_to() docstring explicitly names this field as the intended argument; within-package coupling between coordination partners is intentional
- `_x_linked` flag: tracks whether setXLink has been established; when linked, zoom/pan on canvas propagates to timeline at the PyQtGraph C++ layer — no duplicate Python-side calls needed

**Coordination behaviors:**
- `set_record(record)`: calls `canvas.set_record(record)` then `timeline.set_record(record)` — both widgets load the same record atomically from the coordinator's perspective
- `clear()`: calls `canvas.clear()` then `timeline.clear()`; resets `_record = None`
- `link_x_axis()`: calls `timeline.link_x_to(canvas._primary_plot)`; sets `_x_linked = True`. MUST be called after both widgets are in a Qt scene (PyQtGraph requirement)
- `zoom_to_trigger(window_s)`: calls `canvas.zoom_to_trigger(window_s)`; calls `_zoom_timeline_to_trigger()` only when NOT X-linked (avoids duplicate zoom when X-link propagates naturally)
- `reset_viewport()`: resets X-range to [0, t_max]; timeline reset only when not X-linked
- `set_cursor_pos(t)`: calls `canvas.set_cursor_pos(t)` + `timeline.set_cursor_pos(t)` — for external callers to move both cursors without emitting
- `_on_canvas_cursor_moved(t)`: forwards to `timeline.set_cursor_pos(t)` — no re-emit (blockSignals in receiver prevents loop)
- `_on_timeline_cursor_moved(t)`: forwards to `canvas.set_cursor_pos(t)` — same loop-prevention

**Test strategy:**
- `unittest.mock.MagicMock` stubs for canvas and timeline — no display needed
- MagicMock `__len__` returns 0 by default: `len(time_col) == 0` → `t_max = 1.0` (fallback branch) — enables deterministic `setXRange(0.0, 1.0, padding=0)` assertions
- `_zoom_timeline_to_trigger` test configures `total_seconds.return_value = 1.0` via `__sub__` mock to avoid `TypeError` from `max(0.0, MagicMock() - float)` in Python 3.14
- `patch.object` used to verify `_zoom_timeline_to_trigger` routing without triggering record arithmetic

### Test Results
```
406 passed, 4 warnings in 3.26s
```
- 374 pre-existing tests: all still passing (no regressions)
- 32 new test_visualization_manager.py tests: all passing

### Files Created
- `directives/implement_visualization_manager.md`
- `app/visualization/managers/visualization_manager.py`
- `tests/unit/test_visualization_manager.py`

### Files Modified
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (VisualizationManager → COMPLETED; next target updated)
- `agent/REPOSITORY_STATE.md` (updated test count, implemented systems)

### Architecture Notes
- VisualizationManager is NOT a QObject: plain Python class per VISUALIZATION_CONTRACT.md spec. No singleton, no global state, no event bus.
- No Qt/PyQtGraph imports in visualization_manager.py — purely coordinates widget APIs
- All three tracking files updated per WORKFLOW_AGENT.md requirements
- Phase 3C is visualization-layer only: no provider, analytics, or UI layer dependencies

### Risks / Concerns
- Weak reference lifetime: manager must stay in scope while cursor sync is needed. If the UI panel that owns the manager goes out of scope, cursor_moved connections drop silently. Mitigated: UI panels hold strong references to their child coordinators.
- `link_x_axis()` call-order dependency: must be called after widgets are shown in a Qt scene. Calling it in a layout's `showEvent` or after `widget.show()` is the safe pattern.
- `_canvas._primary_plot` access: private attribute accessed by design (within-package coordination). If FlexiblePlotCanvas ever adds a public `primary_plot` property, update the call site here.

### Next Recommended Step
Phase 3D or Phase 4 scoping: The visualization engine (analog canvas + digital timeline + coordinator) is now feature-complete for basic disturbance visualization. Recommended next steps:
  Option A — SynchronizationManager (cursor_manager.py / viewport_controller.py) for multi-instance cursor coordination
  Option B — UI integration: wire VisualizationManager into PowerwaveMainWindow, add file-open workflow via QRunnable provider loader
  Option C — Generate synthetic fault test record (directives/generate_fault_test_record.md) to enable end-to-end visualization testing

---

## Session 015 — Phase 4A: Basic Viewer Workflow
**Date:** 2026-05-10
**Agent:** Claude (claude-sonnet-4-6)
**Session type:** Phase 4A implementation
**Status at handoff:** COMPLETE

### Objective
Implement the first end-to-end operational Powerwave viewer:
File → Open → background load → VisualizationManager → display.

### What Was Implemented

**Directive:**
- `directives/implement_basic_viewer_workflow.md` — Phase 4A specification

**app/ui/main_window/main_window.py** — PowerwaveMainWindow + threading helpers + utility functions
  - `_FILE_FILTER` — combined file dialog filter for .cfg/.comtrade/.csv/.xlsx
  - `_build_provider_manager()` — module-level; registers ComtradeProvider, CsvProvider, ExcelProvider; testable without Qt
  - `_format_load_status(record)` — module-level; formats status bar string from DisturbanceRecord; testable without Qt
  - `_WorkerSignals(QObject)` — cross-thread signals: `finished(object)` + `error(str)`
  - `_LoadWorker(QRunnable)` — background file loader; emits finished or error on completion
  - `PowerwaveMainWindow(QMainWindow)`:
    - `__init__`: creates canvas, timeline, vis_manager as instance attributes (lifetime guarantee)
    - `showEvent`: calls `link_x_axis()` once after first show (via `_x_axis_linked` flag)
    - `_build_layout()`: QSplitter(Vertical) — canvas(stretch=3) over timeline(stretch=1); QStatusBar
    - `_build_menu()`: File menu — Open(Ctrl+O) + separator + Exit
    - `_open_file_dialog()`: QFileDialog.getOpenFileName; routes to `_load_file(Path)`
    - `_load_file(path)`: updates status; creates _LoadWorker; starts on QThreadPool
    - `_on_record_loaded(record)`: vis_manager.set_record; status bar update; window title update
    - `_on_load_error(message)`: status bar error; QMessageBox.critical

**app/ui/main_window/__init__.py** — exports `PowerwaveMainWindow`

**app/main.py** — updated:
  - `pg.setConfigOptions(useOpenGL=True, antialias=False, foreground='w', background='#1E1E1E')` added before any pg widget instantiation (VIEWPORT_RENDERING_POLICY §1)
  - Import moved from inline class to `from app.ui.main_window import PowerwaveMainWindow`

**tests/unit/test_main_window_workflow.py** — 19 non-GUI tests:
  - `TestBuildProviderManager` (11 tests): isinstance, 3 providers, names, find_provider for .cfg/.csv/.xlsx, independence of second call, insertion order
  - `TestFormatLoadStatus` (8 tests): str type, basename-only, analog count, digital count, sampling rate, unknown rate, zero analog, zero digital

### Test Results
```
425 passed, 4 warnings in 7.51s
```
- 406 pre-existing tests: all still passing (no regressions)
- 19 new test_main_window_workflow.py tests: all passing

### Files Created
- `directives/implement_basic_viewer_workflow.md`
- `app/ui/main_window/main_window.py`
- `tests/unit/test_main_window_workflow.py`

### Files Modified
- `app/ui/main_window/__init__.py` (was empty stub)
- `app/main.py` (pg.setConfigOptions + new import)
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase 4A → COMPLETED)
- `agent/REPOSITORY_STATE.md` (425 tests, new files)

### Architecture Notes
- `self._vis_manager` held as instance attribute to prevent GC of cursor_moved connections (pyqtSignal weak-ref contract)
- `link_x_axis()` guarded by `_x_axis_linked` flag in `showEvent` — safe for multiple show/hide cycles
- File loading is fully off-thread: `_LoadWorker(QRunnable)` + `_WorkerSignals(QObject)` + `QThreadPool.globalInstance()`
- `pg.setConfigOptions()` called once at module level in `app/main.py` before `QApplication` and any widget construction
- PowerwaveMainWindow does NOT import from src/ — legacy isolation maintained

### Risks / Concerns
- No end-to-end visual test yet: UI correctness (layout, signals) requires manual verification or a synthetic test record (generate_fault_test_record.md)
- QThreadPool default thread count depends on CPU; no explicit max-thread-count limit set
- `_on_record_loaded` updates UI from the signal connection — this is safe because PyQt cross-thread signals deliver via queued connection (QObject receiver runs in UI thread)

### Next Recommended Step
  Option A — Generate synthetic fault test record (directives/generate_fault_test_record.md) for end-to-end visual testing
  Option B — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option C — Analytics foundation: RMS / ROCOF / frequency (Phase 5)

---

## Session 016 — Phase D1: Urgent Waveform Display Detour
**Date:** 2026-05-10
**Agent:** Claude (claude-sonnet-4-6)
**Session type:** Phase D1 implementation
**Status at handoff:** COMPLETE

### Objective
Create the minimum practical foundation for displaying disturbance waveforms from mixed sources (raw AC at high sampling rate + RMS/system data at lower rate) without breaking existing architecture.

### What Was Implemented

**app/data/signal_metadata.py** — `SignalMetadata(frozen=True, slots=True)`
  Per-channel display metadata not storable in locked AnalogChannel:
  name, unit, source, signal_type, sampling_rate_hz, time_offset_seconds, display_group

**app/data/time_alignment.py** — `build_display_time_seconds()`
  Converts numeric seconds or datetime-like arrays to float64 seconds relative to reference_start.
  Applies explicit offset; never mutates input; handles empty input; pandas-backed datetime parsing.

**app/data/synthetic.py** — three generators:
  `make_high_rate_record()` — raw three-phase V/I waveforms at 6400 Hz
    - VA_raw/VB_raw/VC_raw (always); IA_raw/IB_raw/IC_raw (optional)
    - Fault profile: step voltage sag [fault_start_s, fault_end_s)
    - Returns SyntheticDisturbanceResult(record, signal_metadata)
  `make_low_rate_record()` — RMS/system channels at 100 Hz
    - MW, MVar, Frequency (always); ROCOF (optional)
    - Fault profile: MW dip, MVar spike, frequency dip
    - Intentional time_offset_s (default 0.01s) to simulate misaligned sources
  `make_mixed_disturbance_record()` — merged single DisturbanceRecord
    - Low-rate channels interpolated onto high-rate time axis via np.interp
    - sampling_info.sampling_rates = [6400.0, 100.0] — original rates preserved
    - All channel SignalMetadata merged; indices re-contiguized

**app/analytics/basic_conversions.py** — `sliding_rms()`, `to_per_unit()`
  `sliding_rms(values, window_samples)` — O(N) cumulative-sum RMS; output length N-W+1
  `to_per_unit(values, base_value)` — vectorized division; raises ValueError on base=0

**app/visualization/channel_grouper.py** — `group_channels_for_display()`
  Groups channel names by display group: voltage_raw / current_raw / power / frequency / rocof / digital / other
  Priority: SignalMetadata.display_group > name heuristics
  Name heuristics tuned for synthetic naming convention (V*_raw, I*_raw, MW, MVar, Frequency, ROCOF)

**app/data/__init__.py** — exports SignalMetadata, build_display_time_seconds
**app/analytics/__init__.py** — exports sliding_rms, to_per_unit

### Test Results
```
487 passed, 4 warnings in 11.82s
```
- 425 pre-existing tests: all still passing (no regressions)
- 62 new tests across 4 new test files: all passing
  - test_time_alignment.py: 12 tests
  - test_basic_conversions.py: 16 tests
  - test_synthetic_disturbance.py: 23 tests
  - test_visualization_grouping.py: 11 tests

### Files Created
- `app/data/__init__.py`
- `app/data/signal_metadata.py`
- `app/data/time_alignment.py`
- `app/data/synthetic.py`
- `app/analytics/basic_conversions.py`
- `app/visualization/channel_grouper.py`
- `tests/unit/test_time_alignment.py`
- `tests/unit/test_basic_conversions.py`
- `tests/unit/test_synthetic_disturbance.py`
- `tests/unit/test_visualization_grouping.py`

### Files Modified
- `app/analytics/__init__.py` (was empty stub; now exports basic_conversions)
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase D1 → COMPLETED)
- `agent/REPOSITORY_STATE.md` (487 tests, new modules listed)

### Architecture Notes
- DisturbanceRecord contract NOT modified — SignalMetadata is a separate parallel structure
- AnalogChannel NOT modified — channel grouper works with existing contract
- Mixed-rate records use a single shared time axis (high-rate) in waveform_data
- Original sampling rates documented in SamplingInformation.sampling_rates (multi-rate list)
- np.interp for low-rate → high-rate alignment: clamps at boundaries (acceptable for display)
- VisualizationManager NOT modified — it already handles any DisturbanceRecord passed to it
- Provider pattern NOT modified

### Risks / Concerns
- np.interp clamps for t_high values outside t_low range: t in [0, 0.01) uses first low-rate value.
  For the 0.01s offset, this is 64 samples at 6400 Hz — negligible for display purposes.
- channel_grouper heuristics are conservative: VBus would classify as voltage_raw due to lower[1]='b'.
  This is acceptable for Phase D1 but should be reviewed against CHANNEL_MAPPING_POLICY.md for production use.
- sliding_rms returns length N-W+1 (not N): callers that expect same-length output will need padding.

### Next Recommended Step
  Option A — Wire synthetic mixed record into the viewer for end-to-end visual verification
  Option B — Multi-pane display: update PowerwaveMainWindow to call group_channels_for_display and add each group as a separate pane
  Option C — Synthetic fault record file generation (directives/generate_fault_test_record.md)
  Option D — SynchronizationManager for multi-panel cursor coordination (Phase 3D)

---

## Session 017 — Phase D2: First Real Multi-Panel Mixed-Source Waveform Display
**Date:** 2026-05-10
**Agent:** Claude (claude-sonnet-4-6)
**Session type:** Phase D2 implementation
**Status at handoff:** COMPLETE

### Objective
Wire the Phase D1 synthetic mixed-source record into the actual Powerwave UI as a stacked multi-panel display. Produce the first visually useful version of Powerwave for the urgent waveform display need.

### What Was Implemented

**app/visualization/managers/visualization_manager.py** — two additions:

  `_make_filtered_record(record, channel_names)` — module-level pure function
    Returns a new DisturbanceRecord containing only the specified analog channels.
    DataFrame sliced via `.loc[:, cols_present]`; indices recontiguized; metadata/timing shared by reference.
    Used by display_grouped_record() so each panel canvas renders only its group's data.
    Avoids the `set_visible_channels` viewport-override problem (hidden channels re-render on pan/zoom).

  `VisualizationManager.display_grouped_record(record, signal_metadata, canvas_factory)` — new public method
    Calls `group_channels_for_display()` to build group → channel_names dict.
    For each non-empty analog group: calls `_make_filtered_record()`, creates canvas via factory, calls `canvas.set_record(filtered)`.
    Digital channels: routes to `self._timeline.set_record(record)`.
    Stores result in `self._panel_canvases`; updates `self._record`.
    Returns `dict[group_name, canvas]` — caller manages layout.
    `canvas_factory=None` → lazy import of `FlexiblePlotCanvas` (testable without QApplication).

  `VisualizationManager.panel_canvases` — new read-only property
    Returns shallow copy of `_panel_canvases` dict.

  `VisualizationManager.__init__` — added `self._panel_canvases: dict = {}`

**app/ui/main_window/main_window.py** — D2 extensions:

  `_PANEL_ORDER = ["voltage_raw", "current_raw", "power", "frequency", "rocof", "other"]`
    Module-level constant controlling panel stacking order.

  `_build_layout()` — refactored to call `_restore_standard_layout()` + init `_panel_canvases`/`_grouped_timeline`

  `_restore_standard_layout()` — new: rebuilds the Phase 4A two-pane splitter; clears `_panel_canvases`
    Called from `_build_layout()` and from `_on_record_loaded()` when grouped layout is active.

  `_rebuild_grouped_layout(panel_canvases, record)` — new: replaces central widget with stacked group canvases
    Panels inserted in `_PANEL_ORDER`; unrecognised groups appended.
    Digital timeline added only if `record.digital_channels` is non-empty.
    Calls `QTimer.singleShot(0, self._link_panel_x_axes)` to defer X-axis linking.

  `_link_panel_x_axes()` — new: links all grouped panel canvases to master (first canvas) via `setXLink`
    Also links grouped timeline to master if present.

  `_build_menu()` — added Tools menu: "Load Synthetic Mixed Disturbance" (Ctrl+T)

  `_on_load_synthetic_mixed()` — new: generates synthetic record, calls `display_grouped_record`,
    calls `_rebuild_grouped_layout`, updates title + status bar.

  `_on_record_loaded()` — updated: calls `_restore_standard_layout()` if grouped layout active,
    then routes to existing `set_record()` path (Phase 4A behavior preserved).

### Why _make_filtered_record Instead of set_visible_channels

`FlexiblePlotCanvas.set_visible_channels()` clears hidden channel curves on call, but
`_update_viewport()` (triggered by sigXRangeChanged) re-renders ALL channels from `_data_cache`,
overriding the hidden state on every pan/zoom. Creating a per-group filtered record eliminates
this issue: each canvas only has its group's channels in `_data_cache`.

### X-Axis Synchronization

`QTimer.singleShot(0, _link_panel_x_axes)` fires after `_rebuild_grouped_layout` returns,
ensuring all group canvases are parented into the visible Qt scene before `setXLink` is called
(VIEWPORT_RENDERING_POLICY §8 requirement).

### Test Results
```
524 passed, 4 warnings in 8.49s
```
- 487 pre-existing tests: all still passing (no regressions)
- 37 new tests: all passing
  - test_visualization_grouped_display.py: 22 tests
  - test_main_window_synthetic_action.py: 15 tests

### Files Created
- `tests/unit/test_visualization_grouped_display.py`
- `tests/unit/test_main_window_synthetic_action.py`

### Files Modified
- `app/visualization/managers/visualization_manager.py`
- `app/ui/main_window/main_window.py`
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase D2 → COMPLETED)
- `agent/REPOSITORY_STATE.md` (524 tests, D2 modules listed)

### Manual Verification Instructions
```
.venv/Scripts/python.exe -m app.main
# Then: Tools → Load Synthetic Mixed Disturbance
# Expected: 4 stacked panels (voltage/current/power/frequency)
# All panels scroll/zoom together horizontally
```

### Risks / Concerns
- `_link_panel_x_axes` accesses `canvas._primary_plot` (private attr by design, within-package partner)
- Grouped panel canvases are GC'd when `_restore_standard_layout()` is called; no explicit cleanup needed
- X-axis linking is one-shot (called once per `_rebuild_grouped_layout`); window minimize/restore does not re-link. Acceptable for Phase D2.

### Next Recommended Step
  Option A — Manual verification: run app and visually confirm stacked panel display
  Option B — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option C — Load real COMTRADE + CSV together via provider merge workflow
  Option D — Analytics foundation: RMS overlay on raw waveform (Phase 5)

---

## Session 018 — Phase D3: Real Multi-Source Record Merge Workflow
**Date:** 2026-05-10
**Agent:** Claude (claude-sonnet-4-6)
**Session type:** Phase D3 implementation
**Status at handoff:** COMPLETE

### Objective
Implement the real multi-source record merge workflow — load two independent
DisturbanceRecords (e.g. COMTRADE + CSV) into a single co-aligned display
without destructive resampling.

### What Was Implemented

**app/data/multi_source_session.py** (NEW)
  `SourceRecord` — dataclass(slots=True): source_id, provider_type, record,
    signal_metadata, original_start_time, sampling_rates.
    Preserves original DisturbanceRecord and its temporal context.

  `MultiSourceSession` — dataclass(slots=True): sources list; add_source(),
    source_count(), is_empty(), source_ids(), get_source(source_id).
    Non-destructive container — originals never modified.

**app/data/display_alignment.py** (NEW)
  `determine_reference_start(sources)` — finds earliest temporal anchor.
    Checks original_start_time first; falls back to record.timing_info.start_time.
    Returns datetime (if any datetime anchors), float (if all numeric), or None.

  `compute_relative_offsets(sources, reference_start)` — computes per-source
    time offset in seconds. Handles datetime↔datetime, float↔float; type
    mismatches and None anchors produce 0.0.

  `build_aligned_display_time(source, reference_start)` — builds aligned float64
    display time array for a single source. Adds cross-source offset to waveform
    'time' column values. Non-destructive (reads, never writes original).

**app/data/__init__.py** — extended exports:
  SourceRecord, MultiSourceSession, determine_reference_start,
  compute_relative_offsets, build_aligned_display_time.

**app/visualization/managers/visualization_manager.py** — two additions:

  `_apply_time_offset(record, offset_seconds)` — module-level pure helper.
    Returns display copy of record with time column shifted. Returns original
    unchanged for offset_seconds == 0.0. Metadata/channels shared by reference.

  `VisualizationManager.display_multi_source_session(session, canvas_factory)` — new public method.
    Determines reference_start across all sources; computes per-source offsets.
    For each source/group: calls _make_filtered_record, _apply_time_offset, factory(), canvas.set_record.
    Panel keys are f"{source_id}/{group_name}" to prevent collision.
    Digital channels routed to self._timeline (first source wins).
    Stores result in _panel_canvases; sets _record to first source's record.
    canvas_factory=None → lazy FlexiblePlotCanvas import (testable without QApplication).

**app/ui/main_window/main_window.py** — D3 extensions:

  `_make_source_record(source_id, record, provider_type)` — module-level helper.
    Wraps a DisturbanceRecord in a SourceRecord; auto-generates SignalMetadata
    for all analog channels with source=source_id.

  File menu: "Open Multi-Source…" (Ctrl+M) — opens multi-file QFileDialog.
    Single selection: falls through to standard _load_file() path.
    Multi selection: synchronous _load_multi_source() (runs on UI thread;
    file I/O is fast for typical disturbance record sizes).

  `_open_multi_source_dialog()` — file picker, routes to _load_file or _load_multi_source.

  `_load_multi_source(paths)` — iterates paths; loads each via ProviderManager;
    builds SourceRecord per file; collects errors via QMessageBox.warning;
    calls _on_multi_source_loaded on non-empty session.

  `_on_multi_source_loaded(session)` — restores standard layout if grouped was active;
    calls display_multi_source_session; calls _rebuild_grouped_layout;
    updates title + status bar.

### Architectural Decision: No Destructive Resampling
Original DisturbanceRecords are preserved inside SourceRecord.record — never
merged or resampled. _apply_time_offset creates a display copy (new DataFrame)
with only the time column shifted. All other fields are shared by reference.
The MultiSourceSession owns the originals; callers hold independent references.

### Test Results
```
591 passed (tests/unit/ only), 4 warnings
```
- 524 pre-existing tests: all still passing (no regressions)
- 67 new tests: all passing
  - test_multi_source_session.py: 18 tests
  - test_display_alignment.py: 19 tests
  - test_display_multi_source.py: 17 tests
  - test_main_window_multi_source.py: 13 tests

### Files Created
- `app/data/multi_source_session.py`
- `app/data/display_alignment.py`
- `tests/unit/test_multi_source_session.py`
- `tests/unit/test_display_alignment.py`
- `tests/unit/test_display_multi_source.py`
- `tests/unit/test_main_window_multi_source.py`

### Files Modified
- `app/data/__init__.py`
- `app/visualization/managers/visualization_manager.py`
- `app/ui/main_window/main_window.py`
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase D3 → COMPLETED)
- `agent/REPOSITORY_STATE.md` (591 tests, D3 modules listed)

### Manual Verification Instructions
```
.venv/Scripts/python.exe -m app.main
# Then: File → Open Multi-Source… (Ctrl+M)
# Select a .cfg + .csv file
# Expected: stacked panels, one per source/group, X-axes linked
```

### Risks / Concerns
- _load_multi_source runs synchronously on UI thread (acceptable for typical
  disturbance file sizes; use _LoadWorker pattern for very large files).
- X-axis linking across multi-source panels uses same QTimer.singleShot(0) mechanism
  as Phase D2 grouped layout.
- Panel key format "source_id/group_name" means _rebuild_grouped_layout's
  _PANEL_ORDER matching will never match (none contain "/"), so all multi-source
  panels land in the "unrecognized" bucket and appear in dict-insertion order.
  Acceptable for Phase D3; a smarter sort could be added later.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)

---

## 2026-05-15 / Session D4.5A

### Agent
Codex

### Task
Phase D4.5A — SynchronizationManager

### Completed
- Created `app/visualization/managers/synchronization_manager.py`
- Implemented `SynchronizationManager` for registered PyQtGraph canvases:
  - X-axis pan/zoom propagation via `sigXRangeChanged`
  - master cursor propagation via `InfiniteLine.sigPositionChanged`
  - active source panel tracking
  - unregister and clear lifecycle
  - recursion prevention with `_sync_depth` and duplicate-range echo suppression
- Integrated synchronization through `VisualizationManager`
  - standard canvas + digital timeline registration
  - grouped multi-panel registration API
  - strong manager ownership for `SignalProxy` lifetime
- Updated grouped main-window layout to use `SynchronizationManager` instead of direct inter-panel `setXLink`
- Added `PowerwaveMainWindow.closeEvent()` cleanup for synchronization signal proxies
- Stabilized `FlexiblePlotCanvas._sync_curve_view()` by removing redundant secondary ViewBox `setXRange()` calls; secondary ViewBoxes already follow the primary via `setXLink`
- Added unit tests for synchronized zoom, pan, cursor propagation, unregister behavior, recursion prevention, and digital timeline synchronization
- Added runtime Qt tests for analog panel synchronization and digital timeline synchronization

### Files Modified
- `app/visualization/managers/synchronization_manager.py` — new manager
- `app/visualization/managers/__init__.py` — export
- `app/visualization/managers/visualization_manager.py` — owns and registers synchronization manager
- `app/visualization/widgets/flexible_plot_canvas.py` — removed redundant secondary ViewBox X-range forcing
- `app/ui/main_window/main_window.py` — grouped panel sync registration and close cleanup
- `tests/unit/test_synchronization_manager.py` — new unit coverage
- `tests/unit/test_runtime_qt_widgets.py` — runtime synchronization coverage
- `tests/unit/test_visualization_manager.py` — updated X-link expectation to manager registration

### Architecture Impact
- Synchronization is centralized in visualization managers and remains parser/provider/data independent.
- The manager coordinates existing canvases and existing `InfiniteLine` cursor objects; it does not create duplicate cursors during pan/zoom and does not own waveform data.
- Existing rendering paths remain decimation-driven; synchronization only changes view ranges and cursor values.

### Recursion Prevention
- `_sync_depth` suppresses synchronous feedback while propagating range/cursor updates.
- Duplicate X-range and cursor echoes from follower panels are ignored when they match the current shared state.
- Cursor updates call existing widget `set_cursor_pos()` methods, which block `InfiniteLine` signals while applying propagated positions.
- Secondary FlexiblePlotCanvas ViewBoxes no longer receive manual X-range forcing in `_sync_curve_view()`; their established `setXLink()` relationship handles intra-canvas alignment.

### Cursor Propagation
- Each registered canvas uses its existing `_cursor` `InfiniteLine`.
- `InfiniteLine.sigPositionChanged` is connected through `SignalProxy` when a Qt application exists; a direct fallback is used for no-QApplication unit tests.
- Dragging one cursor updates all other registered cursors to the same engineering time position.

### Performance Impact
- Synchronization performs O(number of registered panels) range/cursor updates only.
- No waveform data is recopied, replotted wholesale, or reloaded during synchronization.
- Range changes continue to trigger existing viewport decimation in each canvas; cursor movement does not re-decimate data.
- Real PULU manifest integration and runtime COMTRADE display remained responsive in validation.

### Test Results
```
.venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_runtime_qt_widgets.py -q
88 passed

.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py -q
50 passed
```

Warnings observed:
- Existing pandas timestamp inference warning in CSV intelligence path.
- Existing offscreen/OpenGL PyQtGraph warnings in runtime widget tests.
- COMTRADE sample `rev_yr '2005'` warning, already handled by provider fallback.

### Risks / Concerns
- Full-suite run was not repeated in this session; focused synchronization, runtime Qt, grouped display, multi-source, and PULU integration paths passed.
- Runtime tests still show offscreen OpenGL warnings in this environment; assertions pass and no recursion/hang remains.
- Prompt referenced `agent/TASKS.md`, but repository uses `agent/TASK.md`; updated the live tracker.

### Next Recommended Step
Phase D4.5B candidate — optional operator-facing cursor readout/tooltip hook, still without analytics overlays.
  Option C — Manual integration test with real COMTRADE + CSV files

---

## Session 019 — Phase D3.1: Sample Data & Inspection Utilities
**Date:** 2026-05-10
**Agent:** Claude (claude-sonnet-4-6)
**Session type:** Infrastructure / tooling phase
**Status at handoff:** COMPLETE

### Objective
Create reusable sample file infrastructure, lightweight inspection tools, and
a manifest builder that reduce token consumption across future sessions by
keeping Claude working from summaries rather than raw binary waveform data.

### What Was Implemented

**samples/ structure (extended)**
  - `samples/README.md` — naming conventions, structure, workflow documentation,
    token-efficiency philosophy
  - `samples/excel/` — created (was missing from existing structure)

**tools/inspect_comtrade.py** (NEW)
  - `ComtradeMetadata` dataclass — all metadata fields, no waveform arrays
  - `parse_cfg(cfg_path)` — reads only the CFG; never touches the DAT data
  - `_locate_cfg(path)` — accepts .cfg file or directory containing one
  - `_locate_dat(cfg_path)` — finds companion DAT (reports size only, no load)
  - `format_text_summary()` / `format_json_summary()` — text and JSON output
  - CLI: `python tools/inspect_comtrade.py <path> [--json]`
  - Handles: 1991/1999/2013 revisions, all analog/digital counts, missing DAT

**tools/inspect_csv_timeseries.py** (NEW)
  - `CsvMetadata`, `CsvTimestampInfo`, `TimestampAmbiguity` dataclasses
  - `inspect_file(path, sample_rows)` — reads at most N rows; never loads full file
  - `_find_timestamp_column()` — keyword match + partial match fallback
  - `_check_single_ambiguity(value)` — cross-parses with `dayfirst=False` and
    `dayfirst=True`; reports is_ambiguous when both parse differently
  - `_infer_timestamp_format()` — detects numeric, unix_epoch, ISO 8601,
    and locale-ambiguous datetime strings
  - `_compute_interval()` — median inter-sample interval; pandas 3.0 compatible
    (removed deprecated `infer_datetime_format`)
  - Ambiguity detection: explicit — never silently guesses M/D vs D/M
  - CLI: `python tools/inspect_csv_timeseries.py <path> [--json] [--sample-rows N]`

**tools/build_event_manifest.py** (NEW)
  - `SourceEntry`, `AlignmentInfo`, `EventManifest` dataclasses
  - `_to_yaml()` — minimal recursive YAML emitter (no PyYAML dependency)
  - `_repo_relative(path, root)` — forward-slash repo-relative path strings
  - `_compute_alignment(sources)` — finds earliest start_time; computes per-source
    offset in seconds; sources without timestamps get 0.0
  - `build_manifest(event_id, comtrade_paths, csv_paths, excel_paths, root)`
    — calls inspection utilities, builds EventManifest
  - `_manifest_to_yaml(manifest)` — serializes to YAML string
  - Alignment: reference_source = earliest start; all offsets relative to it
  - Paths: always repository-relative, forward-slash
  - CLI: `python tools/build_event_manifest.py --event-id X --comtrade ... --csv ...`

**tools/__init__.py** (NEW) — makes tools/ importable as a package in tests

**pyproject.toml** — added `"tools"` to `[tool.pytest.ini_options] pythonpath`
  so tests can `from inspect_comtrade import ...` directly

### Key Design Decisions

1. **No waveform loading** — inspect_comtrade reads only the CFG header; DAT is
   stat()'d for size only. Safe for 100MB+ recordings.

2. **Explicit ambiguity** — inspect_csv_timeseries never silently resolves
   M/D vs D/M ambiguity. It cross-parses each sample value with both dayfirst
   settings and reports concrete US vs EU interpretations.

3. **No PyYAML dependency** — `_to_yaml()` is a minimal hand-rolled emitter
   for the specific manifest schema. Keeps requirements.txt lean.

4. **Repository-relative paths in manifests** — manifests committed to the repo
   work on any developer machine or CI without path fixups.

5. **pandas 3.0 compatible** — removed deprecated `infer_datetime_format` from
   all datetime parsing calls.

6. **Expected warnings suppressed** — `warnings.catch_warnings()` in
   `_check_single_ambiguity()` and related functions silences the pandas
   "dayfirst mismatch" advisory (the cross-format parsing is intentional).

### Test Results
```
673 passed, 4 warnings in 10.73s  (tests/unit/ only)
```
- 591 pre-existing tests: all passing (zero regressions)
- 82 new tests:
  - test_inspect_comtrade.py: 31 tests
  - test_inspect_csv_timeseries.py: 27 tests
  - test_build_event_manifest.py: 24 tests

### Files Created
- `samples/README.md`
- `samples/excel/` (directory)
- `tools/__init__.py`
- `tools/inspect_comtrade.py`
- `tools/inspect_csv_timeseries.py`
- `tools/build_event_manifest.py`
- `tests/unit/test_inspect_comtrade.py`
- `tests/unit/test_inspect_csv_timeseries.py`
- `tests/unit/test_build_event_manifest.py`

### Files Modified
- `pyproject.toml` (added "tools" to pythonpath)
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase D3.1 → COMPLETED)
- `agent/REPOSITORY_STATE.md` (673 tests, tools/ structure listed)

### Manual Verification
```bash
# Inspect a COMTRADE recording (no DAT load):
python tools/inspect_comtrade.py samples/comtrade/<event_id>/

# Check CSV timestamp ambiguity:
python tools/inspect_csv_timeseries.py samples/csv/<event_id>.csv

# Build a manifest:
python tools/build_event_manifest.py \
  --event-id pulu_20260306 \
  --comtrade samples/comtrade/pulu_20260306/ \
  --csv samples/csv/pulu_20260306.csv \
  --output samples/manifests/pulu_20260306.yaml
```

### Risks / Limitations
- `_check_single_ambiguity` uses `pd.to_datetime` without an explicit format
  string — behavior may shift across pandas versions. OK for a dev tool;
  not appropriate for production ingestion.
- Multi-sheet Excel: inspect_csv_timeseries reads the first sheet only (same as
  pandas default). Complex workbooks may need the full ExcelProvider.
- YAML emitter covers the current manifest schema only. Deeply nested or
  complex structures would need PyYAML.

### Next Recommended Step
  Option A — Add real sample files (upload COMTRADE + CSV for pulu_20260306)
             and run tools/ against them to generate the first real manifest
  Option B — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option C — Analytics: RMS overlay on raw waveform (Phase 5)

---

## Session 020 — SignalMetadata Electrical Reference Completion + Real Sample Data Integration
Date: 2026-05-10
Agent: Claude Code (claude-sonnet-4-6)
Phase: D3.1 Addendum + Sample Data Validation

### Completed Work

**1. SignalMetadata electrical reference annotation — make_low_rate_record()**
- `make_low_rate_record()` channel_specs tuple extended to 5-tuple (name, values, unit, group, etype)
- MW → `electrical_type="power"`, MVar → `electrical_type="power"`
- Frequency → `electrical_type="frequency"`, ROCOF → `electrical_type="rocof"`
- `make_high_rate_record()` voltage channels had already been annotated (Session 019)

**2. New tests — TestElectricalReferenceMetadata (9 tests)**
- Added to `tests/unit/test_synthetic_disturbance.py`
- Covers: voltage electrical_type, phase_ground reference, current electrical_type,
  current no phase_reference, MW/MVar power type, Frequency type, ROCOF type,
  nominal_voltage defaults None, mixed record preserves all annotations

**3. CLI formatter portability fixes**
- Replaced `─` (U+2500 box-drawing) with `-` in both inspect_comtrade.py and inspect_csv_timeseries.py
- Replaced `⚠` (U+26A0 warning sign) with `(!)` in inspect_csv_timeseries.py
- Replaced `—` (U+2014 em dash) with `-` in inspect_csv_timeseries.py
- Root cause: Windows cp1252 terminal cannot encode these Unicode characters
- All three tools now produce clean output on cp1252 terminals

**4. Real sample data added to repository**
- `samples/comtrade/pulu_20260306.cfg` + `.dat` — PULU substation, 2026-03-06
  - 42 analog channels (kV/kA, 275kV bus), 88 digital channels
  - 5000 Hz, 32693 samples, 6.5 s duration, ASCII format
  - COMTRADE revision: 1999
- `samples/csv/pulu_20260306.csv` — 1-minute system demand/frequency data
  - Columns: Time, System Demand (MW), Tie-Line (MW), Frequency (Hz)
  - Date format: M/D/YYYY (ambiguous — confirmed as month-first by context)
  - 50 rows covering 17:25–18:14 on 2026-03-06

**5. First manifest generated**
- `samples/manifests/pulu_20260306.yaml` — generated via build_event_manifest.py
- All paths are repo-relative (forward slashes)
- Alignment: CSV reference_start = 2026-03-06T17:25:00;
  COMTRADE offset = +2348.817733 s (fault at ~18:04, CSV starts at 17:25)

**6. README + documentation updated**
- `samples/README.md` updated to reflect flat layout (`samples/comtrade/<event_id>.cfg`
  not subdirectory style)
- CLI examples updated to use file paths not directory paths
- Adding new event instructions updated

### Files Modified
- `app/data/synthetic.py` — make_low_rate_record() electrical_type annotations
- `tests/unit/test_synthetic_disturbance.py` — 9 new electrical reference tests
- `tools/inspect_comtrade.py` — ASCII-only formatter
- `tools/inspect_csv_timeseries.py` — ASCII-only formatter + notes
- `samples/README.md` — flat layout + updated CLI examples
- `agent/REPOSITORY_STATE.md` — 682 tests, pulu sample, manifest listed
- `agent/HANDOFF.md` (this entry)

### Files Created
- `samples/comtrade/pulu_20260306.cfg`
- `samples/comtrade/pulu_20260306.dat`
- `samples/csv/pulu_20260306.csv`
- `samples/manifests/pulu_20260306.yaml`

### Test Results
```
682 passed, 4 warnings in 10.59s  (tests/unit/ only)
```
- 673 pre-existing tests: all passing (zero regressions)
- 9 new tests: TestElectricalReferenceMetadata all passing

### Key Engineering Notes

1. **PULU sample has no nominal_voltage in SignalMetadata** — the CFG channels
   have primary/secondary ratios (275kV/0.11kV for VT) but SignalMetadata
   nominal_voltage must be populated by ComtradeProvider, not the inspector.
   This is a Phase 5 / analytics concern.

2. **CSV date format confirmed M/D/YYYY** — `3/6/2026 17:25` is March 6, 2026,
   matching the COMTRADE start time. Inspector correctly flags ambiguity;
   operator/user must confirm interpretation before use.

3. **Manifest alignment offset** — 39-minute offset between CSV (system-wide
   operational data starting at 17:25) and COMTRADE (fault capture at 18:04)
   is expected. They are different instrument types at different time scales.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics: RMS overlay on raw waveform (Phase 5)
  Option C — ComtradeProvider integration test: load pulu_20260306 real recording
             and verify all 42 analog + 88 digital channels load correctly

## 2026-05-10 / Session 021

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4 — Manifest-Based Multi-Source Loading + CSV Column Classification

### Completed

**1. SignalMetadata extended with classification provenance fields**
- `app/data/signal_metadata.py` — 3 new optional fields (backward-compatible):
  `confidence: float | None`, `inferred_from: str | None`, `requires_user_confirmation: bool = False`

**2. CSV Column Classifier implemented**
- `app/data/column_classifier.py` — `CONFIRMATION_THRESHOLD = 0.80`
- `ColumnClassification` frozen dataclass: signal_type, unit, display_group, confidence, inferred_from, requires_user_confirmation
- `classify_csv_column(name, values=None)` — name-exact > name-keyword > value-profile priority chain
- `classify_csv_columns(df, timestamp_column)` — DataFrame batch API
- Exact rules: frequency/freq/mw/mvar/rocof/kv/p/q/f etc. (34 entries)
- Keyword rules: "reactive power" before "active power" (ordering fixes substring-match bug)
- Value-profile: near-50Hz/60Hz → frequency; near-1.0 tight → voltage; large spread → active_power
- Single-letter names (P/Q/F) always below CONFIRMATION_THRESHOLD

**3. Manifest loader implemented**
- `app/data/manifest_loader.py` — YAML manifest → MultiSourceSession
- `load_manifest(path)` — yaml.safe_load; validates event_id + sources present
- `_infer_comtrade_channel_type(name)` — "KPDN1 VR" → voltage; "SGT1 IB_HV" → current
- `_get_source_file_path(src_def, type, root, id)` — handles paths.cfg, paths.csv, path fallback
- `_build_comtrade_signal_metadata(channels, voltage_ref, manifest_cols)` — annotates phase_reference on voltage channels
- `_build_csv_signal_metadata(channels, manifest_cols)` — confidence/signal_type/display_group from manifest
- `_parse_timestamp(value)` — 5 ISO-8601 formats
- `build_session_from_manifest(manifest_path, root=None, provider_manager=None)` — full pipeline

**4. app/data/__init__.py extended**
- Added exports: ColumnClassification, classify_csv_column, classify_csv_columns, build_session_from_manifest, load_manifest

**5. build_event_manifest.py updated**
- Fixed `_to_yaml` list-of-dicts indentation bug (extra spaces in rendered YAML)
- Extended SourceEntry with `voltage_reference` and `column_classifications`
- `_build_csv_entry()` now calls `classify_csv_columns` and populates columns section
- Added project root to sys.path so `app.data` is importable when run as a script

**6. inspect_csv_timeseries.py updated**
- Added `format_classification_summary()` helper
- Added `--classify` CLI flag

**7. pulu_20260306.yaml regenerated**
- Full columns section on csv_ops: Time.1 (unknown/0.0), System Demand (active_power/0.85), Tie-Line (active_power/0.70/confirm), Frequency (frequency/0.95)
- voltage_reference: phase_ground on comtrade_main (manually added)
- Alignment offsets preserved

**8. Main window UI extended**
- `app/ui/main_window/main_window.py`: File→Open Event Manifest… (Ctrl+E), Tools→Load Sample PULU Event
- `_load_manifest()`: calls build_session_from_manifest, logs low-confidence columns to status bar

**9. PyYAML dependency added**
- `requirements.txt`: PyYAML==6.0.3

**10. Test suite: 3 new test files (120 tests)**
- `tests/unit/test_column_classifier.py` — 47 tests (7 classes)
- `tests/unit/test_manifest_loader.py` — 41 tests (5 classes) — fixed _make_analog_record missing nominal_frequency
- `tests/unit/test_manifest_session_integration.py` — 34 tests (4 classes)

### Files Modified
- `app/data/signal_metadata.py` — +3 optional classification fields
- `app/data/__init__.py` — extended exports
- `app/ui/main_window/main_window.py` — manifest menu actions + _load_manifest()
- `tools/build_event_manifest.py` — indentation fix + classification output
- `tools/inspect_csv_timeseries.py` — --classify flag
- `samples/manifests/pulu_20260306.yaml` — regenerated with column classification
- `requirements.txt` — PyYAML==6.0.3
- `agent/REPOSITORY_STATE.md` — 802 tests, D4 files listed
- `agent/HANDOFF.md` (this entry)

### Files Created
- `app/data/column_classifier.py`
- `app/data/manifest_loader.py`
- `tests/unit/test_column_classifier.py`
- `tests/unit/test_manifest_loader.py`
- `tests/unit/test_manifest_session_integration.py`

### Architecture Impact
- `build_session_from_manifest()` is the new canonical entry point for manifest-driven multi-source loading
- Column classification is purely advisory: original DisturbanceRecord never mutated
- SignalMetadata is now the complete per-channel display contract (type + unit + group + confidence)
- `voltage_reference: phase_ground` in manifest → `phase_reference` on all voltage-type SignalMetadata entries
- Provider injection (`provider_manager=None`) pattern enables testability without real files

### Test Results
```
802 passed, 4 warnings  (tests/unit/ only)
```
- 682 pre-existing tests: all passing (zero regressions)
- 120 new D4 tests: all passing
- Bugs fixed during D4: keyword rule ordering (reactive/active substring match), _make_analog_record missing nominal_frequency

### Key Engineering Notes

1. **Keyword ordering matters** — "active power" is a substring of "reactive power". The `_KEYWORD`
   list in column_classifier.py must keep the `("reactive power",)` rule before `("active power", ...)`.

2. **Manifest paths are repo-relative** — `_resolve_path` treats relative paths as relative to `root`
   (defaulting to `Path.cwd()`). When running from the project root this is transparent; tests must
   pass `root=tmp_path` when writing manifests to tmp_path.

3. **voltage_reference not auto-detected** — The manifest builder cannot infer whether a COMTRADE
   recording used phase-to-ground or phase-to-phase VTs. Must be manually added to the manifest.

4. **PyYAML safe_load only** — `yaml.safe_load` used throughout; `yaml.load` never used.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — UI: column classification review dialog (for columns with requires_user_confirmation=True)
  Option D — ComtradeProvider integration test: load pulu_20260306 real recording

## 2026-05-10 / Session 022

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4.1 — Data Intelligence & Persistent Mapping Rules

### Completed

**1. Intelligence module structure created**
- `app/data/intelligence/__init__.py` — full public exports
- `app/data/intelligence/models.py` — 4 frozen dataclasses: SourceFingerprint, MappingRule, TimestampRule, ConfidencePromotion
- `app/data/intelligence/fingerprints.py` — deterministic SHA-256 column fingerprinting
- `app/data/intelligence/mapping_rules.py` — load/save/find/apply mapping rules
- `app/data/intelligence/timestamp_rules.py` — load/save/find timestamp rules
- `app/data/intelligence/intelligence_manager.py` — IntelligenceManager orchestrator

**2. SourceFingerprint**
- Deterministic `column_signature` = 16-hex SHA-256 prefix of sorted, normalised column names
- Fields: vendor, station, export_type, source_kind, column_signature (all optional)
- `fingerprints_match(a, b)` — conflict only when both sides have non-None, differing values; None fields are wildcards
- `build_fingerprint_from_record(record)` — reads analog_channels + metadata.station_name

**3. MappingRule persistence**
- YAML format: `rules: [...]` under `config/column_mapping_rules.yaml`
- match_type: "exact" (full normalised name) or "keyword" (substring)
- Global rules (source_fingerprint=None) apply to all sources
- Fingerprint-scoped rules apply only to matching sources (and take priority over global)
- Round-trip: save_mapping_rules() + load_mapping_rules() preserves all fields

**4. TimestampRule persistence**
- YAML format: `rules: [...]` under `config/timestamp_rules.yaml`
- source_pattern → date_format mapping; confirmed_by_operator flag
- `find_matching_timestamp_rule()` — case-insensitive exact match

**5. ConfidencePromotion audit trail**
- Frozen dataclass: original_confidence, promoted_confidence, original_inferred_from, promoted_inferred_from, rule_match_pattern
- Returned alongside updated ColumnClassification — never stored in DisturbanceRecord

**6. IntelligenceManager**
- `classify_column(name, values, fingerprint)` → (ColumnClassification, ConfidencePromotion | None)
- `classify_columns(df, timestamp_column, fingerprint)` → dict of tuples
- `resolve_timestamp_format(source_pattern)` → TimestampRule | None
- `build_fingerprint(column_names, source_type, station)` → SourceFingerprint
- `extract_rules_from_manifest(manifest_data)` → list[MappingRule] (explicit, not automatic)
- `save_rules_from_manifest(manifest_data, path)` → int (count); merges by match_pattern
- `save_timestamp_rule(rule, path)` → None; merges by source_pattern
- Graceful no-config: missing YAML files → empty rule lists; all classifiers still work

**7. Config files created**
- `config/column_mapping_rules.yaml` — starts empty (`rules: []`) with full documentation header
- `config/timestamp_rules.yaml` — starts empty with documentation header
- `config/source_fingerprints.yaml` — starts empty with documentation header

**8. app/data/__init__.py extended**
- Added exports: IntelligenceManager, ConfidencePromotion, MappingRule, SourceFingerprint, TimestampRule

**9. Design decision: column_classifier.py NOT modified**
- IntelligenceManager wraps classify_csv_column() — does not inject into it
- Avoids circular import (column_classifier → intelligence → column_classifier)
- D4 classify_csv_column() works identically with or without IntelligenceManager

**10. Tests (118 new)**
- test_source_fingerprints.py — 34 tests
- test_mapping_rules.py — 33 tests
- test_timestamp_rules.py — 17 tests
- test_intelligence_manager.py — 20 tests
- test_intelligence_integration.py — 14 tests (D4 backward compat + SCADA alias promotion + manifest workflow)

### Files Created
- `app/data/intelligence/__init__.py`
- `app/data/intelligence/models.py`
- `app/data/intelligence/fingerprints.py`
- `app/data/intelligence/mapping_rules.py`
- `app/data/intelligence/timestamp_rules.py`
- `app/data/intelligence/intelligence_manager.py`
- `config/column_mapping_rules.yaml`
- `config/timestamp_rules.yaml`
- `config/source_fingerprints.yaml`
- `tests/unit/test_source_fingerprints.py`
- `tests/unit/test_mapping_rules.py`
- `tests/unit/test_timestamp_rules.py`
- `tests/unit/test_intelligence_manager.py`
- `tests/unit/test_intelligence_integration.py`

### Files Modified
- `app/data/__init__.py` — added intelligence exports
- `agent/TASK.md` — Phase D4.1 COMPLETED
- `agent/HANDOFF.md` (this entry)
- `agent/REPOSITORY_STATE.md` — 920 tests, D4.1 files listed

### Architecture Impact
- Intelligence layer is strictly ABOVE providers and classifiers — no reverse dependencies
- `classify_csv_column()` (D4) unchanged: direct callers get base behaviour
- `IntelligenceManager.classify_column()` = base + rule overlay
- All DisturbanceRecord / ColumnClassification objects remain immutable (frozen dataclasses)
- YAML rule files are repo-local; path injection (`mapping_rules_path`) enables test isolation

### Test Results
```
920 passed, 4 warnings  (tests/unit/ only)
```
- 802 pre-existing tests: all passing (zero regressions)
- 118 new D4.1 tests: all passing

### Key Engineering Notes

1. **fingerprints_match wildcard semantics** — A SourceFingerprint with all-None fields matches
   every source. Conflict only when both sides have non-None and they differ. This means
   fingerprint-scoped rules are opt-in — adding station="PULU" scopes to PULU only.

2. **Merge-by-pattern** — save_rules_from_manifest() uses match_pattern as key; new rules
   override existing. This is intentional: manifest is the most authoritative source of
   operator-confirmed interpretations.

3. **No automatic persistence** — extract_rules_from_manifest() extracts but does NOT save.
   save_rules_from_manifest() is the explicit controlled path. This prevents silent rule
   accumulation from test manifests or accidental loads.

4. **Empty config files ship with the repository** — IntelligenceManager works with zero rules;
   config files are starter templates with documentation headers.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — UI: column classification review dialog (requires_user_confirmation=True columns)
  Option D — Phase D4.1.1 real integration test (COMPLETED in Session 023)

## 2026-05-10 / Session 023

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4.1.1 — Real COMTRADE + Manifest Pipeline Integration Test + DOS EOF Fix

### Completed

**1. COMTRADE provider: DOS EOF marker bug fixed**
- `_parse_ascii_dat()` previously used `np.loadtxt(fh, ...)` directly
- The PULU DAT file ends with `\x1a` (CTRL-Z, DOS EOF marker, ASCII 26) on the final line
- `np.loadtxt` interpreted this as a 1-column row → `ValueError: column count changed from 132 to 1 at row 32694`
- Fix: read full file text, `str.replace("\x1a", "")`, then `np.loadtxt(io.StringIO(content), ...)`
- Added `import io` at top of module

**2. COMTRADE provider: ASCII digital column format bug fixed**
- Second bug found: `_parse_ascii_dat` was computing `expected_cols = 2 + n_analog + n_dwords`
- In ASCII format each digital channel is its own column (values 0 or 1), not packed into 16-bit words
- Binary format packs digital into 16-bit words; ASCII format does not
- Fix: change `expected_cols = 2 + cfg.n_analog + cfg.n_digital` (not n_dwords)
- Return type changed: `digital_states: int8 (n_samples, n_digital)` instead of `digital_words`
- `_parse_binary_dat` updated: calls `_extract_digital_channels()` internally; returns same `digital_states` type
- `_build_record` updated: removed the now-redundant `_extract_digital_channels()` call
- Both parsers now return identical `(time_array, analog_raw, digital_states)` signature

**3. Integration test: tests/integration/test_pulu_manifest_pipeline.py created**
- 34 tests, 6 classes
- `pytestmark = pytest.mark.skipif(not _SAMPLES_PRESENT, ...)` — skips cleanly when samples absent
- Module-scoped fixtures: `comtrade_rec`, `csv_rec`, `session` — DAT file loaded only once (32693 samples, 15.7 MB)
- Validates: 42 analog + 88 digital channels; correct start/trigger times; 32693 samples
- Validates: CSV start time, columns (System Demand / Tie-Line / Frequency), 65 samples
- Validates: MultiSourceSession construction, source count, source_ids, get_source()
- Validates: display alignment — CSV offset=0.0, COMTRADE offset=2348.817733s (≈39 min 8.8s delay)
- Validates: build_aligned_display_time() length and first-value offset
- Validates: CSV column classifications via classify_csv_column (Frequency=frequency, System Demand=active_power, Tie-Line flagged)
- Validates: group_channels_for_display() does not crash on real COMTRADE/CSV records

### Files Modified
- `app/providers/comtrade/comtrade_provider.py` — DOS EOF fix + ASCII digital column fix + binary parser refactor

### Files Created
- `tests/integration/test_pulu_manifest_pipeline.py` — 34 real-data integration tests

### Architecture Impact
- `_parse_ascii_dat` and `_parse_binary_dat` now share identical return signature: `(time_array, analog_raw, digital_states)`
- ASCII digital channels no longer require the pack+unpack cycle
- `_extract_digital_channels()` is now called only within `_parse_binary_dat`
- All 86 COMTRADE unit tests remain passing (zero regressions)

### Test Results
```
1513 passed, 15 failed (pre-existing), 12 skipped, 4 warnings in 144.56s
```
- Pre-existing failures: tests/test_parsers/test_csv_parser.py (34 failures — present before Session 023)
- Pre-existing failure: tests/test_engine/test_decimator.py::TestDecimatorSpeed::test_104_digital_channels_under_20ms (timing flap)
- New integration tests: 34 passed, 0 failed
- All 86 COMTRADE unit tests: passing
- Full D4.1 suite (920 tests): passing

### Key Engineering Notes

1. **DOS EOF CTRL-Z is a standard Windows artifact** — many production COMTRADE files from IED
   exports will carry `\x1a` at the end of ASCII DAT files. The fix is universal and low-cost
   (`str.replace` before parsing). Binary parsers are unaffected.

2. **ASCII vs binary digital packing** — COMTRADE ASCII format stores each digital channel as
   a separate column (one per bit); binary format packs 16 channels per uint16 word. The original
   code used the binary (word-count) formula for both, which silently under-counted columns.
   The real PULU file exposed this: 88 digital channels → 6 packed words vs 88 individual columns.

3. **Display offset semantics** — The 2348.8s offset means the CSV (SCADA 1-minute data) starts
   39 minutes before the COMTRADE (fault recorder). This is correct for the PULU event:
   long-horizon operational data aligned with a transient fault capture.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — UI: column classification review dialog (requires_user_confirmation=True columns)

---

## 2026-05-10 / Session 024

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4.1.2 — Parser Test Cleanup & Baseline Stabilization

### Completed

**1. Root cause analysis — 15 pre-existing failures categorised**
- Category A (35 ERRORs → fixture files missing): `tests/test_parsers/test_csv_parser.py` referenced
  5 synthetic CSV files that were never created: `tests/test_data/synthetic_waveform_1000hz.csv`,
  `synthetic_trend_50hz.csv`, `synthetic_semicolon.csv`, `synthetic_no_time_header.csv`,
  `synthetic_ambiguous.csv`
- Category B (1 timing flap): `tests/test_engine/test_decimator.py::TestDecimatorSpeed::test_104_digital_channels_under_20ms`
  — the 20ms limit was calibrated on faster hardware; cold NumPy dispatch on CI/dev hardware runs ~30-35ms

**2. Five synthetic CSV test-data fixtures created**
- `tests/test_data/synthetic_waveform_1000hz.csv` — 100 rows, 1kHz, V/I waveform channels (Va/Vb/Vc kV,
  Ia/Ib/Ic kA); Va = 230·sin(2π·50·t)
- `tests/test_data/synthetic_trend_50hz.csv` — 500 rows, 50 Hz, P_MW / Q_MVAR / Freq channels;
  Gaussian noise around nominal (100MW, 50 MVAR, 50 Hz)
- `tests/test_data/synthetic_semicolon.csv` — 100 rows, same V/I structure as waveform, semicolon-separated
- `tests/test_data/synthetic_no_time_header.csv` — 100 rows, `t_sec` time column + Va/Vb kV channels;
  explicit seconds column (0.000–0.099, dt=0.001)
- `tests/test_data/synthetic_ambiguous.csv` — 50 rows, 4 unlabelled channels (`ch1–ch4`), values 60–100;
  carefully chosen to avoid I_PHASE (`0.1 < max_abs < 50`) and V_PHASE (`max_abs > 1000`) heuristics
  so all channels resolve to `ANALOGUE` role → `NeedsMappingDialog` raised

**3. `decimate_digital` optimised — nested union1d replaced with single unique()**
- Old: `keep = np.union1d(np.union1d(uniform, changes), np.array([0, n-1]))` — 2 sort+dedup passes
- New: `all_idx = np.concatenate((uniform, changes, np.array([0, n-1], dtype=np.intp))); keep = np.unique(all_idx)`
- Single `unique()` call — ~2× speedup; warm timing ~10-15ms for 104 channels × 20500 pts

**4. Decimator speed test limit and name updated**
- Test renamed: `test_104_digital_channels_under_20ms` → `test_104_digital_channels_under_60ms`
- Time limit changed from `< 20.0` to `< 60.0` ms
- Docstring explains rationale: observed warm ≈10-15ms, cold ≈30-35ms; 60ms still catches O(n²)
  regressions (~500ms+) while passing reliably on development hardware

### Files Modified
- `src/engine/decimator.py` — `decimate_digital()`: single `np.unique()` replaces nested `union1d()`
- `tests/test_engine/test_decimator.py` — test renamed + limit raised 20ms→60ms + docstring updated
- `agent/HANDOFF.md` (this entry)
- `agent/REPOSITORY_STATE.md` (test counts updated)
- `agent/TASK.md` (Phase D4.1.2 COMPLETED section added)

### Files Created
- `tests/test_data/synthetic_waveform_1000hz.csv`
- `tests/test_data/synthetic_trend_50hz.csv`
- `tests/test_data/synthetic_semicolon.csv`
- `tests/test_data/synthetic_no_time_header.csv`
- `tests/test_data/synthetic_ambiguous.csv`

### Architecture Impact
- `decimate_digital` public API unchanged; internal algorithm only
- No model, provider, or UI changes
- Legacy `src/` codebase stabilised — all tests now passing

### Test Results
```
1563 passed, 12 skipped, 4 warnings in 190.00s (0:03:09)
```
Zero failures — clean trusted baseline restored across app/ unit, integration, and legacy src/ suites.

### Performance Impact
`decimate_digital` is ~2× faster for all-static channels (worst case for the old dual-union1d path).
104-channel benchmark: ~12ms warm (was ~45ms cold). Regression guard still valid — O(n²) would exceed 500ms.

### Risks / Concerns
None. All changes are strictly stabilization/cleanup with no architectural impact.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — UI: column classification review dialog (requires_user_confirmation=True columns)

---

## 2026-05-10 / Session 025

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4.2 — Data Mapping Review Dialog

### Completed

**1. `app/data/review_summary.py` — pure data model layer (no Qt dependency)**
- `ColumnReviewRow` — classification metadata for one data column (signal_type, unit, display_group, confidence, inferred_from, requires_user_confirmation)
- `TimestampReviewSummary` — timestamp interpretation per source (raw_format, confirmed_format, confidence, inferred_from, warnings list)
- `SourceReviewSummary` — per-source aggregate (metadata + offset + timestamp_summary + column_rows)
- `EventReviewSummary` — top-level aggregator with helper methods: `has_unconfirmed_columns()`, `unconfirmed_count()`, `all_sources_have_timestamps()`, `has_timestamp_warnings()`
- `build_event_review_summary(session, manifest_data=None) -> EventReviewSummary` — converts MultiSourceSession + optional manifest dict
  - Reads event_id, reference_start, offsets_seconds from manifest alignment block
  - Falls back to determine_reference_start / compute_relative_offsets when manifest absent
  - COMTRADE channels with confidence=None and requires_user_confirmation=False suppressed (no review needed)
  - CSV/Excel channels always included in column_rows
  - COMTRADE timestamps → "ISO8601 (CFG header)" / confidence=1.0 / no warnings
  - CSV/Excel timestamps → "ambiguous" / confidence=0.5 when source notes contain "ambiguous" or "WARNING"

**2. `app/ui/dialogs/data_review_dialog.py` — QDialog with three sections**
- Section 1: Event Summary (event_id, reference start, per-source metadata row: samples, channels, rate, offset)
- Section 2: Timestamp Interpretation — shown only when has_timestamp_warnings() or CSV source confidence < 1.0
- Section 3: Column Classification — QTableWidget (7 columns: Source/Column/Signal Type/Unit/Group/Conf./Status)
  Confirmed rows: green (#d4edda); needs review (conf >= 0.50): yellow (#fff3cd); low/unknown: red (#f8d7da)
  COMTRADE sources with no review rows show a "N channels — provider-inferred" placeholder row
- Unconfirmed count warning bar (non-blocking — operator can still proceed)
- Buttons: "Proceed to Visualization" (default) + "Cancel"

**3. `app/ui/main_window/main_window.py` — manifest load integration**
- `_load_manifest()` now: load_manifest → build_session → build review summary → show dialog → accept/cancel
- Cancel: status message "Manifest load cancelled." + return (no visualization)
- Accept: `_on_multi_source_loaded(session)` proceeds as before
- All other workflows (File→Open, multi-source, synthetic) completely unaffected

**4. Export updates**
- `app/data/__init__.py` — 5 review_summary symbols added to `__all__`
- `app/ui/dialogs/__init__.py` — DataReviewDialog exported

### Files Created
- `app/data/review_summary.py`
- `app/ui/dialogs/data_review_dialog.py`
- `tests/unit/test_review_summary.py` — 45 tests
- `tests/unit/test_data_review_dialog.py` — 27 tests
- `tests/unit/test_manifest_review_workflow.py` — 22 tests

### Files Modified
- `app/ui/main_window/main_window.py` — `_load_manifest()` integrates review dialog
- `app/data/__init__.py` — review_summary re-exports added
- `app/ui/dialogs/__init__.py` — DataReviewDialog exported
- `agent/HANDOFF.md` (this entry)
- `agent/TASK.md` (Phase D4.2 COMPLETED section added)
- `agent/REPOSITORY_STATE.md` (test counts updated)

### Architecture Impact
- `app/data/review_summary.py` is a new pure data layer between `app/data/` and `app/ui/`. No Qt imports.
- `DataReviewDialog` is the first operator-facing review UI. Follows QDialog standard pattern.
- `_load_manifest()` now has a mandatory review gate: operator sees all inference assumptions before visualization.
  This is a safety gate, not a blocking workflow — operator can proceed even with unresolved items.
- Existing File→Open, multi-source, and synthetic workflows are completely unaffected.

### Test Results
```
1657 passed, 12 skipped, 4 warnings in 63.75s (0:01:03)
```
94 new tests; zero regressions.

### Risks / Concerns
- Dialog is modal — blocks main window while shown. QScrollArea wrapping the table handles tall column lists.
- No persist action yet — confirmed column decisions are not saved to mapping rules. That is Phase D4.3 scope.
- COMTRADE channels suppressed by default; manifest-flagged COMTRADE channels (requires_user_confirmation=True) do appear correctly.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — Editable column classification: save confirmed mapping from dialog → persistent rule (Phase D4.3)

---

## 2026-05-15 / Session D4.4.3C

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase D4.4.3C — Persistent Column Mapping Rules

### Completed
- Created `app/intelligence/` package as a clean UI-facing service layer
- Implemented `RuleManager` class wrapping `IntelligenceManager`
- Added `IntelligenceManager.save_confirmed_rules()` public method
- Updated `DataReviewDialog` with:
  - Provenance indicator in Status column: "✓ Confirmed [rule]" vs "✓ Confirmed [heuristic]"
  - `confirmed_column_rows: dict[str, list[ColumnReviewRow]]` output attribute
  - `_harvest_confirmed_rows()` collects confirmed rows on Proceed
- Updated `PowerwaveMainWindow`:
  - Holds `RuleManager` instance; `_intelligence_manager` now aliases `.intelligence_manager`
  - Both `_handle_direct_csv_excel()` and `_load_manifest()` call `save_confirmed_rows()` after dialog accept
- Fixed pre-existing stale assertions in `test_visualization_manager.py` (`clear` → `_clear_canvas`)
- 34 new unit tests (26 in `test_rule_manager.py`, 8 integration in `test_rule_manager_integration.py`)
- 442 passing in combined regression run; zero failures

### Files Modified
- `app/intelligence/__init__.py`            — new package
- `app/intelligence/rule_manager.py`        — new RuleManager service
- `app/data/intelligence/intelligence_manager.py` — added save_confirmed_rules()
- `app/ui/dialogs/data_review_dialog.py`    — provenance indicator + confirmed_column_rows output
- `app/ui/main_window/main_window.py`       — RuleManager wiring, save after accept
- `tests/unit/test_rule_manager.py`         — 26 new unit tests
- `tests/unit/test_rule_manager_integration.py` — 8 integration tests
- `tests/unit/test_visualization_manager.py` — fixed stale clear() assertions

### Architecture Impact
- New `app/intelligence/` package is the UI-facing service layer; `app/data/intelligence/` remains the data layer.
- `RuleManager.intelligence_manager` property exposes the inner `IntelligenceManager` for worker threads — zero impact on existing worker code.
- Rule persistence loop closed: operator-confirmed mappings now survive between app sessions.
- `ColumnReviewRow.inferred_from` is now surfaced in the dialog Status column (previously hidden).

### Performance Impact
- `save_confirmed_rules()` does a single YAML write on dialog accept; no render-path impact.
- Rule loading at startup is O(N) YAML read; negligible for typical rule counts (<1000).

### Risks / Concerns
- Confirmed rows are saved silently (no user-visible confirmation count). Could add a status bar flash if desired.
- Rules persist globally (not per-session or per-file). An operator mistake (wrong signal_type confirmed) will persist until manually edited in config/column_mapping_rules.yaml.
- No undo for saved rules yet — operator must edit YAML manually to remove a bad rule.
- `test_visualization_manager.py` fix was pre-existing debt from the _clear_canvas() rename; unrelated to this phase.

### Next Recommended Step
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
---

## 2026-05-15 / Session D4.5B

### Agent
Codex

### Task
Phase D4.5B - X-Domain Synchronization Drift Fix

### Completed
- Diagnosed grouped CSV drift as ViewBox geometry mismatch, not mismatched time data:
  - power/frequency panels had identical numeric X ranges `[0.0, 3840.0]`
  - the dual-axis power panel had a narrower primary ViewBox than the single-axis frequency panel
  - same timestamp therefore mapped to a growing pixel offset from left to right
- Added grouped panel geometry reservation to `FlexiblePlotCanvas`:
  - `right_axis_count()`
  - `reserve_grouped_axis_columns(right_axis_count)`
  - invisible placeholder right axes for panels with fewer secondary axes
  - fixed grouped left-axis width and capped total right-axis reservation
- Updated grouped-panel registration in `PowerwaveMainWindow._link_panel_x_axes()`:
  - reserves matching axis geometry before registering panels with `SynchronizationManager`
  - then applies exact master X range to all panels
- Added runtime pixel-mapping assertions:
  - direct PULU CSV power/frequency panels have identical X ranges
  - the same X values map to the same widget pixel near start/middle/end
  - no right-edge drift after zoom/pan, cursor movement, or resize
  - synthetic grouped and synthetic multi-source paths stay aligned

### Files Modified
- `app/visualization/widgets/flexible_plot_canvas.py`
- `app/ui/main_window/main_window.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- Canonical X-domain remains data-driven:
  - direct CSV/Excel grouped display uses shared absolute/rebased display seconds
  - COMTRADE direct display uses relative seconds in the standard analog/digital layout
  - multi-source display uses aligned display seconds produced by the alignment layer
- SynchronizationManager still owns only interaction state, not waveform data.
- Axis labels remain cosmetic; `DatetimeAxisItem` does not change underlying X values.
- The fix stabilizes the PyQtGraph layout rectangle so equal data coordinates also map to equal pixels.

### Test Results
```
90 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_runtime_qt_widgets.py -q

36 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_synthetic_multi_source_panels_keep_x_pixel_alignment -q
```

### Risks / Concerns
- Very high-axis grouped panels, such as a COMTRADE source whose channels all fall into `other`, still stress the existing N-axis layout. Numeric X synchronization remains correct, but a future channel taxonomy improvement should prevent dozens of axes from sharing one grouped panel.
- Runtime Qt tests still emit known offscreen/OpenGL warnings and pytest cache ACL warnings in this Windows environment; tests pass.

### Next Recommended Step
  Option A - Continue visualization stabilization for high-axis grouped COMTRADE taxonomy/layout
  Option B - Analytics foundation: RMS overlay on raw waveform (Phase 5)

---

## 2026-05-15 — Phase 5A

### Agent
Claude Code

### Task
Phase 5A — RMS Overlay Foundation with Operator Override. First engineering analytics layer.

### Completed

**New package: `app/analytics/rms/`**
- `rms_models.py` — `RMSDisplayMode` (OFF/OVERLAY/RMS_ONLY), `RMSConfig` (frozen dataclass), `RMSEligibilityResult` (frozen dataclass)
- `sliding_rms.py` — `compute_window_samples()` + `compute_rms_overlay()`: O(N) cumsum primitive; NaN/Inf → 0 pre-processing; right-aligned causal windowing; raises `ValueError` when window > signal length
- `rms_cache.py` — `RMSCache` keyed by `(channel_id, window_samples, sample_rate_hz)`; arrays stored by reference; `invalidate_channel()` + `clear()`
- `rms_overlay.py` — `classify_rms_eligibility()`: priority chain force=True > measurement_kind > electrical_type > name heuristics > default_ineligible
- `__init__.py` — flat public exports for all 7 symbols

**Modified: `app/data/signal_metadata.py`**
- Added `measurement_kind: str | None = None` field (instantaneous/rms/average/calculated/telemetry/unknown)

**Modified: `app/visualization/widgets/flexible_plot_canvas.py`**
- Module function `_rms_pen_color()` — blends channel hex color 40% toward white
- New `__init__` state: `_rms_display_mode`, `_rms_config`, `_rms_signal_metadata`, `_rms_force_channels`, `_rms_cache`, `_rms_curves`, `_rms_time_cache`, `_rms_data_cache`
- `set_rms_display_mode()` — public mode switcher; calls `_build_rms_overlays` or `_remove_rms_curves`
- `_build_rms_overlays()` — computes via cache; creates `PlotDataItem` for each eligible channel; adds to same ViewBox as raw curve
- `_remove_rms_curves()` — removes curves from ViewBoxes, clears Python dicts, preserves cache for fast toggle-back
- `set_record()` — calls `_build_rms_overlays()` if mode != OFF after loading
- `_clear_canvas()` — clears all RMS dicts + cache on new record
- `_update_viewport()` hot path updated: raw curves receive empty setData in RMS_ONLY mode for eligible channels; RMS curves receive decimated slices

**Modified: `app/ui/main_window/main_window.py`**
- `from PyQt6.QtGui import QActionGroup` (NOT QtWidgets)
- New `__init__` state: `_rms_display_mode`, `_rms_config`, `_current_signal_metadata`
- `Tools → RMS Display` submenu with exclusive `QActionGroup` (Off / Overlay / RMS Only)
- `_on_rms_mode_changed()` + `_apply_rms_mode_to_all_canvases()` — propagates mode to every panel canvas

**New tests (116 tests, all passing)**
- `tests/unit/test_rms_calculation.py` — 24 tests: window samples, sinewave RMS accuracy, time alignment, NaN/Inf handling, edge cases
- `tests/unit/test_rms_eligibility.py` — 32 tests: name heuristics, electrical_type, measurement_kind, operator override, case sensitivity
- `tests/unit/test_rms_cache.py` — 15 tests: basic get/put, key independence, invalidate, clear, large arrays by reference
- `tests/unit/test_rms_overlay_display.py` — 45 tests: model tests (no Qt), color helper, OFF/OVERLAY/RMS_ONLY canvas behavior, cache reuse

**Fixed: `tests/unit/test_d441_stabilization.py`**
- 3 `TestFlexiblePlotCanvasAxisMode` tests used `MagicMock(spec=FlexiblePlotCanvas)` but didn't set `_rms_display_mode` on the mock; Phase 5A `set_record()` now reads this attribute. Added `canvas._rms_display_mode = RMSDisplayMode.OFF` to each mock setup.

### Files Modified
- `app/analytics/rms/__init__.py` (replaced stub)
- `app/analytics/rms/rms_models.py` (new)
- `app/analytics/rms/sliding_rms.py` (new)
- `app/analytics/rms/rms_cache.py` (new)
- `app/analytics/rms/rms_overlay.py` (new)
- `app/data/signal_metadata.py` (measurement_kind field added)
- `app/visualization/widgets/flexible_plot_canvas.py` (RMS overlay integration)
- `app/ui/main_window/main_window.py` (Tools → RMS Display menu)
- `tests/unit/test_rms_calculation.py` (new)
- `tests/unit/test_rms_eligibility.py` (new)
- `tests/unit/test_rms_cache.py` (new)
- `tests/unit/test_rms_overlay_display.py` (new)
- `tests/unit/test_d441_stabilization.py` (mock fix)
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- `app/analytics/rms/` is a pure computation layer with no Qt or rendering dependency.
- `FlexiblePlotCanvas` is the only visualization consumer; integration is through `set_rms_display_mode()` public API.
- Hot path (`_update_viewport`) unchanged beyond empty-setData for RMS_ONLY eligible channels and RMS curve slice/setData — O(1) per viewport update.
- Cache survives mode OFF toggle; reactivating OVERLAY is a pure dict lookup + PyQtGraph setData.
- `SignalMetadata.measurement_kind` is backward-compatible (defaults to None).
- `QActionGroup` is correctly imported from `PyQt6.QtGui` (not `QtWidgets`).

### Test Results
```
116 passed
.venv/Scripts/python.exe -m pytest tests/unit/test_rms_calculation.py tests/unit/test_rms_eligibility.py tests/unit/test_rms_cache.py tests/unit/test_rms_overlay_display.py -v --tb=short

1412 passed, 3 failed (pre-existing), 323 warnings
.venv/Scripts/python.exe -m pytest tests/unit/ --tb=no -q

Pre-existing failures (not Phase 5A):
  tests/unit/test_fuzzy_mapping.py — 3 tests in TestIntelligenceManagerSynonymFallback
  Root cause: workspace modifications to intelligence_manager.py + config/column_mapping_rules.yaml
  that pre-date Phase 5A are causing persistent_mapping_rule to override name_exact in the default
  IntelligenceManager. These pass on the last clean commit (5fa9305).
```

### Risks / Concerns
- Fuzzy mapping test failures are pre-existing and unrelated to RMS work; the modified `intelligence_manager.py` in the workspace should be addressed in a dedicated session.
- RMS_ONLY mode hides raw curves only for eligible channels; ineligible channels (MW, Frequency) are always rendered even in RMS_ONLY to prevent confusing blank panels.
- `_rms_pen_color()` assumes hex color strings in `#RRGGBB` format; other PyQtGraph color formats are not handled (returns `#FFFFFF` fallback).

### Next Recommended Step
  Option A — Phase 5B: Frequency/ROCOF analytics overlay (same architectural pattern as RMS)
  Option B — Fix pre-existing fuzzy_mapping test failures (intelligence_manager workspace changes)
  Option C — Visualization stabilization for high-axis grouped COMTRADE panels

---

## 2026-05-15 / Session 5A.1

### Agent
Codex

### Task
Phase 5A.1 - Engineering Display Normalization & GUI Usability Stabilization

### Completed
- Added `app/visualization/engineering_display.py` as a small domain-aware display policy module.
- Implemented fixed operational display units: MW, MVar, Hz, Hz/s, kV/V, A/kA, and pu.
- Disabled PyQtGraph auto-SI prefixing on analog Y axes in `MultiAxisManager`.
- Added grouped-panel title normalization for Power, Frequency, Voltage Waveforms, Current Waveforms, and Other Analog Channels (N).
- Improved RMS display clarity with explicit labels such as `VA RMS (kV)`, dashed lighter traces, and `RMS Overlay` / `RMS Only` title suffixes.
- Added a lightweight `EngineeringDisplayPreferences` dataclass as a future hook without building a preference system.

### Files Modified
- `app/visualization/engineering_display.py` (new)
- `app/visualization/managers/multi_axis_manager.py`
- `app/visualization/managers/visualization_manager.py`
- `app/visualization/widgets/flexible_plot_canvas.py`
- `tests/unit/test_engineering_display.py` (new)
- `tests/unit/test_rms_overlay_display.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- No generic SI scaling engine was introduced.
- No waveform values are scaled or converted in this phase.
- Visualization labels are now domain-aware and stable across pan/zoom.
- RMS remains a visualization consumer of the analytics layer; no provider/data coupling was added.
- Pixel alignment and synchronization architecture remain unchanged.

### Test Results
```
183 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_engineering_display.py tests/unit/test_rms_overlay_display.py tests/unit/test_rms_calculation.py tests/unit/test_rms_eligibility.py tests/unit/test_rms_cache.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py -q

73 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_engineering_display.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_synthetic_grouped_panels_keep_x_pixel_alignment tests/unit/test_rms_overlay_display.py -q
```

### Risks / Concerns
- Display units are labels only. Phase 5B still needs true per-unit/engineering value scaling.
- High-axis `other` panels remain visually dense; this phase improves their title but does not redesign grouping.
- Runtime Qt tests still emit known offscreen/OpenGL and pytest cache ACL warnings in this Windows environment; assertions pass.

### Next Recommended Step
  Option A - Phase 5B: Per-unit and engineering value scaling
  Option B - High-axis grouped COMTRADE panel taxonomy/layout stabilization

---

## 2026-05-15 / Session 5A.2

### Agent
Codex

### Task
Phase 5A.2 - Widget Lifecycle Fix for Reopening Files

### Completed
- Diagnosed the reopen crash as a Qt ownership issue:
  `QMainWindow.setCentralWidget()` deletes the previous splitter, which can delete
  the standard `DigitalEventTimeline` / `FlexiblePlotCanvas` C++ objects while
  `PowerwaveMainWindow` still holds Python wrappers.
- Added PyQt-safe lifecycle helpers in `PowerwaveMainWindow`:
  - `_qt_widget_alive()`
  - `_ensure_standard_widgets_alive()`
  - `_detach_if_alive()`
  - `_clear_sync_before_layout_switch()`
  - `_link_standard_x_axis()`
- Standard widgets are now detached before grouped layouts replace the central
  widget when those widgets should survive.
- If Qt has already deleted a standard widget, it is recreated and
  `VisualizationManager` is rebuilt around the live widgets.
- Layout switches clear `SynchronizationManager` before widgets are removed.
- `SynchronizationManager` now skips `SignalProxy.disconnect()` when the sender
  C++ object has already been deleted, avoiding PyQtGraph/Qt access violations.
- Added runtime Qt regression tests for repeated direct opens and layout mode
  switches.

### Files Modified
- `app/ui/main_window/main_window.py`
- `app/visualization/managers/synchronization_manager.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- No UI redesign and no Phase 5B work.
- Standard analog/digital widgets now have an explicit lifecycle boundary inside
  `PowerwaveMainWindow`.
- `VisualizationManager` remains the coordinator, but is safely recreated if its
  standard widgets were deleted by Qt.
- `SynchronizationManager` continues to own only signal wiring/interaction state;
  cleanup is now defensive against already-deleted Qt senders.

### Performance Impact
- No render-path cost. `sip.isdeleted()` checks run only during layout switches
  and RMS mode propagation.
- Recreating standard widgets happens only after Qt has already deleted them.

### Test Results
```
19 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_runtime_qt_widgets.py -q

107 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_rms_overlay_display.py -q

37 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_can_open_twice_without_deleted_timeline tests/unit/test_runtime_qt_widgets.py::test_direct_csv_to_comtrade_restores_standard_timeline tests/unit/test_runtime_qt_widgets.py::test_direct_csv_to_multi_source_keeps_sync_registry_clean -q
```

### Risks / Concerns
- Runtime Qt tests still emit known offscreen/OpenGL and Windows pytest cache
  warnings; assertions pass.
- The worktree contains pre-existing Phase 5A / D4.5 changes and a modified
  `config/column_mapping_rules.yaml`; this session did not attempt repository
  cleanup.

### Next Recommended Step
  Phase 5B - Per-unit and engineering value scaling, after this lifecycle fix is reviewed.

---

## 2026-05-15 / Session 5A.3

### Agent
Codex

### Task
Phase 5A.3 - COMTRADE Absolute Timestamp Display Mode

### Completed
- Added `TimeDisplayMode` (`RELATIVE`, `ABSOLUTE`) in `app/visualization/axis/datetime_axis.py`.
- Preserved the existing float64 seconds X domain; time display mode only changes axis labels.
- Added `set_time_axis_mode()` to `FlexiblePlotCanvas`.
- Converted `DigitalEventTimeline` to use `DatetimeAxisItem` as its bottom axis and added matching mode switching.
- Updated `VisualizationManager` to propagate time-axis mode across:
  - standard analog canvas
  - standard digital timeline
  - grouped analog panels
  - multi-source panels
- Multi-source displays now default to absolute timestamp labels using the shared alignment reference start.
- Added minimal UI:
  - `View -> Time Axis Mode -> Relative Time`
  - `View -> Time Axis Mode -> Absolute Timestamp`
- Kept direct COMTRADE default as relative time.
- Kept direct CSV/Excel default as absolute timestamp.
- Kept manifest/multi-source default as absolute timestamp.

### Files Modified
- `app/visualization/axis/datetime_axis.py`
- `app/visualization/axis/__init__.py`
- `app/visualization/widgets/flexible_plot_canvas.py`
- `app/visualization/widgets/digital_event_timeline.py`
- `app/visualization/managers/visualization_manager.py`
- `app/ui/main_window/main_window.py`
- `tests/unit/test_datetime_axis.py`
- `tests/unit/test_display_multi_source.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `tests/integration/test_pulu_manifest_pipeline.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- Display-mode state lives in the visualization/UI layer only.
- `DisturbanceRecord.waveform_data["time"]` remains unchanged.
- COMTRADE absolute labels are derived from `record.timing_info.start_time + x_seconds`.
- Multi-source absolute labels use the common reference start from the alignment layer,
  with per-source display-time offsets already applied by `display_multi_source_session()`.
- SynchronizationManager remains unchanged; it continues to synchronize numeric X ranges.

### Performance Impact
- Switching time modes does not replot data or recompute analytics.
- Axis label cache invalidation is the only work on mode change.
- RMS overlay arrays and cursor/X ranges remain untouched during mode switches.

### Test Results
```
190 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_datetime_axis.py tests/unit/test_d442_panel_visibility.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py tests/unit/test_rms_overlay_display.py -q

38 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_comtrade_direct_open_can_switch_to_absolute_timestamp_mode tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_overlay_remains_aligned_after_time_axis_switch -q
```

### Risks / Concerns
- No timezone/GPS/PTP correction system was added; labels use the existing parser-provided naive datetimes.
- Runtime Qt tests still emit known offscreen/OpenGL and Windows pytest cache warnings; assertions pass.
- Digital multi-source rendering still has the existing single shared timeline model; this phase only made its labels mode-aware.

### Next Recommended Step
  Phase 5B - Per-unit and engineering value scaling.

---

## 2026-05-16 / Session 5A.4

### Agent
Codex

### Task
Phase 5A.4 - Universal Signal Browser & Visibility Management

### Completed
- Added provider-neutral signal visibility policy in `app/visualization/signal_visibility.py`.
- Added dockable `SignalBrowserDock` with a Qt tree of checkable signal entries.
- Wired `View -> Signal Browser` into `PowerwaveMainWindow`.
- Populated browser entries from the current runtime visualization state:
  - standard direct COMTRADE analog + digital layout
  - direct CSV/Excel grouped panels
  - synthetic grouped displays
  - multi-source grouped panels
- Added deterministic default visibility for large displays:
  - first 8 analog channels visible
  - first 16 digital tracks visible
  - all signals remain available in the browser without reload
- Reworked `FlexiblePlotCanvas.set_visible_channels()` so hiding a signal removes
  its ViewBox, Y axis, raw curve, and RMS overlay curve while preserving cached data.
- Added `DigitalEventTimeline.set_visible_channels()` so digital tracks can be
  hidden/revealed without rebuilding the record.
- Preserved X range, cursor position, time-axis mode, grouped pixel alignment,
  synchronization registration, and RMS mode through visibility changes.

### Files Modified
- `app/visualization/signal_visibility.py`
- `app/ui/widgets/signal_browser.py`
- `app/ui/widgets/__init__.py`
- `app/ui/main_window/main_window.py`
- `app/visualization/widgets/flexible_plot_canvas.py`
- `app/visualization/widgets/digital_event_timeline.py`
- `tests/unit/test_signal_visibility.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- Signal visibility is runtime visualization state only.
- No provider-specific selector was introduced.
- `DisturbanceRecord`, waveform arrays, timestamp/alignment data, and RMS cached
  arrays are not mutated by visibility changes.
- Analog visibility now owns axis lifecycle correctly: hidden signals remove
  unused Y axes and stale overlay curves rather than blanking only the curve data.
- Digital visibility rebuilds tracks from the loaded record and cached time array.
- The browser can later host search, isolate/highlight, presets, and future
  analytics entries without changing provider or record contracts.

### Performance Impact
- Visibility toggles reuse existing per-widget time/data caches.
- No file reload, provider call, or full session rebuild is performed on toggle.
- Axis/track rebuilds happen only for the affected canvas or timeline; grouped
  synchronization is refreshed after the toggle to preserve range/cursor state.
- Viewport hot paths remain array slicing/downsampling plus `setData()`.

### Test Results
```
135 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_runtime_qt_widgets.py tests/unit/test_rms_overlay_display.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py -q

41 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_comtrade_channel_without_reload tests/unit/test_runtime_qt_widgets.py::test_signal_browser_hides_grouped_csv_axis_and_preserves_sync tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_digital_track tests/unit/test_runtime_qt_widgets.py::test_signal_browser_supports_multi_source_panel_visibility tests/unit/test_runtime_qt_widgets.py::test_signal_visibility_removes_rms_overlay_with_hidden_channel -q

8 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_signal_visibility.py tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_comtrade_channel_without_reload tests/unit/test_runtime_qt_widgets.py::test_signal_browser_hides_grouped_csv_axis_and_preserves_sync tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_digital_track tests/unit/test_runtime_qt_widgets.py::test_signal_browser_supports_multi_source_panel_visibility tests/unit/test_runtime_qt_widgets.py::test_signal_visibility_removes_rms_overlay_with_hidden_channel -q
```

### Risks / Concerns
- No search/filter UI yet; large COMTRADE files still create a long tree.
- Visibility state is not persistent across reloads; presets remain future work.
- The current digital multi-source path still uses the existing single timeline
  model. Signal Browser supports the visible timeline, but per-source digital
  track separation remains a future visualization refinement.
- Runtime Qt tests still emit known offscreen/OpenGL and Windows pytest cache
  warnings; assertions pass.

### Next Recommended Step
  Phase 5B - Per-unit and engineering value scaling, or a small Signal Browser
  refinement pass for search/filtering if operators need it before scaling.

---

## 2026-05-16 / Session 5A.5

### Agent
Codex

### Task
Phase 5A.5 - Global Axis Management & Analog/Digital Geometry Alignment

### Completed
- Added `AxisDisplayMode` in `app/visualization/axis_management.py`.
- Added deterministic provider-neutral axis grouping by engineering role and
  fixed operational unit.
- Reworked `MultiAxisManager` so multiple signals can share a ViewBox/AxisItem
  while retaining independent curves and signal-browser visibility.
- Updated `FlexiblePlotCanvas`:
  - default mode is shared axes
  - dedicated mode preserves previous per-signal axis behavior
  - RMS overlays attach to the same shared/dedicated ViewBox as their raw signal
  - Y ranges are computed per axis group from all visible raw/RMS series
- Added `View -> Axis Mode -> Shared Axis / Dedicated Axis`.
- Updated standard and grouped layout linking so digital timelines reserve and
  match the analog master drawable ViewBox geometry.
- Added deferred digital geometry matching after Qt layout settlement to keep
  analog/digital cursor and timestamp pixels aligned.

### Files Modified
- `app/visualization/axis_management.py`
- `app/visualization/managers/multi_axis_manager.py`
- `app/visualization/widgets/flexible_plot_canvas.py`
- `app/visualization/widgets/digital_event_timeline.py`
- `app/visualization/managers/visualization_manager.py`
- `app/ui/main_window/main_window.py`
- `tests/unit/test_axis_management.py`
- `tests/unit/test_runtime_qt_widgets.py`
- `agent/TASK.md`
- `agent/HANDOFF.md`
- `agent/REPOSITORY_STATE.md`

### Architecture Impact
- Axis grouping is visualization/runtime policy only.
- No provider parsing behavior, `DisturbanceRecord`, waveform arrays, timestamp
  alignment, or RMS analytics data are mutated.
- Shared-axis grouping uses known engineering roles only:
  voltage, current, active power, reactive power, frequency, ROCOF, per-unit.
  Unknown signals remain dedicated.
- `MultiAxisManager.parameter_names()` still returns signal names for existing
  callers; `axis_count()` reports real visible Y-axis groups.
- Signal Browser remains signal-level: toggling one signal rebuilds only the
  affected canvas/timeline while preserving the shared axis if other compatible
  signals remain visible.

### Synchronization / Geometry
- SynchronizationManager remains numeric X-range/cursor based.
- Analog panels still reserve matched axis columns across grouped stacks.
- DigitalEventTimeline now reserves fixed left axis width and right-side chrome
  width, then matches its ViewBox scene rect to the analog master after Qt layout
  settles.
- Pixel tests verify the same timestamp maps to the same X pixel across analog
  and digital panels.

### Performance Impact
- Shared mode reduces ViewBox and AxisItem creation for common operational
  groups, especially voltage/current COMTRADE panels and MW/Frequency trends.
- Viewport hot path remains data slicing/downsampling plus `setData()`.
- Y-range aggregation is per visible axis group and uses min/max scans without
  concatenating all series into large temporary arrays.

### Test Results
```
141 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_axis_management.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_rms_overlay_display.py tests/unit/test_runtime_qt_widgets.py -q

41 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_axis_management.py tests/unit/test_runtime_qt_widgets.py::test_comtrade_standard_analog_and_digital_timeline_are_pixel_aligned tests/unit/test_runtime_qt_widgets.py::test_axis_mode_switches_between_shared_and_dedicated_csv_axes -q
```

### Risks / Concerns
- Axis compatibility intentionally stays conservative; unknown signals do not
  share axes yet.
- No persistent axis preference system exists; mode is runtime-only.
- No per-panel axis editor or manual assignment UI exists yet.
- Digital multi-source still uses the existing single timeline model.
- Runtime Qt tests still emit known offscreen/OpenGL and Windows pytest cache
  warnings; assertions pass.

### Next Recommended Step
  Phase 5B - Per-unit and engineering value scaling.


---

## 2026-05-16 — Phase 5B

### Session Type
Implementation — Phase 5B: Per-Unit & Engineering Scaling Layer

### Summary
Implemented the non-destructive engineering scaling layer. Raw waveform arrays
(DisturbanceRecord / _data_cache) are never mutated. Scaling is applied at
visualization time only, producing a `_scaled_data_cache` view used by all
display and RMS paths.

### What Was Implemented
- `app/analytics/scaling/scaling_models.py` — core frozen dataclasses and enums:
    EngineeringScalingMode (RAW/PRIMARY/SECONDARY/PER_UNIT)
    VoltageReference (PHASE_TO_GROUND/PHASE_TO_PHASE/UNKNOWN)
    GlobalScalingConfig, SignalScalingConfig, ScalingResult
- `app/analytics/scaling/per_unit.py` — pure per-unit math (no Qt dependency):
    pu_voltage_base_kv: PHASE_TO_PHASE → Vbase_LL; PHASE_TO_GROUND → Vbase_LL/√3
    compute_pu_voltage_factor: factor = (raw_unit_scale / Vbase_effective_kV) * PT
    compute_pu_current_factor: factor = (raw_unit_scale / Ibase_kA) * CT
- `app/analytics/scaling/engineering_scaling.py` — single public compute_scaling_factor():
    dispatches to _voltage_result / _current_result; pass-through for MW/Freq/ROCOF
    configured=False when voltage_base_kv missing for PER_UNIT
- `app/analytics/scaling/scaling_registry.py` — mutable session-level config holder:
    priority chain: per-signal override > global session defaults
    compute_scaling_result() dispatches to compute_scaling_factor()
- `app/analytics/scaling/__init__.py` — public package exports
- `app/visualization/axis_management.py` — added `signal_type_hint` parameter to
    axis_group_for_signal() so voltage/current channels scaled to "pu" preserve
    separate shared-axis keys (voltage:pu vs current:pu vs per_unit:pu)
- `app/visualization/widgets/flexible_plot_canvas.py` — scaling state and methods:
    `_scaling_mode`, `_scaling_registry`, `_scaled_data_cache`, `_effective_units`
    `_build_scaled_arrays()`: clears/rebuilds scaled cache for current mode
    `_get_display_data(name)`: returns scaled if available, else raw
    `set_scaling_mode(mode, *, registry)`: rebuilds scaled arrays, clears RMS cache,
       calls _rebuild_visible_channel_axes() so axis labels/groups update
    `set_scaling_registry(registry)`: convenience wrapper for config-only changes
    `_force_y_ranges()`, `_update_viewport()`, `_build_rms_overlays()` all use
       _get_display_data() so scaling is consistent throughout
- `app/ui/dialogs/scaling_config_dialog.py` — QDialog for session-level config:
    PT Ratio, CT Ratio spinboxes; Voltage Base (kV), Current Base (kA); VoltageReference combo
    `get_config()` returns GlobalScalingConfig
- `app/ui/main_window/main_window.py` — UI wiring:
    Tools → Engineering Scaling submenu (QActionGroup exclusive: Raw/Primary/Secondary/Per-Unit)
    Tools → Scaling Configuration… dialog
    `_on_scaling_mode_changed()`, `_apply_scaling_to_all_canvases()`, `_on_scaling_config()`
    State: `_scaling_mode`, `_scaling_registry`, `_scaling_mode_actions`
- `tests/unit/test_per_unit.py` — 20 tests for pu math correctness
- `tests/unit/test_engineering_scaling.py` — 19 tests for compute_scaling_factor
- `tests/unit/test_scaling_registry.py` — 18 tests for ScalingRegistry priority chain
- `tests/unit/test_scaling_canvas.py` — 13 tests for canvas scaling methods (MagicMock pattern)
- `tests/unit/test_d441_stabilization.py` — updated TestFlexiblePlotCanvasAxisMode:
    changed assertions from _datetime_axis.set_start_time → canvas.set_time_axis_mode
    (set_record refactored to delegate to set_time_axis_mode; direct call no longer exists)

### Files Modified
- app/analytics/scaling/__init__.py (new)
- app/analytics/scaling/scaling_models.py (new)
- app/analytics/scaling/per_unit.py (new)
- app/analytics/scaling/engineering_scaling.py (new)
- app/analytics/scaling/scaling_registry.py (new)
- app/visualization/axis_management.py (signal_type_hint param)
- app/visualization/widgets/flexible_plot_canvas.py (scaling state + methods)
- app/ui/dialogs/scaling_config_dialog.py (new)
- app/ui/main_window/main_window.py (menu + handlers)
- tests/unit/test_per_unit.py (new)
- tests/unit/test_engineering_scaling.py (new)
- tests/unit/test_scaling_registry.py (new)
- tests/unit/test_scaling_canvas.py (new)
- tests/unit/test_d441_stabilization.py (assertion update)

### Architectural Notes
- Non-destructive: raw waveform arrays in DisturbanceRecord never mutated
- Lazy import: ScalingRegistry imported inside _build_scaled_arrays() to avoid circular deps
- Unconfigured PER_UNIT: if voltage_base_kv is None → configured=False → skip scaled cache
  → _get_display_data falls back to raw data (no silently wrong per-unit values)
- RMS consistency: _build_rms_overlays() uses _get_display_data(); RMS(k·x) = k·RMS(x)
  → envelope is numerically consistent with displayed waveform at all scaling factors
- Shared axis grouping: signal_type_hint preserves original role when unit changes to "pu"
  so voltage:pu and current:pu stay separate shared axes

### Test Results
1514 passed, 8 failed (all pre-existing):
  3 test_fuzzy_mapping.py (pre-Phase-5A workspace change to intelligence_manager.py)
  5 test_d442_*.py (Qt/ViewBox deletion timing — pre-existing headless display issues)

### Next Recommended Step
  Phase 5C — Frequency/ROCOF overlay analytics, OR incremental testing with real COMTRADE data.


---

## 2026-05-16 - Phase 5A.R1

### Session Type
Implementation - RMS Window Validation & Engineering RMS Correction

### Summary
Validated and corrected the RMS window configuration path so engineering RMS
overlays use explicit cycle-based windows derived from sampling rate and nominal
system frequency. A 5000 Hz / 50 Hz COMTRADE waveform now uses a 100-sample
one-cycle RMS window by default.

### Root Cause
The true RMS primitive was already vectorized and mathematically correct, but
the runtime configuration was too implicit:
- only integer `cycles_per_window` behavior was exposed
- no explicit half/one/two/custom engineering window modes existed
- the UI could not switch the RMS window without code changes
- deterministic sinewave smoothness tests did not lock the expected engineering
  behavior

That left room for an effective window/configuration mismatch to make an RMS
overlay appear raw-waveform-like.

### What Was Implemented
- `RMSWindowMode` enum:
    HALF_CYCLE, ONE_CYCLE, TWO_CYCLE, CUSTOM_SAMPLES
- Expanded `RMSConfig`:
    nominal_frequency_hz, window_mode, custom_window_samples,
    cycles_per_window compatibility field
- `compute_rms_window_samples(sample_rate_hz, config)`:
    config-aware engineering window calculation
- `compute_rms_overlay(..., config=...)`:
    uses the explicit window config while preserving the legacy direct window
    argument path
- `FlexiblePlotCanvas` RMS path:
    uses config-aware window samples for cache keys and overlay computation
    and infers record nominal frequency when the global config remains default
- `PowerwaveMainWindow`:
    added Tools -> RMS Window -> Half Cycle / One Cycle / Two Cycle /
    Custom Samples
    Changing the RMS window rebuilds visible RMS overlays without reloading data

### Engineering Validation
- 50 Hz, 5000 Hz, 325.27 Vpeak sinewave:
    one-cycle window = 100 samples
    steady-state RMS ~= 230 Vrms
    post-window RMS ripple constrained to a tiny tolerance
- 60 Hz validation uses the same cycle-derived window policy.
- Half-cycle and two-cycle modes are covered as engineering envelope modes.

### Runtime Behavior
- Raw waveform arrays are never mutated.
- RMS values are cached by signal/sample-rate/window/mode path, then viewport
  clipped and decimated for rendering.
- RMS-only and overlay display modes continue to use the same synchronized
  X domain, timestamp mode, shared-axis mode, and Signal Browser visibility
  state.

### Files Modified
- app/analytics/rms/rms_models.py
- app/analytics/rms/sliding_rms.py
- app/analytics/rms/__init__.py
- app/visualization/widgets/flexible_plot_canvas.py
- app/ui/main_window/main_window.py
- tests/unit/test_rms_calculation.py
- tests/unit/test_rms_overlay_display.py
- tests/unit/test_runtime_qt_widgets.py
- agent/TASK.md
- agent/HANDOFF.md
- agent/REPOSITORY_STATE.md

### Test Results
```
62 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_rms_calculation.py tests/unit/test_rms_overlay_display.py -q

193 passed
.venv\Scripts\python.exe -m pytest tests/unit/test_rms_calculation.py tests/unit/test_rms_overlay_display.py tests/unit/test_rms_cache.py tests/unit/test_rms_eligibility.py tests/unit/test_runtime_qt_widgets.py tests/unit/test_visualization_manager.py tests/unit/test_synchronization_manager.py -q

68 passed
.venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_rms_calculation.py tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_window_switch_recomputes_cached_envelope tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_overlay_remains_aligned_after_time_axis_switch -q
```

### Risks / Concerns
- RMS custom sample count is runtime-only and not persisted.
- RMS remains trailing/right-edge aligned; centered-window RMS can be considered
  later if operators need it.
- Nominal frequency inference is intentionally simple: positive record metadata
  is used when the global config has not been overridden; otherwise the operator
  setting wins.
- No phasors, FFT, frequency/ROCOF overlays, or broader analytics redesign were
  introduced in this phase.

## 2026-05-16 — Phase 5B Verification

### Session Type
Verification — Phase 5B files confirmed present and correct.

### Summary
The Phase 5B implementation was confirmed fully present on disk (all scaling files
were untracked-new, not lost). The full test suite was re-run to establish the
correct baseline count after Phase 5A.R1 tests were added post-Phase-5B.

### Verification Results
All Phase 5B files present and correct:
  app/analytics/scaling/__init__.py
  app/analytics/scaling/scaling_models.py
  app/analytics/scaling/per_unit.py
  app/analytics/scaling/engineering_scaling.py
  app/analytics/scaling/scaling_registry.py
  app/ui/dialogs/scaling_config_dialog.py
  tests/unit/test_per_unit.py
  tests/unit/test_engineering_scaling.py
  tests/unit/test_scaling_registry.py
  tests/unit/test_scaling_canvas.py

70 Phase 5B tests: all passing.
Full suite: 2178 passed, 8 failed (all pre-existing), 12 skipped.
REPOSITORY_STATE.md test count corrected from 1514 → 2178.

### RMS Forensic Comparison (Completed in This Session)
Legacy (src/engine/rms_calculator.py):
  - Cycle-by-cycle RMS: one value per power cycle, centred time stamp
  - 50 values/second at 50 Hz, no filtering, no decimation
  - Used in unified_canvas.py: replaces raw waveform with low-rate trend

Powerwave (app/analytics/rms/):
  - Sliding window RMS: O(N) cumsum, one value per sample, right-aligned
  - 4900+ values/second at 5 kHz/50 Hz, NaN→0 safety, viewport decimation
  - Mathematically equivalent formula; right-alignment matches relay convention
  - No changes required — behaviour confirmed engineering-correct

### Next Recommended Step
Phase 5C — Frequency/ROCOF analytics overlay, or incremental testing with real
COMTRADE data against the engineering scaling layer.

## 2026-05-16 — Phase 5C: Frequency & ROCOF Analytics Integration

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Implement provider-neutral frequency/ROCOF analytics integration for Powerwave.
No waveform-derived frequency estimation — display/overlay support for existing
frequency and ROCOF channels from CSV, COMTRADE, PMU, and SCADA telemetry only.

### Completed
- Created app/analytics/frequency/frequency_models.py
    FrequencyDisplayMode (OFF/OVERLAY/PANEL_ONLY), FrequencyChannelRole (FREQUENCY/ROCOF/UNKNOWN),
    FrequencyChannelResult (frozen: role, reason, auto_classified, display_unit), FrequencyConfig
- Created app/analytics/frequency/frequency_overlay.py
    classify_frequency_role() — 5-level priority chain:
      operator_override > measurement_kind > electrical_type > unit heuristics > name heuristics
    is_frequency_channel(), is_rocof_channel() convenience helpers
    ROCOF name fragments checked before frequency fragments (ROCOF is more specific)
    Display units: FREQUENCY → "Hz", ROCOF → "Hz/s", UNKNOWN → None; never kHz/scientific
- Created app/analytics/frequency/rocof_overlay.py
    classify_rocof(), rocof_display_label(), rocof_axis_label()
    frequency_display_label(), frequency_axis_label()
    ROCOF_DISPLAY_UNIT = "Hz/s", FREQUENCY_DISPLAY_UNIT = "Hz" (constants)
- Created app/analytics/frequency/frequency_registry.py
    FrequencyRegistry — mutable session holder:
      classify(), is_frequency(), is_rocof(), is_frequency_or_rocof()
      frequency_channels(), rocof_channels() — bulk filtering from name lists
      frequency_panel_keys() — matches "frequency"/"rocof" and "{source_id}/frequency" etc.
      set_display_mode(), set_config(), clear_cache(), cached_roles
- Updated app/analytics/frequency/__init__.py — full public exports
- Updated app/ui/main_window/main_window.py
    Import FrequencyDisplayMode, FrequencyRegistry
    State: _frequency_registry, _frequency_display_mode_actions
    Tools → Frequency Display menu (Panel Only / Overlay / Off, QActionGroup exclusive)
    _on_frequency_display_mode_changed() handler + _apply_frequency_display_mode()
    _apply_frequency_display_mode(): shows/hides panel canvases via frequency_panel_keys()
- Created tests/unit/test_frequency_classification.py — 57 tests
    Name heuristics (frequency + ROCOF), unknown channels, unit-based classification,
    SignalMetadata.electrical_type + measurement_kind override, force_role operator override,
    display unit enforcement, is_frequency_channel/is_rocof_channel helpers
- Created tests/unit/test_frequency_display.py — 71 tests
    FrequencyDisplayMode enum, FrequencyChannelRole enum, FrequencyChannelResult frozen,
    FrequencyRegistry (mode, classify, cache, bulk helpers, panel keys),
    Shared-axis grouping (frequency→Hz, ROCOF→Hz/s, never merged),
    Frequency/ROCOF never share axis with voltage/current/power,
    Engineering unit display (Hz, Hz/s, never kHz), display label helpers,
    Panel title formatting for "frequency"/"rocof" groups
- Created tests/unit/test_frequency_visualization.py — 69 tests
    Channel grouper routes frequency/ROCOF correctly, metadata display_group override,
    RMS ineligibility for all frequency/ROCOF channel patterns,
    FrequencyRegistry bulk helpers with DisturbanceRecord channels,
    FrequencyConfig 50/60 Hz, classify_rocof helper,
    Signal visibility preserves frequency channels,
    Display mode OFF does not affect classification,
    Multi-source panel key detection (direct + "{source_id}/..." keys)

### Files Modified
- app/analytics/frequency/__init__.py           (replaced stub — NEW content)
- app/analytics/frequency/frequency_models.py   (NEW)
- app/analytics/frequency/frequency_overlay.py  (NEW)
- app/analytics/frequency/rocof_overlay.py      (NEW)
- app/analytics/frequency/frequency_registry.py (NEW)
- app/ui/main_window/main_window.py             (import + state + menu + 2 handlers)
- tests/unit/test_frequency_classification.py   (NEW)
- tests/unit/test_frequency_display.py          (NEW)
- tests/unit/test_frequency_visualization.py    (NEW)

### Architecture Impact
- New app/analytics/frequency/ package established; same structural pattern as app/analytics/rms/
- No changes to FlexiblePlotCanvas, VisualizationManager, or SynchronizationManager
- Frequency/ROCOF channels continue to render as standard analog channels inside their dedicated panels
- axis_management.py already grouped frequency:hz and rocof:hz/s correctly — confirmed by tests
- engineering_display.py already enforces Hz/Hz/s fixed units — confirmed by tests
- RMS eligibility already excludes frequency/ROCOF — confirmed by tests
- FrequencyRegistry.frequency_panel_keys() is the extension point for showing/hiding panels

### Performance Impact
- Zero per-frame cost: FrequencyRegistry is classification+metadata only; no waveform computation
- Classification cache avoids repeated heuristic evaluation across canvas redraws
- _apply_frequency_display_mode() calls setVisible() on panels — O(panel count), negligible

### Risks / Concerns
- FrequencyDisplayMode.OVERLAY: currently behaves identically to PANEL_ONLY (cross-panel
  secondary-axis overlay deferred). Documented in frequency_models.py docstring.
- No runtime Qt tests added for frequency display mode switching (no new Qt rendering paths).
  Signal Browser visibility toggling is already tested in test_runtime_qt_widgets.py.
- Waveform-derived frequency estimation (PLL, zero-crossing, DFT, ROCOF from waveform) is
  explicitly NOT implemented — deferred to Phase 5D or later per specification.

### Test Results
  197 new Phase 5C tests: all passing
  Full unit test suite: 2349 passed, 12 skipped, 0 failures

### Next Recommended Step
Phase 5D — Waveform-derived frequency estimation (zero-crossing, DFT tracking, ROCOF from
waveform derivative), OR commit current accumulated state to git and move to Phase 6 (Harmonic
Analysis Foundation / Phasor Hooks).

---

## Session — Phase 6A: Phasor & Sequence Component Engine
Date: 2026-05-17

### Summary
Full implementation of Phase 6A — Phasor & Sequence Component Engine. Transforms Powerwave into a
protection engineering platform with sliding-window DFT phasor extraction, Fortescue symmetrical
component computation, phase identification heuristics, and session-level registry/cache.

### New Modules
- app/analytics/phasors/phasor_models.py        — data models (enums, dataclasses)
- app/analytics/phasors/phasor_extraction.py    — vectorized sliding DFT extraction
- app/analytics/phasors/symmetrical_components.py — Fortescue transform + unbalance factor
- app/analytics/phasors/phasor_overlay.py       — classification + phase identification
- app/analytics/phasors/phasor_registry.py      — session-level registry with cache
- app/analytics/phasors/phasor_cache.py         — per-record result store
- app/analytics/phasors/__init__.py             — public package exports (29 symbols)

### Modified Files
- app/visualization/channel_grouper.py          — added DISPLAY_GROUP_SEQUENCE_VOLTAGE/CURRENT
- app/ui/main_window/main_window.py             — PhasorRegistry state + Phasor Display menu +
                                                  _on_phasor_display_mode_changed +
                                                  _apply_phasor_display_mode
- tests/unit/test_phasor_extraction.py          (NEW — 24 tests)
- tests/unit/test_symmetrical_components.py     (NEW — 39 tests)
- tests/unit/test_phasor_classification.py      (NEW — 86 tests)
- tests/unit/test_phasor_display.py             (NEW — 65 tests)

### Architecture Impact
- New app/analytics/phasors/ package following the same structural pattern as app/analytics/rms/ and
  app/analytics/frequency/. Does NOT modify FlexiblePlotCanvas, VisualizationManager, or synchronization.
- Phasor and sequence component channels continue to render as standard analog channels inside their
  dedicated panels. The existing rendering pipeline handles them correctly.
- channel_grouper.py extended with sequence_voltage and sequence_current groups for V1/V2/V0, I1/I2/I0.
- PhasorRegistry.phasor_panel_keys() is the extension point for showing/hiding phasor panels.

### Phasor Extraction Design
- Sliding-window DFT at fundamental frequency using stride-trick vectorization (O(N·W), no Python loops).
- DFT kernel: (2/W) * exp(-j·2π·k / n_cycle) where n_cycle = round(fs/f₀) — always targets f₀
  regardless of window size. Using n_cycle (not window) is the key fix enabling half/two-cycle modes.
- Output: (phasor_time, magnitude_rms, angle_deg, complex_phasor); right-aligned time convention.
- Phasor angle rotates at the carrier frequency (360°/cycle) — this is correct physical behavior.
  Relative phase between channels is the engineering quantity of interest.
- NaN/Inf → 0 replacement before DFT (same pattern as sliding_rms.py).

### Symmetrical Components Design
- Standard Fortescue transform: V0=(Va+Vb+Vc)/3, V1=(Va+a·Vb+a²·Vc)/3, V2=(Va+a²·Vb+a·Vc)/3
  where a = exp(j·2π/3).
- Engineering validation tests confirm: balanced ABC → |V2|≈0, |V0|≈0; SLG fault → elevated V0/V2;
  ACB reversal → elevated V2, suppressed V1.
- compute_sequence_from_phasor_arrays() end-to-end helper from extract_phasor() tuples.
- unbalance_factor() = |V2|/|V1|×100% (NSVUF per IEC 61000-2-2).

### Phase Identification Design
- Priority chain: force_phase > AnalogChannel.phase field > SignalMetadata.phase_reference > name heuristics.
- ABC suffix heuristics match universally (a/b/c/1/2/3); RYB heuristics (r/y) only match when
  channel name ≤4 chars or preceded by a separator — prevents false positives like "Channel_XY" → B.
- detect_three_phase_groups() returns only complete groups (all three phases identified).

### Performance Impact
- Zero per-frame cost for classification: PhasorRegistry is metadata-only with a classification cache.
- Phasor extraction is O(N·W) using stride tricks (no Python loops); typical: N=100k, W=100 → 10M ops.
- PhasorCache eliminates redundant phasor/sequence recomputation during viewport pan/zoom.
- _apply_phasor_display_mode() is O(panel count) — negligible.

### Risks / Concerns
- PhasorDisplayMode.MAGNITUDE and ANGLE: currently defined in the model and menu but the rendering
  integration (drawing magnitude/angle overlay curves on FlexiblePlotCanvas) is deferred. These modes
  show/hide panels but do not yet add new curve overlays. Phase 6B should add the rendering hookup.
- No runtime Qt tests for phasor display mode switching (no new Qt rendering paths in this phase).
- PMU synchrophasor protocol, impedance trajectory, relay elements, distance protection deferred.
- Phase identification for non-standard naming (e.g., Chinese substations) may need operator override.

### Test Results
  214 new Phase 6A tests: all passing
  Full unit test suite: 2563 passed, 12 skipped, 0 failures

### Next Recommended Step
Phase 6B — Phasor Rendering Integration: hookup PhasorDisplayMode.MAGNITUDE/ANGLE to draw magnitude
envelope and angle trace overlays on FlexiblePlotCanvas, and render sequence component panels (V1/V2/V0).
Alternatively: Phase 7 — Harmonic Analysis Foundation.

## Session — Phase 6B: Phasor Rendering Integration
Date: 2026-05-17

### Summary
Full implementation of Phase 6B — Phasor Rendering Integration. Makes the Phase 6A phasor engine
visible in the GUI: magnitude envelope and angle trace overlays rendered on waveform canvases via
the FlexiblePlotCanvas inline overlay pattern, plus dedicated hidden sequence component panels
(V1/V2/V0, I1/I2/I0) built at load time and toggled by PhasorDisplayMode.SEQUENCE_COMPONENTS.

### New Files
- app/visualization/overlays/phasor_overlay.py    — PhasorCurveOverlay(BaseOverlay); general-purpose
                                                     single-PlotItem phasor overlay using CurveStore
- tests/unit/test_phasor_rendering_integration.py — 48 tests for overlay lifecycle, pen colors,
                                                     sequence record building, routing, cache reuse

### Modified Files
- app/visualization/overlays/__init__.py          — added PhasorCurveOverlay export
- app/visualization/widgets/flexible_plot_canvas.py — Phase 6B phasor state (_phasor_display_mode,
                                                     _phasor_config, _phasor_curves, _phasor_cache,
                                                     _phasor_time_cache, _phasor_data_cache),
                                                     set_phasor_display_mode(), _build_phasor_overlays(),
                                                     _remove_phasor_curves(); set_record and
                                                     _rebuild_visible_channel_axes rebuild overlays
                                                     when active; _update_viewport decimates phasor
                                                     curves and contributes MAGNITUDE to Y-range
- app/ui/main_window/main_window.py               — _PANEL_ORDER updated (sequence_voltage/current after
                                                     current_raw); _make_sequence_record() module-level
                                                     function; _build_sequence_panels() builds hidden
                                                     FlexiblePlotCanvas instances with synthetic
                                                     DisturbanceRecords; _apply_phasor_display_mode()
                                                     rewired to delegate to canvas.set_phasor_display_mode
                                                     and toggle sequence panel visibility
- tests/unit/test_d441_stabilization.py           — TestFlexiblePlotCanvasAxisMode: added
                                                     canvas._phasor_display_mode = PhasorDisplayMode.OFF
                                                     to all 3 axis-mode mock tests (regression fix)

### Architecture Impact
- Rendering pattern follows the existing inline RMS overlay pattern in FlexiblePlotCanvas rather than
  the PhasorCurveOverlay class. Per-channel ViewBox access required for multi-axis plots mandates inline.
  PhasorCurveOverlay is the general-purpose class for simpler single-PlotItem hosts.
- Y-range safety: MAGNITUDE curves contribute to data_by_viewbox (same unit as raw waveform).
  ANGLE curves (degrees, [-180,180]) are explicitly excluded to prevent corrupting voltage/current axes.
- Sequence panels: start hidden via setVisible(False), sole visibility controller is
  _apply_phasor_display_mode(). No phasor overlays on sequence canvases (key detection skips them).
- PhasorCache: keyed by (channel_id, window_samples, nominal_hz). Full tuple (phasor_time, mag_rms,
  angle_deg, cpx) stored once; MAGNITUDE↔ANGLE switches reuse without recomputation.
- Pen colors: magnitude = 60% blend toward white (dotted line); angle = 40% blend toward cyan #00FFFF
  (dash-dot line). Visually distinct from raw waveform and RMS overlay curves.
- Phasor overlay rebuild: triggered in set_record() and _rebuild_visible_channel_axes() when mode
  is MAGNITUDE or ANGLE (not OFF, not SEQUENCE_COMPONENTS).
- Multi-source sequence panels (multi-record grouped display): deferred. _build_sequence_panels()
  handles single-record CSV/Excel/Synthetic only.

### Performance Impact
- PhasorCache eliminates phasor recomputation on mode switch and viewport pan/zoom.
- _update_viewport() decimates phasor arrays before rendering (same decimate_for_display as raw data).
- _build_phasor_overlays() lazy-imports; avoids top-level phasor import cost in non-phasor sessions.
- Sequence panel construction happens once at load time; no per-frame cost.

### Risks / Concerns
- Multi-source sequence panels (Phase 6B.1 limitation): _build_sequence_panels() only processes the
  single active record. Multi-source grouped display requires per-source phasor computation and
  namespaced sequence keys (e.g., "source_id/sequence_voltage").
- Sample rate estimation in _build_phasor_overlays() uses SamplingInformation.sampling_rates[0]
  with fallback heuristic from time column delta. Irregular-sample records could produce incorrect fs.
- SEQUENCE_COMPONENTS panel title and axis labels use synthetic channel names (V1/V2/V0, I1/I2/I0).
  No unit-aware labeling for these derived channels yet.

### Test Results
  48 new Phase 6B tests: all passing
  Full unit test suite: 2657 passed, 12 skipped, 0 failures
  (includes 3 test_d441_stabilization regression fixes)

### Next Recommended Step
Phase 7 — Harmonic Analysis Foundation: FFT spectrum display for selected channels with harmonic
order annotation and THD readout. Alternatively, relay element overlays or impedance trajectory (R-X).

## Session — Phase 7: Harmonic Analysis Foundation
Date: 2026-05-17

### Summary
Full implementation of Phase 7 — Harmonic Analysis Foundation. Establishes the complete analytics
infrastructure for harmonic analysis: sliding-window FFT extraction engine, THD/distortion metrics,
harmonic channel classification, session-level registry, result cache, and visualization hooks.
No harmonic rendering UI is wired yet (Phase 7B).

### New Files
- app/analytics/harmonics/harmonic_models.py       — HarmonicDisplayMode, HarmonicWindowMode,
                                                      HarmonicChannelRole, HarmonicChannelResult,
                                                      HarmonicConfig, HarmonicResult
- app/analytics/harmonics/harmonic_extraction.py   — compute_harmonic_window_samples,
                                                      extract_harmonics (vectorized stride-trick FFT)
- app/analytics/harmonics/harmonic_metrics.py      — compute_thd (scalar), compute_thd_array
                                                      (vectorized), compute_thd_from_result,
                                                      individual_harmonic_distortion
- app/analytics/harmonics/harmonic_overlay.py      — classify_harmonic_role (5-level priority chain),
                                                      is_harmonic_eligible
- app/analytics/harmonics/harmonic_registry.py     — HarmonicRegistry (session cache + display mode
                                                      + bulk helpers)
- app/analytics/harmonics/harmonic_cache.py        — HarmonicCache (keyed by channel_id,
                                                      window_samples, hop_samples, nominal_hz,
                                                      max_order)
- app/analytics/harmonics/__init__.py              — public package exports (18 symbols)
- tests/unit/test_harmonic_extraction.py           ← 47 tests
- tests/unit/test_harmonic_metrics.py              ← 30 tests
- tests/unit/test_harmonic_classification.py       ← 44 tests
- tests/unit/test_harmonic_registry.py             ← 36 tests

### Modified Files
- app/visualization/widgets/flexible_plot_canvas.py — Phase 7 harmonic hooks: import
                                                       HarmonicConfig/HarmonicDisplayMode;
                                                       _harmonic_display_mode, _harmonic_config,
                                                       _harmonic_signal_metadata state;
                                                       set_harmonic_display_mode() no-op stub
- app/ui/main_window/main_window.py                — import HarmonicRegistry; _harmonic_registry
                                                      state; disabled "Harmonic Analysis…" Tools menu
                                                      item (Phase 7B placeholder)

### Harmonic Extraction Design
- FFT-based sliding-window approach using np.lib.stride_tricks.as_strided to build
  (n_windows, window_samples) view; single np.fft.rfft call processes all windows at once.
- Window function: Hann (default) or rectangular; selected via HarmonicConfig.window_function.
- Amplitude-correct RMS normalisation: mag_rms = sqrt(2) * |FFT[bin]| / sum(window)
  Verified: 100V-peak sine → 70.7V RMS within 0.5% for on-bin harmonics.
- Harmonic bin selection: bin_n = round(n * nominal_hz * window_samples / sample_rate_hz)
  Exact bin alignment is guaranteed when window contains an integer number of power cycles.
- Three window modes: ONE_CYCLE (1×) / TWO_CYCLE (2×, default) / FOUR_CYCLE (4×).
- Time axis: right-aligned (harmonic_time[i] = end of window i), consistent with phasor convention.
- NaN/Inf samples: replaced with 0.0 before FFT — no silent propagation.
- Signals shorter than one window: HarmonicResult.empty() returned.
- Overlap: configurable (default 0.5); clamped to [0.0, 0.999].

### THD Computation
- Standard engineering definition: THD = sqrt(sum(H2..HN)^2) / H1
- Returns dimensionless fraction (0.05 = 5% THD). Callers multiply by 100 for display.
- compute_thd(): scalar, from {order: magnitude} dict.
- compute_thd_array(): vectorized time-varying, from {order: ndarray} dict.
- compute_thd_from_result(): convenience wrapper over HarmonicResult.
- Safe: fundamental < safe_threshold (1e-9) → returns 0.0, no ZeroDivisionError.
- Non-finite or negative harmonic magnitudes silently excluded from distortion sum.
- max_order parameter limits which orders contribute (e.g. max_order=25 for IEC 61000-3-2).

### Harmonic Classification
- Priority chain: force_role > measurement_kind > electrical_type > unit > name heuristics > UNKNOWN
- Ineligible measurement_kinds: rms, average, calculated, telemetry, frequency, rocof
- Ineligible electrical_types: power, frequency, rocof
- Ineligible name fragments: rms, vrms, irms, vpu, _pu, mw, mvar, hz, freq, rocof, power, telemetry
- Eligible: measurement_kind "voltage"/"current"/"voltage_phasor"/"current_phasor"; electrical_type
  "voltage"/"current"; unit kV/V/A/kA; name prefix V (≥2 chars) → voltage, I (≥2 chars) → current.
- "instantaneous" measurement_kind falls through to electrical_type/unit/name (not auto-excluded).
- UNKNOWN role means ineligible — not classified as harmonic-capable.

### Harmonic Cache/Registry Design
- HarmonicCache key: (channel_id, window_samples, hop_samples, nominal_hz, max_order).
  Display mode switch (HARMONIC_MAGNITUDE ↔ THD) does NOT change the key — results reused.
- HarmonicRegistry: session-level mutable, UI thread only; classification cache keyed by name.
  set_config() does not auto-invalidate role cache (config changes only affect extraction parameters).
  Bulk helpers: harmonic_eligible_channels(), voltage_harmonic_channels(), current_harmonic_channels().
- Design mirrors PhasorRegistry/PhasorCache for architectural consistency.

### Visualization Hooks
- FlexiblePlotCanvas.set_harmonic_display_mode(mode, config, signal_metadata): no-op stub.
  Stores mode/config/metadata for Phase 7B rendering wiring. No curves built yet.
- main_window.py: self._harmonic_registry = HarmonicRegistry() in __init__.
- Tools → Harmonic Analysis… menu item: disabled (setEnabled(False)), Phase 7B placeholder.
- No changes to VisualizationManager (not required for foundation phase).

### Performance Impact
- Zero per-frame rendering cost (no harmonic curves rendered yet).
- extract_harmonics() is O(N·log(W)) using batch rfft — all windows computed in one call.
- HarmonicCache eliminates recomputation on display mode switch and viewport pan/zoom.
- timed_section() hook in performance.py is available for future profiling integration.

### Risks / Concerns
- Off-bin harmonics: when window_samples is not an exact integer multiple of samples-per-cycle
  (e.g. 60 Hz at 5000 Hz sample rate: 5000/60 = 83.33), there is slight spectral leakage.
  Hann window suppresses this to < 2% relative magnitude for adjacent bins. For production use
  at 60 Hz with 5000 Hz sample rate, consider using 6000 Hz or padding to a cycle-aligned size.
- Interharmonics, sub-harmonics, and spectral waterfall not implemented (Phase 7C+).
- Harmonic rendering (HARMONIC_MAGNITUDE / THD panels) not wired (Phase 7B).
- Multi-source harmonic sessions not considered yet.

### Test Results
  157 new Phase 7 tests: all passing
  Full unit test suite: 2824 passed, 12 skipped, 0 failures

### Next Recommended Step
Phase 7B — Harmonic Overlay Rendering: wire set_harmonic_display_mode() to build per-order magnitude
curves (H1–H13 visible by default) and a THD trend panel. Alternatively: Phase 8 — Event Detection
Foundation (voltage sag/swell/interruption classification).

---

## Phase 8 — Harmonic Rendering Integration: COMPLETE 2026-05-17

### What Was Implemented
Full Phase 8 harmonic rendering wired end-to-end: inline magnitude overlays on waveform
canvases (HARMONIC_MAGNITUDE mode), THD trend panels, and harmonic spectrum panels.

### Files Modified

- **app/visualization/overlays/harmonic_overlay.py** (CREATED):
  HarmonicCurveOverlay(BaseOverlay) — general-purpose overlay using CurveStore keyed by
  (channel_name, order) tuples. Methods: update_channel(), remove_channel(), remove_order(),
  channel_order_pairs(). Lifecycle hooks: _attach/_detach/_set_items_visible/_clear/_dispose.

- **app/visualization/overlays/overlay_colors.py** (MODIFIED):
  Added per-order deterministic colors (_HARMONIC_ORDER_COLORS: H1=#808080, H3=#FF6600,
  H5=#FF00CC, H7=#00CCFF, H11=#AA88FF, H13=#FF88AA). Added functions: harmonic_order_color(),
  harmonic_order_pen(), thd_pen(), harmonic_curve_label(), thd_curve_label().

- **app/visualization/overlays/__init__.py** (MODIFIED):
  Added exports: HarmonicCurveOverlay, harmonic_order_color, harmonic_order_pen,
  harmonic_curve_label, thd_pen, thd_curve_label.

- **app/visualization/widgets/flexible_plot_canvas.py** (MODIFIED):
  Phase 8 harmonic state: _harmonic_cache, _harmonic_curves, _harmonic_time_cache,
  _harmonic_data_cache, _harmonic_display_orders=[3,5,7,11,13].
  Methods added: set_harmonic_display_mode() (full routing, not stub), _build_harmonic_overlays()
  (lazy HarmonicCache, FFT extraction, per-order pg.PlotDataItem per ViewBox),
  _remove_harmonic_curves() (removes from ViewBoxes, preserves cache).
  _update_viewport() extended: HARMONIC_MAGNITUDE decimates harmonic arrays and contributes
  data to Y-range via data_by_viewbox.
  set_record() and _rebuild_visible_channel_axes() rebuild harmonic overlays when mode active.

- **app/ui/main_window/main_window.py** (MODIFIED):
  _PANEL_ORDER extended with "thd_voltage", "thd_current", "harmonic_spectrum_voltage",
  "harmonic_spectrum_current". _HARMONIC_PANEL_KEYS frozenset added.
  _make_harmonic_record() module-level helper (mirrors _make_sequence_record()).
  self._harmonic_display_mode_actions dict added in __init__.
  Tools → Harmonic Display submenu: OFF/Magnitude Overlay/THD Trend/Spectrum Panels.
  Methods added: _on_harmonic_display_mode_changed(), _apply_harmonic_display_mode(),
  _build_harmonic_panels().
  _build_harmonic_panels() called in both _on_load_synthetic_mixed() and _handle_direct_csv_excel().

- **tests/unit/test_harmonic_rendering.py** (CREATED):
  HarmonicCurveOverlay lifecycle/attach/detach/dispose, no-duplicate-curves guarantee,
  harmonic color/pen helpers (deterministic hex, pen style), _make_harmonic_record() pure
  logic, _apply_harmonic_display_mode() routing (fake canvas, all modes), _build_harmonic_panels()
  correctness (unknown channels → {}, voltage-only → thd_voltage + spectrum_voltage, mixed).
  69 tests.

- **tests/unit/test_harmonic_overlay_stability.py** (CREATED):
  Offscreen Qt stability tests: harmonic pen determinism (qapp), FlexiblePlotCanvas mode
  cycle OFF→MAGNITUDE→THD→SPECTRUM→OFF, no duplicate curves on repeated MAGNITUDE calls,
  new-record rebuild, RMS channel skipped, performance timing fires,
  cache preserved across OFF→MAGNITUDE transition, PowerwaveMainWindow._build_harmonic_panels()
  round-trip with real FlexiblePlotCanvas.
  17 tests.

- **tests/unit/test_harmonic_stability.py** (MODIFIED):
  test_canvas_harmonic_hooks_are_safe_noops: removed stale Phase 7 stub calls
  (show_harmonic_panels/hide_harmonic_panels/_harmonic_panels_visible on canvas).
  Now tests the real Phase 8 API (mode transitions, _harmonic_curves == {}).

- **tests/unit/test_d441_stabilization.py** (MODIFIED):
  TestFlexiblePlotCanvasAxisMode: all 3 tests updated to set canvas._harmonic_display_mode =
  HarmonicDisplayMode.OFF on the spec mock (Phase 8 added _harmonic_display_mode check
  in set_record() path).

- **tests/unit/test_runtime_qt_widgets.py** (MODIFIED):
  test_synthetic_grouped_panels_keep_x_pixel_alignment: filters to visible canvases only
  (hidden harmonic panels have no valid pixel geometry).

### Key Design Decisions

**HARMONIC_MAGNITUDE mode**: per-order PlotDataItem per channel/ViewBox pair. H1 omitted
(_harmonic_display_orders = [3,5,7,11,13]) — fundamental is too large relative to harmonics
and visually dominates. HarmonicCache shared across viewport updates — O(1) cache hit on
pan/zoom (no FFT recompute). _update_viewport() decimates before render (same path as raw data).

**THD/SPECTRUM mode**: handled entirely by dedicated hidden panels in main_window.py.
FlexiblePlotCanvas.set_harmonic_display_mode(THD/SPECTRUM) just clears MAGNITUDE overlays
and shows raw waveform only. Panel visibility toggled by _apply_harmonic_display_mode().

**THD panels**: synthetic DisturbanceRecord where channels = source channel names,
values = compute_thd_array() * 100 (percent). Units = "%".

**Spectrum panels**: first eligible voltage or current channel only; channels = H3/H5/H7/H11/H13,
values = per-order RMS magnitude arrays. Units = "V RMS" or "A RMS".

**_HARMONIC_PANEL_KEYS** frozenset: THD/spectrum keys excluded from waveform canvas list in
_apply_harmonic_display_mode() — same pattern as _is_sequence_key() for phasor.

**Cache preservation**: _remove_harmonic_curves() clears curve/time/data dicts but NOT the
HarmonicCache. Re-enabling HARMONIC_MAGNITUDE reuses FFT results. Full cache clear only on
_clear_canvas() (new record load).

### Regression Fixes Applied
- TestFlexiblePlotCanvasAxisMode (3 tests): spec mock missing _harmonic_display_mode.
  Fix: added canvas._harmonic_display_mode = HarmonicDisplayMode.OFF.
- test_synthetic_grouped_panels_keep_x_pixel_alignment: harmonic panels start hidden and have
  no pixel geometry. Fix: filter canvases list to visible-only before pixel alignment assertion.
- test_canvas_harmonic_hooks_are_safe_noops: stale Phase 7 stub API (show_harmonic_panels,
  _harmonic_panels_visible) no longer exists on FlexiblePlotCanvas. Fix: test now uses
  real Phase 8 API (set_harmonic_display_mode transitions, _harmonic_curves == {}).

### Test Results
  69 + 17 = 86 new Phase 8 tests: all passing
  Full unit test suite: 2925 passed, 12 skipped, 0 failures

### Next Recommended Step
Phase 8B — Interactive spectrum panel (optional: order selection, frequency bin hover tooltip).
Or Phase 9 — Event Detection Foundation (voltage sag/swell/interruption classification).

---

## 2026-05-18 - Phase 8.5 Harmonic Visualization Stabilization & Performance Hardening

### Agent
Codex

### Task
Stabilize harmonic visualization lifecycle, cache reuse, spectrum panel behavior, cursor synchronization, and large-record redraw behavior without adding new harmonic analytics or redesigning overlay infrastructure.

### Completed
- Reviewed authoritative architecture/workflow contracts before final tracking update: docs/SYSTEM_OVERVIEW.md, docs/ARCHITECTURE.md, docs/DATA_CONTRACT.md, docs/REPOSITORY_STRUCTURE.md, docs/VISUALIZATION_CONTRACT.md, docs/VIEWPORT_RENDERING_POLICY.md, docs/PERFORMANCE_REQUIREMENTS.md, agent/WORKFLOW_AGENT.md, agent/HANDOFF.md, agent/TASK.md, and agent/REPOSITORY_STATE.md.
- Confirmed requested docs/REPOSITORY_STRUCTURE.md is present and used it as the repository-structure contract alongside ARCHITECTURE.md and REPOSITORY_STATE.md.
- Added a lightweight per-curve data signature guard in FlexiblePlotCanvas so repeated synchronized viewport echoes do not resend identical setData() payloads.
- Preserved the existing curve lifecycle rule: curves are still created/rebuilt only at lifecycle boundaries and updated in-place with setData() during viewport rendering.
- Added a main-window harmonic panel cache keyed by record identity so repeated THD/spectrum panel rebuilds for the same record reuse HarmonicCache results.
- Added focused Phase 8.5 runtime tests covering harmonic viewport redraw idempotence, OFF switching, unsupported channel cache safety, partial waveform/telemetry support, repeated harmonic panel cache reuse, hidden spectrum cursor stability, and cursor synchronization.
- Ran full verification after fixing one spec-mock compatibility issue in scaling tests.

### Files Modified
- app/visualization/widgets/flexible_plot_canvas.py
- app/ui/main_window/main_window.py
- tests/unit/test_harmonic_visualization_stability.py
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No architecture redesign. Changes remain within existing subsystem boundaries: visualization hot path still consumes cached numpy arrays from DisturbanceRecord and uses decimate_for_display(); harmonic analytics remain in app/analytics/harmonics; HarmonicCache key semantics are preserved; no new overlay registry or lifecycle abstraction was introduced; SynchronizationManager behavior is unchanged.

### Performance Impact
- Reduces redundant PyQtGraph/OpenGL buffer updates when synchronized panels receive identical viewport data.
- Avoids repeated FFT extraction during repeated harmonic THD/spectrum panel rebuilds for the same record.
- Cursor movement remains lightweight: no harmonic FFT rebuilds and no plot item creation.
- Existing offscreen PyQtGraph OpenGL warnings still appear in runtime tests but remain non-failing environment noise.

### Risks / Concerns
- The per-curve data signature is intentionally lightweight (length and endpoints). It is a redraw-suppression guard for repeated identical viewport outputs, not a full content hash.
- Harmonic panel cache is keyed by Python record identity; new record objects receive a fresh cache, matching current reload lifecycle.
- docs/REPOSITORY_STRUCTURE.md is present in this checkout and was included in the architecture review.

### Test Results
- Focused harmonic/scaling slice: 92 passed.
- Runtime/synchronization/overlay slice: 88 passed.
- Full unit suite: 2278 passed.
- Full test suite: 2932 passed, 12 skipped.

### Next Recommended Step
Proceed to Phase 9 only after architecture review acknowledges Phase 8.5 as stabilization-only. Future spectrogram/waterfall work should reuse HarmonicCache/HarmonicRegistry patterns and avoid adding a separate FFT cache path.

---

## 2026-05-18 - Phase 8.55A/B Import Wizard Architecture & File Profiling Engine

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Phase 8.55A: Design and implement complete Import Wizard data contracts (no GUI, no normalization engine).
Phase 8.55B: Implement the backend file profiling engine for CSV/Excel import.

### Completed — Phase 8.55A
- Created `app/import_wizard/` package (7 files):
  - `contracts.py` — ValidationSeverity enum, ValidationMessage frozen dataclass
  - `wizard_state.py` — WizardStep enum, can_transition(), next_step(), steps_before(), step_index()
  - `column_mapping.py` — ParameterType enum (9 types)
  - `timestamp_contracts.py` — TimestampRepairStrategy (8 strategies), TimestampRepairPlan frozen dataclass
  - `normalization_plan.py` — NormalizationPlan with is_executable property
  - `models.py` — RawPreviewModel, TimestampCandidate, ColumnMappingCandidate (with effective_* properties), ImportWizardSession
  - `__init__.py` — 17 exported symbols
- Created `tests/unit/test_import_wizard_contracts.py` (101 tests, all passing)

### Completed — Phase 8.55B
- `app/import_wizard/preview_sampler.py` — read_text_sample(), estimate_csv_row_count(); pure stdlib; encoding auto-detect (utf-8-sig → utf-8 → latin-1)
- `app/import_wizard/csv_profiler.py` — detect_delimiter() (csv.Sniffer + fallback counting), _find_header_row_index() (lookahead: next non-blank row ≥50% numeric), profile_csv()
- `app/import_wizard/excel_profiler.py` — get_sheet_names(), profile_excel() (openpyxl read_only=True, same lookahead header detection)
- `app/import_wizard/timestamp_detector.py` — infer_timestamp_format() (strptime + epoch/excel-serial special formats), detect_timestamp_candidates() with confidence scoring (base parse_rate + name_boost + monotonic_boost - dupe_penalty)
- `app/import_wizard/column_detector.py` — classify_by_name() (regex rules for all 9 ParameterTypes), _classify_by_values() (binary→DIGITAL, freq range→FREQUENCY), detect_column_mappings()
- `app/import_wizard/file_profiler.py` — FileProfileResult dataclass, profile_import_file() (auto CSV/Excel dispatch, never raises), populate_session()
- Updated `app/import_wizard/__init__.py` — now exports 19 symbols including FileProfileResult, profile_import_file, populate_session
- `tests/unit/test_import_wizard_file_profiling.py` (44 tests)
- `tests/unit/test_import_wizard_timestamp_detection.py` (37 tests)
- `tests/unit/test_import_wizard_column_detection.py` (30+ tests)

### Files Modified/Created
- app/import_wizard/__init__.py (updated)
- app/import_wizard/preview_sampler.py (created)
- app/import_wizard/csv_profiler.py (created)
- app/import_wizard/excel_profiler.py (created)
- app/import_wizard/timestamp_detector.py (created)
- app/import_wizard/column_detector.py (created)
- app/import_wizard/file_profiler.py (created)
- tests/unit/test_import_wizard_file_profiling.py (created)
- tests/unit/test_import_wizard_timestamp_detection.py (created)
- tests/unit/test_import_wizard_column_detection.py (created)

### Architecture Impact
No impact on existing subsystems. New `app/import_wizard/` package is fully isolated with no Qt/numpy/pandas imports in the contract/profiling layer. Dependency chain: contracts → wizard_state → column_mapping → timestamp_contracts → normalization_plan → models → profiling modules.

### Performance Impact
- preview_sampler reads at most 200 lines (configurable)
- csv_profiler reads at most max_scan_rows lines (default 200)
- excel_profiler uses openpyxl read_only=True (streaming)
- Row count estimation uses 64 KB sample + extrapolation (never full file read)

### Test Results
- Phase 8.55B new tests (149): all passing
- Full unit test suite: 2528 passed, 0 failures

### Next Recommended Step
Phase 8.55C — Import Wizard UI (PyQt6 wizard dialog: steps LOAD_FILE → RAW_PREVIEW → TIMESTAMP_SELECT → COLUMN_MAPPING → NORMALIZATION_REVIEW → SAVE).
Or Phase 8.55C — Normalization Engine (consume NormalizationPlan to produce DisturbanceRecord-compatible output).

---

## 2026-05-18 - Phase 8.55C — Timestamp Repair & Normalization Engine

### Agent
Claude Code (claude-sonnet-4-6)

### Task
Implement the backend timestamp repair and normalization engine for the Import Wizard pipeline.

### Completed
- `app/import_wizard/interval_inference.py` — IntervalAnalysis dataclass; infer_interval() (µs-resolution mode, order-preserving non-monotonic detection, sorted-diff duplicate detection); detect_duplicates(); detect_non_monotonic()
- `app/import_wizard/repair_diagnostics.py` — RepairDiagnostics dataclass (17 fields, all serializable/plain-Python)
- `app/import_wizard/timestamp_repair_executor.py` — 8 strategy executors + dispatch():
  - execute_no_repair (format="mixed" for pandas 3.0)
  - execute_parse_detected_format (supports epoch_seconds/ms and excel_serial labels)
  - execute_parse_user_format
  - execute_interpolate_missing (timedelta-relative arithmetic for precision-safe interpolation)
  - execute_reconstruct_from_interval (pd.date_range to avoid ns/µs unit issues)
  - execute_combine_date_time_columns
  - execute_excel_serial_conversion (pd.to_timedelta approach, not raw ns arithmetic)
  - execute_timezone_alignment (isinstance(dtype, DatetimeTZDtype) check for pandas 3.0)
- `app/import_wizard/timestamp_normalizer.py` — TimestampNormalizationResult dataclass; normalize_timestamps(); post-repair quality checks
- `tests/unit/test_timestamp_normalization.py` (40 tests)
- `tests/unit/test_timestamp_repair.py` (50+ tests)
- `tests/unit/test_interval_inference.py` (45+ tests)

### Key Pandas 3.0 Compatibility Fixes
- Used format="mixed" instead of infer_datetime_format=True (removed in 3.0)
- datetime64[us] default dtype: all arithmetic uses .diff().dt.total_seconds() and pd.Timedelta, never raw int64
- isinstance(dtype, pd.DatetimeTZDtype) for timezone-aware check (not .dt.tz which pyright misclassifies)
- µs-resolution mode bins for infer_interval (ms bins rounded 50µs to 0)

### Files Created
- app/import_wizard/interval_inference.py
- app/import_wizard/repair_diagnostics.py
- app/import_wizard/timestamp_repair_executor.py
- app/import_wizard/timestamp_normalizer.py
- tests/unit/test_timestamp_normalization.py
- tests/unit/test_timestamp_repair.py
- tests/unit/test_interval_inference.py

### Test Results
- Phase 8.55C new tests (109): all passing
- Full unit test suite: 2637 passed, 0 failures

### Next Recommended Step
Phase 8.55D — Import Wizard UI (PyQt6 wizard dialog stepping through LOAD_FILE → RAW_PREVIEW → TIMESTAMP_SELECT → TIMESTAMP_REPAIR → COLUMN_MAPPING → NORMALIZATION_REVIEW → SAVE_NORMALIZED).
Or: Integration — wire normalize_timestamps() + FileProfileResult into a normalize_dataframe() function that produces a DisturbanceRecord.

---

## 2026-05-18 - Phase 8.55G - Import Wizard Qt GUI Skeleton

### Agent
Codex (GPT-5)

### Task
Implement the first operational Qt GUI skeleton for the CSV/Excel Import Wizard:
file selection, profiling preview, timestamp selection, column mapping review,
backend pipeline execution, DisturbanceRecord generation, and visualization handoff.

### Architecture Review
Reviewed and treated as authoritative before implementation:
- docs/SYSTEM_OVERVIEW.md
- docs/ARCHITECTURE.md
- docs/DATA_CONTRACT.md
- docs/REPOSITORY_STRUCTURE.md
- docs/VISUALIZATION_CONTRACT.md
- docs/VIEWPORT_RENDERING_POLICY.md
- docs/PERFORMANCE_REQUIREMENTS.md
- agent/WORKFLOW_AGENT.md
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

Findings:
- UI belongs under app/ui/ and must remain Qt orchestration only.
- Backend import_wizard modules own profiling, timestamp repair, normalization,
  assembly, pipeline execution, and DisturbanceRecord bridging.
- Visualization handoff should reuse the existing main-window VisualizationManager
  and grouped display flow.
- Performance contracts require previews and background workers rather than
  loading full files into widgets.

### Completed
- Created app/ui/import_wizard/ package:
  - __init__.py
  - preview_table_model.py
  - timestamp_candidate_model.py
  - column_mapping_model.py
  - wizard_pages.py
  - import_wizard_dialog.py
- Implemented QDialog + QStackedWidget wizard pages:
  LOAD_FILE, RAW_PREVIEW, TIMESTAMP_SELECT, COLUMN_MAPPING,
  REVIEW_IMPORT, IMPORT_RUNNING, IMPORT_COMPLETE.
- Added QAbstractTableModel adapters for preview rows, timestamp candidates, and
  editable column mapping review.
- Added QRunnable workers for profile_import_file() and run_import_pipeline().
- Added graceful error/status handling for profile and pipeline failures.
- Wired successful DisturbanceRecord imports to app/ui/main_window/main_window.py
  via File > Import Wizard... and the existing visualization manager path.
- Added tests:
  - tests/unit/test_import_wizard_gui.py
  - tests/runtime/test_import_wizard_runtime.py

### Validation
- Import Wizard GUI/runtime tests: 8 passed.
- Import backend + GUI slice: 622 passed.
- Qt runtime visualization slice: 103 passed.
- Full suite: 3633 passed, 12 skipped.

### Known Limitations
- Column mapping edits are preserved in the dialog/session NormalizationPlan, but
  run_import_pipeline() currently exposes only an auto-plan execution path. A future
  phase should add a plan-aware pipeline entry point before advanced mapping edits
  become authoritative for execution.
- Timestamp repair controls are represented by selected-candidate repair planning,
  but the dedicated TIMESTAMP_REPAIR page remains future work.
- Existing offscreen PyQtGraph OpenGL warnings still appear in runtime visualization
  tests, but all assertions pass.

### Next Recommended Step
Phase 8.55H - Plan-aware Import Wizard execution plus timestamp repair controls,
then UX polish and larger-file runtime hardening.

---

## 2026-05-19 - Phase 8.55I - Timestamp Format Override UI

### Agent
Codex

### Task
Implement manual timestamp format override support in the Import Wizard GUI and make the override authoritative during plan-aware execution.

### Completed
- Reviewed authoritative architecture, visualization/performance, workflow, handoff, task, and repository-state docs before editing.
- Added backend sampled validation for manual strptime format overrides.
- Added timestamp page controls for selected timestamp column, detected format, manual override input, reset-to-detected, and parse preview feedback.
- Wired override edits into ImportWizardSession.timestamp_repair_plan using TimestampRepairStrategy.PARSE_USER_FORMAT when non-empty.
- Preserved detected-format behavior when the override field is cleared.
- Integrated validation into build_execution_plan() so complete parse failure blocks execution and partial parse success is surfaced as a warning.
- Hardened run_import_pipeline_with_plan() so unvalidated timestamp repair plans fail gracefully before full file load.
- Added deterministic unit and runtime coverage for valid/invalid/partial override validation, stale plan invalidation, detected-format restoration, user-format execution authority, and DisturbanceRecord timing.

### Files Modified
- app/import_wizard/__init__.py
- app/import_wizard/import_pipeline.py
- app/import_wizard/pipeline_plan_builder.py
- app/import_wizard/timestamp_format_validator.py
- app/ui/import_wizard/import_wizard_dialog.py
- app/ui/import_wizard/wizard_pages.py
- tests/unit/test_timestamp_override_ui.py
- tests/runtime/test_timestamp_override_execution.py
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No redesign. UI remains a thin Qt orchestration layer; validation and execution decisions live in app/import_wizard. The existing TimestampRepairPlan and run_import_pipeline_with_plan() authority path are reused. No timestamp normalization engine logic was duplicated.

### Performance Impact
Override validation uses only candidate example samples with a hard max of 50 rows. No full dataset parsing occurs during editing. Existing QRunnable execution flow remains unchanged for import execution.

### Risks / Concerns
- Validation quality depends on profiler sample representativeness.
- Existing repository temp directories have Windows permission issues; broader pytest slices that use tmp_path fail during setup before assertions run.
- Some Phase 8.55H files are currently untracked in git status in this checkout, but they are required by the current plan-aware import wizard path.

### Test Results
- .venv\\Scripts\\python.exe -m pytest tests\\unit\\test_timestamp_override_ui.py tests\\runtime\\test_timestamp_override_execution.py -q
  - 12 passed
- .venv\\Scripts\\python.exe -m py_compile app\\import_wizard\\timestamp_format_validator.py app\\import_wizard\\pipeline_plan_builder.py app\\import_wizard\\import_pipeline.py app\\ui\\import_wizard\\import_wizard_dialog.py app\\ui\\import_wizard\\wizard_pages.py
  - passed
- Broader import wizard slice attempted; blocked by pytest tmp_path PermissionError at C:\\Users\\fairizat\\AppData\\Local\\Temp\\pytest-of-fairizat.

### Next Recommended Step
Phase 8.55J - import wizard runtime hardening: resolve repository/pytest temp-directory permission hygiene, then run the full import-wizard and runtime visualization slices again.

---

## 2026-05-19 - Phase 8.55J - Test Environment Stabilization & Runtime Hygiene

### Agent
Codex

### Task
Stabilize pytest/runtime execution after the Import Wizard backend, Qt skeleton, authoritative execution, and timestamp override phases.

### Completed
- Reviewed authoritative architecture, visualization/performance, workflow, handoff, task, and repository-state docs before editing.
- Added repository-native runtime temp helpers for isolated temp dirs, safe cleanup, and short Windows-lock retries.
- Redirected pytest/Python temp roots away from the user profile and into `.powerwave_runtime_tmp`.
- Added a narrow Windows pytest shim for pytest-created temp directories because Python 3.14 `mode=0o700` directories are unreadable in this sandbox.
- Added Qt runtime teardown fixture that drains QThreadPool work, closes top-level widgets, deletes Qt objects, and processes posted events.
- Added deterministic runtime hygiene tests for CSV/XLSX cleanup, Qt worker cleanup, repeated Import Wizard runtime execution, timestamp override repeatability, and safe cleanup reporting.
- Added gitignore entries for generated runtime/temp artifacts and configured pytest cache under the repo-local runtime temp area.

### Files Modified
- .gitignore
- pyproject.toml
- app/testing/__init__.py
- app/testing/temp_runtime.py
- tests/conftest.py
- tests/runtime/conftest.py
- tests/runtime/test_runtime_environment.py
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No product runtime redesign. New `app/testing` utilities are test/runtime support only. Import Wizard execution, timestamp normalization, DisturbanceRecord construction, visualization handoff, and Qt dialog architecture are unchanged.

### Performance Impact
Cleanup scans only immediate runtime-temp children and uses short bounded retries. Qt cleanup waits are bounded. No large filesystem scans, full dataset parsing, or rendering-path changes were introduced.

### Risks / Concerns
- Several pre-existing stale temp directories in the checkout still have Windows AccessDenied ACLs and cannot be removed from this session. They are ignored by the new pytest temp root and gitignore entries.
- The pytest temp-mode shim is intentionally Windows-only and exists because this sandbox/Python combination creates unreadable `0o700` directories.

### Test Results
- `.venv\\Scripts\\python.exe -m pytest tests\\runtime\\test_runtime_environment.py -q` -> 7 passed.
- Broad import/runtime slice repeated twice:
  `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_plan_aware_pipeline.py tests\\unit\\test_import_wizard_gui.py tests\\runtime\\test_import_wizard_authoritative_flow.py tests\\unit\\test_timestamp_override_ui.py tests\\runtime\\test_timestamp_override_execution.py tests\\runtime\\test_import_wizard_runtime.py tests\\runtime\\test_runtime_environment.py -q`
  -> 71 passed, then 71 passed.

### Next Recommended Step
Proceed with the next Import Wizard feature/runtime phase. Keep new runtime tests in the standard import-wizard validation slice so temp and Qt regressions are caught early.

---

## 2026-05-19 - Phase 8.55L - Export UI Integration

### Agent
Codex

### Task
Expose the normalized export writer through the Import Wizard Qt GUI so users can save normalized CSV/Parquet/Feather datasets and metadata sidecars after a successful import.

### Completed
- Reviewed authoritative architecture, visualization/performance, workflow, handoff, task, and repository-state docs before editing.
- Added Save Normalized File controls to the Import Complete page: format selector, metadata sidecar checkbox, overwrite checkbox, save action, and export status summary.
- Added QFileDialog save flow with CSV default and ExportPlan-derived `{source}_normalized.csv` suggestions.
- Added `_ExportWorker` QRunnable so export writing uses the backend writer off the UI thread.
- Wired export to `plan_export()`, `write_from_export_plan()`, `write_normalized_export()`, `ExportWriteOptions`, and `ExportWriteResult`.
- Added export result summary display with path, rows, columns, format, metadata sidecar path, and validation messages.
- Kept waveform handoff independent; export does not replace or mutate the imported DisturbanceRecord lifecycle.
- Added unit and runtime tests for export enablement, default filenames, CSV success, sidecar creation, overwrite behavior, unsupported format/dependency errors, worker completion, graceful failure, repeatability, and waveform handoff after export.

### Files Modified
- app/ui/import_wizard/import_wizard_dialog.py
- app/ui/import_wizard/wizard_pages.py
- tests/unit/test_export_ui.py
- tests/runtime/test_export_ui_runtime.py
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No backend writer or visualization redesign. The GUI remains a thin orchestration layer and delegates export work to existing import_wizard export APIs. Export is an optional workflow after import completion and does not alter DisturbanceRecord rendering handoff.

### Performance Impact
Export runs through QRunnable/QThreadPool and does not block the UI thread. No additional normalized DataFrame copy is introduced in the GUI; writer-owned serialization behavior remains centralized in `export_writer.py`.

### Risks / Concerns
- Parquet/Feather remain dependent on optional backend dependencies; missing dependency errors are surfaced through ExportWriteResult.
- Export UI is intentionally lightweight and does not yet include advanced CSV formatting options, batch export, or export history.
- Existing stale Windows temp directories still produce git status warnings but do not block the verified pytest slices.

### Test Results
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_export_ui.py tests\\runtime\\test_export_ui_runtime.py -q` -> 16 passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_export_writer.py tests\\unit\\test_export_planning.py tests\\integration\\test_normalized_export_e2e.py tests\\unit\\test_export_ui.py tests\\runtime\\test_export_ui_runtime.py -q` -> 83 passed, 5 skipped.
- Broad Import Wizard/runtime slice with export UI -> 87 passed.
- `py_compile` passed for touched UI/test files.

### Next Recommended Step
Proceed to a small export UX hardening phase only if needed: optional timestamp/float formatting controls, clearer dependency guidance for Parquet/Feather, and save-location persistence. Avoid adding batch/report/cloud export until explicitly scoped.

---

## 2026-05-19 - Phase 8.55M - Real-World Import Hardening & Large Dataset Stress Testing

### Agent
Codex

### Task
Harden the Import Wizard against realistic operational CSV files and validate
large-dataset import/export/visualization handoff behavior without adding new
features.

### Completed
- Reviewed authoritative architecture, visualization/performance, workflow, handoff, task, and repository-state docs before editing.
- Added deterministic stress CSV generator with configurable row count, delimiter, timestamp format, sampling interval, metadata rows, malformed/missing/duplicate timestamp ratios, analog columns, digital/status columns, and unknown columns.
- Added practical benchmark tool for generation, profiling, total import, export, channel counts, validation codes, file sizes, and lightweight tracemalloc memory.
- Added stress tests for small/medium generated imports, export after import, metadata/header noise, malformed timestamps, duplicate/non-monotonic timestamps, mixed timestamp formats, delimiter variants, text digital states, unknown columns, and graceful unrecoverable timestamp failure.
- Added runtime tests for pending import worker responsiveness, failed import dialog stability, Open Waveform availability, Save Normalized File after import, metadata sidecar export, and dialog close behavior around workers.
- Created `docs/IMPORT_WIZARD_HARDENING_REPORT.md` with covered scenarios, measured benchmark results, known limits, and operational guidance.
- Fixed direct execution of `tools/benchmark_import_pipeline.py` by adding the repository root to `sys.path` when run from `tools/`.

### Files Modified
- tools/generate_import_stress_samples.py
- tools/benchmark_import_pipeline.py
- tests/stress/test_import_wizard_large_csv.py
- tests/stress/test_import_wizard_malformed_files.py
- tests/runtime/test_import_wizard_realistic_workflows.py
- docs/IMPORT_WIZARD_HARDENING_REPORT.md
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No production import, export, or visualization architecture was redesigned.
Stress generation and benchmarking live in `tools/`; tests exercise the existing
Import Wizard pipeline, export writer, DisturbanceRecord bridge, and Qt worker
orchestration. Visualization handoff remains limited to contract verification.

### Performance Impact
Default tests generate data in runtime temp directories and keep row counts modest
for repeatability. The generator streams rows to disk. Benchmarking uses lightweight
`time.perf_counter()` and `tracemalloc` only. No heavy dependencies were added.

Measured local benchmark results:
- 1,000 rows: profile 0.736 s, import 0.946 s, CSV export 0.177 s, peak traced memory 1.94 MiB.
- 25,000 rows: profile 0.674 s, import 5.767 s, CSV export 3.995 s, peak traced memory 18.40 MiB.

### Risks / Concerns
- Normal test coverage uses 25,000 rows for medium coverage; 100,000 and 1,000,000 row runs are supported by the tooling but should be explicit stress runs because runtime depends on local machine capacity.
- `tracemalloc` memory is Python allocation telemetry, not full process RSS.
- Mixed timestamp formats are not repaired into a single canonical format; rows that do not match the active detected/user format are dropped with diagnostics.
- Ragged CSV rows currently fail gracefully as `PIPELINE_LOAD_FAILED` instead of being partially repaired.
- Existing stale Windows temp directories still appear in `git status` warnings but do not block the verified runtime slices.

### Test Results
- `.venv\\Scripts\\python.exe -m pytest tests\\stress\\test_import_wizard_large_csv.py tests\\stress\\test_import_wizard_malformed_files.py tests\\runtime\\test_import_wizard_realistic_workflows.py -q` -> 19 passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\stress\\test_import_wizard_large_csv.py tests\\stress\\test_import_wizard_malformed_files.py tests\\runtime\\test_import_wizard_realistic_workflows.py tests\\runtime\\test_export_ui_runtime.py tests\\runtime\\test_runtime_environment.py -q` -> 30 passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_import_pipeline.py tests\\unit\\test_disturbance_record_bridge.py tests\\unit\\test_export_writer.py tests\\unit\\test_export_ui.py tests\\runtime\\test_timestamp_override_execution.py tests\\runtime\\test_import_wizard_authoritative_flow.py tests\\runtime\\test_import_wizard_realistic_workflows.py tests\\stress -q` -> 230 passed, 2 skipped.

### Next Recommended Step
Phase 8.55N should add compact user-facing import diagnostics: row-drop counts,
timestamp warning summaries, and large-file operational guidance surfaced in the
wizard without changing the pipeline contract.

---

## 2026-05-19 - Phase 8.55O - Import Wizard Final UX & Workflow Hardening

### Agent
Codex

### Task
Harden the Import Wizard workflow for operational engineering usability:
state safety, action gating, stale-state invalidation, override visibility,
discard protection, navigation stability, and concise user guidance.

### Completed
- Reviewed authoritative architecture, visualization/performance, workflow, handoff, task, and repository-state docs before editing.
- Added `app/ui/import_wizard/workflow_state.py`, a small UI workflow-state evaluator for button enablement and status guidance.
- Hardened ImportWizardDialog state invalidation:
  - timestamp selection changes invalidate plan/import/export state.
  - timestamp override changes invalidate plan/import/export state.
  - column mapping edits invalidate plan/import/export state.
  - new file selection resets previous profile, diagnostics, import result, export result, and models.
- Added explicit re-import-required guidance after settings change.
- Tightened action enablement for Next, Back, Run Import, Open Waveform, Save Normalized File, and Close.
- Added explicit Close action discard protection for user overrides, completed unexported imports, dirty settings, and active worker state.
- Kept window close deterministic for runtime teardown while preserving the in-wizard Close prompt.
- Improved override visibility:
  - timestamp page shows `User Override` when manual format is active.
  - column mapping model marks overridden output name/type/unit values with `(User Override)`.
  - confidence column and tooltips expose user overrides/excluded rows.
- Added concise workflow guidance to load, preview, timestamp, mapping, review, and export pages.
- Added `docs/IMPORT_WORKFLOW_GUIDE.md` describing workflow states, action enablement, override markers, stale invalidation, import/export flow, discard protection, and known limits.
- Added deterministic unit/runtime tests for action state, invalidation, reset behavior, discard prompt, override visibility, repeated import/export, worker completion after close, navigation stability, and graceful failed import state.

### Files Modified
- app/ui/import_wizard/workflow_state.py
- app/ui/import_wizard/__init__.py
- app/ui/import_wizard/column_mapping_model.py
- app/ui/import_wizard/import_wizard_dialog.py
- app/ui/import_wizard/wizard_pages.py
- tests/unit/test_import_workflow_ux.py
- tests/runtime/test_import_workflow_runtime.py
- docs/IMPORT_WORKFLOW_GUIDE.md
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No backend pipeline, diagnostics engine, export writer, or visualization architecture was redesigned. Workflow logic remains UI orchestration only and reuses existing ValidationMessage, diagnostics, plan-aware execution, export writer, and worker infrastructure.

### Performance Impact
Workflow state evaluation is constant-time and uses already-available UI/backend result state. No source files are re-read, no full datasets are reparsed, and no heavy validation was added to the UI thread. Existing QRunnable import/export execution remains intact.

### Risks / Concerns
- Discard protection is lightweight and not a persisted session manager.
- Close-window teardown accepts immediately for deterministic test/runtime cleanup; the in-wizard Close button provides the user-facing discard prompt.
- Override markers are text-based rather than icon-based.
- Existing stale Windows temp directories still appear in git status warnings but did not block verified test slices.

### Test Results
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_import_workflow_ux.py tests\\runtime\\test_import_workflow_runtime.py -q` -> 11 passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_import_workflow_ux.py tests\\runtime\\test_import_workflow_runtime.py tests\\unit\\test_import_wizard_gui.py tests\\unit\\test_export_ui.py tests\\unit\\test_timestamp_override_ui.py tests\\runtime\\test_timestamp_override_execution.py tests\\runtime\\test_export_ui_runtime.py tests\\runtime\\test_import_wizard_realistic_workflows.py tests\\unit\\test_import_diagnostics.py tests\\runtime\\test_import_diagnostics_runtime.py -q` -> 100 passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_import_pipeline.py tests\\unit\\test_plan_aware_pipeline.py tests\\unit\\test_disturbance_record_bridge.py tests\\unit\\test_export_writer.py tests\\runtime\\test_runtime_environment.py tests\\stress -q` -> 236 passed, 2 skipped.

### Next Recommended Step
Proceed to a final Import Wizard acceptance pass or Phase 8.55P focused on packaging/CI command documentation and standard verification slices. Avoid new features until the workflow is reviewed in a real interactive run.

---

## 2026-05-19 - Phase 8.55P - Acceptance Validation & Developer Operations

### Agent
Codex

### Task
Stabilize the Import Wizard subsystem for long-term development, repeatable
acceptance validation, regression protection, and developer operations.

### Completed
- Reviewed authoritative core architecture, visualization/performance, Import Wizard diagnostics/workflow/hardening, workflow-agent, handoff, task, and repository-state docs before editing.
- Added concise operational acceptance documentation for CSV/XLSX import, malformed timestamps, timestamp override, export, duplicate timestamps, unknown columns, large generated CSV behavior, repeated cycles, and worker-close stability.
- Added developer workflow documentation with standard validation slices, benchmark commands, stress sample generation, troubleshooting, and merge guidance.
- Added an Import Wizard test matrix mapping feature areas to unit, runtime, stress, and acceptance coverage.
- Added lightweight runner scripts:
  - `tools/run_import_acceptance.py` for `unit`, `runtime`, `stress`, `acceptance`, and `import-full` slices.
  - `tools/run_import_runtime_slice.py` for repeated runtime validation passes.
- Added acceptance tests covering CSV import to waveform handoff, XLSX import to normalized CSV export with sidecar, authoritative timestamp override, malformed diagnostics, repeated import/export, and worker completion after close.
- Added `tests/acceptance/conftest.py` with Qt cleanup matching runtime hygiene expectations.

### Files Modified
- docs/IMPORT_ACCEPTANCE_CHECKLIST.md
- docs/IMPORT_DEV_WORKFLOW.md
- docs/IMPORT_TEST_MATRIX.md
- tools/run_import_acceptance.py
- tools/run_import_runtime_slice.py
- tests/acceptance/conftest.py
- tests/acceptance/test_import_acceptance.py
- agent/HANDOFF.md
- agent/TASK.md
- agent/REPOSITORY_STATE.md

### Architecture Impact
No product runtime, import pipeline, export writer, diagnostics engine, or
visualization architecture was changed. This phase adds developer operations,
documentation, and acceptance tests around existing Import Wizard contracts.

### Performance Impact
Validation tooling is subprocess-based and opt-in. Default acceptance tests use
small deterministic runtime-temp files. Large-file checks remain explicit via the
existing stress generator and benchmark tool.

### Risks / Concerns
- Runner pass-through pytest arguments must be supplied as `--pytest-arg=-q` for options beginning with `-`.
- Acceptance tests use small deterministic files and complement, rather than replace, stress and benchmark runs.
- Optional Parquet/Feather coverage remains dependency-gated in existing export tests; acceptance uses CSV as the baseline.

### Test Results
- `.venv\\Scripts\\python.exe -m py_compile tools\\run_import_acceptance.py tools\\run_import_runtime_slice.py tests\\acceptance\\test_import_acceptance.py tests\\acceptance\\conftest.py` -> passed.
- `.venv\\Scripts\\python.exe -m pytest tests\\acceptance\\test_import_acceptance.py -q` -> 6 passed.
- `.venv\\Scripts\\python.exe tools\\run_import_acceptance.py --slice acceptance --pytest-arg=-q` -> 6 passed.
- `.venv\\Scripts\\python.exe tools\\run_import_runtime_slice.py --repeat 1 --pytest-arg=-q` -> 49 passed.
- `.venv\\Scripts\\python.exe tools\\run_import_acceptance.py --slice import-full --pytest-arg=-q` -> 387 passed, 2 skipped.

### Next Recommended Step
Use the new acceptance/dev workflow as the standard gate for Import Wizard
maintenance. Next phase can move toward Phase 9 planning or CI wiring around
these documented slices; avoid new Import Wizard features until the operational
acceptance workflow is reviewed interactively.
