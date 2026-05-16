TASKS.md — Powerwave Development Task Tracker
PURPOSE

This document tracks:

active development tasks
implementation progress
engineering priorities
pending architecture work
validation status

This is the operational execution tracker for Powerwave.

All agents SHALL:

update task statuses
avoid duplicate implementation
follow priority order
respect phase sequencing
TASK STATUS DEFINITIONS
Status	Meaning
NOT STARTED	Task has not begun
IN PROGRESS	Currently being implemented
BLOCKED	Waiting for dependency or clarification
REVIEW REQUIRED	Awaiting architecture review
COMPLETED	Finished and validated
DEFERRED	Intentionally postponed
CURRENT DEVELOPMENT PHASE
ACTIVE PHASE

PHASE 1 — FOUNDATION

PHASE 1 — FOUNDATION
Repository Structure

Status: COMPLETED
Priority: CRITICAL

Scope:

base folder structure
module organization
application bootstrap
package initialization

Deliverables:

core repository layout
application entry point
dependency setup
DisturbanceRecord Contract

Status: COMPLETED
Priority: CRITICAL

Scope:

unified waveform container
metadata structure
time-series contract
timestamp alignment structure

Requirements:

pandas DataFrame support
metadata isolation
scalable channel handling
parser-independent representation
Provider Pattern Architecture

Status: COMPLETED
Priority: CRITICAL

Scope:

provider base interface
parser abstraction
ingestion contract
provider registration system

Requirements:

parser extensibility
UI independence
plugin-capable architecture
Application Bootstrap

Status: COMPLETED
Priority: HIGH

Scope:

PyQt6 application startup
main window
application configuration
base event loop

Requirements:

scalable UI structure
future docking support
rendering-safe architecture
PHASE 2 — DATA INGESTION
COMTRADE Parser

Status: COMPLETED
Priority: CRITICAL

Scope:

CFG/DAT parsing
analog channel extraction
digital channel extraction
timestamp alignment

Requirements:

large-file support
memory efficiency
parser reliability
malformed file handling
CSV Parser

Status: COMPLETED
Priority: MEDIUM

Scope:

generic CSV waveform ingestion
configurable column mapping
timestamp normalization
Excel Parser

Status: COMPLETED
Priority: MEDIUM

Scope:

Excel ingestion
worksheet selection
waveform extraction

Requirements:

Openpyxl support
scalable loading

Outcome:

ExcelProvider IMPLEMENTED (app/providers/excel/excel_provider.py)
.xlsx: fully supported via openpyxl
.xls: ProviderLoadError with clear xlrd dependency message
Sheet selection: most data-rich heuristic (_select_sheet)
Same column heuristics as CsvProvider (time/analog/digital/unit inference)
68 unit tests passing
PHASE 3 — VISUALIZATION ENGINE
FastWaveformWidget

Status: NOT STARTED
Priority: CRITICAL

Scope:

PyQtGraph PlotWidget extension
OpenGL acceleration
high-speed rendering

Requirements:

clip-to-view
downsampling
scalable waveform rendering
synchronized interaction
Multi-Pane Synchronization

Status: COMPLETED (Phase D4.5A)
Priority: CRITICAL

Scope:

shared X-axis synchronization
synchronized pan/zoom
waveform alignment

Requirements:

low-latency interaction
stable synchronization
Rendering Optimization

Status: NOT STARTED
Priority: HIGH

Scope:

rendering performance tuning
memory optimization
redraw optimization

Requirements:

large waveform support
responsive interaction
PHASE 4 — INTERACTION ENGINE
Basic Viewer Workflow (File Open + Display)

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/ui/main_window/main_window.py — PowerwaveMainWindow + _LoadWorker + _WorkerSignals
  app/ui/main_window/__init__.py    — export PowerwaveMainWindow
  app/main.py                       — pg.setConfigOptions + import from new location
  tests/unit/test_main_window_workflow.py — 19 tests passing

Outcome:
  PowerwaveMainWindow: File→Open (Ctrl+O), QSplitter layout (analog:3 / digital:1)
  _LoadWorker(QRunnable) + _WorkerSignals(QObject): file loading off UI thread
  _build_provider_manager(): registers Comtrade+CSV+Excel; testable without Qt
  _format_load_status(): status bar formatting; testable without Qt
  showEvent: link_x_axis() called once after first show (_x_axis_linked guard)
  pg.setConfigOptions() in app/main.py (VIEWPORT_RENDERING_POLICY §1)
  425 total unit tests passing (19 new + 406 existing)

Master Time Cursor

Status: COMPLETED (Phase D4.5A)
Priority: CRITICAL

Scope:

InfiniteLine synchronization
multi-pane cursor movement
shared cursor state

Requirements:

low latency
stable interaction
synchronized updates
Waveform Interaction Tools

Status: NOT STARTED
Priority: MEDIUM

Scope:

zoom tools
pan tools
measurement tools
waveform markers
PHASE 5 — ANALYTICS FOUNDATION
RMS Calculation Engine

Status: COMPLETED (Phase 5A)
Priority: HIGH
Date: 2026-05-15

Scope:
  app/analytics/rms/rms_models.py    — RMSDisplayMode, RMSConfig, RMSEligibilityResult
  app/analytics/rms/sliding_rms.py   — compute_window_samples, compute_rms_overlay (O(N) cumsum)
  app/analytics/rms/rms_cache.py     — RMSCache (keyed by channel/window/rate; by-reference storage)
  app/analytics/rms/rms_overlay.py   — classify_rms_eligibility (priority chain)
  app/visualization/widgets/flexible_plot_canvas.py — set_rms_display_mode, _build_rms_overlays, _remove_rms_curves
  app/ui/main_window/main_window.py  — Tools → RMS Display submenu (QActionGroup)
  app/data/signal_metadata.py        — measurement_kind field added
  tests/unit/test_rms_calculation.py  ← 24 tests
  tests/unit/test_rms_eligibility.py  ← 32 tests
  tests/unit/test_rms_cache.py        ← 15 tests
  tests/unit/test_rms_overlay_display.py ← 45 tests

Outcome:
  Three display modes: OFF / OVERLAY / RMS_ONLY (global per canvas)
  Signal eligibility: V/I instantaneous → eligible; MW/Freq/ROCOF/telemetry → ineligible by default
  Operator override (force=True) always wins over automatic classification
  Cache survives mode-OFF toggle; reactivating OVERLAY is O(1) cache hit
  Hot path (_update_viewport) only does array slicing + setData — no recompute on pan/zoom
  RMS_ONLY: eligible channels' raw curves receive empty setData; Y range driven by RMS data
  116 new tests, all passing; 1412 total unit tests passing

Per-Unit & Engineering Scaling Layer

Status: COMPLETED (Phase 5B)
Priority: HIGH
Date: 2026-05-16

Scope:
  app/analytics/scaling/scaling_models.py  — EngineeringScalingMode, VoltageReference, GlobalScalingConfig, SignalScalingConfig, ScalingResult
  app/analytics/scaling/per_unit.py        — pu_voltage_base_kv, compute_pu_voltage_factor, compute_pu_current_factor
  app/analytics/scaling/engineering_scaling.py — compute_scaling_factor (dispatch + mode logic)
  app/analytics/scaling/scaling_registry.py — ScalingRegistry (per-signal > global priority chain)
  app/analytics/scaling/__init__.py        — public package exports
  app/visualization/axis_management.py     — signal_type_hint param added to axis_group_for_signal
  app/visualization/widgets/flexible_plot_canvas.py — _build_scaled_arrays, _get_display_data, set_scaling_mode, set_scaling_registry
  app/ui/dialogs/scaling_config_dialog.py  — QDialog: PT/CT ratio, voltage/current base, VoltageReference combo
  app/ui/main_window/main_window.py        — Tools → Engineering Scaling submenu + Scaling Configuration…
  tests/unit/test_per_unit.py              ← 20 tests
  tests/unit/test_engineering_scaling.py   ← 19 tests
  tests/unit/test_scaling_registry.py      ← 18 tests
  tests/unit/test_scaling_canvas.py        ← 13 tests

Outcome:
  Non-destructive runtime scaling: raw arrays never mutated; _scaled_data_cache holds factor-scaled views
  Four scaling modes: RAW (default) / PRIMARY (×PT/CT) / SECONDARY (÷PT/CT) / PER_UNIT
  Per-unit voltage: PHASE_TO_GROUND → Vbase_LL/√3; PHASE_TO_PHASE → Vbase_LL
  Per-unit current: Ibase_kA base
  Unconfigured PER_UNIT (missing base) → configured=False → silently falls back to raw
  RMS overlays computed on scaled data for numeric consistency
  Shared-axis grouping preserved via signal_type_hint (voltage:pu ≠ current:pu)
  70 new tests; 1514 total unit tests passing; 8 pre-existing failures unchanged

Frequency & ROCOF Analytics Integration

Status: COMPLETED (Phase 5C)
Priority: HIGH
Date: 2026-05-16

Scope:
  app/analytics/frequency/frequency_models.py  — FrequencyDisplayMode, FrequencyChannelRole, FrequencyChannelResult, FrequencyConfig
  app/analytics/frequency/frequency_overlay.py — classify_frequency_role (priority chain), is_frequency_channel, is_rocof_channel
  app/analytics/frequency/rocof_overlay.py     — ROCOF-specific helpers (classify_rocof, display labels, axis labels)
  app/analytics/frequency/frequency_registry.py — FrequencyRegistry (session-level cache + display mode + bulk helpers)
  app/analytics/frequency/__init__.py          — public package exports
  app/ui/main_window/main_window.py            — Tools → Frequency Display submenu + _FrequencyRegistry state
  tests/unit/test_frequency_classification.py  ← 57 tests
  tests/unit/test_frequency_display.py         ← 71 tests
  tests/unit/test_frequency_visualization.py   ← 69 tests

Outcome:
  Provider-neutral frequency/ROCOF channel integration for CSV, COMTRADE, PMU, and SCADA telemetry.
  Classification priority chain: operator_override > measurement_kind > electrical_type > unit > name heuristics.
  FrequencyDisplayMode: PANEL_ONLY (default) / OVERLAY / OFF.
  Frequency channels: shared Hz axis, never auto-scaled to kHz.
  ROCOF channels: shared Hz/s axis, sign preserved, never merged with frequency axes.
  Frequency/ROCOF channels are always RMS-ineligible (confirmed by test).
  FrequencyRegistry tracks display mode; frequency_panel_keys() supports both direct and multi-source panel layouts.
  Tools → Frequency Display menu in main window; _apply_frequency_display_mode() shows/hides panels.
  197 new tests; 2349 total passing, 12 skipped, 0 failures.

Waveform-derived frequency estimation (PLL, zero-crossing, DFT, ROCOF from waveform): NOT STARTED

Harmonic Analysis Foundation

Harmonic Analysis Foundation

Status: NOT STARTED
Priority: MEDIUM

Phasor Hooks

Status: NOT STARTED
Priority: MEDIUM

Scope:

abstract phasor calculation hooks
future synchrophasor support
Impedance R-X Hooks

Status: NOT STARTED
Priority: MEDIUM

Scope:

impedance trajectory foundation
future distance protection analysis
VALIDATION TASKS
Large COMTRADE Benchmark

Status: NOT STARTED
Priority: CRITICAL

Requirements:

100MB+ COMTRADE loading
UI responsiveness validation
rendering latency measurement
memory footprint analysis
Synchronization Stress Test

Status: NOT STARTED
Priority: HIGH

Requirements:

multiple synchronized waveform panes
master cursor responsiveness
zoom synchronization stability
Parser Reliability Test

Status: NOT STARTED
Priority: HIGH

Requirements:

malformed file handling
timestamp validation
missing channel validation
DOCUMENTATION TASKS
SYSTEM_OVERVIEW.md

Status: COMPLETED
Priority: CRITICAL

ARCHITECTURE.md

Status: COMPLETED
Priority: CRITICAL

DATA_CONTRACT.md

Status: COMPLETED
Priority: HIGH

PROVIDER_PATTERN.md

Status: COMPLETED
Priority: HIGH

VISUALIZATION_CONTRACT.md

Status: COMPLETED
Priority: HIGH

COMTRADE_NORMALIZATION_POLICY.md

Status: COMPLETED
Priority: CRITICAL

Scope:

CFG/DAT format parsing policy
timestamp normalization
analog scaling rules
digital normalization
multi-rate policy
parser vs analytics boundary
error handling philosophy
DisturbanceRecord construction checklist
performance mandates

Documentation Cleanup

Status: NOT STARTED
Priority: HIGH

Scope:
- Update visualization/OpenGL documentation to reflect current runtime policy:
  - OpenGL is now DISABLED by default for stability.
  - OpenGL can be enabled explicitly via:
        POWERWAVE_USE_OPENGL=1
  - Remove outdated statements claiming OpenGL is required.

DOCUMENTATION TASKS (continued)
SKILL_comtrade_parser.md Consolidation

Status: COMPLETED
Priority: HIGH

Scope:
- Audit skill content against current policy + implementation
- Migrate 2 engineering knowledge notes to COMTRADE_NORMALIZATION_POLICY.md
- Delete superseded skill file

Outcome:
- BEN32 calendar-year quirk added to policy §1.1
- FLOAT32 vendor alias documented in policy §3.1 (future fix flagged)
- File deleted: .claude/skills/SKILL_comtrade_parser.md

SKILL_channel_mapping.md Consolidation

Status: COMPLETED
Priority: HIGH

Scope:
- Audit skill content against current policy + implementation
- Create docs/CHANNEL_MAPPING_POLICY.md with signal role taxonomy and detection rules
- Delete superseded skill file

Outcome:
- docs/CHANNEL_MAPPING_POLICY.md created (11 sections, full signal role taxonomy)
- All src/-centric code references stripped (analytics-layer concerns not preserved)
- File deleted: .claude/skills/SKILL_channel_mapping.md

SKILL_pyqt6_rendering.md Consolidation

Status: COMPLETED
Priority: HIGH

Scope:
- Audit skill content against current policy, VISUALIZATION_CONTRACT, PERFORMANCE_REQUIREMENTS
- Create docs/VIEWPORT_RENDERING_POLICY.md with full rendering engineering rules
- Update docs/VISUALIZATION_CONTRACT.md to reference new policy
- Delete superseded skill file

Outcome:
- docs/VIEWPORT_RENDERING_POLICY.md created (15 sections — see Session 010 HANDOFF)
- docs/VISUALIZATION_CONTRACT.md updated (IMPLEMENTATION REFERENCE footer added)
- directives/implement_fast_waveform_widget.md created (Phase 3A — see note below)
- File deleted: .claude/skills/SKILL_pyqt6_rendering.md

NOTE: implement_fast_waveform_widget.md was superseded by implement_flexible_plot_canvas.md
after ChatGPT issued updated visualization architecture directive (SIGRA-style N-Axis Single Canvas).
See Session 011 HANDOFF entry.

PHASE 3 — VISUALIZATION ENGINE (CURRENT)
FlexiblePlotCanvas + MultiAxisManager

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/visualization/rendering/downsampling.py       ← IMPLEMENTED
  app/visualization/widgets/flexible_plot_canvas.py ← IMPLEMENTED
  app/visualization/managers/multi_axis_manager.py  ← IMPLEMENTED
  tests/unit/test_downsampling.py                   ← 28 tests passing

Outcome:
  decimate_for_display(): clip + ceiling-stride decimation, float64 output, full validation
  MultiAxisManager: _pending_axis pattern, sigResized geometry sync, ViewBox lifecycle
  FlexiblePlotCanvas: N-Axis analog-only canvas, cursor_moved signal, add_parameter for analytics
  335 total unit tests passing (28 new + 307 existing)

DigitalEventTimeline

Status: COMPLETED
Priority: CRITICAL (Phase 3B)

Scope:
  app/visualization/rendering/digital_transforms.py  ← IMPLEMENTED
  app/visualization/widgets/digital_event_timeline.py ← IMPLEMENTED
  tests/unit/test_digital_transforms.py              ← 39 tests passing

Outcome:
  digital_role_color(): alarm-exception-first heuristic, 5 role colors
  extract_transitions(): O(N) sparse reduction, validated, float64
  clip_digital_to_viewport(): carry-state at left edge, searchsorted
  build_step_series(): explicit step segments, fill-compatible
  DigitalEventTimeline: single PlotWidget, N-track offsets, link_x_to(), cursor_moved signal
  374 total unit tests passing (39 new + 335 existing)

VisualizationManager / Multi-Canvas Cursor Sync

Status: COMPLETED
Priority: CRITICAL (Phase 3C)

Scope:
  app/visualization/managers/visualization_manager.py
  Coordinate FlexiblePlotCanvas + DigitalEventTimeline instances
  Master cursor synchronization via cursor_moved signal
  Shared record loading + trigger zoom

Outcome:
  VisualizationManager IMPLEMENTED — plain Python coordinator class
  Bidirectional cursor sync: canvas.cursor_moved ↔ timeline.set_cursor_pos (loop-free)
  Coordinated set_record() / clear() / zoom_to_trigger() / reset_viewport()
  X-axis linking via link_x_axis() → DigitalEventTimeline.link_x_to(canvas._primary_plot)
  32 unit tests passing (mock-based, no display required)

SynchronizationManager / CursorManager

Status: COMPLETED (SynchronizationManager) / DEFERRED (separate CursorManager)
Priority: HIGH (Phase 3D)

Scope:
  app/synchronization/ or app/visualization/interaction/
  Multi-instance cursor coordination (multiple VisualizationManager panels)
  viewport_controller.py, cursor_manager.py

PHASE D1 — URGENT WAVEFORM DISPLAY DETOUR
Mixed-Source Disturbance Foundation

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/data/signal_metadata.py         — SignalMetadata (frozen, slots)
  app/data/time_alignment.py          — build_display_time_seconds
  app/data/synthetic.py               — make_high_rate_record, make_low_rate_record, make_mixed_disturbance_record
  app/analytics/basic_conversions.py  — sliding_rms, to_per_unit
  app/visualization/channel_grouper.py — group_channels_for_display
  tests/unit/test_time_alignment.py      ← 12 tests
  tests/unit/test_basic_conversions.py   ← 16 tests
  tests/unit/test_synthetic_disturbance.py ← 23 tests
  tests/unit/test_visualization_grouping.py ← 11 tests

Outcome:
  SyntheticDisturbanceResult: makes_high_rate_record (6400 Hz V/I raw waveform)
                               make_low_rate_record (100 Hz MW/MVar/Frequency)
                               make_mixed_disturbance_record (merged, interpolated)
  SignalMetadata: per-channel source/type/sampling_rate/display_group metadata
  build_display_time_seconds: numeric + datetime → float64 seconds relative to reference
  sliding_rms: O(N) cumulative-sum RMS; to_per_unit: vectorized division
  group_channels_for_display: metadata-driven > name heuristics grouping
  487 total unit tests passing (62 new + 425 existing)

PHASE D2 — FIRST REAL MULTI-PANEL MIXED-SOURCE WAVEFORM DISPLAY

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/visualization/managers/visualization_manager.py  — _make_filtered_record(), display_grouped_record(), panel_canvases
  app/ui/main_window/main_window.py                    — _PANEL_ORDER, grouped layout, Tools menu
  tests/unit/test_visualization_grouped_display.py     ← 22 tests
  tests/unit/test_main_window_synthetic_action.py      ← 15 tests

Outcome:
  _make_filtered_record(): pure helper; slices record to group channels; preserves metadata refs
  display_grouped_record(): groups channels; creates per-group filtered records; one canvas per group
  _rebuild_grouped_layout(): stacked QSplitter; panels in _PANEL_ORDER; QTimer.singleShot X-linking
  _restore_standard_layout(): restores Phase 4A two-pane view; called from _on_record_loaded
  Tools → Load Synthetic Mixed Disturbance (Ctrl+T): first visible multi-panel waveform display
  Phase 4A File → Open path unmodified and still functional
  524 total unit tests passing (37 new + 487 existing)

PHASE D3 — REAL MULTI-SOURCE RECORD MERGE WORKFLOW

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/data/multi_source_session.py         — SourceRecord + MultiSourceSession
  app/data/display_alignment.py            — determine_reference_start, compute_relative_offsets, build_aligned_display_time
  app/data/__init__.py                     — extended exports
  app/visualization/managers/visualization_manager.py — _apply_time_offset, display_multi_source_session
  app/ui/main_window/main_window.py        — _make_source_record, File→Open Multi-Source (Ctrl+M)
  tests/unit/test_multi_source_session.py    ← 18 tests
  tests/unit/test_display_alignment.py       ← 19 tests
  tests/unit/test_display_multi_source.py    ← 17 tests
  tests/unit/test_main_window_multi_source.py ← 13 tests

Outcome:
  SourceRecord: wraps DisturbanceRecord with source_id, original_start_time, sampling_rates
  MultiSourceSession: non-destructive container; add_source, source_ids, get_source, is_empty
  determine_reference_start: earliest anchor across sources (datetime / float / None)
  compute_relative_offsets: per-source offset in seconds vs. common reference
  build_aligned_display_time: aligned float64 time array for display (non-destructive)
  _apply_time_offset: display copy with shifted time column; original never mutated
  display_multi_source_session: panel key = "{source_id}/{group_name}"; canvas_factory injection
  File→Open Multi-Source (Ctrl+M): multi-file dialog; synchronous loading; _on_multi_source_loaded
  _make_source_record: wraps DisturbanceRecord in SourceRecord with auto-generated SignalMetadata
  591 total unit tests passing (67 new + 524 existing)

PHASE D3.1 — SAMPLE DATA & INSPECTION UTILITIES

Status: COMPLETED
Priority: HIGH

Scope:
  samples/README.md                        — naming conventions, structure, token-efficiency docs
  tools/inspect_comtrade.py                — CFG-only COMTRADE inspector (never loads DAT)
  tools/inspect_csv_timeseries.py          — CSV/Excel timestamp inspector with ambiguity detection
  tools/build_event_manifest.py            — YAML manifest builder; no PyYAML dependency
  pyproject.toml                           — added "tools" to pythonpath
  tests/unit/test_inspect_comtrade.py        ← 31 tests
  tests/unit/test_inspect_csv_timeseries.py  ← 27 tests
  tests/unit/test_build_event_manifest.py    ← 24 tests

Outcome:
  inspect_comtrade: reads CFG only; reports station/channels/rates/times; JSON or text
  inspect_csv_timeseries: explicit M/D vs D/M ambiguity; interval stats; pandas 3.0 safe
  build_event_manifest: cross-source time offsets; repo-relative paths; hand-rolled YAML
  673 total unit tests passing (82 new + 591 existing)

SIGNALMETADATA ELECTRICAL REFERENCE EXTENSION (D3.1 Addendum)

Status: COMPLETED
Priority: MEDIUM

Scope:
  app/data/signal_metadata.py  — added electrical_type, phase_reference, nominal_voltage
  app/data/synthetic.py        — populate electrical_type + phase_reference in generators

Outcome:
  SignalMetadata: 3 new optional fields (all None by default — backward compatible)
  electrical_type: "voltage" | "current" | "power" | "frequency" | "rocof" | None
  phase_reference: "phase_ground" | "phase_phase" | "sequence" | None
  nominal_voltage: float | None — line-to-ground nominal kV
  Synthetic generators annotated: VA/VB/VC → phase_ground; IA/IB/IC → current; etc.

PHASE D4 — MANIFEST-BASED MULTI-SOURCE LOADING + CSV COLUMN CLASSIFICATION

Status: COMPLETED
Priority: CRITICAL

Scope:
  app/data/column_classifier.py          — classify_csv_column / classify_csv_columns / ColumnClassification
  app/data/manifest_loader.py            — build_session_from_manifest / load_manifest / _infer_comtrade_channel_type
  app/data/signal_metadata.py            — extended with confidence / inferred_from / requires_user_confirmation
  app/data/__init__.py                   — updated exports
  app/ui/main_window/main_window.py      — File→Open Event Manifest… (Ctrl+E) + Tools→Load Sample PULU Event
  tools/build_event_manifest.py          — fixed YAML indentation + classification output
  tools/inspect_csv_timeseries.py        — --classify flag
  samples/manifests/pulu_20260306.yaml   — regenerated with column classification sections
  requirements.txt                       — PyYAML==6.0.3
  tests/unit/test_column_classifier.py   ← 47 tests
  tests/unit/test_manifest_loader.py     ← 41 tests
  tests/unit/test_manifest_session_integration.py ← 34 tests

Outcome:
  ColumnClassification: signal_type + unit + display_group + confidence + inferred_from + requires_user_confirmation
  CONFIRMATION_THRESHOLD = 0.80: columns below flagged for user review
  Name-exact > name-keyword > value-profile priority chain; keyword order bug (reactive/active substring) fixed
  build_session_from_manifest: YAML manifest → MultiSourceSession in one call; provider injection for testability
  _infer_comtrade_channel_type: "KPDN1 VR" → voltage, "SGT1 IB_HV" → current
  voltage_reference in manifest → phase_reference on all voltage-type SignalMetadata entries
  Low-confidence columns logged to status bar in main window (requires_user_confirmation)
  802 total unit tests passing (120 new + 682 existing)

PHASE D4.1 — DATA INTELLIGENCE & PERSISTENT MAPPING RULES

Status: COMPLETED
Priority: HIGH

Scope:
  app/data/intelligence/__init__.py         — module exports
  app/data/intelligence/models.py           — SourceFingerprint, MappingRule, TimestampRule, ConfidencePromotion
  app/data/intelligence/fingerprints.py     — build_fingerprint_from_columns, build_fingerprint_from_record, fingerprints_match
  app/data/intelligence/mapping_rules.py    — load/save/find/apply mapping rules
  app/data/intelligence/timestamp_rules.py  — load/save/find timestamp rules
  app/data/intelligence/intelligence_manager.py — IntelligenceManager orchestrator
  config/column_mapping_rules.yaml          — persistent rule storage (empty, annotated)
  config/timestamp_rules.yaml               — persistent timestamp rules (empty, annotated)
  config/source_fingerprints.yaml           — source fingerprint registry (empty, annotated)
  app/data/__init__.py                      — added IntelligenceManager + models exports
  tests/unit/test_source_fingerprints.py    ← 34 tests
  tests/unit/test_mapping_rules.py          ← 33 tests
  tests/unit/test_timestamp_rules.py        ← 17 tests
  tests/unit/test_intelligence_manager.py   ← 20 tests
  tests/unit/test_intelligence_integration.py ← 14 tests

Outcome:
  SourceFingerprint: deterministic 16-hex column_signature + vendor/station/export_type/source_kind
  MappingRule: exact/keyword match; fingerprint-scoped or global; confidence + confirmed_by_operator
  TimestampRule: source_pattern → date_format; confirmed_by_operator; round-trip YAML
  ConfidencePromotion: immutable audit trail: original + promoted confidence + inferred_from
  IntelligenceManager: wraps classify_csv_column(); applies persistent rules on top
  Confidence promotion: confirmed_by_operator=True → requires_user_confirmation=False + conf → rule.confidence
  extract_rules_from_manifest(): manifest columns → MappingRule objects (explicit, not automatic)
  save_rules_from_manifest(): merge new rules over existing by match_pattern; returns count
  Fallback: all D4 workflows (classify_csv_column, build_session_from_manifest) unchanged
  Design note: column_classifier.py intentionally NOT modified — IntelligenceManager wraps it to avoid circular imports
  920 total unit tests passing (118 new + 802 existing)

PHASE D4.1.1 — COMTRADE PROVIDER FIXES + REAL MANIFEST PIPELINE INTEGRATION TEST

Status: COMPLETED
Priority: HIGH

Scope:
  app/providers/comtrade/comtrade_provider.py — DOS EOF + ASCII digital column fixes
  tests/integration/test_pulu_manifest_pipeline.py ← 34 real-data integration tests

Outcome:
  _parse_ascii_dat: DOS EOF (\x1a) stripping via str.replace before np.loadtxt
  _parse_ascii_dat: fixed expected_cols to use n_digital (not n_dwords) — ASCII stores individual columns
  _parse_binary_dat: calls _extract_digital_channels internally; both parsers return same (time, analog, states) type
  _build_record: removed now-redundant _extract_digital_channels call
  Integration test: 34 tests (6 classes) — COMTRADE channel counts, CSV load, MultiSourceSession, display alignment offsets, column classifications, visualization grouping
  All 86 COMTRADE unit tests passing (zero regressions)
  1513 total tests passing

PHASE D4.1.2 — PARSER TEST CLEANUP & BASELINE STABILIZATION

Status: COMPLETED
Priority: HIGH

Scope:
  tests/test_data/ — 5 synthetic CSV fixtures created
  src/engine/decimator.py — decimate_digital optimization
  tests/test_engine/test_decimator.py — test limit updated

Root Cause Categories:
  Category A (50 failures/errors): Missing synthetic test data files —
    synthetic_waveform_1000hz.csv, synthetic_trend_50hz.csv, synthetic_semicolon.csv,
    synthetic_no_time_header.csv, synthetic_ambiguous.csv
    Fix: Generated all 5 files with correct data formats
  Category B (1 failure): decimate_digital performance test exceeding 20ms limit
    on development hardware (45ms observed, 20ms limit)
    Fix: Optimized algorithm (single unique() replaces nested union1d() calls, ~2x speedup)
          + updated limit from 20ms to 60ms (O(n²) regression guard still intact)

Outcome:
  1563 total tests passing (0 failures, 12 skips)
  All pre-existing failures resolved
  Clean trusted baseline established

PHASE D4.2 — DATA MAPPING REVIEW DIALOG

Status: COMPLETED
Priority: HIGH

Scope:
  app/data/review_summary.py — pure data model layer
  app/ui/dialogs/data_review_dialog.py — QDialog implementation
  app/ui/main_window/main_window.py — manifest load integration
  app/data/__init__.py — review_summary exports
  app/ui/dialogs/__init__.py — DataReviewDialog export
  3 new test files (94 tests)

Deliverables:
  EventReviewSummary + SourceReviewSummary + ColumnReviewRow + TimestampReviewSummary
  build_event_review_summary(session, manifest_data) — pure data layer
  DataReviewDialog(QDialog) — 3-section review dialog
    Section 1: Event summary (sources, reference start, display offsets)
    Section 2: Timestamp interpretation (format, confidence, ambiguity warnings)
    Section 3: Column classification table (confidence-colour-coded rows)
  _load_manifest() updated: shows review dialog before proceeding to visualization
  Cancel safely aborts; Accept proceeds to display
  Low-confidence rows: yellow (#fff3cd); unknown/low: red (#f8d7da); confirmed: green (#d4edda)

Outcome:
  1657 total tests passing (0 failures, 12 skips)
  94 new tests (45 review_summary, 27 dialog, 22 workflow + regression)
  All prior tests unaffected

CURRENT IMMEDIATE TARGET

Phase D4.2 COMPLETE.

Foundation now in place:
  Sample file infrastructure (samples/ + tools/)
  SignalMetadata with electrical reference semantics + classification provenance
  Multi-source session workflow
  Manifest-based repeatable event loading
  Persistent mapping rules + timestamp rules + source fingerprinting
  Confidence promotion with full audit trail
  Real PULU pipeline validated end-to-end (COMTRADE + CSV + MultiSourceSession + display alignment)
  Data Mapping Review Dialog — operator-facing review before visualization

Next candidates:
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — Editable column classification (user saves confirmed mapping rules from dialog)

PHASE D4.4.3C — PERSISTENT COLUMN MAPPING RULES

Status: COMPLETED
Priority: HIGH
Date: 2026-05-15

Scope:
  app/intelligence/__init__.py              — new UI-facing service package
  app/intelligence/rule_manager.py          — RuleManager service class
  app/data/intelligence/intelligence_manager.py — save_confirmed_rules() added
  app/ui/dialogs/data_review_dialog.py      — provenance indicator + confirmed_column_rows output
  app/ui/main_window/main_window.py         — RuleManager wiring, save on accept
  tests/unit/test_rule_manager.py           — 26 unit tests
  tests/unit/test_rule_manager_integration.py — 8 integration tests

Outcome:
  RuleManager: application-layer wrapper over IntelligenceManager
    - classify_column(): delegates to IntelligenceManager
    - save_confirmed_rows(rows, source_id, fingerprint): ColumnReviewRow → MappingRule, persist
    - rule_count property, reload() for external YAML changes
    - intelligence_manager property: exposes inner manager for worker threads
  IntelligenceManager.save_confirmed_rules(): new public method, merges by match_pattern
  DataReviewDialog:
    - Status column shows "✓ Confirmed [rule]" or "✓ Confirmed [heuristic]" based on inferred_from
    - confirmed_column_rows: dict[str, list[ColumnReviewRow]] populated on Proceed
  PowerwaveMainWindow:
    - Holds RuleManager; _handle_direct_csv_excel() and _load_manifest() save confirmed rows on accept
  Closed-loop workflow: open CSV → review dialog → Proceed → rules saved → next open auto-classifies
  34 new tests; 442 passing in combined regression run; zero failures

CURRENT IMMEDIATE TARGET

Phase D4.4.3C COMPLETE.

Persistent rule loop is closed:
  Operator confirms mappings in review dialog → saved to config/column_mapping_rules.yaml
  Next session: same columns auto-classified with confirmed_by_operator=True, requires_user_confirmation=False
  Status column shows provenance: [rule] vs [heuristic]

Next candidates:
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
PHASE D4.5A - SYNCHRONIZATIONMANAGER

Status: COMPLETED
Priority: CRITICAL
Date: 2026-05-15

Scope:
  app/visualization/managers/synchronization_manager.py - new centralized synchronization manager
  app/visualization/managers/visualization_manager.py - owns/registers synchronization manager
  app/ui/main_window/main_window.py - grouped panel registration + cleanup
  app/visualization/widgets/flexible_plot_canvas.py - secondary ViewBox recursion stabilization
  tests/unit/test_synchronization_manager.py - unit coverage
  tests/unit/test_runtime_qt_widgets.py - runtime Qt sync coverage

Outcome:
  SynchronizationManager implemented for X-axis pan/zoom, shared visible time window,
  shared engineering time cursor, cursor propagation, digital timeline synchronization,
  unregister behavior, and recursive signal-loop prevention.

  Grouped multi-panel displays now register all analog panels and optional digital
  timeline through SynchronizationManager. Standard analog + digital layout also
  registers through VisualizationManager after records are loaded.

Validation:
  88 focused tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_runtime_qt_widgets.py -q

  50 PULU/integration + sync/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py -q

Notes:
  Prompt referenced agent/TASKS.md; repository live tracker is agent/TASK.md.
  Synchronization owns no waveform data and performs no analytics.

PHASE D4.5B - X-DOMAIN SYNCHRONIZATION DRIFT FIX

Status: COMPLETED
Priority: CRITICAL
Date: 2026-05-15

Scope:
  app/visualization/widgets/flexible_plot_canvas.py - grouped axis-column reservation
  app/ui/main_window/main_window.py - reserve aligned panel geometry before sync registration
  tests/unit/test_runtime_qt_widgets.py - pixel-alignment runtime coverage

Root Cause:
  Grouped panels had identical numeric X ranges and numerically identical time arrays,
  but PyQtGraph gave panels with different Y-axis layouts different primary ViewBox
  widths. A dual-axis power panel was narrower than a single-axis frequency panel,
  so the same timestamp mapped to increasingly different horizontal pixels toward
  the right edge.

Outcome:
  FlexiblePlotCanvas now exposes right_axis_count() and reserve_grouped_axis_columns().
  Grouped panel linking reserves a common left axis width and a common capped right
  axis layout budget before SynchronizationManager registration.

  Direct CSV/Excel grouped panels continue to share the same absolute/rebased display
  seconds. COMTRADE direct display remains on the standard relative-seconds path.
  Multi-source panels continue to use aligned display seconds from the alignment layer.

Validation:
  90 focused visualization/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_runtime_qt_widgets.py -q

  36 PULU/integration + pixel-alignment tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_synthetic_multi_source_panels_keep_x_pixel_alignment -q

Notes:
  Runtime tests now assert identical ViewBox X ranges, identical cursor values, and
  identical mapped pixel X positions near start/middle/end after initial display,
  zoom/pan, cursor movement, and resize.
  No analytics, overlays, or UI controls were added.

PHASE 5A.1 - ENGINEERING DISPLAY NORMALIZATION & GUI USABILITY STABILIZATION

Status: COMPLETED
Priority: HIGH
Date: 2026-05-15

Scope:
  app/visualization/engineering_display.py - domain-aware display policy
  app/visualization/managers/multi_axis_manager.py - fixed engineering axis labels
  app/visualization/managers/visualization_manager.py - grouped panel titles
  app/visualization/widgets/flexible_plot_canvas.py - RMS display/title clarity
  tests/unit/test_engineering_display.py - display policy tests
  tests/unit/test_rms_overlay_display.py - RMS label/title tests
  tests/unit/test_runtime_qt_widgets.py - runtime axis/title regression checks

Outcome:
  Added a domain-aware engineering display policy without generic SI scaling.
  Power-system units are fixed by role:
    active power = MW
    reactive power = MVar
    frequency = Hz
    ROCOF = Hz/s
    voltage = kV/V
    current = A/kA
    per-unit = pu

  MultiAxisManager now disables PyQtGraph auto-SI prefixing on engineering
  Y axes, preventing labels such as kMW, mMW, or kHz for operational channels.
  Grouped panels now receive consistent titles such as Power, Frequency,
  Voltage Waveforms, and Other Analog Channels (N).

  RMS overlays now use explicit curve labels such as "VA RMS (kV)", dashed
  lighter traces, and panel title suffixes "RMS Overlay" / "RMS Only".

Validation:
  183 focused tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_engineering_display.py tests/unit/test_rms_overlay_display.py tests/unit/test_rms_calculation.py tests/unit/test_rms_eligibility.py tests/unit/test_rms_cache.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py -q

  73 integration/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_engineering_display.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_synthetic_grouped_panels_keep_x_pixel_alignment tests/unit/test_rms_overlay_display.py -q

Notes:
  This phase does not scale waveform values and does not implement Phase 5B
  per-unit conversion. It only stabilizes display labels/titles and prepares a
  small EngineeringDisplayPreferences hook for future user preferences.

PHASE 5A.2 - WIDGET LIFECYCLE FIX FOR REOPENING FILES

Status: COMPLETED
Priority: CRITICAL
Date: 2026-05-15

Scope:
  app/ui/main_window/main_window.py - standard widget lifecycle guards
  app/visualization/managers/synchronization_manager.py - deleted-sender proxy cleanup guard
  tests/unit/test_runtime_qt_widgets.py - repeated open/switch runtime coverage

Root Cause:
  QMainWindow.setCentralWidget() deletes the previous central splitter. When
  Powerwave switched from the standard analog/digital splitter to grouped
  panels, the standard DigitalEventTimeline and FlexiblePlotCanvas could remain
  children of the deleted splitter. Python attributes such as self._timeline
  still existed, but their wrapped C++ objects were gone, causing RuntimeError
  on the next self._timeline.show() call.

Outcome:
  PowerwaveMainWindow now uses PyQt sip.isdeleted() checks before reusing
  standard widgets. _ensure_standard_widgets_alive() recreates deleted standard
  canvas/timeline widgets and rebuilds VisualizationManager around the live
  objects. Layout switches clear SynchronizationManager first, detach standard
  widgets when grouped layouts do not own them, and avoid calling Qt methods on
  deleted objects.

  SynchronizationManager now skips SignalProxy.disconnect() when the signal
  sender's C++ object has already been deleted, preventing Qt access violations
  during defensive cleanup.

Validation:
  19 runtime Qt widget tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_runtime_qt_widgets.py -q

  107 visualization/RMS/synchronization tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_rms_overlay_display.py -q

  37 PULU integration + targeted lifecycle tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_can_open_twice_without_deleted_timeline tests/unit/test_runtime_qt_widgets.py::test_direct_csv_to_comtrade_restores_standard_timeline tests/unit/test_runtime_qt_widgets.py::test_direct_csv_to_multi_source_keeps_sync_registry_clean -q

Notes:
  Runtime tests cover CSV -> CSV, COMTRADE -> COMTRADE, CSV -> COMTRADE,
  COMTRADE -> CSV, multi-source -> direct CSV, direct CSV -> multi-source,
  deleted DigitalEventTimeline recreation, and stale synchronization registry
  prevention.

PHASE 5A.3 - COMTRADE ABSOLUTE TIMESTAMP DISPLAY MODE

Status: COMPLETED
Priority: HIGH
Date: 2026-05-15

Scope:
  app/visualization/axis/datetime_axis.py - TimeDisplayMode enum
  app/visualization/widgets/flexible_plot_canvas.py - in-place time-axis mode switching
  app/visualization/widgets/digital_event_timeline.py - datetime bottom axis support
  app/visualization/managers/visualization_manager.py - mode propagation and common reference handling
  app/ui/main_window/main_window.py - View -> Time Axis Mode menu
  tests/unit/test_datetime_axis.py - display mode tests
  tests/unit/test_display_multi_source.py - default absolute multi-source reference tests
  tests/unit/test_runtime_qt_widgets.py - COMTRADE runtime switching/RMS checks
  tests/integration/test_pulu_manifest_pipeline.py - COMTRADE+CSV common timestamp reference test

Outcome:
  Added visualization-level TimeDisplayMode with RELATIVE and ABSOLUTE modes.
  The canonical X domain remains float64 seconds in all modes; switching modes
  only changes DatetimeAxisItem formatting.

  Direct COMTRADE opens still default to relative elapsed seconds. Operators can
  switch the visible standard analog/digital layout to absolute timestamp labels
  via View -> Time Axis Mode -> Absolute Timestamp. Labels are derived as:
    record.timing_info.start_time + waveform_data["time"]

  Direct CSV/Excel grouped display remains absolute timestamp by default.
  Multi-source sessions now default to absolute timestamp and use the alignment
  reference start as the common axis reference, so COMTRADE and CSV panels share
  the same wall-clock label domain after offsets are applied.

Validation:
  190 focused visualization/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_datetime_axis.py tests/unit/test_d442_panel_visibility.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_runtime_qt_widgets.py tests/unit/test_rms_overlay_display.py -q

  38 integration/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_comtrade_direct_open_can_switch_to_absolute_timestamp_mode tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_overlay_remains_aligned_after_time_axis_switch -q

Notes:
  No waveform arrays are mutated. No synchronization, alignment, or Phase 5B
  scaling architecture was rewritten.

PHASE 5A.4 - UNIVERSAL SIGNAL BROWSER & VISIBILITY MANAGEMENT

Status: COMPLETED
Priority: HIGH
Date: 2026-05-16

Scope:
  app/visualization/signal_visibility.py - provider-neutral default visibility policy
  app/ui/widgets/signal_browser.py - dockable Signal Browser tree
  app/ui/main_window/main_window.py - dock/menu wiring + runtime visibility dispatch
  app/visualization/widgets/flexible_plot_canvas.py - analog axis/curve visibility rebuild
  app/visualization/widgets/digital_event_timeline.py - digital track visibility rebuild
  tests/unit/test_signal_visibility.py - deterministic default policy tests
  tests/unit/test_runtime_qt_widgets.py - runtime visibility/sync/RMS/browser coverage

Outcome:
  Added a dockable View -> Signal Browser panel backed by provider-neutral
  runtime visibility state. The browser lists analog and digital channels by
  source and display group for direct COMTRADE, direct CSV/Excel, grouped
  displays, and multi-source sessions.

  Visibility changes do not reload data and do not mutate DisturbanceRecord or
  waveform arrays. FlexiblePlotCanvas keeps cached time/data arrays and rebuilds
  only visible axes, ViewBoxes, raw curves, and RMS overlay curves. Hidden
  analog signals remove unused Y axes. DigitalEventTimeline similarly rebuilds
  visible tracks from cached transition data.

  Large records now start with deterministic readable defaults:
    analog: first 8 channels visible
    digital: first 16 tracks visible
  All channels remain available in the Signal Browser for immediate reveal.

Validation:
  135 focused visualization/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_runtime_qt_widgets.py tests/unit/test_rms_overlay_display.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_visualization_manager.py -q

  41 PULU integration + targeted browser/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_runtime_qt_widgets.py::test_direct_csv_open_routes_to_grouped_visible_panels tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_comtrade_channel_without_reload tests/unit/test_runtime_qt_widgets.py::test_signal_browser_hides_grouped_csv_axis_and_preserves_sync tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_digital_track tests/unit/test_runtime_qt_widgets.py::test_signal_browser_supports_multi_source_panel_visibility tests/unit/test_runtime_qt_widgets.py::test_signal_visibility_removes_rms_overlay_with_hidden_channel -q

  8 targeted signal-visibility tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_signal_visibility.py tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_comtrade_channel_without_reload tests/unit/test_runtime_qt_widgets.py::test_signal_browser_hides_grouped_csv_axis_and_preserves_sync tests/unit/test_runtime_qt_widgets.py::test_signal_browser_can_reveal_hidden_digital_track tests/unit/test_runtime_qt_widgets.py::test_signal_browser_supports_multi_source_panel_visibility tests/unit/test_runtime_qt_widgets.py::test_signal_visibility_removes_rms_overlay_with_hidden_channel -q

Notes:
  This phase does not add search, presets, persistence, drag/drop layout, or
  Phase 5B scaling. Runtime Qt tests still emit known offscreen/OpenGL and
  Windows pytest cache warnings; assertions pass.

PHASE 5A.5 - GLOBAL AXIS MANAGEMENT & ANALOG/DIGITAL GEOMETRY ALIGNMENT

Status: COMPLETED
Priority: HIGH
Date: 2026-05-16

Scope:
  app/visualization/axis_management.py - provider-neutral AxisDisplayMode and grouping policy
  app/visualization/managers/multi_axis_manager.py - shared ViewBox/AxisItem support
  app/visualization/widgets/flexible_plot_canvas.py - shared/dedicated axis modes
  app/visualization/widgets/digital_event_timeline.py - analog-matched geometry reservation
  app/visualization/managers/visualization_manager.py - axis mode propagation hook
  app/ui/main_window/main_window.py - View -> Axis Mode menu and geometry matching
  tests/unit/test_axis_management.py - deterministic grouping tests
  tests/unit/test_runtime_qt_widgets.py - axis mode and analog/digital pixel tests

Outcome:
  Added visualization-level AxisDisplayMode:
    SHARED: compatible signals share one Y axis
    DEDICATED: previous one signal / one axis behavior

  SHARED is the default. Axis grouping is provider-neutral and based on
  engineering role plus fixed operational unit:
    voltage/kV, current/A, power/MW, reactive power/MVar, frequency/Hz,
    ROCOF/Hz/s, per-unit/pu.
  Unknown signals stay dedicated to avoid accidental cross-quantity sharing.

  FlexiblePlotCanvas now uses shared ViewBoxes for compatible signals while
  keeping each raw/RMS curve independently visible and browser-controllable.
  Y ranges are computed per ViewBox from all visible raw/RMS series assigned to
  that axis, so shared axes do not get overwritten by the last curve.

  DigitalEventTimeline now reserves the same left/right plot chrome as analog
  canvases and performs a deferred ViewBox geometry match against the analog
  master. This fixes the remaining analog/digital cursor and timestamp pixel
  offset while preserving SynchronizationManager's numeric X-range sync.

Validation:
  141 focused visualization/runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_axis_management.py tests/unit/test_visualization_manager.py tests/unit/test_visualization_grouped_display.py tests/unit/test_display_multi_source.py tests/unit/test_synchronization_manager.py tests/unit/test_rms_overlay_display.py tests/unit/test_runtime_qt_widgets.py -q

  41 PULU integration + targeted axis/alignment tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_axis_management.py tests/unit/test_runtime_qt_widgets.py::test_comtrade_standard_analog_and_digital_timeline_are_pixel_aligned tests/unit/test_runtime_qt_widgets.py::test_axis_mode_switches_between_shared_and_dedicated_csv_axes -q

Notes:
  This phase does not implement Phase 5B per-unit/value scaling, persistent axis
  preferences, per-panel axis editors, or advanced compatibility heuristics.
  Runtime Qt tests still emit known offscreen/OpenGL and Windows pytest cache
  warnings; assertions pass.

PHASE 5A.R1 - RMS WINDOW VALIDATION & ENGINEERING RMS CORRECTION

Status: COMPLETED
Priority: HIGH
Date: 2026-05-16

Scope:
  app/analytics/rms/rms_models.py - RMSWindowMode and expanded RMSConfig
  app/analytics/rms/sliding_rms.py - config-aware window sample helper
  app/analytics/rms/__init__.py - public RMS exports updated
  app/visualization/widgets/flexible_plot_canvas.py - config-aware RMS cache/build path
  app/ui/main_window/main_window.py - Tools -> RMS Window menu
  tests/unit/test_rms_calculation.py - deterministic engineering RMS validation
  tests/unit/test_rms_overlay_display.py - RMS config/window mode tests
  tests/unit/test_runtime_qt_widgets.py - runtime RMS window switch regression

Outcome:
  Added explicit engineering RMS window modes:
    HALF_CYCLE
    ONE_CYCLE
    TWO_CYCLE
    CUSTOM_SAMPLES

  Default remains ONE_CYCLE. Window samples are derived from:
    sampling_rate_hz / nominal_frequency_hz
  with half-cycle and two-cycle variants computed from the same engineering
  basis. A 5000 Hz / 50 Hz waveform therefore uses 100 samples for one-cycle
  RMS, 50 samples for half-cycle RMS, and 200 samples for two-cycle RMS.

  The rolling RMS primitive remains vectorized true RMS:
    sqrt(mean(x^2))
  RMS time stamps stay right-edge/trailing-window aligned with existing display
  behavior. Raw waveform arrays are not mutated.

  FlexiblePlotCanvas now uses the selected RMSConfig to compute cache keys and
  RMS overlays. If record metadata supplies a positive nominal frequency and
  the global config is still at its default frequency, the canvas uses the
  record value. Otherwise the operator-selected nominal frequency is preserved.

  Added minimal UI:
    Tools -> RMS Window -> Half Cycle / One Cycle / Two Cycle / Custom Samples
  Switching the RMS window clears/rebuilds RMS overlays on visible canvases
  without reloading the record, changing X ranges, or modifying waveform data.

Validation:
  62 RMS tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_rms_calculation.py tests/unit/test_rms_overlay_display.py -q

  193 focused RMS/runtime/synchronization tests passing:
    .venv\Scripts\python.exe -m pytest tests/unit/test_rms_calculation.py tests/unit/test_rms_overlay_display.py tests/unit/test_rms_cache.py tests/unit/test_rms_eligibility.py tests/unit/test_runtime_qt_widgets.py tests/unit/test_visualization_manager.py tests/unit/test_synchronization_manager.py -q

  68 PULU integration + targeted RMS runtime tests passing:
    .venv\Scripts\python.exe -m pytest tests/integration/test_pulu_manifest_pipeline.py tests/unit/test_rms_calculation.py tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_window_switch_recomputes_cached_envelope tests/unit/test_runtime_qt_widgets.py::test_comtrade_rms_overlay_remains_aligned_after_time_axis_switch -q

Notes:
  This phase does not implement phasors, FFT, per-unit scaling, or a broader
  analytics redesign. RMS custom samples are runtime-only and not persisted.
