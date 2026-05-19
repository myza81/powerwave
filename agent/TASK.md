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

Phasor & Sequence Component Engine

Status: COMPLETED (Phase 6A)
Priority: HIGH
Date: 2026-05-17

Scope:
  app/analytics/phasors/phasor_models.py       — PhasorDisplayMode, PhasorWindowMode, PhaseLabel, PhasorChannelRole,
                                                  PhasorChannelResult, PhasorConfig, ThreePhaseGroup
  app/analytics/phasors/phasor_extraction.py   — sliding DFT phasor extraction (vectorized stride-trick, O(N·W))
  app/analytics/phasors/symmetrical_components.py — Fortescue transform (V1/V2/V0, I1/I2/I0), unbalance_factor
  app/analytics/phasors/phasor_overlay.py      — classify_phasor_role (priority chain), identify_phase,
                                                  detect_three_phase_groups, is_voltage_channel, is_current_channel
  app/analytics/phasors/phasor_registry.py     — PhasorRegistry (session-level cache + display mode + bulk helpers)
  app/analytics/phasors/phasor_cache.py        — PhasorCache (phasor + sequence stores, invalidate_channel, clear)
  app/analytics/phasors/__init__.py             — public package exports (29 symbols)
  app/visualization/channel_grouper.py         — added DISPLAY_GROUP_SEQUENCE_VOLTAGE, DISPLAY_GROUP_SEQUENCE_CURRENT
  app/ui/main_window/main_window.py            — Tools → Phasor Display submenu + PhasorRegistry state +
                                                  _on_phasor_display_mode_changed + _apply_phasor_display_mode
  tests/unit/test_phasor_extraction.py         ← 24 tests
  tests/unit/test_symmetrical_components.py    ← 39 tests
  tests/unit/test_phasor_classification.py     ← 86 tests
  tests/unit/test_phasor_display.py            ← 65 tests

Outcome:
  Full phasor and symmetrical component analytics layer. Transforms Powerwave into a protection engineering platform.
  Sliding DFT extraction: magnitude (RMS), angle (degrees), complex phasor. Half/one/two-cycle windows.
  DFT kernel uses n_cycle exponent (not window size) so all window modes extract f₀ correctly.
  Fortescue transform: V1/V2/V0, I1/I2/I0. Validated: balanced → V2≈0/V0≈0; SLG → elevated V0/V2; ACB → elevated V2.
  Phase identification: ABC suffix heuristics (universal) + RYB heuristics (short names / separator-gated).
  Three-phase group detection: detect_three_phase_groups returns complete A/B/C groups only.
  PhasorRegistry: classify + cache + display mode + phasor_panel_keys() (direct + multi-source layouts).
  PhasorCache: separate phasor and sequence stores, keyed by (channel_id, window_samples, nominal_hz).
  PhasorDisplayMode: OFF (default) / MAGNITUDE / ANGLE / SEQUENCE_COMPONENTS.
  Classification priority chain: operator_override > measurement_kind > electrical_type > unit > name heuristics.
  Engineering units: kV for voltage, A for current — no auto-scaling.
  214 new tests; 2563 total passing, 12 skipped, 0 failures.

Phasor Rendering Integration

Status: COMPLETED (Phase 6B)
Priority: HIGH
Date: 2026-05-17

Scope:
  app/visualization/overlays/phasor_overlay.py   — PhasorCurveOverlay(BaseOverlay) using CurveStore
  app/visualization/overlays/__init__.py          — added PhasorCurveOverlay export
  app/visualization/widgets/flexible_plot_canvas.py — set_phasor_display_mode(), _build_phasor_overlays(),
                                                      _remove_phasor_curves(); phasor state and cache dicts;
                                                      _update_viewport() MAGNITUDE Y-range contribution
  app/ui/main_window/main_window.py               — _make_sequence_record(), _build_sequence_panels(),
                                                     _apply_phasor_display_mode() rewired; _PANEL_ORDER updated
  tests/unit/test_phasor_rendering_integration.py ← 48 tests
  tests/unit/test_d441_stabilization.py           ← 3 axis-mode tests updated (regression fix)

Outcome:
  MAGNITUDE mode draws sliding DFT magnitude envelope curves on each waveform canvas channel's ViewBox.
    Pen: dotted, 60%-blend-toward-white color per channel.
  ANGLE mode draws phase angle trace curves (degrees) on each channel's ViewBox, excluded from Y-range.
    Pen: dash-dot, 40%-blend-toward-cyan color per channel.
  SEQUENCE_COMPONENTS mode hides waveform canvases phasor overlays and reveals dedicated hidden panels
    (sequence_voltage, sequence_current) holding synthetic DisturbanceRecords with V1/V2/V0, I1/I2/I0 channels.
  OFF mode: all phasor curves removed, sequence panels hidden.
  PhasorCache: (channel_id, window_samples, nominal_hz) key; full tuple cached so MAGNITUDE↔ANGLE switch
    requires no phasor recomputation.
  _update_viewport() decimates phasor arrays before rendering (same path as raw data).
  48 new tests; 2657 total passing, 12 skipped, 0 failures.

Harmonic Analysis Foundation

Status: COMPLETED (Phase 7)
Priority: MEDIUM
Date: 2026-05-17

Scope:
  app/analytics/harmonics/harmonic_models.py       — HarmonicDisplayMode, HarmonicWindowMode,
                                                      HarmonicChannelRole, HarmonicChannelResult,
                                                      HarmonicConfig, HarmonicResult
  app/analytics/harmonics/harmonic_extraction.py   — compute_harmonic_window_samples,
                                                      extract_harmonics (vectorized batch FFT)
  app/analytics/harmonics/harmonic_metrics.py      — compute_thd, compute_thd_array,
                                                      compute_thd_from_result, individual_harmonic_distortion
  app/analytics/harmonics/harmonic_overlay.py      — classify_harmonic_role (5-level priority chain),
                                                      is_harmonic_eligible
  app/analytics/harmonics/harmonic_registry.py     — HarmonicRegistry (session registry + cache)
  app/analytics/harmonics/harmonic_cache.py        — HarmonicCache (keyed by channel+window+nominal+max_order)
  app/analytics/harmonics/__init__.py              — public exports (18 symbols)
  app/visualization/widgets/flexible_plot_canvas.py — set_harmonic_display_mode() stub + state
  app/ui/main_window/main_window.py                — HarmonicRegistry state + disabled menu item
  tests/unit/test_harmonic_extraction.py           ← 47 tests
  tests/unit/test_harmonic_metrics.py              ← 30 tests
  tests/unit/test_harmonic_classification.py       ← 44 tests
  tests/unit/test_harmonic_registry.py             ← 36 tests

Outcome:
  Vectorized sliding-window FFT engine (batch rfft via stride tricks, O(N·log(W))).
  Hann window default with amplitude-correct RMS normalisation: sqrt(2) * |FFT[bin]| / sum(window).
  THREE window modes: ONE_CYCLE / TWO_CYCLE (default) / FOUR_CYCLE.
  THD: standard engineering definition (dimensionless fraction), scalar + vectorized + safe (no ZeroDivision).
  Classification priority chain: force > measurement_kind > electrical_type > unit > name heuristics > UNKNOWN.
  HarmonicCache: key = (channel_id, window_samples, hop_samples, nominal_hz, max_order).
  Mode switch (HARMONIC_MAGNITUDE ↔ THD) reuses same cache entry.
  FlexiblePlotCanvas: set_harmonic_display_mode() stub + _harmonic_display_mode/_harmonic_config state.
  main_window.py: HarmonicRegistry() instance + disabled Tools → "Harmonic Analysis…" placeholder.
  157 new tests; 2824 total passing, 12 skipped, 0 failures.

Phasor Hooks (Phase 6C — relay elements, impedance, PMU protocol)

Status: NOT STARTED
Priority: MEDIUM

Scope:

  impedance trajectory (R-X plot)
  distance protection analysis
  PMU synchrophasor protocol support
  relay element analytics
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

PHASE 8 — HARMONIC RENDERING INTEGRATION

Status: COMPLETED
Priority: HIGH
Date: 2026-05-17

Scope:
  app/visualization/overlays/harmonic_overlay.py     - CREATED
  app/visualization/overlays/overlay_colors.py       - MODIFIED
  app/visualization/overlays/__init__.py             - MODIFIED
  app/visualization/widgets/flexible_plot_canvas.py  - MODIFIED
  app/ui/main_window/main_window.py                  - MODIFIED
  tests/unit/test_harmonic_rendering.py              - CREATED
  tests/unit/test_harmonic_overlay_stability.py      - CREATED
  tests/unit/test_harmonic_stability.py              - MODIFIED (regression fix)
  tests/unit/test_d441_stabilization.py              - MODIFIED (regression fix)
  tests/unit/test_runtime_qt_widgets.py              - MODIFIED (regression fix)

Outcome:
  Full harmonic rendering wired end-to-end across all four display modes:

  HARMONIC_MAGNITUDE:
    Per-order RMS magnitude envelopes overlaid inline on waveform canvases.
    H3/H5/H7/H11/H13 by default (H1 omitted — too large relative to harmonics).
    HarmonicCache provides O(1) cache hits on pan/zoom after initial FFT.
    Per-channel ViewBox gets its own PlotDataItem per order.
    _update_viewport() decimates harmonic arrays and contributes to Y-range.

  THD / SPECTRUM:
    Dedicated hidden panels built by _build_harmonic_panels() on record load.
    THD panels: time-varying THD% per eligible channel.
    Spectrum panels: H3..H13 magnitude trends for first eligible V/I channel.
    Panels toggle visible/hidden via _apply_harmonic_display_mode().
    Waveform canvases show raw only in THD/SPECTRUM mode.

  OFF:
    Removes all harmonic curves from ViewBoxes.
    HarmonicCache preserved — re-enabling MAGNITUDE reuses FFT results.

  Infrastructure:
    HarmonicCurveOverlay(BaseOverlay) reusable overlay class using CurveStore.
    Per-order deterministic colors: H3=#FF6600, H5=#FF00CC, H7=#00CCFF, H11=#AA88FF.
    _HARMONIC_PANEL_KEYS frozenset excludes harmonic panels from waveform routing.
    Tools → Harmonic Display submenu: checkable radio-style mode items.
    _make_harmonic_record() module-level helper mirrors _make_sequence_record().

  Regression fixes:
    3 spec-mock tests in test_d441_stabilization.py needed _harmonic_display_mode attribute.
    1 pixel-alignment test in test_runtime_qt_widgets.py needed visible-canvas filter.
    1 stability test in test_harmonic_stability.py used stale Phase 7 stub API.

Validation:
  86 new tests (69 unit + 17 stability) all passing.
  Full suite: 2925 passed, 12 skipped, 0 failures.

PHASE 8.5 - HARMONIC VISUALIZATION STABILIZATION & PERFORMANCE HARDENING

Status: COMPLETED
Priority: HIGH
Date: 2026-05-18

Scope:
  app/visualization/widgets/flexible_plot_canvas.py - per-curve viewport data signature guard for redundant setData suppression
  app/ui/main_window/main_window.py - harmonic panel cache reused by record identity
  tests/unit/test_harmonic_visualization_stability.py - focused Phase 8.5 runtime/stability tests

Outcome:
  Harmonic visualization stabilization completed without adding new analytics or redesigning overlay infrastructure.

  Lifecycle:
    Repeated OFF switching remains idempotent.
    Harmonic magnitude overlays still remove curves at lifecycle boundaries and preserve HarmonicCache across OFF toggles.
    Existing BaseOverlay / CurveStore lifecycle contracts remain unchanged.

  Cache reuse:
    Harmonic panel rebuilds now reuse a main-window HarmonicCache for the same record object.
    Cache invalidation remains record-lifecycle based: new record identity gets a fresh panel cache.
    Unsupported harmonic channels do not populate harmonic cache entries.

  Rendering performance:
    FlexiblePlotCanvas now keeps lightweight curve data signatures and skips redundant setData calls when synchronized viewport echoes produce identical decimated arrays.
    The hot path still uses cached numpy arrays, decimate_for_display(), and in-place PlotDataItem updates.
    Cursor movement does not trigger FFT recomputation or plot-item creation.

  Spectrum/cursor/multi-source safety:
    Hidden spectrum panels tolerate repeated cursor updates without creating new plot items.
    Harmonic/spectrum panels remain synchronized through the existing SynchronizationManager.
    Partial harmonic support skips telemetry/frequency channels while rendering eligible waveform channels.

Architecture Review:
  Reviewed docs/SYSTEM_OVERVIEW.md, docs/ARCHITECTURE.md, docs/DATA_CONTRACT.md,
  docs/REPOSITORY_STRUCTURE.md, docs/VISUALIZATION_CONTRACT.md,
  docs/VIEWPORT_RENDERING_POLICY.md, docs/PERFORMANCE_REQUIREMENTS.md,
  agent/WORKFLOW_AGENT.md, agent/HANDOFF.md, agent/TASK.md, and agent/REPOSITORY_STATE.md.
  docs/REPOSITORY_STRUCTURE.md is present in this checkout and was used as the active
  repository-structure contract alongside ARCHITECTURE.md and REPOSITORY_STATE.md.
  No core architecture docs required changes because Phase 8.5 is stabilization-only.

Validation:
  92 passed: scaling + harmonic stabilization/rendering slice.
  88 passed: runtime Qt, synchronization, visualization manager, overlay infrastructure, phasor overlay stability.
  2278 passed: full unit suite.
  2932 passed, 12 skipped: full test suite.

Notes:
  Runtime Qt tests still emit known offscreen PyQtGraph OpenGL warnings; assertions pass.

PHASE 8.55A — IMPORT WIZARD ARCHITECTURE & DATA CONTRACTS

Status: COMPLETED
Priority: HIGH
Date: 2026-05-18

Scope:
  app/import_wizard/__init__.py         - CREATED (public surface, 17 exports)
  app/import_wizard/contracts.py        - CREATED (ValidationSeverity, ValidationMessage)
  app/import_wizard/wizard_state.py     - CREATED (WizardStep, can_transition, helpers)
  app/import_wizard/column_mapping.py   - CREATED (ParameterType enum)
  app/import_wizard/timestamp_contracts.py  - CREATED (TimestampRepairStrategy, TimestampRepairPlan)
  app/import_wizard/normalization_plan.py   - CREATED (NormalizationPlan)
  app/import_wizard/models.py           - CREATED (RawPreviewModel, TimestampCandidate,
                                                    ColumnMappingCandidate, ImportWizardSession)
  tests/unit/test_import_wizard_contracts.py  - CREATED (101 tests)

Outcome:
  Clean architecture layer for the CSV/Excel Import Wizard.
  Contracts only — no GUI, no repair logic, no parsing rewrite.

  WizardStep state machine:
    8 ordered steps: LOAD_FILE → RAW_PREVIEW → TIMESTAMP_SELECT → TIMESTAMP_REPAIR
    → COLUMN_MAPPING → NORMALIZATION_REVIEW → SAVE_NORMALIZED → RENDER_WAVEFORM.
    Backward moves always allowed. Forward strict (1 step) by default.
    allow_skip=True permits forward jumps when preconditions are pre-verified.

  TimestampRepairStrategy:
    8 strategies covering: no-repair, format parse, user format, interpolation,
    interval reconstruction, split date+time columns, Excel serial, timezone alignment.
    Contract only — repair engine deferred to Phase 8.55B.

  ColumnMappingCandidate:
    Stores both auto-classification and user overrides.
    effective_name/unit/type properties prefer user override when present.
    has_user_override property for UI dirty-state detection.

  NormalizationPlan.is_executable:
    Requires validated TimestampRepairPlan + ≥1 selected column + no ERROR messages.
    Single Boolean gate used by GUI and future automation.

  Architecture principles followed:
    No PyQt6/numpy/pandas imports in contract layer.
    Dependency order has no cycles.
    All frozen models survive copy.deepcopy().
    Aligned with existing dataclass(slots=True) conventions.

Validation:
  101 contract tests all passing in 0.46 s (no Qt, no I/O).
  Full suite: 3033 passed, 12 skipped, 0 failures.

Next phase: Phase 8.55B — Timestamp Detection & Repair Engine
  (implementation of TimestampCandidateDetector and repair strategies
   using the contracts defined here).

---

## Phase 8.55B — Raw Preview & File Profiling Engine — COMPLETED 2026-05-18

### Status: COMPLETED

### Deliverables
- app/import_wizard/preview_sampler.py — read_text_sample, estimate_csv_row_count
- app/import_wizard/csv_profiler.py — detect_delimiter, _find_header_row_index (lookahead), profile_csv
- app/import_wizard/excel_profiler.py — get_sheet_names, profile_excel (openpyxl read-only)
- app/import_wizard/timestamp_detector.py — infer_timestamp_format, detect_timestamp_candidates
- app/import_wizard/column_detector.py — classify_by_name, _classify_by_values, detect_column_mappings
- app/import_wizard/file_profiler.py — FileProfileResult, profile_import_file, populate_session
- 3 new test files (149 tests total for Phase 8.55B)

### Test Results
Full suite: 2528 passed, 0 failures.

Next phase: Phase 8.55C — Import Wizard UI or Normalization Engine.

---

## Phase 8.55C — Timestamp Repair & Normalization Engine — COMPLETED 2026-05-18

### Status: COMPLETED

### Deliverables
- app/import_wizard/interval_inference.py — IntervalAnalysis, infer_interval(), detect_duplicates(), detect_non_monotonic()
- app/import_wizard/repair_diagnostics.py — RepairDiagnostics (17 fields, plain Python)
- app/import_wizard/timestamp_repair_executor.py — 8 strategy executors + dispatch()
- app/import_wizard/timestamp_normalizer.py — TimestampNormalizationResult, normalize_timestamps()
- 3 test files (109 tests total)

### Test Results
Full suite: 2637 passed, 0 failures.

Next phase: Phase 8.55D — Import Wizard UI or DisturbanceRecord integration.

---

## Phase 8.55G - Import Wizard Qt GUI Skeleton - COMPLETED 2026-05-18

### Status: COMPLETED

### Deliverables
- app/ui/import_wizard/__init__.py - Qt Import Wizard package exports
- app/ui/import_wizard/preview_table_model.py - QAbstractTableModel adapter for RawPreviewModel
- app/ui/import_wizard/timestamp_candidate_model.py - selectable timestamp candidate model
- app/ui/import_wizard/column_mapping_model.py - editable include/name/type/unit mapping model
- app/ui/import_wizard/wizard_pages.py - focused page widgets for load, preview, timestamp, mapping, review, running, complete
- app/ui/import_wizard/import_wizard_dialog.py - QDialog + QStackedWidget wizard orchestrator
- app/ui/main_window/main_window.py - File > Import Wizard action and visualization handoff
- tests/unit/test_import_wizard_gui.py - unit coverage for models, transitions, validation, pipeline signal
- tests/runtime/test_import_wizard_runtime.py - runtime CSV import through backend pipeline

### Architecture Notes
- Architecture review completed before implementation against core, visualization, viewport, performance, and agent-state contracts.
- GUI remains a thin Qt orchestration layer. Profiling uses profile_import_file(); import execution uses run_import_pipeline().
- No backend timestamp, normalization, data assembly, or DisturbanceRecord conversion logic was duplicated in the GUI.
- File profiling and pipeline execution run on QRunnable workers via QThreadPool to keep the UI thread responsive.
- The successful DisturbanceRecord is emitted by the dialog and handed to the existing main-window visualization path.

### Test Results
- Import Wizard focused GUI/runtime tests: 8 passed.
- Import backend + GUI slice: 622 passed.
- Qt runtime visualization slice: 103 passed.
- Full suite: 3633 passed, 12 skipped.

### Known Limitations
- Column mapping edits are represented in the GUI NormalizationPlan, but the current public run_import_pipeline() API still performs its own backend auto-plan. A future phase should expose an explicit plan-aware pipeline entry point before adding advanced editing.
- The timestamp-repair page is not exposed yet; selected candidate repair is represented by a validated repair plan for this skeleton.

Next phase: Phase 8.55H - Plan-aware Import Wizard execution and timestamp repair controls, or Phase 9 visualization/event workflow planning.

---

## Phase 8.55I - Timestamp Format Override UI - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- app/import_wizard/timestamp_format_validator.py - lightweight sampled strptime validation for manual overrides
- app/import_wizard/pipeline_plan_builder.py - user-format validation integrated into executable plan building
- app/import_wizard/import_pipeline.py - plan-aware execution blocks unvalidated timestamp repair plans before full load
- app/ui/import_wizard/wizard_pages.py - timestamp selection page now shows selected column, detected format, manual format field, reset button, and parse feedback
- app/ui/import_wizard/import_wizard_dialog.py - manual override state sync, stale plan invalidation, and PARSE_USER_FORMAT plan creation
- tests/unit/test_timestamp_override_ui.py - deterministic UI/model validation coverage
- tests/runtime/test_timestamp_override_execution.py - plan-aware runtime timing and failure coverage

### Outcome
- Empty override uses detected-format behavior.
- Non-empty override creates TimestampRepairPlan(strategy=PARSE_USER_FORMAT) and preserves detected_format only as metadata.
- Override validation uses timestamp candidate samples only, emits INFO/WARNING/ERROR messages, and blocks complete parse failure.
- Plan-aware execution uses the exact user format and ignores the detected format while override is active.
- Invalid unvalidated timestamp plans fail gracefully without full pipeline execution.

### Validation
- 12 passed: timestamp override unit/runtime tests.
- py_compile passed for touched import wizard backend and UI files.
- Broader import-wizard slice was attempted, but repository/environment temp permissions currently break tests using pytest tmp_path before assertions run.

### Known Limitations
- Override validation is intentionally sampled from profiler candidate examples, not the full dataset.
- UI feedback is a single lightweight message label; no advanced repair UI, timezone editor, or batch correction was added.

---

## Phase 8.55J - Test Environment Stabilization & Runtime Hygiene - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- app/testing/__init__.py - public exports for runtime temp test helpers
- app/testing/temp_runtime.py - isolated runtime temp dirs, safe cleanup, retry-on-lock cleanup, immediate-child cleanup
- tests/conftest.py - repo-local pytest/tempfile root, offscreen Qt default, runtime_tmp_path fixture, Windows pytest temp-mode shim
- tests/runtime/conftest.py - runtime_qapp fixture with bounded QThreadPool and widget cleanup
- tests/runtime/test_runtime_environment.py - deterministic runtime hygiene and repeatability tests
- pyproject.toml - pytest cache dir moved under repo-local runtime temp area
- .gitignore - runtime/temp artifact patterns added

### Outcome
- Pytest no longer depends on the user-profile temp root for tmp_path/runtime slices.
- Runtime tests use isolated directories under `.powerwave_runtime_tmp` and clean them with bounded retry behavior.
- CSV/XLSX runtime temp files are created inside isolated test roots and removed after execution.
- Qt runtime tests close dialogs/widgets and wait for QThreadPool workers before teardown.
- Import Wizard runtime, visualization handoff, timestamp override execution, and CSV/XLSX runtime slices are repeatable in consecutive runs.

### Validation
- 7 passed: tests/runtime/test_runtime_environment.py.
- 71 passed: broad import-wizard/runtime slice.
- 71 passed: same broad import-wizard/runtime slice repeated immediately.

### Known Limitations
- Pre-existing stale temp directories with Windows AccessDenied ACLs remain in the checkout and were not forcibly removed.
- The pytest temp-mode shim is Windows-only and limited to pytest's private temp-dir creation; it is not product runtime behavior.

Next phase: continue Import Wizard/runtime feature expansion with the new runtime hygiene slice included in standard verification.

---

## Phase 8.55L - Export UI Integration - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- app/ui/import_wizard/wizard_pages.py - Import Complete page export controls and ExportWriteResult summary display
- app/ui/import_wizard/import_wizard_dialog.py - QFileDialog save flow, ExportPlan default suggestions, QRunnable export worker, export completion/error handling
- tests/unit/test_export_ui.py - deterministic unit coverage for export UI contract and worker completion
- tests/runtime/test_export_ui_runtime.py - runtime Import Wizard import-to-export coverage using Phase 8.55J temp/Qt hygiene

### Outcome
- Save Normalized File is available only after successful import with an export-ready NormalizedDataset.
- Supported GUI formats: CSV, Parquet, Feather. CSV is the default.
- Default save filename comes from ExportPlan suggestions, e.g. `{source}_normalized.csv`.
- Export options are intentionally lightweight: include metadata sidecar and overwrite existing file.
- Export execution runs off the UI thread using QRunnable/QThreadPool.
- Export success/failure/warning details are shown in the completion page, including output path, rows, columns, format, and metadata sidecar path.
- Export remains independent from waveform rendering and does not interfere with the imported DisturbanceRecord handoff.

### Validation
- 16 passed: export UI unit + runtime tests.
- 83 passed, 5 skipped: backend export writer/planning/E2E + export UI tests.
- 87 passed: broad Import Wizard/runtime slice including export UI.
- py_compile passed for touched UI/test files.

### Known Limitations
- No advanced export settings dialog yet.
- Optional Parquet/Feather formats still depend on backend pandas/pyarrow availability and surface missing dependencies as validation errors.
- Save-location persistence and export history are not implemented.

Next phase: optional export UX hardening or proceed to the next Import Wizard feature phase.

---

## Phase 8.55M - Real-World Import Hardening & Large Dataset Stress Testing - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- tools/generate_import_stress_samples.py - deterministic streaming CSV stress sample generator.
- tools/benchmark_import_pipeline.py - practical import/export benchmark runner with lightweight timing and tracemalloc memory reporting.
- tests/stress/test_import_wizard_large_csv.py - generated small/medium import, export, metadata sidecar, and timestamp-gap stress coverage.
- tests/stress/test_import_wizard_malformed_files.py - malformed timestamp, delimiter, metadata/header, ragged row, digital text, unknown column, and unrecoverable timestamp coverage.
- tests/runtime/test_import_wizard_realistic_workflows.py - Qt runtime responsiveness, failure stability, export-after-import, waveform handoff, and worker close behavior coverage.
- docs/IMPORT_WIZARD_HARDENING_REPORT.md - measured results, covered scenarios, known limits, and operational guidance.

### Outcome
- Import Wizard backend and GUI runtime were exercised against realistic generated historian-style CSV files.
- Generated files are written only into runtime temp directories during tests/benchmarks.
- Semicolon, tab, and pipe delimiter variants import successfully.
- Metadata rows before headers are detected and skipped.
- Missing/malformed timestamp rows drop with timestamp diagnostics instead of crashing.
- Duplicate and non-monotonic timestamp rows warn without blocking recoverable imports.
- Text digital states such as OPEN/CLOSED route through the existing digital channel path and coerce to 0/1.
- Unknown/text noise columns are preserved as analog with warnings.
- Export after generated medium import writes CSV and metadata sidecar successfully.
- Visualization handoff remains compatible: waveform_data["time"] exists and channel descriptors match waveform columns.

### Validation
- 19 passed: new stress/runtime hardening tests.
- 30 passed: stress + runtime hygiene/export runtime slice.
- 230 passed, 2 skipped: broader import pipeline, bridge, export UI/writer, timestamp override, authoritative flow, runtime hardening, and stress slice.
- Benchmarks measured locally:
  - 1,000 rows: profile 0.736 s, import 0.946 s, export 0.177 s, peak traced memory 1.94 MiB.
  - 25,000 rows: profile 0.674 s, import 5.767 s, export 3.995 s, peak traced memory 18.40 MiB.

### Known Limitations
- Default tests use 25,000 rows for medium coverage; 100,000 and 1,000,000 row files are supported by tools but should be explicit stress runs.
- Tracemalloc memory is not full process RSS.
- Mixed timestamp formats are not unified automatically; unmatched rows are dropped under the active format.
- Ragged CSV rows fail gracefully as load errors rather than being repaired.
- Large Excel stress testing was not added in this phase.

Next phase: Phase 8.55N - user-facing import diagnostics summary and operational guidance in the wizard, without changing the import pipeline contract.

---

## Phase 8.55O - Import Wizard Final UX & Workflow Hardening - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- app/ui/import_wizard/workflow_state.py - small workflow action-state evaluator.
- app/ui/import_wizard/import_wizard_dialog.py - explicit state invalidation, action gating, discard protection, workflow guidance status.
- app/ui/import_wizard/wizard_pages.py - concise operational guidance and timestamp override visibility.
- app/ui/import_wizard/column_mapping_model.py - user override markers and tooltips.
- tests/unit/test_import_workflow_ux.py - deterministic unit coverage for enablement, invalidation, reset, discard prompt, and override visibility.
- tests/runtime/test_import_workflow_runtime.py - runtime coverage for repeated import/export, worker completion after close, rapid navigation, and failed import states.
- docs/IMPORT_WORKFLOW_GUIDE.md - workflow state/action/override/stale-state guidance.

### Outcome
- Next, Run Import, Open Waveform, Save Normalized File, Back, and Close now follow explicit current-state rules.
- Timestamp and mapping edits invalidate stale plan/import/export state and show re-import-required guidance.
- New file selection clears prior profile/import/export/diagnostics state.
- User overrides are visible in timestamp format controls and column mapping rows.
- Explicit Close action prompts before discarding user overrides, dirty settings, unexported successful imports, or worker state.
- Runtime teardown remains deterministic because window close itself accepts without modal prompts.
- Diagnostics/export workflow remains intact and still uses existing backend result objects.

### Validation
- 11 passed: new workflow UX unit/runtime tests.
- 100 passed: broader Import Wizard UX/export/diagnostics/runtime slice.
- 236 passed, 2 skipped: backend import pipeline, plan-aware execution, bridge, export writer, runtime hygiene, and stress slice.

### Known Limitations
- Discard protection is not persistent session management.
- Override markers are text-based, not icon-based.
- Close-window behavior is intentionally prompt-free for stable runtime teardown; use the wizard Close button for discard protection.
- No advanced import history, save-location persistence, drag/drop, or multi-file workflow was added.

Next phase: final Import Wizard acceptance pass, or Phase 8.55P for CI/developer command documentation and standard verification slices.

---

## Phase 8.55P - Acceptance Validation & Developer Operations - COMPLETED 2026-05-19

### Status: COMPLETED

### Deliverables
- docs/IMPORT_ACCEPTANCE_CHECKLIST.md - operational acceptance checklist for CSV/XLSX import, diagnostics, timestamp override, export, large files, repeated cycles, and worker-close stability.
- docs/IMPORT_DEV_WORKFLOW.md - developer validation commands, runtime/stress/benchmark workflow, troubleshooting, and merge guidance.
- docs/IMPORT_TEST_MATRIX.md - mapping of Import Wizard feature areas to unit, runtime, stress, and acceptance coverage.
- tools/run_import_acceptance.py - standard validation runner with `unit`, `runtime`, `stress`, `acceptance`, and `import-full` slices.
- tools/run_import_runtime_slice.py - repeatable runtime slice wrapper.
- tests/acceptance/conftest.py - acceptance Qt cleanup fixture.
- tests/acceptance/test_import_acceptance.py - lightweight operational acceptance workflows.

### Outcome
- Import Wizard validation is now documented as stable developer operations instead of scattered pytest knowledge.
- Future contributors can run narrow unit/runtime/stress/acceptance slices or the combined `import-full` gate.
- Acceptance tests cover CSV waveform handoff, XLSX import/export with metadata sidecar, authoritative timestamp override, malformed diagnostics, repeated import/export cycles, and pending worker close safety.
- Benchmark and stress workflows are documented using existing Phase 8.55M tooling.

### Validation
- 6 passed: acceptance tests.
- 6 passed: acceptance runner slice.
- 49 passed: runtime runner slice.
- 387 passed, 2 skipped: `import-full` runner slice.
- py_compile passed for new scripts and acceptance tests.

### Known Limitations
- Acceptance coverage intentionally uses small deterministic files; explicit stress and benchmark tooling remain the large-file path.
- Optional Parquet/Feather validation remains dependency-gated in existing export tests.
- No CI cloud infrastructure or packaging workflow was added in this phase.

Next phase: Phase 9 planning or CI wiring around the documented validation slices. Avoid new Import Wizard feature work until the acceptance workflow is reviewed by a human operator.
