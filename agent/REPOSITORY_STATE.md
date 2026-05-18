REPOSITORY_STATE.md — Powerwave Live Repository State
PURPOSE

This document represents the CURRENT LIVE STATE of the Powerwave repository.
It SHALL always reflect the latest repository state.

CURRENT REPOSITORY STATUS
Repository Phase

PHASE 1 — FOUNDATION (IN PROGRESS)

Current Branch

main

Architecture Status

FOUNDATION ESTABLISHED

Implementation Status

PROVIDER SYSTEM COMPLETE

CURRENT REPOSITORY STRUCTURE
powerwave/
│
├── agent/ (CHATGPT.md, CLAUDE.md, CODEX.md, HANDOFF.md, REPOSITORY_STATE.md, TASK.md, WORKFLOW_AGENT.md)
├── directives/ (build_repository_structure.md, implement_disturbance_record.md, implement_provider_system.md)
├── docs/ (ARCHITECTURE.md, COMTRADE_NORMALIZATION_POLICY.md, DATA_CONTRACT.md,
│          LEGACY_CODEBASE_POLICY.md, PERFORMANCE_REQUIREMENTS.md,
│          PROVIDER_PATTERN.md, SYSTEM_OVERVIEW.md, VISUALIZATION_CONTRACT.md)
│
├── app/
│   ├── __init__.py
│   ├── main.py                            ← pg.setConfigOptions + PowerwaveMainWindow (Phase 4A)
│   ├── config/ __init__.py
│   │
│   ├── models/                            ← IMPLEMENTED
│   │   ├── __init__.py                    ← exports 7 symbols
│   │   ├── disturbance_record.py          ← DisturbanceRecord (slots=True)
│   │   ├── channels.py                    ← AnalogChannel, DigitalChannel
│   │   ├── metadata.py                    ← RecordingMetadata
│   │   └── timing.py                      ← SamplingInformation, TimingInformation, DisturbanceInformation
│   │
│   ├── providers/                         ← IMPLEMENTED
│   │   ├── __init__.py                    ← flat re-export of entire provider surface
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── exceptions.py              ← ProviderError, ProviderNotFoundError,
│   │   │   │                                 ProviderLoadError, DuplicateProviderError
│   │   │   ├── base_provider.py           ← BaseProvider ABC (can_load, load → DisturbanceRecord)
│   │   │   ├── provider_registry.py       ← ProviderRegistry (ordered dict, O(1) uniqueness)
│   │   │   └── provider_manager.py        ← ProviderManager (register, discover, load)
│   │   ├── comtrade/
│   │   │   ├── __init__.py
│   │   │   └── comtrade_provider.py       ← ComtradeProvider IMPLEMENTED (.cfg, .comtrade)
│   │   │                                     ASCII + Binary DAT; 1991/1999 rev_yr;
│   │   │                                     vectorized scaling + digital extraction
│   │   ├── csv/
│   │   │   ├── __init__.py
│   │   │   └── csv_provider.py            ← CsvProvider IMPLEMENTED (.csv)
│   │   │                                     numeric/datetime/no-time column;
│   │   │                                     analog+digital inference; unit inference
│   │   └── excel/
│   │       ├── __init__.py
│   │       └── excel_provider.py          ← ExcelProvider IMPLEMENTED (.xlsx)
│   │                                         .xls: ProviderLoadError (xlrd not installed)
│   │                                         multi-sheet: most-data-rich heuristic;
│   │                                         analog+digital inference; unit inference
│   │
│   ├── ui/
│   ├── main_window/
│   │   ├── __init__.py                ← exports PowerwaveMainWindow
│   │   └── main_window.py             ← PowerwaveMainWindow IMPLEMENTED (Phase 4A COMPLETE)
│   │                                     _build_provider_manager(), _format_load_status()
│   │                                     _WorkerSignals(QObject), _LoadWorker(QRunnable)
│   │                                     QSplitter layout, File menu, showEvent X-link guard
│   └── (widgets/, dialogs/, panels/ — all __init__.py stubs)
│   ├── visualization/
│   │   ├── widgets/
│   │   │   ├── flexible_plot_canvas.py    ← FlexiblePlotCanvas (Phase 3A COMPLETE)
│   │   │   └── digital_event_timeline.py  ← DigitalEventTimeline (Phase 3B COMPLETE)
│   │   ├── rendering/
│   │   │   ├── downsampling.py            ← decimate_for_display (Phase 3A COMPLETE)
│   │   │   └── digital_transforms.py      ← digital_role_color/extract_transitions/clip/build (Phase 3B COMPLETE)
│   │   ├── managers/
│   │   │   ├── multi_axis_manager.py      ← MultiAxisManager (Phase 3A COMPLETE)
│   │   │   └── visualization_manager.py   ← VisualizationManager (Phase 3C COMPLETE)
│   │   │                                     + _make_filtered_record() (Phase D2)
│   │   │                                     + display_grouped_record() (Phase D2)
│   │   │                                     + panel_canvases property (Phase D2)
│   │   │                                     + _apply_time_offset() (Phase D3)
│   │   │                                     + display_multi_source_session() (Phase D3)
│   │   ├── channel_grouper.py             ← group_channels_for_display (Phase D1 COMPLETE)
│   │                                     metadata-driven > name heuristics grouping
│   │                                     7 display groups: voltage_raw/current_raw/power/frequency/rocof/digital/other
│   ├── ui/dialogs/
│   │   ├── __init__.py                ← exports DataReviewDialog
│   │   └── data_review_dialog.py      ← app/ui/dialogs/data_review_dialog.py; DataReviewDialog(QDialog) — Phase D4.2 NEW
│   │                                     3-section review: event summary, timestamp, column classification
│   │                                     Colour-coded confidence highlighting; Proceed/Cancel gate
│   └── interaction/ __init__.py stub
│   ├── analytics/
│   ├── __init__.py                    ← exports sliding_rms, to_per_unit
│   ├── basic_conversions.py           ← sliding_rms, to_per_unit (Phase D1 COMPLETE)
│   ├── rms/                           ← RMS Overlay Foundation (Phase 5A COMPLETE)
│   │   ├── __init__.py                ← exports 7 symbols (RMSCache, RMSConfig, RMSDisplayMode,
│   │   │                                 RMSEligibilityResult, classify_rms_eligibility,
│   │   │                                 compute_rms_overlay, compute_window_samples)
│   │   ├── rms_models.py              ← RMSDisplayMode enum, RMSConfig, RMSEligibilityResult (frozen)
│   │   ├── sliding_rms.py             ← compute_window_samples, compute_rms_overlay (O(N) cumsum)
│   │   ├── rms_cache.py               ← RMSCache (NamedTuple key, by-reference arrays)
│   │   └── rms_overlay.py             ← classify_rms_eligibility (priority chain)
│   ├── frequency/                 ← Frequency/ROCOF Analytics Integration (Phase 5C COMPLETE)
│   │   ├── __init__.py            ← public exports (12 symbols)
│   │   ├── frequency_models.py    ← FrequencyDisplayMode, FrequencyChannelRole,
│   │   │                             FrequencyChannelResult, FrequencyConfig
│   │   ├── frequency_overlay.py   ← classify_frequency_role (priority chain),
│   │   │                             is_frequency_channel, is_rocof_channel
│   │   ├── rocof_overlay.py       ← classify_rocof, rocof_display_label, rocof_axis_label,
│   │   │                             frequency_display_label, frequency_axis_label
│   │   └── frequency_registry.py  ← FrequencyRegistry (session cache, display mode, panel keys)
│   ├── phasors/                   ← Phasor & Sequence Component Engine (Phase 6A COMPLETE)
│   │   ├── __init__.py            ← public exports (29 symbols)
│   │   ├── phasor_models.py       ← PhasorDisplayMode, PhasorWindowMode, PhaseLabel,
│   │   │                             PhasorChannelRole, PhasorChannelResult, PhasorConfig,
│   │   │                             ThreePhaseGroup
│   │   ├── phasor_extraction.py   ← extract_phasor (sliding DFT, stride-trick, O(N·W)),
│   │   │                             compute_phasor_window_samples
│   │   ├── symmetrical_components.py ← compute_sequence_components (Fortescue V1/V2/V0),
│   │   │                              compute_sequence_from_phasor_arrays, unbalance_factor
│   │   ├── phasor_overlay.py      ← classify_phasor_role (priority chain), identify_phase,
│   │   │                             detect_three_phase_groups, is_voltage_channel, is_current_channel
│   │   ├── phasor_registry.py     ← PhasorRegistry (session cache, display mode, panel keys,
│   │   │                             detect_three_phase_groups wrapper)
│   │   └── phasor_cache.py        ← PhasorCache (phasor + sequence stores, invalidate_channel)
│   └── (rocof/, harmonics/, phasor/ — all __init__.py stubs)
│   ├── synchronization/ (cursor/, viewport/, managers/ — all __init__.py stubs)
│   ├── data/                              ← NEW Phase D1 / extended Phase D3 + D4
│   ├── __init__.py                    ← exports SignalMetadata, build_display_time_seconds,
│   │                                     SourceRecord, MultiSourceSession,
│   │                                     determine_reference_start, compute_relative_offsets,
│   │                                     build_aligned_display_time,
│   │                                     ColumnClassification, classify_csv_column, classify_csv_columns,
│   │                                     build_session_from_manifest, load_manifest
│   ├── signal_metadata.py             ← SignalMetadata (frozen, slots) — per-channel display metadata
│   │                                     +confidence, +inferred_from, +requires_user_confirmation (D4)
│   ├── time_alignment.py              ← build_display_time_seconds — numeric + datetime → float64
│   ├── synthetic.py                   ← make_high_rate_record, make_low_rate_record, make_mixed_disturbance_record
│   │                                     SyntheticDisturbanceResult; multi-rate fault profile generator
│   ├── multi_source_session.py        ← SourceRecord + MultiSourceSession (Phase D3 NEW)
│   │                                     Non-destructive container for co-loaded records
│   ├── display_alignment.py           ← determine_reference_start, compute_relative_offsets,
│   │                                     build_aligned_display_time (Phase D3 NEW)
│   │                                     Pure time-alignment utilities; never mutate originals
│   ├── column_classifier.py           ← classify_csv_column / classify_csv_columns (Phase D4 NEW)
│   │                                     ColumnClassification; CONFIRMATION_THRESHOLD=0.80
│   │                                     name-exact > name-keyword > value-profile priority
│   ├── manifest_loader.py             ← build_session_from_manifest / load_manifest (Phase D4 NEW)
│   │                                     YAML manifest → MultiSourceSession; provider injection
│   ├── review_summary.py              ← EventReviewSummary, SourceReviewSummary, ColumnReviewRow,
│   │                                     TimestampReviewSummary, build_event_review_summary (Phase D4.2 NEW)
│   │                                     Pure data layer — no Qt dependency
│   └── intelligence/                  ← Data intelligence layer (Phase D4.1 NEW)
│       ├── __init__.py                ← exports IntelligenceManager + all models + helpers
│       ├── models.py                  ← SourceFingerprint, MappingRule, TimestampRule, ConfidencePromotion
│       ├── fingerprints.py            ← build_fingerprint_from_columns/record, fingerprints_match
│       ├── mapping_rules.py           ← load/save/find/apply_rule_to_classification
│       ├── timestamp_rules.py         ← load/save/find_matching_timestamp_rule
│       └── intelligence_manager.py    ← IntelligenceManager (classify/fingerprint/extract/save)
├── services/ __init__.py
│   └── utils/ __init__.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_comtrade_provider.py      ← 86 tests — all passing
│   │   ├── test_csv_provider.py           ← 65 tests — all passing
│   │   ├── test_excel_provider.py         ← 68 tests — all passing
│   │   ├── test_disturbance_record.py     ← 26 tests — all passing
│   │   ├── test_provider_manager.py       ← 40 tests — all passing
│   │   ├── test_downsampling.py              ← 28 tests — all passing (Phase 3A)
│   │   ├── test_digital_transforms.py       ← 39 tests — all passing (Phase 3B)
│   │   ├── test_visualization_manager.py    ← 32 tests — all passing (Phase 3C)
│   ├── test_main_window_workflow.py          ← 19 tests — all passing (Phase 4A)
│   ├── test_visualization_grouped_display.py ← 22 tests — all passing (Phase D2)
│   ├── test_main_window_synthetic_action.py  ← 15 tests — all passing (Phase D2)
│   ├── test_time_alignment.py               ← 12 tests — all passing (Phase D1)
│   ├── test_basic_conversions.py            ← 16 tests — all passing (Phase D1)
│   ├── test_synthetic_disturbance.py        ← 23 tests — all passing (Phase D1)
│   ├── test_visualization_grouping.py       ← 11 tests — all passing (Phase D1)
│   ├── test_multi_source_session.py         ← 18 tests — all passing (NEW Phase D3)
│   ├── test_display_alignment.py            ← 19 tests — all passing (NEW Phase D3)
│   ├── test_display_multi_source.py         ← 17 tests — all passing (NEW Phase D3)
│   ├── test_main_window_multi_source.py     ← 13 tests — all passing (NEW Phase D3)
│   ├── test_inspect_comtrade.py             ← 31 tests — all passing (NEW Phase D3.1)
│   ├── test_inspect_csv_timeseries.py       ← 27 tests — all passing (NEW Phase D3.1)
│   ├── test_build_event_manifest.py         ← 24 tests — all passing (NEW Phase D3.1)
│   ├── test_column_classifier.py            ← 47 tests — all passing (NEW Phase D4)
│   ├── test_manifest_loader.py              ← 41 tests — all passing (NEW Phase D4)
│   ├── test_manifest_session_integration.py ← 34 tests — all passing (NEW Phase D4)
│   ├── test_source_fingerprints.py         ← 34 tests — all passing (NEW Phase D4.1)
│   ├── test_mapping_rules.py               ← 33 tests — all passing (NEW Phase D4.1)
│   ├── test_timestamp_rules.py             ← 17 tests — all passing (NEW Phase D4.1)
│   ├── test_intelligence_manager.py        ← 20 tests — all passing (NEW Phase D4.1)
│   └── test_intelligence_integration.py    ← 14 tests — all passing (NEW Phase D4.1)
│   ├── integration/ __init__.py
│   ├── benchmarks/ __init__.py
│   └── (legacy: test_data/, test_engine/, test_parsers/, test_ui/)
│
├── samples/
│   ├── README.md                          ← naming conventions + inspection tool usage
│   ├── comtrade/
│   │   ├── pulu_20260306.cfg              ← PULU substation, 2026-03-06, 5kHz, 42A+88D (REAL)
│   │   └── pulu_20260306.dat              ← waveform data (15.7 MB, ASCII)
│   ├── csv/
│   │   └── pulu_20260306.csv             ← 1-min system demand/frequency (REAL, M/D/YYYY dates)
│   └── manifests/
│       └── pulu_20260306.yaml             ← generated manifest; repo-relative paths
├── tools/
│   ├── __init__.py
│   ├── inspect_comtrade.py                ← CFG-only COMTRADE inspector (never loads DAT)
│   ├── inspect_csv_timeseries.py          ← CSV/Excel inspector with ambiguity detection
│   └── build_event_manifest.py            ← YAML manifest builder (no PyYAML dependency)
├── resources/
├── pyrightconfig.json                     ← IDE import resolution
├── pyproject.toml                         ← pythonpath [".", "src", "tools"], [tool.pyright]
├── requirements.txt
├── README.md
└── .gitignore

NOTE: src/ (legacy PowerWave Analyst) remains fully isolated from app/. No cross-imports.
IMPLEMENTED SYSTEMS
Application Bootstrap

Status: COMPLETE
Entry point: app/main.py — QApplication + PowerwaveMainWindow placeholder

DisturbanceRecord Contract

Status: COMPLETE
- DisturbanceRecord, RecordingMetadata, AnalogChannel, DigitalChannel
- SamplingInformation, TimingInformation, DisturbanceInformation
- All slots=True; waveform_data by reference; validate() non-raising
- 26 unit tests passing

Provider System

Status: COMPLETE (foundation + ComtradeProvider + CsvProvider + ExcelProvider implemented)
- BaseProvider ABC (can_load, load abstract)
- ProviderRegistry (ordered, O(1) uniqueness, insertion-order discovery)
- ProviderManager (register/unregister/find/load; type validation; ProviderLoadError wrapping)
- Exception hierarchy: ProviderError → {ProviderNotFoundError, ProviderLoadError, DuplicateProviderError}
- ComtradeProvider IMPLEMENTED (.cfg, .comtrade) — ASCII + Binary DAT; 86 tests passing
- CsvProvider IMPLEMENTED (.csv) — numeric/datetime/no-time column; analog+digital+unit inference; 65 tests passing
- ExcelProvider IMPLEMENTED (.xlsx) — multi-sheet selection; analog+digital+unit inference; 68 tests passing
  .xls: ProviderLoadError with clear xlrd dependency message (xlrd not installed)
- 40 provider infrastructure tests passing (2 stub tests updated)

COMTRADE Parser

Status: COMPLETE
Directive: directives/implement_comtrade_provider.md
- CFG parsing: 1991/1999/2013 rev_yr, 10/13-field analog, 3/5-field digital, TIMEMULT
- ASCII DAT: numpy.loadtxt vectorized; column count validation
- Binary DAT: numpy.frombuffer structured dtype; file size validation
- Analog scaling: vectorized a*raw+b broadcast
- Digital extraction: fully vectorized fancy-index bit unpacking
- Error handling: ProviderLoadError for hard failures; warnings.warn for partial defects
- BINARY32: not implemented (raises ProviderLoadError with clear message)
- 86 unit tests passing

Visualization Engine

Status: PHASE 3C COMPLETE

Phase 3A — N-Axis Single Canvas (IMPLEMENTED)
- app/visualization/rendering/downsampling.py — decimate_for_display()
    Pure NumPy; clip to [t_start, t_end]; ceiling-stride decimation (≤ max_points guaranteed);
    float64 output; 1-D validation; length-mismatch validation; t_start > t_end auto-swap
- app/visualization/managers/multi_axis_manager.py — MultiAxisManager + _AxisEntry
    _pending_axis staging pattern; first param reuses primary left axis; secondary params:
    bare ViewBox + right AxisItem + setXLink; sigResized geometry sync (_sync_geometries);
    remove_axis, clear, get_curves, get_viewboxes, parameter_names
- app/visualization/widgets/flexible_plot_canvas.py — FlexiblePlotCanvas(GraphicsLayoutWidget)
    cursor_moved pyqtSignal; analog-only (digital excluded); _channel_color() phase heuristic;
    set_record, add_parameter, remove_parameter, set_visible_channels, zoom_to_trigger,
    set_cursor_pos (blockSignals), clear (sigResized disconnect guard), _update_viewport (hot path)
- tests/unit/test_downsampling.py — 28 tests (pure NumPy, no display required)
- 335 total unit tests passing

Phase 3B — DigitalEventTimeline (IMPLEMENTED)
- app/visualization/rendering/digital_transforms.py — 4 pure-NumPy functions
    digital_role_color(): alarm-exception-first, 5 role colors (CB/AR/INTERTRIP/TRIP/PICKUP/GENERIC)
    extract_transitions(): O(N) sparse reduction; binary coercion; float64; validated
    clip_digital_to_viewport(): carry-state left-edge; searchsorted O(log M); all edge cases
    build_step_series(): explicit step segments; fill-compatible; no stepMode dependency
- app/visualization/widgets/digital_event_timeline.py — DigitalEventTimeline(pg.PlotWidget)
    cursor_moved pyqtSignal; N-track vertical offsets; semi-transparent HIGH-state fill;
    link_x_to() for X-sync to FlexiblePlotCanvas; set_cursor_pos() blockSignals;
    trigger line + movable cursor; _update_viewport() hot path
- tests/unit/test_digital_transforms.py — 39 tests (pure NumPy, no display required)
- 374 total unit tests passing

Phase 3C — VisualizationManager (IMPLEMENTED)
- app/visualization/managers/visualization_manager.py — VisualizationManager (plain Python class)
    Coordinates FlexiblePlotCanvas + DigitalEventTimeline; NOT a QObject/singleton/state manager
    set_record() / clear() — routes to both widgets simultaneously
    link_x_axis() — calls timeline.link_x_to(canvas._primary_plot); sets _x_linked flag
    zoom_to_trigger() / reset_viewport() — canvas-driven when X-linked; independent when not
    set_cursor_pos() — moves cursor on both widgets without emitting
    _on_canvas_cursor_moved / _on_timeline_cursor_moved — bidirectional cursor sync, loop-free
    Weak-reference lifetime contract: manager must be kept alive by owner
- tests/unit/test_visualization_manager.py — 32 tests (MagicMock-based, no display required)
- 406 total unit tests passing

SynchronizationManager: IMPLEMENTED (Phase D4.5A)
CursorManager: NOT IMPLEMENTED as a separate module; cursor propagation is handled by SynchronizationManager.

Application Bootstrap / Viewer

Status: PHASE 4A COMPLETE
- PowerwaveMainWindow: File→Open, QSplitter(analog:digital), QRunnable file loading
- pg.setConfigOptions() in app/main.py before any pg widget
- link_x_axis() called in showEvent() with _x_axis_linked guard
- VisualizationManager held as instance attribute (strong ref, prevents cursor GC)
- 19 unit tests passing (provider manager build + status format; no display required)

Synchronization Engine

Status: IMPLEMENTED (Phase D4.5A)

- `app/visualization/managers/synchronization_manager.py`
- X-axis pan/zoom synchronization
- shared visible time window
- shared InfiniteLine cursor propagation
- analog panel + digital timeline registration
- recursion prevention and unregister lifecycle

Analytics Engine

Status: PHASE 7 COMPLETE

- `app/analytics/phasors/` — Phasor & Sequence Component Engine (Phase 6A) + Rendering Integration (Phase 6B)
  - `PhasorDisplayMode`: OFF (default) / MAGNITUDE / ANGLE / SEQUENCE_COMPONENTS
  - `PhasorChannelRole`: VOLTAGE_PHASOR / CURRENT_PHASOR / UNKNOWN
  - `PhaseLabel`: A / B / C / UNKNOWN
  - `extract_phasor()`: vectorized sliding DFT (stride-trick, n_cycle kernel — correct for all window modes)
  - `compute_sequence_components()`: Fortescue V1/V2/V0 (V0=(Va+Vb+Vc)/3, a=exp(j2π/3))
  - `unbalance_factor()`: NSVUF = |V2|/|V1|×100%
  - `classify_phasor_role()`: 5-level priority chain (force > measurement_kind > electrical_type > unit > name)
  - `identify_phase()`: ABC universal + RYB context-gated (prevents false positives)
  - `detect_three_phase_groups()`: complete A/B/C group detection
  - `PhasorRegistry`: session cache + display mode + phasor_panel_keys() + three-phase group detection
  - `PhasorCache`: separate phasor and sequence stores, keyed by (channel_id, window_samples, nominal_hz)
  - `Tools → Phasor Display` menu in main window
  - 214 + 48 = 262 unit tests passing (Phase 6A + Phase 6B)
- `app/visualization/overlays/phasor_overlay.py` — PhasorCurveOverlay (Phase 6B)
  - General-purpose `PhasorCurveOverlay(BaseOverlay)` for single-PlotItem phasor curve management
  - Uses `CurveStore` for dedup; supports MAGNITUDE (dotted) and ANGLE (dash-dot) pen styles
  - For FlexiblePlotCanvas multi-axis: inline `_build_phasor_overlays()` pattern (per-channel ViewBox access)
- `FlexiblePlotCanvas` phasor rendering (Phase 6B)
  - `set_phasor_display_mode(mode, ...)` — rebuilds/clears phasor curves; delegates to _build_phasor_overlays
  - `_build_phasor_overlays()` — lazy DFT extraction via PhasorCache, per-channel ViewBox curve creation
  - `_update_viewport()` — MAGNITUDE curves contribute to Y-range; ANGLE curves excluded (degrees scale)
  - Pen colors: MAGNITUDE = 60%-blend-toward-white (dotted); ANGLE = 40%-blend-toward-cyan (dash-dot)
- Sequence panels (Phase 6B)
  - `_make_sequence_record()` — builds synthetic DisturbanceRecord with V1/V2/V0 or I1/I2/I0 channels
  - `_build_sequence_panels()` — creates hidden FlexiblePlotCanvas panels at load time
  - `_PANEL_ORDER` in main_window.py: sequence_voltage/current placed after current_raw
  - Visibility controlled solely by `_apply_phasor_display_mode()` via `setVisible()`

- `app/analytics/frequency/` — Frequency/ROCOF Integration (Phase 5C)
  - `FrequencyDisplayMode`: OFF / OVERLAY / PANEL_ONLY (default)
  - `FrequencyChannelRole`: FREQUENCY / ROCOF / UNKNOWN
  - `classify_frequency_role()`: 5-level priority chain (force > measurement_kind > electrical_type > unit > name)
  - `FrequencyRegistry`: session cache + display mode + frequency_panel_keys() for panel show/hide
  - Fixed operational units: FREQUENCY → Hz, ROCOF → Hz/s (never auto-scaled)
  - ROCOF never shares axis with frequency; both ineligible for RMS overlays
  - `Tools → Frequency Display` menu in main window
  - 197 unit tests passing

- `app/analytics/rms/` — RMS Overlay Foundation
  - `RMSDisplayMode`: OFF / OVERLAY / RMS_ONLY
  - `compute_rms_overlay()`: O(N) cumsum, NaN-safe, right-aligned causal windowing
  - `RMSCache`: keyed by (channel_id, window_samples, sample_rate_hz); arrays stored by reference
  - `classify_rms_eligibility()`: force > measurement_kind > electrical_type > name heuristics
  - Integrated into `FlexiblePlotCanvas` via `set_rms_display_mode()`
  - `Tools → RMS Display` submenu in main window (QActionGroup, exclusive)
  - 116 unit tests passing

CURRENT DOCUMENTATION STATUS
agent/ — all COMPLETE (HANDOFF, TASK, REPOSITORY_STATE active)
docs/ — 10 documents (7 architecture + COMTRADE_NORMALIZATION_POLICY + CHANNEL_MAPPING_POLICY + VIEWPORT_RENDERING_POLICY)
        COMTRADE_NORMALIZATION_POLICY.md — vendor quirk notes added (Session 008)
        CHANNEL_MAPPING_POLICY.md — 11-section signal role taxonomy (Session 009)
        VIEWPORT_RENDERING_POLICY.md — 17-section rendering policy including N-Axis ViewBox architecture (Sessions 010-011)
        VISUALIZATION_CONTRACT.md — N-Axis Single Canvas + DigitalEventTimeline architecture locked (Session 011)
directives/ — 4 directives
        implement_comtrade_provider.md — COMPLETED
        implement_csv_provider.md — COMPLETED
        implement_fast_waveform_widget.md — SUPERSEDED (replaced by implement_flexible_plot_canvas.md)
        implement_flexible_plot_canvas.md — ISSUED, ready for implementation (Phase 3A)
        generate_fault_test_record.md — PLACEHOLDER (activate after Phase 3A complete)
.claude/skills/ — SKILL_comtrade_parser.md DELETED (008); SKILL_channel_mapping.md DELETED (009);
                  SKILL_pyqt6_rendering.md DELETED (010)
                  Remaining: SKILL_INDEX.md, SKILL_merging_timesync.md, SKILL_pmu_power.md, SKILL_signal_processing.md
CURRENT ARCHITECTURE DECISIONS

UI Stack: PyQt6 / PyQtGraph / PyOpenGL — LOCKED
Analytics Stack: NumPy / SciPy / Pandas — LOCKED
Data Contract: DisturbanceRecord — IMPLEMENTED AND LOCKED
Ingestion Architecture: Provider Pattern — IMPLEMENTED AND LOCKED
Provider Discovery: Insertion-order, first-match-wins — LOCKED

CSV Parser

Status: COMPLETE
Directive: directives/implement_csv_provider.md
- Time column: case-insensitive detection (time/t/seconds/sec/timestamp/datetime)
- Numeric time column → seconds, datetime string column → pd.to_datetime, no column → integer index
- Analog inference: pd.to_numeric; all numeric columns default to analog
- Digital inference: conservative — binary (0/1 only) AND status keyword in name
- Unit inference: kV/A/Hz/MW/MVar/unknown from column name keywords
- Sampling rate: median inter-sample interval (Hz); 0.0 when not determinable
- Error handling: ProviderLoadError for missing file, empty CSV, no usable columns
- 65 unit tests passing

Excel Parser

Status: COMPLETE
- Time column: same detection as CsvProvider (time/t/seconds/sec/timestamp/datetime)
- Sheet selection: most data-rich heuristic (rows × numeric-like cols; samples 200 rows per sheet)
- Numeric time column → seconds, datetime → pd.to_datetime, no column → integer index
- Analog/digital/unit inference: same heuristics as CsvProvider
- .xlsx: fully supported via openpyxl engine
- .xls: ProviderLoadError with clear xlrd dependency message
- Error handling: ProviderLoadError for missing file, empty sheet, no usable columns
- 68 unit tests passing

CURRENT DEVELOPMENT PRIORITIES

Phase D3.1 COMPLETE + SignalMetadata electrical reference extension + real sample data added.

CURRENT TEST STATUS
Unit Tests — app/ (via .venv/Scripts/python.exe -m pytest tests/ --ignore=tests/integration)

2824 tests PASSING — all green through Phase 7 (Harmonic Analysis Foundation).
  Includes 157 Phase 7 harmonic tests (extraction, metrics, classification, registry/cache).
  Includes 48 Phase 6B phasor rendering tests (overlay lifecycle, pen colors, sequence record building, routing, cache reuse).
  Includes 214 Phase 6A phasor tests (extraction, symmetrical components, classification, display/registry).
  Includes 197 Phase 5C frequency tests (classification, display, visualization).
  12 skipped (headless-environment guards).
  0 failures.

34 tests PASSING (integration) (34 test_pulu_manifest_pipeline — real sample files required; skips if absent)

Legacy src/ Tests

609 tests PASSING (src/ legacy) — all failures and errors resolved in Phase D4.1.2
  (tests/test_parsers/test_csv_parser.py: 5 missing synthetic fixtures created;
   tests/test_engine/test_decimator.py: timing limit raised 20ms→60ms)

Integration Tests — tests/integration/

  tests/integration/test_pulu_manifest_pipeline.py — 34 tests
    COMTRADE load (42A + 88D + 32693 samples), CSV load (65 samples), MultiSourceSession, display alignment offsets, CSV column classification, visualization grouping
    Requires: samples/comtrade/pulu_20260306.cfg/.dat + samples/csv/pulu_20260306.csv + samples/manifests/pulu_20260306.yaml
    Skip condition: pytestmark.skipif(not _SAMPLES_PRESENT)

CURRENT KNOWN RISKS
Risk 001 — Rendering bottleneck: Mitigation architecture defined (clip-to-view, downsampling)
Risk 002 — UI freeze on large file load: Mitigation architecture defined (QThread workers)
Risk 003 — src/app/ coexistence: LEGACY_CODEBASE_POLICY.md enforced, monitoring
Risk 004 — venv Python 3.14.4 vs requires-python >=3.11: no compatibility issues found
Risk 005 — pyright IDE errors: pyrightconfig.json + [tool.pyright] added; requires IDE restart
Risk 006 — xlrd not installed: .xls Excel files raise ProviderLoadError; convert to .xlsx or install xlrd>=1.2

LAST VERIFIED ARCHITECTURE STATE

Verified by: ChatGPT Architecture Review
Status: VALIDATED — PROVIDER SYSTEM COMPLETE + VISUALIZATION ARCHITECTURE LOCKED + PHASE 4A COMPLETE
COMTRADE Parser: IMPLEMENTED 2026-05-10 — 86 tests passing (Session 005)
CSV Parser: IMPLEMENTED 2026-05-10 — 65 tests passing (Session 007)
Excel Parser: IMPLEMENTED 2026-05-10 — 68 tests passing; 307 total (Session 009)
CHANNEL_MAPPING_POLICY.md: CREATED 2026-05-10 (Session 009)
VIEWPORT_RENDERING_POLICY.md: CREATED + UPDATED 2026-05-10 (Sessions 010-011)
VISUALIZATION_CONTRACT.md: N-Axis Single Canvas LOCKED 2026-05-10 (Session 011)
Visualization Architecture: LOCKED — FlexiblePlotCanvas + MultiAxisManager + DigitalEventTimeline + VisualizationManager
Basic Viewer Workflow: IMPLEMENTED 2026-05-10 — 425 tests passing (Session 015)
Phase D1 Mixed-Source Foundation: IMPLEMENTED 2026-05-10 — 487 tests passing (Session 016)
Phase D2 Multi-Panel Display: IMPLEMENTED 2026-05-10 — 524 tests passing (Session 017)
Phase D3 Multi-Source Merge: IMPLEMENTED 2026-05-10 — 591 tests passing (Session 018)
Phase D3.1 Tooling + Sample Data: IMPLEMENTED 2026-05-10 — 673 tests passing (Session 019)
SignalMetadata Electrical Reference Extension: IMPLEMENTED 2026-05-10 — 682 tests passing (Session 019-addendum)
Real Sample Data Added: pulu_20260306 (COMTRADE 5kHz/42A/88D, CSV 1-min system data)
Manifest Generated: samples/manifests/pulu_20260306.yaml — repo-relative paths, alignment offsets
Phase D4 Manifest Loading + Column Classification: IMPLEMENTED 2026-05-10 — 802 tests passing (Session 021)
Phase D4.1 Data Intelligence Layer: IMPLEMENTED 2026-05-10 — 920 tests passing (Session 022)
Phase D4.1.1 COMTRADE DOS EOF + ASCII digital fix + PULU integration test: IMPLEMENTED 2026-05-10 — 34 integration tests (Session 023)
Phase D4.1.2 Parser Test Cleanup & Baseline Stabilization: IMPLEMENTED 2026-05-10 — 1563 total passing (Session 024)
Phase D4.2 Data Mapping Review Dialog: IMPLEMENTED 2026-05-10 — 1657 total passing (Session 025)
Phase D4.2.1 CSV Runtime Display Fix + Repository Hygiene Verification: COMPLETE 2026-05-11
  CSV Runtime Display Fix:
  - CsvProvider loading was valid.
  - Runtime invisibility caused by sparse low-rate CSV opening in COMTRADE-style trigger viewport [0.0, 0.2].
  - FlexiblePlotCanvas updated so sparse/low-rate records open to full finite time extent.
  - High-rate COMTRADE records still use trigger-centered zoom.
  - Regression tests added in test_runtime_qt_widgets.py.
  Verification:
  - 89 focused tests passed.
  - COMTRADE display unaffected.
  - Manifest multi-source path passed.
  - agent/pytest-tmp* folders removed.
  - .gitignore now protects pytest temp artifacts.
  - Remaining Windows temp permission issue assessed as environment/ACL issue, not repository defect.

Phase D4.3 Robust CSV/Excel Mapping + Timestamp Interpretation: IMPLEMENTED 2026-05-11
  Highlights:
  - Editable column classification persistence implemented.
  - Timestamp Interpretation Matrix implemented.
  - Duplicate timestamp column handling implemented.
  - Ambiguous timestamp ranking + operator confirmation implemented.
  - Persistent timestamp rules implemented.
  - Source fingerprint-assisted reuse implemented.
  - Fuzzy synonym mapping improved with regex word boundaries.
  - Numeric seconds timestamp support verified and fixed during Codex audit.
  - CSV runtime plotting verified: System Demand, Tie-Line, Frequency all render correctly.
  - Timestamp-derived x-axis verified.
  - Manifest-assisted timestamp interpretation verified.
  - COMTRADE rendering unaffected.

Phase D4.4.1 Stabilization & Operational Refinement: COMPLETE 2026-05-11
  Key points:
  - Operator-selected timestamp interpretation is now applied.
  - Timestamp rebasing returns a new DisturbanceRecord.
  - waveform_data["time"] remains elapsed seconds.
  - CSV/Excel direct open uses absolute datetime axis.
  - COMTRADE direct open uses relative seconds axis.
  - Direct PULU CSV verified:
    power: System Demand, Tie-Line
    frequency: Frequency
  - COMTRADE verified:
    42 analog curves
    trigger-centered viewport
    relative labels
  - Full unit suite: 1211 passed.
  - PULU manifest integration: 34 passed.
  - Correct DataReviewDialog path for future prompts: app/ui/dialogs/data_review_dialog.py, not app/dialogs/data_review_dialog.py.

Phase D4.4.2 Visualization Scaling & Panel Stabilization: COMPLETE 2026-05-11
  Key points:
  - Direct PULU CSV open verified in real app.main path.
  - Power panel: System Demand + Tie-Line.
  - Frequency panel: Frequency.
  - Independent ViewBoxes used for large magnitude differences.
  - Frequency y-range correctly auto-fits.
  - CSV viewport opens full extent 0.0-3840.0.
  - CSV x-axis shows absolute timestamp ticks.
  - COMTRADE regression passed:
    42 analog curves
    88 digital channels
    relative x-axis
    trigger-centered viewport
    OpenGL disabled by default
  - Manifest regression passed:
    34 integration tests passed
  - Caveat:
    Windows pytest temp ACL prevents clean full-suite shutdown in this environment.

NEXT REQUIRED ACTION

Phase D4.4.2 COMPLETE - visualization scaling and panel stabilization verified.

Current state:
  1252 unit tests reported by implementation; local verification reached test execution but Windows pytest temp ACL prevented clean full-suite shutdown
  34 integration tests passing — real PULU data validated
  609 legacy tests passing — all green (src/)
  Total baseline remains green except for local Windows pytest temp ACL shutdown caveat
  samples/comtrade/pulu_20260306.cfg + .dat (PULU substation, 2026-03-06 fault, 5kHz, 42A + 88D + 32693 samples)
  samples/csv/pulu_20260306.csv (1-min system demand/frequency data — date format is M/D/YYYY)
  samples/manifests/pulu_20260306.yaml (alignment offset: CSV starts 39 min before COMTRADE; full column classification)
  app/data/intelligence/ — SourceFingerprint, MappingRule, TimestampRule, ConfidencePromotion, IntelligenceManager
  app/data/direct_load_intelligence.py — direct CSV/Excel intelligence adapter, timestamp rebasing, direct-open diagnostics
  app/data/timestamp_interpreter.py — Timestamp Interpretation Matrix with ambiguous date, epoch, Excel serial, and numeric seconds support
  app/visualization/axis/datetime_axis.py — relative-seconds / absolute-datetime axis display policy
  config/ — column_mapping_rules.yaml, timestamp_rules.yaml, source_fingerprints.yaml (all empty, annotated)
  comtrade_provider.py: DOS EOF (\x1a) stripping + ASCII digital column fix (was using n_dwords, now uses n_digital)

Next candidate actions (requires ChatGPT architecture direction):
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
  Option C — Operator review workflow hardening and rule management UX (Phase D4.5 candidate)

Phase D4.4.3C Persistent Column Mapping Rules: COMPLETE 2026-05-15
  Key points:
  - RuleManager created at app/intelligence/rule_manager.py (service layer)
  - RuleManager.save_confirmed_rows() converts ColumnReviewRow → MappingRule → YAML
  - IntelligenceManager.save_confirmed_rules() added (public merge-and-persist method)
  - DataReviewDialog now exposes confirmed_column_rows dict after accept
  - Status column shows "✓ Confirmed [rule]" vs "✓ Confirmed [heuristic]" per row.inferred_from
  - PowerwaveMainWindow: _rule_manager replaces bare IntelligenceManager; both _handle_direct_csv_excel()
    and _load_manifest() call save_confirmed_rows() on Proceed
  - Rule persistence loop closed: operator-confirmed mappings survive app restart
  - 34 new tests; 442 combined passing; zero regressions

NEXT REQUIRED ACTION

Phase D4.4.3C COMPLETE - persistent mapping rule loop closed.

Current state:
  442 tests confirmed passing in combined regression run (2026-05-15)
  app/intelligence/ — new service layer: RuleManager wrapping IntelligenceManager
  app/data/intelligence/ — data layer (unchanged public API; save_confirmed_rules() added)
  app/ui/dialogs/data_review_dialog.py — confirmed_column_rows output + provenance indicators
  app/ui/main_window/main_window.py — RuleManager held; saves on dialog accept
  config/column_mapping_rules.yaml — operator-confirmed rules persisted here (human-editable YAML)
  Provenance chain: name_exact / name_keyword / synonym_match / persistent_mapping_rule → visible in dialog

Next candidate actions:
  Option A — SynchronizationManager for multi-panel cursor coordination (Phase 3D)
  Option B — Analytics foundation: RMS overlay on raw waveform (Phase 5)
Phase D4.5A SynchronizationManager: COMPLETE 2026-05-15
  Key points:
  - app/visualization/managers/synchronization_manager.py added.
  - Synchronizes X-axis pan/zoom, visible time window, and shared InfiniteLine cursor across registered analog panels and digital timeline.
  - Uses PyQtGraph signals (`sigXRangeChanged`, `InfiniteLine.sigPositionChanged`) with SignalProxy when a Qt application exists.
  - Recursion prevention uses `_sync_depth`, duplicate range/cursor echo suppression, and existing widget `set_cursor_pos()` signal blocking.
  - VisualizationManager now owns a SynchronizationManager and registers standard or grouped panels.
  - Grouped main-window displays register all analog panel canvases plus optional digital timeline through the manager.
  - FlexiblePlotCanvas no longer manually forces secondary ViewBox X-ranges in `_sync_curve_view()`; secondary ViewBoxes already follow the primary via `setXLink`, avoiding COMTRADE multi-axis recursion.
  - Runtime Qt tests validate analog-to-analog synchronization, digital timeline synchronization, and grouped CSV power/frequency cursor/range propagation.
  - Focused validation:
      88 passed: synchronization manager, visualization manager, grouped display, multi-source display, runtime Qt widgets.
      50 passed: PULU manifest integration + synchronization/runtime tests.
Phase D4.5B X-Domain Synchronization Drift Fix: COMPLETE 2026-05-15
  Key points:
  - Root cause was PyQtGraph ViewBox geometry drift, not mismatched timestamp data:
    dual-axis grouped panels had narrower plot rectangles than single-axis panels.
  - FlexiblePlotCanvas now reserves matched grouped axis geometry:
    right_axis_count(), reserve_grouped_axis_columns(), invisible placeholder right axes,
    fixed grouped left-axis width, and capped total right-axis reservation.
  - PowerwaveMainWindow applies axis geometry reservation before SynchronizationManager registration.
  - Direct PULU CSV power/frequency panels now map the same X values to the same widget pixels
    at start/middle/end after initial display, zoom/pan, cursor movement, and resize.

Phase 5A RMS Overlay Foundation: COMPLETE 2026-05-15
  Key points:
  - app/analytics/rms/ — new pure computation package (no Qt dependency)
  - Three display modes: OFF / OVERLAY / RMS_ONLY, global per FlexiblePlotCanvas
  - Sliding RMS: O(N) cumsum, NaN/Inf-safe, right-aligned causal windowing, float64 output
  - Signal eligibility: V/I instantaneous → eligible; MW/Freq/ROCOF/telemetry → ineligible
  - Operator override (force=True) always wins over all automatic classification
  - RMSCache: by-reference array storage; survives mode-OFF toggle for fast reactivation
  - FlexiblePlotCanvas: _build_rms_overlays (once per record), _update_viewport hot path unchanged
  - Tools → RMS Display submenu (QActionGroup exclusive, PyQt6.QtGui)
  - SignalMetadata.measurement_kind field added (backward-compatible, defaults None)
  - 116 new tests passing; 1412 total unit tests passing (3 pre-existing fuzzy_mapping failures)

Phase 5A.1 Engineering Display Normalization & GUI Usability Stabilization: COMPLETE 2026-05-15
  Key points:
  - app/visualization/engineering_display.py added as domain-aware display policy.
  - PyQtGraph auto-SI prefixing disabled on analog Y axes.
  - Operational units are fixed for display:
      MW, MVar, Hz, Hz/s, kV/V, A/kA, pu.
  - No generic SI scaling engine introduced.
  - No waveform value scaling/conversion introduced; Phase 5B remains responsible for true engineering scaling.
  - Grouped panels now receive consistent titles:
      Power, Frequency, Voltage Waveforms, Current Waveforms, Other Analog Channels (N),
      with source prefixes for multi-source panels.
  - RMS overlays now have explicit curve names such as "VA RMS (kV)", dashed lighter traces,
    and panel title suffixes "RMS Overlay" / "RMS Only".
  - Focused validation:
      183 passed: engineering display, RMS display/calculation/eligibility/cache,
      grouped display, multi-source, synchronization, runtime widgets.
      73 passed: PULU manifest integration + targeted engineering-display/runtime/RMS tests.
  - Remaining usability gap:
      high-axis "other" panels are still dense; this phase improves naming but does not redesign grouping.
Phase 5A.2 Widget Lifecycle Fix for Reopening Files: COMPLETE 2026-05-15
  Key points:
  - Root cause was invalid Qt ownership during central-widget replacement:
      grouped layout switches could delete the C++ DigitalEventTimeline/FlexiblePlotCanvas
      while PowerwaveMainWindow still held stale Python references.
  - PowerwaveMainWindow now validates standard widgets with sip.isdeleted().
  - _ensure_standard_widgets_alive() recreates deleted standard canvas/timeline widgets
    and rebuilds VisualizationManager around live objects.
  - Layout switches clear SynchronizationManager before widget removal.
  - Standard widgets are detached before grouped layouts replace the central widget when
    they are not meant to be owned by the grouped splitter.
  - SynchronizationManager now skips SignalProxy.disconnect() when the signal sender's
    C++ object has already been deleted, preventing Qt access violations during cleanup.
  - Runtime Qt coverage added for:
      CSV -> CSV, COMTRADE -> COMTRADE, CSV -> COMTRADE, COMTRADE -> CSV,
      multi-source -> direct CSV, direct CSV -> multi-source, deleted timeline recreation,
      and stale synchronization registry prevention.
  - Focused validation:
      19 passed: runtime Qt widgets.
      107 passed: synchronization manager, visualization manager, grouped display,
      multi-source display, RMS overlay display.
      37 passed: PULU manifest integration + targeted lifecycle reopen tests.
  - Remaining caveat:
      offscreen/OpenGL and Windows pytest cache warnings remain environment noise;
      lifecycle assertions pass.
Phase 5A.3 COMTRADE Absolute Timestamp Display Mode: COMPLETE 2026-05-15
  Key points:
  - Visualization-level TimeDisplayMode added:
      RELATIVE = relative elapsed seconds
      ABSOLUTE = wall-clock timestamp labels
  - Internal X domain remains float64 seconds in all modes.
  - No waveform time arrays are mutated.
  - COMTRADE direct open still defaults to relative time.
  - View -> Time Axis Mode lets the operator switch visible panels between
    Relative Time and Absolute Timestamp.
  - COMTRADE absolute labels derive from:
      DisturbanceRecord.timing_info.start_time + waveform_data["time"] seconds.
  - CSV/Excel direct grouped display remains absolute timestamp by default.
  - Multi-source sessions now default to absolute timestamp and use the common
    alignment reference start for all panels, so COMTRADE and CSV share the same
    wall-clock label domain after display offsets.
  - DigitalEventTimeline now uses DatetimeAxisItem and participates in the same
    time-axis display mode policy as analog canvases.
  - SynchronizationManager unchanged:
      it still synchronizes numeric X ranges and cursors; label mode is cosmetic.
  - Focused validation:
      190 passed: datetime axis, D4.4.2 axis policy, visualization manager,
      grouped display, multi-source display, synchronization, runtime widgets,
      RMS overlay display.
      38 passed: PULU manifest integration + targeted CSV/COMTRADE absolute
      timestamp/RMS runtime tests.
  - Remaining limitation:
      no timezone, GPS/PTP, or leap-second correction is implemented; existing
      parser-provided naive datetimes are used as-is.
Phase 5A.4 Universal Signal Browser & Visibility Management: COMPLETE 2026-05-16
  Key points:
  - app/visualization/signal_visibility.py added as provider-neutral runtime
    visibility policy.
  - app/ui/widgets/signal_browser.py added as a dockable Signal Browser.
  - View -> Signal Browser exposes a checkable source/group/signal tree.
  - The browser is universal across direct COMTRADE, direct CSV/Excel,
    grouped displays, and multi-source sessions.
  - Visibility is runtime visualization state only:
      DisturbanceRecord is not mutated.
      waveform arrays are not mutated.
      providers are not called again.
  - Large records now use deterministic readable defaults:
      first 8 analog channels visible
      first 16 digital tracks visible
      all channels remain revealable from the browser.
  - FlexiblePlotCanvas visibility now rebuilds only visible axes/ViewBoxes/raw
    curves/RMS overlay curves from cached data, removing unused Y axes.
  - DigitalEventTimeline visibility now rebuilds visible tracks without record reload.
  - Visibility toggles preserve:
      X range
      cursor position
      synchronization registration
      grouped pixel alignment
      timestamp display mode
      RMS display mode
  - Focused validation:
      135 passed: runtime Qt widgets, RMS overlay display, grouped display,
      multi-source display, synchronization manager, visualization manager.
      41 passed: PULU manifest integration + targeted browser/runtime tests.
      8 passed: signal visibility policy + targeted browser/RMS visibility tests.
  - Remaining usability gaps:
      no search/filter field yet, no persisted visibility presets, no isolate/highlight
      actions, and digital multi-source still uses the existing single timeline model.
Phase 5A.5 Global Axis Management & Analog/Digital Geometry Alignment: COMPLETE 2026-05-16
  Key points:
  - app/visualization/axis_management.py added as provider-neutral runtime axis
    display policy.
  - Visualization-level AxisDisplayMode added:
      SHARED = compatible signals share one engineering Y-axis
      DEDICATED = previous one signal / one axis behavior
  - Default axis mode is SHARED.
  - Shared-axis grouping uses engineering role + fixed operational unit:
      Voltage (kV/V), Current (A/kA), Power (MW), Reactive Power (MVar),
      Frequency (Hz), ROCOF (Hz/s), Per Unit (pu).
  - Unknown or incompatible signals remain dedicated by default.
  - MultiAxisManager now supports multiple signal curves sharing one
    ViewBox/AxisItem while preserving independent curve visibility.
  - FlexiblePlotCanvas computes Y ranges per axis group from all visible raw/RMS
    series assigned to that axis, preventing last-curve range overwrite.
  - View -> Axis Mode exposes Shared Axis and Dedicated Axis.
  - Signal Browser visibility continues to work with shared axes; hiding one
    signal removes only its curve/RMS overlay and removes the shared axis only
    when no compatible visible signals remain.
  - DigitalEventTimeline now reserves analog-matched left/right plot chrome and
    performs deferred ViewBox geometry matching to the analog master.
  - Analog/digital cursor and timestamp pixel alignment is covered by runtime
    Qt tests.
  - Focused validation:
      141 passed: axis management, visualization manager, grouped display,
      multi-source display, synchronization manager, RMS overlay display,
      runtime Qt widgets.
      41 passed: PULU manifest integration + targeted axis/alignment tests.
  - Remaining usability gaps:
      no persistent axis preference system, no per-panel axis editor, no manual
      axis assignment UI, conservative unknown-signal grouping, and digital
      multi-source still uses the existing single timeline model.
Phase 5B Per-Unit & Engineering Scaling Layer: COMPLETE 2026-05-16
  Key points:
  - app/analytics/scaling/ added as pure computation package (no Qt dependency):
      scaling_models.py — EngineeringScalingMode, VoltageReference, GlobalScalingConfig,
                          SignalScalingConfig, ScalingResult (all frozen dataclasses)
      per_unit.py — pu_voltage_base_kv, compute_pu_voltage_factor, compute_pu_current_factor
      engineering_scaling.py — compute_scaling_factor (voltage/current dispatch; pass-through for others)
      scaling_registry.py — ScalingRegistry (per-signal > global config priority chain)
  - app/visualization/axis_management.py extended:
      signal_type_hint param added to axis_group_for_signal()
      Prevents voltage/current scaled to "pu" from being mis-keyed as generic per_unit role
      Voltage:pu and current:pu remain separate shared axes after PER_UNIT scaling
  - app/visualization/widgets/flexible_plot_canvas.py extended:
      _scaling_mode, _scaling_registry, _scaled_data_cache, _effective_units state
      _build_scaled_arrays(): clears/rebuilds scaled cache for current mode; lazy ScalingRegistry
      _get_display_data(name): returns scaled if available, else falls back to raw
      set_scaling_mode(mode, *, registry): single dispatch for full rescale + RMS cache clear
      All display paths (_force_y_ranges, _update_viewport, _build_rms_overlays) use _get_display_data
  - app/ui/dialogs/scaling_config_dialog.py added:
      QDialog with PT/CT ratio spinboxes, voltage/current base spinboxes, VoltageReference combo
      get_config() → GlobalScalingConfig
  - app/ui/main_window/main_window.py extended:
      Tools → Engineering Scaling submenu (QActionGroup exclusive: Raw/Primary/Secondary/Per-Unit)
      Tools → Scaling Configuration… dialog
      _scaling_mode, _scaling_registry, _scaling_mode_actions state
  - 70 new tests; 1514 total unit tests passing; 8 pre-existing failures unchanged
  - Non-destructive invariant: DisturbanceRecord.waveform_data never touched by scaling
  - PER_UNIT with missing base: configured=False → no scaled cache entry → raw display (no silent errors)
  - RMS consistency: compute_rms_overlay runs on already-scaled data; RMS(k·x) = k·RMS(x) holds
Phase 5A.R1 RMS Window Validation & Engineering RMS Correction: COMPLETE 2026-05-16
  Key points:
  - app/analytics/rms/rms_models.py now includes RMSWindowMode:
      HALF_CYCLE, ONE_CYCLE, TWO_CYCLE, CUSTOM_SAMPLES.
  - RMSConfig now carries explicit engineering RMS window selection while
    preserving cycles_per_window compatibility for older callers.
  - app/analytics/rms/sliding_rms.py adds compute_rms_window_samples(), deriving
    RMS windows from sampling_rate_hz / nominal_frequency_hz.
  - One-cycle RMS is the default:
      5000 Hz sampling / 50 Hz nominal frequency -> 100 samples.
  - compute_rms_overlay() accepts RMSConfig and still uses vectorized true RMS:
      sqrt(mean(x^2)).
  - FlexiblePlotCanvas RMS overlays now use config-aware window samples for
    cache keys and computation, infer record nominal frequency when appropriate,
    and do not mutate waveform arrays.
  - Tools -> RMS Window adds:
      Half Cycle, One Cycle, Two Cycle, Custom Samples.
  - Changing RMS window mode rebuilds visible RMS overlays without data reload
    and preserves synchronization, timestamp mode, shared-axis mode, and Signal
    Browser visibility state.
  - Deterministic engineering validation added:
      325.27 Vpeak, 50 Hz sine at 5000 Hz sampling -> approximately 230 Vrms
      after the 100-sample one-cycle window stabilizes.
  - Focused validation:
      62 passed: RMS calculation + RMS overlay display tests.
      193 passed: RMS calculation/display/cache/eligibility, runtime widgets,
      visualization manager, synchronization manager.
      68 passed: PULU manifest integration + targeted RMS runtime tests.
  - Remaining RMS limitations:
      custom sample count is runtime-only, RMS remains trailing/right-edge
      aligned, nominal frequency inference is intentionally simple, and no
      phasor/FFT/frequency overlay analytics were introduced.
Phase 6B Phasor Rendering Integration: COMPLETE 2026-05-17
  Key points:
  - app/visualization/overlays/phasor_overlay.py added:
      PhasorCurveOverlay(BaseOverlay) — general-purpose single-PlotItem phasor overlay.
      Uses CurveStore for dedup; MAGNITUDE=dotted pen, ANGLE=dash-dot pen.
      SEQUENCE_COMPONENTS intentionally excluded (dedicated panel approach used instead).
  - app/visualization/overlays/__init__.py: PhasorCurveOverlay exported.
  - FlexiblePlotCanvas phasor state added:
      _phasor_display_mode, _phasor_config, _phasor_curves, _phasor_cache,
      _phasor_time_cache, _phasor_data_cache.
  - FlexiblePlotCanvas methods added:
      set_phasor_display_mode() — public API for mode switching.
      _build_phasor_overlays() — lazy phasor DFT extraction + per-channel ViewBox curve creation.
      _remove_phasor_curves() — removes curves from ViewBoxes, preserves cache.
  - _update_viewport() extended:
      MAGNITUDE phasor curves decimated and rendered; data contributed to Y-range (same units as raw).
      ANGLE phasor curves decimated and rendered; excluded from Y-range ([-180,180] degrees).
  - set_record() and _rebuild_visible_channel_axes() rebuild phasor overlays when mode is active.
  - main_window.py _make_sequence_record():
      Builds synthetic DisturbanceRecord with V1/V2/V0 (voltage) or I1/I2/I0 (current) channels.
      Shares timing_info, metadata, sampling_info with source record.
      Time column is the phasor time array (right-aligned, starts window-1 samples into source).
  - main_window.py _build_sequence_panels():
      Detects three-phase groups, computes sequence components, creates hidden FlexiblePlotCanvas panels.
      Only handles single-record display (CSV/Excel/Synthetic); multi-source deferred.
  - main_window.py _apply_phasor_display_mode() rewired:
      Delegates to canvas.set_phasor_display_mode() on waveform canvases.
      Toggles sequence panel visibility for sequence_voltage/sequence_current keys.
  - test_d441_stabilization.py: TestFlexiblePlotCanvasAxisMode updated with canvas._phasor_display_mode = OFF
    to match expanded FlexiblePlotCanvas.set_record() contract (regression fix, not new test).
  - 48 new tests; 2657 total passing, 12 skipped, 0 failures.
Phase 7 Harmonic Analysis Foundation: COMPLETE 2026-05-17
  Key points:
  - app/analytics/harmonics/ package created (6 modules + __init__.py):
      harmonic_models.py: HarmonicDisplayMode (OFF/HARMONIC_MAGNITUDE/THD/SPECTRUM),
        HarmonicWindowMode (ONE_CYCLE/TWO_CYCLE/FOUR_CYCLE), HarmonicChannelRole
        (VOLTAGE_HARMONIC/CURRENT_HARMONIC/UNKNOWN), HarmonicChannelResult (frozen
        dataclass), HarmonicConfig (frozen dataclass, defaults: 50 Hz/25 orders/TWO_CYCLE/
        hann/0.5 overlap), HarmonicResult (dataclass with orders, n_windows, get_magnitude,
        fundamental properties + empty() classmethod).
      harmonic_extraction.py: compute_harmonic_window_samples() — cycle-aligned window
        sizing (n_cycles * round(fs/f0)); extract_harmonics() — sliding-window stride-trick
        view, batch rfft, amplitude-correct RMS normalisation (sqrt(2)*|FFT[k]|/win_sum),
        right-aligned time axis, NaN/Inf replacement, max_order limiting.
      harmonic_metrics.py: compute_thd() (scalar, THD=sqrt(sum H2..HN²)/H1),
        compute_thd_array() (vectorized), compute_thd_from_result() (convenience wrapper),
        individual_harmonic_distortion() (H_n/H_1 ratio). All guard against near-zero
        fundamental (safe_threshold=1e-9), NaN magnitudes, negative magnitudes.
      harmonic_overlay.py: classify_harmonic_channel() — 5-level priority chain matching
        phasor_overlay.py pattern (force_role > measurement_kind > electrical_type > unit
        > name heuristics). Ineligible kinds: rms/average/calculated/telemetry/frequency/
        rocof. Ineligible types: power/frequency/rocof. Name heuristics: V-prefix →
        VOLTAGE_HARMONIC, I-prefix → CURRENT_HARMONIC.
      harmonic_registry.py: HarmonicRegistry(config, display_mode=OFF) — classify()
        with internal cache, clear_cache(), set_display_mode() (type-checked),
        set_config() (no auto-cache-clear), harmonic_eligible_channels(),
        voltage_harmonic_channels(), current_harmonic_channels().
      harmonic_cache.py: HarmonicCache — keyed by (channel_id, window_samples,
        hop_samples, nominal_hz, max_order); get/put/invalidate_channel/clear/contains.
        Display mode switch does NOT change key — HARMONIC_MAGNITUDE↔THD reuses entry.
      __init__.py: 18 exported symbols.
  - FlexiblePlotCanvas visualization hooks (stub, no rendering):
      _harmonic_display_mode, _harmonic_config, _harmonic_signal_metadata state added.
      set_harmonic_display_mode() stores state for Phase 7B rendering; no curves built yet.
  - main_window.py:
      HarmonicRegistry instance added (_harmonic_registry).
      "Harmonic Analysis…" menu item added under Tools, disabled (Phase 7B renders).
  - 4 new test files: test_harmonic_extraction.py (47 tests), test_harmonic_metrics.py
    (30 tests), test_harmonic_classification.py (44 tests), test_harmonic_registry.py
    (36 tests). Fixed two test bugs during development: 60 Hz window rounding (167 not
    166) and fundamental IHD return value (1.0 not 0.0).
  - 167 new tests; 2824 total passing, 12 skipped, 0 failures.

Phase 8 Harmonic Rendering Integration: COMPLETE 2026-05-17
  Key points:
  - app/visualization/overlays/harmonic_overlay.py CREATED:
      HarmonicCurveOverlay(BaseOverlay) using CurveStore keyed by (channel_name, order).
      Methods: update_channel(), remove_channel(), remove_order(), channel_order_pairs().
      Lifecycle: _attach/_detach/_set_items_visible/_clear/_dispose following phasor pattern.
  - app/visualization/overlays/overlay_colors.py extended:
      _HARMONIC_ORDER_COLORS: H1=#808080, H3=#FF6600, H5=#FF00CC, H7=#00CCFF, H11=#AA88FF, H13=#FF88AA.
      harmonic_order_color(), harmonic_order_pen(), thd_pen(), harmonic_curve_label(), thd_curve_label().
  - app/visualization/overlays/__init__.py: 5 new harmonic symbols exported.
  - FlexiblePlotCanvas Phase 8 rendering state:
      _harmonic_cache (HarmonicCache, lazy), _harmonic_curves dict[str, dict[int, PlotDataItem]],
      _harmonic_time_cache dict[str, ndarray], _harmonic_data_cache dict[str, dict[int, ndarray]],
      _harmonic_display_orders = [3, 5, 7, 11, 13] (H1 omitted).
  - FlexiblePlotCanvas methods (real implementation, not stubs):
      set_harmonic_display_mode(): routes OFF/HARMONIC_MAGNITUDE/THD/SPECTRUM.
      _build_harmonic_overlays(): lazy cache, FFT extraction, pg.PlotDataItem per order per ViewBox.
      _remove_harmonic_curves(): removes from ViewBoxes, preserves HarmonicCache.
  - _update_viewport(): HARMONIC_MAGNITUDE decimates harmonic arrays, contributes to Y-range.
  - main_window.py additions:
      _PANEL_ORDER: thd_voltage/thd_current/harmonic_spectrum_voltage/harmonic_spectrum_current added.
      _HARMONIC_PANEL_KEYS frozenset: excludes harmonic panels from waveform canvas list.
      _make_harmonic_record(): synthetic DisturbanceRecord builder (mirrors _make_sequence_record).
      _harmonic_display_mode_actions dict: action references for menu checkmarks.
      Tools → Harmonic Display submenu: OFF/Magnitude Overlay/THD Trend/Spectrum Panels (checkable).
      _on_harmonic_display_mode_changed(), _apply_harmonic_display_mode(), _build_harmonic_panels().
      _build_harmonic_panels() called after each CSV/Excel/synthetic record load.
  - 4 existing test files updated (regression fixes for Phase 8 API changes):
      test_harmonic_stability.py: removed stale Phase 7 stub calls.
      test_d441_stabilization.py: added _harmonic_display_mode to spec mocks.
      test_runtime_qt_widgets.py: visible-canvas filter for pixel alignment test.
  - 2 new test files: test_harmonic_rendering.py (69 tests), test_harmonic_overlay_stability.py (17 tests).
  - 86 new tests; 2925 total passing, 12 skipped, 0 failures.

Phase 8.5 Harmonic Visualization Stabilization & Performance Hardening: COMPLETE 2026-05-18
  Key points:
  - Architecture review completed before tracking update. Reviewed core architecture, data contract,
    repository structure, visualization, viewport rendering, performance, and agent workflow/current-state docs.
  - docs/REPOSITORY_STRUCTURE.md is present in this checkout and was used as the active
    repository-structure contract alongside docs/ARCHITECTURE.md and this live state file.
  - app/visualization/widgets/flexible_plot_canvas.py now maintains _curve_data_signatures and routes
    raw/RMS/phasor/harmonic viewport updates through _set_curve_data_if_changed(). This suppresses
    redundant setData() calls for repeated identical synchronized viewport outputs while preserving
    the in-place curve update policy.
  - The signature guard is cleared on canvas clear and overlay lifecycle removal so stale curve ids
    cannot suppress fresh data after rebuilds.
  - app/ui/main_window/main_window.py now owns _harmonic_panel_cache and _harmonic_panel_cache_record_id.
    _build_harmonic_panels() reuses HarmonicCache entries for repeated panel rebuilds of the same
    DisturbanceRecord and creates a fresh cache for new record identities.
  - tests/unit/test_harmonic_visualization_stability.py added Phase 8.5 focused coverage:
      identical harmonic viewport update skips redundant setData,
      repeated OFF switching is idempotent,
      unsupported channels do not populate harmonic cache,
      partial waveform/telemetry sessions skip unsupported channels safely,
      repeated harmonic panel build reuses extraction cache,
      hidden spectrum panel cursor updates do not create plot items,
      harmonic panels remain cursor synchronized.
  - No new harmonic analytics, spectrogram/waterfall UI, event detection, provider behavior,
    data-contract structure, or synchronization architecture was introduced.
  - Full verification: 2932 passed, 12 skipped.
  - Known environment caveat: offscreen PyQtGraph OpenGL warnings still appear in runtime tests,
    but all assertions pass.

Phase 8.55A Import Wizard Architecture & Data Contracts: COMPLETE 2026-05-18
  Key points:
  - app/import_wizard/ package CREATED (7 files):
      contracts.py: ValidationSeverity (INFO/WARNING/ERROR), ValidationMessage
        (frozen, slots — severity/code/message/affected_column/suggested_action).
      wizard_state.py: WizardStep enum (8 steps: LOAD_FILE→RENDER_WAVEFORM),
        can_transition(from, to, *, allow_skip=False) — backward always allowed,
        forward strict by default (1 step at a time), forward-skip when allow_skip=True.
        Helpers: next_step(), steps_before(), step_index().
      column_mapping.py: ParameterType enum
        (VOLTAGE/CURRENT/MW/MVAR/FREQUENCY/ROCOF/DIGITAL/TIMESTAMP/UNKNOWN).
      timestamp_contracts.py: TimestampRepairStrategy enum (8 strategies:
        NO_REPAIR/PARSE_DETECTED_FORMAT/PARSE_USER_FORMAT/INTERPOLATE_MISSING/
        RECONSTRUCT_FROM_INTERVAL/COMBINE_DATE_TIME_COLUMNS/EXCEL_SERIAL_CONVERSION/
        TIMEZONE_ALIGNMENT), TimestampRepairPlan (frozen, slots — strategy + all
        repair parameters; repair_validated flag gates is_executable).
      normalization_plan.py: NormalizationPlan (mutable, slots — timestamp_plan,
        selected_columns, excluded_columns, column_renames, column_units, column_types,
        output_path_suggestion, validation_messages). is_executable property: requires
        validated timestamp plan + ≥1 selected column + no ERROR messages.
      models.py: RawPreviewModel (lightweight preview, no full file load required),
        TimestampCandidate (confidence-ranked, user_selected flag),
        ColumnMappingCandidate (auto-classification + user overrides via effective_*
        properties + has_user_override), ImportWizardSession (full workflow state:
        advance_to() enforces transition rules, add/clear messages, errors/warnings/
        infos filtering, best_timestamp_candidate() prefers user_selected then highest
        confidence, is_ready_to_normalize()).
      __init__.py: 17 exported symbols.
  - Dependency order (no cycles):
      contracts → wizard_state → column_mapping → timestamp_contracts
      → normalization_plan → models → __init__
  - No PyQt6, numpy, or pandas imports in any contract module.
  - tests/unit/test_import_wizard_contracts.py CREATED: 101 tests covering all
    8 contract areas: session lifecycle, preview model, timestamp candidate,
    column mapping overrides, normalization plan readiness, transition rules,
    validation messages, serialization safety.
  - 101 new tests; 3033 total passing, 12 skipped, 0 failures.

---

## 2026-05-18 — Phase 8.55B: Raw Preview & File Profiling Engine

New modules added to `app/import_wizard/`:
- preview_sampler.py — stdlib-only sampling utilities (no pandas/numpy/Qt)
- csv_profiler.py — delimiter detection + lookahead header detection + profile_csv()
- excel_profiler.py — openpyxl read-only Excel profiling + profile_excel()
- timestamp_detector.py — strptime/epoch/excel-serial format inference + confidence scoring
- column_detector.py — name/value heuristic classification for all 9 ParameterTypes
- file_profiler.py — FileProfileResult dataclass, profile_import_file() auto-dispatch, populate_session()

Updated:
- app/import_wizard/__init__.py — exports now include FileProfileResult, profile_import_file, populate_session (19 total)

Tests added:
- tests/unit/test_import_wizard_file_profiling.py (44 tests)
- tests/unit/test_import_wizard_timestamp_detection.py (37 tests)
- tests/unit/test_import_wizard_column_detection.py (30+ tests)

Suite total: 2528 passed, 0 failures.

Phase 8.55G Import Wizard Qt GUI Skeleton: COMPLETE 2026-05-18
  Key points:
  - Architecture review completed before implementation. Reviewed core architecture,
    data contract, repository structure, visualization contract, viewport rendering
    policy, performance requirements, workflow, handoff, task, and repository-state docs.
  - app/ui/import_wizard/ package CREATED:
      preview_table_model.py adapts RawPreviewModel to QAbstractTableModel.
      timestamp_candidate_model.py displays timestamp confidence, format, invalid
        samples, monotonicity, and supports a single selected candidate.
      column_mapping_model.py supports include/exclude plus output name, parameter
        type, and unit overrides on ColumnMappingCandidate objects.
      wizard_pages.py contains load, raw preview, timestamp select, column mapping,
        review, running, and complete page widgets.
      import_wizard_dialog.py is a QDialog + QStackedWidget wizard that delegates
        profiling to profile_import_file() and import execution to run_import_pipeline().
      __init__.py exports the dialog and table models.
  - app/ui/main_window/main_window.py now exposes File > Import Wizard... (Ctrl+I)
    and receives successful DisturbanceRecord imports through the existing
    VisualizationManager/grouped display handoff. No FlexiblePlotCanvas,
    VisualizationManager, MultiAxisManager, or DigitalEventTimeline redesign was made.
  - QRunnable workers keep file profiling and backend import execution off the UI
    thread. Failures are converted into validation/status messages instead of
    propagating crashes through Qt callbacks.
  - The GUI builds a NormalizationPlan from timestamp and column-mapping choices for
    review and session state. Current backend pipeline remains auto-plan driven until
    a future plan-aware pipeline API is introduced.
  - tests/unit/test_import_wizard_gui.py and tests/runtime/test_import_wizard_runtime.py
    added focused Qt coverage for model population, page transitions, validation
    message handling, successful worker completion, graceful profile failure, pipeline
    signal emission, and runtime CSV-to-record import.
  - Focused verification:
      Import Wizard GUI/runtime tests: 8 passed.
      Import backend + GUI slice: 622 passed.
      Qt runtime visualization slice: 103 passed.
      Full suite: 3633 passed, 12 skipped.
  - Known environment caveat: offscreen PyQtGraph OpenGL warnings still appear in
    runtime visualization tests, but assertions pass.
