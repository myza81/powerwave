# PowerWave Enhancement Version 1 — Intelligence & UX Roadmap

**Baseline commit:** a1ffbf7  
**Phase start date:** 2026-05-22  
**Status:** 🟡 In Progress

---

## Vision

Transform PowerWave from a waveform *viewer* into a waveform *analyst* — a tool that surfaces critical information without requiring the engineer to know where to look. Intelligence must be transparent, proactive but non-intrusive, and never make silent assumptions about data.

### Intelligence Principles

1. **Transparent and accurate** — every computed value is traceable. If a computation is approximate, say so.
2. **Proactive but non-intrusive** — the app surfaces relevant insights automatically but never interrupts the workflow. Suggestions appear; decisions stay with the user.
3. **Progressive disclosure** — simple analysis works out of the box; deeper capabilities are discoverable, not buried.

---

## 4-Tier Intelligence Framework

### Tier 1 — Measurement Intelligence (Interactive Tools)
Direct interaction with the waveform: the engineer places markers and the app computes.

| Capability | Status |
|---|---|
| Two-cursor measurement (Δt, ΔY, frequency, RMS, mean, peak, energy) | ✅ Phase 1 |
| Single-cursor live value readout per channel | ✅ Phase 3 |
| Smart snapping (zero crossings, peaks, cycle boundaries, trigger) | 🔴 Phase 1 |

### Tier 2 — Data Intelligence (Quality & Classification)
What the app knows about the data before the engineer does anything.

| Capability | Status |
|---|---|
| Data quality fingerprint on load (sample rate gaps, clipping, noise floor) | ✅ Phase 4 |
| Recording classification (fault capture, trend, steady-state) | 🔴 Phase 4 |
| Timestamp integrity report | ✅ Implemented (Import Wizard) |

### Tier 3 — Event Intelligence (Automatic Detection)
Finding events in the waveform without manual searching.

| Capability | Status |
|---|---|
| Fault / disturbance detection (voltage dip, overcurrent spike) | ✅ Phase 2 |
| Event timeline markers on the X-axis | ✅ Phase 2 |
| Protection timing extraction (pickup → trip → clear) | 🔴 Phase 6 |
| Digital channel event synchronisation | 🟡 Partial (DigitalEventTimeline) |

### Tier 4 — Analytical Intelligence (Pattern & Correlation)
Higher-order reasoning across signals and sources.

| Capability | Status |
|---|---|
| Fault characterisation via symmetrical components (A-g, AB, 3-phase) | 🔴 Phase 7 |
| Cross-source time correlation (multi-source session) | 🔴 Phase 8 |
| Contextual analytics suggestions (e.g., "Voltage dip detected — run RMS analysis?") | 🔴 Phase 9 |
| Harmonic distortion classification (THD trend, dominant orders) | ✅ Overlay implemented |

---

## Session Canvas Migration Plan

The long-term UX goal is a **single unified canvas** — open a file, get the session canvas with full analytics; "Add Source" to extend to multi-source comparison. The session canvas must reach feature parity with the single-source `FlexiblePlotCanvas` before the File menu is simplified.

### Feature Parity Checklist

| Feature | FlexiblePlotCanvas | SessionCanvas | Migration Phase |
|---|---|---|---|
| N-axis per channel (MultiAxisManager) | ✅ | ✅ | S1 |
| RMS overlay | ✅ | ✅ | S2 |
| Phasor magnitude/angle overlay | ✅ | ✅ | S3 |
| Harmonic magnitude overlay | ✅ | ✅ | S4 |
| Engineering scaling (kV, pu, etc.) | ✅ | ✅ | S5 |
| Two-cursor measurement | ✅ | ✅ | S6 |
| Trigger marker line | ✅ | ✅ | S1 |
| Absolute timestamp axis | ✅ | ✅ | Done |
| Signal browser integration | ✅ | ✅ | S7 |
| Colour management in legend | ✅ | ✅ | Done |
| Y-axis side (left/right auto-assign) | ✅ | ✅ | Done |

### Session Canvas Migration Phases (S-series)

- **S1** — Embed `MultiAxisManager` + trigger markers into `SessionCanvasWidget`
- **S2** — RMS overlay per source channel
- **S3** — Phasor overlay per source channel
- **S4** — Harmonic overlay per source channel
- **S5** — Engineering scaling per source channel
- **S6** — Two-cursor measurement in session canvas (after Phase 1 in single-source)
- **S7** — Signal browser integration for session canvas
- **S8** — Trigger markers per source
- **S9** — File menu simplification (abolish "Open", single entry point)

---

## Enhancement Phases

### Phase 1 — Two-Cursor Measurement Tool ✅

**Priority:** Highest — highest daily-use value for every waveform engineer.

**Goal:** Allow the engineer to place two time markers (A and B) on any waveform panel and instantly see time difference, amplitude difference, frequency, RMS, mean, and peak between them.

#### Architecture

```
FlexiblePlotCanvas
 ├── _cursor_a (pg.InfiniteLine, yellow, existing _cursor)
 ├── _cursor_b (pg.InfiniteLine, cyan, new)
 └── measurement_mode: bool

MeasurementEngine (pure, no Qt)
 └── compute(t_a, t_b, time, data_by_channel, sample_rate, nominal_hz)
     → MeasurementResult

MeasurementPanel (QDockWidget)
 └── update(result: MeasurementResult)
     → renders table of per-channel stats
```

#### MeasurementResult Fields (per channel)

| Metric | Formula | Unit |
|---|---|---|
| Δt | \|t_b − t_a\| | ms |
| Δt cycles | Δt × f_nominal | cycles |
| Frequency from period | 1 / Δt | Hz |
| ΔY | Y(t_b) − Y(t_a) | channel unit |
| RMS (segment) | √(mean(y²)) between markers | channel unit |
| Mean (segment) | mean(y) between markers | channel unit |
| Peak | max(\|y\|) between markers | channel unit |
| Energy (V×I pair) | Σ(V×I×dt) | W·s |

#### Snapping Targets (Phase 1 — basic)

- Zero crossings (nearest sample where sign changes)
- Peaks (local maximum within 3-sample window)
- Trigger marker position

#### UI

- **Enable:** View menu → "Measurement Mode" (Ctrl+M)
- **Cursor A:** existing yellow cursor (draggable)
- **Cursor B:** new cyan cursor (draggable, only visible in measurement mode)
- **Panel:** collapsible dock at bottom or right showing per-channel table
- **Header row:** Δt | cycles | frequency | — (not per channel)
- **Channel rows:** ΔY | RMS | Mean | Peak | Energy

#### Files to Create/Modify

| File | Action |
|---|---|
| `app/visualization/interaction/measurement_engine.py` | Create — pure computation |
| `app/ui/widgets/measurement_panel.py` | Create — QWidget panel |
| `app/visualization/widgets/flexible_plot_canvas.py` | Modify — add cursor B, measurement mode |
| `app/ui/main_window/main_window.py` | Modify — View menu, dock widget |

**Progress:**
- [x] Enhancement document created
- [x] `MeasurementEngine` implemented (`app/visualization/interaction/measurement_engine.py`)
- [x] `MeasurementPanel` widget created (`app/ui/widgets/measurement_panel.py`)
- [x] `FlexiblePlotCanvas` cursor B + measurement mode
- [x] `MainWindow` wiring (View → Measurement Mode Ctrl+M, bottom dock)
- [x] Committed: `1c46a3d`

---

### Phase 2 — Event Detection + Timeline Markers ✅

**Goal:** Automatically detect disturbance events in loaded waveforms and mark them on the time axis.

**Detection targets:**
- Voltage dip (>10% drop below nominal, >0.5 cycle duration)
- Voltage swell (>110% of nominal, >0.5 cycle duration)
- Overcurrent spike (>120% of pre-fault peak, >0.5 cycle duration)
- Frequency deviation (>0.5 Hz from nominal, >0.5 cycle duration)
- Zero-sequence current injection (3I0 / neutral / earth current channels)

**UI:** Color-coded DashDot InfiniteLine markers on each waveform panel at event start (+ DotLine end markers for events >20 ms). Detected Events dock (View menu) lists all events — click to jump canvas to event time.

**Progress:**
- [x] Event detection engine (`app/analytics/events/event_detector.py`)
- [x] Channel role classifier (voltage/current/frequency/zero-sequence by name+unit)
- [x] Pre-fault baseline estimation from trigger time
- [x] Timeline marker rendering in `FlexiblePlotCanvas`
- [x] Event list panel — sortable, click-to-jump (`app/ui/widgets/event_list_panel.py`)
- [x] MainWindow wiring — all load paths, auto-show dock on events found
- [x] Committed: `39a8f59`

---

### Phase 3 — Single-Cursor Live Value Readout ✅ [COMPLETE — commit 7b38ab8]

**Goal:** As the yellow cursor moves, show a floating readout bubble (or status bar) with the interpolated Y value for every visible channel.

**Readout format:** `VA: 11.2 kV  IA: 487 A  freq: 49.98 Hz`

**Progress:**
- [x] Cursor value interpolation (`_compute_cursor_values` via np.searchsorted + linear interp)
- [x] `CursorReadoutBar` widget — 36px dock strip, scrollable channel chips in waveform colours
- [x] `cursor_values_changed` signal on `FlexiblePlotCanvas`
- [x] Wired into `MainWindow` at all three load paths; View menu toggle (Ctrl+R)

---

### Phase 4 — Data Quality Fingerprint ✅ [COMPLETE — commit df923b9]

**Goal:** On file load, silently compute a quality fingerprint and surface a compact status indicator (green/amber/red badge in the signal browser).

**Checks:**
- Sample rate consistency (gaps > 2×median interval)
- Clipping detection (>3 consecutive samples at ADC rail)
- Noise floor vs signal peak (SNR estimate via MAD)
- DC offset ratio |mean| / peak
- Missing/NaN sample percentage

**Progress:**
- [x] `compute_quality_fingerprint()` in `app/analytics/quality/quality_fingerprint.py`
- [x] Signal browser leaf items coloured green/amber/red with issue tooltips
- [x] `QualityReportPanel` dock — sortable per-channel table, auto-shows on WARN/ERROR
- [x] Wired at all three load paths; resets on layout switch

---

### Phase 5 — Fault Characterisation ✅ [COMPLETE — commit 848ac55]

**Goal:** When a voltage dip or overcurrent is detected, run symmetrical components analysis and classify fault type.

**Classification:** A-g, B-g, C-g, AB, BC, CA, AB-g, BC-g, CA-g, ABC, ABC-g

**Uses existing:** `app/analytics/phasors/symmetrical_components.py`

**Progress:**
- [x] `FaultType` enum + `FaultCharacterisation` dataclass
- [x] `identify_voltage_phase_channels()` — heuristic A/B/C name matching
- [x] `classify_fault_from_events()` — single-cycle DFT phasor at fault midpoint + Fortescue + per-phase depression
- [x] `FaultSummaryPanel` — compact two-row dock with phase circles, ground symbol, V₁/V₂/V₀ readout
- [x] Wired inside `_run_event_detection()`; auto-shows on classification

---

### Phase 6 — Protection Timing Extraction ✅ [COMPLETE — commit 17d5f98]

**Goal:** Identify protection relay response timing from digital channels and waveform events.

**Extracted events:**
- Fault inception (voltage dip start)
- Relay pickup (digital channel transition — pickup/start/86)
- Trip command (digital channel — trip/87/51/50/67/21)
- CB open (digital channel — cb/52/breaker/aux)
- Arc extinction (current RMS < 5% of prefault for ≥ 0.5 cycles)
- Reclosure (current recovery > 10% of prefault after clearing)

**Output:** Timing table — pickup delay, trip delay, clearing time, reclose interval (all in ms).

**Progress:**
- [x] Digital channel classification by keyword heuristics (PICKUP/TRIP/CB/RECLOSE)
- [x] `extract_protection_timing()` — transition detection + current extinction + recovery
- [x] `ProtectionTimingPanel` — summary chips + chronological events table with role colours
- [x] Wired at tail of `_run_event_detection()`; auto-shows when >1 milestone found

---

### Phase 7 — Cross-Source Correlation ✅ [COMPLETE — commit 87dbc27]

**Goal:** In multi-source sessions, detect if two sources captured the same event and suggest auto-alignment.

**Method:** FFT cross-correlation of voltage/current signatures over the temporal overlap window.

**Progress:**
- [x] `correlate_source_pair()` — resample to common grid → FFT xcorr → peak lag + confidence
- [x] `correlate_all_pairs()` — all unique pairs in a multi-source session
- [x] `CorrelationReportPanel` — pairwise table (Source A/B | Correlation | Lag | Confidence | Suggested Offset)
- [x] Runs after every multi-source load + after session "Auto Align All"
- [x] Auto-applies high-confidence (≥0.70) offsets to EventAnalysisSession with method='correlation'

---

### Phase 8 — Contextual Analytics Suggestions ✅ [COMPLETE — commit d1bf093]

**Goal:** Based on what the app detects, suggest relevant analytics actions.

**Examples:**
- Voltage dip detected → "Fault/dip detected — view RMS envelope? [Enable]"
- Fault characterised → "A-g fault — view symmetrical components? [Open]"
- High clearing time detected → "Clearing time 85.3 ms detected — view protection timing? [Open]"
- Multi-source correlation → "2 source pair(s) likely same event — view correlation? [Open]"
- Data quality errors → "Data quality errors found — review quality report? [Open]"

**Architecture:**
- `suggestion_engine.py` — pure `SuggestionContext` + `generate_suggestions()` returning prioritised `Suggestion` objects
- `suggestion_bar.py` — `SuggestionBar` QWidget, 44px scrollable chip strip; chips colour-coded by priority (amber=critical, blue=informational, grey=low); each chip has icon + text + [Act] + [×] dismiss
- MainWindow: top-area dock with hidden title bar; maps action_ids to analytics toggles; collapses when all chips dismissed

**Progress:**
- [x] `Suggestion` dataclass + `SuggestionContext` + `generate_suggestions()` in `app/analytics/suggestions/suggestion_engine.py`
- [x] Triggers: voltage dip→RMS, swell→RMS, freq→frequency, fault→phasors+fault panel, protection→timing panel, quality issues→quality panel, multi-source correlation→align
- [x] `SuggestionBar` chip strip widget with dismiss + action signals
- [x] MainWindow top-area dock, hidden title bar, `_build_suggestion_action_map()`
- [x] Analytics pipeline caches events/fault/protection/correlation; `_run_suggestions()` fires after `_run_quality_check()` and after cross-correlation completes

---

---

## File Menu Redesign (S9 — Gated on S1–S8 Feature Parity)

### Problem with the Current Menu

The current File menu presents three overlapping entry points that confuse the user about which one to use:

```
File
 ├── Open…                ← loads into FlexiblePlotCanvas (single-source)
 ├── Import Wizard…       ← CSV/Excel only, separate dialog
 ├── Multi-Source Viewer… ← loads into SessionCanvas
 ├── Open Event Manifest…
 └── Exit
```

The core issue: "Open" and "Multi-Source Viewer" both display waveforms, but they are
separate code paths leading to separate canvases with different feature sets. A user who
opens a file with "Open" and later wants to compare it with another recording has to start
over from "Multi-Source Viewer". There is no natural progression.

### Target State — Unified Entry Point

```
File
 ├── Open…                ← single entry point (see flow below)
 ├── Save Session as Manifest…
 ├── Open Event Manifest…
 ├── ─────────────────
 └── Exit
```

The "Import Wizard" and "Multi-Source Viewer" menu items are removed. Their functionality
is absorbed into the unified Open flow and the session canvas itself.

### Progressive Disclosure Flow

```
User clicks File → Open…
        │
        ▼
   File picker (all supported formats)
        │
        ├── COMTRADE (.cfg/.comtrade) ──────────────┐
        │                                            │
        └── CSV / Excel (.csv/.xlsx) ────────────────┤
                    │                                │
                    ▼                                │
          Import Wizard fires automatically          │
          (no menu item needed — auto-detect)        │
                    │                                │
                    └────────────────────────────────┘
                                 │
                                 ▼
                    Session Canvas opens with one source
                    (full analytics: RMS, phasors, harmonics,
                     measurement tool, engineering scaling)
                                 │
                          ┌──────┴──────┐
                     User wants         User is done
                    to compare          ↓
                         │         Single-source analysis
                         ▼         with all tools available
                  "Add Source" button
                  appears in Session Panel
                         │
                         ▼
                   File picker again
                         │
                         ▼
                   Multi-source session
                   (both sources in session canvas,
                    time-aligned, legend per source,
                    cross-source correlation available)
```

### Key Behaviours

| Behaviour | Detail |
|---|---|
| Import Wizard auto-fire | When user opens a `.csv` or `.xlsx` via Open, the Import Wizard dialog launches automatically. The user never has to know it's a separate flow. |
| Single-source = session with one source | The session canvas always hosts the waveform. No separate `FlexiblePlotCanvas` path. `FlexiblePlotCanvas` becomes an internal implementation detail only. |
| "Add Source" is always available | The Session Panel always shows an "Add Source" button. The user can go multi-source at any time without restarting. |
| Session manifest preserved | File → Save Session as Manifest saves the current session (one or many sources) for later reopening. |
| No data loss on Add Source | Adding a second source never reloads or disturbs the first source's analysis state (cursors, overlays, zoom position). |

### Before / After Comparison

| Scenario | Before | After |
|---|---|---|
| Open a COMTRADE file | File → Open | File → Open |
| Open a CSV file | File → Import Wizard | File → Open (wizard auto-fires) |
| Add a second recording | Restart via Multi-Source Viewer | Click "Add Source" in Session Panel |
| Quick look at one file | File → Open → FlexiblePlotCanvas | File → Open → Session Canvas (same simplicity) |
| Deep analysis (RMS + phasors + measurement) | Available in FlexiblePlotCanvas only | Available in Session Canvas (after S1–S5) |

### Why S9 Must Wait for S1–S8

Removing "Open" today would remove access to:

| Missing from Session Canvas | Covered by |
|---|---|
| N independent Y-axes per channel | S1 |
| RMS overlay | S2 |
| Phasor overlay | S3 |
| Harmonic magnitude overlay | S4 |
| Engineering scaling | S5 |
| Two-cursor measurement | S6 |
| Signal browser | S7 |
| Per-source trigger markers | S8 |

Until all of these are present in the session canvas, the single-source canvas must remain
accessible. S9 is the final step — flip the entry point and retire the old code path.

### Implementation Notes (for when S9 is actioned)

- Remove `_restore_standard_layout()` and `_rebuild_grouped_layout()` from `MainWindow`.
- Remove `FlexiblePlotCanvas` from the central widget stack (keep the class — used by session canvas panels internally via S1).
- Remove "Import Wizard…" and "Multi-Source Viewer…" from File menu.
- Rename "Open…" trigger to call `_on_open_session()` instead of `_open_file_dialog()`.
- `_on_open_session()` auto-fires Import Wizard when file suffix is `.csv`/`.xlsx`.
- The `DigitalEventTimeline` becomes a panel within the session canvas (not a separate splitter widget).

**Progress:**
- [ ] S1–S8 complete (prerequisite)
- [ ] File menu item changes
- [ ] Unified Open handler
- [ ] Import Wizard auto-fire integration
- [ ] FlexiblePlotCanvas removed from main layout
- [ ] Tested: COMTRADE, CSV, multi-source flows

---

## Design Constraints

1. **No silent assumptions** — every inferred value is labelled with its source (e.g., "nominal frequency: 50 Hz from record metadata").
2. **No UI blocking** — all heavy computation runs off the Qt main thread (QThread or ThreadPoolExecutor). The canvas never freezes.
3. **Cache results** — computed analytics (RMS, phasors, harmonics, measurements) are cached per-record so switching display modes is instant.
4. **Additive, not destructive** — new intelligence features are overlaid on the waveform, never replacing it. The raw waveform is always visible unless the user explicitly chooses RMS-only mode.
5. **Feature flags off by default** — new analysis overlays start OFF. The user enables them; the app remembers the preference per-session.

---

## Progress Summary

```
Intelligence Phases
Phase 1 — Two-Cursor Measurement    [██████████] 100% ✅ commit 1c46a3d
Phase 2 — Event Detection           [██████████] 100% ✅ commit 39a8f59
Phase 3 — Live Value Readout        [██████████] 100% ✅ commit 7b38ab8
Phase 4 — Data Quality Fingerprint  [██████████] 100% ✅ commit df923b9
Phase 5 — Fault Characterisation    [██████████] 100% ✅ commit 848ac55
Phase 6 — Protection Timing         [██████████] 100% ✅ commit 17d5f98
Phase 7 — Cross-Source Correlation  [██████████] 100% ✅ commit 87dbc27
Phase 8 — Contextual Suggestions    [██████████] 100% ✅ commit d1bf093

Session Canvas Migration (prerequisite for File menu redesign)
S1 — N-axis + trigger markers       [██████████] 100% ✅ commit 45d7c8c
S2 — RMS overlay                    [██████████] 100% ✅ commit db16f65
S3 — Phasor overlay                 [██████████] 100% ✅ commit 22eaf94
S4 — Harmonic overlay               [██████████] 100% ✅ commit bcef567
S5 — Engineering scaling            [██████████] 100% ✅ commit 5864ffa
S6 — Two-cursor measurement port    [██████████] 100% ✅ commit 45c6cde
S7 — Signal browser                 [██████████] 100% ✅ commit 1f6e7ba
S8 — Per-source trigger markers     [██████████] 100% ✅ implemented in S1

File Menu Redesign
S9 — Unified entry point            [░░░░░░░░░░]   0%  (gated on S1–S8)
```
