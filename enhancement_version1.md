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
| Two-cursor measurement (Δt, ΔY, frequency, RMS, mean, peak, energy) | 🔴 Phase 1 |
| Single-cursor live value readout per channel | 🔴 Phase 3 |
| Smart snapping (zero crossings, peaks, cycle boundaries, trigger) | 🔴 Phase 1 |

### Tier 2 — Data Intelligence (Quality & Classification)
What the app knows about the data before the engineer does anything.

| Capability | Status |
|---|---|
| Data quality fingerprint on load (sample rate gaps, clipping, noise floor) | 🔴 Phase 4 |
| Recording classification (fault capture, trend, steady-state) | 🔴 Phase 4 |
| Timestamp integrity report | ✅ Implemented (Import Wizard) |

### Tier 3 — Event Intelligence (Automatic Detection)
Finding events in the waveform without manual searching.

| Capability | Status |
|---|---|
| Fault / disturbance detection (voltage dip, overcurrent spike) | 🔴 Phase 5 |
| Event timeline markers on the X-axis | 🔴 Phase 5 |
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
| N-axis per channel (MultiAxisManager) | ✅ | ❌ | S1 |
| RMS overlay | ✅ | ❌ | S2 |
| Phasor magnitude/angle overlay | ✅ | ❌ | S3 |
| Harmonic magnitude overlay | ✅ | ❌ | S4 |
| Engineering scaling (kV, pu, etc.) | ✅ | ❌ | S5 |
| Two-cursor measurement | 🔴 Phase 1 | ❌ | S6 (after Phase 1) |
| Trigger marker line | ✅ | ❌ | S1 |
| Absolute timestamp axis | ✅ | ✅ | Done |
| Signal browser integration | ✅ | ✅ | Done |
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

### Phase 1 — Two-Cursor Measurement Tool 🔴 [CURRENT]

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
- [ ] `MeasurementEngine` implemented
- [ ] `MeasurementPanel` widget created
- [ ] `FlexiblePlotCanvas` cursor B + measurement mode
- [ ] `MainWindow` wiring (menu + dock)
- [ ] Committed and tested

---

### Phase 2 — Event Detection + Timeline Markers 🔴

**Goal:** Automatically detect disturbance events in loaded waveforms and mark them on the time axis.

**Detection targets:**
- Voltage dip (>10% drop below nominal, >1 cycle duration)
- Overcurrent spike (>120% of peak nominal)
- Rapid frequency deviation (ROCOF > threshold)
- Zero-sequence current injection (ground fault indicator)

**UI:** Event marker lines on the bottom X-axis (coloured, labelled). Click marker → jump to event + auto-fit Y range.

**Progress:**
- [ ] Event detection engine
- [ ] Timeline marker rendering
- [ ] Event list panel (sortable)

---

### Phase 3 — Single-Cursor Live Value Readout 🔴

**Goal:** As the yellow cursor moves, show a floating readout bubble (or status bar) with the interpolated Y value for every visible channel.

**Readout format:** `VA: 11.2 kV  IA: 487 A  freq: 49.98 Hz`

**Progress:**
- [ ] Cursor value interpolation
- [ ] Floating readout widget

---

### Phase 4 — Data Quality Fingerprint 🔴

**Goal:** On file load, silently compute a quality fingerprint and surface a compact status indicator (green/amber/red badge in the signal browser).

**Checks:**
- Sample rate consistency (gaps > 2×median interval)
- Clipping detection (>3 consecutive samples at ADC rail)
- Noise floor vs signal peak (SNR estimate)
- DC offset magnitude
- Missing/NaN sample percentage

**Progress:**
- [ ] Fingerprint computation
- [ ] Signal browser badge
- [ ] Quality report panel (expandable)

---

### Phase 5 — Fault Characterisation 🔴

**Goal:** When a voltage dip or overcurrent is detected, run symmetrical components analysis and classify fault type.

**Classification:** A-g, B-g, C-g, AB, BC, CA, ABg, BCg, CAg, ABC (3-phase)

**Uses existing:** `app/analytics/phasors/symmetrical_components.py`

**Progress:**
- [ ] Fault classifier
- [ ] Fault summary panel

---

### Phase 6 — Protection Timing Extraction 🔴

**Goal:** Identify protection relay response timing from digital channels and waveform events.

**Extracted events:**
- Fault inception (voltage dip start)
- Relay pickup (digital channel transition)
- Trip command (digital channel)
- Circuit breaker open (current drops to zero)
- Arc extinction / reclosure

**Output:** Timing table — pickup time, trip time, fault clearing time, reclosure interval.

**Progress:**
- [ ] Digital channel mapping to protection events
- [ ] Timing extraction engine
- [ ] Timing report panel

---

### Phase 7 — Cross-Source Correlation 🔴

**Goal:** In multi-source sessions, detect if two sources captured the same event and suggest auto-alignment.

**Method:** Cross-correlation of voltage/current signatures; trigger-time comparison.

**Progress:**
- [ ] Cross-correlation engine
- [ ] Alignment suggestion UI

---

### Phase 8 — Contextual Analytics Suggestions 🔴

**Goal:** Based on what the app detects, suggest relevant analytics actions.

**Examples:**
- Voltage dip detected → "Run RMS analysis? [Yes]"
- High THD detected → "Open Harmonic Spectrum view? [Yes]"
- Multiple sources with close trigger times → "Auto-align sources? [Yes]"

**Progress:**
- [ ] Suggestion engine
- [ ] Non-intrusive notification bar

---

## Design Constraints

1. **No silent assumptions** — every inferred value is labelled with its source (e.g., "nominal frequency: 50 Hz from record metadata").
2. **No UI blocking** — all heavy computation runs off the Qt main thread (QThread or ThreadPoolExecutor). The canvas never freezes.
3. **Cache results** — computed analytics (RMS, phasors, harmonics, measurements) are cached per-record so switching display modes is instant.
4. **Additive, not destructive** — new intelligence features are overlaid on the waveform, never replacing it. The raw waveform is always visible unless the user explicitly chooses RMS-only mode.
5. **Feature flags off by default** — new analysis overlays start OFF. The user enables them; the app remembers the preference per-session.

---

## Progress Summary (Phase 1 in flight)

```
Phase 1 — Two-Cursor Measurement    [░░░░░░░░░░] 0%   ← current
Phase 2 — Event Detection           [░░░░░░░░░░] 0%
Phase 3 — Live Value Readout        [░░░░░░░░░░] 0%
Phase 4 — Data Quality Fingerprint  [░░░░░░░░░░] 0%
Phase 5 — Fault Characterisation    [░░░░░░░░░░] 0%
Phase 6 — Protection Timing         [░░░░░░░░░░] 0%
Phase 7 — Cross-Source Correlation  [░░░░░░░░░░] 0%
Phase 8 — Contextual Suggestions    [░░░░░░░░░░] 0%

Session Canvas Migration
S1 — N-axis + trigger               [░░░░░░░░░░] 0%
S2–S5 — Overlays                    [░░░░░░░░░░] 0%
S6–S9 — Measurement + File menu     [░░░░░░░░░░] 0%
```
