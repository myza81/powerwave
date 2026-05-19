# Phase 9 — Event Analysis Session: Multi-Source Time-Domain Analysis Platform

**Status:** Specification — PENDING (revisit after Phase 8 complete)
**Depends on:** Phase 8.55N complete (3815 tests passing)
**Authored:** 2026-05-19

---

## Background & Motivation

The app currently operates as a single-file waveform viewer with a strong import pre-processor.
The analyst's real workflow is different: a disturbance event produces multiple files simultaneously —
a COMTRADE from the relay, a CSV from the SCADA historian, an Excel export from the energy meter —
each describing the same 200 ms fault from a different vantage point, at a different sampling rate,
with a different clock source.

Phase 9 closes the gap between "load one file" and "analyse one event" by introducing a
first-class session model, interactive time alignment, a unified multi-source canvas, and
per-channel legend/colour control.

### Pain points being solved

| Pain Point | Solution in Phase 9 |
|---|---|
| COMTRADE + CSV cannot be combined on one canvas | Unified session canvas with all sources |
| Time sync between sources is manual/impossible | Per-source offset spinbox + auto-align |
| Different sampling rates clash | Resampling bridge (finest grid, linear interp) |
| No legend — can't tell which colour is which channel | Per-panel ChannelLegendWidget |
| Channel colours clash when two sources share a name | Multi-source saturation/hue strategy |
| Panel layout is fixed at load time | Merge/split panels at runtime |

---

## Core Design Principles

1. **Non-destructive.** `DisturbanceRecord.waveform_data` is never mutated. Time offsets,
   resampling, and display transforms are applied at render time only.
2. **Source-neutral.** The session layer does not care whether a source came from COMTRADE,
   CSV, or Excel. All sources are `DisturbanceRecord` objects by the time they enter the session.
3. **Import Wizard is still the entry point.** Every source must pass through the Import Wizard
   (or direct provider load) before being added to a session.
4. **Incremental.** Each sub-phase is independently testable and shippable. Phases 9A–9C must
   be complete before 9D. Phase 9E can run in parallel with 9D.
5. **Analyst-first language.** UI labels, tooltips, and error messages use engineering vocabulary,
   not internal code names.

---

## Sub-phase Overview

| Sub-phase | Name | Delivers |
|---|---|---|
| 9A | Event Session Data Model | Pure Python session contracts, no Qt |
| 9B | Session Workspace Panel | Dockable UI to manage sources |
| 9C | Interactive Time Alignment | Per-source offset control, auto-align helpers |
| 9D | Unified Session Canvas | Multi-source rendering on shared canvas |
| 9E | Legend, Labels & Colour Management | Per-panel legend, per-channel colour/name override |

---

## Phase 9A — Event Session Data Model

**Location:** `app/sessions/` (new package)
**Dependencies:** `app/data/`, `app/models/` — no Qt, no PyQtGraph

### New files

#### `app/sessions/session_models.py`

```python
@dataclass
class SessionSource:
    source_id: str           # UUID, unique within session
    display_name: str        # user-editable, defaults to filename
    record: DisturbanceRecord
    provider_type: str       # 'comtrade' | 'csv' | 'excel'
    origin_path: str | None  # original file path, informational only
    time_offset_s: float     # view-only shift; positive = shift right in time
    is_active: bool          # False = excluded from canvas and analytics

    # Enhancement 1 — Alignment provenance
    alignment_method: str = "none"
    # 'none'        — no offset applied (default, offset_s == 0.0)
    # 'manual'      — analyst typed or dragged the offset
    # 'auto_trigger'— detect_trigger_time() heuristic
    # 'correlation' — cross-correlation against reference (future)
    # 'imported'    — offset was loaded from a saved session or manifest
    alignment_confidence: float | None = None
    # [0.0, 1.0] for auto methods; None for manual/imported (not applicable)

@dataclass
class SessionChannel:
    source_id: str
    channel_name: str        # canonical name from DisturbanceRecord
    channel_type: str        # 'analog' | 'digital'
    display_name: str        # user override; defaults to channel_name
    color_hex: str | None    # user override; None = auto-assign
    line_style: str          # 'solid' | 'dashed' | 'dotted'
    is_visible: bool
    panel_id: str            # which panel this channel is assigned to

@dataclass
class PanelConfig:
    panel_id: str
    title: str
    channel_refs: list[tuple[str, str]]  # list of (source_id, channel_name)
    panel_type: str          # 'analog' | 'digital'
    is_visible: bool

@dataclass
class AlignedChannelData:
    source_id: str
    channel_name: str
    time: np.ndarray         # float64 seconds, offset applied, within requested window
    values: np.ndarray       # float64, decimated/interpolated to requested resolution
    original_sample_rate_hz: float
    time_offset_s: float     # the offset that was applied (informational)
    unit: str | None
    # Enhancement 3 — PMU / non-uniform sampling flag
    time_is_uniform: bool = True
    # False when the source time array has jitter, dropouts, or missing frames.
    # The resampling path handles both cases correctly; this flag is informational
    # for analytics layers that assume uniform spacing (e.g. FFT window sizing).

# Enhancement 1 — structured result from auto-align operations
@dataclass
class AlignmentResult:
    source_id: str
    suggested_offset_s: float
    alignment_method: str          # matches SessionSource.alignment_method vocabulary
    alignment_confidence: float | None
    reference_time: float | None   # the trigger/event time detected in the source
    notes: str                     # human-readable explanation (shown in Session Panel tooltip)

# Enhancement 4 — per-source data quality metrics
@dataclass
class SourceQualityMetrics:
    source_id: str
    sample_count: int
    inferred_sample_rate_hz: float
    sample_rate_stability: float   # [0.0, 1.0]; 1.0 = perfectly uniform spacing
    missing_data_pct: float        # % of expected samples absent (NaN or gap)
    duplicate_timestamp_pct: float # % of timestamps that are duplicated
    interpolated_pct: float        # % of output samples that were interpolated (not raw)
    resampling_ratio: float        # output_rate / input_rate; > 1 = upsampled, < 1 = downsampled
    time_is_uniform: bool          # mirrors AlignedChannelData.time_is_uniform for the source
```

#### `app/sessions/event_session.py`

```python
class EventAnalysisSession:
    """
    Container for one analyst workspace: N sources + channel registry + panel layout.
    All time offsets are applied lazily at render time — DisturbanceRecord is never mutated.
    """

    # Enhancement 2 — persistence hooks (populated at construction, never mutated)
    session_version: int = 1          # increment when serialisation format changes
    created_at: datetime              # UTC datetime; set once at session creation

    # Source management
    def add_source(record, display_name, provider_type, origin_path) -> str
    def remove_source(source_id) -> None
    def get_source(source_id) -> SessionSource | None
    def list_sources() -> list[SessionSource]
    def set_source_active(source_id, active) -> None

    # Time alignment
    def set_time_offset(source_id, offset_s, method="manual", confidence=None) -> None
    # method and confidence are written to SessionSource.alignment_method/confidence.
    def get_time_offset(source_id) -> float
    def reset_all_offsets() -> None
    def get_global_time_range() -> tuple[float, float]
    # returns intersection of all active source time ranges after offsets applied

    # Enhancement 4 — source quality
    def get_source_quality_metrics(source_id) -> SourceQualityMetrics
    # Computed lazily on first call; cached until source is replaced.

    # Channel registry
    def get_channel(source_id, channel_name) -> SessionChannel | None
    def list_analog_channels(active_only=True) -> list[SessionChannel]
    def list_digital_channels(active_only=True) -> list[SessionChannel]
    def set_channel_display_name(source_id, channel_name, name) -> None
    def set_channel_colour(source_id, channel_name, color_hex) -> None
    def set_channel_visibility(source_id, channel_name, visible) -> None
    def set_channel_panel(source_id, channel_name, panel_id) -> None

    # Panel layout
    def list_panels() -> list[PanelConfig]
    def add_panel(title, panel_type) -> str
    def remove_panel(panel_id) -> None
    def rename_panel(panel_id, title) -> None
    def merge_panels(panel_id_a, panel_id_b) -> str  # returns new panel_id
    def default_layout() -> None  # auto-group by channel type across all sources

    # Data access for rendering
    def build_aligned_data(
        source_id, channel_name,
        t_start, t_end,
        max_points=4000
    ) -> AlignedChannelData
```

#### `app/sessions/alignment_engine.py`

Pure functions. No state.

```python
def apply_time_offset(time_array, offset_s) -> np.ndarray
# Returns new array; does not mutate input.

def resample_to_grid(time_src, values_src, time_grid) -> np.ndarray
# Linear interpolation of (time_src, values_src) onto time_grid.
# Fills NaN outside source coverage — does not extrapolate.
# Enhancement 3 — PMU readiness: time_src is NOT assumed uniformly spaced.
# scipy.interpolate.interp1d with kind='linear' handles irregular spacing correctly.
# Callers must not assume equal dt between adjacent samples.

def build_common_time_grid(sources, t_start, t_end, max_points) -> np.ndarray
# Builds a uniform OUTPUT grid across all active sources for the requested window.
# Grid resolution = median sampling interval of the finest-rate active source,
# capped at max_points. Uses median (not mean) to tolerate dropout-induced gaps
# in PMU or telemetry streams without blowing up the grid density.

def detect_trigger_time(time_array, values_array, threshold_factor=3.0) -> float | None
# Heuristic: returns the time of first sample where |value| > threshold_factor * RMS(first 20%).
# Returns None if no trigger detected.
# time_array may be non-uniform; the function uses actual time values, not index arithmetic.

def suggest_alignment_offsets(sources) -> list[AlignmentResult]
# Runs detect_trigger_time on the first eligible analog channel of each source.
# Returns one AlignmentResult per source with:
#   alignment_method = 'auto_trigger'
#   alignment_confidence = normalised peak-to-RMS ratio at the detected event [0.0, 1.0]
#   suggested_offset_s  = offset that aligns this source's trigger to t=0
#   reference_time      = the raw trigger time detected in the source's own time axis
#   notes               = human-readable explanation
# Sources with no detected trigger get confidence=0.0, offset=0.0, notes explain why.

def assess_time_uniformity(time_array) -> tuple[bool, float]
# Returns (is_uniform, stability_score).
# is_uniform: True if std(diff(time_array)) / mean(diff(time_array)) < 0.01
# stability_score: 1.0 - clipped coefficient of variation of inter-sample intervals [0.0, 1.0]
# Used to populate SessionSource.alignment_confidence and SourceQualityMetrics.sample_rate_stability.

def compute_source_quality(source_id, time_array, values_array) -> SourceQualityMetrics
# Computes all SourceQualityMetrics fields from raw arrays.
# Called lazily by EventAnalysisSession.get_source_quality_metrics().
```

#### `app/sessions/__init__.py`

Exports: `EventAnalysisSession`, `SessionSource`, `SessionChannel`, `PanelConfig`,
`AlignedChannelData`, `AlignmentResult`, `SourceQualityMetrics`, `alignment_engine`.

### Design Notes — Phase 9A

#### Enhancement 3 — PMU & Non-Uniform Sampling

Time arrays throughout the session layer are **not assumed uniformly spaced**. This is a
deliberate architectural decision to support:

- PMU streams with dropouts or missing frames
- SCADA telemetry with jitter or variable poll intervals
- Merged/repaired CSV data where some rows were interpolated during import
- Future streaming sources where sample delivery is not clock-locked

Consequences:
- `resample_to_grid()` uses value-based interpolation (`interp1d`), never index arithmetic
- `build_common_time_grid()` uses **median** inter-sample interval, not mean, to tolerate dropout gaps
- `detect_trigger_time()` uses the actual time axis for threshold crossing, not sample index
- `assess_time_uniformity()` flags non-uniform sources so analytics layers that require
  uniform spacing (e.g. FFT) can issue a warning rather than silently producing wrong results
- `AlignedChannelData.time_is_uniform` propagates this flag to the rendering and analytics layers

No special PMU parser or protocol is introduced in Phase 9. The architecture simply never
assumes uniformity, which means PMU data will work correctly when a PMU provider is added later.

### Tests — Phase 9A

**`tests/unit/test_event_session.py`** — 40+ tests:
- add/remove source; source_id uniqueness
- `session_version` is 1 at construction; `created_at` is set and immutable
- time offset get/set with method/confidence stored; reset_all_offsets clears method to 'none'
- global_time_range with offsets applied; intersection when sources don't fully overlap
- channel registry CRUD; display_name override; colour override
- default_layout groups by channel type across sources
- merge_panels combines channel_refs correctly
- build_aligned_data: offset applied; different sample rates produce same time grid; NaN fill outside coverage
- `get_source_quality_metrics()`: returns correct interpolated_pct and resampling_ratio
- alignment_engine: `apply_time_offset` non-mutating; `resample_to_grid` handles non-uniform input;
  `detect_trigger_time` finds step events; `suggest_alignment_offsets` returns `AlignmentResult` list
  with correct confidence; `assess_time_uniformity` detects jittered vs uniform arrays;
  `compute_source_quality` produces correct missing_data_pct for gap-containing arrays

---

## Phase 9B — Session Workspace Panel

**Location:** `app/ui/session/`
**Dependencies:** Phase 9A, PyQt6

### New files

#### `app/ui/session/session_panel.py` — `SessionPanel(QDockWidget)`

```
┌─ Event Analysis Session ────────────────────────┐
│  [+ Add Source]  [Clear Session]                │
├─────────────────────────────────────────────────┤
│  ┌─ Source 1: pulu_comtrade ─────────────────┐  │
│  │  [✓] COMTRADE  │  42A + 88D  │  5000 Hz   │  │
│  │  Quality: ████████░░ 98%  Uniform  ⓘ      │  │
│  │  Offset: [← ] [-0.000 s] [→]  ● —  [Auto] │  │
│  │  [Expand channels ▼]            [Remove]  │  │
│  └───────────────────────────────────────────┘  │
│  ┌─ Source 2: pulu_csv ──────────────────────┐  │
│  │  [✓] CSV       │  3A + 0D    │  1 Hz      │  │
│  │  Quality: ██████░░░░ 61%  Non-uniform  ⓘ  │  │
│  │  Offset: [← ] [+12.500 s] [→]  ● High (0.91) [Auto] │  │
│  │  [Expand channels ▼]            [Remove]  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

The **Quality** bar is a compact progress bar showing `sample_rate_stability × 100%`.
Hovering the `ⓘ` icon shows a tooltip with all `SourceQualityMetrics` fields:

```
Source: pulu_csv
Samples: 3,840   Rate: 1.00 Hz   Stability: 61%
Missing: 2.1%    Duplicates: 0.0%
Interpolated: 38.5%   Resample ratio: 0.0002×
Time uniformity: Non-uniform (jitter detected)
```

Channel tree (expanded per source):
```
  ▼ Source 1: pulu_comtrade
    ▼ Analog (42)
      [✓] ● VA   VA_comtrade    Panel: Voltage  [✎]
      [✓] ● VB   VB_comtrade    Panel: Voltage  [✎]
    ▼ Digital (88)
      [✓] □ CB1  CB1_status     Panel: Digital  [✎]
```

Key Qt signals emitted by `SessionPanel`:
```python
source_add_requested     = pyqtSignal()
source_remove_requested  = pyqtSignal(str)        # source_id
offset_changed           = pyqtSignal(str, float) # source_id, offset_s
auto_align_requested     = pyqtSignal(str)        # source_id or 'all'
channel_visibility_changed = pyqtSignal(str, str, bool)
channel_colour_change_requested = pyqtSignal(str, str)
channel_panel_changed    = pyqtSignal(str, str, str)
session_cleared          = pyqtSignal()
```

Supporting widgets:
- `app/ui/session/source_row_widget.py` — one collapsible row per `SessionSource`
- `app/ui/session/channel_tree_widget.py` — QTreeWidget for channels within a source

### Integration with `PowerwaveMainWindow`

New menu items:
- `View → Session Panel` — toggles the docked `SessionPanel`
- `File → Add to Session…` — opens Import Wizard; on success calls `session.add_source()`
- `File → New Session` — clears session and panel
- `View → Session Canvas` — available when session has ≥ 1 source

New instance state:
- `_active_session: EventAnalysisSession | None`
- `_session_panel: SessionPanel | None`

The existing `File → Open` direct load remains unchanged and independent of the session.

### Tests — Phase 9B

**`tests/unit/test_session_panel.py`** — signal/model tests (no display)
**`tests/runtime/test_session_panel_runtime.py`** — Qt runtime: add source, offset spinbox,
channel visibility toggle, remove source

---

## Phase 9C — Interactive Time Alignment

**Location:** extends `app/ui/session/` and `app/sessions/alignment_engine.py`

This is the highest-priority analyst-facing feature.
An analyst must be able to shift Source 2 by ±seconds until a fault step in Source 2
visually aligns with the same event in Source 1.

### Offset control UI (per source row)

```
Offset:  [←]  [-12.500 s]  [→]   [Reset]   [Auto-align]
          fine   spinbox    fine
```

- **Spinbox**: range ±9999.999 s, step 0.001 s (1 ms precision), direct keyboard entry
- **Fine arrows (←/→)**: ±1 sample of that source's native sampling interval per click
- **Reset**: sets offset to 0.0
- **Auto-align**: `detect_trigger_time` on the most prominent analog channel of this source;
  sets offset so its trigger aligns to session reference t=0

Session-level alignment toolbar (in `SessionPanel` header):
- **"Align All to Reference"**: runs `suggest_alignment_offsets()` across all active sources;
  writes `alignment_method='auto_trigger'` and `alignment_confidence` into each `SessionSource`
- **"Set as Reference"** (context menu on a source): sets that source's offset to 0.0,
  `alignment_method='manual'`, `alignment_confidence=None`; adjusts all others relative to it

#### Alignment confidence display

Each source row in `SessionPanel` shows a confidence badge beside the offset spinbox:

```
Offset:  [←]  [-12.500 s]  [→]   [Reset]   ● High (0.91)  [Auto-align]
                                             └─ auto_trigger
```

Badge colours: `High ≥ 0.75` = green, `Medium 0.40–0.74` = amber, `Low < 0.40` = red,
`manual` / `imported` = grey (no numeric confidence, labelled method name only).
Hovering the badge shows the full `AlignmentResult.notes` string as a tooltip.

### Non-destructive invariant

`alignment_engine.apply_time_offset()` is called inside `EventAnalysisSession.build_aligned_data()`
before resampling. The canvas receives a pre-shifted time array. `DisturbanceRecord` is never touched.

### Visual offset indicator

Each source's curves carry a thin source-coloured vertical line at its original t=0,
annotated with the source display name and current offset value
(e.g. `pulu_csv  +12.500 s`). This helps the analyst see "where this source's clock zero sits"
on the common time axis.

### Tests — Phase 9C

**`tests/unit/test_alignment_engine.py`** — 20+ tests:
- `apply_time_offset`: non-mutating, float64 precision, zero offset pass-through
- `resample_to_grid`: NaN fill outside coverage, interpolation accuracy, same-rate pass-through
- `build_common_time_grid`: finest rate governs resolution, max_points cap, single source
- `detect_trigger_time`: step event detected, flat signal returns None, short signal edge case
- `suggest_alignment_offsets`: multi-source, partial detection, no-trigger fallback

**`tests/runtime/test_time_alignment_runtime.py`** — Qt runtime:
- Offset spinbox change updates canvas within one `processEvents` cycle
- Auto-align on synthetic fault record produces non-zero offset
- "Align All" applies suggested offsets to all sources simultaneously
- Reset returns offset to 0.0 and canvas re-renders

---

## Phase 9D — Unified Session Canvas

**Location:** `app/ui/session/session_canvas_controller.py`,
`app/visualization/widgets/session_canvas.py`

### Architecture

`SessionCanvasController` owns the canvas layout for the current session:

1. Reads `session.list_panels()` → creates one `FlexiblePlotCanvas` per panel
2. On each viewport update, calls `session.build_aligned_data()` per visible channel →
   gets offset-applied, resampled `AlignedChannelData`
3. Passes arrays to existing `FlexiblePlotCanvas` curve update path via `setData()`
4. Registers all canvases with `SynchronizationManager` → pan/zoom synchronises across panels

```python
class SessionCanvasController:
    def __init__(self, session: EventAnalysisSession, sync_manager: SynchronizationManager)
    def rebuild_layout() -> QWidget       # returns QSplitter with all panel canvases
    def refresh_panel(panel_id) -> None
    def refresh_all() -> None
    def on_offset_changed(source_id, offset_s) -> None   # triggers refresh_all
    def on_channel_visibility_changed(source_id, channel_name, visible) -> None
    def on_panel_merged(panel_id_a, panel_id_b) -> None  # triggers layout rebuild
```

### Resampling policy

When a panel contains channels from sources with different sampling rates:
- Common time grid uses the **finest** sampling rate among active visible channels,
  capped at `max_points` for the current viewport width
- Lower-rate channels → **linear interpolation** onto the common grid (NaN outside coverage)
- Higher-rate channels → **decimation** via existing `decimate_for_display()` pipeline

The analyst sees no gaps or jumps. A 1 Hz CSV channel renders as a smooth interpolated
overlay alongside a 5000 Hz COMTRADE channel. The visual density difference makes the
sampling rate difference naturally apparent without requiring separate canvases.

### Panel merge / split (right-click panel header)

- **Merge with →**: combine two panels into one (channels from both appear together)
- **Split by source**: separate panel's channels back into one-canvas-per-source
- **Split by type**: separate into voltage / current / other within this panel

### Integration with `PowerwaveMainWindow`

New central widget mode `SESSION_CANVAS` added alongside `STANDARD` and `GROUPED`.
Activated by `View → Session Canvas` when session has ≥ 1 source.
`SessionCanvasController` builds the layout and registers with `SynchronizationManager`.

### Tests — Phase 9D

**`tests/unit/test_session_canvas_controller.py`** — 25+ unit tests (mock canvas):
- `rebuild_layout` creates correct number of panels per `list_panels()`
- `refresh_all` calls `build_aligned_data` for each visible channel
- `on_offset_changed` triggers refresh without full rebuild
- Different-rate arrays reach common grid with correct length and NaN fill

**`tests/runtime/test_session_canvas_runtime.py`** — Qt runtime:
- Load COMTRADE + CSV synthetic records → session canvas renders without crash
- Offset change re-renders both sources immediately
- `SynchronizationManager` links all panels: pan in one moves all others
- Merging two panels produces single canvas with combined channel count
- Removing a source clears its curves from all panels

---

## Phase 9E — Legend, Labels & Colour Management

**Location:** `app/ui/session/legend_widget.py`, extensions to `SessionPanel`

### `ChannelLegendWidget(QWidget)`

Compact table docked at the bottom of each `FlexiblePlotCanvas` panel:

```
┌─ Voltage ─────────────────────────────────────────────┐
│  ■ VA  [pulu_comtrade]  kV    ■ VB  [pulu_comtrade]   │
│  ■ VA  [pulu_csv]       kV    ■ VB  [pulu_csv]        │
└───────────────────────────────────────────────────────┘
```

Each row: colour swatch + display name + source badge + unit.

Interactions:
- **Click colour swatch** → `QColorDialog` → stored in `session.set_channel_colour()`
- **Double-click display name** → inline edit → stored in `session.set_channel_display_name()`
- **Right-click row** → context menu: Hide, Move to panel…, Reset colour, Reset name

Toggled via **View → Show Legend** (default: on).

### Multi-source colour strategy

Default auto-assignment (no user override):
- Channels within a source follow existing phase heuristic (A=blue, B=yellow, C=red)
- Source 1: full saturation
- Source 2: 70% saturation, hue shifted +15°
- Source 3: 50% saturation, hue shifted +30°

This ensures VA from COMTRADE (bright blue) and VA from CSV (muted blue) are visually
related but distinct. User override via colour swatch always wins.

### Channel display name rules

- Default: source channel name (e.g. `VA`)
- If two active sources both have `VA`, the second auto-appends source badge: `VA [pulu_csv]`
- User edit to any string overrides the auto-badge
- Reset name restores the auto-generated default

### Tests — Phase 9E

**`tests/unit/test_legend_widget.py`**:
- Colour override stored in `SessionChannel`; auto-colour for source N uses correct saturation
- Display name override; duplicate name auto-badge; reset restores default
- Legend rows match visible channels in panel

**`tests/runtime/test_legend_runtime.py`** — Qt runtime:
- Legend shows correct channel count after add/remove source
- Colour picker change reflects immediately on canvas curve and legend swatch
- Display name edit updates legend label and curve tooltip

---

## Data Flow Summary

```
Import Wizard  ──►  DisturbanceRecord
                         │
                         ▼
              EventAnalysisSession.add_source()
                         │
              ┌──────────┴────────────────────┐
              │  SessionSource                 │  SessionChannel registry
              │  time_offset_s                │  display_name, color_hex
              └──────────┬────────────────────┘
                         │
              build_aligned_data(t_start, t_end, max_points)
                         │
              ┌──────────┴──────────────────────────────┐
              │  apply_time_offset()                     │
              │  build_common_time_grid()                │
              │  resample_to_grid() / decimate()         │
              └──────────┬──────────────────────────────┘
                         │
              AlignedChannelData (time[], values[])
                         │
              SessionCanvasController
                         │
              FlexiblePlotCanvas.setData()  (reuses existing curves)
                         │
              SynchronizationManager  (X-axis + cursor sync across panels)
                         │
              ChannelLegendWidget  (colour + name display per panel)
```

---

## Acceptance Criteria

Phase 9 is complete when all of the following hold:

1. An analyst can load a COMTRADE file and a CSV file into the same session via the Import Wizard
2. Both sources appear in the Session Panel with channel counts, sampling rates, and quality bar visible
3. Adjusting the offset spinbox for one source shifts its waveforms left/right in real time
4. "Auto-align" detects the fault event in both sources, aligns them to t=0, and shows a confidence badge
5. The alignment method ('manual' / 'auto_trigger' / 'imported') is visible per source at all times
6. All channels from both sources are visible on a shared canvas with correct colours, units, and labels
7. A legend shows all channel names and colour swatches; clicking a swatch opens a colour picker
8. Pan and zoom on any panel moves all panels simultaneously
9. Merging two panels produces a single canvas with the combined channel set
10. Removing a source removes all its channels from canvas and legend immediately
11. A non-uniformly-sampled source (jittered CSV) renders correctly without crash or silent error
12. `session_version` and `created_at` are present on every `EventAnalysisSession` instance
13. `SourceQualityMetrics` tooltip is accessible from the Session Panel for every loaded source
14. Full test suite passes with 0 regressions and ≥ 130 new tests across all sub-phases

---

## Non-Goals for Phase 9

Explicitly out of scope:

- Saving/loading a session to disk (Phase 10 candidate)
- Report generation or annotated screenshot export
- Phasor/harmonic analytics operating across sources (existing per-source analytics continue to work)
- GPS/PTP timestamp correction (the offset tool handles this manually)
- More than 5 simultaneous sources (no hard limit, UX not optimised beyond 5)
- Waveform editing or value modification of any kind

---

## Remaining Phase 8 Work (complete before starting Phase 9)

Check `agent/REPOSITORY_STATE.md` and `agent/TASK.md` for current Phase 8 status.
Do not begin Phase 9A until the Phase 8 test baseline is fully green.
