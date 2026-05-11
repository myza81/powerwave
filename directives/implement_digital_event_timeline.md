implement_digital_event_timeline.md — Phase 3B Directive
DIRECTIVE AUTHORITY

Issued by: Claude Code (architecture-guided implementation)
Target agent: Claude / Claude Code
Phase: 3B — Digital Event Timeline
Status: ISSUED AND EXECUTING
Depends on: Phase 3A (FlexiblePlotCanvas — COMPLETE)

─────────────────────────────────────────────────────────────────────────────
MANDATORY PRE-READING
─────────────────────────────────────────────────────────────────────────────

  agent/WORKFLOW_AGENT.md
  agent/CLAUDE.md
  agent/REPOSITORY_STATE.md
  docs/ARCHITECTURE.md
  docs/VISUALIZATION_CONTRACT.md
  docs/VIEWPORT_RENDERING_POLICY.md  — especially §9, §17
  docs/CHANNEL_MAPPING_POLICY.md     — especially §2, §6, §9
  docs/PERFORMANCE_REQUIREMENTS.md
  docs/LEGACY_CODEBASE_POLICY.md
  directives/implement_flexible_plot_canvas.md

─────────────────────────────────────────────────────────────────────────────
OBJECTIVE
─────────────────────────────────────────────────────────────────────────────

Implement DigitalEventTimeline — the Phase 3B companion to FlexiblePlotCanvas.

Architecture: single pg.PlotWidget with N fixed-height horizontal tracks.
  - Each digital channel occupies one track (vertical offset per channel)
  - Binary HIGH state: filled with role-based color
  - Binary LOW state: empty (no fill)
  - No Y-axis zoom/pan; no independent Y-axis per channel
  - X-axis linkable to FlexiblePlotCanvas primary PlotItem
  - Trigger line + master cursor foundation

─────────────────────────────────────────────────────────────────────────────
PHASE 3B SCOPE
─────────────────────────────────────────────────────────────────────────────

IN SCOPE:

  app/visualization/rendering/digital_transforms.py   (pure NumPy, testable)
  app/visualization/widgets/digital_event_timeline.py (Qt widget)
  tests/unit/test_digital_transforms.py               (≥20 tests, no GUI)

NOT IN SCOPE:

  VisualizationManager (multi-widget coordinator)     Phase 3B+ (later)
  SynchronizationManager                              Phase 3B+ (later)
  Full multi-pane dashboard                           Phase 4
  Analytics overlays (RMS, ROCOF)                     Phase 5
  Phasor canvas                                       Phase 5+
  Complementary CB pair merging (OPEN+CLOSE → single) Phase 4+
  Signal role detection engine (analytics layer)      Phase 5

─────────────────────────────────────────────────────────────────────────────
FILE 1 — app/visualization/rendering/digital_transforms.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Pure-NumPy digital channel processing utilities. No Qt. No
         DisturbanceRecord imports. Independently testable without display.

FUNCTION 1 — digital_role_color(name: str) -> str

  Returns display color for a digital channel name using keyword heuristics
  from CHANNEL_MAPPING_POLICY §2 and VIEWPORT_RENDERING_POLICY §9.

  PRIORITY ORDER (first match wins):
    1. Alarm/supervision exception (→ DIG_GENERIC grey):
       keywords: commfail, comm fail, comm_fail, _fail, alarm, warning, mcb, vts
    2. DIG_CB (→ orange #FF8800):
       keywords: gcb, fcb, cb_r, cb_y, cb_b, 52b, 52a, cb open, cb close
    3. DIG_AR (→ blue #4488FF):
       keywords: reclose, autoreclose, ar_, 79ar, auto reclose
    4. DIG_INTERTRIP (→ magenta #FF44FF):
       keywords: intertrip, inter-trip, 50bf, bf_send, teleprotect, direct trip
    5. DIG_TRIP (→ red #FF2222):
       keywords: trip, operated, gen trip, op_
    6. DIG_PICKUP (→ amber #FFAA00):
       keywords: pickup, pick up, _fd, fault det, element start
    7. Default DIG_GENERIC (→ grey #AAAAAA)

  All matching is case-insensitive on the full channel name.

FUNCTION 2 — extract_transitions(time, data) -> (t_out, d_out)

  Signature:
    def extract_transitions(
        time: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

  Purpose: Reduce N sampled points to M transition points (M << N for
           typical digital channels that rarely change state). Returns the
           minimal sparse representation needed for step-function rendering.

  Algorithm:
    1. Validate: both must be 1-D, equal length. Raise ValueError otherwise.
    2. Empty input → return (empty float64, empty float64).
    3. Coerce data to binary: d_bin = (data != 0).astype(np.int8)
    4. changes = np.concatenate([[True], d_bin[1:] != d_bin[:-1]])
       (always include index 0; include every index where value changes)
    5. Append terminal sentinel: time[-1] and d_bin[-1]
       (ensures last segment has a right endpoint for step rendering)
    6. Return (t_out, d_out) as float64 arrays.

  Constraints:
    - FULLY VECTORIZED. No Python loops.
    - Input arrays must not be modified.
    - Output dtype: float64 for both arrays.
    - Raises ValueError with message containing "1-D" for non-1-D input.
    - Raises ValueError with message containing "length mismatch" for unequal lengths.

FUNCTION 3 — clip_digital_to_viewport(t_trans, d_trans, t_start, t_end)
              -> (t_out, d_out)

  Signature:
    def clip_digital_to_viewport(
        t_trans: np.ndarray,
        d_trans: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> tuple[np.ndarray, np.ndarray]:

  Purpose: Clip pre-extracted transition data to [t_start, t_end], adding a
           carry-state point at t_start so the digital state is correctly shown
           at the left viewport edge.

  Algorithm:
    1. Empty input → return (empty float64, empty float64).
    2. t_start > t_end → swap silently.
    3. t_trans[0] > t_end → viewport is entirely before data → return empty.
    4. t_trans[-1] <= t_start → all data before viewport:
         carry final state as flat line: t=[t_start, t_end], d=[d_trans[-1], d_trans[-1]]
    5. Find left-edge state: before_idx = searchsorted(t_trans, t_start, 'right') - 1
    6. If before_idx >= 0: add carry point (t_start, d_trans[before_idx])
    7. Add all transitions strictly within (t_start, t_end] from t_trans.
    8. Append terminal point at t_end with last known state.
    9. Return (t_out, d_out) as float64 arrays.

FUNCTION 4 — build_step_series(t, d, y_offset, track_height) -> (t_step, y_step)

  Signature:
    def build_step_series(
        t: np.ndarray,
        d: np.ndarray,
        y_offset: float = 0.0,
        track_height: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:

  Purpose: Expand (t, d) transition data into explicit step-function line
           segments with Y scaling for multi-track display. Output goes
           directly to curve.setData().

  Algorithm:
    - If len(t) < 2: return (t, d*track_height + y_offset) as float64.
    - n_segs = len(t) - 1
    - t_step: interleave t[:-1] and t[1:] → [t0, t1, t1, t2, t2, ...]
    - y_seg = d[:-1] * track_height + y_offset
    - y_step = np.repeat(y_seg, 2)    → [y0, y0, y1, y1, ...]
    - Return (t_step, y_step) as float64.

  Rationale: Each segment (t[i], y) → (t[i+1], y) is a horizontal line at
             the state held between transitions. fill between y_step and
             y_offset fills only HIGH-state segments.

─────────────────────────────────────────────────────────────────────────────
FILE 2 — app/visualization/widgets/digital_event_timeline.py
─────────────────────────────────────────────────────────────────────────────

DATACLASS _TrackEntry:

  @dataclasses.dataclass
  class _TrackEntry:
      name:     str
      curve:    pg.PlotDataItem
      color:    str
      y_offset: float         # baseline Y in data coordinates
      t_trans:  np.ndarray    # pre-extracted transition times (float64)
      d_trans:  np.ndarray    # pre-extracted transition states (float64)

MODULE-LEVEL CONSTANTS:

  _TRACK_SPACING = 1.5   # vertical distance between track baselines (data coords)
  _TRACK_HEIGHT  = 1.0   # height of HIGH-state fill (data coords)

CLASS: DigitalEventTimeline(pg.PlotWidget)

  pyqtSignal:
    cursor_moved = pyqtSignal(float)   # emits cursor time on user drag

  CONSTRUCTOR __init__(parent=None):
    1. super().__init__(parent=parent, background='#1E1E1E')
    2. self._record: DisturbanceRecord | None = None
    3. self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
    4. self._tracks: dict[str, _TrackEntry] = {}
    5. self._trigger_line: pg.InfiniteLine | None = None
    6. self._cursor: pg.InfiniteLine | None = None
    7. Configure PlotItem:
         plot = self.getPlotItem()
         plot.showGrid(x=True, y=False, alpha=0.2)
         plot.setLabel('bottom', 'Time', units='s')
         plot.showAxis('left')
         plot.setMouseEnabled(x=True, y=False)
    8. Connect: plot.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)

  PUBLIC METHODS:

  set_record(record: DisturbanceRecord) -> None
    1. self.clear()
    2. Store record and cache time array.
    3. For each ch in record.digital_channels: self._add_track(ch)
    4. self._update_y_axis()
    5. self._add_trigger_line()
    6. self._add_cursor()
    7. self._update_viewport()

  link_x_to(view_or_plot: pg.ViewBox | pg.PlotItem) -> None
    self.getPlotItem().getViewBox().setXLink(view_or_plot)
    Purpose: X-link to FlexiblePlotCanvas._primary_plot for shared navigation.

  set_cursor_pos(t: float) -> None
    Move cursor without re-emitting (blockSignals True/False pattern).

  clear() -> None
    1. self.getPlotItem().clear()
    2. self._tracks.clear()
    3. Reset _record, _time_cache, _cursor, _trigger_line to defaults.
    4. self._restore_plot_config()

  PRIVATE METHODS:

  _add_track(ch: DigitalChannel) -> None
    1. track_idx = len(self._tracks) (before inserting)
    2. y_offset = track_idx * _TRACK_SPACING
    3. color = digital_role_color(ch.name)
    4. raw = record.waveform_data[ch.name].to_numpy(dtype=np.float64)
    5. t_trans, d_trans = extract_transitions(self._time_cache, raw)
    6. Create curve: pg.PlotDataItem(
           pen=pg.mkPen(color, width=1.5),
           fillLevel=y_offset,
           brush=pg.mkBrush(color + '55'),
           skipFiniteCheck=True,
       )
    7. self.addItem(curve)
    8. Store _TrackEntry in self._tracks[ch.name].

  _update_y_axis() -> None
    1. n = len(self._tracks)
    2. Set Y-range: [-0.25, (n-1)*_TRACK_SPACING + _TRACK_HEIGHT + 0.25]
    3. Custom tick labels: [(entry.y_offset + _TRACK_HEIGHT/2, entry.name) ...]
       Apply via plot.getAxis('left').setTicks([ticks])

  _add_trigger_line() -> None
    Per VIEWPORT_RENDERING_POLICY §7. Add to self (PlotWidget) at trigger_time_s.

  _add_cursor() -> None
    Per VIEWPORT_RENDERING_POLICY §6. Add movable InfiniteLine.
    Connect sigPositionChanged to _on_cursor_moved.

  _on_cursor_moved(line) -> None
    self.cursor_moved.emit(line.value())

  _on_x_range_changed(_viewbox, x_range) -> None
    t_start, t_end = x_range; self._update_viewport(t_start, t_end)

  _update_viewport(t_start=None, t_end=None) -> None  ← HOT PATH
    1. Guard: if len(self._time_cache) == 0: return
    2. If t_start/t_end None: read from ViewBox.
    3. For each entry in self._tracks.values():
         t_cl, d_cl = clip_digital_to_viewport(t_trans, d_trans, t_start, t_end)
         if len(t_cl) < 2: entry.curve.setData(empty, empty); continue
         t_step, y_step = build_step_series(t_cl, d_cl, entry.y_offset, _TRACK_HEIGHT)
         entry.curve.setData(t_step, y_step)

  _restore_plot_config() -> None
    Restore grid, label, axis config after clear(). Same settings as __init__.

─────────────────────────────────────────────────────────────────────────────
FILE 3 — tests/unit/test_digital_transforms.py
─────────────────────────────────────────────────────────────────────────────

TestDigitalRoleColor (7 tests)
  - TRIP keyword → #FF2222
  - CB keyword (gcb) → #FF8800
  - AR keyword (reclose) → #4488FF
  - INTERTRIP keyword → #FF44FF
  - PICKUP keyword → #FFAA00
  - Unknown name → #AAAAAA (generic)
  - Alarm exception overrides trip: "ALARM_TRIP" → #AAAAAA not #FF2222

TestExtractTransitions (9 tests)
  - Empty arrays → two empty float64 arrays
  - Constant-zero data → 2 points (initial + terminal)
  - Constant-one data → 2 points
  - Single 0→1 transition → 3 points with correct times and states
  - Multiple transitions → correct sparse count
  - Non-binary (e.g. values 0.0 and 2.5) coerced to 0/1
  - 2-D time array raises ValueError matching "1-D"
  - Length mismatch raises ValueError matching "length mismatch"
  - Output dtype is float64 for both arrays

TestClipDigitalToViewport (8 tests)
  - Empty input → two empty float64 arrays
  - All data before viewport → flat line at carry state
  - Viewport entirely before data → two empty arrays
  - Viewport fully within data → carry-state point + in-range transitions + terminal
  - t_start == t_end → at most 2 points returned
  - t_start > t_end → silently swapped, returns non-empty result
  - Viewport extends past data end → terminal at t_end with last state
  - No transitions in viewport but data spans viewport → carry state as flat line

TestBuildStepSeries (5 tests)
  - Single-segment LOW → two points at y_offset
  - Single-segment HIGH → two points at y_offset + track_height
  - Transition 0→1 → 4 points forming step
  - y_offset applied correctly
  - Output dtype is float64

TOTAL: ≥ 29 tests.

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────

1. Python environment: .venv/Scripts/python.exe exclusively.
2. No new dependencies. PyQtGraph, PyQt6, NumPy already installed.
3. No imports from src/ under any circumstances.
4. pg.setConfigOptions() belongs in app/main.py — NOT in this widget.
5. _update_viewport() is the hot path: only clip_digital_to_viewport +
   build_step_series + setData(). No DataFrame ops, no to_numpy().
6. Transition extraction (extract_transitions) happens ONCE on set_record(),
   stored in _TrackEntry.t_trans / _TrackEntry.d_trans. Never re-extracted.
7. digital_transforms.py must remain free of any Qt or PyQtGraph imports.
8. Do NOT modify FlexiblePlotCanvas, MultiAxisManager, or downsampling.py.
9. Do NOT implement VisualizationManager or SynchronizationManager in this phase.

─────────────────────────────────────────────────────────────────────────────
PHASE 3B SUCCESS CRITERIA
─────────────────────────────────────────────────────────────────────────────

  ✓ app/visualization/rendering/digital_transforms.py — implemented and tested
  ✓ app/visualization/widgets/digital_event_timeline.py — implemented
  ✓ tests/unit/test_digital_transforms.py — ≥20 tests, all passing
  ✓ All 335 existing tests continue to pass (no regressions)
  ✓ No imports from src/ in any new file
  ✓ No analytics or provider imports in visualization/
  ✓ agent/HANDOFF.md, agent/TASK.md, agent/REPOSITORY_STATE.md updated
