implement_fast_waveform_widget.md — SUPERSEDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPERSEDED BY: directives/implement_flexible_plot_canvas.md
REASON: Architecture revised to SIGRA-style N-Axis Single Canvas.
  - Class renamed: FastWaveformWidget → FlexiblePlotCanvas
  - Base class: pg.PlotWidget → pg.GraphicsLayoutWidget
  - Architecture: single-axis → N independent Y-axes (ViewBox per parameter)
  - Scope: analog + digital combined → analog-only (Digital Event Timeline separate)
DO NOT IMPLEMENT THIS DIRECTIVE. Use implement_flexible_plot_canvas.md.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

implement_fast_waveform_widget.md — Phase 3A Directive (ORIGINAL, ARCHIVED)
DIRECTIVE AUTHORITY

Issued by: ChatGPT Architecture Orchestrator (archived)
Target agent: Claude / Claude Code
Phase: 3A — Visualization Engine Foundation
Status: READY FOR IMPLEMENTATION

─────────────────────────────────────────────────────────────────────────────
MANDATORY PRE-READING
─────────────────────────────────────────────────────────────────────────────

Before writing any code, read these documents completely:

  agent/WORKFLOW_AGENT.md
  agent/CLAUDE.md
  agent/REPOSITORY_STATE.md
  docs/VISUALIZATION_CONTRACT.md
  docs/VIEWPORT_RENDERING_POLICY.md
  docs/PERFORMANCE_REQUIREMENTS.md
  docs/ARCHITECTURE.md
  docs/DATA_CONTRACT.md
  docs/CHANNEL_MAPPING_POLICY.md     (for color assignment guidance)

After reading, you must understand:
  - FastWaveformWidget inherits pg.PlotWidget (NOT GraphicsLayoutWidget)
  - Curves must use setData() lifecycle (never remove/re-add)
  - Display decimation caps at 4 000 points per curve per viewport update
  - Time array and channel arrays must be cached on record load, not re-extracted per frame
  - DisturbanceRecord access uses new field names (see VIEWPORT_RENDERING_POLICY §11)
  - PyQtGraph global options are set once at app startup (not in widget constructor)

─────────────────────────────────────────────────────────────────────────────
OBJECTIVE
─────────────────────────────────────────────────────────────────────────────

Implement the FastWaveformWidget and supporting rendering infrastructure as the
foundation of the Powerwave visualization engine.

This is Phase 3A. The scope is deliberately narrow: a single, self-contained
waveform rendering widget. Multi-pane layout, synchronization manager, digital
timeline, and analytics overlays are Phase 3B+ concerns and MUST NOT be
implemented here.

─────────────────────────────────────────────────────────────────────────────
PHASE 3A SCOPE
─────────────────────────────────────────────────────────────────────────────

IN SCOPE — implement these:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  app/visualization/rendering/downsampling.py                         │
  │  app/visualization/widgets/fast_waveform_widget.py                   │
  │  tests/unit/test_downsampling.py                                     │
  └──────────────────────────────────────────────────────────────────────┘

NOT IN SCOPE — do NOT implement these in Phase 3A:

  app/visualization/managers/visualization_manager.py   (Phase 3B)
  app/visualization/managers/synchronization_manager.py (Phase 3B)
  app/visualization/interaction/cursor_manager.py       (Phase 3B)
  app/visualization/widgets/digital_signal_widget.py    (Phase 3B)
  app/visualization/rendering/waveform_renderer.py      (defer — all rendering logic is
                                                          in FastWaveformWidget for Phase 3A)
  Any UI docking framework                              (Phase 4)
  Any application-wide state manager (AppState)         (Phase 4)
  Multi-record merge or comparison view                 (Phase 4+)
  Phasor canvas                                         (Phase 5+)
  RMS / ROCOF / harmonic overlay                        (Phase 5)
  PMU streaming rendering                               (Phase 6+)
  Full multi-pane dashboard                             (Phase 3B+)
  Advanced measurement tools                            (Phase 4+)

─────────────────────────────────────────────────────────────────────────────
FILE 1 — app/visualization/rendering/downsampling.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Pure-NumPy display decimation. No PyQt6, no DisturbanceRecord imports.
         Must be independently testable without a display.

REQUIRED FUNCTION:

  def decimate_for_display(
      time: np.ndarray,
      data: np.ndarray,
      t_start: float,
      t_end: float,
      max_points: int = 4_000,
  ) -> tuple[np.ndarray, np.ndarray]:

ALGORITHM REQUIREMENTS:

  1. Input validation:
     - Both arrays must be 1-D and equal length. Raise ValueError otherwise.
     - If either array is empty, return (np.empty(0), np.empty(0)).
     - t_start must be <= t_end. If t_start > t_end, swap them silently.

  2. Clipping:
     mask = (time >= t_start) & (time <= t_end)
     Clip: t_clip = time[mask], d_clip = data[mask]

  3. If len(t_clip) == 0: return (np.empty(0), np.empty(0)).

  4. If len(t_clip) <= max_points: return (t_clip, d_clip) — no decimation needed.

  5. Decimation (integer stride):
     stride = max(1, len(t_clip) // max_points)
     return t_clip[::stride], d_clip[::stride]

CONSTRAINTS:
  - FULLY VECTORIZED — no Python loops over samples.
  - Return dtype must be float64 for both arrays.
  - The function must not modify input arrays.
  - No side effects (no global state, no caching).

─────────────────────────────────────────────────────────────────────────────
FILE 2 — app/visualization/widgets/fast_waveform_widget.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Single-pane waveform rendering widget backed by PyQtGraph + OpenGL.
         Consumes DisturbanceRecord. Renders analog channels. Hosts cursor and
         trigger line. Provides zoom-to-trigger.

IMPORTS ALLOWED:
  - PyQt6, pyqtgraph, numpy
  - app.models (DisturbanceRecord, AnalogChannel)
  - app.visualization.rendering.downsampling (decimate_for_display)
  - Standard library

IMPORTS FORBIDDEN:
  - Any app.providers.* import
  - Any app.analytics.* import
  - Any src.* import

─────────────────────────────────────────────────────────────────────────────
CLASS: FastWaveformWidget(pg.PlotWidget)
─────────────────────────────────────────────────────────────────────────────

CONSTRUCTOR:

  def __init__(self, parent=None, max_display_points: int = 4_000) -> None:

  Required setup steps (in order):
    1. super().__init__(parent=parent, background='#1E1E1E')
    2. self._max_pts = max_display_points
    3. Initialise private state:
         self._record: DisturbanceRecord | None = None
         self._time: np.ndarray = np.empty(0)           # cached time array
         self._channel_data: dict[int, np.ndarray] = {} # index → float64 array
         self._curves: dict[int, pg.PlotDataItem] = {}
         self._cursor: pg.InfiniteLine | None = None
         self._trigger_line: pg.InfiniteLine | None = None
    4. Configure the embedded PlotItem:
         pi = self.getPlotItem()
         pi.showGrid(x=True, y=True, alpha=0.2)
         pi.setLabel('bottom', 'Time', units='s')
    5. Connect viewport change signal:
         self.getPlotItem().getViewBox().sigXRangeChanged.connect(
             self._on_x_range_changed
         )
    6. Declare pyqtSignal:
         cursor_moved = pyqtSignal(float)   # emits time in seconds

─────────────────────────────────────────────────────────────────────────────
REQUIRED PUBLIC METHODS
─────────────────────────────────────────────────────────────────────────────

set_record(record: DisturbanceRecord) -> None

  Purpose: Load a new record and (re-)build all curves.

  Steps:
    1. Store record: self._record = record
    2. Cache numpy arrays (ONCE — avoids per-frame DataFrame access):
         self._time = record.waveform_data['time'].to_numpy(dtype=np.float64)
         self._channel_data = {
             ch.index: record.waveform_data[ch.name].to_numpy(dtype=np.float64)
             for ch in record.analog_channels
         }
    3. Clear existing curves and items:
         self.getPlotItem().clear()
         self._curves.clear()
         self._cursor = None
         self._trigger_line = None
    4. Create one PlotDataItem per analog channel:
         For each ch in record.analog_channels:
           color = _channel_color(ch)
           curve = pg.PlotDataItem(
               pen=pg.mkPen(color, width=1),
               skipFiniteCheck=True,
           )
           curve.setClipToView(True)
           self.getPlotItem().addItem(curve)
           self._curves[ch.index] = curve
    5. Set Y-axis label from first channel unit (or 'Value' if no channels):
         if record.analog_channels:
             self.getPlotItem().setLabel('left',
                 record.analog_channels[0].name,
                 units=record.analog_channels[0].unit
             )
    6. Add trigger line (see VIEWPORT_RENDERING_POLICY §7).
    7. Add movable cursor (see VIEWPORT_RENDERING_POLICY §6).
    8. Call zoom_to_trigger() to set initial viewport.
    9. Call _update_viewport() to populate curve data.


set_visible_channels(indices: list[int]) -> None

  Purpose: Show or hide specific analog channels by channel index.

  For each ch.index in self._curves:
    if ch.index in indices:
      Show curve: if data cached, call _update_curve(ch.index) with current viewport.
    else:
      Hide curve: call curve.setData(np.empty(0), np.empty(0))


zoom_to_trigger(window_s: float = 0.2) -> None

  Purpose: Set X-range to [trigger - window_s, trigger + window_s].
  Implement per VIEWPORT_RENDERING_POLICY §12.
  Guard: if self._record is None or len(self._time) == 0: return.


set_cursor_pos(t: float) -> None

  Purpose: Move master cursor to time t without re-emitting cursor_moved.
  Used by VisualizationManager in Phase 3B to propagate cursor from another pane.

  Implementation:
    if self._cursor is not None:
        self._cursor.blockSignals(True)
        self._cursor.setValue(t)
        self._cursor.blockSignals(False)


clear() -> None

  Purpose: Remove record, clear all curves, remove cursor and trigger line.

  Steps:
    self._record = None
    self._time = np.empty(0)
    self._channel_data.clear()
    self.getPlotItem().clear()
    self._curves.clear()
    self._cursor = None
    self._trigger_line = None

─────────────────────────────────────────────────────────────────────────────
REQUIRED PRIVATE METHODS
─────────────────────────────────────────────────────────────────────────────

_on_x_range_changed(viewbox, x_range: tuple[float, float]) -> None

  Purpose: Slot connected to sigXRangeChanged. Triggers viewport update.

  t_start, t_end = x_range
  self._update_viewport(t_start, t_end)


_update_viewport(t_start: float | None = None, t_end: float | None = None) -> None

  Purpose: Decimate and push updated data to all visible curves.

  If t_start/t_end are None, read current viewport from ViewBox:
    vb = self.getPlotItem().getViewBox()
    t_start, t_end = vb.viewRange()[0]

  For each (idx, curve) in self._curves:
    if self._channel_data contains idx:
      t_dec, d_dec = decimate_for_display(
          self._time, self._channel_data[idx],
          t_start, t_end, self._max_pts
      )
      curve.setData(t_dec, d_dec)


_add_trigger_line() -> None

  Create and add the trigger InfiniteLine per VIEWPORT_RENDERING_POLICY §7.
  Call only from set_record().


_add_cursor() -> None

  Create and add the master cursor InfiniteLine per VIEWPORT_RENDERING_POLICY §6.
  Connect sigPositionChanged to _on_cursor_moved.
  Call only from set_record().


_on_cursor_moved(line: pg.InfiniteLine) -> None

  t = line.value()
  self.cursor_moved.emit(t)


_channel_color(ch: AnalogChannel) -> str

  Purpose: Assign a display color to a channel based on its name.

  Simple heuristic (expand in later phases using CHANNEL_MAPPING_POLICY):
    name_lower = ch.name.lower()
    if any(x in name_lower for x in ('va', 'vr', 'v_a', 'v_r', 'phase_a')):
        return '#FF4444'
    elif any(x in name_lower for x in ('vb', 'vy', 'v_b', 'v_y', 'phase_b')):
        return '#FFCC00'
    elif any(x in name_lower for x in ('vc', 'vb_', 'v_c', 'phase_c')):
        return '#4488FF'
    elif any(x in name_lower for x in ('earth', 'zero', 'neutral', 'vn', 'in', 'ir')):
        return '#44BB44'

  Default rotation by index (fallback):
    COLORS = ['#FF4444', '#FFCC00', '#4488FF', '#44BB44', '#AAAAAA', '#FF8800']
    return COLORS[ch.index % len(COLORS)]

─────────────────────────────────────────────────────────────────────────────
FILE 3 — tests/unit/test_downsampling.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Unit tests for decimate_for_display(). No display or Qt required.

Test classes and minimum coverage:

TestDecimateBelowMaxPoints
  - Data with fewer than max_points samples → returned unchanged
  - All samples in viewport → no masking needed
  - Exact max_points samples → no decimation triggered

TestDecimateAboveMaxPoints
  - Data with 10× max_points → output length ≤ max_points
  - Decimated output is a subset of input (not interpolated)
  - First and approximately last samples are preserved

TestDecimateClipping
  - t_start / t_end clips correctly (samples outside range excluded)
  - Entirely out-of-range viewport → returns empty arrays
  - Edge case: t_start == t_end → at most one sample

TestDecimateEdgeCases
  - Empty input arrays → returns empty arrays
  - Single sample within viewport → returns that sample
  - t_start > t_end → silently swapped, result is non-empty

TestDecimateInputValidation
  - Mismatched array lengths → raises ValueError
  - 2-D input arrays → raises ValueError

TestDecimateReturnTypes
  - Output time dtype is float64
  - Output data dtype is float64
  - Input arrays are not modified (check original unchanged)

TOTAL: target ≥ 25 tests.

Implementation note: FastWaveformWidget tests requiring a QApplication and
display are NOT included in this directive — UI widget tests require a GUI
framework which is not available in all CI environments. The downsampling
module is designed to be independently testable for exactly this reason.
If a headless QApplication can be initialized on the target machine, a
separate test_fast_waveform_widget.py may be added in a subsequent directive.

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────

1. Python environment: .venv/Scripts/python.exe exclusively.
   Verify: .venv/Scripts/python.exe -c "import sys; print(sys.executable)"

2. No new dependencies. PyQtGraph, PyQt6, NumPy are already installed.
   Verify before coding: .venv/Scripts/python.exe -c "import pyqtgraph; print(pyqtgraph.__version__)"

3. No imports from src/ under any circumstances.

4. pg.setConfigOptions() goes in app/main.py, not in FastWaveformWidget.__init__.
   Reason: setConfigOptions must run before any widget is created. Widget tests
   that instantiate FastWaveformWidget must call pg.setConfigOptions() in their
   own setUp / fixture, not rely on the widget to do it.

5. sigXRangeChanged fires on every pan/zoom frame. _update_viewport() must be
   fast: the only work allowed is array slicing and setData() calls.
   No DataFrame operations, no .to_numpy() calls, no analytics.

6. skipFiniteCheck=True on PlotDataItem: this disables PyQtGraph's NaN-boundary
   detection loop. Powerwave waveform data is guaranteed numeric (providers
   ensure this), so the check is unnecessary overhead.

7. Comments: write only when the WHY is non-obvious (see CLAUDE.md). Do not
   annotate WHAT the code does. One-line comments only.

─────────────────────────────────────────────────────────────────────────────
REPOSITORY TRACKING UPDATE REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

After successful implementation and all tests passing:

Update agent/HANDOFF.md (append new session entry):
  - Session number (next after current)
  - Files created / modified
  - Tests passing count

Update agent/TASK.md:
  - FastWaveformWidget: NOT STARTED → COMPLETED
  - Add Rendering Optimization: IN PROGRESS (foundation laid)

Update agent/REPOSITORY_STATE.md:
  - Add FastWaveformWidget to IMPLEMENTED SYSTEMS
  - Update visualization/ stub status
  - Update test count
  - Update NEXT REQUIRED ACTION to Phase 3B

─────────────────────────────────────────────────────────────────────────────
COMPLETION REPORT REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

After implementation, provide:

  1. Summary — what was implemented
  2. Files Created / Modified — exhaustive list
  3. Architectural Impact — how this changes the visualization subsystem
  4. Performance Considerations — rendering latency, memory, decimation behavior
  5. Repository Tracking Updates — confirm HANDOFF, TASK, REPOSITORY_STATE updated
  6. Risks / Concerns — Qt display availability for testing, future sync hooks
  7. Next Recommended Step — Phase 3B scope suggestion

─────────────────────────────────────────────────────────────────────────────
PHASE 3A SUCCESS CRITERIA
─────────────────────────────────────────────────────────────────────────────

Phase 3A is complete when:

  ✓ app/visualization/rendering/downsampling.py exists and is fully implemented
  ✓ app/visualization/widgets/fast_waveform_widget.py exists with all required methods
  ✓ tests/unit/test_downsampling.py passes (≥25 tests)
  ✓ All existing tests continue to pass (no regressions)
  ✓ No imports from src/ anywhere in app/visualization/
  ✓ No analytics, no provider imports in visualization/
  ✓ agent/HANDOFF.md, agent/TASK.md, agent/REPOSITORY_STATE.md updated
