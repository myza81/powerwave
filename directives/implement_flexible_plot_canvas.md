implement_flexible_plot_canvas.md — Phase 3A Directive
DIRECTIVE AUTHORITY

Issued by: ChatGPT Architecture Orchestrator
Target agent: Claude / Claude Code
Phase: 3A — Visualization Engine Foundation
Status: READY FOR IMPLEMENTATION
Supersedes: directives/implement_fast_waveform_widget.md (archived, do not use)

─────────────────────────────────────────────────────────────────────────────
MANDATORY PRE-READING
─────────────────────────────────────────────────────────────────────────────

Before writing any code, read these documents completely:

  agent/WORKFLOW_AGENT.md
  agent/CLAUDE.md
  agent/REPOSITORY_STATE.md
  docs/VISUALIZATION_CONTRACT.md
  docs/VIEWPORT_RENDERING_POLICY.md       — especially §2, §3, §4, §8, §13, §16, §17
  docs/PERFORMANCE_REQUIREMENTS.md
  docs/ARCHITECTURE.md
  docs/DATA_CONTRACT.md
  docs/CHANNEL_MAPPING_POLICY.md

After reading, you must understand:
  - FlexiblePlotCanvas inherits pg.GraphicsLayoutWidget (NOT PlotWidget)
  - N-Axis architecture: one ViewBox per analog parameter, shared X-axis (§16)
  - Digital signals go in DigitalEventTimeline (Phase 3B) — NOT in FlexiblePlotCanvas (§17)
  - Curves use setData() lifecycle — never remove/re-add (§3)
  - Display decimation caps at 4 000 points per curve (§4)
  - Time/data arrays must be cached on record load, not re-extracted per frame (§11)
  - DisturbanceRecord field names: waveform_data, timing_info, analog_channels (§11)
  - pg.setConfigOptions() goes in app/main.py, not in any widget constructor

─────────────────────────────────────────────────────────────────────────────
OBJECTIVE
─────────────────────────────────────────────────────────────────────────────

Implement the FlexiblePlotCanvas widget, MultiAxisManager helper, and display
decimation module as the Phase 3A foundation of the Powerwave visualization engine.

Architecture: SIGRA-style N-Axis Single Canvas.
  - All analog channels rendered in one canvas
  - Each channel gets its own independent ViewBox and color-coded AxisItem
  - All ViewBoxes share the same X-axis (time domain) via setXLink
  - Independent vertical scaling per channel
  - No digital signals in this widget

─────────────────────────────────────────────────────────────────────────────
PHASE 3A SCOPE
─────────────────────────────────────────────────────────────────────────────

IN SCOPE — implement these:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  app/visualization/rendering/downsampling.py                             │
  │  app/visualization/widgets/flexible_plot_canvas.py                       │
  │  app/visualization/managers/multi_axis_manager.py                        │
  │  tests/unit/test_downsampling.py                                         │
  └──────────────────────────────────────────────────────────────────────────┘

NOT IN SCOPE — do NOT implement in Phase 3A:

  Digital signal rendering (DigitalEventTimeline)                Phase 3B
  Visualization manager (multi-widget coordination)              Phase 3B
  Synchronization manager                                        Phase 3B
  Cursor manager (cross-widget cursor sync)                      Phase 3B
  Full multi-pane dashboard                                      Phase 3B
  UI docking framework                                           Phase 4
  Application-wide state manager (AppState)                      Phase 4
  Phasor canvas                                                  Phase 5+
  RMS / ROCOF / harmonic analytics overlay                       Phase 5
  PMU streaming rendering                                        Phase 6+
  Measurement tools                                              Phase 4+

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

ALGORITHM REQUIREMENTS (per VIEWPORT_RENDERING_POLICY §4):

  1. Input validation:
     - Both arrays must be 1-D and equal length. Raise ValueError otherwise.
     - If either array is empty, return (np.empty(0, dtype=np.float64),
                                         np.empty(0, dtype=np.float64)).
     - If t_start > t_end: swap them silently.

  2. Clipping via boolean mask:
     mask = (time >= t_start) & (time <= t_end)
     t_clip = time[mask]
     d_clip = data[mask]

  3. If len(t_clip) == 0: return (np.empty(0, dtype=np.float64),
                                   np.empty(0, dtype=np.float64)).

  4. If len(t_clip) <= max_points: return (t_clip.astype(np.float64),
                                            d_clip.astype(np.float64)).

  5. Stride decimation:
     stride = max(1, len(t_clip) // max_points)
     return t_clip[::stride].astype(np.float64), d_clip[::stride].astype(np.float64)

CONSTRAINTS:
  - FULLY VECTORIZED. No Python loops over samples.
  - Return dtype must be float64 for both arrays.
  - Input arrays must not be modified (check: return views or copies, not in-place).
  - No side effects.

─────────────────────────────────────────────────────────────────────────────
FILE 2 — app/visualization/managers/multi_axis_manager.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Manages ViewBox and AxisItem lifecycle for the N-Axis Single Canvas.
         This is a pure Python helper class — not a widget, not a QObject.

DATACLASS for axis metadata:

  @dataclasses.dataclass
  class _AxisEntry:
      name: str
      viewbox: pg.ViewBox
      axis_item: pg.AxisItem
      curve: pg.PlotDataItem
      color: str

CLASS: MultiAxisManager

  def __init__(self, primary_plot: pg.PlotItem, layout: pg.GraphicsLayoutWidget) -> None:
      self._primary = primary_plot
      self._layout = layout
      self._axes: dict[str, _AxisEntry] = {}   # name → _AxisEntry
      self._right_col = 2    # next available right column in the layout grid
      # Connect geometry sync to primary ViewBox resize
      self._primary.getViewBox().sigResized.connect(self._sync_geometries)

  def add_axis(self, name: str, unit: str, color: str) -> pg.ViewBox:
      """Create and register a new ViewBox + AxisItem for a parameter.

      Returns the new ViewBox. The caller creates the PlotDataItem and
      adds it to the returned ViewBox.
      """
      if name in self._axes:
          raise ValueError(f"Parameter '{name}' already registered")

      # First parameter uses the primary PlotItem's axis — no new ViewBox needed
      if not self._axes:
          # Primary axis already exists; return primary ViewBox directly
          self._primary.setLabel('left', name, units=unit)
          self._primary.getAxis('left').setPen(pg.mkPen(color))
          return self._primary.getViewBox()

      # Secondary parameters: new ViewBox
      vb = pg.ViewBox()
      vb.setXLink(self._primary)
      self._primary.scene().addItem(vb)

      axis = pg.AxisItem(orientation='right')
      axis.setLabel(name, units=unit)
      axis.setPen(pg.mkPen(color))
      axis.setTextPen(pg.mkPen(color))
      axis.linkToView(vb)

      self._layout.addItem(axis, row=0, col=self._right_col)
      self._right_col += 1

      # Initial geometry sync
      vb.setGeometry(self._primary.getViewBox().sceneBoundingRect())
      vb.linkedViewChanged(self._primary.getViewBox(), vb.XAxis)

      return vb

  def register(self, name: str, viewbox: pg.ViewBox, axis_item: pg.AxisItem,
               curve: pg.PlotDataItem, color: str) -> None:
      """Register the completed axis entry after add_axis() + curve creation."""
      self._axes[name] = _AxisEntry(
          name=name, viewbox=viewbox, axis_item=axis_item, curve=curve, color=color
      )

  def remove_axis(self, name: str) -> None:
      """Remove a parameter's ViewBox, AxisItem, and curve from the scene."""
      if name not in self._axes:
          return
      entry = self._axes.pop(name)
      entry.viewbox.scene().removeItem(entry.viewbox)
      entry.axis_item.scene().removeItem(entry.axis_item)

  def get_viewboxes(self) -> list[pg.ViewBox]:
      """Return all secondary ViewBoxes (excludes primary)."""
      primary_vb = self._primary.getViewBox()
      return [e.viewbox for e in self._axes.values() if e.viewbox is not primary_vb]

  def get_curves(self) -> dict[str, pg.PlotDataItem]:
      """Return name → curve mapping for all registered parameters."""
      return {name: e.curve for name, e in self._axes.items()}

  def _sync_geometries(self) -> None:
      """Update all secondary ViewBox geometries to match the primary ViewBox.
      Called automatically via sigResized — see VIEWPORT_RENDERING_POLICY §16.4.
      """
      scene_rect = self._primary.getViewBox().sceneBoundingRect()
      primary_vb = self._primary.getViewBox()
      for entry in self._axes.values():
          if entry.viewbox is not primary_vb:
              entry.viewbox.setGeometry(scene_rect)
              entry.viewbox.linkedViewChanged(primary_vb, entry.viewbox.XAxis)

─────────────────────────────────────────────────────────────────────────────
FILE 3 — app/visualization/widgets/flexible_plot_canvas.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: N-Axis Single Canvas for analog waveform rendering.
         Renders all analog channels from a DisturbanceRecord in a single
         canvas with independent Y-axes, shared X-axis.
         Does NOT render digital channels.

IMPORTS ALLOWED:
  - PyQt6, pyqtgraph, numpy
  - app.models (DisturbanceRecord, AnalogChannel)
  - app.visualization.rendering.downsampling (decimate_for_display)
  - app.visualization.managers.multi_axis_manager (MultiAxisManager)
  - Standard library

IMPORTS FORBIDDEN:
  - Any app.providers.* import
  - Any app.analytics.* import
  - Any src.* import

─────────────────────────────────────────────────────────────────────────────
CLASS: FlexiblePlotCanvas(pg.GraphicsLayoutWidget)
─────────────────────────────────────────────────────────────────────────────

pyqtSignal:

  cursor_moved = pyqtSignal(float)   # emits cursor time in seconds

CONSTRUCTOR:

  def __init__(self, parent=None, max_display_points: int = 4_000) -> None:

  Steps (in order):
    1. super().__init__(parent=parent, background='#1E1E1E')
    2. self._max_pts = max_display_points
    3. Initialise state:
         self._record: DisturbanceRecord | None = None
         self._time_cache: np.ndarray = np.empty(0, dtype=np.float64)
         self._data_cache: dict[str, np.ndarray] = {}   # ch.name → float64 array
         self._cursor: pg.InfiniteLine | None = None
         self._trigger_line: pg.InfiniteLine | None = None
    4. Create primary PlotItem in the layout:
         self._primary_plot = self.addPlot(row=0, col=0)
         self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
         self._primary_plot.setLabel('bottom', 'Time', units='s')
    5. Create MultiAxisManager:
         self._axis_manager = MultiAxisManager(self._primary_plot, self)
    6. Connect viewport change:
         self._primary_plot.getViewBox().sigXRangeChanged.connect(
             self._on_x_range_changed
         )

─────────────────────────────────────────────────────────────────────────────
REQUIRED PUBLIC METHODS
─────────────────────────────────────────────────────────────────────────────

set_record(record: DisturbanceRecord) -> None

  Purpose: Load a new record. Build all axes and curves. Zoom to trigger.

  Steps:
    1. Clear existing state (call self.clear()).
    2. Store record: self._record = record
    3. Cache numpy arrays ONCE:
         self._time_cache = record.waveform_data['time'].to_numpy(dtype=np.float64)
         self._data_cache = {
             ch.name: record.waveform_data[ch.name].to_numpy(dtype=np.float64)
             for ch in record.analog_channels
         }
    4. For each ch in record.analog_channels:
         a. Determine color: _channel_color(ch)
         b. vb = self._axis_manager.add_axis(ch.name, ch.unit, color)
         c. Create curve:
              if vb is self._primary_plot.getViewBox():
                  curve = pg.PlotDataItem(pen=..., skipFiniteCheck=True)
                  curve.setClipToView(True)
                  self._primary_plot.addItem(curve)
              else:
                  curve = pg.PlotDataItem(pen=..., skipFiniteCheck=True)
                  curve.setClipToView(True)
                  vb.addItem(curve)
         d. Register: self._axis_manager.register(ch.name, vb, axis_item, curve, color)
            NOTE: For the primary ViewBox, axis_item = self._primary_plot.getAxis('left')
    5. Add trigger line: self._add_trigger_line()
    6. Add cursor: self._add_cursor()
    7. Zoom to trigger: self.zoom_to_trigger()
    8. Update viewport: self._update_viewport()


add_parameter(name: str, data: np.ndarray, unit: str = 'unknown',
              color: str | None = None) -> None

  Purpose: Add a single parameter to the canvas without a full DisturbanceRecord.
           Useful for overlaying derived signals or analytics results.

  Steps:
    1. If name already registered, call remove_parameter(name) first.
    2. Determine color (use argument or rotate from COLORS list).
    3. vb = self._axis_manager.add_axis(name, unit, effective_color)
    4. Create and add curve to vb (or primary if first param).
    5. Cache data: self._data_cache[name] = data.astype(np.float64)
    6. Call _update_viewport() to populate curve.


remove_parameter(name: str) -> None

  Purpose: Remove a parameter's ViewBox, axis, and curve from the canvas.

  Steps:
    1. Call self._axis_manager.remove_axis(name)
    2. Remove from self._data_cache.
    3. No viewport update needed — axis is gone.


set_visible_channels(names: list[str]) -> None

  Purpose: Show or hide parameters by name.

  For each name in self._axis_manager.get_curves():
    if name in names:
      Repopulate curve via _update_curve(name) with current viewport.
    else:
      Hide: curve.setData(np.empty(0), np.empty(0))


zoom_to_trigger(window_s: float = 0.2) -> None

  Purpose: Set X-range centred on trigger time ± window_s.
  Implement per VIEWPORT_RENDERING_POLICY §12.
  Guard: if self._record is None or len(self._time_cache) == 0: return.
  Call self._primary_plot.setXRange(t_start, t_end, padding=0).


set_cursor_pos(t: float) -> None

  Purpose: Move cursor without re-emitting cursor_moved (for Phase 3B sync).

  if self._cursor is not None:
      self._cursor.blockSignals(True)
      self._cursor.setValue(t)
      self._cursor.blockSignals(False)


clear() -> None

  Purpose: Remove all parameters, cursor, trigger line. Reset state.

  Steps:
    1. self._primary_plot.clear()   # removes all items including curves/lines
    2. For each secondary ViewBox in _axis_manager.get_viewboxes():
         Remove all items from the ViewBox.
         Remove the ViewBox from the scene.
    3. Re-add primary axis config after clear():
         self._primary_plot.showGrid(x=True, y=True, alpha=0.2)
         self._primary_plot.setLabel('bottom', 'Time', units='s')
    4. Rebuild MultiAxisManager (reset to clean state):
         self._axis_manager = MultiAxisManager(self._primary_plot, self)
    5. Reset cache:
         self._record = None
         self._time_cache = np.empty(0, dtype=np.float64)
         self._data_cache.clear()
         self._cursor = None
         self._trigger_line = None

─────────────────────────────────────────────────────────────────────────────
REQUIRED PRIVATE METHODS
─────────────────────────────────────────────────────────────────────────────

_on_x_range_changed(viewbox, x_range: tuple[float, float]) -> None

  t_start, t_end = x_range
  self._update_viewport(t_start, t_end)


_update_viewport(t_start: float | None = None, t_end: float | None = None) -> None

  If t_start/t_end are None, read from primary ViewBox:
    t_start, t_end = self._primary_plot.getViewBox().viewRange()[0]

  For each (name, curve) in self._axis_manager.get_curves().items():
      if name in self._data_cache:
          t_dec, d_dec = decimate_for_display(
              self._time_cache, self._data_cache[name],
              t_start, t_end, self._max_pts
          )
          curve.setData(t_dec, d_dec)


_add_trigger_line() -> None

  Per VIEWPORT_RENDERING_POLICY §7:
  Compute trigger_time_s from self._record.timing_info.
  Create InfiniteLine(movable=False, pen='#FF4444', width=2, DotLine, label='T').
  Add to self._primary_plot (NOT to secondary ViewBoxes).
  Store as self._trigger_line.


_add_cursor() -> None

  Per VIEWPORT_RENDERING_POLICY §6:
  Create InfiniteLine(movable=True, pen='#FFFF00', width=1.5, DashLine).
  Connect sigPositionChanged to self._on_cursor_moved.
  Add to self._primary_plot.
  Store as self._cursor.


_on_cursor_moved(line: pg.InfiniteLine) -> None

  self.cursor_moved.emit(line.value())


_channel_color(ch: AnalogChannel) -> str

  Per VIEWPORT_RENDERING_POLICY §9 and CHANNEL_MAPPING_POLICY color guidance.
  Simple heuristic — use channel name to detect phase:

    name = ch.name.lower()
    if any(x in name for x in ('_a', 'va', 'ia', 'vr', 'ir', 'phase_a', 'ph_a')):
        return '#FF4444'   # Phase A — red
    if any(x in name for x in ('_b', 'vb', 'ib', 'vy', 'iy', 'phase_b', 'ph_b')):
        return '#FFCC00'   # Phase B — amber
    if any(x in name for x in ('_c', 'vc', 'ic', 'vb_', 'phase_c', 'ph_c')):
        return '#4488FF'   # Phase C — blue
    if any(x in name for x in ('earth', 'zero', 'neutral', 'vn', 'in_', '3i0', '3u0')):
        return '#44BB44'   # Earth/zero-sequence — green

    # Fallback rotation by channel index
    COLORS = ['#FF4444', '#FFCC00', '#4488FF', '#44BB44', '#AAAAAA', '#FF8800']
    return COLORS[ch.index % len(COLORS)]

─────────────────────────────────────────────────────────────────────────────
FILE 4 — tests/unit/test_downsampling.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Unit tests for decimate_for_display(). No display or Qt required.

Test classes (minimum):

TestDecimateBelowMaxPoints (3 tests)
  - Input with fewer than max_points samples → returned unchanged (same length)
  - Input with exactly max_points samples → no decimation triggered
  - Input with 1 sample → returned as-is

TestDecimateAboveMaxPoints (4 tests)
  - 10× max_points → output length ≤ max_points
  - Decimated output values are a subset of input values (not interpolated)
  - Output time values are a subset of input time values
  - Stride decimation: output[0] == input[0] (first sample preserved)

TestDecimateClipping (5 tests)
  - t_start/t_end clips correctly (samples outside range excluded)
  - All samples after t_end excluded
  - All samples before t_start excluded
  - Entirely out-of-range viewport → returns empty arrays (length 0)
  - t_start == t_end → at most one sample returned

TestDecimateEdgeCases (5 tests)
  - Empty time array → returns two empty arrays
  - Empty data array → returns two empty arrays
  - Single sample within viewport → returns that sample
  - t_start > t_end → silently swapped, returns non-empty result
  - Very large dataset (1M points) → returns ≤ max_points without error

TestDecimateInputValidation (4 tests)
  - Mismatched array lengths → raises ValueError
  - 2-D time array → raises ValueError
  - 2-D data array → raises ValueError
  - Scalar input (0-D) → raises ValueError

TestDecimateReturnTypes (5 tests)
  - Output time dtype is float64
  - Output data dtype is float64
  - Float32 input → float64 output
  - Integer input → float64 output
  - Input arrays not modified (originals unchanged after call)

TOTAL: target ≥ 26 tests.

Note on FastWaveformWidget / FlexiblePlotCanvas widget tests:
  UI widget tests require a QApplication and display server. These are NOT
  included in Phase 3A. The downsampling module is designed to be fully
  testable without a display, isolating the critical performance-path logic.
  Widget integration tests will be added as a separate test file when a
  headless display environment is confirmed.

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────

1. Python environment: .venv/Scripts/python.exe exclusively.
   Verify: .venv/Scripts/python.exe -c "import sys; print(sys.executable)"

2. No new dependencies. PyQtGraph, PyQt6, NumPy already installed.
   Verify: .venv/Scripts/python.exe -c "import pyqtgraph; print(pyqtgraph.__version__)"

3. No imports from src/ under any circumstances.

4. pg.setConfigOptions() belongs in app/main.py. The widget MUST NOT call it
   in __init__. Widget-level tests that need PyQtGraph must call setConfigOptions
   in their own setUp or use a conftest.py fixture.

5. _on_x_range_changed fires on every pan/zoom frame. _update_viewport() must
   only perform: read viewport range → decimate → setData(). No DataFrame ops,
   no to_numpy(), no analytics, no file I/O.

6. skipFiniteCheck=True on all PlotDataItem instances (waveform data is
   guaranteed numeric by providers — NaN detection loop is unnecessary overhead).

7. Comments: only when WHY is non-obvious. No docstring novels. No WHAT comments.

8. The add_parameter() / remove_parameter() API is designed for future use by
   analytics overlays (Phase 5) and multi-record comparison (Phase 4+). Implement
   it correctly even though no caller uses it in Phase 3A.

9. MultiAxisManager right-column axis placement: for Phase 3A, all secondary
   axes stack on the right margin (columns 2, 3, 4...). Left/right alternation
   is a Phase 3B enhancement.

─────────────────────────────────────────────────────────────────────────────
REPOSITORY TRACKING UPDATE REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

After successful implementation and all tests passing:

Update agent/HANDOFF.md — append new session entry with:
  - Session number
  - Files created / modified
  - Test count (passing)
  - Key implementation decisions

Update agent/TASK.md:
  - FastWaveformWidget task: mark as SUPERSEDED (replaced by FlexiblePlotCanvas)
  - Add FlexiblePlotCanvas task entry → COMPLETED
  - Multi-Pane Synchronization: update to note Phase 3B dependency on FlexiblePlotCanvas

Update agent/REPOSITORY_STATE.md:
  - Add FlexiblePlotCanvas, MultiAxisManager, downsampling to IMPLEMENTED SYSTEMS
  - Update visualization/ stub status to PARTIALLY IMPLEMENTED (Phase 3A)
  - Update test count
  - Update NEXT REQUIRED ACTION to Phase 3B

─────────────────────────────────────────────────────────────────────────────
COMPLETION REPORT REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

After implementation, provide:

  1. Summary — what was implemented
  2. Files Created / Modified — exhaustive list
  3. Architectural Impact — how N-Axis canvas changes the visualization subsystem
  4. Performance Considerations — per-viewport rendering cost with N ViewBoxes
  5. Repository Tracking Updates — confirm HANDOFF, TASK, REPOSITORY_STATE updated
  6. Risks / Concerns — known PyQtGraph ViewBox edge cases, widget test limitations
  7. Next Recommended Step — Phase 3B scope (DigitalEventTimeline + VisualizationManager)

─────────────────────────────────────────────────────────────────────────────
PHASE 3A SUCCESS CRITERIA
─────────────────────────────────────────────────────────────────────────────

Phase 3A is complete when:

  ✓ app/visualization/rendering/downsampling.py — implemented and tested
  ✓ app/visualization/managers/multi_axis_manager.py — implemented
  ✓ app/visualization/widgets/flexible_plot_canvas.py — implemented
  ✓ tests/unit/test_downsampling.py — ≥26 tests, all passing
  ✓ All existing tests continue to pass (307 tests, no regressions)
  ✓ No imports from src/ in any app/visualization/ file
  ✓ No analytics, no provider imports in visualization/
  ✓ agent/HANDOFF.md, agent/TASK.md, agent/REPOSITORY_STATE.md updated
