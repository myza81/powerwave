implement_visualization_manager.md — Phase 3C Directive
DIRECTIVE AUTHORITY

Issued by: Claude Code (architecture-guided, per VISUALIZATION_CONTRACT.md mandate)
Target agent: Claude / Claude Code
Phase: 3C — VisualizationManager (Coordination Layer)
Status: ISSUED AND EXECUTING
Depends on: Phase 3A (FlexiblePlotCanvas — COMPLETE), Phase 3B (DigitalEventTimeline — COMPLETE)

─────────────────────────────────────────────────────────────────────────────
MANDATORY PRE-READING
─────────────────────────────────────────────────────────────────────────────

  agent/WORKFLOW_AGENT.md
  agent/CLAUDE.md
  agent/REPOSITORY_STATE.md
  docs/ARCHITECTURE.md
  docs/VISUALIZATION_CONTRACT.md
  docs/VIEWPORT_RENDERING_POLICY.md  — especially §6.3, §8, §12
  docs/PERFORMANCE_REQUIREMENTS.md
  docs/LEGACY_CODEBASE_POLICY.md
  directives/implement_flexible_plot_canvas.md
  directives/implement_digital_event_timeline.md

─────────────────────────────────────────────────────────────────────────────
OBJECTIVE
─────────────────────────────────────────────────────────────────────────────

Implement VisualizationManager — the Phase 3C coordination layer that wires
FlexiblePlotCanvas (analog) and DigitalEventTimeline (digital) together.

This component is a pure coordinator. It owns neither widget. It wires them.

─────────────────────────────────────────────────────────────────────────────
PHASE 3C SCOPE
─────────────────────────────────────────────────────────────────────────────

IN SCOPE:

  app/visualization/managers/visualization_manager.py    (coordinator class)
  tests/unit/test_visualization_manager.py               (focused unit tests)

NOT IN SCOPE — do NOT implement:

  SynchronizationManager                         Phase 3D (future)
  cursor_manager.py                              Phase 3D (future)
  viewport_controller.py                         Phase 3D (future)
  Full multi-pane dashboard layout               Phase 4
  Application-wide state manager (AppState)      Phase 4+
  File loading workflow / provider orchestration Phase 4+
  Analytics overlays (RMS, ROCOF)                Phase 5
  Phasor canvas                                  Phase 5+
  Multi-record merging                           Phase 4+
  Measurement panel / annotation engine          Phase 4+
  Event bus / pub-sub system                     Not in scope (ever)

DO NOT MODIFY:
  FlexiblePlotCanvas, DigitalEventTimeline, MultiAxisManager
  DisturbanceRecord, any provider, any analytics module

─────────────────────────────────────────────────────────────────────────────
FILE 1 — app/visualization/managers/visualization_manager.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Coordinates FlexiblePlotCanvas and DigitalEventTimeline. A pure
         Python coordinator class — not a widget, not a QObject, not a
         singleton. No application state.

IMPORTS ALLOWED:
  - app.models (DisturbanceRecord)
  - app.visualization.widgets.flexible_plot_canvas (FlexiblePlotCanvas)
  - app.visualization.widgets.digital_event_timeline (DigitalEventTimeline)
  - Standard library only

IMPORTS FORBIDDEN:
  - Any app.providers.* import
  - Any app.analytics.* import
  - Any app.services.* import
  - Any app.ui.* import
  - Any src.* import
  - pyqtgraph, PyQt6

CLASS: VisualizationManager

  CONSTRUCTOR:

    def __init__(
        self,
        canvas: FlexiblePlotCanvas,
        timeline: DigitalEventTimeline,
    ) -> None:

    Steps:
      1. Store canvas and timeline references.
      2. self._record: DisturbanceRecord | None = None
      3. self._x_linked: bool = False
      4. Wire bidirectional cursor sync:
           canvas.cursor_moved.connect(self._on_canvas_cursor_moved)
           timeline.cursor_moved.connect(self._on_timeline_cursor_moved)

  PROPERTIES (read-only):

    canvas → FlexiblePlotCanvas   — returns self._canvas
    timeline → DigitalEventTimeline — returns self._timeline
    record → DisturbanceRecord | None — returns self._record
    is_x_linked → bool — returns self._x_linked

  PUBLIC METHODS:

  link_x_axis() -> None
    Purpose: Link DigitalEventTimeline X-axis to FlexiblePlotCanvas.
    MUST be called after both widgets are added to a visible Qt scene.
    setXLink() on unparented items is undefined behavior (VIEWPORT_RENDERING_POLICY §8).

    Algorithm:
      self._timeline.link_x_to(self._canvas._primary_plot)
      self._x_linked = True

    After linking: canvas zoom/pan propagates to timeline automatically via
    PyQtGraph X-link. zoom_to_trigger() and reset_viewport() need only act on
    the canvas when linked.

  set_record(record: DisturbanceRecord) -> None
    Algorithm:
      self._record = record
      self._canvas.set_record(record)
      self._timeline.set_record(record)

  clear() -> None
    Algorithm:
      self._record = None
      self._canvas.clear()
      self._timeline.clear()

  zoom_to_trigger(window_s: float = 0.2) -> None
    Algorithm:
      self._canvas.zoom_to_trigger(window_s)
      if not self._x_linked:
          self._zoom_timeline_to_trigger(window_s)

    Rationale: When X-linked, canvas zoom propagates to timeline automatically.
    When not linked (e.g. before link_x_axis() is called), timeline is zoomed
    independently.

  reset_viewport() -> None
    Purpose: Reset the viewport to show the full time range of the record.
    Guard: if self._record is None: return

    Algorithm:
      time_col = self._record.waveform_data["time"]
      t_max = float(time_col.iloc[-1]) if len(time_col) > 0 else 1.0
      self._canvas._primary_plot.setXRange(0.0, t_max, padding=0)
      if not self._x_linked:
          self._timeline.getPlotItem().setXRange(0.0, t_max, padding=0)

  set_cursor_pos(t: float) -> None
    Purpose: Move master cursor on both widgets simultaneously (no signal emit).
    Algorithm:
      self._canvas.set_cursor_pos(t)
      self._timeline.set_cursor_pos(t)

  PRIVATE METHODS:

  _on_canvas_cursor_moved(t: float) -> None
    self._timeline.set_cursor_pos(t)

  _on_timeline_cursor_moved(t: float) -> None
    self._canvas.set_cursor_pos(t)

  _zoom_timeline_to_trigger(window_s: float) -> None
    Purpose: Independently zoom the timeline when not X-linked.
    Guard: if self._record is None: return

    Algorithm:
      t_trig = (
          self._record.timing_info.trigger_time
          - self._record.timing_info.start_time
      ).total_seconds()
      time_col = self._record.waveform_data["time"]
      t_max = float(time_col.iloc[-1]) if len(time_col) > 0 else 0.0
      t_start = max(0.0, t_trig - window_s)
      t_end = min(t_max, t_trig + window_s)
      self._timeline.getPlotItem().setXRange(t_start, t_end, padding=0)

─────────────────────────────────────────────────────────────────────────────
FILE 2 — tests/unit/test_visualization_manager.py
─────────────────────────────────────────────────────────────────────────────

PURPOSE: Unit tests for coordination logic using MagicMock stubs.
         No Qt display required — all widget dependencies are mocked.

Test classes and target coverage:

TestConstruction (6 tests)
  - canvas property returns the canvas passed to constructor
  - timeline property returns the timeline passed to constructor
  - record is None initially
  - is_x_linked is False initially
  - canvas.cursor_moved.connect called with _on_canvas_cursor_moved
  - timeline.cursor_moved.connect called with _on_timeline_cursor_moved

TestSetRecord (4 tests)
  - canvas.set_record called with the record
  - timeline.set_record called with the record
  - record property reflects the loaded record
  - second set_record replaces first (both widgets called twice)

TestClear (4 tests)
  - canvas.clear called on clear()
  - timeline.clear called on clear()
  - record is None after clear()
  - clear() without prior record does not raise

TestCursorSync (5 tests)
  - _on_canvas_cursor_moved forwards t to timeline.set_cursor_pos
  - _on_timeline_cursor_moved forwards t to canvas.set_cursor_pos
  - set_cursor_pos calls canvas.set_cursor_pos(t)
  - set_cursor_pos calls timeline.set_cursor_pos(t)
  - forwarded value is preserved exactly (no transformation)

TestLinkXAxis (3 tests)
  - link_x_axis calls timeline.link_x_to with canvas._primary_plot
  - is_x_linked is True after link_x_axis()
  - is_x_linked is False before link_x_axis() (initial state)

TestZoomToTrigger (4 tests)
  - canvas.zoom_to_trigger called with window_s
  - _zoom_timeline_to_trigger NOT called when x_linked
  - _zoom_timeline_to_trigger called when not x_linked (patch.object)
  - default window_s of 0.2 passed to canvas.zoom_to_trigger

TestResetViewport (4 tests)
  - reset_viewport is no-op when record is None
  - canvas._primary_plot.setXRange called with (0.0, t_max, padding=0)
  - timeline.getPlotItem() NOT called when x_linked
  - timeline.getPlotItem().setXRange called when not x_linked

TestZoomTimelineToTrigger (2 tests)
  - _zoom_timeline_to_trigger is no-op when record is None
  - _zoom_timeline_to_trigger calls timeline.getPlotItem().setXRange

TOTAL: ≥ 30 tests.

─────────────────────────────────────────────────────────────────────────────
CURSOR SYNCHRONIZATION DESIGN
─────────────────────────────────────────────────────────────────────────────

Loop prevention:
  Both FlexiblePlotCanvas.set_cursor_pos() and DigitalEventTimeline.set_cursor_pos()
  use blockSignals(True/False) — they do NOT re-emit cursor_moved.
  Therefore: canvas_cursor_moved → _on_canvas_cursor_moved → timeline.set_cursor_pos()
  does NOT trigger _on_timeline_cursor_moved. No signal loop.

Lifetime contract:
  Qt's pyqtSignal.connect() stores a weak reference to bound methods when the
  receiver is not a QObject. The VisualizationManager instance MUST be kept alive
  by the caller for as long as cursor sync is needed. If the manager is
  garbage-collected, connections are silently dropped.

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────

1. Python environment: .venv/Scripts/python.exe exclusively.
2. No new dependencies.
3. No imports from src/ under any circumstances.
4. VisualizationManager is a plain Python class — no QObject, no singleton,
   no global state. The VISUALIZATION_CONTRACT shows: class VisualizationManager: pass
5. Tests use unittest.mock.MagicMock for canvas and timeline — no display needed.
6. Do NOT call pg.setConfigOptions() or import pyqtgraph in visualization_manager.py.
7. VisualizationManager accesses self._canvas._primary_plot by design — this is
   the established coupling between coordinator and canvas (documented in
   DigitalEventTimeline.link_x_to() docstring).
8. Comments only when WHY is non-obvious. No docstring novels.

─────────────────────────────────────────────────────────────────────────────
REPOSITORY TRACKING UPDATE REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────

After successful implementation and all tests passing:

Update agent/HANDOFF.md — append Session 014 entry.

Update agent/TASK.md:
  - VisualizationManager: NOT STARTED → COMPLETED
  - Update current immediate target

Update agent/REPOSITORY_STATE.md:
  - Add visualization_manager.py to implemented systems
  - Update test count
  - Update NEXT REQUIRED ACTION to Phase 3D or Phase 4

─────────────────────────────────────────────────────────────────────────────
PHASE 3C SUCCESS CRITERIA
─────────────────────────────────────────────────────────────────────────────

  ✓ app/visualization/managers/visualization_manager.py — implemented
  ✓ tests/unit/test_visualization_manager.py — ≥ 30 tests, all passing
  ✓ All 374 existing tests continue to pass (no regressions)
  ✓ No imports from src/ in any new file
  ✓ No analytics, provider, or UI imports in visualization_manager.py
  ✓ agent/HANDOFF.md, agent/TASK.md, agent/REPOSITORY_STATE.md updated
