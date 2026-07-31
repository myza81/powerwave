VIEWPORT_RENDERING_POLICY.md — Powerwave Rendering Engineering Policy
PURPOSE

This document defines the mandatory low-level rendering engineering rules for
Powerwave's visualization subsystem.

It complements VISUALIZATION_CONTRACT.md, which defines what the visualization
engine must do. This document defines how it must be implemented — covering
PyQtGraph initialization, curve lifecycle, decimation, cursor and trigger patterns,
dark theme, threading, and common mistakes to avoid.

This is the authoritative implementation reference for:

  app/visualization/widgets/fast_waveform_widget.py
  app/visualization/rendering/downsampling.py
  app/visualization/managers/visualization_manager.py
  app/visualization/interaction/cursor_manager.py

All implementation agents SHALL read this policy before touching any rendering code.

─────────────────────────────────────────────────────────────────────────────
§1 — PYQTGRAPH GLOBAL CONFIGURATION
─────────────────────────────────────────────────────────────────────────────

PyQtGraph global options MUST be set once at application startup, before any
widget is instantiated. This call belongs in app/main.py or the top-level
application bootstrap — never inside a widget constructor.

MANDATORY CALL:

  import pyqtgraph as pg

  pg.setConfigOptions(
      useOpenGL=True,      # REQUIRED — enables GPU rendering pipeline
      antialias=False,     # REQUIRED — keep False; antialiasing kills performance
                           #            at high point counts (>100k samples/channel)
      foreground='w',
      background='#1E1E1E',
  )

These options MUST be set before any pg.PlotWidget or pg.GraphicsLayoutWidget
is created. They cannot be changed at runtime after widget creation.

RATIONALE — antialias=False:
  Antialiasing in PyQtGraph applies per-line-segment CPU blending, which is
  catastrophic for waveforms with millions of samples. The dark background +
  thin lines (width=1) look sharp without antialiasing at any display density.
  Antialiasing may be considered ONLY for static export/print rendering, never
  for interactive display.

─────────────────────────────────────────────────────────────────────────────
§2 — WIDGET ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────

§2.1 FlexiblePlotCanvas — N-Axis Single Canvas (analog signals)

CANONICAL INHERITANCE:

  class FlexiblePlotCanvas(pg.GraphicsLayoutWidget):
      ...

FlexiblePlotCanvas renders ALL analog parameters in a single canvas with N
independent Y-axes (one ViewBox per parameter, shared X-axis). This is the
SIGRA-style N-Axis Single Canvas architecture mandated by VISUALIZATION_CONTRACT.md.

pg.GraphicsLayoutWidget is the correct base because it provides scene management,
margin layout, and axis placement needed for the multi-axis pattern.

Digital signals are NOT rendered in FlexiblePlotCanvas. They have a separate
dedicated component — see §17 and VISUALIZATION_CONTRACT.md Digital Event Timeline.

§2.2 MultiAxisManager — ViewBox and axis lifecycle manager

  class MultiAxisManager:
      ...

MultiAxisManager is a helper class (not a widget) responsible for:
  - Creating and tracking ViewBoxes for each parameter
  - Linking secondary ViewBoxes to the primary ViewBox (setXLink)
  - Adding and positioning AxisItem instances
  - Updating secondary ViewBox geometries on scene resize

§2.3 Required instance fields of FlexiblePlotCanvas

  class FlexiblePlotCanvas(pg.GraphicsLayoutWidget):
      _record: DisturbanceRecord | None
      _axis_manager: MultiAxisManager
      _curves: dict[str, pg.PlotDataItem]         # param name → curve
      _time_cache: np.ndarray                     # cached time array (float64)
      _data_cache: dict[str, np.ndarray]          # param name → float64 array
      _cursor: pg.InfiniteLine | None             # movable master time cursor
      _trigger_line: pg.InfiniteLine | None       # static trigger marker
      _primary_plot: pg.PlotItem                  # hosts X-axis + first parameter

─────────────────────────────────────────────────────────────────────────────
§3 — CURVE LIFECYCLE RULE (CRITICAL)
─────────────────────────────────────────────────────────────────────────────

LAW: ALWAYS update curves using setData(). NEVER remove and re-add items.

CORRECT (GPU-efficient in-place update):
  curve.setData(x_decimated, y_decimated)

FORBIDDEN (causes visual flicker, destroys GPU buffer, kills performance):
  self.removeItem(curve)
  new_curve = pg.PlotDataItem(...)
  self.addItem(new_curve)

This law applies to ALL curve updates — viewport pan, zoom, channel toggle,
new record load, cursor movement, everything.

Exception — initial curve creation:
  Curves are created ONCE per channel when set_record() is called, via
  pg.PlotDataItem(pen=...), then added to the plot with addItem(). After
  that initial creation, only setData() is ever called on them.

Curve creation pattern:
  curve = pg.PlotDataItem(
      pen=pg.mkPen(color, width=1),
      skipFiniteCheck=True,   # skip NaN-boundary detection — faster
  )
  self.addItem(curve)
  self._curves[ch.index] = curve

─────────────────────────────────────────────────────────────────────────────
§4 — DISPLAY DECIMATION POLICY
─────────────────────────────────────────────────────────────────────────────

§4.1 Maximum display points per viewport

The rendering engine SHALL decimate waveform data before passing to setData().

TARGET: ≤ 4 000 points per curve per viewport update.

RATIONALE:
  A typical monitor shows at most 2560 horizontal pixels. Rendering more than
  ~4000 points per curve produces no additional visual information but degrades
  GPU throughput significantly. The 4000-point budget provides a 1.5× overdraw
  safety margin.

This limit is a practical heuristic. Future implementations may adapt the limit
based on actual widget pixel width (e.g., width_px × 2.0).

§4.2 Decimation function signature (canonical)

  def decimate_for_display(
      time: np.ndarray,
      data: np.ndarray,
      t_start: float,
      t_end: float,
      max_points: int = 4_000,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Clip waveform to [t_start, t_end] and decimate to at most max_points.

      Returns (time_decimated, data_decimated) as float64 arrays.
      Both input arrays must be the same length.
      """

Implementation requirements:
  1. Boolean mask: mask = (time >= t_start) & (time <= t_end)
  2. If visible samples ≤ max_points: return mask-filtered arrays directly.
  3. If visible samples > max_points: apply integer stride decimation.
     stride = max(1, len_visible // max_points)
     Return time[mask][::stride], data[mask][::stride]
  4. No interpolation. Stride decimation preserves waveform peak fidelity
     better than averaging for transient disturbance signals.
  5. MUST be vectorized (NumPy boolean mask + fancy index). No Python loops.

§4.3 DisturbanceRecord access pattern for decimation

New DisturbanceRecord uses waveform_data (a pandas DataFrame):

  time_array = record.waveform_data['time'].to_numpy()
  ch_data    = record.waveform_data[ch.name].to_numpy()

These to_numpy() calls return views where possible (zero-copy). The time and
channel columns must be extracted ONCE on record load and cached as numpy arrays
to avoid repeated DataFrame column extraction during viewport updates.

§4.4 When to re-decimate

Re-decimation is triggered by:
  - Viewport pan (x-range changed)
  - Viewport zoom (x-range changed)
  - New record loaded (full redraw)
  - Channel visibility toggled (show: populate; hide: setData([], []))

NOT triggered by:
  - Y-axis zoom (vertical range change — no x-decimation needed)
  - Cursor movement (cursor does not affect waveform data)
  - Window resize (acceptable to leave at previous decimation level)

─────────────────────────────────────────────────────────────────────────────
§5 — CLIP-TO-VIEW POLICY
─────────────────────────────────────────────────────────────────────────────

Clip-to-view rendering is achieved by the decimation function (§4) — only
samples within [t_start, t_end] are passed to setData(). This is the primary
clip-to-view mechanism and requires no additional PyQtGraph configuration.

PyQtGraph's built-in clip-to-view option (PlotDataItem.setClipToView) may be
enabled as a secondary guard:

  curve.setClipToView(True)

This should be set on curve creation. It instructs PyQtGraph's C++ layer to
skip off-screen line segments, providing a secondary rendering speedup on top
of the Python-level decimation.

The two clip mechanisms are complementary. Both should be enabled.

─────────────────────────────────────────────────────────────────────────────
§6 — CURSOR RENDERING RULES
─────────────────────────────────────────────────────────────────────────────

§6.1 Master time cursor (movable)

CANONICAL CONSTRUCTION:

  cursor = pg.InfiniteLine(
      pos=0.0,
      angle=90,
      movable=True,
      pen=pg.mkPen('#FFFF00', width=1.5, style=Qt.PenStyle.DashLine),
  )

Colors:
  Primary cursor (cursor_a):   '#FFFF00'  (yellow)
  Secondary cursor (cursor_b): '#FF8800'  (orange)

§6.2 Cursor position propagation

When the cursor is dragged, it emits sigPositionChanged. The handler MUST
NOT perform heavy computation inline — it must propagate the position
via a Qt signal to any registered observers:

  cursor.sigPositionChanged.connect(self._on_cursor_moved)

  def _on_cursor_moved(self, line: pg.InfiniteLine) -> None:
      t = line.value()
      self.cursor_moved.emit(t)   # pyqtSignal(float)

The signal is the boundary. Any cursor-time-dependent computation
(RMS readout, measurement panel update) happens in slots that receive
this signal — never inline in the sigPositionChanged handler.

§6.3 Multi-pane cursor synchronization

When multiple FastWaveformWidget instances are managed by VisualizationManager,
cursor synchronization is achieved by connecting the cursor_moved signal to
a slot in the VisualizationManager that forwards the position to all other
widgets. Each widget provides a set_cursor_pos(t: float) method that moves
its InfiniteLine without re-emitting sigPositionChanged (to avoid signal loops).

This synchronization architecture is Phase 3B scope. FastWaveformWidget
implements cursor_moved signal emission (Phase 3A scope) but the cross-widget
wiring is deferred.

─────────────────────────────────────────────────────────────────────────────
§7 — TRIGGER LINE RULES
─────────────────────────────────────────────────────────────────────────────

CANONICAL CONSTRUCTION:

  trigger_line = pg.InfiniteLine(
      pos=trigger_time_s,
      angle=90,
      movable=False,
      pen=pg.mkPen('#FF4444', width=2, style=Qt.PenStyle.DotLine),
      label='T',
      labelOpts={'color': '#FF4444', 'position': 0.95},
  )

Rules:
  - ALWAYS movable=False — the trigger is a fixed event marker
  - Position is in seconds relative to record start:
      trigger_time_s = (
          record.timing_info.trigger_time - record.timing_info.start_time
      ).total_seconds()
  - If timing_info.trigger_time == timing_info.start_time, the trigger line
    is still rendered at t=0. This is intentional — it marks the recording
    start when no separate trigger time is available.
  - The trigger line is added to the same PlotItem as the waveform curves.
    It must be added AFTER curves to ensure it renders on top.
  - On record clear or new record load, the old trigger line MUST be removed
    with removeItem() before creating the new one. Unlike curves, trigger
    lines are recreated on each new record (there is exactly one per widget).

─────────────────────────────────────────────────────────────────────────────
§8 — X-AXIS SYNCHRONIZATION
─────────────────────────────────────────────────────────────────────────────

Within FlexiblePlotCanvas, X-axis synchronization between secondary ViewBoxes
and the primary PlotItem is achieved via setXLink (see §16.3 for full pattern).

  secondary_viewbox.setXLink(primary_plot)

This links the secondary ViewBox's X-range to the primary PlotItem's ViewBox,
ensuring all pan/zoom operations propagate automatically at the PyQtGraph/C++
level — zero Python overhead per pan event.

For multi-pane dashboard coordination (VisualizationManager scope, Phase 3B+),
separate PlotItems across multiple widgets are synchronized via:
  plot_item_b.setXLink(plot_item_a)

setXLink() MUST be called after both the source and target have been added to
a scene. Calling it on an unparented item is undefined behavior in PyQtGraph.

The link is one-directional in declaration but bidirectional in behavior —
any pan/zoom on either linked item updates both.

─────────────────────────────────────────────────────────────────────────────
§9 — DARK ENGINEERING THEME
─────────────────────────────────────────────────────────────────────────────

Powerwave uses a dark engineering theme. These are the canonical color values.

Background and grid:
  BACKGROUND   = '#1E1E1E'    # near-black canvas
  GRID_COLOR   = '#333333'    # dark grey grid lines

Phase waveform colors (analog channels):
  PHASE_A      = '#FF4444'    # red   — R/A phase voltage/current
  PHASE_B      = '#FFCC00'    # amber — Y/B phase voltage/current
  PHASE_C      = '#4488FF'    # blue  — B/C phase voltage/current
  EARTH_ZERO   = '#44BB44'    # green — earth/zero-sequence/neutral

Digital signal colors:
  DIG_GENERIC  = '#AAAAAA'    # grey  — generic digital channels
  DIG_TRIP     = '#FF2222'    # red   — trip signals
  DIG_CB       = '#FF8800'    # orange — CB status
  DIG_PICKUP   = '#FFAA00'    # amber — relay pickup

Cursor and marker colors:
  CURSOR_A     = '#FFFF00'    # yellow — primary master cursor
  CURSOR_B     = '#FF8800'    # orange — secondary cursor
  TRIGGER      = '#FF4444'    # red    — trigger line

Text and axis:
  TEXT_COLOR   = '#DDDDDD'
  AXIS_COLOR   = '#888888'

Grid configuration:
  plot.showGrid(x=True, y=True, alpha=0.2)

Alpha=0.2 ensures the grid is visible without competing with waveforms.

Channel color assignment:
  Phase channels are detected by name (see CHANNEL_MAPPING_POLICY.md §3 for
  phase naming conventions). Default assignment for unrecognized channels
  rotates through PHASE_A, PHASE_B, PHASE_C, EARTH_ZERO in channel index order.
  Digital channels use DIG_GENERIC unless a signal role is detected.

─────────────────────────────────────────────────────────────────────────────
§10 — UI THREAD PROTECTION RULES
─────────────────────────────────────────────────────────────────────────────

§10.1 Law: heavy operations MUST NOT run on the UI thread

Operations that block the Qt event loop cause the application to freeze,
invalidating the UI responsiveness requirement in PERFORMANCE_REQUIREMENTS.md.

Operations that MUST run in a worker thread:
  - File loading (any provider: COMTRADE, CSV, Excel)
  - Waveform preprocessing (large array operations)
  - Analytics computation (RMS, ROCOF, FFT)
  - Large DataFrame normalization

Operations that MAY run on the UI thread (bounded cost):
  - setData() calls with decimated arrays (≤4000 points)
  - InfiniteLine position updates
  - Show/hide toggle on existing curves
  - Reading cursor position from InfiniteLine.value()

§10.2 Canonical QRunnable worker pattern

  from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject

  class _WorkerSignals(QObject):
      finished = pyqtSignal(object)    # emits result on success
      error    = pyqtSignal(str)       # emits error message on failure

  class _LoadWorker(QRunnable):
      def __init__(self, provider_manager, path: Path):
          super().__init__()
          self.provider_manager = provider_manager
          self.path = path
          self.signals = _WorkerSignals()

      def run(self) -> None:
          try:
              record = self.provider_manager.load(self.path)
              self.signals.finished.emit(record)
          except Exception as exc:
              self.signals.error.emit(str(exc))

  # In the widget or main window:
  def load_file(self, path: Path) -> None:
      worker = _LoadWorker(self._provider_manager, path)
      worker.signals.finished.connect(self._on_record_loaded)
      worker.signals.error.connect(self._on_load_error)
      QThreadPool.globalInstance().start(worker)

  def _on_record_loaded(self, record: DisturbanceRecord) -> None:
      # This slot executes on the UI thread — safe to call set_record()
      self.set_record(record)

§10.3 Rules for cross-thread signal emission

  - Worker signals use Qt.ConnectionType.QueuedConnection automatically when
    sender and receiver live in different threads. This is the default when
    connecting a QRunnable's QObject signals to a UI-thread slot.
  - NEVER call UI widget methods directly from a worker thread.
  - NEVER store a reference to a widget inside a QRunnable and call methods on it.
  - The only cross-thread communication mechanism is Qt signals/slots.

─────────────────────────────────────────────────────────────────────────────
§11 — DISTURBANCERECORD ACCESS PATTERN FOR RENDERING
─────────────────────────────────────────────────────────────────────────────

New DisturbanceRecord (app/models/) uses these fields. All rendering code
MUST use these field names. Legacy src/ field names are prohibited.

  CORRECT (new DisturbanceRecord API)      FORBIDDEN (legacy src/ API)
  ─────────────────────────────────────    ──────────────────────────────────
  record.analog_channels                   record.analogue_channels
  record.waveform_data['time']             record.time_array
  record.waveform_data[ch.name]            ch.raw_data
  ch.index                                 ch.channel_id
  ch.name                                  ch.name (same — OK)
  ch.unit                                  ch.unit (same — OK)
  record.timing_info.trigger_time          record.trigger_time
  record.timing_info.start_time            record.start_time
  record.sampling_info.sampling_rates[0]   record.sample_rate
  record.digital_channels                  record.digital_channels (same — OK)

Time array extraction (cache on record load — do NOT call per-viewport):
  self._time  = record.waveform_data['time'].to_numpy(dtype=np.float64)
  self._data  = {
      ch.index: record.waveform_data[ch.name].to_numpy(dtype=np.float64)
      for ch in record.analog_channels
  }

Trigger time extraction:
  trigger_time_s = (
      record.timing_info.trigger_time - record.timing_info.start_time
  ).total_seconds()

─────────────────────────────────────────────────────────────────────────────
§12 — ZOOM-TO-TRIGGER ALGORITHM
─────────────────────────────────────────────────────────────────────────────

The zoom-to-trigger operation centers the viewport on the trigger event with
a configurable half-window. It is the default view on record load.

  def zoom_to_trigger(self, window_s: float = 0.2) -> None:
      if self._record is None:
          return
      t_trig = (
          self._record.timing_info.trigger_time
          - self._record.timing_info.start_time
      ).total_seconds()
      t_max = self._time[-1] if len(self._time) > 0 else 0.0
      t_start = max(0.0, t_trig - window_s)
      t_end   = min(t_max, t_trig + window_s)
      self.setXRange(t_start, t_end, padding=0)
      # setXRange triggers sigXRangeChanged, which triggers viewport update

─────────────────────────────────────────────────────────────────────────────
§13 — RENDERING ANTI-PATTERNS (FORBIDDEN)
─────────────────────────────────────────────────────────────────────────────

The following patterns are explicitly forbidden in all Powerwave rendering code.

Anti-pattern 1 — Curve recreation on viewport update:
  FORBIDDEN:  self.removeItem(curve); curve = PlotDataItem(...); self.addItem(curve)
  CORRECT:    curve.setData(x_dec, y_dec)
  WHY:        removeItem/addItem destroys the GPU buffer and causes visible flicker.
              setData() updates the buffer in-place via the OpenGL pipeline.

Anti-pattern 2 — Python loop over samples:
  FORBIDDEN:  for i, sample in enumerate(data): processed[i] = transform(sample)
  CORRECT:    processed = np.vectorize_operation(data)
  WHY:        Python loop on 1M+ samples takes seconds. NumPy operates in C,
              taking milliseconds on the same data.

Anti-pattern 3 — Heavy computation on UI thread:
  FORBIDDEN:  def on_open_clicked(self): record = provider.load(path); self.set_record(record)
  CORRECT:    Dispatch to QRunnable; receive DisturbanceRecord via finished signal.
  WHY:        Synchronous file loading blocks the Qt event loop. On a 100MB
              COMTRADE file this can freeze the UI for several seconds.

Anti-pattern 4 — Rendering raw (undecimated) waveform data:
  FORBIDDEN:  curve.setData(time_array, channel_data)  # all 6M+ samples
  CORRECT:    t_dec, d_dec = decimate_for_display(time_array, channel_data, t0, t1)
              curve.setData(t_dec, d_dec)              # ≤4000 samples
  WHY:        Passing 6 million points to PyQtGraph/OpenGL saturates the GPU
              vertex buffer and causes rendering latency of multiple seconds.

Anti-pattern 5 — Calling to_numpy() inside the viewport update hot path:
  FORBIDDEN:  curve.setData(record.waveform_data['time'].to_numpy(), ...)  # per update
  CORRECT:    Cache arrays on record load (§11); reference cache in update loop.
  WHY:        DataFrame column extraction + to_numpy() involves Python overhead
              per call. Inside sigXRangeChanged this fires on every pan/zoom frame.

─────────────────────────────────────────────────────────────────────────────
§14 — OUT-OF-SCOPE FOR VIEWPORT_RENDERING_POLICY
─────────────────────────────────────────────────────────────────────────────

The following topics are related to rendering but are explicitly NOT covered
here — they belong to later-phase directives or separate policy documents:

  - PhasorCanvas (Phase 5+ — QPainter-based, separate widget)
  - Multi-pane dashboard layout (Phase 3B — VisualizationManager scope)
  - Application-wide state singleton (AppState) (Phase 4+ — deferred)
  - Analytics overlays (RMS, ROCOF, harmonic envelopes) (Phase 5)
  - PMU streaming rendering (Phase 6+)
  - Impedance R-X trajectory canvas (Phase 5+)

─────────────────────────────────────────────────────────────────────────────
§16 — N-AXIS VIEWBOX MULTI-PARAMETER ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────

This section defines the mandatory implementation pattern for the N-Axis Single
Canvas (FlexiblePlotCanvas). This is the canonical multi-axis architecture for
all analog parameter rendering in Powerwave.

§16.1 Architecture overview

Each analog parameter displayed in FlexiblePlotCanvas has its own independent
ViewBox. All ViewBoxes share the same X-axis (time domain) but have completely
independent Y-axis ranges, scales, and labels.

One canvas = N parameters = N ViewBoxes = N AxisItems = 1 shared X-axis.

The primary PlotItem (PlotItem_0) hosts the X-axis and the first parameter.
All additional parameters use bare ViewBox objects linked to PlotItem_0's ViewBox.

§16.2 Primary PlotItem setup

  primary_plot = canvas.addPlot(row=0, col=0)
  primary_plot.showGrid(x=True, y=True, alpha=0.2)
  primary_plot.setLabel('bottom', 'Time', units='s')
  # First parameter's curve goes into primary_plot directly

§16.3 Additional parameter ViewBox pattern

For each parameter beyond the first:

  # 1. Create a bare ViewBox
  vb = pg.ViewBox()

  # 2. Link X-axis to primary — critical for synchronized horizontal navigation
  vb.setXLink(primary_plot)

  # 3. Add ViewBox to the scene (it is NOT in the GraphicsLayout grid)
  primary_plot.scene().addItem(vb)

  # 4. Create a curve in the secondary ViewBox
  curve = pg.PlotDataItem(pen=pg.mkPen(color, width=1), skipFiniteCheck=True)
  vb.addItem(curve)

  # 5. Create a color-coded AxisItem (alternating left/right for readability)
  #    Axis index 0 = primary left axis (already exists on primary_plot)
  #    Axis index 1 = first right axis
  #    Axis index 2 = second left axis (if using both sides)
  axis = pg.AxisItem(orientation='right')
  axis.setLabel(name, units=unit)
  axis.setPen(pg.mkPen(color))
  axis.setTextPen(pg.mkPen(color))

  # 6. Link AxisItem to secondary ViewBox
  axis.linkToView(vb)

  # 7. Add axis to the canvas layout (controls margin placement)
  canvas.addItem(axis, row=0, col=len(existing_params) + 1)

§16.4 Geometry synchronization — MANDATORY

Secondary ViewBoxes have no automatic size management. Their geometry MUST be
updated manually whenever the primary PlotItem's ViewBox is resized.

REQUIRED CONNECTION (set up once on canvas creation):

  primary_plot.getViewBox().sigResized.connect(self._sync_viewbox_geometry)

  def _sync_viewbox_geometry(self) -> None:
      scene_rect = self._primary_plot.getViewBox().sceneBoundingRect()
      for vb in self._secondary_viewboxes:
          vb.setGeometry(scene_rect)
          vb.linkedViewChanged(self._primary_plot.getViewBox(), vb.XAxis)

FAILURE TO DO THIS causes secondary ViewBoxes to remain at zero geometry,
making curves invisible or incorrectly positioned.

§16.5 Axis positioning strategy

For N parameters, axes alternate between right and left margins to avoid
over-crowding a single side:

  Parameter 0: primary left axis (managed by primary PlotItem)
  Parameter 1: right axis, column +1
  Parameter 2: left axis, column -1 (insert before column 0)
  Parameter 3: right axis, column +2
  ...

For Phase 3A, simpler right-only axis stacking is acceptable. Left/right
alternation may be added in Phase 3B.

§16.6 Independent Y-axis scaling

Each ViewBox maintains independent Y-range. Setting Y-range in one ViewBox
has no effect on other ViewBoxes. This is automatic — no additional work needed.

Users may reset the Y-range to auto with:
  vb.enableAutoRange(axis=pg.ViewBox.YAxis)

Manual Y-range:
  vb.setYRange(y_min, y_max, padding=0)

§16.6.1 Merged panel axis guardrails

Panel merge is a visual layout operation. It SHALL combine waveform panels into
one canvas with a shared X-axis, but it SHALL NOT collapse unrelated signals
onto one Y-axis.

Correct merge behavior:
  - Validate that all selected panels have a compatible X-axis.
  - Preserve every channel's unit, signal type, and display metadata.
  - Rebuild the destination FlexiblePlotCanvas using the normal multi-axis
    manager.
  - Group axes by engineering meaning, at minimum by signal type + unit.
  - Keep different units on independent Y-axes.
  - Keep different signal types on independent Y-axes even when the unit string
    happens to match, unless an explicit user override exists.

Incorrect merge behavior:
  - Concatenating channel values onto one shared Y-axis.
  - Replacing channel units with a mixed or arbitrary unit to make the merge
    fit one axis.
  - Treating panel title as the axis grouping key.
  - Merging panels with incompatible time/sample-index vectors without warning.

Examples:
  Power(MW) + Frequency(Hz) => one canvas, MW axis + Hz axis
  Frequency(Hz) + ROCOF(Hz/s) => one canvas, Hz axis + Hz/s axis
  Voltage(kV) + Current(A) => one canvas, kV axis + A axis

UX guardrail:
  If a merge would create many Y-axes, warn the user that the result may be hard
  to read and offer cancel/continue.

§16.7 Cursor and trigger line in N-Axis canvas

InfiniteLine items added to primary_plot are automatically visible across the
full canvas height because the primary PlotItem's scene covers the full widget.

However, InfiniteLines added to secondary ViewBoxes are NOT visible in the
primary PlotItem's visual region. The cursor MUST be added to primary_plot only:

  self._primary_plot.addItem(self._cursor)
  self._primary_plot.addItem(self._trigger_line)

The cursor and trigger line will visually span the full canvas height due to
the InfiniteLine's nature (infinite vertical extent) regardless of which
PlotItem they are added to, as long as they are in the primary scene.

§16.8 setXLink() behavior

setXLink() synchronizes X-range (time) between the secondary ViewBox and the
primary PlotItem's ViewBox. This means:
  - Zooming in the primary plot → all secondary ViewBoxes follow
  - Panning in a secondary ViewBox → primary plot follows (bidirectional)
  - Y-range is completely independent (exactly as required)

setXLink() operates at the PyQtGraph C++ layer — zero Python overhead per
pan/zoom frame. This is the correct mechanism for shared time-axis synchronization.

§16.9 Performance note for N-Axis canvas

With N parameters, the viewport update hot path calls setData() N times per
sigXRangeChanged event. Each call is O(k) where k ≤ max_display_points (4000).
For 10 parameters: 10 × 4000 = 40k points per viewport update. This is fast
at the OpenGL level — no special optimization needed beyond standard decimation.

─────────────────────────────────────────────────────────────────────────────
§17 — DIGITAL EVENT TIMELINE (SEPARATE COMPONENT)
─────────────────────────────────────────────────────────────────────────────

Digital signals (breaker status, relay trips, pickups, alarms) are NOT rendered
in FlexiblePlotCanvas. They require a separate dedicated component.

§17.1 Architecture separation rationale

Digital signals are binary state changes (0/1), not continuous waveforms. They:
  - Do not have a meaningful Y-axis range or scale
  - Are most informative as step function / horizontal state tracks
  - Benefit from different visual treatment (color fill, labels, state annotation)
  - Typically occupy much less vertical space than analog waveforms

Mixing digital signals into the N-Axis ViewBox architecture would require
dummy Y-axes with [0, 1] range, wasting vertical canvas space and complicating
the axis management logic.

§17.2 Canonical separation

  FlexiblePlotCanvas:    analog parameters only (N ViewBoxes, N Y-axes)
  DigitalEventTimeline:  digital channels only (separate widget below canvas)

Both widgets share the SAME X-axis time reference. Their time axes must be
synchronized (X-linked or driven by the same viewport state).

§17.3 DigitalEventTimeline rendering model

Each digital channel occupies one horizontal track (fixed-height row).
Within each track:
  - Low state (0): dim fill or empty
  - High state (1): bright fill or distinct color

Rendering approach: PyQtGraph step function (connect='finite') or QGraphicsItem
custom rendering. Use channel-specific colors from the dark engineering palette
(§9). No Y-axis. No cursor readout (cursor position is shown on analog canvas).

§17.4 Phase scope

DigitalEventTimeline is NOT in Phase 3A scope. It is scheduled for Phase 3B.

Phase 3A implements FlexiblePlotCanvas (analog only).
Phase 3B implements DigitalEventTimeline + VisualizationManager (wires them together).

─────────────────────────────────────────────────────────────────────────────
§15 — DOCUMENT AUTHORITY AND REFERENCES
─────────────────────────────────────────────────────────────────────────────

This document is authoritative for rendering implementation details.

Related policy documents:
  docs/VISUALIZATION_CONTRACT.md     — WHAT the visualization engine must do
  docs/PERFORMANCE_REQUIREMENTS.md   — Rendering performance targets
  docs/CHANNEL_MAPPING_POLICY.md     — Signal role taxonomy for color assignment
  docs/DATA_CONTRACT.md              — DisturbanceRecord full field reference

For architecture boundaries (what rendering must NOT do):
  docs/ARCHITECTURE.md               — Subsystem separation rules
  docs/LEGACY_CODEBASE_POLICY.md     — src/ isolation rules

Implementation directive:
  directives/implement_flexible_plot_canvas.md — Phase 3A build specification
