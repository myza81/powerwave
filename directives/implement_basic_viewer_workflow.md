DIRECTIVE: implement_basic_viewer_workflow.md
Phase 4A — First Operational Viewer

OBJECTIVE

Wire VisualizationManager into PowerwaveMainWindow to produce the first end-to-end
working Powerwave viewer: File → Open → load → display.

SCOPE

Implement:
  app/ui/main_window/main_window.py   — PowerwaveMainWindow + threading helpers + module-level utilities
  app/ui/main_window/__init__.py      — export PowerwaveMainWindow
  app/main.py                         — pg.setConfigOptions() + import from new location
  tests/unit/test_main_window_workflow.py — ~18 non-GUI unit tests

DO NOT implement:
  - Analytics panels, docking, preferences/settings dialogs
  - Measurement panel, multi-record, plugin loader
  - Synchronization manager / multi-panel cursor coordination

DO NOT modify:
  - DisturbanceRecord, providers, visualization architecture
  - FlexiblePlotCanvas, DigitalEventTimeline, VisualizationManager
  - downsampling.py, digital_transforms.py, multi_axis_manager.py

ARCHITECTURE

  app/main.py
    pg.setConfigOptions(useOpenGL=True, antialias=False, foreground='w', background='#1E1E1E')
    QApplication + PowerwaveMainWindow (imported from app.ui.main_window)

  app/ui/main_window/main_window.py — four module-level + two classes

    _FILE_FILTER (str constant)
      Supported Files (*.cfg *.comtrade *.csv *.xlsx);;COMTRADE;;CSV;;Excel;;All Files (*)

    _build_provider_manager() -> ProviderManager
      Module-level function; testable without Qt.
      Registers ComtradeProvider, CsvProvider, ExcelProvider in that order.

    _format_load_status(record: DisturbanceRecord) -> str
      Module-level function; testable without Qt.
      Returns: "<filename> | <N> analog, <M> digital | Fs = <rate> Hz"
      Uses Path(record.metadata.source_file).name for the filename component.
      If sampling_rates is empty → "unknown" for rate.

    _WorkerSignals(QObject)
      finished = pyqtSignal(object)   ← emits DisturbanceRecord
      error    = pyqtSignal(str)      ← emits error message string

    _LoadWorker(QRunnable)
      __init__(provider_manager, path: Path)
      run(): calls provider_manager.load(path); emits finished or error

    PowerwaveMainWindow(QMainWindow)
      __init__():
        self._provider_manager = _build_provider_manager()
        self._canvas = FlexiblePlotCanvas()
        self._timeline = DigitalEventTimeline()
        self._vis_manager = VisualizationManager(self._canvas, self._timeline)
        self._x_axis_linked = False
        _build_layout()
        _build_menu()

      showEvent(event):
        if not self._x_axis_linked:
            self._vis_manager.link_x_axis()
            self._x_axis_linked = True

      _build_layout():
        QSplitter(Vertical): canvas (stretch=3) / timeline (stretch=1)
        setCentralWidget(splitter)
        setStatusBar(QStatusBar()); statusBar().showMessage("Ready")

      _build_menu():
        File menu: Open (Ctrl+O) → _open_file_dialog(); separator; Exit → close()

      _open_file_dialog():
        QFileDialog.getOpenFileName with _FILE_FILTER
        if path selected: _load_file(Path(path))

      _load_file(path: Path):
        statusBar().showMessage(f"Loading: {path.name} …")
        _LoadWorker; connect finished → _on_record_loaded; connect error → _on_load_error
        QThreadPool.globalInstance().start(worker)

      _on_record_loaded(record: DisturbanceRecord):
        self._vis_manager.set_record(record)
        statusBar().showMessage(_format_load_status(record))
        setWindowTitle(f"Powerwave — {station_or_filename}")

      _on_load_error(message: str):
        statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Load Error", message)

THREADING CONTRACT

  File loading MUST NOT block the UI thread.
  Use QRunnable + QThreadPool (VIEWPORT_RENDERING_POLICY §10).
  _WorkerSignals(QObject) cross-thread signal delivery is required.
  Worker holds a strong reference to provider_manager and path.

X-AXIS LINK TIMING

  link_x_axis() MUST be called after both widgets are in a visible Qt scene.
  Call in showEvent() with _x_axis_linked guard (call once only).
  Do NOT call in __init__ before show().

VISUAIZATION MANAGER LIFETIME

  self._vis_manager MUST be an instance attribute of PowerwaveMainWindow.
  pyqtSignal stores weak reference to bound methods on non-QObject receivers.
  Allowing the manager to be GC'd silently drops cursor_moved connections.

PYQTGRAPH CONFIGURATION

  pg.setConfigOptions() MUST be called in app/main.py before any pg widget
  is instantiated (VIEWPORT_RENDERING_POLICY §1).
  Options: useOpenGL=True, antialias=False, foreground='w', background='#1E1E1E'

TESTS

  tests/unit/test_main_window_workflow.py
  No Qt display, no QApplication.
  Test only: _build_provider_manager(), _format_load_status()

  TestBuildProviderManager (10 tests):
    - Returns ProviderManager instance
    - available_providers() returns list of length 3
    - "comtrade" in available_providers()
    - "csv" in available_providers()
    - "excel" in available_providers()
    - ComtradeProvider registered: can_load(".cfg") is True
    - CsvProvider registered: can_load(".csv") is True
    - ExcelProvider registered: can_load(".xlsx") is True
    - Second call returns independent instance
    - Discovery order: comtrade first (index 0), csv second (index 1)

  TestFormatLoadStatus (8 tests):
    - Returns str
    - Contains source_file basename (not full path)
    - Contains analog count formatted as "<N> analog"
    - Contains digital count formatted as "<M> digital"
    - Contains sampling rate formatted as "<rate> Hz"
    - "unknown" rate when sampling_rates is empty
    - Zero analog channels → "0 analog"
    - Zero digital channels → "0 digital"

VALIDATION

  All existing 406 tests continue to pass.
  18 new non-GUI tests pass.
  Total: 424 tests passing.

COMPLETION CRITERIA

  [x] directives/implement_basic_viewer_workflow.md created (this file)
  [ ] app/ui/main_window/main_window.py implemented
  [ ] app/ui/main_window/__init__.py exports PowerwaveMainWindow
  [ ] app/main.py updated (pg.setConfigOptions + new import)
  [ ] tests/unit/test_main_window_workflow.py — 18 tests passing
  [ ] All 424 tests passing
  [ ] agent/HANDOFF.md updated (Session 015 entry)
  [ ] agent/TASK.md updated (Phase 4A COMPLETED)
  [ ] agent/REPOSITORY_STATE.md updated (424 tests, new files listed)
