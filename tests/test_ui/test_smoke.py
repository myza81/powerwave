"""
tests/test_ui/test_smoke.py

Milestone 1D smoke tests — import and attribute checks only.

These tests do NOT instantiate any Qt widget or require a display.  They
verify that every module can be imported without side effects and that the
key classes / attributes exist with the expected interface.

The real Milestone 1D test is manual:
  .venv/bin/python src/main.py
  File > Open → load a COMTRADE .cfg file → waveforms must appear.
"""

from __future__ import annotations

import importlib
import types


# ── Import smoke tests ────────────────────────────────────────────────────────

class TestImports:
    """All Milestone 1D modules must be importable without raising."""

    def test_app_state_importable(self) -> None:
        mod = importlib.import_module("src.core.app_state")
        assert isinstance(mod, types.ModuleType)

    def test_thread_manager_importable(self) -> None:
        mod = importlib.import_module("src.core.thread_manager")
        assert isinstance(mod, types.ModuleType)

    def test_waveform_panel_importable(self) -> None:
        mod = importlib.import_module("src.ui.waveform_panel")
        assert isinstance(mod, types.ModuleType)

    def test_channel_canvas_importable(self) -> None:
        mod = importlib.import_module("src.ui.channel_canvas")
        assert isinstance(mod, types.ModuleType)

    def test_main_importable(self) -> None:
        mod = importlib.import_module("src.main")
        assert isinstance(mod, types.ModuleType)


# ── AppState signal attributes ────────────────────────────────────────────────

class TestAppState:
    """app_state singleton must expose all required signals."""

    def test_app_state_singleton_exists(self) -> None:
        from core.app_state import app_state
        assert app_state is not None

    def test_record_loaded_signal_exists(self) -> None:
        from core.app_state import app_state
        assert hasattr(app_state, "record_loaded")

    def test_cursor_moved_signal_exists(self) -> None:
        from core.app_state import app_state
        assert hasattr(app_state, "cursor_moved")

    def test_channel_toggled_signal_exists(self) -> None:
        from core.app_state import app_state
        assert hasattr(app_state, "channel_toggled")

    def test_view_mode_changed_signal_exists(self) -> None:
        from core.app_state import app_state
        assert hasattr(app_state, "view_mode_changed")


# ── WorkerSignals attributes ──────────────────────────────────────────────────

class TestWorkerSignals:
    """WorkerSignals must have finished, error, and progress."""

    def test_finished_attribute(self) -> None:
        from core.thread_manager import WorkerSignals
        assert hasattr(WorkerSignals, "finished")

    def test_error_attribute(self) -> None:
        from core.thread_manager import WorkerSignals
        assert hasattr(WorkerSignals, "error")

    def test_progress_attribute(self) -> None:
        from core.thread_manager import WorkerSignals
        assert hasattr(WorkerSignals, "progress")


# ── Worker class interface ────────────────────────────────────────────────────

class TestWorker:
    """Worker must accept a callable and expose a signals attribute."""

    def test_worker_has_run_method(self) -> None:
        from core.thread_manager import Worker
        assert callable(getattr(Worker, "run", None))

    def test_worker_constructor_accepts_callable(self) -> None:
        from core.thread_manager import Worker
        # Constructing Worker should not raise (no QApplication needed for
        # QRunnable — it is not a QObject)
        w = Worker(lambda: 42)
        assert w.fn() == 42

    def test_worker_signals_attribute(self) -> None:
        from core.thread_manager import Worker
        w = Worker(lambda: None)
        assert hasattr(w, "signals")
        from core.thread_manager import WorkerSignals
        assert isinstance(w.signals, WorkerSignals)


# ── run_in_thread callable ────────────────────────────────────────────────────

class TestRunInThread:
    """run_in_thread must be a callable exported from thread_manager."""

    def test_run_in_thread_is_callable(self) -> None:
        from core.thread_manager import run_in_thread
        assert callable(run_in_thread)


# ── ChannelCanvas class interface ─────────────────────────────────────────────

class TestChannelCanvasInterface:
    """ChannelCanvas class must expose the required public methods."""

    def test_load_record_method_exists(self) -> None:
        from ui.channel_canvas import ChannelCanvas
        assert callable(getattr(ChannelCanvas, "load_record", None))

    def test_update_channel_visibility_method_exists(self) -> None:
        from ui.channel_canvas import ChannelCanvas
        assert callable(
            getattr(ChannelCanvas, "update_channel_visibility", None)
        )


# ── WaveformPanel class interface ─────────────────────────────────────────────

class TestWaveformPanelInterface:
    """LabelPanel (renamed from WaveformPanel) must expose the load_record method."""

    def test_load_record_method_exists(self) -> None:
        from ui.waveform_panel import LabelPanel
        assert callable(getattr(LabelPanel, "load_record", None))


# ── MainWindow class interface ────────────────────────────────────────────────

class TestMainWindowInterface:
    """MainWindow and the main() entry point must be present."""

    def test_main_window_class_exists(self) -> None:
        from main import MainWindow
        assert MainWindow is not None

    def test_main_function_exists(self) -> None:
        from main import main
        assert callable(main)
