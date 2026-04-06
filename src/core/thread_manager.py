"""
src/core/thread_manager.py

QThreadPool worker infrastructure (LAW 2 — UI thread is sacred).

All file I/O, parsing, and heavy computation must be dispatched off the UI
thread.  This module provides:
  WorkerSignals  — QObject carrying finished / error / progress signals
  Worker         — QRunnable that wraps any callable
  run_in_thread  — convenience one-liner for the most common use case

Usage::

    from src.core.thread_manager import run_in_thread

    run_in_thread(
        parser.load, path,
        on_done=self._on_record_loaded,
        on_error=self._on_load_error,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal


# ── WorkerSignals ─────────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    """Signals emitted by a Worker upon completion or failure.

    QRunnable itself cannot carry signals (it is not a QObject), so signals
    live in a separate QObject instance owned by each Worker.
    """

    finished = pyqtSignal(object)   # emits the callable's return value
    error    = pyqtSignal(str)      # emits a human-readable error message
    progress = pyqtSignal(int)      # emits 0-100 percent complete


# ── Worker ────────────────────────────────────────────────────────────────────

class Worker(QRunnable):
    """Wraps an arbitrary callable for execution on the global QThreadPool.

    Args:
        fn:      Callable to execute off the UI thread.
        *args:   Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Signals (via ``self.signals``):
        finished(object): emitted with ``fn``'s return value on success.
        error(str):       emitted with a formatted error message on exception.
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Execute the wrapped callable; emit finished or error."""
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — intentional catch-all
            self.signals.error.emit(
                f"{type(exc).__name__}: {exc}"
            )


# ── run_in_thread ─────────────────────────────────────────────────────────────

def run_in_thread(
    fn: Callable,
    *args: Any,
    on_done: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
    **kwargs: Any,
) -> Worker:
    """Dispatch ``fn(*args, **kwargs)`` to the global QThreadPool.

    Args:
        fn:       Callable to execute off the UI thread.
        *args:    Positional arguments for ``fn``.
        on_done:  Optional slot connected to ``Worker.signals.finished``.
                  Called on the UI thread (Qt auto-connection) with the
                  callable's return value.
        on_error: Optional slot connected to ``Worker.signals.error``.
                  Called on the UI thread with a human-readable error string.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The ``Worker`` instance (already submitted to the thread pool).
    """
    worker = Worker(fn, *args, **kwargs)
    if on_done is not None:
        worker.signals.finished.connect(on_done)
    if on_error is not None:
        worker.signals.error.connect(on_error)
    QThreadPool.globalInstance().start(worker)
    return worker
