"""Cross-panel X-range and cursor synchronization for waveform views.

The manager owns interaction wiring only. It does not own waveform data, know
about providers, or perform rendering/analytics work.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication


class _DirectSignalConnection:
    """Fallback signal connection used when no QCoreApplication exists."""

    def __init__(self, signal: Any, slot: Any) -> None:
        self._signal = signal
        self._slot = lambda *args: slot(args)
        signal.connect(self._slot)

    def disconnect(self) -> None:
        try:
            self._signal.disconnect(self._slot)
        except (TypeError, RuntimeError, AttributeError):
            pass


@dataclasses.dataclass
class _RegisteredCanvas:
    canvas: Any
    plot_item: pg.PlotItem
    viewbox: pg.ViewBox
    cursor: pg.InfiniteLine | None
    x_proxy: pg.SignalProxy | _DirectSignalConnection | None
    cursor_proxy: pg.SignalProxy | _DirectSignalConnection | None


class SynchronizationManager:
    """Synchronize time navigation and cursor position across plot canvases.

    Registered canvases may be FlexiblePlotCanvas, DigitalEventTimeline, a
    PlotWidget, or a PlotItem-like object exposing a ViewBox. The manager uses
    PyQtGraph signals only; there is no polling, timer-driven repaint, or data
    ownership in this layer.
    """

    def __init__(self, *, rate_limit_hz: int = 120) -> None:
        self._registered: dict[int, _RegisteredCanvas] = {}
        self._master_canvas_id: int | None = None
        self._active_source_canvas_id: int | None = None
        self._sync_depth = 0
        self._rate_limit_hz = rate_limit_hz
        self._visible_x_range: tuple[float, float] | None = None
        self._cursor_pos: float | None = None

    @property
    def registered_count(self) -> int:
        return len(self._registered)

    @property
    def active_source_canvas(self) -> Any | None:
        entry = self._registered.get(self._active_source_canvas_id or -1)
        return entry.canvas if entry is not None else None

    @property
    def visible_x_range(self) -> tuple[float, float] | None:
        return self._visible_x_range

    @property
    def cursor_pos(self) -> float | None:
        return self._cursor_pos

    def register_canvas(self, canvas: Any, *, set_as_master: bool = False) -> None:
        """Register a plot canvas for X-range and cursor synchronization."""
        canvas_id = id(canvas)
        if canvas_id in self._registered:
            if set_as_master:
                self.set_master_canvas(canvas)
            return

        plot_item = self._extract_plot_item(canvas)
        viewbox = plot_item.getViewBox()
        cursor = self._extract_cursor(canvas)

        x_proxy = self._connect_signal(
            viewbox.sigXRangeChanged,
            lambda args, cid=canvas_id: self._on_x_range_changed(cid, args),
        )
        cursor_proxy = None
        if cursor is not None:
            cursor_proxy = self._connect_signal(
                cursor.sigPositionChanged,
                lambda args, cid=canvas_id: self._on_cursor_changed(cid, args),
            )

        self._registered[canvas_id] = _RegisteredCanvas(
            canvas=canvas,
            plot_item=plot_item,
            viewbox=viewbox,
            cursor=cursor,
            x_proxy=x_proxy,
            cursor_proxy=cursor_proxy,
        )

        if self._master_canvas_id is None or set_as_master:
            self.set_master_canvas(canvas)

    def unregister_canvas(self, canvas: Any) -> None:
        """Remove a canvas from synchronization and disconnect signal proxies."""
        canvas_id = id(canvas)
        entry = self._registered.pop(canvas_id, None)
        if entry is None:
            return
        self._disconnect_entry(entry)

        if self._master_canvas_id == canvas_id:
            self._master_canvas_id = next(iter(self._registered), None)
        if self._active_source_canvas_id == canvas_id:
            self._active_source_canvas_id = None

    def clear(self) -> None:
        """Disconnect all registered canvases."""
        for entry in list(self._registered.values()):
            self._disconnect_entry(entry)
        self._registered.clear()
        self._master_canvas_id = None
        self._active_source_canvas_id = None
        self._visible_x_range = None
        self._cursor_pos = None
        self._sync_depth = 0

    def set_master_canvas(self, canvas: Any) -> None:
        """Mark the canvas whose current range/cursor should seed sync state."""
        canvas_id = id(canvas)
        if canvas_id not in self._registered:
            self.register_canvas(canvas)
            return
        self._master_canvas_id = canvas_id
        self._active_source_canvas_id = canvas_id
        self._visible_x_range = self._read_x_range(self._registered[canvas_id])
        cursor = self._registered[canvas_id].cursor
        if cursor is not None:
            self._cursor_pos = float(cursor.value())

    def register_many(
        self,
        canvases: Iterable[Any],
        *,
        master_canvas: Any | None = None,
    ) -> None:
        for canvas in canvases:
            self.register_canvas(canvas)
        if master_canvas is not None:
            self.set_master_canvas(master_canvas)

    def synchronize_x_range(
        self,
        source_canvas: Any,
        x_range: tuple[float, float] | list[float],
    ) -> None:
        """Propagate a visible X-range from one canvas to all others."""
        if self._sync_depth > 0:
            return
        source_id = id(source_canvas)
        if source_id not in self._registered:
            return

        t_start, t_end = self._normalize_range(x_range)
        if (
            self._visible_x_range is not None
            and self._ranges_close(self._visible_x_range, (t_start, t_end))
            and source_id != self._active_source_canvas_id
        ):
            return
        self._visible_x_range = (t_start, t_end)
        self._active_source_canvas_id = source_id

        self._sync_depth += 1
        try:
            for canvas_id, entry in self._registered.items():
                if canvas_id == source_id:
                    continue
                self._set_x_range(entry, t_start, t_end)
        finally:
            self._sync_depth -= 1

    def synchronize_cursor(self, source_canvas: Any, t: float) -> None:
        """Propagate cursor position from one canvas to all others."""
        if self._sync_depth > 0:
            return
        source_id = id(source_canvas)
        if source_id not in self._registered:
            return

        pos = float(t)
        if (
            self._cursor_pos is not None
            and abs(self._cursor_pos - pos) <= 1e-12
            and source_id != self._active_source_canvas_id
        ):
            return
        self._cursor_pos = pos
        self._active_source_canvas_id = source_id

        self._sync_depth += 1
        try:
            for canvas_id, entry in self._registered.items():
                if canvas_id == source_id:
                    continue
                self._set_cursor(entry, pos)
        finally:
            self._sync_depth -= 1

    def _on_x_range_changed(self, canvas_id: int, args: tuple[Any, ...]) -> None:
        if self._sync_depth > 0:
            return
        entry = self._registered.get(canvas_id)
        if entry is None:
            return
        payload = self._unwrap_signal_proxy_args(args)
        if len(payload) < 2:
            return
        self.synchronize_x_range(entry.canvas, payload[1])

    def _on_cursor_changed(self, canvas_id: int, args: tuple[Any, ...]) -> None:
        if self._sync_depth > 0:
            return
        entry = self._registered.get(canvas_id)
        if entry is None:
            return
        payload = self._unwrap_signal_proxy_args(args)
        if not payload:
            return
        line = payload[0]
        if hasattr(line, "value"):
            self.synchronize_cursor(entry.canvas, float(line.value()))

    def _set_x_range(
        self,
        entry: _RegisteredCanvas,
        t_start: float,
        t_end: float,
    ) -> None:
        # FlexiblePlotCanvas provides normalize_viewport() to keep sparse
        # trends and secondary axes stable after an X-range change.
        if hasattr(entry.canvas, "normalize_viewport"):
            entry.canvas.normalize_viewport(t_start, t_end)
        else:
            entry.plot_item.setXRange(t_start, t_end, padding=0)

    def _set_cursor(self, entry: _RegisteredCanvas, t: float) -> None:
        if hasattr(entry.canvas, "set_cursor_pos"):
            entry.canvas.set_cursor_pos(t)
            return
        if entry.cursor is not None:
            entry.cursor.blockSignals(True)
            entry.cursor.setValue(t)
            entry.cursor.blockSignals(False)

    def _disconnect_entry(self, entry: _RegisteredCanvas) -> None:
        if entry.x_proxy is not None and self._qt_object_alive(entry.viewbox):
            entry.x_proxy.disconnect()
        if (
            entry.cursor_proxy is not None
            and entry.cursor is not None
            and self._qt_object_alive(entry.cursor)
        ):
            entry.cursor_proxy.disconnect()

    def _connect_signal(self, signal: Any, slot: Any) -> pg.SignalProxy | _DirectSignalConnection:
        if QCoreApplication.instance() is None:
            return _DirectSignalConnection(signal, slot)
        return pg.SignalProxy(
            signal,
            delay=0,
            rateLimit=self._rate_limit_hz,
            slot=slot,
        )

    @staticmethod
    def _qt_object_alive(obj: Any) -> bool:
        try:
            return not sip.isdeleted(obj)
        except (TypeError, RuntimeError):
            return True

    @staticmethod
    def _extract_plot_item(canvas: Any) -> pg.PlotItem:
        if hasattr(canvas, "_primary_plot"):
            return canvas._primary_plot
        if hasattr(canvas, "getPlotItem"):
            return canvas.getPlotItem()
        if hasattr(canvas, "getViewBox"):
            return canvas
        raise TypeError("Canvas must expose _primary_plot, getPlotItem(), or getViewBox()")

    @staticmethod
    def _extract_cursor(canvas: Any) -> pg.InfiniteLine | None:
        cursor = getattr(canvas, "_cursor", None)
        if cursor is not None and hasattr(cursor, "sigPositionChanged"):
            return cursor
        return None

    @staticmethod
    def _read_x_range(entry: _RegisteredCanvas) -> tuple[float, float]:
        x_range = entry.viewbox.viewRange()[0]
        return float(x_range[0]), float(x_range[1])

    @staticmethod
    def _normalize_range(x_range: tuple[float, float] | list[float]) -> tuple[float, float]:
        if len(x_range) != 2:
            raise ValueError("x_range must contain exactly two values")
        return float(x_range[0]), float(x_range[1])

    @staticmethod
    def _ranges_close(
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> bool:
        return abs(left[0] - right[0]) <= 1e-12 and abs(left[1] - right[1]) <= 1e-12

    @staticmethod
    def _unwrap_signal_proxy_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(args) == 1 and isinstance(args[0], tuple):
            return args[0]
        return args
