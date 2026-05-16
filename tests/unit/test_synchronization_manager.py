from __future__ import annotations

from app.visualization.managers.synchronization_manager import SynchronizationManager


class _Signal:
    def __init__(self) -> None:
        self._slots = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self._slots):
            slot(args)


class _Proxy:
    def __init__(self, signal, delay=0.3, rateLimit=0, slot=None, **_kw) -> None:
        self.signal = signal
        self.slot = slot
        self.disconnected = False
        if slot is not None:
            signal.connect(slot)

    def disconnect(self) -> None:
        self.disconnected = True


class _ViewBox:
    def __init__(self) -> None:
        self.sigXRangeChanged = _Signal()
        self.range = [0.0, 1.0]

    def viewRange(self):
        return [list(self.range), [0.0, 1.0]]

    def setXRange(self, t_start: float, t_end: float, padding=0) -> None:
        self.range = [t_start, t_end]
        self.sigXRangeChanged.emit(self, tuple(self.range))


class _PlotItem:
    def __init__(self) -> None:
        self.viewbox = _ViewBox()

    def getViewBox(self):
        return self.viewbox

    def setXRange(self, t_start: float, t_end: float, padding=0) -> None:
        self.viewbox.setXRange(t_start, t_end, padding=padding)


class _Cursor:
    def __init__(self) -> None:
        self.sigPositionChanged = _Signal()
        self._value = 0.0
        self._blocked = False

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        self._value = value
        if not self._blocked:
            self.sigPositionChanged.emit(self)

    def blockSignals(self, blocked: bool) -> None:
        self._blocked = blocked


class _AnalogCanvas:
    def __init__(self) -> None:
        self._primary_plot = _PlotItem()
        self._cursor = _Cursor()
        self.normalized_ranges: list[tuple[float, float]] = []
        self.cursor_positions: list[float] = []

    def normalize_viewport(self, t_start: float, t_end: float) -> None:
        self.normalized_ranges.append((t_start, t_end))
        self._primary_plot.setXRange(t_start, t_end, padding=0)

    def set_cursor_pos(self, t: float) -> None:
        self.cursor_positions.append(t)
        self._cursor.blockSignals(True)
        self._cursor.setValue(t)
        self._cursor.blockSignals(False)


class _DigitalTimeline:
    def __init__(self) -> None:
        self._plot_item = _PlotItem()
        self._cursor = _Cursor()
        self.cursor_positions: list[float] = []

    def getPlotItem(self):
        return self._plot_item

    def set_cursor_pos(self, t: float) -> None:
        self.cursor_positions.append(t)
        self._cursor.blockSignals(True)
        self._cursor.setValue(t)
        self._cursor.blockSignals(False)


def test_synchronized_zoom_updates_all_registered_canvases(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    master = _AnalogCanvas()
    follower = _AnalogCanvas()

    manager.register_many([master, follower], master_canvas=master)
    manager.synchronize_x_range(master, (2.0, 5.0))

    assert follower.normalized_ranges == [(2.0, 5.0)]
    assert manager.visible_x_range == (2.0, 5.0)


def test_synchronized_pan_uses_active_source_panel(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    a = _AnalogCanvas()
    b = _AnalogCanvas()
    manager.register_many([a, b], master_canvas=a)

    manager.synchronize_x_range(b, (10.0, 12.0))

    assert a.normalized_ranges == [(10.0, 12.0)]
    assert manager.active_source_canvas is b


def test_cursor_propagation_updates_all_other_cursors(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    a = _AnalogCanvas()
    b = _AnalogCanvas()
    manager.register_many([a, b], master_canvas=a)

    manager.synchronize_cursor(a, 3.25)

    assert b.cursor_positions == [3.25]
    assert b._cursor.value() == 3.25
    assert manager.cursor_pos == 3.25


def test_unregister_stops_range_and_cursor_propagation(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    a = _AnalogCanvas()
    b = _AnalogCanvas()
    manager.register_many([a, b], master_canvas=a)
    manager.unregister_canvas(b)

    manager.synchronize_x_range(a, (4.0, 6.0))
    manager.synchronize_cursor(a, 9.0)

    assert b.normalized_ranges == []
    assert b.cursor_positions == []
    assert manager.registered_count == 1


def test_recursive_x_range_signal_is_ignored_during_propagation(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    a = _AnalogCanvas()
    b = _AnalogCanvas()
    c = _AnalogCanvas()
    manager.register_many([a, b, c], master_canvas=a)

    manager.synchronize_x_range(a, (7.0, 8.0))

    assert b.normalized_ranges == [(7.0, 8.0)]
    assert c.normalized_ranges == [(7.0, 8.0)]


def test_digital_timeline_synchronizes_range_and_cursor(monkeypatch) -> None:
    monkeypatch.setattr("pyqtgraph.SignalProxy", _Proxy)
    manager = SynchronizationManager()
    analog = _AnalogCanvas()
    digital = _DigitalTimeline()
    manager.register_canvas(analog, set_as_master=True)
    manager.register_canvas(digital)

    manager.synchronize_x_range(analog, (1.0, 1.5))
    manager.synchronize_cursor(analog, 1.25)

    assert digital.getPlotItem().getViewBox().viewRange()[0] == [1.0, 1.5]
    assert digital.cursor_positions == [1.25]
