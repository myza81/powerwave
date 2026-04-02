# SKILL: PyQt6 UI & PyQtGraph Rendering

## Trigger
Load this skill when implementing anything in `src/ui/` or when working with
PyQt6 widgets, signals/slots, docking panels, or PyQtGraph canvases.

---

## PYQT6 PATTERNS

### Main Window Shell
```python
from PyQt6.QtWidgets import (QMainWindow, QDockWidget, QToolBar,
                              QFileDialog, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool
from PyQt6.QtGui import QAction
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PowerWave Analyst")
        self.setMinimumSize(1280, 800)
        self._setup_central_widget()
        self._setup_docks()
        self._setup_toolbar()
        self._setup_menu()
        self._connect_signals()

    def _setup_docks(self):
        # Channel panel — left
        self.channel_dock = QDockWidget("Channels", self)
        self.channel_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.channel_dock)

        # Measurement panel — right
        self.measure_dock = QDockWidget("Measurements", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.measure_dock)

        # Analysis tabs — bottom
        self.analysis_dock = QDockWidget("Analysis", self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.analysis_dock)
```

### Thread Manager Pattern (LAW 2 — never block UI thread)
```python
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject

class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # emits result
    error    = pyqtSignal(str)      # emits error message
    progress = pyqtSignal(int)      # emits 0-100

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

# Usage in MainWindow:
def _load_file(self, path: str):
    worker = Worker(self._parse_file, path)
    worker.signals.finished.connect(self._on_record_loaded)
    worker.signals.error.connect(self._on_load_error)
    QThreadPool.globalInstance().start(worker)

def _on_record_loaded(self, record):
    # This runs on UI thread — safe to update UI here
    self.canvas.set_record(record)
```

### Signals & Slots — Cross-Layer Communication
```python
# In core/app_state.py — central signal hub
class AppState(QObject):
    cursor_moved    = pyqtSignal(int, float)    # (cursor_id, time_seconds)
    record_loaded   = pyqtSignal(object)        # DisturbanceRecord
    view_mode_changed = pyqtSignal(str)         # 'instantaneous'|'rms_half'|'rms_full'
    channel_toggled = pyqtSignal(int, bool)     # (channel_id, visible)

app_state = AppState()   # singleton — import and use everywhere
```

---

## PYQTGRAPH WAVEFORM CANVAS

### Canvas Setup (Critical Settings)
```python
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotDataItem, InfiniteLine, LinearRegionItem

pg.setConfigOptions(
    useOpenGL=True,          # REQUIRED — GPU rendering
    antialias=False,         # keep False for performance at high point counts
    foreground='w',
    background='#1E1E1E',    # dark engineering theme
)

class ChannelCanvas(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('#1E1E1E')
        self._plots = {}         # channel_id → PlotItem
        self._curves = {}        # channel_id → PlotDataItem
        self._cursors = {}       # cursor_id → InfiniteLine
        self._trigger_line = None
```

### Stacked Multi-Channel Layout
```python
def _setup_channels(self, record: DisturbanceRecord):
    self.clear()
    self._plots.clear()
    self._curves.clear()

    for i, ch in enumerate(record.analogue_channels):
        if not ch.visible:
            continue
        plot = self.addPlot(row=i, col=0)
        plot.setLabel('left', ch.name, units=ch.unit)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setXLink(self._plots[0] if self._plots else None)  # link X axes

        curve = PlotDataItem(pen=pg.mkPen(ch.colour, width=1))
        plot.addItem(curve)
        self._plots[ch.channel_id] = plot
        self._curves[ch.channel_id] = curve
```

### Viewport Update — ALWAYS use setData(), never remove/re-add
```python
def update_viewport(self, t_start: float, t_end: float):
    for ch_id, curve in self._curves.items():
        record = self._record
        ch = record.analogue_channels[ch_id]
        if not ch.visible:
            continue
        # Decimate for current viewport
        t_dec, d_dec = decimate_for_display(
            record.time_array, ch.raw_data, t_start, t_end
        )
        curve.setData(t_dec, d_dec)   # GPU update — DO NOT use removeItem/addItem
```

### Cursors
```python
def _add_cursors(self):
    for cursor_id, color in [(0, '#FFFF00'), (1, '#FF8800')]:
        line = InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine)
        )
        line.sigPositionChanged.connect(
            lambda l, cid=cursor_id: app_state.cursor_moved.emit(cid, l.value())
        )
        self._cursors[cursor_id] = line
        for plot in self._plots.values():
            plot.addItem(line)
```

### Trigger Line
```python
def _add_trigger_line(self, trigger_time_s: float):
    self._trigger_line = InfiniteLine(
        pos=trigger_time_s, angle=90, movable=False,
        pen=pg.mkPen('#FF4444', width=2, style=Qt.PenStyle.DotLine),
        label='T', labelOpts={'color': '#FF4444', 'position': 0.95}
    )
    for plot in self._plots.values():
        plot.addItem(self._trigger_line)
```

### Zoom-to-Fault
```python
def zoom_to_fault(self, record: DisturbanceRecord, window_s: float = 0.2):
    """Centre view on trigger ± window_s seconds."""
    t_trig = (record.trigger_time - record.start_time).total_seconds()
    t_start = max(0, t_trig - window_s)
    t_end   = min(record.time_array[-1], t_trig + window_s)
    for plot in self._plots.values():
        plot.setXRange(t_start, t_end, padding=0)
```

---

## PHASOR CANVAS (ui/phasor_canvas.py)

```python
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
import numpy as np

class PhasorCanvas(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        radius = min(cx, cy) - 40

        # Draw reference circle
        painter.setPen(QPen(QColor('#444444'), 1))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Draw phasor arrow
        def draw_phasor(phasor: complex, colour: str, label: str):
            mag = abs(phasor) / self._v_scale  # normalise to canvas
            angle_rad = np.angle(phasor)
            end_x = cx + radius * mag * np.cos(angle_rad)
            end_y = cy - radius * mag * np.sin(angle_rad)  # Y flipped
            painter.setPen(QPen(QColor(colour), 2))
            painter.drawLine(QPointF(cx, cy), QPointF(end_x, end_y))
```

---

## COLOUR SCHEME (dark engineering theme)

```python
COLOURS = {
    'background':  '#1E1E1E',
    'grid':        '#333333',
    'phase_a':     '#FF4444',   # red
    'phase_b':     '#FFCC00',   # yellow
    'phase_c':     '#4488FF',   # blue
    'earth':       '#44BB44',   # green
    'digital':     '#AAAAAA',   # grey
    'cursor_a':    '#FFFF00',   # yellow cursor
    'cursor_b':    '#FF8800',   # orange cursor
    'trigger':     '#FF4444',   # red trigger line
    'dig_trip':    '#FF2222',
    'dig_cb':      '#FF8800',
    'dig_pickup':  '#FFAA00',
    'text':        '#DDDDDD',
    'axis':        '#888888',
}
```

---

## COMMON MISTAKES TO AVOID

```
❌ plot.removeItem(curve); plot.addItem(new_curve)  → causes flicker, kills performance
✅ curve.setData(x, y)                               → GPU-efficient in-place update

❌ for sample in data: process(sample)              → Python loop on millions of points
✅ NumPy vectorised operations on whole arrays       → C-speed processing

❌ Heavy computation in button click handler         → freezes UI (violates LAW 2)
✅ Dispatch to QThreadPool Worker, update via signal → smooth UI always

❌ plot.setData(time_array, raw_data)               → rendering 6M+ points = freeze
✅ Decimate first: x, y = decimator(time, data, t0, t1)  → max 4000 points
```
