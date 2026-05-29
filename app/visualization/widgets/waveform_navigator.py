"""WaveformNavigatorStrip — compact time-ruler with a draggable viewport window.

Shows the full time span of the session as a horizontal ruler.  A highlighted
region marks the currently visible window in the main canvas panels; dragging
the region pans all panels, dragging its edges zooms them in/out.

No waveform traces are drawn inside the strip — the handles speak for
themselves and keep the widget as compact as possible.
"""
from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


_REGION_FILL  = QColor(64, 160, 255, 45)   # subtle blue fill
_REGION_LINE  = "#40A0FF"                   # solid blue handle lines
_HANDLE_WIDTH = 3                           # px — thick so users can see & grab them
_BG_COLOR     = "#1A1A1A"
_HEIGHT_PX    = 52                          # compact; axis takes ~20 px inside


def _fmt_span(dt: float) -> str:
    """Human-readable span label for the viewport."""
    if dt < 0.001:
        return f"{dt * 1e6:.0f} µs"
    if dt < 1.0:
        return f"{dt * 1000:.1f} ms"
    return f"{dt:.3f} s"


class WaveformNavigatorStrip(QWidget):
    """Compact navigator ruler: drag the blue region to pan, drag edges to zoom.

    Public API
    ----------
    set_viewport(t_start, t_end)    — move the region to match canvas zoom
    set_time_range(t_min, t_max)    — tell the navigator the full data extent
    viewport_changed(float, float)  — emitted when the user drags
    """

    viewport_changed = pyqtSignal(float, float)   # (t_start, t_end) — user dragged

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._t_min: float = 0.0
        self._t_max: float = 1.0
        self._region_updating: bool = False
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(_BG_COLOR)
        # Bottom border line to separate navigator from the panels below
        self._glw.setStyleSheet(
            "pg--GraphicsLayoutWidget { border-top: 1px solid #2A2A2A;"
            " border-bottom: 1px solid #333; }"
        )

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.hideAxis("left")
        self._plot.showAxis("bottom")

        # Style the time axis: small monospace font, muted ticks
        axis = self._plot.getAxis("bottom")
        tick_font = QFont("Consolas", 8)
        axis.setTickFont(tick_font)
        axis.setTextPen(pg.mkPen("#777777"))
        axis.setPen(pg.mkPen("#333333"))

        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=False, y=False)
        # Keep the Y range fixed — there is no waveform content
        self._plot.setYRange(0, 1, padding=0)

        # ── Viewport region ──────────────────────────────────────────────────
        self._region = pg.LinearRegionItem(
            values=(self._t_min, self._t_max),
            orientation="vertical",
            brush=pg.mkBrush(_REGION_FILL),
            pen=pg.mkPen(_REGION_LINE, width=_HANDLE_WIDTH),
            movable=True,
            swapMode="push",
        )
        # Make both edge lines equally thick and visibly blue
        for line in self._region.lines:
            line.setPen(pg.mkPen(_REGION_LINE, width=_HANDLE_WIDTH))

        self._region.sigRegionChanged.connect(self._on_region_changed)
        self._plot.addItem(self._region)

        # Click anywhere in the plot to jump (center viewport on that point)
        self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

        layout.addWidget(self._glw)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_time_range(self, t_min: float, t_max: float) -> None:
        """Set the full data time extent — fixes the ruler X range."""
        if t_min >= t_max:
            return
        self._t_min = t_min
        self._t_max = t_max
        self._plot.setXRange(t_min, t_max, padding=0.01)

    def set_viewport(self, t_start: float, t_end: float) -> None:
        """Move the region to reflect an external X-range change (no signal)."""
        self._region_updating = True
        try:
            self._region.setRegion((t_start, t_end))
        finally:
            self._region_updating = False

    # No-op stubs — kept so the controller can call them without guarding
    def update_channel(self, key: str, time, values, color: str) -> None:  # noqa: ARG002
        pass

    def remove_channel(self, key: str) -> None:  # noqa: ARG002
        pass

    def clear_channels(self) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _on_region_changed(self) -> None:
        if self._region_updating:
            return
        t_start, t_end = self._region.getRegion()
        self.viewport_changed.emit(float(t_start), float(t_end))

    def _on_scene_clicked(self, event) -> None:
        """Left-click outside the region → center the current viewport on that x."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        vb = self._plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        view_pt = vb.mapSceneToView(pos)
        t = float(view_pt.x())

        r_start, r_end = self._region.getRegion()
        # Only jump if click is outside the current region
        if r_start <= t <= r_end:
            return
        span = r_end - r_start
        new_start = max(self._t_min, min(t - span / 2.0, self._t_max - span))
        new_end = new_start + span
        self._region.setRegion((new_start, new_end))
