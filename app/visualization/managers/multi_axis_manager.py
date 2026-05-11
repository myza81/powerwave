from __future__ import annotations

import dataclasses

import pyqtgraph as pg


@dataclasses.dataclass
class _AxisEntry:
    name: str
    viewbox: pg.ViewBox
    axis_item: pg.AxisItem
    curve: pg.PlotDataItem
    color: str


class MultiAxisManager:
    """ViewBox and AxisItem lifecycle manager for the N-Axis Single Canvas.

    Each call to add_axis() followed by register() adds one independent Y-axis
    for a named parameter. The first parameter reuses the primary PlotItem's
    existing left axis. Subsequent parameters get new ViewBoxes linked to the
    primary via setXLink().

    Geometry synchronization is wired automatically via sigResized so that
    secondary ViewBoxes always align with the primary PlotItem's scene rect.
    """

    def __init__(
        self,
        primary_plot: pg.PlotItem,
        layout: pg.GraphicsLayoutWidget,
    ) -> None:
        self._primary = primary_plot
        self._layout = layout
        self._axes: dict[str, _AxisEntry] = {}
        self._pending_axis: dict[str, pg.AxisItem] = {}
        self._right_col = 2
        self._primary.getViewBox().sigResized.connect(self._sync_geometries)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_axis(self, name: str, unit: str, color: str) -> pg.ViewBox:
        """Create and stage a new ViewBox + AxisItem for the named parameter.

        Returns the ViewBox the caller must add curves into.
        Must be followed by register() to complete the entry.
        Raises ValueError if name is already registered.
        """
        if name in self._axes:
            raise ValueError(f"Parameter '{name}' is already registered")

        # First parameter reuses the primary PlotItem's left axis
        if not self._axes and not self._pending_axis:
            self._primary.setLabel("left", name, units=unit)
            left_ax: pg.AxisItem = self._primary.getAxis("left")
            left_ax.setPen(pg.mkPen(color))
            self._pending_axis[name] = left_ax
            return self._primary.getViewBox()

        # Secondary parameters: independent ViewBox + right-side AxisItem
        vb = pg.ViewBox()
        vb.setXLink(self._primary)
        scene = self._primary.scene() or self._layout.scene()
        if scene is None:
            raise RuntimeError("Cannot add secondary plot axis without a Qt scene")
        scene.addItem(vb)

        axis = pg.AxisItem(orientation="right")
        axis.setLabel(name, units=unit)
        axis.setPen(pg.mkPen(color))
        axis.setTextPen(pg.mkPen(color))
        axis.linkToView(vb)

        self._layout.addItem(axis, row=0, col=self._right_col)
        self._right_col += 1

        # Align new ViewBox to primary before the first resize event
        primary_rect = self._primary.getViewBox().sceneBoundingRect()
        vb.setGeometry(primary_rect)
        vb.linkedViewChanged(self._primary.getViewBox(), vb.XAxis)

        self._pending_axis[name] = axis
        return vb

    def register(
        self,
        name: str,
        viewbox: pg.ViewBox,
        curve: pg.PlotDataItem,
        color: str,
    ) -> None:
        """Complete the axis entry after add_axis() and curve creation."""
        axis_item = self._pending_axis.pop(name)
        self._axes[name] = _AxisEntry(
            name=name,
            viewbox=viewbox,
            axis_item=axis_item,
            curve=curve,
            color=color,
        )

    def remove_axis(self, name: str) -> None:
        """Remove a parameter's ViewBox, AxisItem, and curve from the scene."""
        if name not in self._axes:
            return
        entry = self._axes.pop(name)
        primary_vb = self._primary.getViewBox()
        if entry.viewbox is not primary_vb:
            if entry.viewbox.scene():
                entry.viewbox.scene().removeItem(entry.viewbox)
            if entry.axis_item.scene():
                entry.axis_item.scene().removeItem(entry.axis_item)

    def clear(self) -> None:
        """Remove all secondary ViewBoxes and axes. Reset to initial state."""
        primary_vb = self._primary.getViewBox()
        for entry in list(self._axes.values()):
            if entry.viewbox is not primary_vb:
                if entry.viewbox.scene():
                    entry.viewbox.scene().removeItem(entry.viewbox)
                if entry.axis_item.scene():
                    entry.axis_item.scene().removeItem(entry.axis_item)
        self._axes.clear()
        self._pending_axis.clear()

    def get_viewboxes(self) -> list[pg.ViewBox]:
        """Return secondary ViewBoxes only (primary ViewBox excluded)."""
        primary_vb = self._primary.getViewBox()
        return [e.viewbox for e in self._axes.values() if e.viewbox is not primary_vb]

    def get_curves(self) -> dict[str, pg.PlotDataItem]:
        """Return name → curve mapping for all registered parameters."""
        return {name: e.curve for name, e in self._axes.items()}

    def parameter_names(self) -> list[str]:
        """Return names of all registered parameters in insertion order."""
        return list(self._axes.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry synchronization
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_geometries(self) -> None:
        """Keep secondary ViewBox geometries aligned with the primary ViewBox.

        Connected to sigResized — fires automatically on widget resize.
        Per VIEWPORT_RENDERING_POLICY §16.4: without this, secondary ViewBoxes
        stay at zero geometry and their curves are invisible.
        """
        primary_vb = self._primary.getViewBox()
        scene_rect = primary_vb.sceneBoundingRect()
        for entry in self._axes.values():
            if entry.viewbox is not primary_vb:
                entry.viewbox.setGeometry(scene_rect)
                entry.viewbox.linkedViewChanged(primary_vb, entry.viewbox.XAxis)
