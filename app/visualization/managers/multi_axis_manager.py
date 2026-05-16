from __future__ import annotations

import dataclasses

import pyqtgraph as pg

from app.visualization.engineering_display import EngineeringAxisLabel, format_axis_label


@dataclasses.dataclass
class _AxisEntry:
    name: str
    axis_key: str
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
        self._pending_axis: dict[str, tuple[str, pg.AxisItem]] = {}
        self._axis_views: dict[str, tuple[pg.ViewBox, pg.AxisItem]] = {}
        self._right_col = 1
        self._primary.getViewBox().sigResized.connect(self._sync_geometries)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def add_axis(
        self,
        name: str,
        unit: str,
        color: str,
        *,
        axis_key: str | None = None,
        axis_label: EngineeringAxisLabel | None = None,
    ) -> pg.ViewBox:
        """Create or reuse a ViewBox + AxisItem for the named parameter.

        Returns the ViewBox the caller must add curves into.
        Must be followed by register() to complete the entry.
        Raises ValueError if name is already registered.
        """
        if name in self._axes:
            raise ValueError(f"Parameter '{name}' is already registered")

        key = axis_key or name
        if key in self._axis_views:
            vb, axis = self._axis_views[key]
            self._pending_axis[name] = (key, axis)
            return vb

        label = axis_label or format_axis_label(name, unit)

        # First parameter reuses the primary PlotItem's left axis
        if not self._axis_views and not self._pending_axis:
            self._primary.setLabel("left", label.text, units=label.unit)
            left_ax: pg.AxisItem = self._primary.getAxis("left")
            left_ax.enableAutoSIPrefix(False)
            left_ax.setPen(pg.mkPen(color))
            left_ax.setTextPen(pg.mkPen(color))
            vb = self._primary.getViewBox()
            self._axis_views[key] = (vb, left_ax)
            self._pending_axis[name] = (key, left_ax)
            return vb

        # Secondary parameters: independent ViewBox + right-side AxisItem
        vb = pg.ViewBox()
        vb.setXLink(self._primary)
        scene = self._primary.scene() or self._layout.scene()
        if scene is None:
            raise RuntimeError("Cannot add secondary plot axis without a Qt scene")
        scene.addItem(vb)

        axis = pg.AxisItem(orientation="right")
        axis.enableAutoSIPrefix(False)
        axis.setLabel(label.text, units=label.unit)
        axis.setPen(pg.mkPen(color))
        axis.setTextPen(pg.mkPen(color))
        axis.linkToView(vb)

        self._layout.addItem(axis, row=0, col=self._right_col)
        self._right_col += 1

        # Align new ViewBox to primary before the first resize event
        primary_rect = self._primary.getViewBox().sceneBoundingRect()
        vb.setGeometry(primary_rect)
        vb.linkedViewChanged(self._primary.getViewBox(), vb.XAxis)

        self._axis_views[key] = (vb, axis)
        self._pending_axis[name] = (key, axis)
        return vb

    def register(
        self,
        name: str,
        viewbox: pg.ViewBox,
        curve: pg.PlotDataItem,
        color: str,
    ) -> None:
        """Complete the axis entry after add_axis() and curve creation."""
        axis_key, axis_item = self._pending_axis.pop(name)
        self._axes[name] = _AxisEntry(
            name=name,
            axis_key=axis_key,
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
        if any(e.axis_key == entry.axis_key for e in self._axes.values()):
            return
        self._axis_views.pop(entry.axis_key, None)
        primary_vb = self._primary.getViewBox()
        if entry.viewbox is not primary_vb:
            if entry.viewbox.scene():
                entry.viewbox.scene().removeItem(entry.viewbox)
            self._remove_axis_item(entry.axis_item)

    def clear(self) -> None:
        """Remove all secondary ViewBoxes and axes. Reset to initial state."""
        primary_vb = self._primary.getViewBox()
        removed: set[int] = set()
        for viewbox, axis_item in list(self._axis_views.values()):
            if viewbox is primary_vb:
                continue
            viewbox_id = id(viewbox)
            if viewbox_id in removed:
                continue
            removed.add(viewbox_id)
            if viewbox.scene():
                viewbox.scene().removeItem(viewbox)
            self._remove_axis_item(axis_item)
        self._axes.clear()
        self._pending_axis.clear()
        self._axis_views.clear()
        self._right_col = 1

    def get_viewboxes(self) -> list[pg.ViewBox]:
        """Return secondary ViewBoxes only (primary ViewBox excluded)."""
        primary_vb = self._primary.getViewBox()
        viewboxes: list[pg.ViewBox] = []
        seen: set[int] = set()
        for viewbox, _axis in self._axis_views.values():
            if viewbox is primary_vb or id(viewbox) in seen:
                continue
            seen.add(id(viewbox))
            viewboxes.append(viewbox)
        return viewboxes

    def get_curves(self) -> dict[str, pg.PlotDataItem]:
        """Return name → curve mapping for all registered parameters."""
        return {name: e.curve for name, e in self._axes.items()}

    def parameter_names(self) -> list[str]:
        """Return names of all registered parameters in insertion order."""
        return list(self._axes.keys())

    def axis_count(self) -> int:
        """Return the number of visible Y-axis groups."""
        return len(self._axis_views)

    def axis_keys(self) -> list[str]:
        """Return visible Y-axis group keys in insertion order."""
        return list(self._axis_views.keys())

    def _remove_axis_item(self, axis_item: pg.AxisItem) -> None:
        try:
            self._layout.removeItem(axis_item)
        except Exception:
            if axis_item.scene():
                axis_item.scene().removeItem(axis_item)

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
