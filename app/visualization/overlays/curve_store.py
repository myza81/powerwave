from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pyqtgraph as pg

from app.visualization.overlays.overlay_render_policy import apply_curve_policy


class CurveStore:
    """Manage reusable PyQtGraph curves for visualization overlays."""

    def __init__(self) -> None:
        self._curves: dict[Hashable, pg.PlotDataItem] = {}
        self._plot_items: dict[Hashable, Any] = {}

    def get_or_create_curve(
        self,
        key: Hashable,
        plot_item: Any,
        *,
        name: str | None = None,
        pen: Any = None,
    ) -> pg.PlotDataItem:
        if key in self._curves:
            return self._curves[key]

        curve = pg.PlotDataItem(
            name=name,
            pen=pen,
            skipFiniteCheck=True,
        )
        apply_curve_policy(curve)
        if hasattr(plot_item, "addItem"):
            plot_item.addItem(curve)
        self._curves[key] = curve
        self._plot_items[key] = plot_item
        return curve

    def remove_curve(self, key: Hashable) -> pg.PlotDataItem | None:
        curve = self._curves.pop(key, None)
        plot_item = self._plot_items.pop(key, None)
        if curve is None:
            return None
        self._clear_curve(curve)
        if plot_item is not None and hasattr(plot_item, "removeItem"):
            try:
                plot_item.removeItem(curve)
            except Exception:  # noqa: BLE001
                pass
        return curve

    def clear_data(self) -> None:
        for curve in self._curves.values():
            self._clear_curve(curve)

    def dispose(self) -> None:
        for key in list(self._curves.keys()):
            self.remove_curve(key)
        self._curves.clear()
        self._plot_items.clear()

    def keys(self) -> list[Hashable]:
        return list(self._curves.keys())

    def get_curve(self, key: Hashable) -> pg.PlotDataItem | None:
        """Return a cached curve without creating a new item."""
        return self._curves.get(key)

    @staticmethod
    def _clear_curve(curve: pg.PlotDataItem) -> None:
        empty = np.empty(0, dtype=np.float64)
        curve.setData(empty, empty)
