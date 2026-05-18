"""Concrete harmonic magnitude visualization overlay using BaseOverlay and CurveStore.

HarmonicCurveOverlay manages per-order RMS magnitude curves on any PyQtGraph
plot host.  It is a general-purpose reusable overlay that follows the same
lifecycle contract as PhasorCurveOverlay.

For FlexiblePlotCanvas multi-axis integration, the inline
_build_harmonic_overlays() approach in flexible_plot_canvas.py is used because
it requires per-channel ViewBox access.  HarmonicCurveOverlay is useful for
simpler single-PlotItem hosts (e.g. a standalone spectrum panel) or as a
building block in future Phase 8B interactive spectrum rendering.

Key design: each (channel_name, harmonic_order) pair maps to exactly one
PlotDataItem managed by CurveStore — no duplicate items, safe to call
update_channel() repeatedly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.visualization.overlays.base_overlay import BaseOverlay
from app.visualization.overlays.curve_store import CurveStore
from app.visualization.overlays.overlay_colors import (
    OverlayTheme,
    harmonic_curve_label,
    harmonic_order_pen,
)


class HarmonicCurveOverlay(BaseOverlay):
    """Manages harmonic magnitude overlay curves on a PyQtGraph plot host.

    Each (channel_name, harmonic_order) pair maps to one PlotDataItem via
    CurveStore — creating a new item only on the first call for that pair,
    reusing it on subsequent calls.

    Lifecycle:
      1. Create.
      2. Call attach(canvas) — extracts the primary PlotItem from the canvas.
      3. Call update_channel() for each channel/order to add/refresh curves.
      4. Call set_visible(False/True) to hide/show all curves atomically.
      5. Call clear() to zero data while keeping curves attached.
      6. Call dispose() to permanently remove all curves and release resources.
    """

    def __init__(self, *, theme: OverlayTheme | None = None) -> None:
        super().__init__()
        self._theme = theme
        self._store = CurveStore()
        self._plot_item: Any = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update_channel(
        self,
        channel_name: str,
        harmonic_time: np.ndarray,
        magnitude: np.ndarray,
        order: int,
        *,
        pen_width: float = 1.5,
    ) -> None:
        """Add or refresh the harmonic magnitude curve for (channel_name, order).

        Creates a new PlotDataItem on the first call for this pair; reuses the
        existing one on subsequent calls (CurveStore deduplication — no
        duplicate scene items regardless of call count).

        Args:
            channel_name:  Unique channel identifier.
            harmonic_time: Right-aligned time array from extract_harmonics().
            magnitude:     RMS magnitude array for this harmonic order.
            order:         Harmonic order integer (1 = fundamental, 3 = 3rd, …).
            pen_width:     Line width override (default 1.5).
        """
        if self._disposed or self._plot_item is None:
            return
        key = (channel_name, order)
        pen = harmonic_order_pen(order, width=pen_width)
        label = harmonic_curve_label(channel_name, order)
        curve = self._store.get_or_create_curve(
            key, self._plot_item, name=label, pen=pen
        )
        curve.setVisible(self._visible)
        curve.setData(
            np.asarray(harmonic_time, dtype=np.float64),
            np.asarray(magnitude, dtype=np.float64),
        )

    def remove_channel(self, channel_name: str) -> None:
        """Remove all harmonic curves for a given channel (all orders)."""
        keys_to_remove = [
            k for k in self._store.keys()
            if isinstance(k, tuple) and len(k) == 2 and k[0] == channel_name
        ]
        for key in keys_to_remove:
            self._store.remove_curve(key)

    def remove_order(self, channel_name: str, order: int) -> None:
        """Remove the curve for a specific (channel_name, order) pair."""
        self._store.remove_curve((channel_name, order))

    def channel_order_pairs(self) -> list[tuple[str, int]]:
        """Return all currently managed (channel_name, order) pairs."""
        return [
            k for k in self._store.keys()
            if isinstance(k, tuple) and len(k) == 2
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # BaseOverlay hook implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _attach(self, canvas: Any) -> None:
        if hasattr(canvas, "_primary_plot"):
            self._plot_item = canvas._primary_plot
        elif hasattr(canvas, "addItem"):
            self._plot_item = canvas
        else:
            self._plot_item = None

    def _detach(self) -> None:
        self._store.dispose()
        self._plot_item = None

    def _set_items_visible(self, visible: bool) -> None:
        for key in self._store.keys():
            curve = self._store.get_curve(key)
            if curve is not None:
                curve.setVisible(visible)

    def _clear(self) -> None:
        self._store.clear_data()

    def _dispose(self) -> None:
        self._store.dispose()
        self._plot_item = None
