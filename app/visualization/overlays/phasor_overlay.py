"""Concrete phasor visualization overlay using BaseOverlay and CurveStore.

PhasorCurveOverlay manages per-channel phasor magnitude or angle curves on
any PyQtGraph plot host. It is a general-purpose reusable overlay.

Note: For FlexiblePlotCanvas multi-axis integration, the inline
_build_phasor_overlays() approach in flexible_plot_canvas.py is used because
it requires per-channel ViewBox access. PhasorCurveOverlay is useful for
simpler single-PlotItem hosts or standalone phasor display panels.

SEQUENCE_COMPONENTS rendering is intentionally excluded — sequence component
data is rendered via dedicated FlexiblePlotCanvas instances (separate panels).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.analytics.phasors.phasor_models import PhasorDisplayMode
from app.visualization.overlays.base_overlay import BaseOverlay
from app.visualization.overlays.curve_store import CurveStore
from app.visualization.overlays.overlay_colors import (
    OverlayTheme,
    phasor_curve_label,
    phasor_pen,
)


def _phasor_pen(color: str, mode: PhasorDisplayMode) -> Any:
    """Backward-compatible wrapper for the centralized phasor pen policy."""
    return phasor_pen("", mode, base_color=color)


class PhasorCurveOverlay(BaseOverlay):
    """Manages phasor magnitude or angle overlay curves on a PyQtGraph plot host.

    Lifecycle:
      1. Create with a PhasorDisplayMode.
      2. Call attach(canvas) — extracts the primary PlotItem from the canvas.
      3. Call update_channel() for each channel to add/refresh curves.
      4. Call set_visible(False/True) to hide/show all curves together.
      5. Call clear() to wipe data while keeping curves in place.
      6. Call dispose() to permanently remove curves and release resources.

    Mode must be MAGNITUDE or ANGLE. SEQUENCE_COMPONENTS is not supported here.
    """

    def __init__(
        self,
        mode: PhasorDisplayMode = PhasorDisplayMode.MAGNITUDE,
        *,
        theme: OverlayTheme | None = None,
    ) -> None:
        super().__init__()
        if mode not in (PhasorDisplayMode.MAGNITUDE, PhasorDisplayMode.ANGLE):
            raise ValueError(
                f"PhasorCurveOverlay supports MAGNITUDE or ANGLE, not {mode!r}"
            )
        self._mode = mode
        self._theme = theme
        self._store = CurveStore()
        self._plot_item: Any = None

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> PhasorDisplayMode:
        """The phasor display mode this overlay was created with."""
        return self._mode

    def update_channel(
        self,
        channel_name: str,
        phasor_time: np.ndarray,
        phasor_data: np.ndarray,
        *,
        color: str | None = None,
        signal_role: str | None = None,
    ) -> None:
        """Add or refresh a phasor curve for one channel.

        Creates a new curve on the first call; reuses the existing one on
        subsequent calls (via CurveStore — no duplicate PlotDataItems).

        Args:
            channel_name: Unique curve identifier (usually the channel name).
            phasor_time:  Right-aligned time array from extract_phasor().
            phasor_data:  Magnitude (RMS units) or angle (degrees) array.
            color:        Optional base color; theme policy still controls styling.
            signal_role:  Optional "voltage" or "current" role hint.
        """
        if self._disposed or self._plot_item is None:
            return
        pen = phasor_pen(
            channel_name,
            self._mode,
            theme=self._theme,
            signal_role=signal_role,
            base_color=color,
        )
        label = phasor_curve_label(channel_name, self._mode)
        curve = self._store.get_or_create_curve(
            channel_name,
            self._plot_item,
            name=label,
            pen=pen,
        )
        curve.setVisible(self._visible)
        curve.setData(
            np.asarray(phasor_time, dtype=np.float64),
            np.asarray(phasor_data, dtype=np.float64),
        )

    def remove_channel(self, channel_name: str) -> None:
        """Remove the curve for one channel and release its PlotDataItem."""
        self._store.remove_curve(channel_name)

    def channel_names(self) -> list[str]:
        """Return the names of all currently managed channels."""
        return [str(k) for k in self._store.keys()]

    # ─────────────────────────────────────────────────────────────────────
    # BaseOverlay hook implementations
    # ─────────────────────────────────────────────────────────────────────

    def _attach(self, canvas: Any) -> None:
        """Resolve and cache the primary PlotItem from the canvas host."""
        if hasattr(canvas, "_primary_plot"):
            self._plot_item = canvas._primary_plot
        elif hasattr(canvas, "addItem"):
            self._plot_item = canvas
        else:
            self._plot_item = None

    def _detach(self) -> None:
        """Dispose all curves (removes them from the scene) and clear state."""
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
