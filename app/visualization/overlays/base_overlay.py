from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VisualizationOverlay(Protocol):
    """Lifecycle contract for visualization overlays."""

    def attach(self, canvas: Any) -> None:
        """Bind the overlay to a canvas or plot host."""

    def detach(self) -> None:
        """Remove visual items from the canvas while keeping the overlay reusable."""

    def set_visible(self, visible: bool) -> None:
        """Set overlay visibility, safely before or after attach."""

    def update_view_range(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> None:
        """Refresh overlay data for the current visible X range."""

    def clear(self) -> None:
        """Clear plotted data while keeping the overlay reusable."""

    def dispose(self) -> None:
        """Permanently release visual items and references."""


class BaseOverlay:
    """Small reusable base for overlays with an idempotent lifecycle.

    Subclasses can override the protected hook methods to manage concrete
    plot items. Repeated attach/detach calls are safe and will not create
    duplicate items when attaching to the same canvas.
    """

    def __init__(self) -> None:
        self._canvas: Any | None = None
        self._visible = True
        self._disposed = False

    @property
    def canvas(self) -> Any | None:
        return self._canvas

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def disposed(self) -> bool:
        return self._disposed

    def attach(self, canvas: Any) -> None:
        if self._disposed:
            raise RuntimeError("Cannot attach a disposed overlay")
        if self._canvas is canvas:
            self._set_items_visible(self._visible)
            return
        if self._canvas is not None:
            self.detach()
        self._canvas = canvas
        self._attach(canvas)
        self._set_items_visible(self._visible)

    def detach(self) -> None:
        if self._canvas is None:
            return
        self._detach()
        self._canvas = None

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if not self._disposed:
            self._set_items_visible(self._visible)

    def update_view_range(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> None:
        if self._disposed:
            return
        self._update_view_range(x_min, x_max)

    def clear(self) -> None:
        if self._disposed:
            return
        self._clear()

    def dispose(self) -> None:
        if self._disposed:
            return
        self.detach()
        self.clear()
        self._dispose()
        self._disposed = True

    def _attach(self, canvas: Any) -> None:
        pass

    def _detach(self) -> None:
        pass

    def _set_items_visible(self, visible: bool) -> None:
        pass

    def _update_view_range(
        self,
        x_min: float | None,
        x_max: float | None,
    ) -> None:
        pass

    def _clear(self) -> None:
        pass

    def _dispose(self) -> None:
        pass
