from __future__ import annotations

from collections.abc import Hashable

from app.visualization.overlays.base_overlay import VisualizationOverlay


class OverlayRegistry:
    """Track visualization overlays by owner and key."""

    def __init__(self) -> None:
        self._overlays: dict[Hashable, dict[Hashable, VisualizationOverlay]] = {}

    def register(
        self,
        owner_id: Hashable,
        key: Hashable,
        overlay: VisualizationOverlay,
    ) -> None:
        owner_overlays = self._overlays.setdefault(owner_id, {})
        if key in owner_overlays:
            raise ValueError(
                f"Overlay key {key!r} is already registered for owner {owner_id!r}"
            )
        owner_overlays[key] = overlay

    def get(
        self,
        owner_id: Hashable,
        key: Hashable,
    ) -> VisualizationOverlay | None:
        return self._overlays.get(owner_id, {}).get(key)

    def unregister(
        self,
        owner_id: Hashable,
        key: Hashable,
        *,
        dispose: bool = True,
    ) -> VisualizationOverlay | None:
        owner_overlays = self._overlays.get(owner_id)
        if not owner_overlays or key not in owner_overlays:
            return None
        overlay = owner_overlays.pop(key)
        if dispose:
            overlay.dispose()
        if not owner_overlays:
            self._overlays.pop(owner_id, None)
        return overlay

    def clear_owner(self, owner_id: Hashable, *, dispose: bool = True) -> None:
        owner_overlays = self._overlays.pop(owner_id, {})
        if dispose:
            for overlay in owner_overlays.values():
                overlay.dispose()

    def set_visible(
        self,
        owner_id: Hashable,
        key: Hashable,
        visible: bool,
    ) -> bool:
        overlay = self.get(owner_id, key)
        if overlay is None:
            return False
        overlay.set_visible(visible)
        return True

    def keys(self, owner_id: Hashable) -> list[Hashable]:
        return list(self._overlays.get(owner_id, {}).keys())
