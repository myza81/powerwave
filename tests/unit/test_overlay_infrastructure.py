from __future__ import annotations

from app.visualization.overlays import BaseOverlay, CurveStore, OverlayRegistry
from app.visualization.performance import timed_section


class _OverlayStub(BaseOverlay):
    def __init__(self) -> None:
        super().__init__()
        self.attach_count = 0
        self.detach_count = 0
        self.clear_count = 0
        self.dispose_count = 0
        self.visibility_calls: list[bool] = []

    def _attach(self, canvas) -> None:
        self.attach_count += 1

    def _detach(self) -> None:
        self.detach_count += 1

    def _set_items_visible(self, visible: bool) -> None:
        self.visibility_calls.append(visible)

    def _clear(self) -> None:
        self.clear_count += 1

    def _dispose(self) -> None:
        self.dispose_count += 1


class _FakePlotItem:
    def __init__(self) -> None:
        self.items = []

    def addItem(self, item) -> None:
        self.items.append(item)

    def removeItem(self, item) -> None:
        if item in self.items:
            self.items.remove(item)


class TestBaseOverlay:
    def test_visibility_before_attach_is_safe(self) -> None:
        overlay = _OverlayStub()
        overlay.set_visible(False)
        assert overlay.visible is False
        assert overlay.visibility_calls == [False]

    def test_repeated_attach_to_same_canvas_is_idempotent(self) -> None:
        canvas = object()
        overlay = _OverlayStub()
        overlay.attach(canvas)
        overlay.attach(canvas)
        assert overlay.attach_count == 1
        assert overlay.canvas is canvas


class TestOverlayRegistry:
    def test_register_prevents_duplicate_owner_key(self) -> None:
        registry = OverlayRegistry()
        registry.register("canvas-1", "phasor", _OverlayStub())
        try:
            registry.register("canvas-1", "phasor", _OverlayStub())
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate overlay key did not raise")

    def test_get_unregister_and_clear_owner_lifecycle(self) -> None:
        registry = OverlayRegistry()
        first = _OverlayStub()
        second = _OverlayStub()
        registry.register("canvas-1", "a", first)
        registry.register("canvas-1", "b", second)

        assert registry.get("canvas-1", "a") is first
        removed = registry.unregister("canvas-1", "a", dispose=False)
        assert removed is first
        assert first.disposed is False
        assert registry.keys("canvas-1") == ["b"]

        registry.clear_owner("canvas-1")
        assert second.disposed is True
        assert registry.keys("canvas-1") == []

    def test_set_visible_returns_false_for_missing_overlay(self) -> None:
        registry = OverlayRegistry()
        assert registry.set_visible("missing", "key", True) is False

    def test_set_visible_updates_registered_overlay(self) -> None:
        registry = OverlayRegistry()
        overlay = _OverlayStub()
        registry.register("canvas-1", "a", overlay)
        assert registry.set_visible("canvas-1", "a", False) is True
        assert overlay.visible is False


class TestCurveStore:
    def test_reuses_same_curve_for_same_key(self) -> None:
        plot = _FakePlotItem()
        store = CurveStore()

        first = store.get_or_create_curve("mag", plot, name="Magnitude")
        second = store.get_or_create_curve("mag", plot, name="Magnitude 2")

        assert first is second
        assert plot.items == [first]
        assert store.keys() == ["mag"]

    def test_remove_curve_removes_from_plot_item(self) -> None:
        plot = _FakePlotItem()
        store = CurveStore()
        curve = store.get_or_create_curve("angle", plot)

        removed = store.remove_curve("angle")

        assert removed is curve
        assert plot.items == []
        assert store.keys() == []

    def test_dispose_removes_all_curves(self) -> None:
        plot = _FakePlotItem()
        store = CurveStore()
        first = store.get_or_create_curve("a", plot)
        second = store.get_or_create_curve("b", plot)

        store.dispose()

        assert first not in plot.items
        assert second not in plot.items
        assert store.keys() == []


class TestTimedSection:
    def test_sink_not_called_when_disabled(self) -> None:
        calls = []
        with timed_section("overlay", enabled=False, sink=lambda *args: calls.append(args)):
            pass
        assert calls == []

    def test_sink_called_when_enabled(self) -> None:
        calls = []
        with timed_section("overlay", enabled=True, sink=lambda *args: calls.append(args)):
            pass
        assert len(calls) == 1
        name, elapsed_ms = calls[0]
        assert name == "overlay"
        assert elapsed_ms >= 0.0
