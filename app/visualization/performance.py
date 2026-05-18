from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter

TimingSink = Callable[[str, float], None]


@contextmanager
def timed_section(
    name: str,
    *,
    enabled: bool = False,
    sink: TimingSink | None = None,
) -> Iterator[None]:
    """Optionally time a block and send elapsed milliseconds to a sink."""
    if not enabled:
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        if sink is not None:
            sink(name, (perf_counter() - start) * 1000.0)
