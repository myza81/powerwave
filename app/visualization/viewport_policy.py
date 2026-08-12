"""Initial-viewport selection policy (Stage 2).

Answers one question: *which portion of the session should the engineer see
first?* That is deliberately a different question from *what time range exists
in the session?*, which is ``SessionCanvasController._session_window()``'s job
and is left completely untouched by this module.

    SESSION DATA DOMAIN   _session_window()          feeds build_aligned_data,
                                                     RMS/phasor/harmonic,
                                                     navigator, clipping,
                                                     "Fit All"
    INITIAL VIEWPORT      select_initial_viewport()  feeds one setXRange at
                                                     first activation

Keeping them separate is what stops a display preference from silently
narrowing the window analytics compute over.

Coordinate system
-----------------
Everything here works in SESSION TIME — ``raw_elapsed_time + time_offset_s``.
Stage 1 (``app.sessions.absolute_alignment``) is the single alignment
authority; this module never looks at ``start_time``/``trigger_time`` to
re-derive a wall-clock offset, it only reads ``SessionSource.time_offset_s``
that Stage 1 already computed. The one exception is the trigger's position
*within its own record* — ``trigger_time - start_time`` — which is intra-record
geometry, not cross-source alignment.

Purity
------
No Qt, no session mutation, no offset writes, no record modification, no
resampling, no interpolation. Reads metadata and time arrays; returns two
scalars or None.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from app.models.disturbance_record import DisturbanceRecord
    from app.sessions.session_models import SessionSource

# Fraction of the session data domain beyond which event focusing stops being
# worth it. When the candidate window already covers at least this much of the
# domain, the analyst gains little from a narrower initial view and would lose
# surrounding context, so the policy declines (returns None) and normal
# full-session behaviour is retained. Half the domain is a deliberately plain
# rule: no user-facing threshold, no per-provider tuning.
MAX_BENEFIT_FRACTION = 0.5

# Padding added on each side of the logical event window, as a fraction of its
# width — matches the deprecated multi-source selector this policy replaces.
_PAD_FRACTION = 0.05
_MIN_PAD_S = 0.001

# Below two active sources there is no cross-source visibility problem to
# solve, and single-source sessions must keep behaving exactly as they do now.
MIN_SOURCES_FOR_EVENT_FOCUS = 2


# ---------------------------------------------------------------------------
# Pure geometry helpers
#
# Adapted from the (deprecated, never reactivated) VisualizationManager
# helpers, with one substantive change: each takes a scalar offset_seconds and
# adds it on read, instead of requiring the caller to have materialised a
# shifted copy of the record via _apply_time_offset(). That copy is what made
# the old path expensive; Stage 1's lazy scalar offset makes it unnecessary.
# ---------------------------------------------------------------------------


def _time_column(record: "DisturbanceRecord") -> np.ndarray | None:
    wf = record.waveform_data
    if wf is None or "time" not in getattr(wf, "columns", ()) or wf.empty:
        return None
    t = wf["time"].to_numpy(dtype=np.float64)
    return t if len(t) else None


def time_extent(
    record: "DisturbanceRecord", offset_seconds: float = 0.0
) -> tuple[float, float] | None:
    """First and last sample of *record* in session time, or None when empty."""
    t = _time_column(record)
    if t is None:
        return None
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return lo + offset_seconds, hi + offset_seconds


def trigger_display_time(
    record: "DisturbanceRecord", offset_seconds: float = 0.0
) -> float | None:
    """Session-time X of *record*'s trigger, or None when it has no real one.

    Returns None when ``trigger_time == start_time``. That equality is not a
    guess: ``app.import_wizard.disturbance_record_bridge`` sets
    ``trigger_time=start_time`` unconditionally for every CSV/Excel import
    because those formats carry no independent trigger, so equality is the
    concrete signal that a source has no event marker of its own. Without this
    rule a two-hour SCADA trend would anchor the viewport at its first sample
    purely because of that synthetic trigger.

    The rule costs a rare false negative — a COMTRADE recording whose trigger
    genuinely coincides with its first sample is not offered as an anchor — and
    that source still contributes its samples for bracketing. Preferring a
    missed anchor over a meaningless one is the safe direction.
    """
    timing = getattr(record, "timing_info", None)
    if timing is None or timing.trigger_time is None or timing.start_time is None:
        return None
    if timing.trigger_time == timing.start_time:
        return None
    try:
        local_trigger_s = (timing.trigger_time - timing.start_time).total_seconds()
    except (TypeError, AttributeError):
        return None
    return float(offset_seconds) + float(local_trigger_s)


def nearest_sample_bounds(
    record: "DisturbanceRecord",
    center_s: float,
    offset_seconds: float = 0.0,
) -> tuple[float, float] | None:
    """Session-time X of the real samples bracketing *center_s*.

    Returns ``(at_or_before, at_or_after)``. Both are actual recorded sample
    positions — nothing is interpolated, and no synthetic sample is invented at
    *center_s*. Uses ``np.searchsorted`` (O(log N)) rather than scanning.
    """
    t = _time_column(record)
    if t is None:
        return None
    local_center = center_s - offset_seconds
    idx = int(np.searchsorted(t, local_center, side="left"))
    lo_idx = max(0, min(idx - 1, len(t) - 1))
    hi_idx = max(0, min(idx, len(t) - 1))
    return float(t[lo_idx]) + offset_seconds, float(t[hi_idx]) + offset_seconds


def padded_range(t_start: float, t_end: float) -> tuple[float, float]:
    """Widen a logical window by _PAD_FRACTION on each side."""
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    width = t_end - t_start
    if width <= 0:
        width = max(abs(t_start) * 0.01, 1.0)
    pad = max(width * _PAD_FRACTION, _MIN_PAD_S)
    return t_start - pad, t_end + pad


def effective_sample_rate(record: "DisturbanceRecord") -> float:
    """Highest declared sampling rate, falling back to the median interval.

    Used only to rank event-anchor candidates, so the highest-resolution
    recording of an event wins over a sparse trend that happens to share it.
    """
    info = getattr(record, "sampling_info", None)
    rates = list(getattr(info, "sampling_rates", None) or [])
    positive = [float(r) for r in rates if r and float(r) > 0.0]
    if positive:
        return max(positive)
    t = _time_column(record)
    if t is None or len(t) < 2:
        return 0.0
    diffs = np.diff(t)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 0.0
    median_dt = float(np.median(positive_diffs))
    return 1.0 / median_dt if median_dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def select_initial_viewport(
    session,
    *,
    max_benefit_fraction: float = MAX_BENEFIT_FRACTION,
) -> tuple[float, float] | None:
    """Return the initial visible X range, or None to keep full-session view.

    None is returned — deliberately, not as a failure — when:
      - fewer than two active sources exist (single-source sessions unchanged);
      - no source offers a real event anchor (trend + trend, no trigger);
      - the candidate window would already cover most of the data domain, so
        focusing would cost context without buying visibility.

    Never mutates *session*.
    """
    sources: Sequence["SessionSource"] = [
        s for s in session.list_sources() if s.is_active
    ]
    if len(sources) < MIN_SOURCES_FOR_EVENT_FOCUS:
        return None

    extents: dict[str, tuple[float, float]] = {}
    for source in sources:
        extent = time_extent(source.record, source.time_offset_s)
        if extent is not None:
            extents[source.source_id] = extent
    if not extents:
        return None

    # Session data domain, unpadded. Computed here rather than imported from
    # SessionCanvasController._session_window() so this module stays Qt-free;
    # it is the same union, minus that function's 2% display margin, and is
    # used only as the denominator of the benefit guard.
    domain_lo = min(lo for lo, _ in extents.values())
    domain_hi = max(hi for _, hi in extents.values())
    domain_span = domain_hi - domain_lo
    if domain_span <= 0:
        return None

    # --- event anchor: highest effective sample rate among sources whose real
    #     trigger falls inside their own extent -----------------------------
    candidates: list[tuple[float, float, str]] = []
    for source in sources:
        extent = extents.get(source.source_id)
        if extent is None:
            continue
        trigger_s = trigger_display_time(source.record, source.time_offset_s)
        if trigger_s is None or not (extent[0] <= trigger_s <= extent[1]):
            continue
        candidates.append(
            (effective_sample_rate(source.record), trigger_s, source.source_id)
        )
    if not candidates:
        return None

    _rate, event_s, anchor_id = max(candidates, key=lambda c: c[0])
    window_lo, window_hi = extents[anchor_id]

    # --- expand to the real bracketing samples of every other in-range source
    for source in sources:
        if source.source_id == anchor_id:
            continue
        extent = extents.get(source.source_id)
        # A source whose recording does not span the event contributes nothing:
        # stretching the viewport to reach its nearest sample could span hours.
        if extent is None or not (extent[0] <= event_s <= extent[1]):
            continue
        bounds = nearest_sample_bounds(source.record, event_s, source.time_offset_s)
        if bounds is None:
            continue
        window_lo = min(window_lo, bounds[0])
        window_hi = max(window_hi, bounds[1])

    window_lo = max(window_lo, domain_lo)
    window_hi = min(window_hi, domain_hi)
    if window_hi <= window_lo:
        return None

    if (window_hi - window_lo) >= max_benefit_fraction * domain_span:
        return None

    return padded_range(window_lo, window_hi)
