"""Calculated Signal Resolution Service (Phase 2C-3).

Sits between the two layers that must never touch each other directly::

    EventAnalysisSession
        owns definitions, results, dependencies, freshness
            |
    CalculatedSignalResolutionService (this module)
        resolves session data and orchestrates recalculation
            |
    app.calculated_signals.engine.calculate_signal()
        performs numerical calculation only

EventAnalysisSession never imports or calls calculate_signal() (see
app/sessions/event_session.py's Phase 2C-2 section docstring); this service
is the only place that bridges live session state to the numerically pure
Phase 2B engine. It holds a session reference and orchestrates, but never
duplicates, session state: dependency validity, source/channel lookup, and
result storage are all delegated back to the session's own API.

Full resolution only, never display resolution: inputs are built by reading
DisturbanceRecord.waveform_data directly and applying only
alignment_engine.apply_time_offset() -- this service deliberately does not
call EventAnalysisSession.build_aligned_data(), because that method clips to
a view window and decimates to at most max_points for on-screen rendering.
A calculated signal's numerical result must reflect every sample in the
overlap region, not a display-decimated subset.

preview_definition() (Phase 3A) resolves and calculates a definition that is
not yet, and may never be, registered in the session -- e.g. for a UI
preview -- reusing the exact same input-resolution and calculation path as
resolve_one() but never touching session state at all.

Synchronous only: every method in this module runs to completion on the
calling thread. No QRunnable, QThreadPool, or other worker/thread
infrastructure is used or assumed here -- a future UI phase that wants a
responsive offset-drag experience is responsible for deciding when to call
this service (e.g. on editingFinished rather than every valueChanged tick),
not this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.calculated_signals.engine import (
    CalculatedSignalEngineError,
    CalculationEngineConfig,
    calculate_signal,
)
from app.calculated_signals.expression import CalculatedSignalExpressionError
from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    ChannelRef,
    ResolvedAnalogInput,
)
from app.sessions import alignment_engine

if TYPE_CHECKING:
    from app.sessions.event_session import EventAnalysisSession

__all__ = [
    "CalculatedSignalResolutionError",
    "ResolutionFailure",
    "ResolutionBatchResult",
    "CalculatedSignalResolutionService",
]


class CalculatedSignalResolutionError(RuntimeError):
    """A calculated signal could not be resolved and/or recalculated.

    Covers both resolution failures (unresolvable dependency: missing or
    inactive source, missing/digital/non-numeric channel) and calculation
    failures (calculate_signal() raised a CalculatedSignalEngineError) --
    callers that only care "did resolve_one succeed" do not need to
    distinguish the two; the message explains which occurred.

    Deliberately distinct from CalculatedSignalEngineError: that family
    reports purely numerical/expression problems from the Phase 2B engine
    and is raised by calculate_signal() itself. This type is raised by the
    resolution layer, including for failures calculate_signal() never sees
    (e.g. a dependency that is not resolvable at all).
    """


@dataclass(frozen=True, slots=True)
class ResolutionFailure:
    """One calc_id's resolution/calculation failure, from a batch resolve.

    Deliberately small -- a calc_id and a human-readable message, not a
    stored exception or traceback -- since batch callers (a future
    recalculation controller, or a test) only need to know which signals
    failed and why, not reconstruct the original exception.
    """

    calc_id: str
    message: str


@dataclass(frozen=True, slots=True)
class ResolutionBatchResult:
    """The outcome of resolving more than one calculated signal at once.

    successful holds the freshly computed results, in the order they were
    resolved; failures holds one ResolutionFailure per calc_id that could
    not be resolved. A batch method never raises on an individual failure --
    it continues past it and reports it here instead, so one broken
    calculated signal cannot block every other one in the same batch.
    """

    successful: tuple[CalculatedSignalResult, ...]
    failures: tuple[ResolutionFailure, ...]


class CalculatedSignalResolutionService:
    """Resolves calculated-signal dependencies against a live session and
    invokes the Phase 2B engine to (re)calculate them.

    Holds a reference to exactly one EventAnalysisSession, supplied at
    construction, and an optional CalculationEngineConfig applied to every
    calculate_signal() call this service makes. Stateless beyond that: no
    caching, no queued work, no background recalculation -- every resolve_*
    call does its work synchronously and returns.
    """

    def __init__(
        self,
        session: "EventAnalysisSession",
        *,
        engine_config: CalculationEngineConfig | None = None,
    ) -> None:
        self._session = session
        self._engine_config = engine_config

    # -------------------------------------------------------------------------
    # Input resolution
    # -------------------------------------------------------------------------

    def _resolve_input(self, variable: str, ref: ChannelRef) -> ResolvedAnalogInput:
        """Resolve one ChannelRef to a full-resolution ResolvedAnalogInput.

        validate eligibility -> fetch source -> fetch analog data column ->
        fetch time column -> apply source.time_offset_s -> construct
        ResolvedAnalogInput. Never mutates the SessionSource, the
        DisturbanceRecord, or its waveform_data: raw_time/raw_values are
        converted to new float64 arrays, and apply_time_offset() itself
        returns a new array (time_array + offset_s), so nothing here writes
        back into session or record state.
        """
        eligibility = self._session.check_calculated_input_eligibility(ref)
        if not eligibility.is_valid:
            raise CalculatedSignalResolutionError(
                f"variable {variable!r} ({ref.source_id!r}/{ref.channel_name!r}) is "
                f"not a valid analog input: {eligibility.eligibility.value}"
            )

        source = self._session.get_source(ref.source_id)
        assert source is not None  # guaranteed by is_valid eligibility above
        record = source.record

        raw_time = record.waveform_data.get("time")
        raw_values = record.waveform_data.get(ref.channel_name)
        assert raw_time is not None and raw_values is not None  # guaranteed by eligibility

        time_arr = np.asarray(raw_time, dtype=np.float64)
        values_arr = np.asarray(raw_values, dtype=np.float64)
        shifted_time = alignment_engine.apply_time_offset(time_arr, source.time_offset_s)

        unit: str | None = None
        for ch in record.analog_channels:
            if ch.name == ref.channel_name:
                unit = ch.unit or None
                break

        return ResolvedAnalogInput(
            variable=variable,
            time=shifted_time,
            values=values_arr,
            unit=unit,
            source_id=ref.source_id,
            channel_name=ref.channel_name,
        )

    def _build_inputs(
        self, definition: CalculatedSignalDefinition
    ) -> dict[str, ResolvedAnalogInput]:
        """Resolve every declared binding, not only those the expression
        references -- calculate_signal() itself tolerates and ignores extra
        entries (Phase 2B's existing unused-binding policy), so this method
        does not need to know which bindings are actually referenced.
        """
        return {
            variable: self._resolve_input(variable, ref)
            for variable, ref in definition.variable_bindings.items()
        }

    # -------------------------------------------------------------------------
    # Single-signal resolution
    # -------------------------------------------------------------------------

    def resolve_one(self, calc_id: str) -> CalculatedSignalResult:
        """Resolve and recalculate one calculated signal.

        On success, stores the new result via
        session.set_calculated_signal_result() and returns it.

        On failure -- unresolvable dependency, an expression error from
        Phase 2A's validate_expression() (CalculatedSignalExpressionError),
        or a numerical/unit failure from Phase 2B's calculate_signal()
        (CalculatedSignalEngineError) -- the session's previously stored
        result, if any, is left completely untouched (including an ERROR
        result from a prior failure), and CalculatedSignalResolutionError is
        raised. This service never fabricates a placeholder/empty result and
        never destroys last-known-good data on a failed recalculation.

        Does not modify the calculated signal's definition or its
        dependency indexes -- both remain the session's sole responsibility.
        """
        definition = self._session.get_calculated_signal_definition(calc_id)
        if definition is None:
            raise CalculatedSignalResolutionError(f"unknown calc_id: {calc_id!r}")

        dependency_status = self._session.get_dependency_status(calc_id)
        if not dependency_status.is_resolvable:
            raise CalculatedSignalResolutionError(
                f"calc_id {calc_id!r} has unresolvable dependencies: "
                f"missing_sources={list(dependency_status.missing_sources)}, "
                f"inactive_sources={list(dependency_status.inactive_sources)}, "
                f"missing_channels={list(dependency_status.missing_channels)}, "
                f"digital_channels={list(dependency_status.digital_channels)}"
            )

        inputs = self._build_inputs(definition)

        try:
            result = calculate_signal(definition, inputs, config=self._engine_config)
        except (CalculatedSignalEngineError, CalculatedSignalExpressionError) as exc:
            raise CalculatedSignalResolutionError(
                f"calc_id {calc_id!r} failed to calculate: {exc}"
            ) from exc

        self._session.set_calculated_signal_result(calc_id, result)
        return result

    # -------------------------------------------------------------------------
    # Preview (Phase 3A)
    # -------------------------------------------------------------------------

    def preview_definition(self, definition: CalculatedSignalDefinition) -> CalculatedSignalResult:
        """Resolve and calculate *definition* without registering it, or
        storing anything, in the session -- for UI preview only.

        Reuses exactly the same input-resolution (_resolve_input via
        _build_inputs) and calculation (calculate_signal) path as
        resolve_one(), so a preview's numerical result is identical to what
        creating and resolving the same definition would produce. Two
        differences from resolve_one(): *definition* is supplied directly
        rather than looked up by an already-registered calc_id (a preview
        happens before the user has committed to creating anything), and
        nothing is ever written back to the session -- not the definition,
        not a result, not a dependency index entry. Each binding is
        validated by _resolve_input()'s own eligibility check (the same
        eligibility contract enforced everywhere else in this module), so a
        preview failure reports the same reason a resolve_one() failure
        would for the same inputs.

        Raises CalculatedSignalResolutionError on any resolution or
        calculation failure, exactly like resolve_one().
        """
        inputs = self._build_inputs(definition)

        try:
            return calculate_signal(definition, inputs, config=self._engine_config)
        except (CalculatedSignalEngineError, CalculatedSignalExpressionError) as exc:
            raise CalculatedSignalResolutionError(
                f"failed to calculate {definition.name!r}: {exc}"
            ) from exc

    # -------------------------------------------------------------------------
    # Batch resolution
    # -------------------------------------------------------------------------

    def _resolve_batch(self, calc_ids: "tuple[str, ...] | list[str]") -> ResolutionBatchResult:
        successful: list[CalculatedSignalResult] = []
        failures: list[ResolutionFailure] = []
        for calc_id in calc_ids:
            try:
                successful.append(self.resolve_one(calc_id))
            except CalculatedSignalResolutionError as exc:
                failures.append(ResolutionFailure(calc_id=calc_id, message=str(exc)))
        return ResolutionBatchResult(successful=tuple(successful), failures=tuple(failures))

    def resolve_all_stale(self) -> ResolutionBatchResult:
        """Recalculate every calculated signal currently STALE.

        Order is session.get_stale_calculated_signal_ids()'s own
        deterministic (insertion) order. A failure on one calc_id does not
        prevent the remaining stale signals from being attempted.
        """
        return self._resolve_batch(self._session.get_stale_calculated_signal_ids())

    def resolve_for_source(self, source_id: str) -> ResolutionBatchResult:
        """Recalculate every calculated signal that depends on *source_id*,
        regardless of its current status (OK, STALE, ERROR, or never
        calculated).

        Uses session.get_calculated_dependents_for_source() -- the
        session's own reverse dependency index -- rather than scanning
        every calculated signal's bindings by hand.
        """
        return self._resolve_batch(self._session.get_calculated_dependents_for_source(source_id))

    def resolve_all(self) -> ResolutionBatchResult:
        """Recalculate every calculated signal defined in the session,
        regardless of current status.

        Order matches session.list_calculated_signals()'s own deterministic
        (insertion) order.
        """
        calc_ids = tuple(entry.definition.calc_id for entry in self._session.list_calculated_signals())
        return self._resolve_batch(calc_ids)
