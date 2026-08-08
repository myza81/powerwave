"""Unit tests for Sprint 1B timing-reference compatibility detection in
app.sessions.timing_compatibility.

Covers classification (Step 18), the pairwise compatibility matrix
(Step 19), and session-level assessment (Step 20). Pure detection tests --
this module never repairs, shifts, or mutates anything, and neither do
these tests: every test that checks "no mutation" asserts it explicitly.

Uses only generic, synthetic session fixtures -- no filename, station, or
event identity is special-cased anywhere in this file or in production code.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.models.channels import AnalogChannel
from app.models.disturbance_record import DisturbanceRecord
from app.models.metadata import RecordingMetadata
from app.models.timing import SamplingInformation, TimingInformation
from app.sessions.event_session import EventAnalysisSession
from app.sessions.timing_compatibility import (
    TimingCompatibilityLevel,
    TimingReferenceClass,
    assess_pair_compatibility,
    assess_session_timing_compatibility,
    classify_source_timing,
    classify_timing_reference,
)

_EPOCH_FALLBACK = datetime(2000, 1, 1, 0, 0, 0)  # the known sentinel, see module docstring


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _timing(
    *,
    start: datetime = datetime(2026, 1, 1, 10, 0, 0),
    timezone: str | None = None,
    timing_reference: str = "absolute",
    time_axis_unit: str | None = None,
) -> TimingInformation:
    return TimingInformation(
        start_time=start,
        trigger_time=start,
        timezone=timezone,
        timing_reference=timing_reference,
        time_axis_unit=time_axis_unit,
    )


def _record(timing: TimingInformation, n: int = 10) -> DisturbanceRecord:
    t = np.linspace(0, 1, n)
    df = pd.DataFrame({"time": t, "A": np.arange(n, dtype=float)})
    return DisturbanceRecord(
        metadata=RecordingMetadata(
            station_name="GenericStation", recorder_name="GenericRecorder",
            source_file="generic.csv", provider_type="csv", nominal_frequency=50.0,
        ),
        waveform_data=df,
        analog_channels=[AnalogChannel(name="A", unit="MW", index=0)],
        digital_channels=[],
        sampling_info=SamplingInformation(sampling_rates=[10.0], samples_per_rate=[n]),
        timing_info=timing,
    )


def _add(sess: EventAnalysisSession, timing: TimingInformation, name: str) -> str:
    return sess.add_source(_record(timing), name, "csv")


# Named constructors for each classification category -- one place to keep
# them consistent across classification, matrix, and session tests.
def _absolute_aware() -> TimingInformation:
    return _timing(timezone="UTC")


def _absolute_naive() -> TimingInformation:
    return _timing(timezone=None)


def _elapsed_unanchored_explicit() -> TimingInformation:
    return _timing(timing_reference="relative_elapsed", time_axis_unit="s")


def _elapsed_unanchored_via_fallback_anchor() -> TimingInformation:
    """timing_reference defaults to "absolute" but start_time is the known
    no-real-timestamp sentinel -- must NOT be trusted as a real anchor."""
    return _timing(start=_EPOCH_FALLBACK, timing_reference="absolute")


def _reconstructed() -> TimingInformation:
    return _timing(timing_reference="synthetic_elapsed", time_axis_unit="s")


def _sample_index() -> TimingInformation:
    return _timing(timing_reference="sample_index", time_axis_unit="sample")


def _unknown() -> TimingInformation:
    return _timing(timing_reference="some_future_repair_strategy_not_yet_known")


# ─────────────────────────────────────────────────────────────────────────────
# Step 18 — Classification
# ─────────────────────────────────────────────────────────────────────────────


class TestClassification:
    def test_absolute_aware(self) -> None:
        assert classify_timing_reference(_absolute_aware()) == TimingReferenceClass.ABSOLUTE_AWARE

    def test_absolute_naive(self) -> None:
        assert classify_timing_reference(_absolute_naive()) == TimingReferenceClass.ABSOLUTE_NAIVE

    def test_elapsed_unanchored_explicit(self) -> None:
        assert (
            classify_timing_reference(_elapsed_unanchored_explicit())
            == TimingReferenceClass.ELAPSED_UNANCHORED
        )

    def test_elapsed_unanchored_via_fallback_anchor_not_trusted_as_absolute(self) -> None:
        """The core 'don't trust the label at face value' case: timing_reference
        says "absolute" (the field's own default) but start_time is the known
        placeholder sentinel used by csv_provider/excel_provider/disturbance_
        record_bridge when no real timestamp was found."""
        assert (
            classify_timing_reference(_elapsed_unanchored_via_fallback_anchor())
            == TimingReferenceClass.ELAPSED_UNANCHORED
        )

    def test_reconstructed(self) -> None:
        assert classify_timing_reference(_reconstructed()) == TimingReferenceClass.RECONSTRUCTED

    def test_sample_index(self) -> None:
        assert classify_timing_reference(_sample_index()) == TimingReferenceClass.SAMPLE_INDEX

    def test_unknown(self) -> None:
        assert classify_timing_reference(_unknown()) == TimingReferenceClass.UNKNOWN

    def test_comtrade_style_absolute_with_real_cfg_parsed_anchor(self) -> None:
        """COMTRADE never sets timing_reference explicitly (verified against
        app/providers/comtrade/comtrade_provider.py); its start_time always
        comes from a successfully parsed CFG date/time line (the parser
        raises rather than falling back to a placeholder). This must
        classify as a genuine absolute anchor, not as unanchored merely
        because waveform_data["time"] itself holds elapsed seconds."""
        comtrade_like = _timing(start=datetime(2024, 6, 15, 3, 22, 10, 500000), timezone=None)
        assert classify_timing_reference(comtrade_like) == TimingReferenceClass.ABSOLUTE_NAIVE

    def test_classify_source_timing_preserves_offset_and_method(self) -> None:
        sess = EventAnalysisSession()
        sid = _add(sess, _absolute_naive(), "Source A")
        sess.set_time_offset(sid, 2.5, method="manual")
        source = sess.get_source(sid)
        profile = classify_source_timing(source)
        assert profile.source_id == sid
        assert profile.display_name == "Source A"
        assert profile.time_offset_s == pytest.approx(2.5)
        assert profile.alignment_method == "manual"
        assert profile.reference_class == TimingReferenceClass.ABSOLUTE_NAIVE

    def test_classification_does_not_mutate_timing_information(self) -> None:
        timing = _absolute_naive()
        snapshot = (timing.start_time, timing.trigger_time, timing.timezone, timing.timing_reference)
        classify_timing_reference(timing)
        assert (timing.start_time, timing.trigger_time, timing.timezone, timing.timing_reference) == snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Step 19 — Compatibility matrix
# ─────────────────────────────────────────────────────────────────────────────


def _profile_for(sess: EventAnalysisSession, sid: str):
    return classify_source_timing(sess.get_source(sid))


class TestCompatibilityMatrix:
    def _pair_level(self, timing_a: TimingInformation, timing_b: TimingInformation) -> TimingCompatibilityLevel:
        sess = EventAnalysisSession()
        sid_a = _add(sess, timing_a, "A")
        sid_b = _add(sess, timing_b, "B")
        result = assess_pair_compatibility(_profile_for(sess, sid_a), _profile_for(sess, sid_b))
        return result.level

    def test_absolute_aware_plus_absolute_aware_is_compatible(self) -> None:
        assert self._pair_level(_absolute_aware(), _absolute_aware()) == TimingCompatibilityLevel.COMPATIBLE

    def test_absolute_aware_plus_absolute_naive_requires_review(self) -> None:
        assert self._pair_level(_absolute_aware(), _absolute_naive()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_absolute_naive_plus_absolute_naive_requires_review(self) -> None:
        """Both anchored, but timezone equivalence is assumed, not proven."""
        assert self._pair_level(_absolute_naive(), _absolute_naive()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_absolute_plus_unanchored_elapsed_requires_review(self) -> None:
        assert (
            self._pair_level(_absolute_naive(), _elapsed_unanchored_explicit())
            == TimingCompatibilityLevel.REQUIRES_REVIEW
        )

    def test_absolute_plus_sample_index_requires_review(self) -> None:
        assert self._pair_level(_absolute_naive(), _sample_index()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_absolute_plus_reconstructed_requires_review(self) -> None:
        assert self._pair_level(_absolute_naive(), _reconstructed()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_two_anchored_sources_requires_review_when_both_naive(self) -> None:
        """Covers the 'anchored elapsed + anchored elapsed' case from a
        COMTRADE-like record paired with another COMTRADE-like record --
        both anchored (naive), so REQUIRES_REVIEW (timezone assumption),
        matching the naive+naive rule."""
        comtrade_a = _timing(start=datetime(2024, 6, 15, 3, 0, 0))
        comtrade_b = _timing(start=datetime(2024, 6, 15, 3, 0, 5))
        assert self._pair_level(comtrade_a, comtrade_b) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_unanchored_elapsed_plus_unanchored_elapsed_requires_review(self) -> None:
        assert (
            self._pair_level(_elapsed_unanchored_explicit(), _elapsed_unanchored_explicit())
            == TimingCompatibilityLevel.REQUIRES_REVIEW
        )

    def test_sample_index_plus_sample_index_requires_review(self) -> None:
        """Matching index 0 does not prove a common real-world time."""
        assert self._pair_level(_sample_index(), _sample_index()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_reconstructed_plus_reconstructed_requires_review(self) -> None:
        assert self._pair_level(_reconstructed(), _reconstructed()) == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_unknown_plus_absolute_aware_is_unknown(self) -> None:
        assert self._pair_level(_unknown(), _absolute_aware()) == TimingCompatibilityLevel.UNKNOWN

    def test_unknown_plus_unanchored_elapsed_is_unknown(self) -> None:
        assert self._pair_level(_unknown(), _elapsed_unanchored_explicit()) == TimingCompatibilityLevel.UNKNOWN

    def test_unknown_plus_unknown_is_unknown(self) -> None:
        assert self._pair_level(_unknown(), _unknown()) == TimingCompatibilityLevel.UNKNOWN

    def test_result_is_factual_not_judgmental(self) -> None:
        """Compatible does not claim correctness -- just that metadata supports comparison."""
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_aware(), "A")
        sid_b = _add(sess, _absolute_aware(), "B")
        result = assess_pair_compatibility(_profile_for(sess, sid_a), _profile_for(sess, sid_b))
        assert "correct" not in result.message.lower()
        assert "can be compared directly" in result.message


# ─────────────────────────────────────────────────────────────────────────────
# Step 20 — Session assessment
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionAssessment:
    def test_zero_sources(self) -> None:
        sess = EventAnalysisSession()
        assessment = assess_session_timing_compatibility(sess)
        assert assessment.level == TimingCompatibilityLevel.COMPATIBLE
        assert not assessment.has_warning
        assert assessment.pair_results == ()

    def test_one_source_no_alarming_warning(self) -> None:
        sess = EventAnalysisSession()
        _add(sess, _absolute_naive(), "A")
        assessment = assess_session_timing_compatibility(sess)
        assert assessment.level == TimingCompatibilityLevel.COMPATIBLE
        assert not assessment.has_warning
        assert "No" in assessment.summary or "Only one" in assessment.summary

    def test_two_compatible_sources_no_warning(self) -> None:
        sess = EventAnalysisSession()
        _add(sess, _absolute_aware(), "A")
        _add(sess, _absolute_aware(), "B")
        assessment = assess_session_timing_compatibility(sess)
        assert assessment.level == TimingCompatibilityLevel.COMPATIBLE
        assert not assessment.has_warning

    def test_mixed_timing_references_warns(self) -> None:
        sess = EventAnalysisSession()
        _add(sess, _absolute_naive(), "A")
        _add(sess, _elapsed_unanchored_explicit(), "B")
        assessment = assess_session_timing_compatibility(sess)
        assert assessment.level == TimingCompatibilityLevel.REQUIRES_REVIEW
        assert assessment.has_warning

    def test_three_source_mixed_session(self) -> None:
        sess = EventAnalysisSession()
        _add(sess, _absolute_naive(), "Relay")
        _add(sess, _elapsed_unanchored_explicit(), "SCADA")
        _add(sess, _sample_index(), "Legacy")
        assessment = assess_session_timing_compatibility(sess)
        assert len(assessment.source_profiles) == 3
        assert len(assessment.pair_results) == 3  # C(3,2)
        assert assessment.level == TimingCompatibilityLevel.REQUIRES_REVIEW

    def test_inactive_source_ignored(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        sess.set_source_active(sid_b, False)
        assessment = assess_session_timing_compatibility(sess)
        assert len(assessment.source_profiles) == 1
        assert assessment.level == TimingCompatibilityLevel.COMPATIBLE

    def test_source_removal_clears_warning(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        assert assess_session_timing_compatibility(sess).has_warning
        sess.remove_source(sid_b)
        assert not assess_session_timing_compatibility(sess).has_warning

    def test_reactivation_restores_assessment(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        sess.set_source_active(sid_b, False)
        assert not assess_session_timing_compatibility(sess).has_warning
        sess.set_source_active(sid_b, True)
        assert assess_session_timing_compatibility(sess).has_warning

    def test_offsets_included_in_source_profiles(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        sess.set_time_offset(sid_b, -0.42, method="manual")
        assessment = assess_session_timing_compatibility(sess)
        by_name = {p.display_name: p for p in assessment.source_profiles}
        assert by_name["B"].time_offset_s == pytest.approx(-0.42)
        assert by_name["A"].time_offset_s == pytest.approx(0.0)

    def test_manual_alignment_acknowledged_without_claiming_correctness(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        assessment_before = assess_session_timing_compatibility(sess)
        assert not assessment_before.manual_alignment_present

        sess.set_time_offset(sid_b, 1.5, method="manual")
        assessment_after = assess_session_timing_compatibility(sess)
        assert assessment_after.manual_alignment_present
        assert "manual session alignment" in assessment_after.summary
        assert "correct" not in assessment_after.summary.lower()
        # the raw level is unaffected by manual alignment -- it's a
        # separate fact, not a fix (Step 7's explicit distinction)
        assert assessment_after.level == assessment_before.level

    def test_no_mutation_of_session_state(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        sess.set_time_offset(sid_b, 3.3, method="manual")

        offsets_before = {s.source_id: s.time_offset_s for s in sess.list_sources()}
        methods_before = {s.source_id: s.alignment_method for s in sess.list_sources()}
        active_before = {s.source_id: s.is_active for s in sess.list_sources()}
        panels_before = [p.panel_id for p in sess.list_panels()]
        timing_refs_before = {
            s.source_id: s.record.timing_info.timing_reference for s in sess.list_sources()
        }

        assess_session_timing_compatibility(sess)
        assess_session_timing_compatibility(sess)  # called twice for good measure

        offsets_after = {s.source_id: s.time_offset_s for s in sess.list_sources()}
        methods_after = {s.source_id: s.alignment_method for s in sess.list_sources()}
        active_after = {s.source_id: s.is_active for s in sess.list_sources()}
        panels_after = [p.panel_id for p in sess.list_panels()]
        timing_refs_after = {
            s.source_id: s.record.timing_info.timing_reference for s in sess.list_sources()
        }

        assert offsets_before == offsets_after
        assert methods_before == methods_after
        assert active_before == active_after
        assert panels_before == panels_after
        assert timing_refs_before == timing_refs_after

    def test_no_waveform_data_mutation(self) -> None:
        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        before = sess.get_source(sid_a).record.waveform_data.copy(deep=True)

        assess_session_timing_compatibility(sess)

        after = sess.get_source(sid_a).record.waveform_data
        pd.testing.assert_frame_equal(before, after)

    def test_no_calculated_signal_invalidation(self) -> None:
        from app.calculated_signals.models import CalculatedSignalDefinition, ChannelRef
        from app.calculated_signals.resolver import CalculatedSignalResolutionService

        sess = EventAnalysisSession()
        sid_a = _add(sess, _absolute_naive(), "A")
        sid_b = _add(sess, _elapsed_unanchored_explicit(), "B")
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="Net", expression="a + a",
            variable_bindings={"a": ChannelRef(sid_a, "A")}, reference_variable="a",
        )
        sess.add_calculated_signal(defn)
        CalculatedSignalResolutionService(sess).resolve_one("c1")
        status_before = sess.get_calculated_signal_result("c1").status

        assess_session_timing_compatibility(sess)

        status_after = sess.get_calculated_signal_result("c1").status
        assert status_before == status_after
