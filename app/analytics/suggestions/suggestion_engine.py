"""Contextual analytics suggestion engine — pure computation, no Qt.

Inspects the results produced by Phases 2-7 and emits a prioritised list
of actionable suggestions to surface in the SuggestionBar.

Usage:
    suggestions = generate_suggestions(context)
    for s in suggestions:
        bar.add_suggestion(s)

Each Suggestion carries an ``action_id`` that MainWindow maps to a callable.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.analytics.events.event_detector import DetectedEvent
    from app.analytics.quality.quality_fingerprint import RecordQuality
    from app.analytics.fault.fault_classifier import FaultCharacterisation
    from app.analytics.protection.timing_extractor import ProtectionTimingResult
    from app.analytics.correlation.cross_correlator import CrossCorrelationResult


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Suggestion:
    """A single contextual suggestion chip."""

    id: str           # unique stable key (used for deduplication / dismissal)
    text: str         # short human-readable sentence shown on the chip
    action_id: str    # maps to a callable in MainWindow._suggestion_action_map
    action_label: str # label on the [Act] button, e.g. "Enable", "Open", "Align"
    priority: int     # lower = more important (shown left-to-right)
    icon: str = "💡"  # single emoji prefix on the chip


@dataclasses.dataclass
class SuggestionContext:
    """All analytics results needed to derive suggestions."""

    events: list = dataclasses.field(default_factory=list)
    quality: object = None                   # RecordQuality | None
    fault: object = None                     # FaultCharacterisation | None
    protection: object = None               # ProtectionTimingResult | None
    correlation_results: list = dataclasses.field(default_factory=list)
    is_multi_source: bool = False

    # Current analytics state — suppress suggestions for already-active modes
    rms_active: bool = False
    phasors_active: bool = False
    harmonics_active: bool = False

    # Dock visibility — suppress "open panel" suggestions if already visible
    quality_dock_visible: bool = False
    fault_dock_visible: bool = False
    protection_dock_visible: bool = False
    correlation_dock_visible: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestions(ctx: SuggestionContext) -> list[Suggestion]:
    """Return a prioritised list of contextual suggestions.

    Suggestions are ordered by ascending priority (0 = most important).
    Duplicates (same ``id``) are removed.
    """
    seen: set[str] = set()
    out: list[Suggestion] = []

    def _add(s: Suggestion) -> None:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)

    # ── Event-driven suggestions ──────────────────────────────────────────────
    event_types: set[str] = {
        getattr(e, "event_type", "") for e in ctx.events
    }

    has_voltage_dip    = any("dip" in t or "sag" in t for t in event_types)
    has_voltage_swell  = any("swell" in t for t in event_types)
    has_overcurrent    = any("current" in t or "overcurrent" in t for t in event_types)
    has_freq_deviation = any("freq" in t for t in event_types)

    if (has_voltage_dip or has_overcurrent) and not ctx.rms_active:
        _add(Suggestion(
            id="voltage_dip_rms",
            text="Fault/dip detected — view RMS envelope?",
            action_id="enable_rms",
            action_label="Enable",
            priority=10,
            icon="📉",
        ))

    if has_voltage_swell and not ctx.rms_active:
        _add(Suggestion(
            id="voltage_swell_rms",
            text="Voltage swell detected — view RMS envelope?",
            action_id="enable_rms",
            action_label="Enable",
            priority=11,
            icon="📈",
        ))

    if has_freq_deviation:
        _add(Suggestion(
            id="freq_deviation_frequency",
            text="Frequency deviation detected — enable frequency trend?",
            action_id="enable_frequency",
            action_label="Enable",
            priority=20,
            icon="〰️",
        ))

    # ── Fault characterisation suggestions ────────────────────────────────────
    if ctx.fault is not None:
        fault_type = getattr(getattr(ctx.fault, "fault_type", None), "value", "Unknown")
        if not ctx.phasors_active:
            _add(Suggestion(
                id="fault_phasors",
                text=f"{fault_type} fault — view symmetrical components?",
                action_id="enable_phasors",
                action_label="Open",
                priority=5,
                icon="⚡",
            ))
        if not ctx.fault_dock_visible:
            _add(Suggestion(
                id="show_fault_panel",
                text=f"Fault characterised: {fault_type} — open fault summary?",
                action_id="show_fault",
                action_label="Open",
                priority=6,
                icon="⚡",
            ))

    # ── 3-phase phasor suggestion (without a fault) ───────────────────────────
    elif ctx.events and not ctx.phasors_active:
        # Suggest phasors whenever we have multi-channel voltage events
        voltage_events = [
            e for e in ctx.events
            if "voltage" in getattr(e, "event_type", "")
        ]
        if len(voltage_events) >= 1:
            _add(Suggestion(
                id="threephase_phasors",
                text="3-phase voltages present — view phasor diagram?",
                action_id="enable_phasors",
                action_label="Open",
                priority=30,
                icon="🔄",
            ))

    # ── Protection timing suggestion ──────────────────────────────────────────
    if ctx.protection is not None and not ctx.protection_dock_visible:
        clearing_ms = getattr(ctx.protection, "clearing_time_ms", None)
        if clearing_ms is not None:
            _add(Suggestion(
                id="show_protection_panel",
                text=f"Clearing time {clearing_ms:.1f} ms detected — view protection timing?",
                action_id="show_protection",
                action_label="Open",
                priority=7,
                icon="🛡️",
            ))
        else:
            _add(Suggestion(
                id="show_protection_panel",
                text="Protection events detected — view relay timing?",
                action_id="show_protection",
                action_label="Open",
                priority=8,
                icon="🛡️",
            ))

    # ── Data quality suggestions ──────────────────────────────────────────────
    if ctx.quality is not None:
        from app.analytics.quality import QualityGrade
        grade = getattr(ctx.quality, "overall_grade", QualityGrade.OK)
        if grade == QualityGrade.ERROR and not ctx.quality_dock_visible:
            _add(Suggestion(
                id="show_quality_error",
                text="Data quality errors found — review quality report?",
                action_id="show_quality",
                action_label="Open",
                priority=3,
                icon="🔴",
            ))
        elif grade == QualityGrade.WARN and not ctx.quality_dock_visible:
            _add(Suggestion(
                id="show_quality_warn",
                text="Data quality warnings — review quality report?",
                action_id="show_quality",
                action_label="Open",
                priority=15,
                icon="🟡",
            ))

    # ── Multi-source correlation suggestions ──────────────────────────────────
    if ctx.is_multi_source and ctx.correlation_results:
        same_event_pairs = [
            r for r in ctx.correlation_results if getattr(r, "same_event_likely", False)
        ]
        if same_event_pairs and not ctx.correlation_dock_visible:
            _add(Suggestion(
                id="show_correlation_panel",
                text=f"{len(same_event_pairs)} source pair(s) likely same event — view correlation?",
                action_id="show_correlation",
                action_label="Open",
                priority=12,
                icon="🔗",
            ))
        # Auto-align suggestion only if unaligned (correlation offset not applied)
        unaligned = [
            r for r in same_event_pairs
            if abs(getattr(r, "lag_ms", 0.0)) > 1.0
        ]
        if unaligned:
            _add(Suggestion(
                id="auto_align_sources",
                text=f"{len(unaligned)} source(s) have >1 ms lag — auto-align?",
                action_id="auto_align",
                action_label="Align",
                priority=2,
                icon="📌",
            ))

    # ── Harmonic suggestion ───────────────────────────────────────────────────
    # Only add if we have disturbance events but no harmonic mode active
    if ctx.events and not ctx.harmonics_active and not ctx.fault:
        _add(Suggestion(
            id="explore_harmonics",
            text="Disturbance recorded — explore harmonic spectrum?",
            action_id="enable_harmonics",
            action_label="Open",
            priority=40,
            icon="〰",
        ))

    out.sort(key=lambda s: s.priority)
    return out
