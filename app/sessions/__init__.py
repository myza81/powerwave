"""Event Analysis Session package — Phase 9A.

Public surface (Qt-free):

Models:
    SessionSource        One loaded DisturbanceRecord + display metadata + alignment state.
    SessionChannel       One channel with user-editable display properties.
    PanelConfig          Layout configuration for one display panel.
    AlignedChannelData   Render-ready output from build_aligned_data().
    AlignmentResult      Structured result from auto-align operations.
    SourceQualityMetrics Per-source data quality summary.

Session:
    EventAnalysisSession Multi-source analyst workspace.

Engine:
    alignment_engine     Pure-function module; import directly for engine functions.
"""
from __future__ import annotations

from app.sessions.session_models import (
    ALIGNMENT_METHODS,
    AlignedChannelData,
    AlignmentResult,
    PanelConfig,
    SessionChannel,
    SessionSource,
    SourceQualityMetrics,
)
from app.sessions.event_session import EventAnalysisSession
from app.sessions import alignment_engine

__all__ = [
    # Models
    "AlignedChannelData",
    "AlignmentResult",
    "ALIGNMENT_METHODS",
    "PanelConfig",
    "SessionChannel",
    "SessionSource",
    "SourceQualityMetrics",
    # Session
    "EventAnalysisSession",
    # Engine module
    "alignment_engine",
]
