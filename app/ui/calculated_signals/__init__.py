"""Calculated Signals UI (Phase 3A) — creation and preview only.

Contains the dialog that lets an analyst assemble a CalculatedSignalDefinition
from existing analog channels and create it in the active
EventAnalysisSession. This package holds no mathematical, unit, alignment,
or interpolation logic of its own -- it only builds definitions and calls
the existing backend (app.calculated_signals, app.sessions.event_session)
for validation, preview, and creation.

Session Canvas rendering of calculated signals is out of scope for this
phase; see app.ui.calculated_signals.calculated_signal_dialog's module
docstring for the exact boundary.

Public API:
  CalculatedSignalDialog       -- the creation/preview dialog
  AnalogInputSelectorDialog    -- analog-only, source-grouped channel picker
  SelectedAnalogChannel        -- one channel chosen from the selector
"""
from app.ui.calculated_signals.analog_input_selector import (
    AnalogInputSelectorDialog,
    SelectedAnalogChannel,
)
from app.ui.calculated_signals.calculated_signal_dialog import CalculatedSignalDialog

__all__ = [
    "CalculatedSignalDialog",
    "AnalogInputSelectorDialog",
    "SelectedAnalogChannel",
]
