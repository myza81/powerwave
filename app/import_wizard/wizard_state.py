"""Wizard step definitions and state-transition rules.

Transition policy
-----------------
- Backward: always allowed to any earlier step (user may re-edit any prior choice).
- Forward (strict): only one step at a time — prevents skipping required setup steps.
- Forward (skip): allowed when allow_skip=True — the GUI may pass this when it has
  already verified that the skipped step's preconditions are satisfied (e.g.
  timestamps are already clean, so TIMESTAMP_REPAIR may be skipped).
"""
from __future__ import annotations

import enum


class WizardStep(enum.Enum):
    LOAD_FILE            = "load_file"
    RAW_PREVIEW          = "raw_preview"
    TIMESTAMP_SELECT     = "timestamp_select"
    TIMESTAMP_REPAIR     = "timestamp_repair"
    COLUMN_MAPPING       = "column_mapping"
    NORMALIZATION_REVIEW = "normalization_review"
    SAVE_NORMALIZED      = "save_normalized"
    RENDER_WAVEFORM      = "render_waveform"


_STEP_ORDER: list[WizardStep] = [
    WizardStep.LOAD_FILE,
    WizardStep.RAW_PREVIEW,
    WizardStep.TIMESTAMP_SELECT,
    WizardStep.TIMESTAMP_REPAIR,
    WizardStep.COLUMN_MAPPING,
    WizardStep.NORMALIZATION_REVIEW,
    WizardStep.SAVE_NORMALIZED,
    WizardStep.RENDER_WAVEFORM,
]

_STEP_INDEX: dict[WizardStep, int] = {s: i for i, s in enumerate(_STEP_ORDER)}


def can_transition(
    from_step: WizardStep,
    to_step: WizardStep,
    *,
    allow_skip: bool = False,
) -> bool:
    """Return True when the transition from_step → to_step is permitted.

    Args:
        from_step:  Current wizard step.
        to_step:    Requested target step.
        allow_skip: When True, forward jumps of more than one step are permitted.
                    The caller is responsible for verifying preconditions before
                    passing allow_skip=True.
    """
    from_idx = _STEP_INDEX[from_step]
    to_idx   = _STEP_INDEX[to_step]

    if to_idx <= from_idx:
        return True  # any backward move is always allowed

    if allow_skip:
        return True  # caller has verified preconditions

    return to_idx == from_idx + 1  # strict forward: one step at a time


def next_step(step: WizardStep) -> WizardStep | None:
    """Return the immediately following step, or None at the last step."""
    idx = _STEP_INDEX[step]
    if idx + 1 < len(_STEP_ORDER):
        return _STEP_ORDER[idx + 1]
    return None


def steps_before(step: WizardStep) -> list[WizardStep]:
    """Return all steps that precede *step* in order."""
    return _STEP_ORDER[: _STEP_INDEX[step]]


def step_index(step: WizardStep) -> int:
    """Return the zero-based ordinal position of *step*."""
    return _STEP_INDEX[step]
