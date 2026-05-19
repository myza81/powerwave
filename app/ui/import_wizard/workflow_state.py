"""Lightweight workflow-state evaluation for the Import Wizard UI.

This module is intentionally UI-facing and small. It centralizes action
enablement decisions so the dialog can keep button state and user guidance in
sync without duplicating backend validation logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.import_wizard import WizardStep


@dataclass(frozen=True, slots=True)
class WorkflowActionState:
    """Current Import Wizard action availability and guidance."""

    can_go_back: bool
    can_go_next: bool
    can_import: bool
    can_open_waveform: bool
    can_export: bool
    can_close: bool
    status_text: str


def evaluate_workflow_actions(
    *,
    step: WizardStep,
    step_index: int,
    has_path: bool,
    has_profile: bool,
    profile_has_errors: bool,
    has_timestamp_candidates: bool,
    has_included_columns: bool,
    plan_is_executable: bool,
    import_running: bool,
    export_running: bool,
    import_success: bool,
    export_ready: bool,
    settings_dirty_since_import: bool,
) -> WorkflowActionState:
    """Return action enablement for the current wizard state."""
    busy = import_running or export_running
    can_go_back = step_index > 0 and not import_running
    can_close = not busy
    can_open = import_success and not export_running and not settings_dirty_since_import
    can_export = export_ready and not export_running and not settings_dirty_since_import

    if busy:
        return WorkflowActionState(
            can_go_back=can_go_back,
            can_go_next=False,
            can_import=False,
            can_open_waveform=can_open,
            can_export=can_export,
            can_close=can_close,
            status_text="Worker is running. Actions will re-enable when it completes.",
        )

    if settings_dirty_since_import:
        dirty_text = "Import settings changed. Re-import required before export or waveform handoff."
    else:
        dirty_text = ""

    can_next = False
    can_import = False
    status = dirty_text

    if step == WizardStep.LOAD_FILE:
        can_next = has_path
        status = status or ("Choose a CSV or Excel file." if not has_path else "Ready to profile selected file.")
    elif step == WizardStep.RAW_PREVIEW:
        can_next = has_profile and not profile_has_errors
        status = status or ("Preview ready. Continue to timestamp selection." if can_next else "Profile a readable file before continuing.")
    elif step == WizardStep.TIMESTAMP_SELECT:
        can_next = has_timestamp_candidates
        status = status or ("Timestamp selection ready." if can_next else "No timestamp candidate is available; choose another file or adjust the source.")
    elif step == WizardStep.COLUMN_MAPPING:
        can_next = has_included_columns
        status = status or ("Mapping ready. Review the execution plan." if can_next else "Include at least one data column before import.")
    elif step == WizardStep.NORMALIZATION_REVIEW:
        can_import = plan_is_executable
        status = status or ("Plan is ready to import." if can_import else "Review plan issues before running import.")
    elif step == WizardStep.SAVE_NORMALIZED:
        status = status or "Import is running." if import_running else "Import execution state."
    elif step == WizardStep.RENDER_WAVEFORM:
        if import_success:
            status = status or "Import complete. Save normalized data or open the waveform."
        else:
            status = status or "Import did not complete successfully. Go back, adjust settings, and retry."

    return WorkflowActionState(
        can_go_back=can_go_back,
        can_go_next=can_next,
        can_import=can_import,
        can_open_waveform=can_open,
        can_export=can_export,
        can_close=can_close,
        status_text=status,
    )
