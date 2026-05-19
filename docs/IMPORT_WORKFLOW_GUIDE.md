# Import Wizard Workflow Guide

Phase: 8.55O - Import Wizard Final UX & Workflow Hardening

Date: 2026-05-19

## Purpose

The Import Wizard is an operational engineering workflow for turning CSV/Excel
files into normalized datasets and `DisturbanceRecord` objects. The UI remains a
thin orchestration layer: profiling, validation, normalization, export writing,
and waveform conversion stay in backend modules.

## Workflow States

The wizard uses a linear workflow:

1. Load File
2. Raw Preview
3. Timestamp
4. Column Mapping
5. Review
6. Import Running
7. Complete

Backward navigation is allowed when no import worker is running. Forward
navigation is enabled only when the current step has enough valid state.

## Action Enablement

Actions follow these rules:

| Action | Enabled When |
| --- | --- |
| Next | Current step has the required input |
| Back | A previous page exists and import is not running |
| Run Import | Review plan is executable |
| Open Waveform | Current import succeeded and settings have not changed |
| Save Normalized File | Current import succeeded and dataset is export-ready |
| Close | No import/export worker is running |

The workflow status label explains why a step is waiting or why re-import is
required.

## Override Visibility

Timestamp format overrides are shown as `User Override` on the timestamp page
when the manual format field is non-empty. Empty override fields return to
auto-detected behavior.

Column mapping overrides are marked directly in the mapping table:

- Output name: `(User Override)`
- Type: `(User Override)`
- Unit: `(User Override)`
- Confidence column: `User Override` marker when the row has user changes or is excluded
- Tooltips explain which field was overridden

## Stale-State Invalidation

The wizard invalidates downstream state when upstream user decisions change.

Examples:

- Changing timestamp selection invalidates review/import/export state.
- Changing timestamp format override invalidates review/import/export state.
- Changing column mappings invalidates review/import/export state.
- Loading a new file clears the previous profile, import result, export result,
  diagnostics, and model state.

After a completed import, any mapping or timestamp change sets the workflow to
`Re-import required`. Export and Open Waveform remain disabled until the import
is rerun.

## Import And Export Flow

The Review step is the last validation checkpoint before execution. Import runs
through the plan-aware backend worker and uses the current GUI-authoritative
plan.

On completion:

- Diagnostics summarize import quality and repair behavior.
- Open Waveform continues to the visualization workflow.
- Save Normalized File writes the normalized dataset and optional metadata
  sidecar through the export backend.

Metadata sidecars are recommended because they preserve audit information:
source traceability, canonical names, and repair strategy.

## Discard Protection

The explicit Close action prompts before discarding:

- user timestamp overrides
- user mapping overrides
- completed imports that have not been exported
- changed settings after import
- active worker state

This is lightweight protection only. The wizard does not persist sessions or
maintain import history.

## Error And Empty States

Messages should explain impact and next action:

- No timestamp candidate: choose another file or provide a timestamp column.
- No data columns: include at least one data column.
- Invalid override: correct the format before import.
- Failed import: go back, adjust settings, and retry.
- Failed export: retry with a valid path/format/options.

Avoid exposing stack traces or backend internals in user-facing text.

## Performance Philosophy

Workflow UX logic must not re-read source files, parse full datasets, or run
heavy validation on the UI thread. The wizard keeps:

- sampled preview behavior
- background worker execution
- backend-owned validation
- diagnostics derived from existing result objects

No visualization architecture changes are part of this workflow hardening phase.

## Known Limits

- Discard protection is not a full session manager.
- Close-window teardown accepts immediately for deterministic runtime cleanup;
  the in-wizard Close action provides the user-facing discard prompt.
- Override markers are text-based rather than icon-based.
- Advanced import history and save-location persistence remain future work.
