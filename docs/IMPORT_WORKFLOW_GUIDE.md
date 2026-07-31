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
3. Time Axis
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

## Time Axis Selection

The Time Axis step is responsible for selecting or defining the waveform X-axis.

Supported Time Axis modes:

| Mode | Required input | Output axis |
| --- | --- | --- |
| Auto-detect | Candidate source column | App chooses absolute timestamp or elapsed time when confidence is high |
| Absolute timestamp | Parseable timestamp column and optional strptime override | Elapsed seconds from first valid timestamp |
| Elapsed time values | Numeric duration column and unit (`seconds`, `milliseconds`, `minutes`) | Source duration converted to seconds |
| Synthetic time from sample rate | Sample rate or sample interval | Generated elapsed seconds from row order |
| Sample index | Ordered rows and at least one data column | Sample number / sequence index, not time |

For absolute timestamp sources, the import backend derives
`waveform_data["time"]` as seconds elapsed from the first valid timestamp and
preserves the real `TimingInformation.start_time` when available.

For relative elapsed-time sources, the import backend treats the duration values
as the authoritative engineering time axis. Values are converted to seconds and
preserved in `waveform_data["time"]`. Any synthetic datetime anchor used for
compatibility must not be shown to the operator as a real recording timestamp.

Elapsed-time auto-detection must be conservative. Numeric columns should only be
auto-selected as elapsed time when the column name and monotonic sample pattern
strongly indicate a time axis.

For synthetic time from sample rate, the operator must provide exactly one
timing basis: sample rate or sample interval. The wizard SHALL validate that the
value is positive and finite. The generated X-axis is elapsed seconds, but the
Review and Diagnostics pages SHALL identify it as synthetic timing.

For Sample index mode, the wizard SHALL NOT require a timestamp candidate,
timestamp format, elapsed-time unit, sample rate, or sample interval. Guardrails
are limited to confirming that at least one signal/data column and at least one
row are available. The X-axis SHALL be labelled Sample Index and SHALL NOT be
used for duration, frequency, event timing, or cross-record synchronization.

Manual mode overrides SHALL be explicit and auditable:

- Switching an absolute timestamp candidate to elapsed-time mode is allowed only
  when the selected column is numeric and monotonic enough to be a duration
  axis; otherwise the wizard blocks and recommends Sample index or Synthetic
  time from sample rate.
- Switching an elapsed-time candidate to absolute timestamp mode is allowed only
  when the selected values parse as timestamps with the selected/detected
  format.
- Switching to Synthetic time from sample rate ignores timestamp column values
  and uses row order plus the operator-provided timing basis.
- Switching to Sample index ignores all timestamp/duration values and uses row
  order only.

## Override Visibility

Timestamp format overrides are shown as `User Override` on the Time Axis page
when the manual format field is non-empty. Empty override fields return to
auto-detected behavior.

Manual strptime format overrides apply only to Absolute timestamp mode. Relative
elapsed-time columns expose unit confirmation. Synthetic time from sample rate
exposes sample-rate/sample-interval input. Sample index exposes no timing
parameter input.

Column mapping overrides are marked directly in the mapping table:

- Output name: `(User Override)`
- Type: `(User Override)`
- Unit: `(User Override)`
- Confidence column: `User Override` marker when the row has user changes or is excluded
- Tooltips explain which field was overridden

Changing a column Type SHALL immediately apply the default Unit for that type
(`V`, `A`, `MW`, `Mvar`, `Hz`, or `Hz/s`). The Unit column remains editable so
operators can override the default when the source uses a different engineering
unit.

The selected time-axis column SHALL NOT be presented as an editable signal
mapping in the Column Mapping table. The page should show the chosen time axis
as contextual read-only information and list only waveform/data columns for
include, output name, type, and unit review. Backend planning still treats the
selected timestamp/duration source as excluded from output channels.

## Stale-State Invalidation

The wizard invalidates downstream state when upstream user decisions change.

Examples:

- Changing time-axis selection invalidates review/import/export state.
- Changing time-axis mode, timestamp format override, elapsed-time unit, sample
  rate, sample interval, or Sample index selection invalidates
  review/import/export state.
- Changing column mappings invalidates review/import/export state.
- Loading a new file clears the previous profile, import result, export result,
  diagnostics, and model state.

After a completed import, any mapping or time-axis change sets the workflow to
`Re-import required`. Export and Open Waveform remain disabled until the import
is rerun.

## Import And Export Flow

The Review step is the last validation checkpoint before execution. Import runs
through the plan-aware backend worker and uses the current GUI-authoritative
plan.

Entering the Review step SHALL rebuild the execution plan from the current
timestamp and column-mapping state, including when the operator navigates by the
left step list. When the plan is not executable, the Review page and workflow
status SHALL show the blocking reason instead of only disabling Run Import.

On completion:

- Diagnostics summarize import quality and repair behavior.
- Open Waveform continues to the visualization workflow.
- Save Normalized File writes the normalized dataset and optional metadata
  sidecar through the export backend.

Metadata sidecars are recommended because they preserve audit information:
source traceability, canonical names, and repair strategy.

## Discard Protection

The explicit Close action prompts before discarding:

- user time-axis overrides
- user mapping overrides
- completed imports that have not been exported
- changed settings after import
- active worker state

This is lightweight protection only. The wizard does not persist sessions or
maintain import history.

## Error And Empty States

Messages should explain impact and next action:

- No time-axis candidate: choose another file, provide a timestamp/duration
  column, select Synthetic time from sample rate, or select Sample index.
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
