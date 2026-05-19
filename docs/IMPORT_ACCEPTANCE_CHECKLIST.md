# Import Acceptance Checklist

Phase: 8.55P - Acceptance Validation & Developer Operations

Date: 2026-05-19

## Purpose

This checklist defines the practical acceptance scenarios for the Import Wizard
subsystem. It is intended for release-readiness checks, contributor handoff, and
repeatable local validation before Phase 9 work.

## Standard Acceptance Scenarios

| Scenario | Expected Result | Validation Slice |
| --- | --- | --- |
| CSV import | Import succeeds and Open Waveform emits a `DisturbanceRecord` | acceptance, runtime |
| XLSX import | Import succeeds and normalized CSV export can be written | acceptance |
| Malformed timestamps | Import completes when recoverable, with clear diagnostics for dropped rows | stress, runtime |
| Invalid timestamp override | Execution is blocked with validation messages; wizard stays usable | unit, runtime |
| Timestamp override | User format is authoritative and detected format is ignored | unit, runtime, acceptance |
| Export normalized CSV | CSV is written and metadata sidecar is created when enabled | runtime, acceptance |
| Duplicate timestamps | Import warns without crashing or silently hiding the issue | stress |
| Unknown columns | Columns are preserved or warned according to mapping diagnostics | stress |
| Large generated CSV | UI/runtime path remains responsive; explicit benchmark records measured results | stress, benchmark |
| Repeated import/export cycles | Runtime remains stable with no stale export state | runtime, acceptance |
| Dialog close around workers | Worker completion after close does not crash or leak Qt state | runtime, acceptance |

## Acceptance Rules

- All waveform data accepted by the UI must flow through `DisturbanceRecord`.
- Import and export must run through existing worker-backed paths.
- Runtime tests must use repository temp hygiene; generated files stay in runtime
  temp directories.
- Warnings are acceptable when they describe recoverable data quality issues.
- Complete timestamp parse failure, unsupported source files, and invalid export
  paths must fail gracefully without crashing the dialog.

## Recommended Acceptance Command

```powershell
.venv\Scripts\python.exe tools\run_import_acceptance.py --slice acceptance
```

Run `--slice import-full` before major Import Wizard merges.

## Manual Spot Check

1. Open the Import Wizard from the main window.
2. Import a small historian-style CSV with timestamp, analog, and digital columns.
3. Confirm diagnostics show timestamp strategy, channel counts, and validation messages.
4. Open the waveform and verify the relative time axis starts at zero.
5. Save a normalized CSV and confirm the metadata sidecar exists.
6. Change a timestamp override or mapping and confirm re-import is required.

## Known Acceptance Limits

- Default acceptance tests use small deterministic files; large-file validation is
  explicit through stress and benchmark tools.
- Parquet and Feather acceptance depends on optional dependencies and is covered
  by backend/export tests when available.
- This checklist does not replace engineering review for new ingestion formats or
  visualization architecture changes.
