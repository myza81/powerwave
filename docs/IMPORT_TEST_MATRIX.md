# Import Test Matrix

Phase: 8.55P - Acceptance Validation & Developer Operations

Date: 2026-05-19

## Purpose

This matrix maps Import Wizard feature areas to the tests and operational slices
that protect them.

## Coverage Matrix

| Feature Area | Unit Coverage | Runtime Coverage | Stress Coverage | Acceptance Coverage |
| --- | --- | --- | --- | --- |
| File profiling | `test_import_wizard_file_profiling.py` | realistic workflow tests | malformed and delimiter stress tests | CSV/XLSX operational flows |
| Timestamp detection | `test_import_wizard_timestamp_detection.py` | timestamp override execution | malformed/mixed timestamp stress tests | authoritative override test |
| Timestamp override | `test_timestamp_override_ui.py` | `test_timestamp_override_execution.py` | mixed timestamp stress tests | override-to-record timing test |
| Plan-aware execution | `test_plan_aware_pipeline.py` | authoritative flow tests | generated import tests | CSV import/open waveform |
| Column mapping | `test_import_wizard_gui.py`, `test_import_workflow_ux.py` | workflow runtime tests | unknown/digital column stress tests | export and repeated workflow tests |
| Normalized dataset assembly | `test_import_pipeline.py` | realistic workflow tests | large CSV stress tests | CSV/XLSX export paths |
| DisturbanceRecord bridge | `test_disturbance_record_bridge.py` | visualization handoff runtime tests | visualization handoff stress tests | Open Waveform signal test |
| Export writer | `test_export_writer.py`, `test_export_planning.py` | export UI runtime tests | export after medium import | XLSX import to normalized CSV |
| Export UI | `test_export_ui.py` | `test_export_ui_runtime.py` | export stress tests | repeated import/export test |
| Diagnostics panel | `test_import_diagnostics.py` | diagnostics runtime tests | malformed file diagnostics | malformed diagnostics visibility |
| Workflow invalidation | `test_import_workflow_ux.py` | workflow runtime tests | n/a | repeated workflow stability |
| Runtime hygiene | n/a | `test_runtime_environment.py` | realistic workflow tests | close during worker test |

## Standard Slice Contents

### unit

Fast deterministic Import Wizard tests covering backend planning, diagnostics,
GUI models, timestamp override UI behavior, export writer, and export UI models.

### runtime

Qt/runtime validation for Import Wizard workers, temp hygiene, export UI,
diagnostics, timestamp override execution, and realistic workflow stability.

### stress

Generated CSV tests for large-ish deterministic files, malformed timestamps,
delimiter variants, metadata/header noise, duplicate timestamps, digital text
states, unknown columns, export after import, and visualization handoff.

### acceptance

Small operational workflows that combine import, diagnostics, export, timestamp
override authority, waveform handoff, repeated cycles, and dialog-close safety.

### import-full

The combined Import Wizard regression surface: unit, runtime, stress, and
acceptance slices.

## Regression Risks And Guards

| Risk | Primary Guard |
| --- | --- |
| UI-thread blocking during import/export | runtime and acceptance worker tests |
| Stale import/export state after setting edits | workflow UX/runtime tests |
| Timestamp override silently falling back to detected format | timestamp override runtime and acceptance tests |
| Metadata sidecar not written | export UI/runtime and acceptance tests |
| Hidden row drops or malformed timestamp loss | diagnostics and stress tests |
| Worker completion after dialog close crash | runtime and acceptance close tests |
| Visualization handoff missing time axis | bridge/runtime/stress/acceptance handoff tests |

## Adding Future Tests

Add tests to the narrowest layer that catches the regression:

- Pure planning or validation behavior belongs in `tests/unit/`.
- Qt worker, dialog, or cleanup behavior belongs in `tests/runtime/`.
- Generated malformed or larger data belongs in `tests/stress/`.
- Cross-step operational workflows belong in `tests/acceptance/`.
