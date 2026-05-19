# Import Developer Workflow

Phase: 8.55P - Acceptance Validation & Developer Operations

Date: 2026-05-19

## Purpose

This guide gives contributors a repeatable way to validate Import Wizard changes
without reverse-engineering the test suite. Use the repository virtual
environment for all commands.

## Standard Validation Slices

| Slice | Purpose | Command |
| --- | --- | --- |
| unit | Fast local validation for import planning, diagnostics, UI models, export UI | `.venv\Scripts\python.exe tools\run_import_acceptance.py --slice unit` |
| runtime | Qt worker, temp hygiene, export runtime, visualization handoff | `.venv\Scripts\python.exe tools\run_import_acceptance.py --slice runtime` |
| stress | Generated malformed and medium CSV coverage | `.venv\Scripts\python.exe tools\run_import_acceptance.py --slice stress` |
| acceptance | Small operational end-to-end workflows | `.venv\Scripts\python.exe tools\run_import_acceptance.py --slice acceptance` |
| import-full | Unit + runtime + stress + acceptance Import Wizard coverage | `.venv\Scripts\python.exe tools\run_import_acceptance.py --slice import-full` |

`tools\run_import_runtime_slice.py` is a convenience wrapper for repeated runtime
passes:

```powershell
.venv\Scripts\python.exe tools\run_import_runtime_slice.py --repeat 2
```

## Recommended Development Loop

1. Run the narrow unit test that covers the code you changed.
2. Run `--slice acceptance` for operational smoke coverage.
3. Run `--slice runtime` when touching Qt, workers, temp files, export, or
   visualization handoff.
4. Run `--slice stress` when touching profiling, timestamp handling, delimiter
   detection, column classification, or export after import.
5. Run `--slice import-full` before handing off a substantial Import Wizard phase.

## Benchmark Workflow

Use the benchmark tool for explicit performance checks. Do not run large stress
benchmarks as part of the default test loop.

```powershell
.venv\Scripts\python.exe tools\benchmark_import_pipeline.py --rows 1000
.venv\Scripts\python.exe tools\benchmark_import_pipeline.py --rows 25000 --json .powerwave_runtime_tmp\benchmarks\import_25000.json
```

The benchmark reports generation, profiling, import, export time, channel counts,
validation codes, file sizes, and `tracemalloc` memory. The memory value is Python
allocation telemetry, not full process RSS.

Observed local results from Phase 8.55M:

| Rows | Profile | Import | CSV Export | Peak traced memory |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.736 s | 0.946 s | 0.177 s | 1.94 MiB |
| 25,000 | 0.674 s | 5.767 s | 3.995 s | 18.40 MiB |

## Stress Sample Generation

Generate deterministic CSV files into runtime temp locations only:

```powershell
.venv\Scripts\python.exe tools\generate_import_stress_samples.py .powerwave_runtime_tmp\samples\stress.csv --rows 1000 --digital-text
```

Useful variants:

- `--delimiter ";"` for semicolon historian exports.
- `--metadata-rows 3` for header-noise files.
- `--malformed-ratio 0.02` for timestamp repair diagnostics.
- `--duplicate-ratio 0.02` for duplicate timestamp warnings.

## Troubleshooting

- If Qt tests hang, run the runtime slice only and check worker cleanup failures
  first: `tests\runtime\test_runtime_environment.py`.
- If export tests fail on Parquet or Feather, confirm whether optional
  dependencies are installed. CSV export is the baseline acceptance path.
- If timestamp override behavior regresses, run both timestamp override unit and
  runtime tests plus the acceptance slice.
- If stale temp directories appear in `git status`, do not delete unrelated
  user state. The active test root is `.powerwave_runtime_tmp`.

## Merge Guidance

Before merging Import Wizard changes:

- Run `--slice unit` for backend/UI model changes.
- Run `--slice runtime` for Qt worker, dialog, export, or waveform handoff changes.
- Run `--slice stress` for profiling, timestamp, delimiter, or data-quality changes.
- Run `--slice import-full` for phase-level work.
