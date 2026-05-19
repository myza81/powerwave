# Import Wizard Hardening Report

Phase: 8.55M - Real-World Import Hardening & Large Dataset Stress Testing

Date: 2026-05-19

## Scope

This phase validates the existing Import Wizard pipeline under realistic CSV
conditions. It does not add new Import Wizard features or modify visualization
architecture.

Covered areas:

- Deterministic large CSV sample generation.
- Backend import pipeline repeatability.
- Malformed timestamp handling.
- Metadata/header noise.
- Delimiter variants: comma, semicolon, tab, pipe.
- Duplicate, missing, and non-monotonic timestamps.
- Mixed timestamp formats.
- Text digital/status columns.
- Unknown/text noise columns.
- Export after import.
- DisturbanceRecord and waveform handoff after import.
- Qt runtime worker responsiveness and cleanup behavior.

## Stress Sample Generator

`tools/generate_import_stress_samples.py` creates deterministic CSV files and
streams rows directly to disk. Generated files are intended for runtime temp
directories and are not committed.

Supported configuration:

- Row count: small 1,000, medium 100,000, large 1,000,000 through presets.
- Timestamp format and sampling interval.
- Delimiter.
- Metadata rows before the header.
- Malformed, duplicate, and missing timestamp ratios.
- Analog columns including voltage, current, MW, MVar, and frequency.
- Digital/status columns with numeric or text states.
- Unknown/text noise columns.

The normal test suite uses smaller runtime-generated files to keep execution
deterministic and fast. The tool supports larger files for explicit local or CI
stress runs.

## Benchmark Tool

`tools/benchmark_import_pipeline.py` runs a practical benchmark against the
backend pipeline. It reports:

- File generation time.
- Profiling latency.
- Total import latency.
- CSV export latency.
- Final row count and channel counts.
- Validation message codes.
- Source/export file sizes.
- Tracemalloc current and peak memory.

The tool uses repository runtime temp directories and cleans generated files
after the benchmark. Memory numbers are Python allocation measurements from
`tracemalloc`, not full process RSS.

## Measured Results

Measured locally in this workspace with `.venv\Scripts\python.exe` on
2026-05-19.

| Rows | Result | Profile | Import | CSV Export | Channels | Peak traced memory |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1,000 | success | 0.736 s | 0.946 s | 0.177 s | 7 analog, 2 digital | 1.94 MiB |
| 25,000 | success | 0.674 s | 5.767 s | 3.995 s | 7 analog, 2 digital | 18.40 MiB |

Performance observations:

- Small generated files are comfortably interactive.
- The 25,000-row path completes predictably and exports successfully.
- Export is a meaningful part of total runtime for medium samples.
- Larger 100,000 and 1,000,000 row runs should be treated as explicit stress
  runs, not default unit/runtime test work.

## Runtime Hardening Results

New runtime coverage verifies:

- Import worker can remain pending while the dialog stays responsive.
- Completed import enables Open Waveform.
- Successful import keeps Save Normalized File available.
- CSV export with metadata sidecar works after realistic import.
- Failed ragged CSV import returns a pipeline failure without crashing the
  dialog.
- Closing the dialog after worker completion drains cleanly.
- Closing while an import worker is pending does not crash when the worker
  later completes in the deterministic test harness.

## Malformed File Results

Recoverable inputs:

- Metadata rows before header are skipped.
- Semicolon, tab, and pipe-delimited files import successfully.
- Blank and malformed timestamp values are dropped with timestamp diagnostics.
- Duplicate timestamps warn without blocking import.
- Non-monotonic timestamps warn without blocking import.
- Mixed timestamp formats drop unmatched rows when a detected format is in
  force.
- Text digital statuses such as OPEN/CLOSED route to digital channels and are
  coerced to 0/1.
- Unknown columns are preserved as analog channels with warnings.

Unrecoverable inputs:

- Files with no detectable timestamp candidate fail gracefully with validation
  messages.
- Ragged rows that pandas cannot load fail through `PIPELINE_LOAD_FAILED`
  without crashing the wizard.

## Visualization Handoff

Stress tests verify the imported `DisturbanceRecord` remains compatible with
the existing visualization contract:

- `record.waveform_data["time"]` exists.
- The time axis starts from the imported timestamp origin.
- Analog and digital channel descriptors match waveform columns.
- Digital text statuses are routed through the existing digital channel path.

No visualization architecture changes were made.

## Export Verification

Export after import is covered for generated medium data:

- CSV normalized export succeeds.
- Metadata sidecar is created when enabled.
- Row counts match the normalized dataset.
- Runtime tests confirm export remains available after successful GUI import.

Parquet and Feather remain dependency-gated through the existing export writer.

## Known Limits

- Default tests use 25,000 rows for medium runtime coverage to avoid slow local
  and CI runs. The generator and benchmark support 100,000 and 1,000,000 row
  samples for explicit stress runs.
- Memory reporting is lightweight and based on `tracemalloc`; it does not
  represent total process RSS or native allocations inside pandas.
- Mixed timestamp formats are not automatically repaired into a unified format;
  unmatched rows are dropped when a specific detected or user format is used.
- Ragged CSV rows are reported as load failures rather than partially repaired.
- No Excel large-file benchmark was added in this phase; Excel runtime remains
  covered by existing small deterministic tests.

## Operational Guidance

- Use the generator for reproducible import samples instead of checking large
  files into the repository.
- Run `tools/benchmark_import_pipeline.py --rows 100000` before large customer
  file validation.
- Treat all import warnings as engineering diagnostics. Duplicate,
  non-monotonic, and dropped timestamp warnings are expected for imperfect
  historian exports.
- Use the manual timestamp override UI when mixed or regional timestamp formats
  are detected incorrectly.

## Recommended Next Phase

Phase 8.55N should focus on operational import diagnostics:

- A compact import diagnostics summary in the wizard.
- Optional user-visible row-drop summary for malformed timestamp rows.
- Documented large-file acceptance guidance for field users.
