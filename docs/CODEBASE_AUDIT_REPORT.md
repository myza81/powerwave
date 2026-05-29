# Powerwave Codebase Audit Report

Date: 2026-05-29
Scope: read-only architecture, UI/UX, visualization, data handling, cleanup, performance, and test coverage audit.

## Latest Re-Audit Update After User Fixes

Date: 2026-05-29

The latest follow-up audit shows the validation baseline is now green. The earlier collection blocker and later 52-failure baseline have both been resolved. Packaging has been moved toward the `app/` package, `src/` is explicitly marked legacy, the active COMTRADE provider no longer reads binary DAT files through `Path.read_bytes()`, BINARY32 fails earlier with a clearer provider error, `VisualizationManager` is marked deprecated, cursor extraction recognizes the session canvas hover cursor, and `SessionChannel.line_width` now has a backward-compatible default.

Current validation:

- `.venv\Scripts\python.exe -m pytest --collect-only -q` -> 4023 tests collected in 5.91 seconds.
- `.venv\Scripts\python.exe -m pytest -q` -> 4004 passed, 19 skipped, 11 warnings in 118.49 seconds.

### Re-Audit Finding Changes

| Prior finding | Current status | Notes |
| --- | --- | --- |
| C1: pytest could not collect | Closed | Collection now succeeds and the full suite passes. |
| C2: packaging points at `src` | Mostly fixed | `pyproject.toml` now uses `where = ["."]` and `include = ["app*"]`. `src/LEGACY.md` documents `src/` as non-canonical. |
| H1: active/legacy architecture split | Improved, still open | `src/` is labeled legacy, but legacy modules and tests remain in the repository and `pythonpath` still includes `src`. |
| H3: COMTRADE binary full-file read | Improved | Active `app/providers/comtrade/comtrade_provider.py` now uses file size plus `np.fromfile` for BINARY. Legacy `src/parsers/comtrade_parser.py` still uses `read_bytes()`. |
| H3: BINARY32 accepted then rejected late | Improved, still unsupported | Active provider now rejects BINARY32 before building the record. Capability reporting and user-facing workflow should still say it is unsupported. |
| H5: signal-browser tests against deleted widget | Closed for test baseline | The full suite now passes with deleted widget files still absent. Continue treating those files as cleanup/history candidates. |
| H6: visualization ownership unclear | Improved, still open | `VisualizationManager` docstring now says deprecated and retained for legacy tests, but docs/tests still reference it heavily. |
| M4: cursor synchronization fragmentation | Improved | `SynchronizationManager._extract_cursor` now supports `_hover_cursor`, reducing the session canvas sync gap. Cursor A/B ownership is still separate. |
| RC1: 52 failing tests | Closed | The suite now passes: 4004 passed, 19 skipped. |
| RC2: `SessionChannel.line_width` API drift | Closed | `SessionChannel.line_width` now defaults to `1.0`. |

### Remaining Findings After Green Baseline

| ID | Severity | Type | Affected files | Evidence | Recommended fix | Phase |
| --- | --- | --- | --- | --- | --- | --- |
| RC3 | High | Architecture / Cleanup | `app/visualization/managers/visualization_manager.py`, `tests/unit/test_visualization_manager.py`, `tests/unit/test_visualization_grouped_display.py`, `tests/integration/test_pulu_manifest_pipeline.py`, docs | `VisualizationManager` is now marked deprecated, but many tests and docs still treat it as a core path. | Create a deprecation plan: keep tests only for compatibility behavior, move primary visualization tests to session canvas/controller, and update visualization docs. | Phase 1 |
| RC4 | Medium | Architecture / Test Hygiene | `pyproject.toml`, `src/`, `tests/test_*` | Packaging now includes `app*`, but pytest `pythonpath` still includes `"src"` and legacy tests still collect. | Keep `src` only if needed for legacy tests. Otherwise migrate/remove legacy tests and remove `src` from pytest `pythonpath`. | Phase 1 |

## 1. Executive Summary

Powerwave has a strong modern core: an `app.models.DisturbanceRecord` data contract, provider classes for COMTRADE/CSV/Excel, a dedicated Import Wizard pipeline, grouped waveform panels, session-level multi-source alignment, and a substantial test corpus. The intended architecture is well documented in `docs/ARCHITECTURE.md`, `docs/PROVIDER_PATTERN.md`, `docs/DATA_CONTRACT.md`, `docs/VISUALIZATION_CONTRACT.md`, and `docs/IMPORT_WORKFLOW_GUIDE.md`.

The main risk is not absence of architecture. It is architectural split-brain: the canonical `app/` stack coexists with a legacy `src/` stack, duplicated data models, duplicated parsers, overlapping visualization managers, and tests that still exercise paths that appear partially removed. Packaging now targets `app*`, but pytest still includes `src` on `pythonpath`, and legacy modules remain in the repository.

Validation is currently green. The suite collects 4023 tests and completes with 4004 passed, 19 skipped, and 11 warnings. The earlier stale main-window test failures and `SessionChannel.line_width` fixture/API drift have been resolved.

The CSV/Excel user workflow is mostly implemented through the Import Wizard, but direct provider paths still exist and can bypass preview, timestamp repair, labeling, normalization, and export. COMTRADE remains a direct provider path with no equivalent preview/confirmation workflow. Visualization has strong session-canvas machinery, but there are overlapping older grouped-display paths and fragmented cursor synchronization behavior.

## 2. Current System Strengths

- The canonical `app.models.DisturbanceRecord` is a clean central container for analog data, digital events, metadata, trigger time, sample rate, channel metadata, and source path.
- Provider classes isolate COMTRADE, CSV, and Excel loading behind `ProviderBase` and `ProviderManager`.
- The Import Wizard has a serious workflow model: file profiling, raw preview, timestamp selection, timestamp normalization, column mapping, review/import, export, and waveform handoff.
- `EventAnalysisSession` provides non-destructive multi-source offsets, source/channel metadata, panel routing, and render-ready aligned channel data.
- Grouped panels cover voltage, current, power, frequency, digital, and other channels, with `mw`, `mvar`, `frequency`, and `rocof` routing.
- The visualization layer uses PyQtGraph downsampling/clip-to-view in key widgets and has dedicated digital-event rendering.
- The repository contains extensive unit, runtime, stress, and acceptance tests, with prior documentation showing broad coverage.
- Documentation is unusually strong for a desktop scientific application and gives maintainers a clear target architecture.

## 3. Critical Findings

| ID | Severity | Type | Affected files | Evidence | Recommended fix | Phase |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Critical | None currently blocking | Test suite | Full suite is green: 4004 passed, 19 skipped. | Keep this baseline protected while addressing architectural cleanup. | Phase 0 |
| C2 | High | Architecture / Legacy Boundary | `pyproject.toml`, `src/`, `app/` | Packaging now includes `app*`, and `src/LEGACY.md` marks `src` as non-canonical. However, duplicate `DisturbanceRecord` classes remain and pytest `pythonpath` still includes `src`. | Finish the legacy boundary: migrate/remove remaining legacy tests, remove `src` from pytest path when possible, and maintain a deletion map. | Phase 1 |

## 4. High-Priority Findings

| ID | Severity | Type | Affected files | Evidence | Recommended fix | Phase |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | High | Architecture | `src/models/*`, `src/parsers/*`, `src/ui/*`, `src/engine/*`, `app/models/*`, `app/providers/*` | Legacy `src/` contains separate parsers, models, UI, and engine code. Current docs and UI point to `app/`, but legacy tests still exist. | Create a deprecation map: canonical, still-used, legacy, and delete-candidate modules. Do not remove until tests and packaging are aligned. | Phase 1 |
| H2 | High | Architecture / UX | `app/ui/main_window/main_window.py`, `app/providers/csv/csv_provider.py`, `app/providers/excel/excel_provider.py`, `app/import_wizard/*` | Main window routes CSV/Excel to `ImportWizardDialog` at `main_window.py:1627`, but `CsvProvider` and `ExcelProvider` still perform full direct loads with `pd.read_csv` and `pd.read_excel` at `csv_provider.py:202` and `excel_provider.py:252`. | Make Import Wizard the only interactive CSV/Excel path. Keep direct providers only for trusted, already-normalized programmatic loads, and name that contract explicitly. | Phase 1 |
| H3 | High | Performance / Bug | `app/providers/comtrade/comtrade_provider.py`, `src/parsers/comtrade_parser.py` | Active COMTRADE binary loading now uses `np.fromfile`, which improves the prior full `read_bytes()` risk. BINARY32 is rejected earlier. Legacy `src/parsers/comtrade_parser.py` still uses `read_bytes()`, and BINARY32 remains unsupported in the active provider. | Add large-file tests around 100 MB for the active provider, keep BINARY32 capability messaging explicit, and avoid legacy parser use in production paths. | Phase 2 |
| H4 | High | UX / Workflow | `app/ui/main_window/main_window.py`, `app/ui/import_wizard/*`, `app/providers/comtrade/*` | The required flow includes raw preview, timestamp repair, labeling, normalized save, and render. CSV/Excel mostly follow it; COMTRADE loads directly through provider/session flow at `main_window.py:1635-1638`. | Define separate workflows: COMTRADE quick-open with metadata review, and CSV/Excel normalization wizard. Surface the distinction in UI. | Phase 2 |
| H5 | High | Test Gap / Cleanup | `tests/unit/test_runtime_qt_widgets.py`, `app/ui/widgets/*` | `tests/unit/test_runtime_qt_widgets.py` was rewritten away from signal-browser expectations, while `app/ui/widgets/signal_browser.py` and `cursor_readout_bar.py` remain deleted in the worktree. | Confirm the replacement workflow is fully covered by session-canvas/waveform-navigator tests before deleting old widget references from docs and history. | Phase 0 |
| H6 | High | Architecture / Visualization | `app/visualization/managers/visualization_manager.py`, `app/ui/session/session_canvas_controller.py`, `app/visualization/widgets/session_canvas.py` | `VisualizationManager` remains heavily tested and documented, but the active session flow uses `EventAnalysisSession` and `SessionCanvasController`. | Choose the primary visualization orchestration path. Mark the other as compatibility, then migrate tests and docs accordingly. | Phase 1 |

## 5. Medium / Low-Priority Improvements

| ID | Severity | Type | Affected files | Evidence | Recommended fix | Phase |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Medium | Data Handling | `app/providers/csv/csv_provider.py`, `app/providers/excel/excel_provider.py` | Both providers duplicate `_infer_unit`, `_is_digital_column`, and related classification helpers. | Move shared column classification/unit inference into one tested module used by providers and Import Wizard. | Phase 2 |
| M2 | Medium | UX | `app/ui/main_window/main_window.py` | `_CSV_EXCEL_SUFFIXES` includes `.xls` at line 69, but `_FILE_FILTER` only advertises `.xlsx` for Excel at line 63. | Align file dialogs with supported suffixes and dependency availability. If `.xls` needs xlrd, explain that before load. | Phase 1 |
| M3 | Medium | Data Handling | `app/import_wizard/import_pipeline.py`, `app/providers/excel/excel_provider.py` | Excel sheet selection is automatic: the wizard loads a selected/default sheet and the provider picks a likely data-rich sheet. No explicit user sheet-selection screen was found. | Add an Excel sheet selection step when a workbook has multiple sheets with plausible data. | Phase 3 |
| M4 | Medium | Visualization | `app/visualization/managers/synchronization_manager.py`, `app/visualization/widgets/session_canvas.py` | Sync manager `_extract_cursor` looks for `_cursor`, while session canvas uses `_cursor_a` and `_cursor_b`; session controller separately synchronizes cursor A/B. | Consolidate the cursor synchronization contract so one manager owns range and cursor state for all canvases. | Phase 2 |
| M5 | Medium | Performance | `app/import_wizard/data_assembler.py`, `app/import_wizard/export_writer.py`, `app/visualization/managers/visualization_manager.py` | Multiple full DataFrame copies exist: `raw_dataframe.copy()`, normalized timestamp copies, selected-column copies, visualization filtered copies. | Audit copies by ownership. Retain copies only where mutation or isolation is required. | Phase 3 |
| M6 | Medium | Reliability | `app/ui/session/session_canvas_controller.py`, visualization widgets | Several broad `except Exception: pass` blocks protect UI painting but can hide render failures. | Route suppressed render errors into a throttled diagnostic logger/status pane. | Phase 2 |
| M7 | Low | Cleanup | `build.spec` | Empty build spec file exists. | Remove or populate when packaging strategy is settled. | Phase 4 |
| M8 | Low | Developer Experience | `pyproject.toml`, `.venv` | Project declares Python 3.11, but the observed venv error path used Python 3.14. | Standardize the development Python version and document it in setup instructions. | Phase 1 |

## 6. UI/UX Workflow Gap Analysis

Required sequence:

1. User opens CSV/Excel/COMTRADE file.
2. System previews raw data.
3. User selects timestamp column.
4. System suggests timestamp format.
5. User confirms or repairs timestamp.
6. System normalizes file.
7. System labels columns and units.
8. User corrects labels if needed.
9. System saves clean normalized file.
10. System renders waveform.

Current support:

- CSV/Excel through the main window mostly follows this flow via `ImportWizardDialog`.
- The wizard has file profiling, preview, timestamp selection, mapping, review/import, complete/export/open stages.
- Timestamp normalization and repair support exists, including format override and reconstruction paths.
- Waveform handoff is supported after import.

Workflow gaps:

- COMTRADE does not use the same review flow. It opens directly through the provider and session loader. This is acceptable for validated COMTRADE pairs, but the UX should make that explicit and still show metadata/channel review before plotting.
- Direct `CsvProvider` and `ExcelProvider` paths can still load data without the wizard's preview/repair/label/export sequence.
- Excel multi-sheet selection is not clearly exposed to the user before parsing.
- Timestamp repair is technical rather than guided. The user can override formats, but there is no dedicated mixed-format/duplicate/irregular timestamp repair screen.
- The file dialog advertises `.xlsx` but not `.xls`, while provider constants include `.xls`.
- Error messages are often raw exception strings from workers/providers. Industrial operators need actionable messages: failed sheet, failed timestamp column, unsupported DAT encoding, missing dependency, duplicate timestamps, or no plottable channels.

## 7. Visualization Audit

Strengths:

- Session rendering is routed through `EventAnalysisSession`, `SessionCanvasController`, and `SessionCanvasWidget`.
- Default panel routing covers voltage, current, power, frequency/ROCOF, digital, and other channels.
- `SessionCanvasWidget.update_curve` enables PyQtGraph downsampling and clip-to-view behavior.
- Digital event rendering has a dedicated timeline/track path.
- X-axis synchronization exists through `SynchronizationManager` and controller-level registration.

Risks:

- There are two visualization orchestration concepts: `VisualizationManager` and session-canvas management. Both are tested and documented, but their ownership boundary is unclear.
- Cursor synchronization is fragmented. `FlexiblePlotCanvas` has `_cursor`; session canvas has `_cursor_a` and `_cursor_b`; `SynchronizationManager` extracts only `_cursor`; `SessionCanvasController` synchronizes A/B separately.
- Some render exceptions are swallowed, so a waveform can fail to display while import succeeds without an obvious diagnostic.
- Digital visibility and repacking are more complex than analog curve visibility, increasing regression risk when channels are hidden/shown.
- Old grouped-display tests may pass against mocks without exercising real Qt/PyQtGraph behavior.

Cases where waveform may not display even though import succeeds:

- All selected channels classify as unsupported/non-plottable.
- Timestamp normalization succeeds but produces non-finite, duplicate-heavy, or non-monotonic time values not accepted by plotting paths.
- Provider returns a valid `DisturbanceRecord` with empty `waveform_data` or channel metadata mismatch.
- Session panel visibility or deleted signal browser/waveform navigator state hides channels.
- A broad render exception is swallowed during `_paint_panel` or digital repack.

## 8. Data Handling Audit

Strengths:

- `DisturbanceRecord` supports both waveform data and digital events.
- `EventAnalysisSession` applies offsets virtually and does not mutate source records.
- Multi-source global time range handles intersection and falls back to union when sources do not overlap.
- Raw waveform and lower-rate RMS/trend data can coexist on the same displayed time domain because each channel keeps its own sample positions.

Risks:

- CSV/Excel direct providers and Import Wizard each contain timestamp, unit, and classification logic. This invites drift.
- Direct providers perform best-effort timestamp parsing and can silently fall back in ways that are unclear to the user.
- Mixed-format, duplicate, missing, and irregular timestamps are better handled in the wizard than in direct providers.
- Multi-source alignment is currently display-oriented. It clips and decimates per channel but does not resample all sources onto a shared grid for cross-channel analytics.
- Unit inference depends heavily on names and units. Apparent-power units such as `VA` are valid, but missing or ambiguous units can still misroute channels.

## 9. Dead Code / Unused Code Candidates

These are candidates only. Do not delete without a dependency map and passing tests.

- `src/` tree: appears to contain legacy models, parsers, UI, and engine code parallel to the canonical `app/` architecture.
- `src/main.py`: old entrypoint parallel to `app/main.py`.
- `src/parsers/*`: legacy COMTRADE/CSV/Excel/PMU parser path parallel to providers.
- `src/models/disturbance_record.py`: duplicate data model.
- `app/visualization/managers/visualization_manager.py`: possibly older grouped display manager; still tested and documented, so classify before deletion.
- `app/data/display_alignment.py` and `app/data/multi_source_session.py`: may overlap with `app/sessions/event_session.py`.
- `app/providers/csv/csv_provider.py` and `app/providers/excel/excel_provider.py`: duplicate helper logic that should be centralized even if providers remain.
- Deleted-in-worktree widgets `app/ui/widgets/signal_browser.py` and `app/ui/widgets/cursor_readout_bar.py`: tests and agent docs still mention signal browser behavior.
- `build.spec`: empty packaging artifact.

## 10. Performance Risks

| Risk | Severity | Evidence | Impact | Recommendation |
| --- | --- | --- | --- | --- |
| COMTRADE large-file handling still needs proof | High | Active provider now uses `np.fromfile`; no 100 MB+ validation was observed in this re-audit. Legacy `src/parsers/comtrade_parser.py` still uses `read_bytes()`. | 100 MB+ files may still expose memory or latency problems, especially if legacy paths are invoked. | Add explicit active-provider large COMTRADE benchmarks and block legacy parser use in production. |
| BINARY32 unsupported | Medium / High | `_VALID_DAT_FORMATS` includes BINARY32, and active provider rejects it early with a clearer error. | Users with COMTRADE 2013 float32 data still cannot load those records. | Either implement BINARY32 or make unsupported status visible in file capability checks and UI messaging. |
| Full DataFrame copies in import/export | Medium | `data_assembler.py:159`, `data_assembler.py:302`, `export_writer.py:278` | Large CSV/XLSX import can double or triple memory footprint. | Track ownership and avoid copies when immutable. |
| Full direct CSV/Excel reads | Medium | `pd.read_csv` and `pd.read_excel` direct provider paths | Large files bypass wizard diagnostics and progress. | Reserve direct providers for normalized files or run them in a worker with limits. |
| Render-time per-channel decimation | Medium | `EventAnalysisSession.build_aligned_data` builds arrays per channel per view | Many channels/sources can repeat expensive masks and decimation. | Cache viewport slices per source/time window and invalidate on offset/visibility changes. |
| Optional OpenGL | Low / Medium | OpenGL depends on environment flag in `app/main.py` | Production performance may vary by launch environment. | Make rendering mode visible in diagnostics and document recommended deployment defaults. |

## 11. Test Coverage Gaps

Observed validation:

- `pytest -q` could not run because `pytest` was not on PATH.
- `.venv\Scripts\python.exe -m pytest --collect-only -q` collected 4023 tests in 5.91 seconds.
- `.venv\Scripts\python.exe -m pytest -q` completed with 4004 passed, 19 skipped, and 11 warnings in 118.49 seconds.

Gaps and risks:

- Full pass/fail/skip counts are now available, and the suite is green.
- The prior tests that expected removed or renamed `PowerwaveMainWindow` private methods have been migrated or removed.
- `SessionChannel.line_width` now has a default and no longer breaks existing fixtures.
- Several visualization-manager tests are mock-based and may not catch real Qt/PyQtGraph render failures.
- Large COMTRADE stress coverage around 100 MB+ and BINARY32 appears missing or not currently runnable.
- Packaging tests should assert that the installed package exposes the intended `app` entrypoint and not only legacy `src`.
- UI workflow acceptance should cover the exact required sequence from raw preview through normalized save and waveform render for CSV and Excel.
- COMTRADE acceptance should cover missing DAT, unsupported DAT format, BINARY/BINARY32 handling, channel metadata review, and successful waveform display.

## 12. Recommended Remediation Roadmap

Phase 0: Restore validation

- Preserve the current green suite while making architectural cleanup changes.
- Keep replacement session-canvas tests in place for workflows that previously depended on removed `PowerwaveMainWindow` private methods.
- Keep `SessionChannel.line_width` backward compatibility covered by tests.
- Add a short smoke command for import-to-waveform so future audits start from a known baseline.

Phase 1: Settle architecture ownership

- Decide canonical package root and update `pyproject.toml`.
- Label `src/` modules as canonical, compatibility, or legacy.
- Decide whether `VisualizationManager` or session-canvas orchestration is the primary display path.
- Align file dialogs, docs, and provider capability reporting.

Phase 2: Harden ingestion and visualization

- Make Import Wizard the only user-facing CSV/Excel normalization path.
- Add COMTRADE metadata/channel review before plotting.
- Centralize unit inference and column classification.
- Add early, actionable errors for unsupported COMTRADE variants.
- Consolidate cursor synchronization and expose render errors through diagnostics.

Phase 3: Large-data and workflow polish

- Reduce DataFrame/NumPy copies in import/export/session rendering.
- Add Excel sheet selection for multi-sheet files.
- Add timestamp repair screens for duplicate, missing, irregular, and mixed-format timestamps.
- Add viewport slice caching for dense multi-source sessions.

Phase 4: Cleanup and packaging

- Remove or quarantine legacy modules only after tests and packaging are green.
- Remove empty/obsolete files.
- Update repository structure docs to match the final architecture.
- Add CI gates for packaging, import workflow, runtime Qt smoke, visualization smoke, and large-file performance slices.

## 13. Suggested Next Development Phases

1. Validation Recovery: fix the broken runtime test file, reconcile removed widgets/tests, and produce a green or clearly triaged test baseline.
2. Architecture Convergence: choose and document canonical package/data/visualization paths, then update packaging and tests to match.
3. Import Workflow Unification: enforce the CSV/Excel wizard path and define the COMTRADE review path.
4. Industrial File Hardening: implement large COMTRADE memory strategy, BINARY32 policy/support, and 100 MB+ performance gates.
5. Visualization Reliability: consolidate sync/cursor ownership, surface render errors, and add real Qt/PyQtGraph acceptance smoke tests.
6. Cleanup: retire legacy modules and duplicate helpers after the above phases are validated.
