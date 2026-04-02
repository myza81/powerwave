# SKILL INDEX — PowerWave Analyst (Final)

## Quick Reference — Which skill to load

| Task / Module                        | Skill File                         |
|--------------------------------------|------------------------------------|
| comtrade_parser.py / .cfg / .dat     | SKILL_comtrade_parser.md           |
| signal_role_detector.py              | SKILL_channel_mapping.md           |
| channel_mapping_dialog.py            | SKILL_channel_mapping.md           |
| csv_parser.py / excel_parser.py      | SKILL_channel_mapping.md           |
| pmu_csv_parser.py                    | SKILL_pmu_power.md                 |
| power_calculator.py                  | SKILL_pmu_power.md                 |
| pmu_record.py / power_canvas.py      | SKILL_pmu_power.md                 |
| rms_calculator.py                    | SKILL_signal_processing.md         |
| phasor_calculator.py                 | SKILL_signal_processing.md         |
| symmetrical_components.py            | SKILL_signal_processing.md         |
| fft_analyzer.py                      | SKILL_signal_processing.md         |
| frequency_tracker.py                 | SKILL_signal_processing.md         |
| decimator.py                         | SKILL_signal_processing.md         |
| channel_canvas.py / main_window.py   | SKILL_pyqt6_rendering.md           |
| phasor_canvas.py / fft_canvas.py     | SKILL_pyqt6_rendering.md           |
| Any UI freeze or performance bug     | SKILL_pyqt6_rendering.md           |
| merged_session.py                    | SKILL_merging_timesync.md          |
| time_sync_engine.py                  | SKILL_merging_timesync.md          |
| merge_manager.py / sync_panel.py     | SKILL_merging_timesync.md          |
| Any wrong measurement value          | SKILL_signal_processing.md         |
| Any power calculation error          | SKILL_pmu_power.md                 |
| Any channel not detected correctly   | SKILL_channel_mapping.md           |
| Any CFG/DAT parsing error            | SKILL_comtrade_parser.md           |

## File List
- SKILL_comtrade_parser.md   — CFG/DAT all formats, BEN32+NARI quirks, bay extraction
- SKILL_channel_mapping.md   — Full taxonomy, R/Y/B+Chinese detection, dialog, profiles
- SKILL_signal_processing.md — RMS, phasor, FFT, decimation, frequency, symmetrical comp
- SKILL_pyqt6_rendering.md   — PyQt6 patterns, PyQtGraph, thread safety, colour scheme
- SKILL_pmu_power.md         — PMU CSV parsing, Malaysian grid, P/Q/S/PF formulas
- SKILL_merging_timesync.md  — MergedSession, GPS align, cross-correlation
