implement_comtrade_provider.md — Powerwave COMTRADE Provider Implementation Directive
STATUS: READY FOR IMPLEMENTATION
ISSUER: ChatGPT (Architecture)
EXECUTOR: Claude Code
DATE: 2026-05-10

───────────────────────────────────────────────────────────────────────────────
OBJECTIVE
───────────────────────────────────────────────────────────────────────────────

Implement ComtradeProvider.load() — the first real parser in the Powerwave platform.

Replace the existing NotImplementedError stub with a production-quality COMTRADE
ingestion implementation that parses CFG + DAT files and returns a fully normalized
DisturbanceRecord.

Primary normalization reference: docs/COMTRADE_NORMALIZATION_POLICY.md

───────────────────────────────────────────────────────────────────────────────
SCOPE
───────────────────────────────────────────────────────────────────────────────

IN SCOPE
  app/providers/comtrade/comtrade_provider.py — replace stub with real implementation
  tests/unit/test_comtrade_provider.py        — new comprehensive test file

OUT OF SCOPE (do not implement)
  Binary32 / COMTRADE 2013 float32 DAT format
  Timezone correction (store None, apply correction at application layer)
  Skew time correction (preserve skew value, correction belongs in analytics)
  PS ratio conversion (preserve primary/secondary values, do not apply)
  Resampling or interpolation of any kind
  Signal role detection, phase normalization (A/B/C mapping)
  Per-unit (PU) normalization
  RMS, phasor, or any analytics
  UI integration, threading, caching
  ProviderManager redesign
  DisturbanceRecord contract changes
  src/ imports of any kind

───────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION REQUIREMENTS
───────────────────────────────────────────────────────────────────────────────

1. CFG PARSING
   - Parse rev_yr from line 1: default "1999" if absent or unrecognized
   - Parse channel counts (TT,##A,##D or legacy nA,nD formats)
   - Parse analog channel definitions: 10-field (1991) and 13-field (1999+)
   - Parse digital channel definitions: 3-field and 5-field variants
   - Parse lf (nominal frequency): default 50.0 if absent
   - Parse nrates + (samp, endsamp) pairs
   - Parse start_time and trigger_time as datetime objects (no tz conversion)
   - Parse ft (ASCII or BINARY): raise ProviderLoadError for unrecognized ft
   - Parse TIMEMULT (1999+ only): default 1.0 if absent

2. ASCII DAT PARSING
   - Use vectorized numpy.loadtxt (no per-row Python loops)
   - Validate column count: must match 2 + n_analog + ceil(n_digital / 16)
   - Extract time array: (raw_timestamp × TIMEMULT) / 1_000_000 (seconds)
   - Extract analog columns and digital word columns

3. BINARY DAT PARSING
   - Use numpy.frombuffer with structured dtype (no per-row Python loops)
   - Validate file size: must be exact multiple of row_size
   - Row layout: uint32(n) + uint32(ts) + int16*nA + uint16*nDw (little-endian)
   - Extract time array: (raw_timestamp × TIMEMULT) / 1_000_000 (seconds)

4. ANALOG SCALING
   - Apply: physical = a × raw + b (vectorized broadcast)
   - For ASCII and Binary only — Binary32 is out of scope
   - Warn (not error) if a == 0.0

5. DIGITAL EXTRACTION
   - Vectorized bit extraction: (words[:, d//16] >> (d%16)) & 0x1
   - Store as int8 (0 or 1)
   - Preserve normal_state in DigitalChannel — do NOT invert

6. DISTURBANCERECORD CONSTRUCTION
   Per docs/COMTRADE_NORMALIZATION_POLICY.md Section 13:
   - metadata.provider_type = "COMTRADE"
   - metadata.timezone = None
   - timing_info.timezone = None
   - timing_info.time_multiplier = TIMEMULT value
   - disturbance_info = None
   - waveform_data column order: time, [analog channels], [digital channels]

7. ERROR HANDLING
   - CFG or DAT unreadable: ProviderLoadError with file path in message
   - DAT not found: ProviderLoadError
   - Binary size mismatch: ProviderLoadError
   - ASCII column mismatch: ProviderLoadError
   - Empty DAT: ProviderLoadError
   - Invalid ft: ProviderLoadError
   - Malformed timestamp: ProviderLoadError
   - DAT sample count mismatch (partial file): warnings.warn, continue
   - a == 0.0: warnings.warn, continue

───────────────────────────────────────────────────────────────────────────────
TESTING REQUIREMENTS
───────────────────────────────────────────────────────────────────────────────

All tests in: tests/unit/test_comtrade_provider.py

Required test coverage:
  - CFG parsing: 1991 format, 1999 format with TIMEMULT, multi-rate, default rev_yr
  - ASCII DAT: basic parse, analog scaling correctness, digital extraction, column mismatch
  - Binary DAT: basic parse, analog scaling correctness, digital extraction, size mismatch
  - Full load: 1991 ASCII end-to-end, 1999 ASCII with TIMEMULT, Binary end-to-end
  - Multi-rate: SamplingInformation constructed correctly
  - Error handling: missing CFG, missing DAT, empty DAT, invalid ft
  - DisturbanceRecord validity: validate() returns [] for well-formed records

Also update:
  tests/unit/test_provider_manager.py
  → TestComtradeProviderStub.test_load_raises_not_implemented
  → Change to test_load_raises_provider_load_error_for_missing_file
  → Expect ProviderLoadError (file missing), not NotImplementedError

───────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
───────────────────────────────────────────────────────────────────────────────

- Use .venv/Scripts/python.exe for all test execution
- No imports from src/
- No new external dependencies (numpy, pandas already available)
- All private helpers are module-level functions prefixed with _
- ComtradeProvider class methods only: can_load(), load()
- No class state — parser must be stateless and thread-safe
- Follow LEGACY_CODEBASE_POLICY.md strictly

───────────────────────────────────────────────────────────────────────────────
COMPLETION CRITERIA
───────────────────────────────────────────────────────────────────────────────

- All existing tests still pass (66 tests)
- New COMTRADE tests pass (target: 30+ additional tests)
- ComtradeProvider().load(valid_cfg_path) returns valid DisturbanceRecord
- DisturbanceRecord.validate() returns [] for well-formed records
- agent/HANDOFF.md, agent/TASK.md, agent/REPOSITORY_STATE.md updated
