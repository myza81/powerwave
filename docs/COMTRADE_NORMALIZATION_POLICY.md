COMTRADE_NORMALIZATION_POLICY.md — Powerwave COMTRADE Ingestion Normalization Policy
PURPOSE

This document defines the mandatory normalization policy that the COMTRADE provider
implementation SHALL follow when parsing CFG and DAT files and constructing DisturbanceRecord.

It is the authoritative pre-implementation reference for ComtradeProvider.load().

This document covers:

  CFG structure parsing policy
  DAT format handling policy (ASCII, Binary, Binary32/Float32)
  Timestamp normalization policy
  Analog channel scaling policy
  Digital channel normalization policy
  Multi-rate sampling policy
  Engineering unit normalization
  Phase naming normalization
  Parser responsibilities and boundaries
  Analytics layer exclusions
  Error handling philosophy

This document does NOT contain implementation code.
This document does NOT modify the DisturbanceRecord contract.
This document does NOT modify the provider architecture.

ARCHITECTURAL CONTEXT

ComtradeProvider exists at:

  app/providers/comtrade/comtrade_provider.py

Its sole public contract is:

  load(path: Path) -> DisturbanceRecord

Downstream systems (visualization, analytics, synchronization) consume only DisturbanceRecord.
No COMTRADE-specific structure shall leak beyond this boundary.

The normalization layer lives entirely inside ComtradeProvider.load() (and private helpers).

───────────────────────────────────────────────────────────────────────────────
SECTION 1 — CFG FILE PARSING POLICY
───────────────────────────────────────────────────────────────────────────────

1.1 SUPPORTED REVISIONS

The parser SHALL support all three COMTRADE revision years:

  1991   — original COMTRADE standard
  1999   — adds TIMEMULT, TIMECODE, LEAPSEC, extended fields
  2013   — adds Binary32 (float32) DAT, extended CFG fields

Rev year detection:

  Line 1 of CFG: "station_name,rec_dev_id[,rev_yr]"

  If the third field is absent → treat as 1991.
  If the third field is a year string (e.g. "1999", "2013") → use it.
  If the third field is an unexpected value → log warning, default to 1999.

  Per IEC 60255-24: an absent rev_yr is standard for 1991 files.
  Per vendor reality: some 1999 files omit rev_yr entirely → safe to default 1999.

  Known vendor quirk — BEN32 (Malaysia):
    Some BEN32 files write an actual calendar year in the rev_yr field instead of
    the standard revision year (e.g. "2005", "2024" instead of "1991"/"1999"/"2013").
    These must be treated as 1999-format files. The implementation handles this by
    defaulting any unrecognised rev_yr value to "1999".

1.2 FIELD COUNT TOLERANCE

Analog channel CFG lines have 13 fields in 1999/2013, 10 fields in 1991:

  1991 (10 fields): an#, ch_id, ph, ccbm, uu, a, b, skew, min, max
  1999+ (13 fields): adds primary, secondary, PS flag

The parser SHALL:

  Accept both 10-field and 13-field analog lines without raising an error.
  If primary/secondary are absent → set primary_ratio and secondary_ratio to None.
  Never hard-code field count as a constant — parse defensively.

Digital channel CFG lines:

  1991/1999 (5 fields): dn#, ch_id, ph, ccbm, y
  Some vendors omit ph and ccbm → 3 fields: dn#, ch_id, y

  The parser SHALL:

    Accept both 3-field and 5-field digital lines.
    Missing ph → None.
    Missing ccbm → None.
    y (normal state) SHALL always be extracted — it is the minimum required field.

1.3 STATION AND DEVICE METADATA

Extract from CFG line 1:

  station_name: strip whitespace
  rec_dev_id:   strip whitespace — may be empty → use empty string, not None

These map directly to RecordingMetadata.station_name and RecordingMetadata.recorder_name.

1.4 NOMINAL FREQUENCY

CFG line: "lf" (line frequency)

  Accepted values: 50, 60 (Hz)
  If absent or unparseable → default 50.0 Hz, log warning.

Maps to RecordingMetadata.nominal_frequency.

1.5 TIME MULTIPLIER (TIMEMULT)

Present in COMTRADE 1999+ CFG files.

  TIMEMULT is a floating-point multiplier applied to all DAT timestamp fields.
  Default value: 1.0 (if absent or rev_yr = 1991).
  Unit: microseconds per unit of the raw timestamp integer.

  Example: TIMEMULT = 0.1 → raw timestamp × 0.1 μs = time in microseconds.

Policy:

  ALWAYS read TIMEMULT if the line is present.
  If absent → use 1.0 (microsecond resolution assumed).
  Apply TIMEMULT before any time array construction.
  Store in TimingInformation.time_multiplier.

───────────────────────────────────────────────────────────────────────────────
SECTION 2 — TIMESTAMP NORMALIZATION POLICY
───────────────────────────────────────────────────────────────────────────────

2.1 START TIME AND TRIGGER TIME

CFG provides two absolute timestamps:

  start_time:   the first sample time
  trigger_time: the event trigger time

Format: "DD/MM/YYYY,HH:MM:SS.SSSSSS"

  Subsecond precision: up to 6 decimal digits (microseconds).
  Parse using datetime.strptime with "%d/%m/%Y,%H:%M:%S.%f".

  If the subsecond field is absent → treat as 0 microseconds.
  If the timestamp string is malformed → raise ProviderLoadError.

These map to:

  TimingInformation.start_time   (datetime, no timezone — COMTRADE is local wall-clock)
  TimingInformation.trigger_time (datetime, no timezone)

Timezone handling:

  COMTRADE CFG contains NO timezone field (IEEE C37.111 standard).
  Stored timestamps are LOCAL wall-clock time at the recording substation.
  The parser SHALL NOT assume UTC.
  The parser SHALL NOT apply timezone conversion.
  Timezone correction is an application-level responsibility (outside the parser).
  Store TimingInformation.timezone = None.

2.2 TIME ARRAY CONSTRUCTION

The time array represents elapsed seconds from start_time for each sample.

DAT timestamp field:

  Each DAT row contains a raw timestamp integer (n field).
  The physical time in microseconds = n × TIMEMULT.
  The time in seconds = (n × TIMEMULT) / 1_000_000.

Policy:

  Build the time array as numpy float64 (seconds from start).
  Do NOT store absolute datetime objects per sample.
  Absolute timestamps are only start_time and trigger_time (scalar datetimes).
  The waveform_data DataFrame time column SHALL be: seconds elapsed from start_time (float64).

Multi-rate time arrays:

  When multiple sampling rates are present (nrates > 1), the DAT rows are contiguous
  across rate sections. Each section uses the same DAT timestamp field (n).
  The time array is constructed from all n values across all sections uniformly.
  The time array is NON-UNIFORM at rate section boundaries — this is correct and expected.

2.3 TRIGGER TIME INDEX

The trigger time SHALL be locatable in the time array:

  trigger_offset_seconds = (trigger_time - start_time).total_seconds()

  This value is used by analytics and visualization for pre/post-fault alignment.
  The parser SHALL NOT embed this offset in the time array.
  It is derived from TimingInformation when needed downstream.

───────────────────────────────────────────────────────────────────────────────
SECTION 3 — DAT FORMAT HANDLING POLICY
───────────────────────────────────────────────────────────────────────────────

3.1 FORMAT DETECTION

CFG specifies the DAT format on the ft (file type) line:

  ASCII   → ASCII DAT
  BINARY  → Binary (16-bit integer, little-endian) DAT
  BINARY32 → Binary32 (IEEE 754 float32) DAT — COMTRADE 2013

The parser SHALL detect the format from CFG, not from file extension or content sniffing.

If ft is absent → default ASCII (COMTRADE 1991 behavior).
If ft is an unrecognized value → raise ProviderLoadError with a clear message.

Known vendor quirk — FLOAT32 alias:
  Some vendor IEDs (observed in the field) write "FLOAT32" instead of "BINARY32"
  as the ft field value. "FLOAT32" is not a standard COMTRADE format name but is
  functionally identical to BINARY32 (IEEE 754 float32 samples, same binary layout).
  Future implementation should accept "FLOAT32" as an alias for BINARY32 rather than
  raising ProviderLoadError, to improve compatibility with non-conforming vendor files.

3.2 ASCII DAT FORMAT

Each row contains:

  n, timestamp, A1, A2, ..., An, D1

  Where:
    n          = sample number (1-based integer)
    timestamp  = raw time units (integer, apply TIMEMULT for seconds)
    A1..An     = raw analog values (integers or floats as text)
    D1         = packed digital word (unsigned integer, 16 channels per word)

Multiple digital words per row: one per 16 digital channels (ceiling division).

Parsing policy:

  Use numpy.loadtxt or vectorized string split — avoid Python-loop row iteration.
  Strip header rows before parsing.
  Validate column count against CFG nA (analog count) + nD (digital word count) + 2.
  If column count mismatches → raise ProviderLoadError.

3.3 BINARY DAT FORMAT

Each row is a fixed-size binary record:

  Field       Type              Size
  n           uint32            4 bytes
  timestamp   uint32            4 bytes
  A[0..nA-1]  int16 × nA        2 bytes each
  D[0..nDw-1] uint16 × nDw      2 bytes each (nDw = ceil(nD / 16))

Total row size: 8 + (2 × nA) + (2 × nDw) bytes.

Parsing policy:

  Use numpy.frombuffer for bulk binary read — avoid Python struct loops.
  Read entire DAT file as bytes, then reshape using calculated row size.
  Validate file size: file_size % row_size == 0.
  If file size is not a multiple of row size → raise ProviderLoadError.

Byte order:

  IEEE C37.111 specifies little-endian for binary files.
  Use numpy dtype with '<' prefix (e.g., '<i2', '<u4').

3.4 BINARY32 (FLOAT32) DAT FORMAT

Present in COMTRADE 2013 only.

Each row contains:

  Field       Type              Size
  n           uint32            4 bytes
  timestamp   uint32            4 bytes
  A[0..nA-1]  float32 × nA      4 bytes each
  D[0..nDw-1] uint16 × nDw      2 bytes each

Row size: 8 + (4 × nA) + (2 × nDw) bytes.

Parsing policy:

  Same bulk numpy.frombuffer approach as Binary.
  float32 values are ALREADY scaled — they represent physical engineering values.
  DO NOT apply analog scaling (a, b factors) to Binary32 analog values.
  They are stored ready-to-use.
  Digital words are unpacked the same way as Binary format.

3.5 DAT FILE LOCATION

The CFG file specifies the associated DAT file path.

Location rules:

  1. If the CFG contains an explicit data filename → use it.
  2. Otherwise → replace CFG extension with .dat (same directory, same stem).
  3. If DAT file does not exist → raise ProviderLoadError with path shown.

For .comtrade files (COMTRADE 2013 combined format):

  CFG and DAT are contained in the same file.
  The parser SHALL handle combined-format files without requiring a separate .dat file.

───────────────────────────────────────────────────────────────────────────────
SECTION 4 — ANALOG CHANNEL SCALING POLICY
───────────────────────────────────────────────────────────────────────────────

4.1 PHYSICAL VALUE CALCULATION

For ASCII and Binary DAT (16-bit integer raw values):

  physical_value = (a × raw_integer) + b

  Where:
    a  = channel multiplier (CFG field 6: floating-point)
    b  = channel offset     (CFG field 7: floating-point)

This SHALL be applied to every sample of every analog channel.
It SHALL be applied vectorized across the full numpy array.

It SHALL NOT be applied to Binary32 channels (values are pre-scaled).

4.2 SKEW CORRECTION

CFG field 8 specifies per-channel time skew (in microseconds):

  skew = time offset between this channel's ADC sampling and the timestamp reference.

Policy:

  Skew correction is NOT applied in the parser.
  The skew value SHALL be preserved and stored in AnalogChannel.description
  (or a dedicated field if added to the contract later).
  Reason: skew correction requires time-array resampling, which belongs in analytics.

4.3 PRIMARY / SECONDARY RATIO (PS FLAG)

CFG fields 11, 12, 13 (1999+ only):

  primary   = primary CT/VT ratio value
  secondary = secondary CT/VT ratio value
  PS        = 'P' (primary) or 'S' (secondary)

Policy:

  Store primary_ratio and secondary_ratio in AnalogChannel as-is.
  The parser SHALL NOT apply PS conversion.
  The data in waveform_data is stored at the a/b-scaled value level.
  If PS = 'P': stored values are primary-referred.
  If PS = 'S': stored values are secondary-referred.
  Downstream analytics decides whether to convert.

4.4 MISSING OR ZERO SCALE FACTORS

If 'a' = 0.0 in CFG:

  This is a malformed CFG — physical value would always be 'b'.
  Log a warning identifying the channel.
  Do NOT raise an error — continue with a = 0.0.
  The channel will appear as a flat line (constant at 'b').

If 'a' is absent from CFG:

  Default a = 1.0, b = 0.0.
  Log a warning.

4.5 WAVEFORM DATA STORAGE

All analog channels SHALL be stored in waveform_data as float64 columns.

  Column naming: use the channel name from CFG (ch_id field), stripped of whitespace.
  If ch_id is empty → use generated name: "A{n}" where n is the 1-based analog index.

The time column SHALL be named "time" and stored as float64 (seconds).

Column order in DataFrame:

  time, [analog channels in CFG order], [digital channels in CFG order]

This column order SHALL remain stable and deterministic.

───────────────────────────────────────────────────────────────────────────────
SECTION 5 — DIGITAL CHANNEL NORMALIZATION POLICY
───────────────────────────────────────────────────────────────────────────────

5.1 BIT EXTRACTION

Digital channels are packed into 16-bit unsigned integers (words) in the DAT.

  Channel d (0-indexed) occupies bit (d % 16) of word (d // 16).
  Extraction: state = (word >> (d % 16)) & 0x1

This SHALL be performed vectorized across all samples using numpy bitwise operations.
Never iterate per-sample.

5.2 NORMAL STATE

Each digital channel has a "normal state" (y field in CFG, value 0 or 1).

Policy:

  The parser SHALL store the normal_state value in DigitalChannel.normal_state.
  The parser SHALL NOT invert the bit based on normal_state.
  The raw extracted bit is stored as-is.
  The analytics / visualization layer interprets the logical state using normal_state.

Reason: preserving raw bit values allows downstream to choose whether to display
as raw or as "active/normal" logical state. This is a display decision, not a parse decision.

5.3 DIGITAL CHANNEL NAMING

Use ch_id from CFG, stripped of whitespace.
If ch_id is empty → use generated name: "D{n}" where n is the 1-based digital index.

Digital channel values stored in waveform_data SHALL be int8 (0 or 1).

5.4 DIGITAL CHANNEL COLUMNS IN DATAFRAME

Digital channels SHALL be stored as separate columns in waveform_data (not as packed words).
Each digital channel has its own named column.

  Reason: downstream visualization and analytics consume individual channel state arrays,
  not packed bit words.

───────────────────────────────────────────────────────────────────────────────
SECTION 6 — MULTI-RATE SAMPLING POLICY
───────────────────────────────────────────────────────────────────────────────

6.1 NRATES STRUCTURE

CFG specifies:

  nrates  = number of sampling rate sections
  For each section: samp (rate in Hz), endsamp (last sample index for this section)

Example:

  2       ← two rate sections
  5000,3000  ← 5000 Hz for first 3000 samples
  500,3500   ← 500 Hz for next 500 samples (samples 3001–3500)

6.2 SAMPLING INFORMATION CONTRACT

All rate sections SHALL be stored in SamplingInformation:

  sampling_rates:    [5000.0, 500.0]
  samples_per_rate:  [3000, 500]

These map directly from CFG nrates lines.

6.3 TIME ARRAY WITH MULTI-RATE

The time array is built from the raw DAT timestamp column (all nTotal samples).
It is NOT constructed from sampling rates.

  Reason: DAT timestamps are the authoritative sample timing.
  Computed ideal-rate timestamps may diverge due to ADC jitter or timing drift.

The result is a non-uniform time array across rate boundaries — this is correct.

6.4 SINGLE DOMINANT RATE

For analytics that require a single representative sample rate:

  Use the first (highest) sampling rate from SamplingInformation.sampling_rates[0].
  Store in RecordingMetadata — no dedicated field exists; it is derivable from SamplingInformation.

The parser SHALL NOT compute a blended or averaged rate.

───────────────────────────────────────────────────────────────────────────────
SECTION 7 — PHASE NAMING NORMALIZATION POLICY
───────────────────────────────────────────────────────────────────────────────

7.1 PHASE FIELD EXTRACTION

CFG analog channel field 3 (ph) contains the phase identifier.

  Store directly in AnalogChannel.phase — do NOT normalize the raw value.

Reason: the analytics layer (signal role detector) maps phase values to A/B/C/N
conventions. The parser preserves the source string faithfully.

7.2 KNOWN PHASE VARIANTS (for documentation only)

Common values found in real files:

  BEN32 (Malaysia):  R, Y, B, N  (maps to A, B, C, N at analytics layer)
  NARI (China):       a, b, c, N  (maps to A, B, C, N at analytics layer)
  IEC standard:       A, B, C, N
  ABB:                L1, L2, L3, N (some variants)
  GE/SEL:             A, B, C, N   (standard)

The parser SHALL NOT translate these. Preserve as-is.
Signal role detection and phase normalization belong in the analytics/signal processing layer.

7.3 CHANNEL UNIT (uu FIELD)

CFG analog channel field 5 (uu) contains the engineering unit string.

  Store directly in AnalogChannel.unit — do NOT translate or normalize.

Common values: kV, V, A, kA, MW, MVAr, Hz, RPM, %
The analytics layer interprets unit context for scaling and display.

───────────────────────────────────────────────────────────────────────────────
SECTION 8 — ENGINEERING UNIT NORMALIZATION POLICY
───────────────────────────────────────────────────────────────────────────────

8.1 NO AUTOMATIC UNIT CONVERSION IN THE PARSER

The COMTRADE parser SHALL NOT:

  convert kV to V
  convert kA to A
  convert % to a fraction
  convert RPM to rad/s
  normalize any unit to a canonical base unit

Reason: the a/b scaling factors in CFG already encode the final physical values
in the units specified by the uu field. No further unit conversion is required
to obtain the engineering-meaningful value.

8.2 WHAT THE PARSER DOES

The parser SHALL:

  apply a/b scaling to produce physical values (analog channels, non-Binary32)
  store values with the unit string exactly as declared in CFG

Example:

  CFG: uu = "kV", a = 0.01, b = 0.0
  Raw DAT value = 27500
  Physical = 0.01 × 27500 + 0.0 = 275.0 (unit: kV)
  Stored as 275.0 with unit = "kV"

8.3 PU NORMALIZATION

Per-unit normalization is NOT a parser responsibility.
PU mode is a display-layer calculation that divides channel values by a user-specified
nominal base value. This belongs in the visualization or analytics layer.

───────────────────────────────────────────────────────────────────────────────
SECTION 9 — SAMPLING INTEGRITY PRESERVATION POLICY
───────────────────────────────────────────────────────────────────────────────

9.1 NO RESAMPLING IN THE PARSER

The parser SHALL NOT:

  interpolate between samples
  upsample to a common rate
  downsample for display
  align multi-rate sections to a common grid
  fill gaps with NaN

Reason: resampling changes the engineering meaning of the data. Interpolated values
are synthetic. For protection engineering analysis, only measured samples are trustworthy.

9.2 MISSING SAMPLE DETECTION

If the number of rows in DAT does not match the total expected samples (sum of samples_per_rate):

  Log a warning: "DAT sample count mismatch: expected N, found M"
  Store the actual available samples without padding.
  Do NOT raise an error (partial files may be valid for analysis).

If DAT is completely empty:

  Raise ProviderLoadError.

9.3 SAMPLE COUNT VALIDATION

After parsing, the parser SHALL verify:

  len(waveform_data) > 0                            (non-empty)
  len(waveform_data.columns) > 1                    (at least time + one channel)
  len(analog_channels) >= 0 (zero is allowed — digital-only files exist)
  len(digital_channels) >= 0 (zero is allowed — analog-only files exist)

These are the minimum validity checks. Full DisturbanceRecord.validate() is called
by the ProviderManager after load() returns.

───────────────────────────────────────────────────────────────────────────────
SECTION 10 — PARSER RESPONSIBILITIES VS ANALYTICS LAYER
───────────────────────────────────────────────────────────────────────────────

10.1 WHAT BELONGS IN THE PARSER (ComtradeProvider)

The parser is responsible for:

  CFG file reading and structural parsing
  DAT file reading (ASCII, Binary, Binary32)
  Analog a/b scaling (for ASCII and Binary only)
  Digital bit extraction from packed words
  Time array construction (seconds from start_time)
  SamplingInformation construction from nrates
  TimingInformation construction (start_time, trigger_time, TIMEMULT)
  RecordingMetadata construction (station, device, nominal freq, source file, provider type)
  AnalogChannel list construction (name, unit, phase, index, scale, offset, ratios)
  DigitalChannel list construction (name, index, normal_state)
  DisturbanceRecord construction and return

10.2 WHAT DOES NOT BELONG IN THE PARSER

The parser SHALL NOT perform:

  Signal role detection (V_PHASE, I_EARTH, etc.)
  Phase normalization (R/Y/B → A/B/C)
  Per-unit conversion
  RMS calculation
  Phasor calculation
  Frequency analysis
  Symmetrical component calculation
  Skew time correction
  PS ratio conversion
  Timezone conversion
  Downsampling or decimation
  Channel grouping by bay name
  Any rendering or visualization

These belong in:

  analytics layer (app/analytics/)
  signal role detector (future app/services/ or app/analytics/)
  visualization layer (app/visualization/)
  application-level settings (app/config/)

10.3 BOUNDARY RULE

If a processing step changes the meaning of a sample value or requires domain knowledge
beyond what the COMTRADE standard defines → it belongs outside the parser.

The parser converts raw DAT bytes/text into physical engineering values.
Everything beyond that is analytics.

───────────────────────────────────────────────────────────────────────────────
SECTION 11 — WHAT MUST BE PRESERVED FROM RAW COMTRADE
───────────────────────────────────────────────────────────────────────────────

11.1 PRESERVED AS-IS

The following SHALL be preserved without modification:

  station_name            (from CFG line 1)
  rec_dev_id              (from CFG line 1)
  rev_yr                  (from CFG line 1)
  lf (nominal frequency)  (from CFG)
  start_time              (from CFG, as datetime — no tz conversion)
  trigger_time            (from CFG, as datetime — no tz conversion)
  TIMEMULT                (from CFG, stored in TimingInformation.time_multiplier)
  channel ph (phase)      (from CFG, stored in AnalogChannel.phase)
  channel uu (unit)       (from CFG, stored in AnalogChannel.unit)
  channel a, b            (from CFG, applied to compute values; also stored in AnalogChannel)
  channel normal_state    (digital channels, from CFG)
  primary, secondary      (from CFG, stored in AnalogChannel — not applied)
  PS flag                 (from CFG, stored in AnalogChannel.description or future field)
  sampling_rates[]        (from CFG nrates section)
  samples_per_rate[]      (from CFG nrates section)

11.2 DERIVED DURING PARSE (NOT FROM CFG DIRECTLY)

  time array              (derived from DAT n column × TIMEMULT, unit: seconds)
  analog physical values  (derived: a × raw + b for ASCII/Binary; direct for Binary32)
  digital bit states      (derived: bit extraction from packed DAT words)

11.3 NOT PRESERVED (DISCARDED)

  Raw DAT integer values  (only physical values are kept)
  Packed digital words    (unpacked to individual bit columns)
  CFG line structure      (not stored — only extracted values are kept)

───────────────────────────────────────────────────────────────────────────────
SECTION 12 — PARSER ERROR HANDLING PHILOSOPHY
───────────────────────────────────────────────────────────────────────────────

12.1 FAIL-SAFE DEFAULTS FOR BENIGN DEFECTS

Some CFG files from real-world relays are structurally valid but contain minor
non-conformances. The parser SHALL apply fail-safe defaults rather than refusing to load:

  Missing rev_yr       → default "1999", log warning
  Missing TIMEMULT     → default 1.0, log warning
  Missing lf           → default 50.0, log warning
  10-field analog line → treat primary/secondary as None, no error
  3-field digital line → treat ph/ccbm as None, no error
  Empty ch_id          → use generated name ("A{n}" or "D{n}"), log info

12.2 HARD FAILURES (RAISE ProviderLoadError)

The parser SHALL raise ProviderLoadError for:

  CFG file not found or unreadable
  DAT file not found or unreadable
  DAT format ft is unrecognized
  Binary DAT file size is not a multiple of row_size
  ASCII DAT column count does not match CFG channel declaration
  Completely empty DAT (zero rows)
  start_time or trigger_time is malformed beyond recovery
  nA + nD declared but channel definition lines are fewer than declared

Every ProviderLoadError SHALL include:

  A human-readable message identifying the specific file and defect.
  __cause__ chaining if wrapping a lower-level exception.

12.3 WARNINGS (LOG ONLY, CONTINUE)

The parser SHALL log a warning (not raise) for:

  DAT sample count mismatch (partial file)
  a = 0.0 scale factor on analog channel
  Unrecognized PS flag value
  rev_yr not in {1991, 1999, 2013}

12.4 NO SILENT PARTIAL FAILURE

If the parser cannot construct a complete DisturbanceRecord with all declared channels,
it SHALL either:

  Include the partial channel with a warning (sample count mismatch cases), or
  Raise ProviderLoadError (structural defect cases)

It SHALL NOT silently return a record with fewer channels than declared without warning.

───────────────────────────────────────────────────────────────────────────────
SECTION 13 — DISTURBANCERECORD CONSTRUCTION CHECKLIST
───────────────────────────────────────────────────────────────────────────────

After parsing, the parser constructs DisturbanceRecord as follows:

  metadata = RecordingMetadata(
      station_name   = <from CFG line 1, stripped>
      recorder_name  = <rec_dev_id from CFG line 1, stripped>
      source_file    = <absolute path to CFG file as string>
      provider_type  = "COMTRADE"
      nominal_frequency = <lf from CFG, float>
      device_id      = None  (may be populated from rec_dev_id in future)
      timezone       = None  (COMTRADE has no timezone field)
  )

  analog_channels = [AnalogChannel(...) for each CFG analog line]
  digital_channels = [DigitalChannel(...) for each CFG digital line]

  sampling_info = SamplingInformation(
      sampling_rates    = [rate1, rate2, ...]
      samples_per_rate  = [endsamp1, endsamp2 - endsamp1, ...]
  )

  timing_info = TimingInformation(
      start_time       = <parsed datetime from CFG>
      trigger_time     = <parsed datetime from CFG>
      time_multiplier  = <TIMEMULT, default 1.0>
      timezone         = None
  )

  waveform_data = pd.DataFrame({
      "time": <float64 array, seconds from start>,
      "<ch_id_1>": <float64 array, physical values>,
      "<ch_id_2>": <float64 array, physical values>,
      ...
      "<dig_ch_id_1>": <int8 array, 0 or 1>,
      ...
  })

  disturbance_info = None  (no structured disturbance info in COMTRADE CFG)

  return DisturbanceRecord(
      metadata=metadata,
      waveform_data=waveform_data,
      analog_channels=analog_channels,
      digital_channels=digital_channels,
      sampling_info=sampling_info,
      timing_info=timing_info,
      disturbance_info=None,
  )

───────────────────────────────────────────────────────────────────────────────
SECTION 14 — PERFORMANCE REQUIREMENTS FOR THE PARSER
───────────────────────────────────────────────────────────────────────────────

14.1 TARGET FILE SIZE

The parser SHALL support COMTRADE files up to 100MB+ without UI blocking.
All parsing SHALL execute on a worker thread (not the Qt UI thread).

14.2 REQUIRED VECTORIZED OPERATIONS

The following SHALL use numpy vectorized operations, never Python loops:

  ASCII DAT parsing:    numpy.loadtxt or vectorized string-split
  Binary DAT parsing:   numpy.frombuffer then structured array indexing
  Analog a/b scaling:   (a_array * raw_array) + b_array   (broadcast multiply/add)
  Digital unpacking:    (words_array >> bit_offsets) & 0x1  (broadcast shift/mask)
  Time array:           n_array * timemult / 1_000_000     (scalar multiply/divide)

14.3 MEMORY EFFICIENCY

  Avoid intermediate copies of the full waveform array.
  Build waveform_data from numpy arrays (not row-by-row DataFrame append).
  Use pd.DataFrame constructor with a dict of pre-built arrays.
  Store float64 for time and analog values; int8 for digital.

14.4 THREADING NOTE

The parser itself is not responsible for thread management.
The calling layer (application service, future UI integration) places the parser call
on a QThread worker. The parser must be thread-safe and stateless (no shared mutable state).

───────────────────────────────────────────────────────────────────────────────
NORMALIZATION POLICY SUMMARY TABLE
───────────────────────────────────────────────────────────────────────────────

Concern                         Parser Action           Downstream Action
──────────────────────────────────────────────────────────────────────────────
Rev year detection              Detect, default 1999    N/A
Field count tolerance           Accept 10 or 13 fields  N/A
Timestamp parsing               Parse as datetime       N/A
Timezone conversion             None (not done)         App-level setting
TIMEMULT application            Applied to time array   N/A
start_time / trigger_time       Stored as-is            Trigger alignment
Analog a/b scaling              Applied (ASCII/Binary)  N/A
Binary32 pre-scaled values      Used directly           N/A
Skew correction                 Preserved in channel    Analytics layer
PS ratio conversion             Preserved, not applied  Analytics layer
Digital bit extraction          Vectorized              N/A
Normal state inversion          Not applied             Visualization layer
Phase normalization (R→A etc.)  Not applied             Signal role detector
Unit conversion (kV→V etc.)     Not applied             Analytics layer
PU normalization                Not applied             Visualization layer
RMS / phasor / FFT              Not performed           Analytics engines
Bay grouping                    Not performed           App/analytics layer
Skew time correction            Not performed           Analytics layer
Resampling                      Not performed           Analytics layer
Downsampling                    Not performed           Visualization engine
Timezone application            Not performed           App settings
──────────────────────────────────────────────────────────────────────────────

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURAL ALIGNMENT DECLARATION
───────────────────────────────────────────────────────────────────────────────

This policy is aligned with:

  docs/ARCHITECTURE.md        — provider layer responsibilities, strict layer isolation
  docs/DATA_CONTRACT.md       — DisturbanceRecord structure and normalization requirements
  docs/PROVIDER_PATTERN.md    — provider isolation rules, error handling requirements
  docs/PERFORMANCE_REQUIREMENTS.md — vectorized parsing, memory efficiency, threading
  docs/LEGACY_CODEBASE_POLICY.md   — no src/ imports; ComtradeProvider is independent
  agent/REPOSITORY_STATE.md   — DisturbanceRecord and provider system locked; parser next

ComtradeProvider.load() SHALL be the sole implementation boundary.
No downstream system (visualization, analytics, synchronization) shall require
knowledge of COMTRADE format internals.

FINAL PRINCIPLE

The COMTRADE parser converts a file on disk into a DisturbanceRecord in memory.
Nothing more. Nothing less.

All interpretation, normalization beyond a/b scaling, analytics, and display adaptation
belong downstream. The parser's contract is to faithfully extract and minimally transform
what the COMTRADE standard defines — and to fail clearly when the file does not conform.
