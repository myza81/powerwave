implement_csv_provider.md — Powerwave CSV Provider Implementation Directive
PURPOSE

This directive implements the CSV ingestion provider for Powerwave.

The CSV provider is responsible for loading generic waveform data from .csv files and normalizing it into:

DisturbanceRecord

This provides a simple, flexible ingestion path for:

exported recorder data
engineering analysis files
manually prepared waveform datasets
future visualization testing

This directive implements only CSV ingestion.

It does NOT implement:

Excel ingestion
waveform rendering
analytics calculations
synchronization logic
UI file import flow
IMPLEMENTATION OBJECTIVES

The implementation shall:

replace the CsvProvider stub with real CSV loading
parse CSV files into pandas DataFrame
infer or accept a time column
normalize waveform data into DisturbanceRecord
create AnalogChannel entries from numeric columns
create DigitalChannel entries from binary/state columns where practical
populate metadata, timing, and sampling information
provide meaningful ProviderLoadError failures
remain parser-independent, UI-independent, and analytics-independent
TARGET LOCATION

Implement primarily in:

app/providers/csv/csv_provider.py

Tests:

tests/unit/test_csv_provider.py

Update if needed:

app/providers/csv/init.py
agent/HANDOFF.md
agent/TASK.md
agent/REPOSITORY_STATE.md

REQUIRED BEHAVIOR

CsvProvider shall support:

.csv extension detection
loading CSV through pandas
empty-file detection
missing/invalid data handling
numeric waveform columns
optional time column
basic digital channel inference
DisturbanceRecord output
CSV FORMAT ASSUMPTION

Initial CSV support shall be pragmatic and simple.

Supported structure:

time, VA, VB, VC, IA, IB, IC, FREQ

or:

timestamp, VA, VB, VC, IA, IB, IC

or:

VA, VB, VC, IA, IB, IC

If no time/timestamp column exists:

create relative sample index time using row order
sampling rate may remain unknown or inferred only if safely possible
TIME COLUMN POLICY

Recognized time columns:

time
t
seconds
sec
timestamp
datetime

Case-insensitive matching is acceptable.

Relative Time

If time column is numeric:

treat as seconds
preserve as relative time column
Timestamp

If timestamp/datetime column is parseable:

parse using pandas datetime parsing
populate start_time from first timestamp
populate trigger_time equal to start_time unless unavailable
No Time Column

If no time column exists:

create time column as sample index in seconds if sampling rate can be inferred
otherwise create time column as integer sample index converted to float

Do not overbuild sampling inference.

CHANNEL CLASSIFICATION POLICY
Analog Channels

Numeric columns shall be treated as analog channels by default.

Examples:

VA
VB
VC
IA
IB
IC
FREQ
MW
MVAR
Digital Channels

Columns may be treated as digital if:

values are only 0/1
boolean type
column name suggests status/trip/pickup/breaker

Examples:

trip
pickup
breaker
status
cb
relay

Digital inference should remain conservative.

If uncertain:

classify numeric columns as analog.
UNIT POLICY

CSV does not provide reliable embedded units.

Initial unit handling:

voltage-like names: kV
current-like names: A
frequency-like names: Hz
power-like names: MW / MVar
digital channels: state
unknown analogs: unknown

Do not overbuild unit inference.

METADATA POPULATION

RecordingMetadata shall be populated as follows:

station_name: "Unknown"
recorder_name: "CSV"
source_file: CSV file path/name
provider_type: "csv"
nominal_frequency: 50.0 unless unavailable

Optional fields may remain None.

SAMPLING INFORMATION

SamplingInformation shall be populated pragmatically.

If numeric time column exists:

estimate sampling rate from median time difference if possible
samples_per_rate = total sample count

If no reliable time exists:

sampling_rates may be empty or [0.0] depending on current model expectations
samples_per_rate = total sample count

Do not perform resampling.

Do not enforce uniform sampling beyond lightweight validation.

TIMING INFORMATION

TimingInformation shall be populated as follows:

If timestamp column exists:

start_time = first timestamp
trigger_time = first timestamp

If only relative time exists:

start_time = current neutral/default datetime or minimum supported fallback
trigger_time = start_time

Keep timezone handling simple.

Do not assume timezone.

DATA NORMALIZATION RULES

The returned DisturbanceRecord shall contain:

waveform_data as pandas DataFrame
normalized time/timestamp column
analog channel columns
digital channel columns
metadata
sampling_info
timing_info

Do not return raw CSV parser objects.

Do not leak pandas parsing options outside provider.

ERROR HANDLING REQUIREMENTS

Raise ProviderLoadError for:

file not found
unsupported extension
empty CSV
no usable waveform columns
malformed CSV
unreadable file
invalid DataFrame construction

Error messages should be clear and actionable.

Avoid silent failures.

PERFORMANCE REQUIREMENTS

CSV provider should use pandas efficiently.

Avoid:

per-row Python parsing
unnecessary DataFrame copying
deep copying waveform data
expensive validation loops

Preferred:

pandas vectorized operations
column-level inference
lightweight validation
ARCHITECTURE RULES
Rule 1 — Provider Isolation

CsvProvider shall not import:

PyQt6
PyQtGraph
OpenGL
visualization modules
Rule 2 — Analytics Isolation

CsvProvider shall not calculate:

RMS
ROCOF
harmonics
phasors
Rule 3 — Legacy Isolation

Do not import from:

src/

Rule 4 — DisturbanceRecord Enforcement

CsvProvider.load() shall return:

DisturbanceRecord

TEST REQUIREMENTS

Create:

tests/unit/test_csv_provider.py

Tests shall cover:

can_load() for .csv
can_load() rejects non-CSV
load valid CSV with numeric time column
load valid CSV with timestamp column
load valid CSV without time column
analog channel inference
digital channel inference
unit inference
sampling rate estimation
empty CSV failure
malformed CSV failure
no usable waveform columns failure
missing file failure
DisturbanceRecord validation passes

Run:

.venv/Scripts/python.exe -m pytest tests/unit/test_csv_provider.py tests/unit/test_provider_manager.py tests/unit/test_disturbance_record.py -v

ENVIRONMENT REQUIREMENT

All execution and testing must use:

.venv/

Do not use:

global Python
system pip
unrelated external environments
IMPLEMENTATION CONSTRAINTS

Do NOT implement:

Excel provider
UI import workflow
waveform rendering
analytics calculations
synchronization logic
resampling
timezone correction
advanced column mapping UI
REQUIRED REPOSITORY TRACKING UPDATES

After implementation, update:

agent/HANDOFF.md
agent/TASK.md
agent/REPOSITORY_STATE.md

Also fix agent/REPOSITORY_STATE.md if it still incorrectly lists COMTRADE parser as the next task.

COMPLETION REPORT REQUIREMENTS

The implementation report shall include:

1. Summary

What was implemented.

2. Files Created / Modified

List all created or modified files.

3. Architecture Alignment

Confirm:

provider isolation
DisturbanceRecord enforcement
analytics independence
visualization independence
legacy isolation
4. Validation / Test Results

Confirm:

tests added
tests executed through .venv
test results
5. Performance Considerations

Explain CSV loading and inference performance.

6. Repository Tracking Updates

Confirm updates to:

agent/HANDOFF.md
agent/TASK.md
agent/REPOSITORY_STATE.md
7. Risks / Concerns

Identify any concern.

8. Next Recommended Step

Suggested continuation.

NEXT EXPECTED DIRECTIVE

After CSV provider:

directives/implement_excel_provider.md

FINAL PRINCIPLE

CSV ingestion is a practical engineering import path.

It must remain:

simple
robust
parser-independent
visualization-independent
lightweight
suitable for waveform viewer testing

Avoid overengineering.
Keep normalization predictable.
Return clean DisturbanceRecord objects.