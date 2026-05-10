implement_disturbance_record.md — Powerwave DisturbanceRecord Implementation Directive
PURPOSE

This directive implements the first core internal data contract for Powerwave:

DisturbanceRecord

This is the foundational model used by:

providers
parsers
analytics
visualization
synchronization
future benchmarking

The objective is to create a clean, typed, parser-independent waveform container.

This directive does NOT implement:

COMTRADE parsing
CSV parsing
Excel parsing
waveform rendering
analytics calculations
synchronization logic

Only the data contract foundation.

IMPLEMENTATION OBJECTIVES

The implementation shall:

create the DisturbanceRecord model
create supporting metadata models
create channel models
create timing and sampling models
provide lightweight validation helpers
remain parser-independent
remain visualization-independent
support future high-performance workflows
TARGET LOCATION

Implement inside:

app/models/

Recommended files:

app/models/disturbance_record.py
app/models/channels.py
app/models/metadata.py
app/models/timing.py
app/models/init.py

Keep implementation simple and clear.

REQUIRED MODEL STRUCTURE
1. DisturbanceRecord

DisturbanceRecord shall represent one normalized disturbance recording.

It shall contain:

metadata
waveform_data
analog_channels
digital_channels
sampling_info
timing_info
disturbance_info

Conceptual structure:

@dataclass(slots=True)
class DisturbanceRecord:
    metadata: RecordingMetadata
    waveform_data: pd.DataFrame
    analog_channels: list[AnalogChannel]
    digital_channels: list[DigitalChannel]
    sampling_info: SamplingInformation
    timing_info: TimingInformation
    disturbance_info: DisturbanceInformation | None = None
2. RecordingMetadata

RecordingMetadata shall contain recording-level information.

Required fields:

station_name
recorder_name
source_file
provider_type
nominal_frequency

Recommended optional fields:

device_id
location
timezone
comments
3. AnalogChannel

AnalogChannel shall describe one analog waveform channel.

Required fields:

name
unit
index

Recommended optional fields:

phase
description
scale
offset
primary_ratio
secondary_ratio

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
4. DigitalChannel

DigitalChannel shall describe one digital/binary channel.

Required fields:

name
index

Recommended optional fields:

normal_state
description

Examples:

breaker status
relay trip
protection pickup
autoreclose status
5. SamplingInformation

SamplingInformation shall describe waveform sampling.

Required fields:

sampling_rates
samples_per_rate

Recommended optional fields:

samples_per_cycle
nominal_frequency

Must support future COMTRADE multi-rate sampling.

6. TimingInformation

TimingInformation shall describe time alignment.

Required fields:

start_time
trigger_time

Recommended optional fields:

time_multiplier
timezone

Use python datetime types.

Must remain compatible with python-dateutil.

7. DisturbanceInformation

DisturbanceInformation shall describe optional event context.

Recommended fields:

event_type
notes
tags

This is optional and may be expanded later.

WAVEFORM DATA REQUIREMENTS

DisturbanceRecord shall store waveform samples using:

pandas.DataFrame

The DataFrame shall:

contain one row per sample
contain one column per signal
support timestamp or relative time alignment
avoid parser-specific structures
avoid duplicated waveform copies

Recommended columns:

time or timestamp
analog channel columns
digital channel columns

Example:

time | VA | VB | VC | IA | IB | IC | FREQ

VALIDATION REQUIREMENTS

Implement lightweight validation methods.

DisturbanceRecord should support:

validate()
channel_names()
analog_channel_names()
digital_channel_names()
has_channel(name)
sample_count()
duration_seconds()

Validation should check:

waveform_data is a pandas DataFrame
waveform_data is not empty
analog channel names exist in DataFrame columns where applicable
digital channel names exist in DataFrame columns where applicable
timing information is valid
sampling information is valid

Do not overbuild validation.

Keep validation lightweight.

PERFORMANCE REQUIREMENTS

The model must remain:

lightweight
low overhead
suitable for large DataFrames
non-copying where possible

Avoid:

automatic deep copying of waveform_data
per-sample validation loops
expensive validation on construction
heavy computed properties

Use:

dataclass(slots=True) where practical
direct references to DataFrame
vectorized assumptions
ARCHITECTURE RULES
Rule 1 — Parser Independence

DisturbanceRecord shall not import or depend on:

COMTRADE parser
CSV parser
Excel parser
provider manager
Rule 2 — Visualization Independence

DisturbanceRecord shall not import or depend on:

PyQt6
PyQtGraph
OpenGL
visualization widgets
Rule 3 — Analytics Independence

DisturbanceRecord shall not import or depend on:

RMS engines
ROCOF engines
harmonic modules

Analytics consume DisturbanceRecord, not the other way around.

Rule 4 — Legacy Isolation

Do not import from:

src/

Legacy code may be inspected only as reference.

No direct dependency from app/models/ to src/ is allowed.

TEST REQUIREMENTS

Create unit tests under:

tests/unit/test_disturbance_record.py

Tests shall cover:

creating a valid DisturbanceRecord
retrieving channel names
checking channel existence
sample_count()
duration_seconds()
validation passes for valid data
validation fails for empty waveform_data
validation fails if declared channel missing from DataFrame

Keep tests focused and lightweight.

ENVIRONMENT REQUIREMENT

All Python execution, testing, and dependency usage must use:

.venv/

Do not use:

system Python
global pip
unrelated external environments

Testing should be run through the repository virtual environment.

IMPLEMENTATION CONSTRAINTS

Do NOT implement:

parser logic
provider logic
waveform rendering
analytics calculations
synchronization logic
UI logic

Only implement data models and tests.

EXPECTED OUTPUT

After implementation, the repository should contain:

app/models/disturbance_record.py
app/models/channels.py
app/models/metadata.py
app/models/timing.py
tests/unit/test_disturbance_record.py

Updated:

app/models/init.py
agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md

COMPLETION REPORT REQUIREMENTS

The implementation report shall include:

1. Summary

What was implemented.

2. Files Created / Modified

List all created or modified files.

3. Architecture Alignment

Confirm:

parser independence
visualization independence
analytics independence
legacy isolation
4. Validation

Confirm:

tests added
tests executed through .venv
test results
5. Repository Tracking Updates

Confirm updates to:

agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md
6. Risks / Concerns

Identify any concern.

7. Next Recommended Step

Suggested continuation.

NEXT EXPECTED DIRECTIVE

After this directive:

directives/implement_provider_system.md

FINAL PRINCIPLE

DisturbanceRecord is the core internal data contract of Powerwave.

It must remain:

simple
stable
parser-independent
visualization-independent
analytics-friendly
scalable for large waveform data

Protect this contract carefully.