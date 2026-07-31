DATA_CONTRACT.md — Powerwave Unified Data Contract
PURPOSE

This document defines the unified internal data structure used throughout Powerwave.

All waveform-related systems SHALL communicate using:

DisturbanceRecord

This contract ensures:

parser independence
consistent downstream processing
analytics compatibility
visualization consistency
future scalability

No parser-specific structures shall exist outside provider modules.

CORE DESIGN PRINCIPLE

Powerwave uses a single normalized waveform representation.

All ingestion providers:

COMTRADE
CSV
Excel
future providers

must normalize their outputs into:

DisturbanceRecord

This prevents:

UI/parser coupling
analytics/parser coupling
duplicated waveform handling logic
HIGH-LEVEL STRUCTURE
DisturbanceRecord
├── metadata
├── waveform_data
├── analog_channels
├── digital_channels
├── sampling_information
├── timing_information
└── disturbance_information
PRIMARY OBJECTIVES

The contract must support:

large waveform datasets
synchronized visualization
high-speed analytics
multi-provider ingestion
future extensibility

The structure must remain:

lightweight
scalable
parser-agnostic
analytics-friendly
DISTURBANCERECORD STRUCTURE

Example conceptual structure:

@dataclass
class DisturbanceRecord:
    metadata: RecordingMetadata
    waveform_data: pd.DataFrame
    analog_channels: list[AnalogChannel]
    digital_channels: list[DigitalChannel]
    sampling_info: SamplingInformation
    timing_info: TimingInformation
    disturbance_info: DisturbanceInformation

This structure may evolve incrementally but the architectural intent must remain unchanged.

WAVEFORM DATA
PRIMARY STORAGE

Waveform samples SHALL use:

pandas.DataFrame

Reason:

vectorized operations
analytics compatibility
scalable indexing
mature ecosystem support
REQUIRED CHARACTERISTICS

Waveform data must support:

high-frequency samples
multiple channels
synchronized timestamps
scalable slicing
efficient viewport extraction
RECOMMENDED STRUCTURE
time | VA | VB | VC | IA | IB | IC | FREQ

Each column represents a normalized signal channel.

TIME AXIS REQUIREMENTS

The waveform_data time column SHALL represent the X-axis used for waveform
display and analytics.

Required internal representation:

time

Type:

float64

Meaning:

seconds elapsed along the record's time axis for time-based modes, or sample
number for sample-index mode.

This column is authoritative for rendering. Visualization code SHALL use this
normalized X-axis column and SHALL NOT require one absolute datetime value per
sample.

SOURCE TIME SEMANTICS

Providers and import pipelines SHALL support five time-axis modes:

1. Auto-detected time axis

The import workflow MAY auto-select one of the supported concrete timing modes
when the source column names, values, and sampling pattern make the intent clear.
Auto-detection SHALL be conservative; ambiguous numeric columns SHALL NOT be
silently treated as elapsed time.

2. Absolute timestamp-based recordings

Examples:

2026-03-06 18:04:09.318
2026-03-06 18:04:09.328
COMTRADE start/trigger timestamps

For absolute sources:

start_time is the real first-sample timestamp.
trigger_time is the real trigger timestamp when available.
waveform_data["time"] is seconds elapsed from start_time.

3. Relative elapsed-time recordings

Examples:

-0.002
0.008
0.018
0.028

For relative elapsed sources:

the source duration values are the real engineering time axis.
waveform_data["time"] SHALL preserve those elapsed values after unit conversion
to seconds.
start_time MAY use a synthetic compatibility origin.
trigger_time MAY equal start_time unless a trigger offset is known.
absolute calendar meaning SHALL NOT be inferred from the synthetic origin.

Elapsed source units MAY include:

seconds
milliseconds
minutes

Detection of elapsed-time columns SHALL be conservative. Numeric columns SHALL
only be auto-selected as elapsed time when column naming and monotonicity make
the intent clear enough for engineering use.

4. Synthetic elapsed time from sampling interval

This mode is for sources that have no usable timestamp/duration column but have
a known sample rate or sample interval.

For synthetic elapsed-time sources:

the operator SHALL provide either sample rate or sample interval.
waveform_data["time"] SHALL be generated as elapsed seconds from row order.
the generated axis SHALL be labelled as synthetic elapsed time in diagnostics.
start_time MAY use a synthetic compatibility origin.
absolute calendar meaning SHALL NOT be inferred.

5. Sample-index axis

This mode is for plotting ordered data series without timing metadata.

For sample-index sources:

no timestamp column is required.
no sample rate or interval is required.
waveform_data["time"] MAY carry sample indices for renderer compatibility.
TimingInformation.timing_reference SHALL be "sample_index" or equivalent
metadata SHALL identify the axis as non-time.
time_axis_unit SHALL be "sample" or "index".
visualization SHALL label the X-axis as Sample Index, not Time (s).
sample index values SHALL NOT be used for duration, frequency, event timing, or
cross-record synchronization calculations.

ABSOLUTE TIMESTAMP REQUIREMENTS

Absolute timestamps SHALL:

remain precise
support high-resolution alignment
support cross-record synchronization

Preferred handling:

numpy datetime64
pandas datetime index
python-dateutil compatibility
CHANNEL CONTRACTS
ANALOG CHANNEL

Analog channels represent:

voltage
current
MW
MVar
frequency
analog measurements

Example structure:

@dataclass
class AnalogChannel:
    name: str
    phase: str | None
    unit: str
    scale: float
    offset: float

Analog channel metadata is authoritative for visualization axis grouping.
Display features such as panel merge SHALL preserve AnalogChannel.unit and any
available signal role/type metadata. Merged displays SHALL NOT discard units or
replace them with a mixed arbitrary unit in order to force unrelated channels
onto one Y-axis.

When multiple waveform panels are merged into one canvas, the DisturbanceRecord
contract still requires each analog channel to keep its own name, unit, scale,
offset, and role metadata. Visualization may derive axis groups from these
fields, but it SHALL NOT mutate the source record to make incompatible channels
appear compatible.
DIGITAL CHANNEL

Digital channels represent:

breaker status
relay operation
protection pickup
binary state signals

Example structure:

@dataclass
class DigitalChannel:
    name: str
    normal_state: int
METADATA CONTRACT

Metadata SHALL include:

station name
recorder ID
source filename
nominal frequency
trigger time
timezone
provider type

Example:

@dataclass
class RecordingMetadata:
    station_name: str
    recorder_name: str
    source_file: str
    nominal_frequency: float
SAMPLING INFORMATION

Sampling information SHALL support:

multiple sampling rates
COMTRADE multi-rate structures
waveform alignment

Example:

@dataclass
class SamplingInformation:
    sampling_rates: list[float]
    samples_per_rate: list[int]
TIMING INFORMATION

Timing information SHALL support:

trigger alignment
event timing
synchronized analysis
absolute and relative elapsed source semantics

Example:

@dataclass
class TimingInformation:
    start_time: datetime
    trigger_time: datetime
    time_multiplier: float

TimingInformation SHALL distinguish the meaning of start_time and the X-axis:

absolute
    start_time and trigger_time represent real recording timestamps.

relative_elapsed
    waveform_data["time"] is the authoritative time axis. start_time is only a
    synthetic compatibility anchor unless documented otherwise.

synthetic_elapsed
    waveform_data["time"] is generated from row order and an operator-supplied
    sample rate or interval. start_time is synthetic unless documented
    otherwise.

sample_index
    waveform_data["time"] represents ordered sample indices for display only.
    start_time and trigger_time SHALL NOT be interpreted as real event timing.

Implementations SHALL preserve the source timing mode and original elapsed-time
unit when known, either directly on TimingInformation or through recording
metadata until the TimingInformation schema is extended.
DISTURBANCE INFORMATION

Disturbance context SHALL support:

event classification
disturbance tagging
analysis annotations

Example:

@dataclass
class DisturbanceInformation:
    event_type: str | None
    notes: str | None
NORMALIZATION REQUIREMENTS

All providers SHALL normalize:

Channel Names

Consistent naming conventions.

Example:

VA
VB
VC
IA
IB
IC
Units

Units SHALL remain explicit.

Examples:

kV
A
MW
Hz
Time Axis Format

Absolute, relative elapsed, and synthetic elapsed axes SHALL use unified
seconds-based representation:

waveform_data["time"] as float64 seconds.

Absolute source timestamps SHALL be normalized into start_time/trigger_time plus
relative seconds. Relative elapsed source values SHALL be preserved as relative
seconds and SHALL NOT be displayed as synthetic calendar timestamps.

Sample-index axes SHALL be explicitly marked as non-time. Implementations MAY
store sample indices in waveform_data["time"] for renderer compatibility, but
the axis unit SHALL be "sample" or "index", and visualization SHALL NOT label or
treat the values as seconds.

PROVIDER RESPONSIBILITIES

Each provider SHALL:

parse raw source format
normalize metadata
normalize channels
construct DisturbanceRecord
return parser-independent structure

Providers SHALL NOT:

expose raw parser internals
expose proprietary structures
leak provider-specific logic
ANALYTICS REQUIREMENTS

Analytics systems SHALL:

consume DisturbanceRecord only
remain provider-independent
avoid direct parser interaction

This enables:

analytics portability
easier testing
future provider expansion
VISUALIZATION REQUIREMENTS

Visualization systems SHALL:

consume normalized waveform data
remain parser-agnostic
avoid file-format assumptions

Visualization SHALL NOT:

know parser internals
depend on provider logic
MEMORY MANAGEMENT PRINCIPLES

The contract must remain efficient for:

large COMTRADE files
high-frequency recordings
multi-channel datasets

Avoid:

duplicated waveform storage
unnecessary conversions
redundant arrays

Preferred:

shared references
vectorized operations
lazy processing where possible
FUTURE EXTENSIBILITY

The contract shall support future:

PMU data
live streaming
database-backed records
synchronized multi-record analysis
AI analytics annotations

without major redesign.

CONTRACT RULES
RULE 1 — SINGLE INTERNAL FORMAT

All waveform data SHALL use:

DisturbanceRecord
RULE 2 — PARSER ISOLATION

Parser-specific structures SHALL remain inside provider modules.

RULE 3 — ANALYTICS ISOLATION

Analytics systems SHALL remain provider-independent.

RULE 4 — VISUALIZATION ISOLATION

Visualization systems SHALL remain parser-independent.

Visualization systems SHALL also remain unit-safe. Canvas merge, overlay, and
panel layout features must use the normalized channel metadata in
DisturbanceRecord and SHALL NOT collapse channels with different engineering
units or signal roles onto one shared Y-axis.

RULE 5 — PERFORMANCE FIRST

The contract SHALL support:

scalable slicing
low-latency rendering
vectorized analytics
FINAL PRINCIPLE

DisturbanceRecord is the foundation contract of Powerwave.

All major systems depend on:

its stability
consistency
scalability
parser independence

Protect the contract carefully.
Evolve incrementally.
Avoid uncontrolled structural expansion.
