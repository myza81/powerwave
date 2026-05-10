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
timestamp | VA | VB | VC | IA | IB | IC | FREQ

Each column represents a normalized signal channel.

TIMESTAMP REQUIREMENTS

Timestamps SHALL:

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

Example:

@dataclass
class TimingInformation:
    start_time: datetime
    trigger_time: datetime
    time_multiplier: float
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
Timestamp Format

All timestamps SHALL use unified internal formatting.

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