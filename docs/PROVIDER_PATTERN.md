PROVIDER_PATTERN.md — Powerwave Data Provider Architecture
PURPOSE

This document defines the provider-pattern architecture used for waveform ingestion in Powerwave.

The provider system is responsible for:

file ingestion
parser abstraction
waveform normalization
metadata extraction
DisturbanceRecord creation

The architecture is designed to:

isolate parsing logic
prevent UI/parser coupling
support future extensibility
enable plugin-capable ingestion
CORE ARCHITECTURE PRINCIPLE

All waveform sources SHALL be abstracted through a unified provider interface.

The UI, analytics engine, and visualization systems SHALL NEVER interact directly with parser-specific logic.

All ingestion providers must return:

DisturbanceRecord

This ensures:

parser independence
scalable analytics
visualization consistency
future extensibility
ARCHITECTURE OVERVIEW
Input File
    ↓
Provider Detection
    ↓
Concrete Provider
    ↓
Normalization
    ↓
DisturbanceRecord
    ↓
Application Systems
PROVIDER RESPONSIBILITIES

Providers are responsible for:

file parsing
metadata extraction
timestamp normalization
channel normalization
waveform extraction
DisturbanceRecord construction

Providers SHALL NOT:

manipulate UI
perform rendering
perform heavy analytics
know application state
PROVIDER INTERFACE

All providers SHALL inherit from:

BaseProvider

Example conceptual structure:

class BaseProvider(ABC):

    @abstractmethod
    def load(self, path: str) -> DisturbanceRecord:
        pass

    @abstractmethod
    def can_load(self, path: str) -> bool:
        pass
REQUIRED PROVIDERS

Initial providers:

1. COMTRADE Provider

Responsible for:

CFG parsing
DAT parsing
analog extraction
digital extraction
multi-rate handling

Output:

normalized DisturbanceRecord
2. CSV Provider

Responsible for:

generic CSV ingestion
column mapping
timestamp interpretation

Output:

normalized DisturbanceRecord
3. Excel Provider

Responsible for:

Excel ingestion
worksheet selection
waveform extraction

Output:

normalized DisturbanceRecord
PROVIDER DETECTION FLOW

Example conceptual flow:

File Selected
    ↓
Provider Manager
    ↓
Find Compatible Provider
    ↓
Execute Provider.load()
    ↓
Return DisturbanceRecord
PROVIDER MANAGER

The provider manager is responsible for:

provider registration
provider discovery
provider selection

Example conceptual structure:

class ProviderManager:

    def register(self, provider):
        pass

    def load(self, path) -> DisturbanceRecord:
        pass
PROVIDER REGISTRATION

Providers should support scalable registration.

Preferred architecture:

centralized provider registry
automatic provider discovery
plugin-capable expansion
NORMALIZATION REQUIREMENTS

All providers SHALL normalize:

Channel Names

Examples:

VA
VB
VC
IA
IB
IC

Avoid:

provider-specific naming leakage
Timestamp Structure

All timestamps SHALL:

use unified formatting
support high-resolution precision
support synchronized analysis
Units

All units SHALL remain explicit.

Examples:

kV
A
Hz
MW
MVar
COMTRADE PROVIDER REQUIREMENTS

The COMTRADE provider must support:

IEEE C37.111
multi-rate sampling
analog channels
digital channels
timestamp alignment
trigger information
scaling factors

The provider must remain:

memory efficient
scalable
reliable for large files
CSV PROVIDER REQUIREMENTS

CSV provider shall support:

configurable timestamp column
configurable delimiter
configurable encoding
flexible channel mapping

The provider must tolerate:

imperfect datasets
missing columns
engineering-user workflows
EXCEL PROVIDER REQUIREMENTS

Excel provider shall support:

multiple worksheets
configurable sheet selection
waveform extraction
scalable loading

Preferred engine:

Openpyxl
ERROR HANDLING REQUIREMENTS

Providers SHALL:

validate input files
detect malformed structures
provide meaningful errors
fail gracefully

Avoid:

silent parsing failures
partial invalid structures
inconsistent outputs
PERFORMANCE REQUIREMENTS

Providers must support:

large COMTRADE files (>100MB)
scalable waveform loading
responsive UI workflows

Preferred strategies:

lazy loading where possible
vectorized parsing
minimized memory duplication

Avoid:

unnecessary waveform copying
blocking UI thread
repeated conversions
THREADING REQUIREMENTS

Heavy parsing SHALL NOT block UI thread.

Preferred mechanisms:

QThread
worker architecture
asynchronous loading patterns
PROVIDER ISOLATION RULES
RULE 1 — PROVIDERS SHALL RETURN ONLY NORMALIZED DATA

Providers SHALL return:

DisturbanceRecord

Providers SHALL NOT expose:

raw parser internals
provider-specific structures
RULE 2 — PROVIDERS SHALL REMAIN UI-INDEPENDENT

Providers SHALL NOT:

know widgets
know rendering systems
know application state
RULE 3 — PROVIDERS SHALL REMAIN ANALYTICS-INDEPENDENT

Providers SHALL NOT:

calculate RMS
calculate phasors
perform signal analytics

Those responsibilities belong to:

analytics engine
PROVIDER DIRECTORY STRUCTURE

Recommended structure:

app/providers/
│
├── base/
│   ├── base_provider.py
│   └── provider_manager.py
│
├── comtrade/
│   └── comtrade_provider.py
│
├── csv/
│   └── csv_provider.py
│
└── excel/
    └── excel_provider.py
FUTURE EXTENSIBILITY

The provider architecture shall support future:

PMU streams
proprietary recorder formats
database-backed ingestion
live acquisition
cloud-based sources

without redesigning:

visualization
analytics
UI systems
TESTING REQUIREMENTS

Provider testing SHALL include:

malformed files
timestamp validation
channel consistency
scaling correctness
large-file handling
multi-rate validation
FINAL PRINCIPLE

The provider system is the ingestion foundation of Powerwave.

Its responsibilities are:

isolate parsing complexity
normalize waveform data
preserve scalability
protect downstream systems from parser-specific logic

All providers must remain:

modular
scalable
performant
isolated

Protect the ingestion boundary carefully.