ARCHITECTURE.md — Powerwave System Architecture
PURPOSE

This document defines the core technical architecture of Powerwave.

It specifies:

subsystem boundaries
repository organization
data flow
module responsibilities
architectural contracts
implementation constraints

This document is the primary technical reference for all implementation agents.

ARCHITECTURAL PHILOSOPHY

Powerwave follows these core principles:

Performance-first engineering
Strict modularity
Contract-based data flow
Incremental scalability
Visualization-centric architecture
Parser-independent ingestion
Future analytics extensibility
HIGH-LEVEL ARCHITECTURE
┌──────────────────────┐
│      UI LAYER        │
│ PyQt6 / PyQtGraph    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ VISUALIZATION ENGINE │
│ Rendering / Sync     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ANALYTICS ENGINE    │
│ Signal Processing    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ DATA CONTRACT LAYER  │
│ DisturbanceRecord    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ DATA PROVIDERS       │
│ COMTRADE / CSV / XLS │
└──────────────────────┘
REPOSITORY STRUCTURE
powerwave/
│
├── agent/
├── directives/
├── docs/
│
├── app/
│   ├── main.py
│   ├── config/
│   ├── ui/
│   ├── visualization/
│   ├── analytics/
│   ├── providers/
│   ├── models/
│   ├── synchronization/
│   ├── services/
│   └── utils/
│
├── tests/
│
├── benchmarks/
│
└── resources/
SUBSYSTEM ARCHITECTURE
1. UI LAYER

Location:

app/ui/

Responsibilities:

main window
menus/toolbars
docking system
workspace management
interaction controls

Primary technology:

PyQt6

The UI layer SHALL NOT:

parse files
perform analytics
directly manipulate waveform computation
2. VISUALIZATION ENGINE

Location:

app/visualization/

Responsibilities:

waveform rendering
plot synchronization
OpenGL rendering
viewport optimization
master cursor interaction

Primary technology:

PyQtGraph
PyOpenGL

Core components:

FastWaveformWidget
Plot synchronization manager
Cursor synchronization engine

Key requirements:

clip-to-view rendering
downsampling
scalable interaction
low redraw latency
3. ANALYTICS ENGINE

Location:

app/analytics/

Responsibilities:

RMS calculations
frequency analysis
ROCOF calculations
harmonic analysis
phasor calculations
impedance calculations

Primary technology:

NumPy
SciPy
Pandas

The analytics layer SHALL:

operate independently from UI
remain parser-agnostic
consume DisturbanceRecord only
4. DATA PROVIDER LAYER

Location:

app/providers/

Responsibilities:

file ingestion
parser abstraction
normalization
metadata extraction

Supported providers:

COMTRADE
CSV
Excel

Architecture pattern:

provider pattern
plugin-capable parser system

Each provider SHALL:

normalize output
return DisturbanceRecord
remain isolated from UI systems
5. DATA MODEL LAYER

Location:

app/models/

Responsibilities:

unified waveform contracts
metadata structures
normalized internal representations

Primary model:

DisturbanceRecord

DisturbanceRecord SHALL contain:

metadata
timestamps
waveform DataFrame
channel definitions
disturbance context

This is the single internal waveform contract.

6. SYNCHRONIZATION ENGINE

Location:

app/synchronization/

Responsibilities:

synchronized X-axis behavior
shared viewport state
master time cursor
synchronized zoom/pan

Key requirements:

low interaction latency
stable synchronization
scalable multi-pane support
7. SERVICES LAYER

Location:

app/services/

Responsibilities:

orchestration logic
application coordination
workflow management
service abstraction

Examples:

file loading service
waveform registry
session management
CORE DATA FLOW
Input File
    ↓
Provider Parser
    ↓
Normalization
    ↓
DisturbanceRecord
    ↓
Analytics Engine
    ↓
Visualization Engine
    ↓
User Interaction
DISTURBANCERECORD CONTRACT

The unified waveform structure SHALL include:

Metadata

Examples:

station name
recording device
trigger time
sampling rate
frequency
Time-Series Data

Primary storage:

pandas DataFrame
Channel Definitions

Examples:

analog channels
digital channels
engineering units
scaling information
Time Alignment

Requirements:

precise timestamp handling
relative elapsed-time handling for duration-based waveform records
cross-source synchronization capability
PROVIDER PATTERN ARCHITECTURE

All parsers SHALL inherit from a common provider interface.

Example structure:

class BaseProvider:
    def load(self, path) -> DisturbanceRecord:
        pass

Concrete providers:

ComtradeProvider
CsvProvider
ExcelProvider

Benefits:

parser independence
plugin extensibility
isolated ingestion logic
VISUALIZATION ARCHITECTURE

Core rendering component:

FastWaveformWidget

Inheritance:

pyqtgraph.PlotWidget
    └── FastWaveformWidget

Required optimizations:

OpenGL acceleration
clip-to-view
downsampling
incremental redraw
SYNCHRONIZATION ARCHITECTURE

Synchronization shall use:

centralized synchronization manager
shared viewport coordination
shared InfiniteLine cursor

Required synchronization:

zoom
pan
cursor movement
PERFORMANCE ARCHITECTURE
REQUIRED OPTIMIZATION STRATEGIES
Rendering
clip-to-view
downsampling
OpenGL acceleration
Computation
vectorized NumPy operations
minimized memory copies
Interaction
incremental updates
minimized redraw scope
THREADING STRATEGY

Heavy operations SHALL NOT block UI thread.

Examples:

COMTRADE loading
waveform preprocessing
analytics computation

Preferred mechanisms:

QThread
worker patterns
asynchronous task execution
TESTING ARCHITECTURE

Location:

tests/

Testing categories:

parser validation
synchronization validation
rendering stability
analytics correctness
performance benchmarks
BENCHMARKING ARCHITECTURE

Location:

benchmarks/

Benchmarks shall measure:

file loading time
rendering latency
synchronization responsiveness
memory usage
interaction smoothness
FUTURE EXTENSIBILITY

Architecture shall support future:

PMU integration
live streaming
AI-assisted analytics
disturbance databases
waveform clustering
relay inference engines

without major redesign.

ARCHITECTURAL RULES
RULE 1 — STRICT MODULARITY

Subsystems SHALL remain isolated.

RULE 2 — SINGLE DATA CONTRACT

All waveform data SHALL use:

DisturbanceRecord
RULE 3 — PERFORMANCE FIRST

All implementation decisions SHALL prioritize:

rendering speed
responsiveness
memory efficiency
RULE 4 — PHASED DEVELOPMENT

The system SHALL evolve incrementally.

Avoid:

speculative abstractions
premature complexity
uncontrolled expansion
FINAL PRINCIPLE

Powerwave architecture is designed for:

industrial-scale disturbance analysis
scalable waveform rendering
future analytics growth
long-term maintainability

Protect architecture integrity.
Optimize continuously.
Build incrementally.
