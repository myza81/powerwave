VISUALIZATION_CONTRACT.md — Powerwave Visualization Architecture Contract
PURPOSE

This document defines the visualization architecture and rendering contracts used throughout Powerwave.

The visualization engine is the core user-facing subsystem of Powerwave.

It is responsible for:

high-performance waveform rendering
synchronized multi-pane visualization
low-latency interaction
scalable waveform navigation
engineering-grade waveform inspection

This document defines:

rendering architecture
visualization responsibilities
synchronization contracts
interaction rules
performance requirements
CORE VISUALIZATION PHILOSOPHY

Powerwave is a visualization-centric engineering platform.

The rendering engine must prioritize:

responsiveness
scalability
synchronization stability
rendering efficiency
engineering usability

The system is expected to handle:

large disturbance records
high-frequency waveform data
simultaneous multi-channel rendering
real-time interaction

without UI lag.

VISUALIZATION ARCHITECTURE
DisturbanceRecord
        ↓
Visualization Manager
        ↓
┌─────────────────────────┐  ┌──────────────────────────┐
│  FlexiblePlotCanvas     │  │  DigitalEventTimeline     │
│  (N-Axis analog canvas) │  │  (digital state tracks)   │
└─────────────────────────┘  └──────────────────────────┘
        ↓                              ↓
PyQtGraph Rendering Engine  (shared X-axis domain)
        ↓
OpenGL Rendering Pipeline

X-AXIS CONTRACT

Visualization consumes DisturbanceRecord.waveform_data["time"] as the
authoritative X-axis.

Supported representations:

float64 seconds for absolute, relative elapsed, and synthetic elapsed axes
float64 sample indices for sample-index axes

Supported source semantics:

absolute timestamp source
    The time column is seconds elapsed from the real first-sample timestamp.

relative elapsed-time source
    The time column is the source duration axis converted to seconds. The
    display SHALL show elapsed time values, not a synthetic calendar timestamp.

synthetic elapsed source
    The time column is generated from row order and an operator-provided sample
    rate or interval. The display SHALL label it as elapsed time and indicate
    synthetic timing in cursor/diagnostic wording when available.

sample-index source
    The X-axis is ordered sample number only. The display SHALL label it as
    Sample Index, not Time (s). Sample-index axes SHALL NOT be used for duration
    measurement, frequency inference, event timing, or cross-record time
    synchronization.

The visualization layer SHALL remain parser-agnostic. It may inspect normalized
TimingInformation or metadata to choose axis labels and cursor readout wording,
but it SHALL NOT re-parse source files or infer file-format-specific timing
rules.

PRIMARY VISUALIZATION COMPONENTS

Analog rendering widget:

FlexiblePlotCanvas

Inheritance:

pyqtgraph.GraphicsLayoutWidget
    └── FlexiblePlotCanvas

Analog axis manager:

MultiAxisManager (helper class, not a widget)

Digital rendering widget (Phase 3B):

DigitalEventTimeline (separate widget, below FlexiblePlotCanvas)

N-AXIS SINGLE CANVAS ARCHITECTURE

Powerwave uses SIGRA-style N-Axis Single Canvas visualization for analog signals.

Architecture mandates:

One shared X-axis across all parameters in a record
One independent ViewBox per analog parameter
One color-coded AxisItem per parameter
N ViewBoxes = N independent Y-axis scales
Unlimited analog parameters on a single canvas

This provides:

True independent Y-axis scaling per parameter
No fixed stacking layout (all parameters co-exist on one canvas)
Engineering-grade multi-parameter comparison at a glance
Shared X-axis navigation (all parameters zoom/pan together)

Axis behavior:

Axes may appear on left or right canvas margins
Axes are color-coded to match their waveform
Axes are dynamically added and removed with parameters
Margins grow automatically as axes are added

Implementation:

Primary PlotItem hosts the X-axis and first parameter
Additional parameters use pg.ViewBox() linked via setXLink(primary_plot)
Geometry synchronized on sigResized (see VIEWPORT_RENDERING_POLICY §16)
Procedural axis generation — axes are not pre-allocated

MERGED CANVAS CONTRACT

Powerwave may allow users to merge multiple waveform panels into one visual
canvas for comparison. A merge operation SHALL mean:

one shared X-axis canvas
multiple independent Y-axis groups

A merge operation SHALL NOT mean collapsing every waveform onto one shared
Y-axis. Different engineering units, signal roles, and scales must remain
independent.

Examples:

Power (MW) + Frequency (Hz)
    One canvas, shared X-axis.
    Power remains on a MW Y-axis.
    Frequency remains on a Hz Y-axis.

ROCOF (Hz/s) + Frequency (Hz)
    One canvas, shared X-axis.
    ROCOF remains on a Hz/s Y-axis.
    Frequency remains on a Hz Y-axis.

Voltage (kV) + Current (A)
    One canvas, shared X-axis.
    Voltage and current remain on separate Y-axes.

Merge guardrails:

The selected panels SHALL share a compatible X-axis before merge. If the time
or sample-index vectors differ and cannot be safely aligned, the UI SHALL warn
or block the merge.

The visualization layer SHALL preserve channel unit and signal role metadata
when constructing a merged canvas.

Axis grouping in a merged canvas SHALL be based on engineering meaning, not on
panel origin. At minimum, the grouping key must include normalized signal type
and unit. Same-unit but different-role channels SHOULD NOT be silently forced
onto one scale without a clear user override.

If a merge would produce too many independent Y-axes to read comfortably, the UI
SHOULD warn the user before proceeding.

DIGITAL EVENT TIMELINE

Digital channels (breaker status, relay trips, pickups, alarms) are rendered
in a SEPARATE component: DigitalEventTimeline.

Architecture mandates:

Digital signals are NOT rendered in FlexiblePlotCanvas
Each digital channel occupies one fixed-height horizontal track
Binary state display: high/low with color fill
No Y-axis, no independent scaling
Shares the same time X-axis as FlexiblePlotCanvas (X-linked or driven)

Phase scope:
FlexiblePlotCanvas: Phase 3A
DigitalEventTimeline: Phase 3B

FLEXIBLEPLOTCANVAS RESPONSIBILITIES

FlexiblePlotCanvas SHALL:

render analog waveform channels
manage N independent Y-axes (one ViewBox per parameter)
support synchronized X-axis navigation (setXLink)
support viewport clipping and display decimation
host the master time cursor and trigger line
support low-latency viewport updates
support color-coded axes matching waveform colors
support dynamic parameter add/remove

FlexiblePlotCanvas SHALL NOT:

render digital signals (DigitalEventTimeline responsibility)
parse files
perform heavy analytics
contain provider logic
REQUIRED RENDERING FEATURES
1. OPENGL ACCELERATION

Rendering SHALL support:

useOpenGL=True

Purpose:

GPU-assisted rendering
scalable waveform display
reduced rendering bottlenecks
2. CLIP-TO-VIEW

Rendering SHALL support:

viewport-based rendering
visible-region-only plotting

Purpose:

reduce unnecessary rendering
improve large-waveform performance
3. DOWNSAMPLING

Rendering SHALL support:

adaptive downsampling
viewport-aware sample reduction

Purpose:

maintain UI responsiveness
prevent rendering overload
4. INCREMENTAL REDRAW

Rendering SHALL minimize:

full redraw operations
unnecessary scene refreshes

Preferred:

localized updates
incremental rendering
MULTI-PANE VISUALIZATION

Powerwave SHALL support:

multiple synchronized waveform panes
shared time axis
coordinated interaction

Examples:

voltage pane
current pane
frequency pane
MW/MVar pane
digital signal pane
SYNCHRONIZATION CONTRACT

All waveform panes SHALL support synchronized:

zoom
pan
cursor movement
viewport updates

Synchronization SHALL remain:

low latency
stable
scalable
MASTER TIME CURSOR

The visualization engine SHALL support:

shared InfiniteLine cursor
synchronized cross-pane movement

Behavior:

dragging cursor in one pane updates all panes
all visible waveforms align to same timestamp
relative elapsed-time recordings align to the same elapsed-time value

Purpose:

engineering-grade disturbance investigation
VISUALIZATION MANAGER

A centralized visualization manager SHALL coordinate:

pane registration
synchronization
cursor coordination
shared viewport state

Conceptual structure:

class VisualizationManager:
    pass
WAVEFORM LAYERING

Visualization SHALL support:

multi-channel overlay
analog layering
digital signal visualization

The system must support:

scalable overlay management
engineering-friendly waveform comparison
AXIS CONTRACTS
X-AXIS

The X-axis SHALL represent:

elapsed seconds for time-based records, or sample index for sequence-only
records

All panes SHALL:

share X-axis alignment
maintain synchronized navigation
Y-AXIS

Each pane SHALL support:

independent engineering units
scalable vertical zoom
configurable scaling

Merged canvases SHALL preserve this rule. Combining panels SHALL NOT combine
their Y-axis scales unless the signals are explicitly compatible by signal type
and unit.

Examples:

kV
A
Hz
MW
pu
INTERACTION CONTRACTS

The visualization engine SHALL support:

zoom
pan
cursor movement
waveform inspection
channel visibility control

Future support:

measurements
annotations
event markers
engineering notes
DATA CONTRACT REQUIREMENT

Visualization SHALL consume ONLY:

DisturbanceRecord

Visualization SHALL NOT:

know parser internals
depend on provider structures

This ensures:

parser independence
visualization stability
analytics portability
PERFORMANCE REQUIREMENTS
TARGET PERFORMANCE

The visualization engine must support:

large COMTRADE files
high sample-rate rendering
responsive interaction
synchronized multi-pane navigation

Expected workload:

128+ samples/cycle
multi-second recordings
multiple simultaneous channels
MEMORY REQUIREMENTS

Visualization SHALL minimize:

waveform duplication
unnecessary array copies
redundant rendering buffers

Preferred:

shared references
viewport extraction
lightweight redraw operations
UI THREAD REQUIREMENTS

Heavy visualization operations SHALL NOT:

block Qt event loop
freeze UI interaction

Preferred:

asynchronous preprocessing
incremental rendering updates
DIGITAL SIGNAL VISUALIZATION

The engine SHALL support:

breaker status visualization
relay pickup indication
binary state display

Digital visualization shall remain:

synchronized with analog waveforms
scalable for many channels
FUTURE VISUALIZATION EXTENSIBILITY

The architecture shall support future:

phasor overlays
impedance trajectories
harmonic visualization
AI-generated annotations
multi-record comparison
live streaming visualization

without redesigning:

rendering core
synchronization engine
VISUALIZATION DIRECTORY STRUCTURE

app/visualization/
│
├── widgets/
│   ├── flexible_plot_canvas.py        ← Phase 3A — N-Axis analog canvas
│   └── digital_event_timeline.py     ← Phase 3B — digital state tracks
│
├── managers/
│   ├── multi_axis_manager.py          ← Phase 3A — ViewBox/axis lifecycle
│   ├── visualization_manager.py       ← Phase 3B — wires canvas + timeline
│   └── synchronization_manager.py     ← Phase 3B — cross-widget X-sync
│
├── rendering/
│   └── downsampling.py                ← Phase 3A — decimate_for_display()
│
└── interaction/
    ├── cursor_manager.py              ← Phase 3B
    └── viewport_controller.py        ← Phase 3B
VISUALIZATION RULES
RULE 1 — PERFORMANCE FIRST

All rendering decisions SHALL prioritize:

responsiveness
low redraw latency
scalable rendering
RULE 2 — STRICT ISOLATION

Visualization SHALL remain:

parser-independent
analytics-independent
RULE 3 — SYNCHRONIZATION IS MANDATORY

All panes SHALL support:

synchronized navigation
synchronized cursor movement
RULE 4 — ENGINEERING USABILITY FIRST

The visualization engine is built for:

protection engineers
disturbance analysts
operational engineering workflows

Interaction behavior must support:

rapid disturbance investigation
waveform comparison
engineering analysis
FINAL PRINCIPLE

The visualization engine is the heart of Powerwave.

Its architecture must prioritize:

rendering performance
synchronization stability
scalability
engineering usability

Build for industrial-scale waveform analysis.
Protect rendering efficiency carefully.
Optimize interaction continuously.

IMPLEMENTATION REFERENCE

For low-level rendering implementation rules (PyQtGraph initialization,
curve lifecycle, decimation policy, cursor/trigger patterns, color scheme,
UI-thread protection, DisturbanceRecord field access, anti-patterns):

  docs/VIEWPORT_RENDERING_POLICY.md

This document defines HOW to implement the contracts specified here.
VISUALIZATION_CONTRACT.md defines WHAT. VIEWPORT_RENDERING_POLICY.md defines HOW.
