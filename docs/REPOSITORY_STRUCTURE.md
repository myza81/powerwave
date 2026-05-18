# REPOSITORY_STRUCTURE.md

# Powerwave Repository Structure Contract

Version: 1.0  
Status: AUTHORITATIVE  
Scope: Entire repository structure, subsystem ownership, architectural boundaries, lifecycle responsibility, and future expansion zones.

---

# 1. PURPOSE

This document defines the canonical repository topology for Powerwave.

Its goals are to:

- prevent architectural drift
- prevent duplicate subsystem creation
- maintain clean separation of concerns
- preserve long-term maintainability
- define subsystem ownership boundaries
- guide future feature expansion
- ensure consistent agent implementation behavior

This document is an architectural contract.

All contributors and agents must follow this structure unless an intentional architectural migration is approved and documented.

---

# 2. HIGH-LEVEL ARCHITECTURE

Powerwave is divided into the following major layers:

┌─────────────────────────────────────┐
│ UI Layer                            │
├─────────────────────────────────────┤
│ Visualization Layer                 │
├─────────────────────────────────────┤
│ Analytics Layer                     │
├─────────────────────────────────────┤
│ Data & Session Layer                │
├─────────────────────────────────────┤
│ Provider / Ingestion Layer          │
├─────────────────────────────────────┤
│ Infrastructure / Utilities Layer    │
└─────────────────────────────────────┘

Each layer has strict responsibilities.

Cross-layer leakage should be avoided.

3. TOP-LEVEL REPOSITORY STRUCTURE
powerwave/
│
├── app/
├── docs/
├── tests/
├── samples/
├── tools/
├── agent/
├── scripts/
├── resources/
└── requirements/

4. APPLICATION LAYER STRUCTURE (app/)
app/
│
├── analytics/
├── providers/
├── visualization/
├── ui/
├── data/
├── sessions/
├── synchronization/
├── infrastructure/
├── configuration/
└── utilities/

5. ANALYTICS LAYER (app/analytics/)

Purpose:
All engineering calculations and signal analysis.

Analytics modules must:

remain UI-independent
avoid direct PyQt dependencies
return structured models/results
support caching and vectorization
preserve deterministic behavior

Structure
app/analytics/
│
├── phasors/
├── harmonics/
├── frequency/
├── transients/
├── events/
├── impedance/
├── power_quality/
└── common/

5.1 PHASORS (analytics/phasors/)

Responsibilities:

sliding DFT
phasor extraction
sequence components
phasor classification
phasor caching
phasor registry/state

Must NOT:

render GUI elements
own visualization logic
5.2 HARMONICS (analytics/harmonics/)

Responsibilities:

FFT extraction
harmonic orders
THD/TDD computation
harmonic classification
harmonic caching
harmonic registry/state

Future:

interharmonics
resonance metrics
harmonic event analytics

Must remain rendering-independent.

5.3 FREQUENCY (analytics/frequency/)

Responsibilities:

frequency extraction
ROCOF
frequency classification
frequency overlays support models
5.4 TRANSIENTS (analytics/transients/)

Reserved for:

transient event extraction
wavefront analysis
traveling-wave logic
transient energy analytics
5.5 EVENTS (analytics/events/)

Reserved for:

breaker operations
fault inception detection
relay operation correlation
sequence-of-events analytics
5.6 IMPEDANCE (analytics/impedance/)

Reserved for:

R-X trajectories
impedance loci
distance protection analytics
5.7 COMMON (analytics/common/)

Shared:

vector math
signal utilities
reusable engineering helpers
reusable models

Must avoid subsystem coupling.

6. PROVIDER LAYER (app/providers/)

Purpose:
File ingestion and format parsing.

Structure:
app/providers/
│
├── base/
├── comtrade/
├── csv/
├── excel/
└── future/

Responsibilities:

parse external formats
normalize into DisturbanceRecord
preserve metadata integrity

Must NOT:

perform analytics
perform rendering
own visualization logic

Provider Pattern is mandatory.

7. DATA LAYER (app/data/)

Purpose:
Core data contracts and shared models.

Contains:

DisturbanceRecord
channel models
metadata contracts
sampling models
timing models

This is the canonical internal representation layer.

All analytics and visualization operate from these contracts.

8. SESSION LAYER (app/sessions/)

Purpose:
Manage:

multi-source sessions
aligned recordings
source relationships
synchronized timebases

Responsibilities:

alignment logic
source ownership
merged display sessions

Must NOT:

perform rendering
perform analytics
9. SYNCHRONIZATION LAYER (app/synchronization/)

Purpose:
Global synchronization coordination.

Examples:

cursor synchronization
linked X-axis behavior
multi-panel coordination
multi-source alignment synchronization

This layer coordinates state only.

Must NOT:

perform heavy rendering
perform analytics
10. VISUALIZATION LAYER (app/visualization/)

Purpose:
All rendering, overlays, plot coordination, and viewport management.

Structure
app/visualization/
│
├── widgets/
├── overlays/
├── managers/
├── grouping/
├── themes/
├── performance/
└── utilities/

10.1 WIDGETS (visualization/widgets/)

Contains:

FlexiblePlotCanvas
DigitalEventTimeline
specialized rendering widgets

Responsibilities:

plot rendering
viewport interaction
OpenGL rendering
synchronized display

Must NOT:

perform engineering calculations directly
10.2 OVERLAYS (visualization/overlays/)

Purpose:
Lifecycle-managed overlay rendering.

Contains:

BaseOverlay
Harmonic overlays
Phasor overlays
future transient overlays

Responsibilities:

attach/detach lifecycle
CurveStore reuse
overlay visibility
overlay rebuild coordination

Must use:

OverlayRegistry
CurveStore
OverlayRenderPolicy
10.3 MANAGERS (visualization/managers/)

Purpose:
Visualization orchestration.

Examples:

VisualizationManager
MultiAxisManager

Responsibilities:

panel coordination
layout coordination
overlay routing
synchronization integration

Managers coordinate.
Managers should not own engineering algorithms.

10.4 GROUPING (visualization/grouping/)

Purpose:
Channel grouping logic.

Examples:

waveform grouping
sequence grouping
harmonic grouping

Must remain deterministic.

10.5 THEMES (visualization/themes/)

Purpose:
Centralized visual styling.

Examples:

overlay themes
waveform colors
spectrum colors
sequence colors

Future:

user customization
dark/light themes
accessibility themes

No rendering logic should hardcode colors.

10.6 PERFORMANCE (visualization/performance/)

Purpose:
Performance instrumentation and viewport policies.

Responsibilities:

timing hooks
viewport optimization
rendering telemetry
performance contracts
11. UI LAYER (app/ui/)

Purpose:
Application windows, menus, dialogs, and UI orchestration.

Structure:
app/ui/
│
├── main_window/
├── dialogs/
├── panels/
├── controls/
└── menus/

Responsibilities:

user interaction
menu routing
display mode switching
application-level orchestration

UI must NOT:

implement analytics directly
bypass visualization contracts
12. INFRASTRUCTURE LAYER (app/infrastructure/)

Purpose:
Shared low-level infrastructure.

Examples:

threading
logging
async coordination
future plugin infrastructure

Must remain subsystem-neutral.

13. CONFIGURATION LAYER (app/configuration/)

Purpose:
Centralized configuration/state.

Future:

theme persistence
user preferences
rendering settings
workspace layouts

Must avoid direct subsystem ownership.

14. UTILITIES (app/utilities/)

Purpose:
General reusable helpers.

Must:

remain generic
avoid business logic
avoid visualization ownership

15. TEST STRUCTURE (tests/)
tests/
│
├── unit/
├── integration/
├── runtime/
├── performance/
└── regression/

15.1 UNIT TESTS

Purpose:
Small deterministic subsystem validation.

Examples:

phasor extraction
FFT extraction
cache validation
overlay lifecycle
15.2 INTEGRATION TESTS

Purpose:
Subsystem coordination tests.

Examples:

visualization + analytics
multi-source alignment
overlay routing
15.3 RUNTIME TESTS

Purpose:
Qt/OpenGL rendering behavior.

Examples:

plot lifecycle
viewport synchronization
overlay stability
15.4 PERFORMANCE TESTS

Purpose:
Large dataset responsiveness validation.

Examples:

100MB COMTRADE
high sample-rate rendering
FFT rendering load
15.5 REGRESSION TESTS

Purpose:
Prevent previously fixed bugs from returning.

Must be added for:

overlay duplication bugs
synchronization regressions
cache regressions
16. SAMPLE DATA (samples/)

Purpose:
Known engineering reference datasets.

Contains:

COMTRADE samples
CSV samples
alignment manifests
engineering validation datasets

Must remain immutable/reference-oriented.

17. TOOLS (tools/)

Purpose:
Engineering support tooling.

Examples:

manifest generators
COMTRADE inspectors
FFT inspection utilities
synthetic waveform generators

Tools must not become runtime dependencies.

18. AGENT DIRECTORY (agent/)

Purpose:
AI orchestration governance.

Contains:

workflow contracts
handoff state
task tracking
repository state
implementation directives

Agents must update:

HANDOFF.md
TASK.md
REPOSITORY_STATE.md

after meaningful architectural changes.

19. ARCHITECTURAL RULES
19.1 Analytics Must Not Render

Analytics modules:

compute
classify
cache

They must not:

manipulate Qt widgets
own plot logic
19.2 Visualization Must Not Recompute Analytics

Visualization layer:

consumes cached results
renders efficiently

It must not:

repeatedly recompute FFT/DFT unnecessarily
19.3 Overlay Lifecycle Contract Is Mandatory

All overlays must:

inherit BaseOverlay
use CurveStore
respect OverlayRegistry
support attach/detach/dispose
19.4 Theme Centralization Is Mandatory

No hardcoded colors in rendering logic.

All colors/styles must route through:

overlay theme system
centralized theme helpers
19.5 Multi-Source Safety Is Mandatory

All visualization/analytics systems must:

tolerate unsupported channels
tolerate partial support
preserve synchronized timebases
19.6 Rendering Performance Philosophy

Rendering must:

reuse arrays
reuse PlotDataItems
use setData()
use decimation
avoid excessive allocations
20. FUTURE EXPANSION ZONES

Reserved future areas:

analytics/transients/
analytics/events/
analytics/impedance/
visualization/spectrograms/
visualization/export/
plugins/
automation/
reporting/

These areas must follow the same architectural principles.

21. AUTHORITATIVE STATUS

This document is authoritative for:

repository topology
subsystem ownership
architectural boundaries
lifecycle ownership
future expansion structure

All future phases must conform to this contract unless an intentional architectural migration is approved and documented.