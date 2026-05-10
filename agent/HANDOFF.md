HANDOFF.md — Powerwave Cross-Agent Handoff Protocol
PURPOSE

This document defines the standard handoff protocol between:

ChatGPT
Claude
Claude Code
Codex

The objective is to ensure:

continuity between implementation sessions
architecture consistency
reduced duplicated work
reduced AI confusion
traceable engineering progress

Every significant implementation session SHALL update this document.

HANDOFF RULES
RULE 1 — ALWAYS UPDATE AFTER IMPLEMENTATION

After completing meaningful work, the implementation agent SHALL update:

completed work
modified files
current architecture state
unresolved concerns
next recommended step

This prevents:

repeated work
architecture drift
context loss
RULE 2 — DO NOT REMOVE HISTORY

Do NOT delete prior handoff entries.

Append new entries chronologically.

Engineering history must remain traceable.

RULE 3 — KEEP ENTRIES TECHNICAL

Handoff entries must contain:

implementation details
architecture impact
known limitations
next engineering actions

Avoid:

conversational summaries
vague descriptions
non-technical commentary
HANDOFF ENTRY FORMAT

Every handoff SHALL use this structure:

## DATE / SESSION

### Agent
Claude / Codex / ChatGPT / Claude Code

### Task
Short task description

### Completed
- item
- item
- item

### Files Modified
- path/file.py
- path/file.py

### Architecture Impact
Description of affected systems

### Performance Impact
Rendering/runtime/memory considerations

### Risks / Concerns
Known issues or future concerns

### Next Recommended Step
Suggested continuation
CURRENT SYSTEM STATUS
Architecture Status

IN PROGRESS

Repository Status

INITIALIZATION

Rendering Engine

NOT STARTED

Data Provider System

NOT STARTED

COMTRADE Parser

NOT STARTED

Synchronization Engine

NOT STARTED

ACTIVE DEVELOPMENT PRIORITIES

Current development order:

Repository structure
DisturbanceRecord contract
Provider pattern
COMTRADE parser
FastWaveformWidget
Multi-pane synchronization
Master time cursor
Performance optimization
IMPLEMENTATION TRACKING
PHASE 1 — FOUNDATION

Status: NOT STARTED

Scope:

repository structure
core contracts
application bootstrap
PHASE 2 — DATA INGESTION

Status: NOT STARTED

Scope:

provider pattern
COMTRADE parser
CSV parser
Excel parser
PHASE 3 — VISUALIZATION ENGINE

Status: NOT STARTED

Scope:

FastWaveformWidget
OpenGL acceleration
synchronized rendering
multi-pane infrastructure
PHASE 4 — INTERACTION ENGINE

Status: NOT STARTED

Scope:

master time cursor
synchronized zoom/pan
waveform interaction
PHASE 5 — ANALYTICS FOUNDATION

Status: NOT STARTED

Scope:

RMS
ROCOF
frequency analysis
harmonic foundation
phasor hooks
KNOWN ARCHITECTURAL RULES
RULE — SINGLE DATA CONTRACT

All waveform data shall use:

DisturbanceRecord

No parser-specific structures shall leak outside parser modules.

RULE — PERFORMANCE FIRST

The system must support:

large COMTRADE files
high-frequency waveform rendering
synchronized multi-pane interaction
industrial-scale disturbance analysis

Preferred techniques:

vectorization
downsampling
clip-to-view
incremental redraw
GPU acceleration
RULE — MODULAR ISOLATION

UI modules SHALL NOT:

parse files
perform heavy analytics

Parser modules SHALL NOT:

know UI state

Analysis modules SHALL NOT:

depend on widgets
ENGINEERING DECISION LOG
Decision 001

Powerwave uses:

PyQt6
PyQtGraph
PyOpenGL

Reason:
Industrial-grade high-performance rendering requirements.

Decision 002

Unified waveform contract:

DisturbanceRecord

Reason:
Prevent parser leakage and maintain modularity.

Decision 003

Provider-pattern architecture for ingestion.

Reason:
Future parser extensibility without UI modification.

OPEN CONCERNS
Concern 001

Need benchmark methodology for:

100MB+ COMTRADE loading
rendering latency
memory footprint

Status:
Pending implementation phase.

Concern 002

Need synchronization strategy validation for:

multiple waveform panes
master cursor performance

Status:
Pending visualization phase.

NEXT IMMEDIATE TARGET

Current immediate target:

docs/SYSTEM_OVERVIEW.md

This document will define:

overall system purpose
application modules
architectural philosophy
major subsystem interaction

It becomes the primary technical overview document for the entire platform.

---

## 2026-05-10 — Session 002

### Agent
Claude Code

### Task
Repository bootstrap — initialize approved directory structure per directives/build_repository_structure.md

### Completed
- Created app/ package with all 20 subdirectory packages (each with __init__.py)
- Created app/main.py — minimal QApplication + PowerwaveMainWindow placeholder
- Created tests/unit/, tests/integration/, tests/benchmarks/ with __init__.py
- Created resources/ directory
- Created README.md with project description, stack, and setup instructions
- Verified app/main.py syntax with Python compiler

### Files Modified
- app/__init__.py (new)
- app/main.py (new)
- app/config/__init__.py (new)
- app/ui/__init__.py (new)
- app/ui/main_window/__init__.py (new)
- app/ui/widgets/__init__.py (new)
- app/ui/dialogs/__init__.py (new)
- app/ui/panels/__init__.py (new)
- app/visualization/__init__.py (new)
- app/visualization/widgets/__init__.py (new)
- app/visualization/rendering/__init__.py (new)
- app/visualization/managers/__init__.py (new)
- app/visualization/interaction/__init__.py (new)
- app/analytics/__init__.py (new)
- app/analytics/rms/__init__.py (new)
- app/analytics/frequency/__init__.py (new)
- app/analytics/rocof/__init__.py (new)
- app/analytics/harmonics/__init__.py (new)
- app/analytics/phasor/__init__.py (new)
- app/providers/__init__.py (new)
- app/providers/base/__init__.py (new)
- app/providers/comtrade/__init__.py (new)
- app/providers/csv/__init__.py (new)
- app/providers/excel/__init__.py (new)
- app/models/__init__.py (new)
- app/synchronization/__init__.py (new)
- app/synchronization/cursor/__init__.py (new)
- app/synchronization/viewport/__init__.py (new)
- app/synchronization/managers/__init__.py (new)
- app/services/__init__.py (new)
- app/utils/__init__.py (new)
- tests/unit/__init__.py (new)
- tests/integration/__init__.py (new)
- tests/benchmarks/__init__.py (new)
- resources/ (new empty directory)
- README.md (new)

### Architecture Impact
Foundation layer established. All subsystem boundaries defined by directory structure:
UI layer (app/ui/), visualization engine (app/visualization/), analytics engine (app/analytics/),
provider layer (app/providers/), data model layer (app/models/), synchronization engine (app/synchronization/),
services layer (app/services/), utilities (app/utils/).
No existing src/ code was modified.

### Performance Impact
None — bootstrap only. No rendering, computation, or data loading implemented.

### Risks / Concerns
- Existing src/ contains the PowerWave Analyst codebase (separate project). The new app/ is cleanly isolated.
- tests/ directory contains both the legacy test_engine/test_parsers/test_ui subdirectories and the new unit/integration/benchmarks layout. These coexist without conflict.
- requirements.txt retains pinned versions from existing project — acceptable for now.

### Next Recommended Step
Implement DisturbanceRecord contract per directives/implement_disturbance_record.md (to be created by ChatGPT).
Bootstrap is complete and implementation-ready.