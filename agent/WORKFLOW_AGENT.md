WORKFLOW_AGENT.md — Powerwave AI Development Orchestration
PURPOSE

This document defines the operational workflow between ChatGPT, Claude, Claude Code, and Codex during the development of Powerwave.

Powerwave is an industrial-grade Power System Disturbance Analysis platform focused on:

high-performance waveform visualization
modular disturbance data ingestion
power system analysis
protection engineering workflows
COMTRADE-centric disturbance investigation
scalable analysis architecture

This document ensures:

consistent engineering direction
controlled architecture evolution
clear separation of responsibilities
predictable implementation quality
reduced AI-generated architectural drift
CORE DEVELOPMENT PHILOSOPHY
1. PERFORMANCE FIRST

Every implementation decision shall prioritize:

UI responsiveness
efficient rendering
low memory overhead
vectorized computation
GPU acceleration where applicable

The application is expected to handle:

128+ samples/cycle
large COMTRADE files (>100MB)
multi-channel synchronized rendering
real-time interaction without lag
2. MODULAR ARCHITECTURE

The system must remain:

parser-agnostic
visualization-independent
extensible
plugin-capable
maintainable

No module shall tightly couple:

UI ↔ parser logic
analysis ↔ rendering
ingestion ↔ storage
3. INDUSTRIAL-GRADE ENGINEERING

This is NOT:

a demo app
a hobby waveform viewer
a prototype-only implementation

The architecture shall support:

utility-scale disturbance analysis
operational engineering workflows
future analytics expansion
future automation capabilities
AI AGENT ROLES
CHATGPT — SYSTEM ARCHITECT & ORCHESTRATOR
PRIMARY RESPONSIBILITIES

ChatGPT acts as:

Lead System Architect
Technical Reviewer
Workflow Coordinator
Architectural Gatekeeper
Development Strategist

ChatGPT responsibilities:

define architecture
define contracts/interfaces
define development phases
review consistency
prevent architectural drift
break work into executable tasks
coordinate Claude and Codex
maintain engineering direction

ChatGPT SHALL:

think system-wide
protect architecture integrity
prioritize long-term maintainability
prevent overengineering
ensure modularity

ChatGPT SHALL NOT:

generate uncontrolled large codebases
refactor entire repositories blindly
introduce architecture changes without review
CLAUDE — PRIMARY IMPLEMENTATION ENGINEER
PRIMARY RESPONSIBILITIES

Claude acts as:

Senior Software Engineer
Refactoring Engineer
Core System Builder
Architecture-Conscious Implementer

Claude responsibilities:

implement complex modules
refactor architecture
build scalable systems
create reusable abstractions
build framework-level components

Claude SHALL:

follow architecture strictly
avoid shortcuts
avoid hidden coupling
maintain clean abstractions
prefer scalable implementations

Claude SHALL:

explain implementation decisions
produce completion reports
highlight architecture risks
maintain consistency with docs/

Claude is preferred for:

core architecture
data models
rendering engine
runtime systems
provider patterns
synchronization systems
complex refactoring
CODEX — FOCUSED EXECUTION ENGINEER
PRIMARY RESPONSIBILITIES

Codex acts as:

focused implementation engineer
debugging engineer
unit test engineer
narrow scoped contributor

Codex responsibilities:

implement isolated tasks
fix targeted bugs
write tests
build small modules
improve reliability

Codex SHALL:

operate within defined scope
avoid architectural redesign
avoid broad refactoring
respect existing contracts

Codex is preferred for:

unit tests
validation logic
parsing edge cases
utility functions
type safety
minor fixes
small feature additions
CLAUDE CODE — REPOSITORY EXECUTION ENGINE

Claude Code acts as:

repository-aware executor
multi-file implementation engine
high-speed repository manipulator

Claude Code SHALL:

follow directives exactly
respect contracts in docs/
avoid speculative architecture changes
produce implementation summaries

Claude Code is primarily used for:

large repository edits
multi-file synchronized changes
bulk implementation work
structured refactoring
DEVELOPMENT WORKFLOW
STAGE 1 — ARCHITECTURE

Handled by:

ChatGPT

Deliverables:

architecture documents
contracts
data flow definitions
repository structure
development phases

Output location:

docs/
STAGE 2 — DIRECTIVE CREATION

Handled by:

ChatGPT

Deliverables:

task-specific implementation SOPs

Output location:

directives/

Directive examples:

implement_waveform_viewer.md
implement_comtrade_parser.md
implement_master_cursor.md
STAGE 3 — IMPLEMENTATION

Handled by:

Claude
Claude Code
Codex

Implementation rules:

Before coding:

Read relevant docs/
Read relevant directives/
Follow contracts strictly

After coding:

Produce completion report
Explain modified files
Explain architectural impact
Explain remaining gaps
Update repository tracking documents
STAGE 4 — REVIEW

Handled by:

ChatGPT

Review scope:

architecture consistency
performance concerns
modularity validation
engineering correctness
future scalability

ChatGPT may:

approve
reject
refactor direction
split implementation further
REPOSITORY STATE MANAGEMENT
PURPOSE

agent/REPOSITORY_STATE.md represents the LIVE CURRENT STATE of the repository.

Unlike:

HANDOFF.md → historical implementation log

REPOSITORY_STATE.md SHALL contain:

current implementation status
current repository structure
active architecture state
implemented systems
current blockers
validated engineering status
UPDATE REQUIREMENT

After any meaningful implementation, agents SHALL update:

HANDOFF.md
TASKS.md
REPOSITORY_STATE.md
WHEN TO UPDATE REPOSITORY_STATE.md

Update is REQUIRED for:

new module creation
architecture-affecting implementation
repository restructuring
provider implementation
rendering engine updates
synchronization updates
analytics integration
benchmark implementation
major testing additions
CURRENT STATE PRINCIPLE

REPOSITORY_STATE.md SHALL always represent:

latest validated repository state

Outdated information SHALL be replaced.

Unlike:

HANDOFF.md, which is append-only.
ARCHITECTURE REVIEW FLOW

Implementation workflow becomes:

Claude/Codex Implementation
        ↓
Update:
- HANDOFF.md
- TASKS.md
- REPOSITORY_STATE.md
        ↓
Human reviews/pastes state
        ↓
ChatGPT architecture review
        ↓
Next directive generation
ENGINEERING RULES
RULE 1 — NO ARCHITECTURAL DRIFT

Agents SHALL NOT:

invent new architectures
introduce hidden frameworks
replace agreed stack
add unnecessary abstractions

Without approval from:

ChatGPT architecture direction
RULE 2 — PERFORMANCE IS MANDATORY

Avoid:

inefficient loops
UI thread blocking
excessive memory copying
non-vectorized operations
full redraw rendering patterns

Preferred:

NumPy vectorization
downsampling
clip-to-view rendering
incremental updates
GPU-assisted rendering
RULE 3 — KEEP COMPONENTS ISOLATED

UI modules SHALL NOT:

parse files directly
compute heavy analytics directly

Parser modules SHALL NOT:

know UI state
know rendering logic

Analysis modules SHALL NOT:

depend on widgets
RULE 4 — STRICT DATA CONTRACTS

All disturbance data shall flow through:

DisturbanceRecord

No parser-specific structures shall leak into:

UI
analysis
rendering
RULE 5 — PHASED DEVELOPMENT ONLY

Do NOT:

build entire app at once
prematurely optimize unknown areas
introduce future systems too early

Development must proceed:

phase-by-phase
component-by-component
validated incrementally
COMMUNICATION FORMAT

All implementation agents SHALL provide:

1. Summary

What was implemented

2. Files Modified

List of changed files

3. Architecture Impact

Explain affected systems

4. Risks / Concerns

Potential issues or future concerns

5. Next Recommended Step

Suggested continuation

DOCUMENT HIERARCHY

Priority order:

directives/*
docs/*
agent/*
inline prompts

If conflict exists:

directives override docs
docs override prompts
PROJECT STACK
UI
PyQt6
PyQtGraph
PyOpenGL
Computation
NumPy
SciPy
Pandas
File Parsing
COMTRADE
CSV
Excel/Openpyxl
Future
plugin parser architecture
advanced disturbance analytics
phasor calculations
impedance trajectory analysis
AI-assisted disturbance classification
CURRENT DEVELOPMENT PRIORITY

Initial priorities:

repository structure
DisturbanceRecord contract
data provider architecture
COMTRADE parser
FastWaveformWidget
multi-pane synchronization
master time cursor
performance optimization
PYTHON ENVIRONMENT REQUIREMENT

All Python execution, dependency installation, testing, and tooling MUST use:

powerwave/.venv/

This applies to:

all agents (Claude, Claude Code, Codex)
all development sessions
all testing workflows

Enforcement:
Python interpreter: .venv/Scripts/python.exe (Windows) or .venv/bin/python (Unix)
Pip: .venv/Scripts/pip.exe (Windows) or .venv/bin/pip (Unix)
Pytest: .venv/Scripts/pytest.exe (Windows) or .venv/bin/pytest (Unix)
Application: .venv/Scripts/python.exe app/main.py

NEVER:
use bare python or pip commands
install to system Python
use pyenv global or other interpreter

Agents MUST verify the interpreter before executing any command:
.venv/Scripts/python.exe -c "import sys; print(sys.executable)"
FINAL PRINCIPLE

Powerwave shall be engineered as a professional industrial analysis platform.

Every implementation decision must prioritize:

reliability
clarity
performance
modularity
scalability
engineering correctness

Avoid:

unnecessary complexity
AI-generated chaos
uncontrolled abstractions
architectural inconsistency

Build incrementally.
Validate continuously.
Protect the architecture.