LEGACY_CODEBASE_POLICY.md — Powerwave Legacy Codebase Separation Policy
PURPOSE

This document defines the architectural policy governing the coexistence of:

src/  → legacy PowerWave Analyst codebase
app/  → new Powerwave platform architecture

The purpose of this policy is to:

prevent architecture contamination
preserve clean-system evolution
control legacy reuse safely
avoid uncontrolled coupling
maintain long-term maintainability

This document is mandatory for:

Claude
Codex
Claude Code
future implementation agents
CURRENT REPOSITORY REALITY

The repository currently contains:

src/

which represents:

legacy implementation
earlier PowerWave Analyst architecture
pre-orchestration codebase

The repository now also contains:

app/

which represents:

the new Powerwave architecture
orchestrated modular architecture
future scalable platform foundation

These two systems SHALL remain isolated.

CORE ARCHITECTURAL PRINCIPLE

The new architecture in:

app/

is the authoritative future architecture.

The legacy architecture in:

src/

is considered:

reference material
migration source
implementation knowledge source

NOT:

architectural authority
STRICT SEPARATION RULE
RULE 1 — NO DIRECT MIXING

The following is prohibited:

importing legacy modules directly into new architecture
copying large legacy subsystems blindly
coupling new architecture to old architecture

Specifically:

app/
SHALL NOT directly depend on
src/

unless explicitly approved.

LEGACY CODE REUSE POLICY
ALLOWED REUSE

The following MAY be reused from src/:

isolated algorithms
mathematical calculations
utility logic
waveform processing snippets
proven engineering calculations

ONLY IF:

reviewed
refactored
architecture-aligned
dependency-clean
PROHIBITED REUSE

The following SHALL NOT be copied blindly:

UI architecture
rendering architecture
application structure
tightly coupled modules
old synchronization systems
legacy state management
implicit architecture patterns
MIGRATION PHILOSOPHY

Powerwave is NOT:

a refactor of src/
a gradual patching exercise

Powerwave is:

a new architecture
informed by lessons from src/
selectively reusing validated engineering logic

This is a controlled rebuild.

MIGRATION STRATEGY

Migration shall occur:

Step 1 — Build Clean Foundation

Inside:

app/
Step 2 — Evaluate Legacy Components

Potential candidates:

waveform math
engineering utilities
proven calculations
Step 3 — Controlled Refactor

Before reuse:

isolate logic
remove coupling
align contracts
validate architecture fit
Step 4 — Reintegrate Safely

Reintegrated code SHALL:

follow DisturbanceRecord contract
follow provider architecture
follow visualization contract
follow synchronization architecture
IMPORT RULES
PROHIBITED

Example prohibited behavior:

from src.old_module import *

or

import src.viewer

inside:

app/
ALLOWED

Temporary evaluation/testing MAY occur:

inside migration sandbox scripts
isolated benchmarking
architecture review experiments

ONLY IF:

clearly isolated
non-production
not merged into app/
FUTURE MIGRATION AREA

Potential future migration candidates:

Engineering Utilities

Examples:

RMS calculations
signal processing helpers
waveform math
COMTRADE Knowledge

Examples:

parsing edge-case knowledge
engineering handling logic
UI/Rendering Reference

May be referenced conceptually only.

Must NOT:

be copied directly without review
ARCHITECTURE AUTHORITY

The following documents override legacy implementation behavior:

docs/ARCHITECTURE.md
docs/DATA_CONTRACT.md
docs/PROVIDER_PATTERN.md
docs/VISUALIZATION_CONTRACT.md
docs/PERFORMANCE_REQUIREMENTS.md

Legacy implementation patterns SHALL NOT override documented architecture.

AGENT RESPONSIBILITIES

All implementation agents SHALL:

treat src/ as legacy reference only
preserve clean architecture boundaries
avoid accidental coupling
avoid convenience-driven reuse

Before reusing legacy code:

explicitly justify reuse
explain architecture alignment
explain dependency isolation
RECOMMENDED FUTURE STRUCTURE

Future repository evolution MAY become:

powerwave/
│
├── app/                 ← active architecture
├── src_legacy/          ← archived legacy system

But:

no migration required yet
no restructuring required now

This is only a future possibility.

KNOWN ARCHITECTURAL RISK
Risk

Future implementation agents may:

unintentionally reuse old patterns
import legacy modules directly
bypass new architecture contracts
Mitigation

Mandatory:

architecture review
repository state tracking
strict orchestration workflow
explicit migration policy
ENGINEERING PRINCIPLE

The existence of a working legacy system does NOT justify:

bypassing architecture
reusing tightly coupled code
contaminating the new platform

Engineering discipline takes priority over convenience.

FINAL PRINCIPLE

src/ is legacy reference material.

app/ is the future Powerwave platform.

The new architecture must remain:

clean
modular
scalable
maintainable
performance-focused

Reuse knowledge carefully.
Reuse code selectively.
Protect architecture aggressively.