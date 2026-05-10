build_repository_structure.md — Powerwave Repository Bootstrap Directive
PURPOSE

This directive initializes the foundational repository structure for Powerwave.

This is the first implementation directive.

The goal is to:

establish clean repository organization
initialize architectural boundaries
prepare scalable subsystem layout
avoid future restructuring chaos

This directive focuses only on:

repository structure
package initialization
bootstrap scaffolding

This directive does not implement:

analytics logic
waveform rendering logic
parser logic
synchronization logic

Only the structural foundation.

IMPLEMENTATION OBJECTIVES

The implementation shall:

create the approved repository structure
initialize Python packages
establish subsystem boundaries
prepare scalable application layout
prepare future implementation phases

The implementation shall remain:

minimal
clean
scalable
architecture-aligned
REQUIRED REPOSITORY STRUCTURE

The repository shall contain:

powerwave/
│
├── agent/
├── directives/
├── docs/
│
├── app/
│   ├── main.py
│   │
│   ├── config/
│   ├── ui/
│   │   ├── main_window/
│   │   ├── widgets/
│   │   ├── dialogs/
│   │   └── panels/
│   │
│   ├── visualization/
│   │   ├── widgets/
│   │   ├── rendering/
│   │   ├── managers/
│   │   └── interaction/
│   │
│   ├── analytics/
│   │   ├── rms/
│   │   ├── frequency/
│   │   ├── rocof/
│   │   ├── harmonics/
│   │   └── phasor/
│   │
│   ├── providers/
│   │   ├── base/
│   │   ├── comtrade/
│   │   ├── csv/
│   │   └── excel/
│   │
│   ├── models/
│   ├── synchronization/
│   │   ├── cursor/
│   │   ├── viewport/
│   │   └── managers/
│   │
│   ├── services/
│   └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
├── resources/
│
├── requirements.txt
├── README.md
└── .gitignore
REQUIRED PACKAGE INITIALIZATION

Every Python package directory shall include:

__init__.py

This includes:

app/
app/config/
app/ui/
app/ui/main_window/
app/ui/widgets/
app/ui/dialogs/
app/ui/panels/
app/visualization/
app/visualization/widgets/
app/visualization/rendering/
app/visualization/managers/
app/visualization/interaction/
app/analytics/
app/analytics/rms/
app/analytics/frequency/
app/analytics/rocof/
app/analytics/harmonics/
app/analytics/phasor/
app/providers/
app/providers/base/
app/providers/comtrade/
app/providers/csv/
app/providers/excel/
app/models/
app/synchronization/
app/synchronization/cursor/
app/synchronization/viewport/
app/synchronization/managers/
app/services/
app/utils/
tests/
tests/unit/
tests/integration/
tests/benchmarks/
APPLICATION ENTRY POINT

Create:

app/main.py

Purpose:

Powerwave application bootstrap
future Qt application initialization

At this phase:

minimal implementation only

Expected minimal behavior:

creates QApplication
creates a placeholder main window
starts Qt event loop

Do not implement:

complex UI
docking system
waveform viewer
file loading logic
rendering engine
MINIMAL MAIN.PY EXPECTATION

The implementation may use a simple placeholder such as:

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow


class PowerwaveMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Powerwave")
        self.resize(1200, 800)


def main() -> int:
    app = QApplication(sys.argv)
    window = PowerwaveMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

This is only a bootstrap placeholder.

A proper main window module may be introduced in later directives.

REQUIREMENTS FILE

Create:

requirements.txt

Initial dependencies:

PyQt6
pyqtgraph
PyOpenGL
numpy
scipy
pandas
openpyxl
python-dateutil

Keep minimal for now.

Do not add unnecessary dependencies.

README INITIALIZATION

Create:

README.md

Minimum content should include:

project name
short description
current status
technology stack
development philosophy
setup placeholder

Example sections:

# Powerwave

Powerwave is an industrial-grade Power System Disturbance Analysis platform focused on high-performance waveform visualization, modular disturbance data ingestion, and protection engineering workflows.

## Status

Initial repository bootstrap.

## Stack

- PyQt6
- PyQtGraph
- PyOpenGL
- NumPy
- SciPy
- Pandas
- Openpyxl

## Development Philosophy

- Performance first
- Modular architecture
- Provider-based ingestion
- Contract-based data flow
- Incremental implementation
GITIGNORE INITIALIZATION

Create:

.gitignore

Include at minimum:

__pycache__/
*.py[cod]
*.pyo
*.pyd

.venv/
venv/
env/

.pytest_cache/
.mypy_cache/
.ruff_cache/

build/
dist/
*.egg-info/

.DS_Store
Thumbs.db

.idea/
.vscode/

*.log
IMPLEMENTATION CONSTRAINTS
DO NOT IMPLEMENT YET

This directive shall not implement:

DisturbanceRecord
waveform rendering engine
COMTRADE parser
CSV parser
Excel parser
synchronization engine
analytics logic
master cursor logic
OpenGL optimization
advanced UI systems

Only repository structure initialization.

REQUIRED ENGINEERING RULES
1. Clean Structure Only

Avoid:

unnecessary files
speculative modules
premature abstractions
2. Follow Approved Architecture

The implementation shall follow:

agent/WORKFLOW_AGENT.md
agent/CLAUDE.md
agent/REPOSITORY_STATE.md
docs/ARCHITECTURE.md
docs/SYSTEM_OVERVIEW.md

Do not invent alternative structures.

3. Minimal Bootstrap

Only create:

essential initialization
scalable package structure
foundation scaffolding

Avoid:

overbuilding
future speculative systems
EXPECTED OUTPUT

Expected implementation result:

repository structure initialized
packages initialized
application bootstrap exists
requirements file exists
README exists
.gitignore exists
future implementation ready
REQUIRED REPOSITORY TRACKING UPDATES

After implementation, update:

agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md
COMPLETION REPORT REQUIREMENTS

Implementation report shall include:

1. Summary

Repository structure created.

2. Files Created / Modified

List all created or modified directories/files.

3. Architecture Alignment

Confirm alignment with:

docs/ARCHITECTURE.md
docs/SYSTEM_OVERVIEW.md
docs/PERFORMANCE_REQUIREMENTS.md
4. Repository Tracking Updates

Confirm updates to:

agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md
5. Risks / Concerns

Potential future concerns.

6. Next Recommended Step

Suggested continuation.

NEXT EXPECTED DIRECTIVE

After repository bootstrap:

directives/implement_disturbance_record.md
FINAL PRINCIPLE

This phase establishes the architectural foundation of Powerwave.

The objective is:

clean structure
scalable layout
future-safe organization

Keep implementation minimal.
Protect architectural clarity.
Avoid premature complexity.