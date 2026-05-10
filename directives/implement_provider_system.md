implement_provider_system.md — Powerwave Provider System Implementation Directive
PURPOSE

This directive implements the provider foundation architecture for Powerwave.

The provider system is responsible for:

ingestion abstraction
parser orchestration
provider registration
provider discovery
normalized DisturbanceRecord loading

This directive establishes the scalable ingestion foundation before implementing:

COMTRADE parsing
CSV parsing
Excel parsing

This phase implements ONLY:

provider interfaces
provider manager
provider registry
provider contracts
provider validation structure

No real parsing logic yet.

IMPLEMENTATION OBJECTIVES

The implementation shall:

implement BaseProvider
implement ProviderManager
implement provider registration
implement provider discovery
implement provider loading workflow
enforce DisturbanceRecord contract
remain parser-independent
support future plugin expansion

The architecture must remain:

modular
scalable
lightweight
extensible
TARGET LOCATION

Implement inside:

app/providers/

Recommended structure:

app/providers/base/base_provider.py
app/providers/base/provider_manager.py
app/providers/base/provider_registry.py
app/providers/base/exceptions.py
app/providers/init.py

Create placeholder provider stubs:

app/providers/comtrade/comtrade_provider.py
app/providers/csv/csv_provider.py
app/providers/excel/excel_provider.py

No real parsing logic yet.

REQUIRED COMPONENTS
1. BaseProvider

Implement abstract provider contract.

Recommended structure:

class BaseProvider(ABC):

    provider_name: str = "base"

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        pass

    @abstractmethod
    def load(self, path: Path) -> DisturbanceRecord:
        pass

Purpose:

enforce provider interface consistency
ensure parser independence
2. ProviderManager

ProviderManager shall:

register providers
manage provider discovery
resolve compatible provider
execute loading workflow

Recommended responsibilities:

register_provider()
unregister_provider()
available_providers()
find_provider()
load()
3. Provider Registry

Provider registry shall:

maintain provider collection
prevent duplicate registration
support future plugin expansion

Keep implementation lightweight.

Avoid:

dynamic plugin loading
filesystem scanning
premature complexity
4. Exceptions

Implement provider exceptions.

Recommended:

ProviderError
ProviderNotFoundError
ProviderLoadError
DuplicateProviderError

Keep hierarchy simple.

PLACEHOLDER PROVIDERS

Create lightweight provider stubs:

ComtradeProvider
CsvProvider
ExcelProvider

Each should:

inherit BaseProvider
define provider_name
define placeholder can_load()
define placeholder load()

Do NOT implement real parsing yet.

Placeholder load() may raise:

NotImplementedError
PROVIDER DISCOVERY FLOW

ProviderManager.load(path) should conceptually:

Find matching provider
        ↓
Validate provider
        ↓
Execute provider.load()
        ↓
Return DisturbanceRecord
PROVIDER MATCHING REQUIREMENTS

can_load(path) shall determine compatibility.

Examples:

COMTRADE:

.cfg
.comtrade

CSV:

.csv

Excel:

.xlsx
.xls

Keep matching lightweight.

DISTURBANCERECORD CONTRACT REQUIREMENT

All providers SHALL return:

DisturbanceRecord

No provider-specific structures may escape providers.

This is mandatory.

VALIDATION REQUIREMENTS

ProviderManager shall validate:

provider uniqueness
provider inheritance
provider compatibility
provider existence before loading

Errors shall remain:

explicit
predictable
debuggable

Avoid:

silent failures
ambiguous provider selection
PERFORMANCE REQUIREMENTS

The provider architecture must remain:

lightweight
scalable
low-overhead
suitable for large-file workflows

Avoid:

unnecessary file reads
eager loading behavior
speculative caching

No heavy optimization yet.

Only architecture foundation.

THREADING REQUIREMENTS

Do NOT implement threading yet.

However:

The architecture shall remain compatible with future:

QThread workers
asynchronous loading
background preprocessing

Do not block future expansion.

ARCHITECTURE RULES
Rule 1 — Parser Isolation

Providers shall isolate parsing logic from:

UI
visualization
analytics
Rule 2 — DisturbanceRecord Enforcement

Providers shall return ONLY:

DisturbanceRecord

Rule 3 — No Visualization Dependency

Providers shall not import:

PyQt6
PyQtGraph
OpenGL
Rule 4 — No Analytics Dependency

Providers shall not depend on:

RMS engines
ROCOF engines
phasor engines
Rule 5 — Legacy Isolation

Do not import from:

src/

Legacy code may be referenced only for engineering understanding.

TEST REQUIREMENTS

Create tests under:

tests/unit/test_provider_manager.py

Tests shall cover:

provider registration
duplicate registration rejection
provider discovery
provider loading routing
provider not found handling
provider validation behavior

Use lightweight mock providers.

Do NOT test real parsing yet.

ENVIRONMENT REQUIREMENT

All execution and testing must use:

.venv/

Do not use:

global Python
system pip
unrelated environments
IMPLEMENTATION CONSTRAINTS

Do NOT implement yet:

COMTRADE parsing
CSV parsing
Excel parsing
waveform rendering
analytics logic
synchronization logic
UI logic

Only provider architecture foundation.

EXPECTED OUTPUT

After implementation:

app/providers/base/base_provider.py
app/providers/base/provider_manager.py
app/providers/base/provider_registry.py
app/providers/base/exceptions.py

Placeholder providers:

app/providers/comtrade/comtrade_provider.py
app/providers/csv/csv_provider.py
app/providers/excel/excel_provider.py

Tests:

tests/unit/test_provider_manager.py

Updated:

agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md

COMPLETION REPORT REQUIREMENTS

Implementation report shall include:

1. Summary

What was implemented.

2. Files Created / Modified

List all created or modified files.

3. Architecture Alignment

Confirm:

provider isolation
DisturbanceRecord enforcement
parser independence
legacy isolation
4. Validation

Confirm:

tests added
tests executed through .venv
test results
5. Repository Tracking Updates

Confirm updates to:

agent/HANDOFF.md
agent/TASKS.md
agent/REPOSITORY_STATE.md
6. Risks / Concerns

Potential future concerns.

7. Next Recommended Step

Suggested continuation.

NEXT EXPECTED DIRECTIVE

After provider foundation:

directives/implement_comtrade_provider.md

FINAL PRINCIPLE

The provider architecture is the ingestion backbone of Powerwave.

It must remain:

modular
parser-independent
scalable
lightweight
extensible

Protect the ingestion boundary carefully.