generate_fault_test_record.md — Codex Follow-Up Directive (PLACEHOLDER)
DIRECTIVE AUTHORITY

Issued by: ChatGPT Architecture Orchestrator
Target agent: Codex
Phase: 3A support / visualization stress testing
Status: PLACEHOLDER — do not implement yet

Prerequisite: directives/implement_flexible_plot_canvas.md must be COMPLETED first.

─────────────────────────────────────────────────────────────────────────────
PURPOSE
─────────────────────────────────────────────────────────────────────────────

Generate synthetic disturbance datasets for visualization stress testing and
rendering validation of FlexiblePlotCanvas.

These datasets allow testing without requiring real COMTRADE files and provide
controlled waveforms with known characteristics for rendering correctness checks.

─────────────────────────────────────────────────────────────────────────────
REQUIRED SYNTHETIC DATASET SPECIFICATION
─────────────────────────────────────────────────────────────────────────────

Target file: tests/fixtures/synthetic_fault_record.py

The module shall provide a factory function:

  def make_fault_record(
      sample_rate_hz: float = 4_000.0,
      duration_s: float = 0.5,
      fault_start_s: float = 0.15,
      fault_end_s: float = 0.35,
      nominal_freq_hz: float = 50.0,
      include_digital: bool = True,
  ) -> DisturbanceRecord:

─────────────────────────────────────────────────────────────────────────────
REQUIRED ANALOG CHANNELS
─────────────────────────────────────────────────────────────────────────────

  Channel    Unit   Signal description
  ────────   ────   ──────────────────────────────────────────────────────────
  VA         kV     Phase A voltage — 110kV nominal, voltage dip during fault
  VB         kV     Phase B voltage — 110kV nominal, maintained during fault
  VC         kV     Phase C voltage — 110kV nominal, maintained during fault
  IA         A      Phase A current — 1kA nominal, 5kA overcurrent during fault
  IB         A      Phase B current — 1kA nominal, maintained during fault
  IC         A      Phase C current — 1kA nominal, maintained during fault
  FREQ       Hz     System frequency — 50.0 nominal, slight dip during fault (optional)
  ROCOF      Hz/s   Rate of change of frequency — 0 nominal (optional)

─────────────────────────────────────────────────────────────────────────────
REQUIRED DIGITAL CHANNELS (if include_digital=True)
─────────────────────────────────────────────────────────────────────────────

  Channel    Signal description
  ────────   ──────────────────────────────────────────────────────────────────
  CB_status  Circuit breaker status — 1=closed, opens at fault_end_s
  TRIP_A     Phase A trip signal — pulses high at fault_start_s + 80ms
  PICKUP_A   Phase A relay pickup — goes high at fault_start_s + 5ms

─────────────────────────────────────────────────────────────────────────────
WAVEFORM CHARACTERISTICS
─────────────────────────────────────────────────────────────────────────────

Pre-fault (t < fault_start_s):
  VA = 110/√3 × √2 × sin(2π × 50 × t)           # 63.5kV peak, 50Hz
  VB = VA shifted -120°
  VC = VA shifted +120°
  IA = 1000 × √2 × sin(2π × 50 × t - φ)         # 1kA peak, lagging
  IB, IC = IA with phase shifts

During fault (fault_start_s ≤ t < fault_end_s):
  VA = VA_prefault × 0.15                          # voltage collapse on A phase
  IA = 5000 × √2 × sin(2π × 50 × t)               # 5kA fault current

Post-fault (t ≥ fault_end_s):
  All channels return to pre-fault levels (breaker opened)

Trigger time: fault_start_s (TimingInformation.trigger_time)

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION CONSTRAINTS
─────────────────────────────────────────────────────────────────────────────

  - Use NumPy vectorized operations only — no sample loops
  - Return a valid DisturbanceRecord (must pass record.validate())
  - Use app.models imports only — no src/ imports
  - No file I/O — generate arrays in-memory
  - The function must be deterministic (same inputs → same outputs)
  - station_name = "Synthetic Fault Station"
  - recorder_name = "SyntheticGenerator"
  - provider_type = "synthetic"

─────────────────────────────────────────────────────────────────────────────
TEST FILE
─────────────────────────────────────────────────────────────────────────────

Target: tests/unit/test_synthetic_fault_record.py

Minimum coverage:
  - Record validates successfully (record.validate() == [])
  - Correct number of analog channels
  - Correct number of digital channels (if include_digital=True)
  - Time array length matches expected sample count
  - VA channel has voltage dip during fault (value < 50% of pre-fault)
  - IA channel has overcurrent during fault (value > 3kA peak)
  - Trigger time is set correctly
  - Sampling rate matches sample_rate_hz

─────────────────────────────────────────────────────────────────────────────
ACTIVATION CONDITION
─────────────────────────────────────────────────────────────────────────────

This directive SHALL NOT be implemented until:

  1. directives/implement_flexible_plot_canvas.md is COMPLETED
  2. app/visualization/widgets/flexible_plot_canvas.py is implemented
  3. ChatGPT explicitly activates this directive

This is a placeholder only. Do not implement prematurely.
