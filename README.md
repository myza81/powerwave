# Powerwave

Powerwave is an industrial-grade Power System Disturbance Analysis platform focused on high-performance waveform visualization, modular disturbance data ingestion, and protection engineering workflows.

## Status

Initial repository bootstrap. Phase 1 — Foundation in progress.

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

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

## Running

From the project root:

```bash
python -m app.main
```

If you need to run the file path directly, set the project root on
`PYTHONPATH`:

```bash
PYTHONPATH=. python app/main.py
```

OpenGL rendering is disabled by default; enable it with:

```bash
POWERWAVE_USE_OPENGL=1 python -m app.main
```
