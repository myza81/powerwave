"""Run standard Import Wizard validation slices.

This script is intentionally small: it groups repository-native pytest commands
for contributors and prints concise pass/fail summaries.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SLICE_PATHS: dict[str, list[str]] = {
    "unit": [
        "tests/unit/test_import_wizard_gui.py",
        "tests/unit/test_import_workflow_ux.py",
        "tests/unit/test_timestamp_override_ui.py",
        "tests/unit/test_import_diagnostics.py",
        "tests/unit/test_export_ui.py",
        "tests/unit/test_export_writer.py",
        "tests/unit/test_export_planning.py",
        "tests/unit/test_import_pipeline.py",
        "tests/unit/test_plan_aware_pipeline.py",
        "tests/unit/test_disturbance_record_bridge.py",
    ],
    "runtime": [
        "tests/runtime/test_runtime_environment.py",
        "tests/runtime/test_import_wizard_runtime.py",
        "tests/runtime/test_import_wizard_authoritative_flow.py",
        "tests/runtime/test_timestamp_override_execution.py",
        "tests/runtime/test_export_ui_runtime.py",
        "tests/runtime/test_import_diagnostics_runtime.py",
        "tests/runtime/test_import_wizard_realistic_workflows.py",
        "tests/runtime/test_import_workflow_runtime.py",
    ],
    "stress": [
        "tests/stress",
    ],
    "acceptance": [
        "tests/acceptance/test_import_acceptance.py",
    ],
}

SLICE_PATHS["import-full"] = [
    *SLICE_PATHS["unit"],
    *SLICE_PATHS["runtime"],
    *SLICE_PATHS["stress"],
    *SLICE_PATHS["acceptance"],
]


def run_slice(slice_name: str, *, extra_pytest_args: list[str] | None = None) -> int:
    paths = SLICE_PATHS[slice_name]
    cmd = [sys.executable, "-m", "pytest", *paths, *(extra_pytest_args or [])]
    print(f"Import Wizard validation slice: {slice_name}")
    print("Command:", " ".join(cmd))
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"{status}: {slice_name} completed in {elapsed:.2f}s")
    return completed.returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Import Wizard validation slices.")
    parser.add_argument(
        "--slice",
        choices=sorted(SLICE_PATHS),
        default="acceptance",
        help="Validation slice to run.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed through to pytest. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_slice(args.slice, extra_pytest_args=args.pytest_arg)


if __name__ == "__main__":
    raise SystemExit(main())
