"""Run the Import Wizard runtime validation slice, optionally repeated."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_import_acceptance import run_slice


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable Import Wizard runtime validation.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runtime passes.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed through to pytest. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repeat = max(1, args.repeat)
    for index in range(repeat):
        print(f"Runtime pass {index + 1}/{repeat}")
        code = run_slice("runtime", extra_pytest_args=args.pytest_arg)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
