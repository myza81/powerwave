"""Practical benchmark runner for the Import Wizard pipeline.

The script generates a deterministic CSV into the runtime temp root, runs the
backend import/export path, and reports coarse stage timings. It intentionally
uses only repository APIs and lightweight standard-library measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.import_wizard import ExportWriteOptions, write_normalized_export
from app.import_wizard.file_profiler import profile_import_file
from app.import_wizard.import_pipeline import run_import_pipeline
from app.testing.temp_runtime import runtime_temp_dir
from tools.generate_import_stress_samples import StressSampleConfig, generate_import_stress_csv


def benchmark_import_pipeline(
    *,
    row_count: int,
    output_json: str | Path | None = None,
    delimiter: str = ",",
    metadata_rows: int = 0,
    malformed_ratio: float = 0.0,
    duplicate_ratio: float = 0.0,
    missing_ratio: float = 0.0,
    export_csv: bool = True,
) -> dict[str, Any]:
    """Run a benchmark and return a serializable result dictionary."""
    config = StressSampleConfig(
        row_count=row_count,
        delimiter=delimiter,
        metadata_rows=metadata_rows,
        malformed_timestamp_ratio=malformed_ratio,
        duplicate_timestamp_ratio=duplicate_ratio,
        missing_timestamp_ratio=missing_ratio,
        digital_text_values=True,
    )

    tracemalloc.start()
    with runtime_temp_dir("import-benchmark") as root:
        source = root / f"stress_{row_count}.csv"
        t0 = time.perf_counter()
        generate_import_stress_csv(source, config)
        generate_seconds = time.perf_counter() - t0
        source_size = source.stat().st_size

        t0 = time.perf_counter()
        profile = profile_import_file(str(source))
        profile_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = run_import_pipeline(str(source))
        import_seconds = time.perf_counter() - t0

        export_seconds = None
        export_success = None
        export_size = None
        if export_csv and result.dataset is not None and result.dataset.is_export_ready():
            export_path = root / "stress_normalized.csv"
            t0 = time.perf_counter()
            export_result = write_normalized_export(
                result.dataset,
                export_path,
                options=ExportWriteOptions(include_metadata_sidecar=True, overwrite=True),
            )
            export_seconds = time.perf_counter() - t0
            export_success = export_result.success
            export_size = export_path.stat().st_size if export_path.exists() else 0

        current_mem, peak_mem = tracemalloc.get_traced_memory()

    tracemalloc.stop()
    summary: dict[str, Any] = {
        "config": asdict(config),
        "source_size_bytes": source_size,
        "timings_seconds": {
            "generate": round(generate_seconds, 6),
            "profile": round(profile_seconds, 6),
            "total_import": round(import_seconds, 6),
            "export_csv": round(export_seconds, 6) if export_seconds is not None else None,
        },
        "profile": {
            "provider_type": profile.provider_type,
            "delimiter": profile.delimiter,
            "header_row_index": profile.raw_preview.header_row_index,
            "preview_rows": len(profile.raw_preview.preview_rows),
            "timestamp_candidates": len(profile.timestamp_candidates),
            "messages": [m.code for m in profile.validation_messages],
        },
        "pipeline": {
            "success": result.success,
            "normalized_rows": result.diagnostics.normalized_row_count,
            "analog_channels": result.diagnostics.analog_channel_count,
            "digital_channels": result.diagnostics.digital_channel_count,
            "warnings": result.diagnostics.warning_count,
            "errors": result.diagnostics.error_count,
            "message_codes": [m.code for m in result.validation_messages],
        },
        "export": {
            "success": export_success,
            "output_size_bytes": export_size,
        },
        "memory": {
            "tracemalloc_current_bytes": current_mem,
            "tracemalloc_peak_bytes": peak_mem,
        },
    }

    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Import Wizard backend pipeline.")
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--metadata-rows", type=int, default=0)
    parser.add_argument("--malformed-ratio", type=float, default=0.0)
    parser.add_argument("--duplicate-ratio", type=float, default=0.0)
    parser.add_argument("--missing-ratio", type=float, default=0.0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = benchmark_import_pipeline(
        row_count=args.rows,
        output_json=args.json,
        delimiter=args.delimiter,
        metadata_rows=args.metadata_rows,
        malformed_ratio=args.malformed_ratio,
        duplicate_ratio=args.duplicate_ratio,
        missing_ratio=args.missing_ratio,
        export_csv=not args.no_export,
    )
    timings = result["timings_seconds"]
    pipeline = result["pipeline"]
    memory = result["memory"]
    print("Import Wizard benchmark")
    print(f"rows: {pipeline['normalized_rows']:,} / success: {pipeline['success']}")
    print(
        "timings: "
        f"profile={timings['profile']:.3f}s, "
        f"import={timings['total_import']:.3f}s, "
        f"export={timings['export_csv'] if timings['export_csv'] is not None else 'n/a'}s"
    )
    print(
        "channels: "
        f"analog={pipeline['analog_channels']}, digital={pipeline['digital_channels']}"
    )
    print(f"peak traced memory: {memory['tracemalloc_peak_bytes'] / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
