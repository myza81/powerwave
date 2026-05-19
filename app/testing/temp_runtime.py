"""Deterministic runtime temp helpers for tests.

The helpers keep pytest/runtime artifacts out of user-profile temp directories,
which are a common source of permission and stale-lock failures on Windows.
They intentionally avoid global mutable state and expensive filesystem scans.
"""
from __future__ import annotations

import contextlib
import gc
import os
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_DEFAULT_ROOT_NAME = ".powerwave_runtime_tmp"


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Result of a best-effort cleanup operation."""

    path: Path
    removed: bool
    attempts: int
    error: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_temp_root(base: str | os.PathLike[str] | None = None) -> Path:
    """Return the root used for runtime/test temporary artifacts."""
    configured = base or os.environ.get("POWERWAVE_RUNTIME_TMP")
    root = Path(configured) if configured else _repo_root() / _DEFAULT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def isolated_runtime_dir(
    prefix: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Create a unique isolated temp directory under the runtime root."""
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prefix)
    path = runtime_temp_root(root) / f"{safe_prefix}-{os.getpid()}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@contextlib.contextmanager
def runtime_temp_dir(
    prefix: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Iterator[Path]:
    """Context manager that creates and safely removes an isolated temp dir."""
    path = isolated_runtime_dir(prefix, root=root)
    try:
        yield path
    finally:
        safe_rmtree(path)


def _make_writable(path: str) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _remove_readonly(func, path: str, _exc_info) -> None:
    _make_writable(path)
    func(path)


def safe_rmtree(
    path: str | os.PathLike[str],
    *,
    retries: int = 4,
    delay_seconds: float = 0.05,
) -> CleanupResult:
    """Remove a file or directory with short Windows-lock retries.

    Returns a result instead of raising so teardown paths can report useful
    diagnostics without masking the original test failure.
    """
    target = Path(path)
    if not target.exists():
        return CleanupResult(target, True, 0)

    last_error: str | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, onerror=_remove_readonly)
            else:
                _make_writable(str(target))
                target.unlink(missing_ok=True)
            return CleanupResult(target, True, attempt)
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            gc.collect()
            if attempt < attempts:
                time.sleep(delay_seconds * attempt)

    return CleanupResult(target, False, attempts, last_error)


def cleanup_runtime_children(
    root: str | os.PathLike[str] | None = None,
    *,
    prefix: str | None = None,
    retries: int = 2,
) -> list[CleanupResult]:
    """Best-effort cleanup of immediate children under the runtime root."""
    base = runtime_temp_root(root)
    results: list[CleanupResult] = []
    try:
        children = list(base.iterdir())
    except OSError as exc:
        return [CleanupResult(base, False, 0, f"{type(exc).__name__}: {exc}")]

    for child in children:
        if prefix is not None and not child.name.startswith(prefix):
            continue
        results.append(safe_rmtree(child, retries=retries))
    return results
