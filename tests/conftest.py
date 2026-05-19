"""Repository-wide pytest runtime hygiene.

Pytest and tempfile default to user-profile temp locations. On Windows those
locations can become inaccessible after interrupted Qt/runtime sessions, which
prevents tmp_path from even setting up. Keep transient runtime artifacts inside
a repo-local temp root while still letting pytest manage its own basetemp.
"""
from __future__ import annotations

import os
import tempfile
from getpass import getuser
from pathlib import Path

import pytest

from app.testing.temp_runtime import cleanup_runtime_children, isolated_runtime_dir, safe_rmtree


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pytest_temp_root() -> Path:
    configured = os.environ.get("POWERWAVE_PYTEST_TMP")
    if configured:
        root = Path(configured)
    else:
        root = Path.cwd() / ".powerwave_runtime_tmp" / "pytest-processes" / f"run-{os.getpid()}"
    root.mkdir(mode=0o777, parents=True, exist_ok=True)
    return root.resolve()


def _configure_process_temp_root() -> Path:
    root = _pytest_temp_root()
    os.environ["TMP"] = str(root)
    os.environ["TEMP"] = str(root)
    os.environ["TMPDIR"] = str(root)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(root)
    tempfile.tempdir = str(root)

    # Pytest creates Windows temp roots with mode=0o700. In this sandbox that
    # ACL shape can become unreadable to the same process, so pre-create the
    # predictable parent with normal inherited permissions.
    pytest_user_root = root / f"pytest-of-{getuser() or 'unknown'}"
    pytest_user_root.mkdir(mode=0o777, parents=True, exist_ok=True)
    return root


def _patch_pytest_windows_tmp_mode() -> None:
    if os.name != "nt":
        return

    import _pytest.pathlib as pytest_pathlib
    import _pytest.tmpdir as pytest_tmpdir

    original = pytest_pathlib.make_numbered_dir
    if getattr(original, "_powerwave_mode_patch", False):
        return

    def make_numbered_dir(root: Path, prefix: str, mode: int = 0o700) -> Path:
        return original(root, prefix, 0o777 if mode == 0o700 else mode)

    make_numbered_dir._powerwave_mode_patch = True  # type: ignore[attr-defined]
    pytest_pathlib.make_numbered_dir = make_numbered_dir
    pytest_tmpdir.make_numbered_dir = make_numbered_dir


def _ensure_pytest_cache_dir(config) -> None:
    cache_dir = Path(str(config.getini("cache_dir"))).resolve()
    cache_dir.mkdir(mode=0o777, parents=True, exist_ok=True)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    """Redirect Python's process temp root before tmp_path is used."""
    _configure_process_temp_root()
    _patch_pytest_windows_tmp_mode()
    _ensure_pytest_cache_dir(config)


@pytest.fixture()
def runtime_tmp_path():
    """Isolated temp directory with cleanup resilient to short Windows locks."""
    path = isolated_runtime_dir("test")
    try:
        yield path
    finally:
        result = safe_rmtree(path)
        if not result.removed:
            raise AssertionError(f"Failed to clean runtime temp {result.path}: {result.error}")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Clean stray runtime fixture dirs without touching pytest's basetemp."""
    cleanup_runtime_children(prefix="test-", retries=1)
