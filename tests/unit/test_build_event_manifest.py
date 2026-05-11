"""Unit tests for tools/build_event_manifest.py.

Uses minimal synthetic CFG + CSV fixtures.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from build_event_manifest import (
    AlignmentInfo,
    EventManifest,
    SourceEntry,
    _compute_alignment,
    _manifest_to_yaml,
    _repo_relative,
    _to_yaml,
    build_manifest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_MINIMAL_CFG = """\
PULU,REC01,1999
9,6A,3D
1,VA,A,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
2,VB,B,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
3,VC,C,,kV,1.0,0.0,0,-32767,32767,1.0,1.0,P
4,IA,A,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
5,IB,B,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
6,IC,C,,A,1.0,0.0,0,-32767,32767,1.0,1.0,P
1,TRIP,,,0
2,START,,,0
3,ALARM,,,0
50
1
5000,32000
06/03/2026,18:04:08.817733
06/03/2026,18:04:09.317733
BINARY
1
"""

_OPS_CSV = """\
Timestamp,MW,MVar,Frequency
2026-03-06 17:25:00,100.0,50.0,49.98
2026-03-06 17:26:00,105.0,48.0,50.01
2026-03-06 17:27:00,110.0,52.0,50.00
"""


def _make_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Return (cfg_path, csv_path)."""
    cfg_dir = tmp_path / "comtrade" / "pulu_20260306"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "event.cfg"
    cfg.write_text(_MINIMAL_CFG, encoding="latin-1")

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv = csv_dir / "pulu_20260306.csv"
    csv.write_text(_OPS_CSV, encoding="utf-8")
    return cfg, csv


# ─────────────────────────────────────────────────────────────────────────────
# TestToYaml
# ─────────────────────────────────────────────────────────────────────────────


class TestToYaml:
    def test_scalar_string(self) -> None:
        assert _to_yaml("hello") == "hello"

    def test_string_with_colon_quoted(self) -> None:
        result = _to_yaml("key: value")
        assert result.startswith('"') and result.endswith('"')

    def test_integer(self) -> None:
        assert _to_yaml(42) == "42"

    def test_float(self) -> None:
        assert _to_yaml(3.14) == "3.14"

    def test_none(self) -> None:
        assert _to_yaml(None) == "null"

    def test_bool_true(self) -> None:
        assert _to_yaml(True) == "true"

    def test_bool_false(self) -> None:
        assert _to_yaml(False) == "false"

    def test_empty_list(self) -> None:
        assert _to_yaml([]) == "[]"

    def test_list_of_strings(self) -> None:
        result = _to_yaml(["a", "b"])
        assert "- a" in result
        assert "- b" in result

    def test_empty_dict(self) -> None:
        assert _to_yaml({}) == "{}"

    def test_simple_dict(self) -> None:
        result = _to_yaml({"key": "value"})
        assert "key: value" in result

    def test_nested_dict(self) -> None:
        result = _to_yaml({"outer": {"inner": 1}})
        assert "outer:" in result
        assert "inner: 1" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestRepoRelative
# ─────────────────────────────────────────────────────────────────────────────


class TestRepoRelative:
    def test_relative_to_root(self, tmp_path: Path) -> None:
        root = tmp_path
        file = tmp_path / "samples" / "comtrade" / "event.cfg"
        file.parent.mkdir(parents=True)
        file.touch()
        rel = _repo_relative(file, root)
        assert rel == "samples/comtrade/event.cfg"

    def test_uses_forward_slashes(self, tmp_path: Path) -> None:
        root = tmp_path
        file = tmp_path / "a" / "b" / "c.txt"
        file.parent.mkdir(parents=True)
        file.touch()
        rel = _repo_relative(file, root)
        assert "\\" not in rel


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeAlignment
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAlignment:
    def _make_source(self, sid: str, start: str | None) -> SourceEntry:
        return SourceEntry(
            source_id=sid,
            source_type="comtrade",
            paths={},
            start_time=start,
            trigger_time=None,
            sampling_rates_hz=[],
            channel_names=[],
        )

    def test_empty_sources_returns_empty_offsets(self) -> None:
        result = _compute_alignment([])
        assert result.offsets == {}

    def test_reference_is_earliest(self) -> None:
        s1 = self._make_source("a", "2026-03-06T17:25:00")
        s2 = self._make_source("b", "2026-03-06T18:04:08")
        result = _compute_alignment([s1, s2])
        assert result.reference_source == "a"

    def test_offset_computed_correctly(self) -> None:
        s1 = self._make_source("a", "2026-03-06T17:25:00")
        s2 = self._make_source("b", "2026-03-06T18:04:08.817733")
        result = _compute_alignment([s1, s2])
        # 39 min + 8.817733 s = 2348.817733 s
        assert result.offsets["b"] == pytest.approx(2348.817733, abs=0.01)
        assert result.offsets["a"] == pytest.approx(0.0)

    def test_source_with_no_start_time_gets_zero_offset(self) -> None:
        s1 = self._make_source("a", "2026-03-06T17:25:00")
        s2 = self._make_source("b", None)
        result = _compute_alignment([s1, s2])
        assert result.offsets["b"] == pytest.approx(0.0)

    def test_all_none_start_times(self) -> None:
        s1 = self._make_source("a", None)
        s2 = self._make_source("b", None)
        result = _compute_alignment([s1, s2])
        assert result.reference_start is None


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildManifest
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildManifest:
    def test_comtrade_only_builds_manifest(self, tmp_path: Path) -> None:
        cfg, _ = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="pulu_20260306",
            comtrade_paths=[cfg.parent],
            csv_paths=[],
            excel_paths=[],
            root=tmp_path,
        )
        assert manifest.event_id == "pulu_20260306"
        assert len(manifest.sources) == 1
        assert manifest.sources[0].source_type == "comtrade"

    def test_mixed_sources_manifest(self, tmp_path: Path) -> None:
        cfg, csv = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="pulu_20260306",
            comtrade_paths=[cfg.parent],
            csv_paths=[csv],
            excel_paths=[],
            root=tmp_path,
        )
        assert len(manifest.sources) == 2
        types = {s.source_type for s in manifest.sources}
        assert "comtrade" in types
        assert "csv" in types

    def test_source_ids_assigned(self, tmp_path: Path) -> None:
        cfg, _ = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="test",
            comtrade_paths=[cfg.parent],
            csv_paths=[],
            excel_paths=[],
            root=tmp_path,
        )
        assert manifest.sources[0].source_id == "comtrade_main"

    def test_channel_names_in_comtrade_source(self, tmp_path: Path) -> None:
        cfg, _ = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="test",
            comtrade_paths=[cfg.parent],
            csv_paths=[],
            excel_paths=[],
            root=tmp_path,
        )
        src = manifest.sources[0]
        assert "VA" in src.channel_names
        assert "TRIP" in src.channel_names

    def test_alignment_reference_is_earliest(self, tmp_path: Path) -> None:
        cfg, csv = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="test",
            comtrade_paths=[cfg.parent],
            csv_paths=[csv],
            excel_paths=[],
            root=tmp_path,
        )
        # CSV starts at 17:25, COMTRADE starts at 18:04 → CSV is earlier
        assert manifest.alignment.reference_source == "csv_ops"

    def test_comtrade_offset_positive(self, tmp_path: Path) -> None:
        cfg, csv = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="test",
            comtrade_paths=[cfg.parent],
            csv_paths=[csv],
            excel_paths=[],
            root=tmp_path,
        )
        # COMTRADE starts 39 min + 8.8 s after CSV
        comtrade_offset = manifest.alignment.offsets.get("comtrade_main", 0.0)
        assert comtrade_offset == pytest.approx(2348.817733, abs=1.0)

    def test_paths_are_repo_relative(self, tmp_path: Path) -> None:
        cfg, _ = _make_fixtures(tmp_path)
        manifest = build_manifest(
            event_id="test",
            comtrade_paths=[cfg.parent],
            csv_paths=[],
            excel_paths=[],
            root=tmp_path,
        )
        src = manifest.sources[0]
        for p in src.paths.values():
            assert not Path(p).is_absolute(), f"Expected relative path, got: {p}"


# ─────────────────────────────────────────────────────────────────────────────
# TestManifestToYaml
# ─────────────────────────────────────────────────────────────────────────────


class TestManifestToYaml:
    def _make_manifest(self, tmp_path: Path) -> EventManifest:
        cfg, csv = _make_fixtures(tmp_path)
        return build_manifest(
            event_id="pulu_20260306",
            comtrade_paths=[cfg.parent],
            csv_paths=[csv],
            excel_paths=[],
            root=tmp_path,
        )

    def test_yaml_contains_event_id(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path)
        yaml = _manifest_to_yaml(manifest)
        assert "pulu_20260306" in yaml

    def test_yaml_contains_source_ids(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path)
        yaml = _manifest_to_yaml(manifest)
        assert "comtrade_main" in yaml
        assert "csv_ops" in yaml

    def test_yaml_contains_alignment(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path)
        yaml = _manifest_to_yaml(manifest)
        assert "alignment" in yaml
        assert "reference_source" in yaml

    def test_yaml_ends_with_newline(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path)
        yaml = _manifest_to_yaml(manifest)
        assert yaml.endswith("\n")
