"""Cross-path semantic classification consistency: CsvProvider.load() and the
Import Wizard's headless pipeline must agree on the *semantic category* and
*unit family* for the same source column, even though they intentionally
preserve different names (direct providers keep the original header; the
Wizard canonicalises it) and may hold different confidence values.

These tests use several independent, generic headers spanning demand,
generation, tie-line, active/reactive power, voltage, current, and frequency
terminology to prove the shared classifier generalises rather than being
tuned to any single fixture. samples/csv/pulu_20260306.csv is exercised once,
at the end, purely as a real-world regression check -- no rule in production
code depends on its filename, station, or exact values.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from app.import_wizard.import_pipeline import run_import_pipeline
from app.providers.csv.csv_provider import CsvProvider

# (header, sample values, expected semantic category, expected unit family)
_GENERIC_CASES = [
    ("System Demand", [18700.0, 18712.0, 18705.0, 18730.0, 18711.0], "power", "MW"),
    ("Grid Demand", [4500.0, 4510.0, 4495.0, 4520.0, 4501.0], "power", "MW"),
    ("Net Generation", [820.0, 815.0, 830.0, 825.0, 818.0], "power", "MW"),
    ("Tie Line MW", [45.0, 46.5, 44.2, 47.1, 45.8], "power", "MW"),
    ("Reactive Power", [12.5, 13.1, 12.8, 13.4, 12.9], "power", "MVAr"),
    ("Bus Voltage", [230.1, 230.4, 229.8, 230.0, 230.2], "voltage", "kV"),
    ("Frequency", [50.02, 50.01, 49.99, 50.03, 50.0], "frequency", "Hz"),
]


def _write_csv(tmp_path: Path, header: str, values: list[float]) -> Path:
    lines = [f"Time,{header}"]
    for i, v in enumerate(values):
        lines.append(f"2026-01-01 00:00:{i:02d},{v}")
    p = tmp_path / "generic.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _semantic_category(unit: str | None, parameter_type: str | None) -> str:
    """Collapse the two taxonomies (direct-provider signal_type strings and
    Wizard ParameterType.value strings) into one coarse category for
    comparison, without asserting the exact vocabulary matches.
    """
    text = f"{unit or ''} {parameter_type or ''}".lower()
    tokens = set(text.split())
    if "voltage" in text or "kv" in tokens or "v" in tokens:
        return "voltage"
    if "current" in text or "amp" in text:
        return "current"
    if "reactive" in text or "mvar" in tokens:
        return "power"
    if "active" in text or "mw" in tokens or "power" in tokens:
        return "power"
    if "frequency" in text or "hz" in tokens:
        return "frequency"
    return "other"


@pytest.mark.parametrize("header,values,expected_category,expected_unit_family", _GENERIC_CASES)
def test_semantic_category_matches_across_paths(
    tmp_path: Path, header, values, expected_category, expected_unit_family
) -> None:
    csv_path = _write_csv(tmp_path, header, values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct_record = CsvProvider().load(csv_path)
    direct_channel = next(c for c in direct_record.analog_channels if c.name == header)

    result = run_import_pipeline(str(csv_path), provider_type="csv")
    assert result.success
    wizard_channel = next(
        c for c in result.record.analog_channels
        if header.lower().replace(" ", "").replace("-", "") in c.name.lower().replace("_", "")
    )

    direct_category = _semantic_category(direct_channel.unit, direct_channel.parameter_type)
    wizard_category = _semantic_category(wizard_channel.unit, wizard_channel.parameter_type)

    assert direct_category == expected_category, (
        f"direct provider: {header!r} -> unit={direct_channel.unit!r} "
        f"parameter_type={direct_channel.parameter_type!r}"
    )
    assert wizard_category == expected_category, (
        f"wizard: {header!r} -> unit={wizard_channel.unit!r} "
        f"parameter_type={wizard_channel.parameter_type!r}"
    )
    assert direct_category == wizard_category

    # Neither path renamed the DataFrame column / channel in a way that loses
    # the original source header on the direct-provider side.
    assert direct_channel.name == header


# ─────────────────────────────────────────────────────────────────────────────
# PULU fixture — one real-world regression check, not a source of any rule.
# ─────────────────────────────────────────────────────────────────────────────

_CSV_PATH = Path("samples/csv/pulu_20260306.csv")


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="sample fixture not present")
def test_pulu_system_demand_no_longer_diverges_as_voltage() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct_record = CsvProvider().load(_CSV_PATH)
    direct_channel = next(c for c in direct_record.analog_channels if c.name == "System Demand")

    result = run_import_pipeline(str(_CSV_PATH), provider_type="csv")
    wizard_channel = next(
        c for c in result.record.analog_channels if "system_demand" in c.name.lower()
    )

    assert _semantic_category(direct_channel.unit, direct_channel.parameter_type) == "power"
    assert _semantic_category(wizard_channel.unit, wizard_channel.parameter_type) == "power"
    assert "voltage" not in wizard_channel.name
    assert wizard_channel.parameter_type != "voltage"
