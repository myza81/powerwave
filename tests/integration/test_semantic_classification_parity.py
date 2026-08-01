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

Documented exception (approved): direct CSV/Excel providers now apply the
same confidence discipline to `unit` that already applied to
`parameter_type` -- an unconfirmed classification (confidence below the
shared CONFIRMATION_THRESHOLD) populates neither field. The Import Wizard's
own orchestration was explicitly out of scope for that change and still
surfaces a low-confidence unit/type as an editable suggestion. For headers
whose only evidence is a medium-confidence *name* keyword (0.70-0.79, e.g.
"Grid Demand", "Net Generation" at 0.78) with no independent name-only
fallback (see app.providers.csv.csv_provider._infer_unit) to corroborate it,
direct providers therefore now legitimately land on "other" while the Wizard
still shows "power". This is intentional and is asserted explicitly below,
not treated as a parity failure.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from app.data.column_classifier import classify_csv_column
from app.import_wizard.import_pipeline import run_import_pipeline
from app.providers.csv.csv_provider import CsvProvider
from app.providers.excel.excel_provider import ExcelProvider

# (header, sample values, expected semantic category, expected unit family)
# All of these are >=0.80 confidence via the shared classifier, or (Tie Line
# MW) also independently confirmed by the provider's plain name-substring
# unit fallback -- so direct and Wizard are expected to fully agree.
_GENERIC_CASES = [
    ("System Demand", [18700.0, 18712.0, 18705.0, 18730.0, 18711.0], "power", "MW"),
    ("Tie Line MW", [45.0, 46.5, 44.2, 47.1, 45.8], "power", "MW"),
    ("Reactive Power", [12.5, 13.1, 12.8, 13.4, 12.9], "power", "MVAr"),
    ("Bus Voltage", [230.1, 230.4, 229.8, 230.0, 230.2], "voltage", "kV"),
    ("Frequency", [50.02, 50.01, 49.99, 50.03, 50.0], "frequency", "Hz"),
]

# (header, sample values) -- medium-confidence (0.78) name-only matches with
# no independent corroborating evidence. Direct providers now correctly
# withhold the guess (unit stays "unknown", parameter_type stays None); the
# Wizard is unaffected and still surfaces "power" as an editable suggestion.
_DOCUMENTED_DIVERGENCE_CASES = [
    ("Grid Demand", [4500.0, 4510.0, 4495.0, 4520.0, 4501.0]),
    ("Net Generation", [820.0, 815.0, 830.0, 825.0, 818.0]),
]


def _write_csv(tmp_path: Path, header: str, values: list[float]) -> Path:
    lines = [f"Time,{header}"]
    for i, v in enumerate(values):
        lines.append(f"2026-01-01 00:00:{i:02d},{v}")
    p = tmp_path / "generic.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_xlsx(tmp_path: Path, header: str, values: list[float]) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Time", header])
    for i, v in enumerate(values):
        ws.append([f"2026-01-01 00:00:{i:02d}", v])
    p = tmp_path / "generic.xlsx"
    wb.save(str(p))
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


@pytest.mark.parametrize("header,values", _DOCUMENTED_DIVERGENCE_CASES)
def test_medium_confidence_name_only_divergence_is_documented_and_safe(
    tmp_path: Path, header, values
) -> None:
    """See module docstring: this is an approved, explicit exception to
    cross-path parity, not a defect. Direct providers must not silently
    populate unit/parameter_type from an unconfirmed guess; the Wizard is
    unaffected and may still surface the same guess as an editable
    suggestion for the user to confirm.
    """
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

    # Direct provider: unconfirmed, so neither field is populated.
    assert direct_channel.unit == "unknown"
    assert direct_channel.parameter_type is None
    assert _semantic_category(direct_channel.unit, direct_channel.parameter_type) == "other"

    # Wizard: unaffected by this change, still surfaces the low-confidence
    # suggestion (orchestration/UI decides whether to gate it, out of scope
    # for this task).
    assert _semantic_category(wizard_channel.unit, wizard_channel.parameter_type) == "power"

    assert direct_channel.name == header


# ─────────────────────────────────────────────────────────────────────────────
# Removed magnitude-band fallback: neutral header, no name evidence.
# ─────────────────────────────────────────────────────────────────────────────

_NEUTRAL_HEADERS = ["Column 1", "Value", "Signal"]

# (dataset label, values) -- each of these previously triggered an
# electrical-type guess from magnitude/statistics alone in at least one path;
# none of them may do so any more. Frequency is intentionally excluded here
# and covered by its own test below, since it remains the one approved
# value-only exception.
_REMOVED_BAND_DATASETS = [
    ("near_1.0", [0.98, 1.01, 0.99, 1.02, 1.00, 1.01, 0.99]),
    ("noisy_132", [125.0, 138.0, 128.0, 140.0, 122.0, 135.0, 130.0, 145.0, 120.0, 133.0]),
    ("noisy_275", [265.0, 285.0, 268.0, 288.0, 262.0, 282.0, 270.0, 290.0, 260.0, 278.0]),
    ("near_18700", [18700.0, 18712.0, 18705.0, 18730.0, 18711.0, 18698.0, 18720.0]),
    ("positive_mw_like", [120.0, 140.0, 118.0, 145.0, 121.0, 110.0, 133.0, 108.0, 150.0, 125.0]),
    ("negative_mw_like", [-120.0, -140.0, -118.0, -145.0, -121.0, -110.0, -133.0, -108.0, -150.0, -125.0]),
]


@pytest.mark.parametrize("header", _NEUTRAL_HEADERS)
@pytest.mark.parametrize("label,values", _REMOVED_BAND_DATASETS)
def test_removed_magnitude_bands_stay_unknown_across_all_paths(
    tmp_path: Path, label, values, header
) -> None:
    shared = classify_csv_column(header, values)
    assert shared.signal_type is None, f"shared classifier: {label}/{header} -> {shared.signal_type!r}"

    csv_path = _write_csv(tmp_path, header, values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csv_record = CsvProvider().load(csv_path)
    csv_channel = next(c for c in csv_record.analog_channels if c.name == header)
    assert csv_channel.unit == "unknown", f"CsvProvider: {label}/{header} -> unit={csv_channel.unit!r}"
    assert csv_channel.parameter_type is None

    xlsx_path = _write_xlsx(tmp_path, header, values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xlsx_record = ExcelProvider().load(xlsx_path)
    xlsx_channel = next(c for c in xlsx_record.analog_channels if c.name == header)
    assert xlsx_channel.unit == "unknown", f"ExcelProvider: {label}/{header} -> unit={xlsx_channel.unit!r}"
    assert xlsx_channel.parameter_type is None

    result = run_import_pipeline(str(csv_path), provider_type="csv")
    assert result.success
    wizard_channel = result.record.analog_channels[0]
    assert wizard_channel.parameter_type not in (
        "voltage", "current", "mw", "mvar", "rocof",
    ), f"wizard: {label}/{header} -> parameter_type={wizard_channel.parameter_type!r}"


def test_frequency_remains_the_one_approved_value_only_exception(tmp_path: Path) -> None:
    values = [49.5, 50.6, 49.2, 50.8, 49.7, 50.3, 49.9, 50.1, 49.6, 50.4]
    header = "Column 1"

    shared = classify_csv_column(header, values)
    assert shared.signal_type == "frequency"
    assert shared.requires_user_confirmation is True

    result = run_import_pipeline(str(_write_csv(tmp_path, header, values)), provider_type="csv")
    assert result.success
    wizard_channel = result.record.analog_channels[0]
    assert wizard_channel.parameter_type == "frequency"


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
