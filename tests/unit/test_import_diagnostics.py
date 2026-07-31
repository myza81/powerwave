"""Unit tests for the Import Diagnostics Summary layer (Phase 8.55N).

Coverage
--------
1.  Basic field population from a successful import
2.  Timestamp repair actions — dropped rows
3.  Timestamp repair actions — duplicate timestamps
4.  Timestamp repair actions — interpolate_missing strategy
5.  Timestamp repair actions — reconstruct_from_interval strategy
6.  Timestamp repair actions — user format override
7.  Confidence labels: High / Medium / Low / N/A
8.  Validation message grouping (errors / warnings / infos separate)
9.  Large-file guidance triggered above threshold
10. Small-file — no large-file guidance
11. Export guidance with successful ExportWriteResult
12. Export guidance with failed ExportWriteResult
13. User-overridden columns counted correctly
14. Low-confidence columns listed
15. Classification confidence label reflects overall confidence
16. Failed import — summary still builds without crash
17. No dataset — fallback to PipelineDiagnostics counts
18. data_completeness_pct computed correctly
19. has_data_loss property
20. render_diagnostics_text includes all key sections
21. render_diagnostics_text — failure text shown for failed import
22. Sidecar metadata guidance included when sidecar written
23. Provider type passed through
24. elapsed_seconds mapped to import_duration_s
25. No dataset — excluded_column_count and user_overridden_count are 0
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.import_wizard.column_mapping import ParameterType
from app.import_wizard.contracts import ValidationMessage, ValidationSeverity
from app.import_wizard.diagnostics_summary import (
    ImportDiagnosticsSummary,
    _confidence_label,
    build_import_diagnostics,
)
from app.import_wizard.export_writer import ExportWriteResult
from app.import_wizard.import_pipeline import ImportPipelineResult, PipelineDiagnostics
from app.import_wizard.models import (
    ColumnMappingCandidate,
    ImportWizardSession,
    RawPreviewModel,
    TimestampCandidate,
)
from app.import_wizard.normalized_dataset import (
    AssemblyDiagnostics,
    NormalizedDataset,
    ParameterMetadata,
)
from app.import_wizard.timestamp_contracts import TimestampRepairPlan, TimestampRepairStrategy
from app.ui.import_wizard.diagnostics_panel import render_diagnostics_text
from app.ui.import_wizard.diagnostics_panel import render_diagnostics_html


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_diag(**kw) -> PipelineDiagnostics:
    defaults = dict(
        source_file_path="/data/test.csv",
        provider_type="csv",
        normalized_row_count=100,
        analog_channel_count=5,
        digital_channel_count=2,
    )
    defaults.update(kw)
    return PipelineDiagnostics(**defaults)


def _make_dataset(
    *,
    rows: int = 100,
    dropped: int = 0,
    duplicates: int = 0,
    invalid_values: int = 0,
    params: list[ParameterMetadata] | None = None,
    excluded: list[str] | None = None,
) -> NormalizedDataset:
    ts = pd.date_range("2026-01-01", periods=rows, freq="20ms")
    df = pd.DataFrame({"timestamp": ts, "va": range(rows)})
    if params is None:
        params = [
            ParameterMetadata(
                canonical_name="va",
                parameter_type=ParameterType.VOLTAGE,
                unit="V",
                source_name="VA",
                source_index=1,
                confidence=0.9,
            )
        ]
    asm = AssemblyDiagnostics(
        total_rows=rows + dropped,
        normalized_rows=rows,
        dropped_rows=dropped,
        duplicate_timestamp_count=duplicates,
        invalid_value_count=invalid_values,
    )
    return NormalizedDataset(
        data=df,
        timestamp_column="timestamp",
        parameters=params,
        excluded_columns=list(excluded or []),
        validation_messages=[],
        diagnostics=asm,
        source_path="/data/test.csv",
        source_file_name="test.csv",
        timestamp_repair_strategy="parse_detected_format",
        is_valid=True,
    )


def _make_session() -> ImportWizardSession:
    return ImportWizardSession(source_path="/data/test.csv", provider_type="csv")


def _make_candidate(confidence: float = 0.92) -> TimestampCandidate:
    return TimestampCandidate(
        column_name="timestamp",
        column_index=0,
        confidence=confidence,
        detected_format="%Y-%m-%d %H:%M:%S.%f",
    )


def _make_repair_plan(
    strategy: TimestampRepairStrategy = TimestampRepairStrategy.PARSE_DETECTED_FORMAT,
    **kw,
) -> TimestampRepairPlan:
    defaults = dict(
        strategy=strategy,
        detected_format="%Y-%m-%d %H:%M:%S.%f",
        repair_validated=True,
    )
    defaults.update(kw)
    return TimestampRepairPlan(**defaults)


_UNSET: object = object()


def _make_result(
    *,
    success: bool = True,
    dataset: NormalizedDataset | None | object = _UNSET,
    candidate: TimestampCandidate | None = None,
    repair_plan: TimestampRepairPlan | None = None,
    diag_kw: dict | None = None,
    messages: list[ValidationMessage] | None = None,
    row_estimate: int = 100,
) -> ImportPipelineResult:
    if dataset is _UNSET:
        dataset = _make_dataset()
    if candidate is None:
        candidate = _make_candidate()
    if repair_plan is None:
        repair_plan = _make_repair_plan()

    preview = RawPreviewModel(
        column_names=["timestamp", "va"],
        preview_rows=[],
        row_count_estimate=row_estimate,
    )
    from app.import_wizard.file_profiler import FileProfileResult
    profile = FileProfileResult(
        raw_preview=preview,
        provider_type="csv",
        timestamp_candidates=[candidate],
        column_mappings=[],
    )
    diag = _make_diag(**(diag_kw or {}))
    session = _make_session()
    return ImportPipelineResult(
        session=session,
        profile=profile,
        selected_candidate=candidate,
        repair_plan=repair_plan,
        normalization_result=None,
        dataset=dataset,  # type: ignore[arg-type]
        bridge_result=None,
        record=None,
        diagnostics=diag,
        success=success,
        validation_messages=list(messages or []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic field population
# ─────────────────────────────────────────────────────────────────────────────


def test_basic_field_population() -> None:
    result = _make_result()
    summary = build_import_diagnostics(result)

    assert summary.success is True
    assert summary.provider_type == "csv"
    assert summary.source_file_name == "test.csv"
    assert summary.analog_channels == 5
    assert summary.digital_channels == 2
    assert summary.normalized_rows == 100
    assert summary.timestamp_column == "timestamp"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repair actions — dropped rows
# ─────────────────────────────────────────────────────────────────────────────


def test_repair_actions_dropped_rows() -> None:
    dataset = _make_dataset(rows=92, dropped=8)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert summary.dropped_rows == 8
    assert any("8" in a and "removed" in a for a in summary.repair_actions)
    assert summary.has_data_loss is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Repair actions — duplicate timestamps
# ─────────────────────────────────────────────────────────────────────────────


def test_repair_actions_duplicate_timestamps() -> None:
    dataset = _make_dataset(duplicates=3)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert summary.duplicate_timestamps == 3
    assert any("duplicate" in a.lower() for a in summary.repair_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Repair actions — interpolate_missing
# ─────────────────────────────────────────────────────────────────────────────


def test_repair_actions_interpolate_missing() -> None:
    plan = _make_repair_plan(strategy=TimestampRepairStrategy.INTERPOLATE_MISSING)
    result = _make_result(repair_plan=plan)
    summary = build_import_diagnostics(result)

    assert any("interpolat" in a.lower() for a in summary.repair_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Repair actions — reconstruct_from_interval
# ─────────────────────────────────────────────────────────────────────────────


def test_repair_actions_reconstruct_from_interval() -> None:
    plan = _make_repair_plan(
        strategy=TimestampRepairStrategy.RECONSTRUCT_FROM_INTERVAL,
        sampling_interval_seconds=0.02,
    )
    result = _make_result(repair_plan=plan)
    summary = build_import_diagnostics(result)

    assert any("reconstruct" in a.lower() for a in summary.repair_actions)
    assert any("0.02" in a for a in summary.repair_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Repair actions — user format override
# ─────────────────────────────────────────────────────────────────────────────


def test_repair_actions_user_format() -> None:
    plan = _make_repair_plan(
        strategy=TimestampRepairStrategy.PARSE_USER_FORMAT,
        user_format="%d/%m/%Y %H:%M:%S",
    )
    result = _make_result(repair_plan=plan)
    summary = build_import_diagnostics(result)

    assert summary.timestamp_format_source == "user override"
    assert summary.timestamp_format == "%d/%m/%Y %H:%M:%S"
    assert any("user-specified" in a.lower() or "user" in a.lower() for a in summary.repair_actions)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Confidence labels
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.95, "High"),
        (0.85, "High"),
        (0.84, "Medium"),
        (0.60, "Medium"),
        (0.59, "Low"),
        (0.0, "Low"),
        (None, "N/A"),
    ],
)
def test_confidence_label_thresholds(confidence, expected) -> None:
    assert _confidence_label(confidence) == expected


def test_timestamp_confidence_label_high() -> None:
    result = _make_result(candidate=_make_candidate(confidence=0.93))
    summary = build_import_diagnostics(result)
    assert summary.timestamp_confidence_label == "High"
    assert summary.timestamp_confidence == pytest.approx(0.93)


def test_timestamp_confidence_label_low() -> None:
    result = _make_result(candidate=_make_candidate(confidence=0.40))
    summary = build_import_diagnostics(result)
    assert summary.timestamp_confidence_label == "Low"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Validation message grouping
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_message_grouping() -> None:
    messages = [
        ValidationMessage(ValidationSeverity.ERROR, "E1", "critical error"),
        ValidationMessage(ValidationSeverity.WARNING, "W1", "minor warning"),
        ValidationMessage(ValidationSeverity.WARNING, "W2", "another warning"),
        ValidationMessage(ValidationSeverity.INFO, "I1", "info note"),
    ]
    result = _make_result(messages=messages)
    summary = build_import_diagnostics(result)

    assert len(summary.errors) == 1
    assert len(summary.warnings) == 2
    assert len(summary.infos) == 1
    assert summary.has_errors is True
    assert summary.has_warnings is True
    assert summary.errors[0].code == "E1"


def test_empty_validation_messages() -> None:
    result = _make_result(messages=[])
    summary = build_import_diagnostics(result)
    assert summary.errors == []
    assert summary.warnings == []
    assert summary.has_errors is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. Large-file guidance triggered above threshold
# ─────────────────────────────────────────────────────────────────────────────


def test_large_file_guidance_triggered() -> None:
    result = _make_result(row_estimate=200_000)
    summary = build_import_diagnostics(result)

    assert len(summary.large_file_guidance) > 0
    assert any("200,000" in g or "large" in g.lower() for g in summary.large_file_guidance)
    assert any("parquet" in g.lower() or "feather" in g.lower() for g in summary.large_file_guidance)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Small file — no large-file guidance
# ─────────────────────────────────────────────────────────────────────────────


def test_small_file_no_large_guidance() -> None:
    result = _make_result(row_estimate=500)
    summary = build_import_diagnostics(result)
    assert summary.large_file_guidance == []


# ─────────────────────────────────────────────────────────────────────────────
# 11. Export guidance with successful ExportWriteResult
# ─────────────────────────────────────────────────────────────────────────────


def test_export_guidance_success() -> None:
    result = _make_result()
    export_result = ExportWriteResult(
        success=True,
        output_path="/out/test.csv",
        format_used="csv",
        rows_written=100,
        columns_written=3,
        metadata_path="/out/test.normalized.json",
        validation_messages=[],
        diagnostics_summary="Exported 100 rows",
    )
    summary = build_import_diagnostics(result, export_result)

    assert any("100" in g for g in summary.export_guidance)
    assert any("sidecar" in g.lower() for g in summary.export_guidance)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Export guidance with failed ExportWriteResult
# ─────────────────────────────────────────────────────────────────────────────


def test_export_guidance_failure() -> None:
    result = _make_result()
    export_result = ExportWriteResult(
        success=False,
        output_path=None,
        format_used=None,
        rows_written=0,
        columns_written=0,
        metadata_path=None,
        validation_messages=[
            ValidationMessage(ValidationSeverity.ERROR, "EXPORT_FILE_EXISTS", "File exists")
        ],
        diagnostics_summary="Export failed.",
    )
    summary = build_import_diagnostics(result, export_result)

    assert any("File exists" in g or "issue" in g.lower() for g in summary.export_guidance)


# ─────────────────────────────────────────────────────────────────────────────
# 13. User-overridden columns counted correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_user_overridden_columns_counted() -> None:
    params = [
        ParameterMetadata("va", ParameterType.VOLTAGE, "V", "VA", 1, user_overridden=True, confidence=0.9),
        ParameterMetadata("ia", ParameterType.CURRENT, "A", "IA", 2, user_overridden=True, confidence=0.88),
        ParameterMetadata("mw", ParameterType.MW, "MW", "MW", 3, user_overridden=False, confidence=0.85),
    ]
    dataset = _make_dataset(params=params)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert summary.user_overridden_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 14. Low-confidence columns listed
# ─────────────────────────────────────────────────────────────────────────────


def test_low_confidence_columns_listed() -> None:
    params = [
        ParameterMetadata("va", ParameterType.VOLTAGE, "V", "VA", 1, confidence=0.9),
        ParameterMetadata("unknown1", ParameterType.UNKNOWN, None, "X1", 2, confidence=0.35),
        ParameterMetadata("unknown2", ParameterType.UNKNOWN, None, "X2", 3, confidence=0.45),
    ]
    dataset = _make_dataset(params=params)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert "unknown1" in summary.low_confidence_columns
    assert "unknown2" in summary.low_confidence_columns
    assert "va" not in summary.low_confidence_columns


# ─────────────────────────────────────────────────────────────────────────────
# 15. Classification confidence label
# ─────────────────────────────────────────────────────────────────────────────


def test_classification_confidence_label_high() -> None:
    params = [
        ParameterMetadata("va", ParameterType.VOLTAGE, "V", "VA", 1, confidence=0.92),
        ParameterMetadata("ia", ParameterType.CURRENT, "A", "IA", 2, confidence=0.88),
    ]
    dataset = _make_dataset(params=params)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert summary.classification_confidence_label == "High"


def test_classification_confidence_label_low_when_mostly_low() -> None:
    params = [
        ParameterMetadata("a", ParameterType.UNKNOWN, None, "A", 1, confidence=0.3),
        ParameterMetadata("b", ParameterType.UNKNOWN, None, "B", 2, confidence=0.4),
        ParameterMetadata("c", ParameterType.VOLTAGE, "V", "C", 3, confidence=0.9),
    ]
    dataset = _make_dataset(params=params)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    assert summary.classification_confidence_label in ("Low", "Medium")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Failed import — summary builds without crash
# ─────────────────────────────────────────────────────────────────────────────


def test_failed_import_no_crash() -> None:
    result = _make_result(success=False, dataset=None)
    summary = build_import_diagnostics(result)

    assert summary.success is False
    assert summary.analog_channels == 5  # from PipelineDiagnostics defaults
    assert isinstance(summary.repair_actions, list)
    assert isinstance(summary.errors, list)


# ─────────────────────────────────────────────────────────────────────────────
# 17. No dataset — fallback to PipelineDiagnostics counts
# ─────────────────────────────────────────────────────────────────────────────


def test_no_dataset_falls_back_to_diag_counts() -> None:
    result = _make_result(dataset=None, diag_kw=dict(
        source_file_path="/data/test.csv",
        provider_type="csv",
        normalized_row_count=250,
        analog_channel_count=10,
        digital_channel_count=3,
    ))
    summary = build_import_diagnostics(result)

    assert summary.normalized_rows == 250
    assert summary.total_rows == 250
    assert summary.dropped_rows == 0
    assert summary.analog_channels == 10
    assert summary.digital_channels == 3


# ─────────────────────────────────────────────────────────────────────────────
# 18. data_completeness_pct
# ─────────────────────────────────────────────────────────────────────────────


def test_data_completeness_pct() -> None:
    dataset = _make_dataset(rows=92, dropped=8)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)

    pct = summary.data_completeness_pct
    assert pct is not None
    assert abs(pct - 92.0) < 1.0


def test_data_completeness_pct_zero_rows() -> None:
    result = _make_result(dataset=None, diag_kw=dict(
        source_file_path="/data/test.csv",
        provider_type="csv",
        normalized_row_count=0,
    ))
    summary = build_import_diagnostics(result)
    assert summary.data_completeness_pct is None


# ─────────────────────────────────────────────────────────────────────────────
# 19. has_data_loss property
# ─────────────────────────────────────────────────────────────────────────────


def test_has_data_loss_false_when_no_drops() -> None:
    result = _make_result(dataset=_make_dataset(dropped=0))
    summary = build_import_diagnostics(result)
    assert summary.has_data_loss is False


def test_has_data_loss_true_when_drops() -> None:
    result = _make_result(dataset=_make_dataset(dropped=5))
    summary = build_import_diagnostics(result)
    assert summary.has_data_loss is True


# ─────────────────────────────────────────────────────────────────────────────
# 20. render_diagnostics_text includes all key sections
# ─────────────────────────────────────────────────────────────────────────────


def test_render_text_includes_all_sections() -> None:
    dataset = _make_dataset(rows=50, dropped=2, duplicates=1)
    params = [
        ParameterMetadata("va", ParameterType.VOLTAGE, "V", "VA", 1, confidence=0.92),
        ParameterMetadata("ia", ParameterType.CURRENT, "A", "IA", 2, confidence=0.88),
    ]
    dataset = _make_dataset(rows=50, dropped=2, duplicates=1, params=params)
    messages = [
        ValidationMessage(ValidationSeverity.WARNING, "WARN_TEST", "test warning"),
    ]
    result = _make_result(dataset=dataset, messages=messages)
    summary = build_import_diagnostics(result)
    text = render_diagnostics_text(summary)

    assert "DATA SUMMARY" in text
    assert "TIMESTAMP" in text
    assert "CHANNEL CLASSIFICATION" in text
    assert "VALIDATION" in text
    assert "50" in text   # normalized rows
    assert "2" in text    # dropped rows
    assert "test warning" in text


def test_render_html_uses_structured_sections_and_alert_colour() -> None:
    dataset = _make_dataset(rows=50, dropped=2, duplicates=1)
    result = _make_result(dataset=dataset)
    summary = build_import_diagnostics(result)
    html = render_diagnostics_html(summary)

    assert "Data Summary" in html
    assert "Timestamp" in html
    assert "Channel Classification" in html
    assert "Rows imported" in html
    assert "#B71C1C" in html


# ─────────────────────────────────────────────────────────────────────────────
# 21. render_diagnostics_text — failure status shown
# ─────────────────────────────────────────────────────────────────────────────


def test_render_text_shows_failure_status() -> None:
    result = _make_result(success=False, dataset=None)
    summary = build_import_diagnostics(result)
    text = render_diagnostics_text(summary)
    assert "failed" in text.lower() or "✗" in text


# ─────────────────────────────────────────────────────────────────────────────
# 22. Sidecar metadata guidance included when sidecar written
# ─────────────────────────────────────────────────────────────────────────────


def test_sidecar_guidance_included() -> None:
    result = _make_result()
    export_result = ExportWriteResult(
        success=True,
        output_path="/out/test.csv",
        format_used="csv",
        rows_written=100,
        columns_written=3,
        metadata_path="/out/test.normalized.json",
        validation_messages=[],
        diagnostics_summary="",
    )
    summary = build_import_diagnostics(result, export_result)
    assert any("sidecar" in g.lower() for g in summary.export_guidance)


# ─────────────────────────────────────────────────────────────────────────────
# 23. Provider type passed through
# ─────────────────────────────────────────────────────────────────────────────


def test_provider_type_passed_through() -> None:
    diag = _make_diag(provider_type="excel")
    from app.import_wizard.file_profiler import FileProfileResult
    preview = RawPreviewModel(column_names=[], preview_rows=[])
    profile = FileProfileResult(raw_preview=preview, provider_type="excel")
    session = _make_session()
    session.provider_type = "excel"
    result = ImportPipelineResult(
        session=session,
        profile=profile,
        selected_candidate=_make_candidate(),
        repair_plan=_make_repair_plan(),
        normalization_result=None,
        dataset=_make_dataset(),
        bridge_result=None,
        record=None,
        diagnostics=diag,
        success=True,
        validation_messages=[],
    )
    summary = build_import_diagnostics(result)
    assert summary.provider_type == "excel"


# ─────────────────────────────────────────────────────────────────────────────
# 24. elapsed_seconds → import_duration_s
# ─────────────────────────────────────────────────────────────────────────────


def test_elapsed_seconds_mapped() -> None:
    diag = _make_diag(elapsed_seconds=3.14)
    from app.import_wizard.file_profiler import FileProfileResult
    preview = RawPreviewModel(column_names=[], preview_rows=[])
    profile = FileProfileResult(raw_preview=preview, provider_type="csv")
    result = ImportPipelineResult(
        session=_make_session(),
        profile=profile,
        selected_candidate=_make_candidate(),
        repair_plan=_make_repair_plan(),
        normalization_result=None,
        dataset=_make_dataset(),
        bridge_result=None,
        record=None,
        diagnostics=diag,
        success=True,
        validation_messages=[],
    )
    summary = build_import_diagnostics(result)
    assert summary.import_duration_s == pytest.approx(3.14)


# ─────────────────────────────────────────────────────────────────────────────
# 25. No dataset — excluded/override counts are 0
# ─────────────────────────────────────────────────────────────────────────────


def test_no_dataset_excluded_and_override_counts_zero() -> None:
    result = _make_result(dataset=None)
    summary = build_import_diagnostics(result)
    assert summary.excluded_column_count == 0
    assert summary.user_overridden_count == 0
    assert summary.low_confidence_columns == []
    assert summary.classification_confidence_label == "N/A"
