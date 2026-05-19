"""Lightweight timestamp format override validation.

This module validates user-entered strptime format strings against sampled
timestamp values from the profiler. It is intentionally small and backend-only:
no Qt imports and no full-dataset parsing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.import_wizard.contracts import ValidationMessage, ValidationSeverity


@dataclass(frozen=True, slots=True)
class TimestampFormatValidationResult:
    """Result of validating a manual timestamp format against sample values."""

    format_text: str
    sample_count: int
    parsed_count: int
    validation_messages: list[ValidationMessage] = field(default_factory=list)
    exception_text: str | None = None

    @property
    def parse_rate(self) -> float:
        return self.parsed_count / self.sample_count if self.sample_count else 0.0

    @property
    def is_valid(self) -> bool:
        return not any(
            message.severity == ValidationSeverity.ERROR
            for message in self.validation_messages
        )


def validate_user_timestamp_format(
    format_text: str,
    samples: list[str],
    *,
    affected_column: str | None = None,
    max_samples: int = 50,
) -> TimestampFormatValidationResult:
    """Validate a user format with a bounded sample set.

    Empty samples are ignored. Complete parse failure is an ERROR and blocks
    execution. Partial success is a WARNING so the user can still proceed when
    a few malformed source rows will be dropped by the pipeline.
    """
    fmt = format_text.strip()
    clean_samples = [str(value).strip() for value in samples if str(value).strip()][:max_samples]

    if not fmt:
        return TimestampFormatValidationResult(
            format_text=fmt,
            sample_count=len(clean_samples),
            parsed_count=0,
            validation_messages=[
                ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="PLAN_USING_DETECTED_FORMAT",
                    message="Manual timestamp format is empty; using detected format.",
                    affected_column=affected_column,
                )
            ],
        )

    if not clean_samples:
        return TimestampFormatValidationResult(
            format_text=fmt,
            sample_count=0,
            parsed_count=0,
            validation_messages=[
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="PLAN_TIMESTAMP_PARSE_FAILED",
                    message="No timestamp samples are available to validate the manual format.",
                    affected_column=affected_column,
                    suggested_action="Choose a detected timestamp column or clear the override.",
                )
            ],
        )

    parsed = 0
    exception_text: str | None = None
    for sample in clean_samples:
        try:
            datetime.strptime(sample, fmt)
            parsed += 1
        except ValueError as exc:
            if exception_text is None:
                exception_text = str(exc)

    messages: list[ValidationMessage] = []
    if parsed == len(clean_samples):
        messages.append(
            ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="PLAN_TIMESTAMP_FORMAT_VALID",
                message=f"Parsed {parsed}/{len(clean_samples)} samples successfully.",
                affected_column=affected_column,
            )
        )
    elif parsed == 0:
        detail = f" First failure: {exception_text}" if exception_text else ""
        messages.append(
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="PLAN_TIMESTAMP_PARSE_FAILED",
                message=(
                    f"Manual timestamp format parsed 0/{len(clean_samples)} samples."
                    f"{detail}"
                ),
                affected_column=affected_column,
                suggested_action="Correct the format or clear the override.",
            )
        )
    else:
        detail = f" First failure: {exception_text}" if exception_text else ""
        messages.append(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="PLAN_PARTIAL_TIMESTAMP_PARSE",
                message=(
                    f"Manual timestamp format parsed {parsed}/{len(clean_samples)} "
                    f"samples successfully.{detail}"
                ),
                affected_column=affected_column,
                suggested_action="Review malformed rows before importing.",
            )
        )

    return TimestampFormatValidationResult(
        format_text=fmt,
        sample_count=len(clean_samples),
        parsed_count=parsed,
        validation_messages=messages,
        exception_text=exception_text,
    )
