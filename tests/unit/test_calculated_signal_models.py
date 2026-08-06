"""Unit tests for app.calculated_signals.models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import numpy as np
import pytest

from app.calculated_signals.models import (
    CalculatedSignalDefinition,
    CalculatedSignalResult,
    CalculationStatus,
    ChannelRef,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# ChannelRef
# ─────────────────────────────────────────────────────────────────────────────


class TestChannelRef:
    def test_valid_values(self) -> None:
        ref = ChannelRef(source_id="src-1", channel_name="Ia")
        assert ref.source_id == "src-1"
        assert ref.channel_name == "Ia"

    def test_empty_source_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            ChannelRef(source_id="", channel_name="Ia")

    def test_empty_channel_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_name"):
            ChannelRef(source_id="src-1", channel_name="")

    def test_whitespace_only_source_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            ChannelRef(source_id="   ", channel_name="Ia")

    def test_whitespace_only_channel_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_name"):
            ChannelRef(source_id="src-1", channel_name="\t\n")

    def test_leading_trailing_whitespace_is_preserved_not_trimmed(self) -> None:
        # Only blank-ness is validated; the original string is never rewritten.
        ref = ChannelRef(source_id="src-1", channel_name=" Ia ")
        assert ref.channel_name == " Ia "

    def test_equality(self) -> None:
        a = ChannelRef(source_id="src-1", channel_name="Ia")
        b = ChannelRef(source_id="src-1", channel_name="Ia")
        c = ChannelRef(source_id="src-2", channel_name="Ia")
        assert a == b
        assert a != c

    def test_hashable_and_usable_as_dict_key(self) -> None:
        a = ChannelRef(source_id="src-1", channel_name="Ia")
        b = ChannelRef(source_id="src-1", channel_name="Ia")
        d = {a: "value"}
        assert d[b] == "value"
        assert hash(a) == hash(b)

    def test_immutable(self) -> None:
        ref = ChannelRef(source_id="src-1", channel_name="Ia")
        with pytest.raises(FrozenInstanceError):
            ref.source_id = "src-2"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# CalculatedSignalDefinition
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculatedSignalDefinition:
    def _bindings(self) -> dict[str, ChannelRef]:
        return {
            "A": ChannelRef(source_id="s1", channel_name="Ia"),
            "B": ChannelRef(source_id="s1", channel_name="Ib"),
        }

    def test_valid_definition(self) -> None:
        defn = CalculatedSignalDefinition(
            calc_id="c1",
            name="Sum",
            expression="A + B",
            variable_bindings=self._bindings(),
            reference_variable="A",
        )
        assert defn.calc_id == "c1"
        assert defn.reference_variable == "A"
        assert defn.interpolation == "linear"
        assert defn.output_unit is None
        assert defn.created_at.tzinfo is not None

    def test_empty_calc_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="calc_id"):
            CalculatedSignalDefinition(
                calc_id="", name="Sum", expression="A + B",
                variable_bindings=self._bindings(), reference_variable="A",
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            CalculatedSignalDefinition(
                calc_id="c1", name="", expression="A + B",
                variable_bindings=self._bindings(), reference_variable="A",
            )

    def test_empty_expression_rejected(self) -> None:
        with pytest.raises(ValueError, match="expression"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="",
                variable_bindings=self._bindings(), reference_variable="A",
            )

    def test_empty_binding_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="variable_bindings"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A + B",
                variable_bindings={}, reference_variable="A",
            )

    @pytest.mark.parametrize("bad_name", ["1A", "A-B", "A B", "A.B", ""])
    def test_invalid_variable_names_rejected(self, bad_name: str) -> None:
        with pytest.raises(ValueError):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A",
                variable_bindings={bad_name: ChannelRef("s1", "Ia")},
                reference_variable=bad_name,
            )

    def test_underscore_prefixed_variable_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="_"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="_A",
                variable_bindings={"_A": ChannelRef("s1", "Ia")},
                reference_variable="_A",
            )

    def test_keyword_variable_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="keyword"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="for",
                variable_bindings={"for": ChannelRef("s1", "Ia")},
                reference_variable="for",
            )

    def test_reserved_abs_binding_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="abs",
                variable_bindings={"abs": ChannelRef("s1", "Ia")},
                reference_variable="abs",
            )

    def test_non_channelref_binding_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="ChannelRef"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A",
                variable_bindings={"A": "not-a-channelref"},  # type: ignore[dict-item]
                reference_variable="A",
            )

    def test_reference_variable_missing_from_bindings_rejected(self) -> None:
        with pytest.raises(ValueError, match="reference_variable"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A + B",
                variable_bindings=self._bindings(), reference_variable="C",
            )

    def test_unsupported_interpolation_rejected(self) -> None:
        with pytest.raises(ValueError, match="interpolation"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A + B",
                variable_bindings=self._bindings(), reference_variable="A",
                interpolation="cubic",
            )

    def test_naive_created_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            CalculatedSignalDefinition(
                calc_id="c1", name="Sum", expression="A + B",
                variable_bindings=self._bindings(), reference_variable="A",
                created_at=datetime.now(),  # naive
            )

    def test_default_created_at_is_timezone_aware_utc(self) -> None:
        before = _utc_now()
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="Sum", expression="A + B",
            variable_bindings=self._bindings(), reference_variable="A",
        )
        after = _utc_now()
        assert defn.created_at.tzinfo is not None
        assert before - timedelta(seconds=1) <= defn.created_at <= after + timedelta(seconds=1)

    def test_default_created_at_uses_factory_not_import_time(self) -> None:
        # Two definitions created a moment apart must not share a timestamp
        # frozen at module-import time.
        first = CalculatedSignalDefinition(
            calc_id="c1", name="Sum", expression="A + B",
            variable_bindings=self._bindings(), reference_variable="A",
        )
        import time as _time
        _time.sleep(0.01)
        second = CalculatedSignalDefinition(
            calc_id="c2", name="Sum", expression="A + B",
            variable_bindings=self._bindings(), reference_variable="A",
        )
        assert second.created_at > first.created_at

    def test_binding_map_is_defensively_owned(self) -> None:
        original = self._bindings()
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="Sum", expression="A + B",
            variable_bindings=original, reference_variable="A",
        )
        original["A"] = ChannelRef("s2", "different")
        # Mutating the caller's dict after construction must not affect the definition.
        assert defn.variable_bindings["A"] == ChannelRef("s1", "Ia")

    def test_binding_map_is_immutable(self) -> None:
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="Sum", expression="A + B",
            variable_bindings=self._bindings(), reference_variable="A",
        )
        assert isinstance(defn.variable_bindings, MappingProxyType)
        with pytest.raises(TypeError):
            defn.variable_bindings["C"] = ChannelRef("s1", "Ic")  # type: ignore[index]

    def test_definition_itself_is_immutable(self) -> None:
        defn = CalculatedSignalDefinition(
            calc_id="c1", name="Sum", expression="A + B",
            variable_bindings=self._bindings(), reference_variable="A",
        )
        with pytest.raises(FrozenInstanceError):
            defn.name = "Renamed"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# CalculatedSignalResult
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculatedSignalResult:
    def _ok_result(self, **overrides) -> CalculatedSignalResult:
        kwargs = dict(
            calc_id="c1",
            time=np.array([0.0, 1.0, 2.0]),
            values=np.array([10.0, 20.0, 30.0]),
            validity_mask=np.array([True, True, False]),
            unit="A",
            status=CalculationStatus.OK,
            error_message=None,
            computed_at=_utc_now(),
        )
        kwargs.update(overrides)
        return CalculatedSignalResult(**kwargs)

    def test_valid_arrays(self) -> None:
        result = self._ok_result()
        assert len(result.time) == 3
        assert result.time.dtype == np.float64
        assert result.validity_mask.dtype == bool

    def test_arrays_are_copied_not_aliased(self) -> None:
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([10.0, 20.0, 30.0])
        m = np.array([True, True, False])
        result = self._ok_result(time=t, values=v, validity_mask=m)
        t[0] = 999.0
        v[0] = 999.0
        m[0] = False
        assert result.time[0] == 0.0
        assert result.values[0] == 10.0
        assert result.validity_mask[0] == True  # noqa: E712

    def test_one_dimensional_enforced_for_time(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            self._ok_result(time=np.array([[0.0, 1.0], [2.0, 3.0]]))

    def test_one_dimensional_enforced_for_values(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            self._ok_result(values=np.array([[10.0, 20.0]]))

    def test_one_dimensional_enforced_for_validity_mask(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            self._ok_result(validity_mask=np.array([[True, False]]))

    def test_length_mismatch_time_values(self) -> None:
        with pytest.raises(ValueError, match="identical length"):
            self._ok_result(time=np.array([0.0, 1.0, 2.0]), values=np.array([10.0, 20.0]))

    def test_length_mismatch_validity_mask(self) -> None:
        with pytest.raises(ValueError, match="identical length"):
            self._ok_result(validity_mask=np.array([True, False]))

    def test_non_numeric_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            self._ok_result(time=np.array(["a", "b", "c"]))

    def test_non_numeric_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            self._ok_result(values=np.array(["a", "b", "c"]))

    def test_object_dtype_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            self._ok_result(time=np.array([0.0, "x", 2.0], dtype=object))

    def test_validity_mask_accepts_int_array(self) -> None:
        result = self._ok_result(validity_mask=np.array([1, 0, 1]))
        assert result.validity_mask.dtype == bool
        assert list(result.validity_mask) == [True, False, True]

    def test_validity_mask_rejects_string_array(self) -> None:
        with pytest.raises(ValueError, match="validity_mask"):
            self._ok_result(validity_mask=np.array(["true", "false", "true"]))

    def test_warnings_list_is_copied_defensively(self) -> None:
        warnings_list = ["something"]
        result = self._ok_result(warnings=warnings_list)
        warnings_list.append("mutated after construction")
        assert result.warnings == ["something"]

    def test_naive_computed_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            self._ok_result(computed_at=datetime.now())

    def test_error_status_requires_message(self) -> None:
        with pytest.raises(ValueError, match="error_message"):
            self._ok_result(status=CalculationStatus.ERROR, error_message=None)

    def test_error_status_rejects_blank_message(self) -> None:
        with pytest.raises(ValueError, match="error_message"):
            self._ok_result(status=CalculationStatus.ERROR, error_message="   ")

    def test_error_status_with_message_is_valid(self) -> None:
        result = self._ok_result(status=CalculationStatus.ERROR, error_message="division failed")
        assert result.status == CalculationStatus.ERROR
        assert result.error_message == "division failed"

    def test_ok_status_does_not_require_message(self) -> None:
        result = self._ok_result(status=CalculationStatus.OK, error_message=None)
        assert result.error_message is None

    def test_stale_status_does_not_require_message(self) -> None:
        result = self._ok_result(status=CalculationStatus.STALE, error_message=None)
        assert result.status == CalculationStatus.STALE

    def test_mutable_fields_can_be_reassigned(self) -> None:
        # CalculatedSignalResult is intentionally mutable (unlike the
        # definition) so a future owner can mark it stale in place.
        result = self._ok_result()
        result.status = CalculationStatus.STALE
        assert result.status == CalculationStatus.STALE

    def test_equality_for_identical_results(self) -> None:
        t, v, m = np.array([0.0]), np.array([1.0]), np.array([True])
        now = _utc_now()
        a = self._ok_result(time=t, values=v, validity_mask=m, computed_at=now)
        b = self._ok_result(time=t.copy(), values=v.copy(), validity_mask=m.copy(), computed_at=now)
        # Dataclass equality compares array fields with NumPy's __eq__,
        # which is elementwise; assert field-by-field instead of `a == b`.
        assert np.array_equal(a.time, b.time)
        assert np.array_equal(a.values, b.values)
        assert np.array_equal(a.validity_mask, b.validity_mask)
        assert a.status == b.status
