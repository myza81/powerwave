"""Tests for the shared, UI-independent channel-name matching helper.

This module (app.data.channel_name_matching) is consumed by
app.data.column_classifier, app.data.intelligence.intelligence_manager,
app.import_wizard.column_detector, and the CSV/Excel providers'
_infer_unit() -- see those modules' own tests for classifier-level
behaviour. These tests cover only the shared primitives themselves:
tokenization, exact/prefix token matching, phrase matching, and
status/control qualifier detection.
"""
from __future__ import annotations

from app.data.channel_name_matching import (
    has_exact_token,
    has_status_qualifier,
    has_token_phrase,
    has_token_prefix,
    tokenize_channel_name,
)


class TestTokenizeChannelName:
    def test_whitespace_separated(self) -> None:
        assert tokenize_channel_name("Voltage Status") == ("voltage", "status")

    def test_underscore_separated(self) -> None:
        assert tokenize_channel_name("Voltage_Status") == ("voltage", "status")
        assert tokenize_channel_name("P_TOTAL") == ("p", "total")

    def test_hyphen_separated(self) -> None:
        assert tokenize_channel_name("Voltage-Status") == ("voltage", "status")

    def test_slash_separated(self) -> None:
        assert tokenize_channel_name("df/dt") == ("df", "dt")

    def test_camel_case(self) -> None:
        assert tokenize_channel_name("VoltageStatus") == ("voltage", "status")
        assert tokenize_channel_name("CurrentState") == ("current", "state")

    def test_all_caps_underscore_form(self) -> None:
        assert tokenize_channel_name("CURRENT_STATE") == ("current", "state")

    def test_compact_status_forms_stay_one_token(self) -> None:
        # "MWStatus" has no lowercase-to-uppercase transition (M,W,S are all
        # uppercase, then a lowercase run) -- deliberately not split, since
        # the same shape is used by "MVar"/"MW"/"ROCOF" abbreviations that
        # must NOT be split apart. Staying one opaque token still correctly
        # fails to match any known measurement token.
        assert tokenize_channel_name("MWStatus") == ("mwstatus",)

    def test_relay_tokens_preserved(self) -> None:
        for name, expected in [
            ("Vab", ("vab",)),
            ("Vbc", ("vbc",)),
            ("Ia", ("ia",)),
            ("I0", ("i0",)),
            ("V1", ("v1",)),
            ("V2", ("v2",)),
        ]:
            assert tokenize_channel_name(name) == expected

    def test_abbreviations_not_split(self) -> None:
        assert tokenize_channel_name("MW") == ("mw",)
        assert tokenize_channel_name("MVar") == ("mvar",)
        assert tokenize_channel_name("MVAr") == ("mvar",)
        assert tokenize_channel_name("ROCOF") == ("rocof",)

    def test_unrelated_collision_words_stay_whole(self) -> None:
        for name in ["Occurrence", "Example", "Impulse", "Interval"]:
            tokens = tokenize_channel_name(name)
            assert tokens == (name.lower(),)

    def test_empty_and_blank(self) -> None:
        assert tokenize_channel_name("") == ()
        assert tokenize_channel_name("   ") == ()


class TestHasExactToken:
    def test_matches_whole_token(self) -> None:
        assert has_exact_token("Va", ("va", "vb"))
        assert has_exact_token("Bus Voltage", ("voltage",))

    def test_does_not_match_substring(self) -> None:
        assert not has_exact_token("Occurrence", ("curr",))
        assert not has_exact_token("Example", ("amp",))

    def test_accepts_pre_tokenized_input(self) -> None:
        assert has_exact_token(("va", "status"), ("va",))


class TestHasTokenPrefix:
    def test_matches_token_start(self) -> None:
        assert has_token_prefix("Voltage", ("volt",))
        assert has_token_prefix("Current", ("curr",))

    def test_does_not_match_mid_word(self) -> None:
        assert not has_token_prefix("Occurrence", ("curr",))
        assert not has_token_prefix("Example", ("amp",))

    def test_empty_prefixes_never_match(self) -> None:
        assert not has_token_prefix("Voltage", ())


class TestHasTokenPhrase:
    def test_single_word_phrase(self) -> None:
        assert has_token_phrase("Bus Voltage", "voltage")

    def test_multi_word_phrase_requires_contiguous_tokens(self) -> None:
        assert has_token_phrase("Bus Voltage", "bus voltage")
        assert has_token_phrase("BusVoltage", "bus voltage")
        assert not has_token_phrase("Voltage Bus", "bus voltage")

    def test_no_match_when_tokens_not_adjacent(self) -> None:
        assert not has_token_phrase("Bus Something Voltage", "bus voltage")


class TestHasStatusQualifier:
    def test_detects_common_qualifiers(self) -> None:
        for name in [
            "Voltage Status", "VoltageStatus", "Voltage Alarm", "Voltage State",
            "Voltage Control", "Current Status", "Current State", "Current Alarm",
            "Frequency Alarm", "Frequency Status", "MW Status",
            "Active Power Alarm", "Reactive Power Status",
        ]:
            assert has_status_qualifier(name), f"{name!r} should carry a qualifier"

    def test_compact_form_without_delimiter_is_not_a_detected_qualifier(self) -> None:
        # "MWStatus" has no delimiter and no lowercase-to-uppercase
        # transition (see TestTokenizeChannelName), so it tokenizes to one
        # opaque token ("mwstatus") that has_status_qualifier does not
        # recognise. This is safe by a different mechanism: that same
        # opaque token also does not match any measurement token ("mw"),
        # so callers still end up with no analog measurement -- see
        # test_column_classifier.py / test_import_wizard_column_detection.py
        # for the end-to-end proof.
        assert not has_status_qualifier("MWStatus")

    def test_no_false_positive_on_valid_measurement_names(self) -> None:
        for name in [
            "Voltage", "Bus Voltage", "Phase Voltage", "Va", "Vab", "Current",
            "Phase Current", "Ia", "Active Power", "Real Power", "System Demand",
            "Total Demand", "MW", "Reactive Power", "MVar", "MVAr", "Frequency",
            "System Frequency", "Freq", "ROCOF", "df/dt",
        ]:
            assert not has_status_qualifier(name), f"{name!r} should not carry a qualifier"

    def test_no_false_positive_on_domain_ambiguous_words(self) -> None:
        # "load"/"output"/"demand" are intentionally excluded from the
        # qualifier list -- they are also legitimate parts of real
        # measurement names in this repository's own vocabulary.
        assert not has_status_qualifier("Load Demand")
        assert not has_status_qualifier("Plant Output")
        assert not has_status_qualifier("Net Generation")

    def test_no_false_positive_on_unrelated_collision_words(self) -> None:
        for name in ["Occurrence", "Example", "Input", "Index", "Interval", "Info", "Variable", "Pump", "Impulse"]:
            assert not has_status_qualifier(name)
