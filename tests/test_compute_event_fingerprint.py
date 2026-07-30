"""Tests for oprim.compute_event_fingerprint (oprim 2.24.0)."""

from __future__ import annotations

import re

import pytest

from oprim.compute_event_fingerprint import compute_event_fingerprint

_TYPE = "TypeError"
_VALUE = "unsupported operand type(s) for +: 'int' and 'str'"
_FUNC = "process_payment"
_FILE = "aegis/payment.py"


def test_deterministic_same_inputs() -> None:
    fp1 = compute_event_fingerprint(
        exception_type=_TYPE,
        exception_value=_VALUE,
        top_frame_function=_FUNC,
        top_frame_filename=_FILE,
    )
    fp2 = compute_event_fingerprint(
        exception_type=_TYPE,
        exception_value=_VALUE,
        top_frame_function=_FUNC,
        top_frame_filename=_FILE,
    )
    assert fp1 == fp2


def test_different_exception_type_gives_different_fp() -> None:
    fp1 = compute_event_fingerprint(exception_type="TypeError", exception_value=_VALUE)
    fp2 = compute_event_fingerprint(exception_type="ValueError", exception_value=_VALUE)
    assert fp1 != fp2


def test_different_exception_value_gives_different_fp() -> None:
    fp1 = compute_event_fingerprint(exception_type=_TYPE, exception_value="msg A")
    fp2 = compute_event_fingerprint(exception_type=_TYPE, exception_value="msg B")
    assert fp1 != fp2


def test_different_top_frame_function_gives_different_fp() -> None:
    fp1 = compute_event_fingerprint(exception_type=_TYPE, top_frame_function="fn_a")
    fp2 = compute_event_fingerprint(exception_type=_TYPE, top_frame_function="fn_b")
    assert fp1 != fp2


def test_different_top_frame_filename_gives_different_fp() -> None:
    fp1 = compute_event_fingerprint(exception_type=_TYPE, top_frame_filename="a.py")
    fp2 = compute_event_fingerprint(exception_type=_TYPE, top_frame_filename="b.py")
    assert fp1 != fp2


def test_none_exception_value_same_as_empty_string() -> None:
    fp_none = compute_event_fingerprint(exception_type=_TYPE, exception_value=None)
    fp_empty = compute_event_fingerprint(exception_type=_TYPE, exception_value="")
    assert fp_none == fp_empty


def test_custom_fingerprint_overrides_default() -> None:
    custom = ["payment-flow", "retry-exhausted"]
    fp_custom = compute_event_fingerprint(
        exception_type="SomeOtherError",
        exception_value="whatever",
        custom_fingerprint=custom,
    )
    fp_same_custom = compute_event_fingerprint(
        exception_type="DifferentError",
        exception_value="different message",
        custom_fingerprint=custom,
    )
    assert fp_custom == fp_same_custom


def test_different_custom_fingerprint_lengths_differ() -> None:
    fp1 = compute_event_fingerprint(exception_type=_TYPE, custom_fingerprint=["a"])
    fp2 = compute_event_fingerprint(exception_type=_TYPE, custom_fingerprint=["a", "b"])
    assert fp1 != fp2


def test_returns_64_char_hex() -> None:
    fp = compute_event_fingerprint(exception_type=_TYPE, exception_value=_VALUE)
    assert len(fp) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", fp) is not None


def test_empty_exception_type_raises() -> None:
    with pytest.raises(ValueError, match="exception_type cannot be empty"):
        compute_event_fingerprint(exception_type="")


def test_null_byte_separator_no_collision() -> None:
    # "a|b" + "c"  vs  "a" + "b|c" — would collide with "|" separator
    # With "\x00" separator these are distinct
    fp1 = compute_event_fingerprint(exception_type="a|b", exception_value="c")
    fp2 = compute_event_fingerprint(exception_type="a", exception_value="b|c")
    assert fp1 != fp2


def test_all_none_optional_fields_deterministic() -> None:
    fp = compute_event_fingerprint(exception_type=_TYPE)
    assert len(fp) == 64
    assert fp == compute_event_fingerprint(exception_type=_TYPE)


# --- dual-form import verification ---


def test_dual_form_direct_import() -> None:
    from oprim.compute_event_fingerprint import compute_event_fingerprint as cef

    assert callable(cef)


def test_dual_form_top_level_import() -> None:
    from oprim import compute_event_fingerprint as cef

    assert callable(cef)


def test_in_all() -> None:
    import oprim

    assert "compute_event_fingerprint" in oprim.__all__
