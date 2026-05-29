"""Tests for oprim.canonical_json (RFC 8785 JCS)."""

import pytest

from oprim.serialization.canonical import canonical_json


def test_canonical_json_simple_dict():
    result = canonical_json({"a": 1, "b": 2})
    assert result == '{"a":1,"b":2}'


def test_canonical_json_sorts_keys():
    result = canonical_json({"b": 1, "a": 2})
    assert result == '{"a":2,"b":1}'


def test_canonical_json_nested_dict_sorts_all_levels():
    result = canonical_json({"z": {"b": 1, "a": 2}, "a": 3})
    assert result == '{"a":3,"z":{"a":2,"b":1}}'


def test_canonical_json_no_whitespace():
    result = canonical_json({"key": "value", "num": 42})
    assert " " not in result


def test_canonical_json_unicode_string():
    result = canonical_json("日本語")
    assert "日本語" in result


def test_canonical_json_invalid_type_raises():
    with pytest.raises(TypeError):
        canonical_json({1: "int key"})
    with pytest.raises(TypeError):
        canonical_json(object())


def test_canonical_json_deterministic():
    obj = {"z": 1, "a": [3, 1, 2], "m": {"k": 0}}
    assert canonical_json(obj) == canonical_json(obj)


def test_canonical_json_none():
    assert canonical_json(None) == "null"


def test_canonical_json_bool():
    assert canonical_json(True) == "true"
    assert canonical_json(False) == "false"


def test_canonical_json_list():
    assert canonical_json([1, 2, 3]) == "[1,2,3]"


def test_canonical_json_nan_raises():
    import math
    with pytest.raises(ValueError):
        canonical_json(math.nan)


@pytest.mark.academic_reference
def test_canonical_json_rfc8785_test_vectors():
    """RFC 8785 §3 determinism and sorting test vectors."""
    # Simple object
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    # Nested object
    assert canonical_json({"c": {"d": 4, "e": 5}, "a": 1}) == '{"a":1,"c":{"d":4,"e":5}}'
    # Integer vs float representation
    assert canonical_json(1) == "1"
    assert canonical_json(1.0) == "1"
    # Array preserved order
    assert canonical_json([3, 1, 2]) == "[3,1,2]"
    # Empty objects
    assert canonical_json({}) == "{}"
    assert canonical_json([]) == "[]"


def test_canonical_json_zero_float():
    assert canonical_json(0.0) == "0"


def test_canonical_json_non_integer_float():
    result = canonical_json(1.5)
    assert "1.5" in result


def test_canonical_json_inf_raises():
    import math
    with pytest.raises(ValueError):
        canonical_json(math.inf)
    with pytest.raises(ValueError):
        canonical_json(-math.inf)
