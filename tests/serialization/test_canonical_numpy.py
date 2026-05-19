"""Tests: canonical_json handles numpy scalars correctly (numpy 2.x regression guard)."""
import json

import numpy as np
import pytest

from oprim.serialization.canonical import canonical_json


def test_float64_fractional():
    """np.float64(0.5) must not become 'np.float64(0.5)' in numpy ≥ 2.0."""
    result = canonical_json(np.float64(0.5))
    assert result == "0.5", f"got {result!r}"


def test_float64_integer_valued():
    """np.float64(100.0) should be serialised as integer '100', not '100.0'."""
    result = canonical_json(np.float64(100.0))
    assert result == "100", f"got {result!r}"


def test_int64():
    result = canonical_json(np.int64(42))
    assert result == "42", f"got {result!r}"


def test_numpy_bool_true():
    result = canonical_json(np.bool_(True))
    assert result == "true", f"got {result!r}"


def test_numpy_bool_false():
    result = canonical_json(np.bool_(False))
    assert result == "false", f"got {result!r}"


def test_nested_dict_with_numpy_values():
    """dict containing numpy scalars must round-trip via canonical_json correctly."""
    payload = {
        "target_notional_usd": np.float64(5000.5),
        "symbol": "BTC-USDT",
        "quantity": np.int64(2),
    }
    result = canonical_json(payload)
    parsed = json.loads(result)
    assert parsed["target_notional_usd"] == 5000.5
    assert parsed["quantity"] == 2
    assert "np.float64" not in result
    assert "np.int64" not in result


def test_nested_dict_deterministic():
    """Two dicts with same numpy values must produce identical canonical JSON."""
    a = canonical_json({"v": np.float64(1.5), "w": np.float64(2.5)})
    b = canonical_json({"v": np.float64(1.5), "w": np.float64(2.5)})
    assert a == b


def test_list_of_numpy_floats():
    result = canonical_json([np.float64(0.1), np.float64(0.2), np.float64(0.3)])
    parsed = json.loads(result)
    assert len(parsed) == 3
    assert "np.float64" not in result


def test_np_sign_result():
    """np.sign() returns np.float64 — must serialize as native int/float."""
    val = 1000.0 * np.sign(0.7)
    result = canonical_json(val)
    assert result == "1000", f"got {result!r}"
    assert "np" not in result
