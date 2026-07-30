from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from oprim._exceptions import OprimError
from oprim.mathlib_lookup import mathlib_lookup


def test_mathlib_lookup_unique_hit(mocker):
    # Mock opener
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)

    # Mock response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "hits": [
            {
                "name": "Nat.add_comm",
                "module": "Mathlib.Algebra.Group.Nat",
                "type": "∀ (n m : ℕ), n + m = m + n"
            }
        ]
    }).encode("utf-8")
    mock_opener.open.return_value.__enter__.return_value = mock_resp

    result = mathlib_lookup(identifier="Nat.add_comm")

    assert result.count == 1
    assert result.hits[0].name == "Nat.add_comm"
    assert result.hits[0].module == "Mathlib.Algebra.Group.Nat"
    assert result.hits[0].type_signature == "∀ (n m : ℕ), n + m = m + n"


def test_mathlib_lookup_multiple_hits(mocker):
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "hits": [
            {"name": "hit1", "module": "mod1", "type": "sig1"},
            {"name": "hit2", "module": "mod2", "type": "sig2"}
        ]
    }).encode("utf-8")
    mock_opener.open.return_value.__enter__.return_value = mock_resp

    result = mathlib_lookup(identifier="some_id")
    assert result.count == 2
    assert len(result.hits) == 2


def test_mathlib_lookup_no_hits(mocker):
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"hits": []}).encode("utf-8")
    mock_opener.open.return_value.__enter__.return_value = mock_resp

    result = mathlib_lookup(identifier="absent")
    assert result.count == 0
    assert result.hits == []


def test_mathlib_lookup_api_error(mocker):
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)

    mock_resp = MagicMock()
    mock_resp.status = 500
    mock_opener.open.return_value.__enter__.return_value = mock_resp

    with pytest.raises(OprimError, match="API request failed with status 500"):
        mathlib_lookup(identifier="error")


def test_mathlib_lookup_timeout(mocker):
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)
    mock_opener.open.side_effect = Exception("Connect timeout")

    with pytest.raises(OprimError, match="API request timed out"):
        mathlib_lookup(identifier="timeout")


def test_mathlib_lookup_invalid_json(mocker):
    mock_opener = MagicMock()
    mocker.patch("oprim.mathlib_lookup.make_ssrf_safe_opener", return_value=mock_opener)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"invalid json"
    mock_opener.open.return_value.__enter__.return_value = mock_resp

    with pytest.raises(OprimError, match="Invalid JSON response"):
        mathlib_lookup(identifier="badjson")


def test_mathlib_lookup_empty_identifier():
    with pytest.raises(OprimError, match="Identifier cannot be empty"):
        mathlib_lookup(identifier="")
