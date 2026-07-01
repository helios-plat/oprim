"""Tests for oprim.keyword_alert_checker."""

from __future__ import annotations

import pytest

from oprim.keyword_alert_checker import keyword_alert_checker


def test_exact_match_found():
    result = keyword_alert_checker(text="alert: disk full", keywords=["disk"])
    assert result["has_alerts"] is True
    assert any(m["keyword"] == "disk" for m in result["matches"])


def test_exact_match_not_found():
    result = keyword_alert_checker(text="everything is fine", keywords=["error"])
    assert result["has_alerts"] is False
    assert result["matches"] == []


def test_case_insensitive_match():
    result = keyword_alert_checker(
        text="Alert: System Failure", keywords=["alert"], case_sensitive=False
    )
    assert result["has_alerts"] is True


def test_case_sensitive_no_match():
    result = keyword_alert_checker(
        text="Alert: System Failure", keywords=["alert"], case_sensitive=True
    )
    assert result["has_alerts"] is False


def test_regex_match():
    result = keyword_alert_checker(
        text="error code 404 detected", keywords=[r"error\s+code\s+\d+"], mode="regex"
    )
    assert result["has_alerts"] is True


def test_multiple_keywords_each_with_positions():
    result = keyword_alert_checker(
        text="foo bar foo baz bar", keywords=["foo", "bar"], mode="exact"
    )
    assert len(result["matches"]) == 2
    foo_match = next(m for m in result["matches"] if m["keyword"] == "foo")
    bar_match = next(m for m in result["matches"] if m["keyword"] == "bar")
    assert len(foo_match["positions"]) == 2
    assert len(bar_match["positions"]) == 2


def test_empty_text_returns_zero_matches():
    result = keyword_alert_checker(text="", keywords=["error"])
    assert result["total_matches"] == 0
    assert result["has_alerts"] is False


def test_invalid_regex_sets_error():
    result = keyword_alert_checker(text="some text", keywords=["[invalid"], mode="regex")
    assert result["error"] is not None


def test_total_matches_sum_correct():
    result = keyword_alert_checker(text="foo foo bar", keywords=["foo", "bar"], mode="exact")
    expected = sum(m["count"] for m in result["matches"])
    assert result["total_matches"] == expected


def test_positions_are_list_of_ints():
    result = keyword_alert_checker(text="error here and error there", keywords=["error"])
    assert result["has_alerts"] is True
    for m in result["matches"]:
        assert isinstance(m["positions"], list)
        for pos in m["positions"]:
            assert isinstance(pos, int)


def test_has_alerts_true_when_any_match():
    result = keyword_alert_checker(text="normal text with one error", keywords=["ok", "error"])
    assert result["has_alerts"] is True


def test_fuzzy_match_finds_near_match():
    result = keyword_alert_checker(text="erroe detected", keywords=["error"], mode="fuzzy")
    # "erroe" has edit distance 1 from "error" → should match
    assert result["has_alerts"] is True


def test_empty_keywords_returns_no_matches():
    result = keyword_alert_checker(text="some text", keywords=[])
    assert result["total_matches"] == 0
    assert result["has_alerts"] is False
