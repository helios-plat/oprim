"""Tests for oprim.searxng_search."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from oprim.searxng_search import searxng_search

MOCK_RESPONSE = {
    "results": [
        {
            "title": "Result 1",
            "url": "https://r1.example.com",
            "content": "content 1",
            "engine": "bing",
            "score": 0.9,
        },
        {
            "title": "Result 2",
            "url": "https://r2.example.com",
            "content": "content 2",
            "engine": "duckduckgo",
            "score": 0.7,
        },
    ]
}

MOCK_5_RESULTS = {
    "results": [
        {
            "title": f"Result {i}",
            "url": f"https://r{i}.example.com",
            "content": f"content {i}",
            "engine": "bing",
            "score": 0.9 - i * 0.1,
        }
        for i in range(1, 6)
    ]
}


def _mock_opener(mock_body: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_body).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_resp
    return mock_opener


def _patch_opener(mock_body: dict):
    return patch(
        "obase.http.dns_pinned_transport.make_ssrf_safe_opener",
        return_value=_mock_opener(mock_body),
    )


# ── test 1: returns dict with required keys ──────────────────────────────────


def test_returns_required_keys():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    assert set(result.keys()) >= {"query", "results", "total", "error"}


# ── test 2: results contain required fields ───────────────────────────────────


def test_result_items_have_required_fields():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    for item in result["results"]:
        assert set(item.keys()) >= {"title", "url", "content", "engine", "score"}


# ── test 3: total == len(results) ─────────────────────────────────────────────


def test_total_equals_len_results():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    assert result["total"] == len(result["results"])


# ── test 4: query preserved in result dict ────────────────────────────────────


def test_query_preserved():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="hello world", searxng_url="http://172.17.0.2:8080")
    assert result["query"] == "hello world"


# ── test 5: max_results limits output ────────────────────────────────────────


def test_max_results_limits_output():
    with _patch_opener(MOCK_5_RESULTS):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080", max_results=2)
    assert len(result["results"]) == 2
    assert result["total"] == 2


# ── test 6: empty query → no HTTP call, empty results ────────────────────────


def test_empty_query_no_http_call():
    with patch("urllib.request.urlopen") as mock_urlopen:
        with patch("obase.http.dns_pinned_transport.make_ssrf_safe_opener") as mock_ssrf:
            result = searxng_search(query="   ", searxng_url="http://172.17.0.2:8080")
    mock_urlopen.assert_not_called()
    mock_ssrf.assert_not_called()
    assert result["results"] == []
    assert result["total"] == 0


# ── test 7: HTTP error → error field set, no raise ───────────────────────────


def test_http_error_sets_error_field():
    with patch(
        "obase.http.dns_pinned_transport.make_ssrf_safe_opener",
        side_effect=Exception("ssrf unavailable"),
    ):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    assert result["error"] is not None
    assert "connection refused" in result["error"]
    assert result["results"] == []


# ── test 8: categories parameter included in API URL ─────────────────────────


def test_categories_in_url():
    captured_urls = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"results": []}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with patch(
        "obase.http.dns_pinned_transport.make_ssrf_safe_opener",
        side_effect=ImportError("not available"),
    ):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            searxng_search(
                query="test",
                searxng_url="http://172.17.0.2:8080",
                categories=["general", "news"],
            )

    assert captured_urls, "urlopen should have been called"
    assert "categories=general%2Cnews" in captured_urls[0] or "categories=" in captured_urls[0]


# ── test 9: time_range parameter included in API URL ─────────────────────────


def test_time_range_in_url():
    captured_urls = []

    def fake_urlopen(req, timeout=None):
        captured_urls.append(req.full_url)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"results": []}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with patch(
        "obase.http.dns_pinned_transport.make_ssrf_safe_opener",
        side_effect=ImportError("not available"),
    ):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            searxng_search(
                query="test",
                searxng_url="http://172.17.0.2:8080",
                time_range="week",
            )

    assert captured_urls, "urlopen should have been called"
    assert "time_range=week" in captured_urls[0]


# ── test 10: score is float ───────────────────────────────────────────────────


def test_score_is_float():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    for item in result["results"]:
        assert isinstance(item["score"], float)


# ── test 11: error=None on successful fetch ───────────────────────────────────


def test_error_none_on_success():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    assert result["error"] is None


# ── test 12: result values match mock data ────────────────────────────────────


def test_result_values_match_mock():
    with _patch_opener(MOCK_RESPONSE):
        result = searxng_search(query="test", searxng_url="http://172.17.0.2:8080")
    assert result["results"][0]["title"] == "Result 1"
    assert result["results"][0]["url"] == "https://r1.example.com"
    assert result["results"][0]["engine"] == "bing"
    assert result["results"][1]["title"] == "Result 2"
    assert result["results"][1]["engine"] == "duckduckgo"
