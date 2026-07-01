"""Tests for oprim.url_fetch_ssrf_safe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.url_fetch_ssrf_safe import url_fetch_ssrf_safe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    *,
    status: int = 200,
    content_type: str = "text/html",
    body: bytes = b"hello world",
):
    resp = MagicMock()
    resp.status = status
    resp.headers = {"Content-Type": content_type}
    resp.read = MagicMock(return_value=body)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_opener(resp):
    opener = MagicMock()
    opener.open = MagicMock(return_value=resp)
    return opener


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ssrf_blocked_private_ip_returns_error():
    from obase.http.dns_pinned_transport import SSRFBlockedError

    opener = MagicMock()
    opener.open.side_effect = SSRFBlockedError("blocked")

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://192.168.1.1/secret")

    assert result["error"] == "ssrf_blocked"
    assert result["body_bytes"] == b""


def test_successful_fetch_returns_body():
    resp = _make_mock_response(body=b"response body")
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/data")

    assert result["body_bytes"] == b"response body"
    assert result["error"] is None


def test_timeout_parameter_passed_to_opener():
    resp = _make_mock_response()
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener) as mock_make:
        url_fetch_ssrf_safe(url="http://example.com/", timeout=30)

    mock_make.assert_called_once_with(timeout=30)


def test_max_bytes_truncation():
    resp = _make_mock_response(body=b"x" * 100)
    resp.read = MagicMock(return_value=b"x" * 10)  # simulates truncation at read level
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/", max_bytes=10)

    assert len(result["body_bytes"]) <= 10


def test_content_type_extracted_from_headers():
    resp = _make_mock_response(content_type="application/json")
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/api")

    assert result["content_type"] == "application/json"


def test_status_code_returned():
    resp = _make_mock_response(status=404)
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/missing")

    assert result["status_code"] == 404


def test_url_in_result():
    resp = _make_mock_response()
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/page")

    assert result["url"] == "http://example.com/page"


def test_error_none_on_success():
    resp = _make_mock_response()
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/")

    assert result["error"] is None


def test_ssrf_blocked_error_caught():
    from obase.http.dns_pinned_transport import SSRFBlockedError

    opener = MagicMock()
    opener.open.side_effect = SSRFBlockedError("private range")

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://10.0.0.1/")

    assert result["error"] == "ssrf_blocked"
    assert result["status_code"] is None


def test_headers_forwarded_to_request():
    import urllib.request

    resp = _make_mock_response()
    opener = _make_opener(resp)

    captured_req = {}

    def _open(req, timeout=None):
        captured_req["headers"] = dict(req.headers)
        return resp

    opener.open = _open

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        url_fetch_ssrf_safe(url="http://example.com/", headers={"X-Custom": "value"})

    assert captured_req["headers"].get("X-custom") == "value"


def test_body_text_decoded():
    resp = _make_mock_response(body=b"hello utf8")
    opener = _make_opener(resp)

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/")

    assert result["body_text"] == "hello utf8"


def test_generic_exception_sets_error():
    opener = MagicMock()
    opener.open.side_effect = ConnectionError("network down")

    with patch("oprim.url_fetch_ssrf_safe.make_ssrf_safe_opener", return_value=opener):
        result = url_fetch_ssrf_safe(url="http://example.com/")

    assert result["error"] is not None
    assert "network down" in result["error"]
