"""Tests for oprim.http_post_webhook — single-shot webhook delivery (B2)."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from oprim.http_post_webhook import MAX_RESPONSE_BODY_BYTES, WebhookResult, http_post_webhook

URL = "https://example.com/webhook"
PAYLOAD = {"event": "test", "value": 42}


def _make_response(status_code: int, text: str = "ok") -> httpx.Response:
    """Build a minimal httpx.Response for mocking."""
    return httpx.Response(status_code=status_code, text=text)


class TestWebhookSuccess:
    def test_200_returns_success(self) -> None:
        mock_resp = _make_response(200, "received")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is True
        assert result.status_code == 200
        assert result.elapsed_ms > 0
        assert result.error is None

    def test_201_returns_success(self) -> None:
        mock_resp = _make_response(201, "created")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is True
        assert result.status_code == 201


class TestWebhookClientErrors:
    def test_404_returns_http_4xx(self) -> None:
        mock_resp = _make_response(404, "not found")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.error == "http_4xx"
        assert result.status_code == 404

    def test_500_returns_http_5xx(self) -> None:
        mock_resp = _make_response(500, "server error")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.error == "http_5xx"
        assert result.status_code == 500

    def test_502_returns_http_5xx(self) -> None:
        mock_resp = _make_response(502, "bad gateway")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.error == "http_5xx"

    def test_302_returns_http_3xx(self) -> None:
        """follow_redirects=False means 302 is returned as-is and labelled http_3xx."""
        mock_resp = _make_response(302, "")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.status_code == 302
        assert result.error == "http_3xx"


class TestWebhookNetworkErrors:
    def test_timeout_returns_timeout_error(self) -> None:
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.TimeoutException("timed out")
            )
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.error == "timeout"
        assert result.status_code is None
        assert result.elapsed_ms >= 0

    def test_connect_error_returns_connect_failed(self) -> None:
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("connection refused")
            )
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.status_code is None
        assert result.error is not None
        assert result.error.startswith("connect_failed:")

    def test_unexpected_exception_returns_unexpected(self) -> None:
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = RuntimeError(
                "boom"
            )
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.status_code is None
        assert result.error is not None
        assert result.error.startswith("unexpected:")


class TestWebhookPayload:
    def test_payload_not_serializable_via_mock(self) -> None:
        """Force json.dumps to raise TypeError to cover payload_not_serializable path."""
        with patch("oprim.http_post_webhook.json.dumps", side_effect=TypeError("bad")):
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is False
        assert result.status_code is None
        assert result.error is not None
        assert result.error.startswith("payload_not_serializable:")

    def test_complex_types_serialize_with_default_str(self) -> None:
        """default=str means object() in payload still allows POST to proceed."""
        mock_resp = _make_response(200, "ok")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload={"key": object()})
        assert result.success is True


class TestWebhookHeaders:
    def test_signature_header_added(self) -> None:
        """Verify X-Aegis-Signature header is present when signature is passed."""
        mock_resp = _make_response(200, "ok")
        captured_headers: dict[str, str] = {}

        def capture_post(url: str, content: str, headers: dict[str, str]) -> httpx.Response:
            captured_headers.update(headers)
            return mock_resp

        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = capture_post
            result = http_post_webhook(url=URL, payload=PAYLOAD, signature="sig-abc123")

        assert result.success is True
        assert captured_headers.get("X-Aegis-Signature") == "sig-abc123"

    def test_custom_signature_header_name(self) -> None:
        """Custom signature_header kwarg is used instead of default."""
        mock_resp = _make_response(200, "ok")
        captured_headers: dict[str, str] = {}

        def capture_post(url: str, content: str, headers: dict[str, str]) -> httpx.Response:
            captured_headers.update(headers)
            return mock_resp

        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = capture_post
            http_post_webhook(
                url=URL,
                payload=PAYLOAD,
                signature="mysig",
                signature_header="X-Custom-Sig",
            )

        assert captured_headers.get("X-Custom-Sig") == "mysig"
        assert "X-Aegis-Signature" not in captured_headers

    def test_custom_headers_merged(self) -> None:
        """Extra headers passed via headers= kwarg appear in the request."""
        mock_resp = _make_response(200, "ok")
        captured_headers: dict[str, str] = {}

        def capture_post(url: str, content: str, headers: dict[str, str]) -> httpx.Response:
            captured_headers.update(headers)
            return mock_resp

        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.side_effect = capture_post
            http_post_webhook(
                url=URL,
                payload=PAYLOAD,
                headers={"X-Source": "helios", "X-Version": "2"},
            )

        assert captured_headers.get("X-Source") == "helios"
        assert captured_headers.get("X-Version") == "2"


class TestWebhookResponseBody:
    def test_response_body_truncated_to_4096(self) -> None:
        long_body = "x" * 8000
        mock_resp = _make_response(200, long_body)
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.success is True
        assert len(result.response_body) <= MAX_RESPONSE_BODY_BYTES
        assert len(result.response_body) == MAX_RESPONSE_BODY_BYTES

    def test_short_response_body_preserved(self) -> None:
        mock_resp = _make_response(200, "short")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert result.response_body == "short"


class TestWebhookResult:
    def test_result_is_webhook_result_instance(self) -> None:
        mock_resp = _make_response(200, "ok")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post_webhook(url=URL, payload=PAYLOAD)
        assert isinstance(result, WebhookResult)

    def test_follow_redirects_false_is_passed_to_client(self) -> None:
        """Verify httpx.Client is instantiated with follow_redirects=False."""
        mock_resp = _make_response(200, "ok")
        with patch("oprim.http_post_webhook.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            http_post_webhook(url=URL, payload=PAYLOAD)
        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs.get("follow_redirects") is False
