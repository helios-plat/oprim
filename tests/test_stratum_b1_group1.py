"""Tests for Stratum Batch 1 Group 1: template_render, crypto_token_generate,
file_size_limiter, file_type_detector, http_post."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import patch

import httpx
import pytest

from oprim._exceptions import OprimError
from oprim.crypto_token_generate import crypto_token_generate
from oprim.file_size_limiter import SizeLimitResult, file_size_limiter
from oprim.file_type_detector import FileTypeInfo, file_type_detector
from oprim.http_post import HTTPResponse, http_post
from oprim.template_render import template_render

_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# template_render
# ---------------------------------------------------------------------------


class TestTemplateRender:
    def test_simple_variable_substitution(self) -> None:
        result = template_render(template="Hello {{ name }}", context={"name": "Wiki"})
        assert result == "Hello Wiki"

    def test_multiple_variables(self) -> None:
        result = template_render(
            template="{{ greeting }} {{ name }}!",
            context={"greeting": "Hi", "name": "World"},
        )
        assert result == "Hi World!"

    def test_strict_undefined_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="undefined_variable"):
            template_render(template="Hello {{ missing }}", context={}, strict=True)

    def test_non_strict_renders_empty_for_missing(self) -> None:
        result = template_render(template="Hello {{ missing }}", context={}, strict=False)
        assert "missing" not in result
        assert result == "Hello "

    def test_loop_template(self) -> None:
        result = template_render(
            template="{% for item in items %}{{ item }} {% endfor %}",
            context={"items": ["a", "b", "c"]},
        )
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_syntax_error_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="template_syntax_error"):
            template_render(template="{% if %}", context={})

    def test_numeric_context_value(self) -> None:
        result = template_render(template="Count: {{ n }}", context={"n": 42})
        assert result == "Count: 42"


# ---------------------------------------------------------------------------
# crypto_token_generate
# ---------------------------------------------------------------------------


class TestCryptoTokenGenerate:
    def test_default_length_32_produces_43_chars(self) -> None:
        token = crypto_token_generate()
        assert len(token) == 43

    def test_url_safe_chars_only(self) -> None:
        import re

        for _ in range(20):
            token = crypto_token_generate(url_safe=True)
            assert re.fullmatch(r"[A-Za-z0-9_\-=]+", token), f"invalid chars in {token!r}"

    def test_100_tokens_are_unique(self) -> None:
        tokens = {crypto_token_generate() for _ in range(100)}
        assert len(tokens) == 100

    def test_length_less_than_1_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            crypto_token_generate(length=0)

    def test_negative_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            crypto_token_generate(length=-5)

    def test_url_safe_false_returns_hex(self) -> None:
        import re

        token = crypto_token_generate(length=32, url_safe=False)
        assert len(token) == 64
        assert re.fullmatch(r"[0-9a-f]+", token), f"not hex: {token!r}"

    def test_custom_length_url_safe(self) -> None:
        token = crypto_token_generate(length=16, url_safe=True)
        # 16 bytes → ceil(16*4/3) = 22 chars (base64url pads, token_urlsafe trims padding)
        assert len(token) > 0

    def test_custom_length_hex(self) -> None:
        token = crypto_token_generate(length=16, url_safe=False)
        assert len(token) == 32


# ---------------------------------------------------------------------------
# file_size_limiter
# ---------------------------------------------------------------------------


class TestFileSizeLimiter:
    def test_miniprogram_within_limit(self) -> None:
        result = file_size_limiter(file_size=10 * _MB, client_type="miniprogram")
        assert result.allowed is True
        assert result.limit == 20 * _MB
        assert result.reason is None

    def test_miniprogram_exceeds_limit(self) -> None:
        result = file_size_limiter(file_size=21 * _MB, client_type="miniprogram")
        assert result.allowed is False
        assert result.reason is not None

    def test_official_account_limit(self) -> None:
        result = file_size_limiter(file_size=20 * _MB, client_type="official_account")
        assert result.allowed is True
        assert result.limit == 20 * _MB

    def test_desktop_large_file_allowed(self) -> None:
        result = file_size_limiter(file_size=300 * _MB, client_type="desktop")
        assert result.allowed is True
        assert result.limit == 500 * _MB

    def test_desktop_exceeds_limit(self) -> None:
        result = file_size_limiter(file_size=501 * _MB, client_type="desktop")
        assert result.allowed is False

    def test_web_within_limit(self) -> None:
        result = file_size_limiter(file_size=50 * _MB, client_type="web")
        assert result.allowed is True
        assert result.limit == 100 * _MB

    def test_web_boundary_exactly_at_limit(self) -> None:
        result = file_size_limiter(file_size=100 * _MB, client_type="web")
        assert result.allowed is True

    def test_web_one_byte_over_limit(self) -> None:
        result = file_size_limiter(file_size=100 * _MB + 1, client_type="web")
        assert result.allowed is False

    def test_zero_size_always_allowed(self) -> None:
        result = file_size_limiter(file_size=0, client_type="miniprogram")
        assert result.allowed is True

    def test_negative_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            file_size_limiter(file_size=-1, client_type="web")

    def test_result_is_size_limit_result(self) -> None:
        result = file_size_limiter(file_size=1024, client_type="web")
        assert isinstance(result, SizeLimitResult)
        assert result.client_type == "web"
        assert result.file_size == 1024


# ---------------------------------------------------------------------------
# file_type_detector
# ---------------------------------------------------------------------------


class TestFileTypeDetector:
    def test_text_file_detected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("hello world")
            path = pathlib.Path(f.name)
        try:
            result = file_type_detector(file_path=path)
            assert "text" in result.mime_type
            assert result.extension == ".txt"
            assert result.is_supported is True
        finally:
            path.unlink(missing_ok=True)

    def test_category_document_for_text(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("plain text content here")
            path = pathlib.Path(f.name)
        try:
            result = file_type_detector(file_path=path)
            assert result.category in {"document", "code"}
        finally:
            path.unlink(missing_ok=True)

    def test_python_file_category_code(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("print('hello')\n")
            path = pathlib.Path(f.name)
        try:
            result = file_type_detector(file_path=path)
            # magic may detect as text/x-script.python or text/plain — extension is .py
            assert result.extension == ".py"
            assert result.category in {"code", "document"}
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="file_not_found"):
            file_type_detector(file_path=pathlib.Path("/tmp/does_not_exist_xyz123.txt"))

    def test_result_is_file_type_info(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("test content")
            path = pathlib.Path(f.name)
        try:
            result = file_type_detector(file_path=path)
            assert isinstance(result, FileTypeInfo)
        finally:
            path.unlink(missing_ok=True)

    def test_json_file_detected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('{"key": "value"}')
            path = pathlib.Path(f.name)
        try:
            result = file_type_detector(file_path=path)
            assert result.extension == ".json"
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# http_post
# ---------------------------------------------------------------------------


URL = "https://api.example.com/data"


def _make_response(
    status_code: int, json_body: dict | None = None, text: str = "ok"
) -> httpx.Response:
    if json_body is not None:
        import json as _json

        return httpx.Response(status_code=status_code, text=_json.dumps(json_body))
    return httpx.Response(status_code=status_code, text=text)


class TestHttpPost:
    def test_200_returns_http_response(self) -> None:
        mock_resp = _make_response(200, json_body={"result": "ok"})
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post(url=URL, json_data={"key": "val"})
        assert result.status_code == 200
        assert isinstance(result, HTTPResponse)
        assert result.elapsed_ms >= 0

    def test_json_body_parsed(self) -> None:
        mock_resp = _make_response(200, json_body={"message": "hello"})
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post(url=URL, json_data={})
        assert isinstance(result.body, dict)
        assert result.body == {"message": "hello"}

    def test_text_body_fallback(self) -> None:
        mock_resp = _make_response(200, text="plain text response")
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post(url=URL)
        assert isinstance(result.body, str)
        assert result.body == "plain text response"

    def test_timeout_raises_oprim_error(self) -> None:
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException(
                "timed out"
            )
            with pytest.raises(OprimError, match="timeout"):
                http_post(url=URL)

    def test_connect_error_raises_oprim_error(self) -> None:
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "connection refused"
            )
            with pytest.raises(OprimError, match="connect_failed"):
                http_post(url=URL)

    def test_unexpected_exception_raises_oprim_error(self) -> None:
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = RuntimeError("boom")
            with pytest.raises(OprimError, match="unexpected"):
                http_post(url=URL)

    def test_headers_returned(self) -> None:
        mock_resp = _make_response(200, json_body={})
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = http_post(url=URL)
        assert isinstance(result.headers, dict)

    def test_no_json_data_sends_none(self) -> None:
        mock_resp = _make_response(201, json_body={"created": True})
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_inst = mock_cls.return_value.__enter__.return_value
            mock_inst.post.return_value = mock_resp
            result = http_post(url=URL)
        assert result.status_code == 201
        call_kwargs = mock_inst.post.call_args.kwargs
        assert call_kwargs.get("json") is None

    def test_follow_redirects_true(self) -> None:
        mock_resp = _make_response(200, json_body={})
        with patch("oprim.http_post.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            http_post(url=URL)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("follow_redirects") is True
