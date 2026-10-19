"""Tests for oprim.llm.llm_call — provider dispatch over httpx (DashScope / NIM).

The original tests targeted the pre-v3.10.6 implementation (dashscope SDK + anthropic
SDK with retry loops). llm_call now talks to DashScope / NVIDIA NIM over plain httpx
and the Claude branch is an explicit stub, so the tests mock ``httpx.post`` instead.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from oprim.errors import LLMError, LLMRateLimitError
from oprim.llm.llm_call import LLMResponse, llm_call


@pytest.fixture(autouse=True)
def _no_config_keys(monkeypatch):
    """Isolate from any local oprim config / env API keys."""
    fake_cfg = MagicMock()
    fake_cfg.get.return_value = None
    # ``oprim.llm.llm_call`` the attribute is the re-exported function; patch the module.
    monkeypatch.setattr(importlib.import_module("oprim.llm.llm_call"), "cfg", fake_cfg)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)


def _ok(json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body
    return resp


def _http_error(status: int) -> MagicMock:
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status, request=request)
    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )
    return resp


_DASHSCOPE_BODY = {"output": {"choices": [{"message": {"content": "The answer is 42."}}]}}


class TestDashScope:
    def test_missing_key_raises(self):
        with pytest.raises(LLMError, match="DASHSCOPE_API_KEY not set"):
            llm_call("hi", provider="qwen3_dashscope")

    def test_success(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        with patch("httpx.post", return_value=_ok(_DASHSCOPE_BODY)) as mock_post:
            result = llm_call("What is the answer?", provider="qwen3_dashscope")
        assert isinstance(result, LLMResponse)
        assert result.text == "The answer is 42."
        assert result.model == "qwen-plus"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"

    def test_custom_model_and_system_prompt(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        with patch("httpx.post", return_value=_ok(_DASHSCOPE_BODY)) as mock_post:
            result = llm_call(
                "hello", provider="qwen3_dashscope", model="qwen-max", system="You are helpful"
            )
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "qwen-max"
        assert result.model == "qwen-max"
        messages = body["input"]["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_rate_limit_raises(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        with patch("httpx.post", return_value=_http_error(429)):
            with pytest.raises(LLMRateLimitError):
                llm_call("prompt", provider="qwen3_dashscope")

    def test_http_error_raises(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        with patch("httpx.post", return_value=_http_error(500)):
            with pytest.raises(LLMError, match="HTTP error 500"):
                llm_call("prompt", provider="qwen3_dashscope")

    def test_transport_failure_raises(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        with patch("httpx.post", side_effect=RuntimeError("boom")):
            with pytest.raises(LLMError, match="DashScope call failed"):
                llm_call("prompt", provider="qwen3_dashscope")


class TestNvidiaNim:
    def test_missing_key_raises(self):
        with pytest.raises(LLMError, match="NVIDIA_NIM_API_KEY not set"):
            llm_call("hi", provider="nvidia_nim")

    def test_success(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nim-key")
        body = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }
        with patch("httpx.post", return_value=_ok(body)):
            result = llm_call("hi", provider="nvidia_nim", model="m", system="sys")
        assert result.text == "ok"
        assert result.model == "m"
        assert result.stop_reason == "stop"
        assert (result.input_tokens, result.output_tokens) == (3, 5)

    def test_rate_limit_raises(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nim-key")
        with patch("httpx.post", return_value=_http_error(429)):
            with pytest.raises(LLMRateLimitError):
                llm_call("hi", provider="nvidia_nim")


class TestDispatch:
    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            llm_call("prompt", provider="nonexistent")

    def test_claude_is_explicit_stub(self):
        with pytest.raises(LLMError, match="llm_complete"):
            llm_call("prompt", provider="claude")
