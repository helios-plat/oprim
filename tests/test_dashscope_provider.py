"""Tests for oprim.providers.dashscope — B5 OpenAI-compat rewrite + JSON coercion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.errors import LLMError, LLMRateLimitError
from oprim.providers.dashscope import _coerce_llm_json, _make_llm_caller


def _make_http_response(status_code: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


def _compat_body(content: str, *, model: str = "qwen-plus") -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": model,
    }


class TestDashScopeCaller:
    def test_success_hits_compat_endpoint(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(200, _compat_body("hello world"))

        with patch("httpx.post", return_value=http_resp) as mock_post:
            result = caller(messages=[{"role": "user", "content": "hi"}])

        assert result.get("content") == "hello world"
        assert mock_post.call_args.args[0] == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

    async def test_result_is_awaitable(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(200, _compat_body("hello"))

        with patch("httpx.post", return_value=http_resp):
            result = caller(messages=[{"role": "user", "content": "hi"}])
            awaited = await result

        assert awaited["content"] == "hello"
        assert awaited["output"]["choices"][0]["message"]["content"] == "hello"

    def test_rate_limit_raises(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(429, {"error": "Throttling"})

        with patch("httpx.post", return_value=http_resp), pytest.raises(LLMRateLimitError):
            caller(messages=[{"role": "user", "content": "hi"}])

    def test_server_error_raises(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(500, {"error": "Internal error"})

        with patch("httpx.post", return_value=http_resp), pytest.raises(LLMError):
            caller(messages=[{"role": "user", "content": "hi"}])

    def test_custom_model_forwarded(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(200, _compat_body("ok", model="qwen-max"))

        with patch("httpx.post", return_value=http_resp) as mock_post:
            caller(messages=[{"role": "user", "content": "hi"}], model="qwen-max")

        assert mock_post.call_args.kwargs["json"]["model"] == "qwen-max"

    def test_placeholder_api_key_used_in_auth_header(self) -> None:
        caller = _make_llm_caller(default_model="qwen-plus")
        http_resp = _make_http_response(200, _compat_body("ok"))

        with (
            patch("oprim._config.cfg.get", return_value="sk-placeholder"),
            patch("httpx.post", return_value=http_resp) as mock_post,
        ):
            caller(messages=[{"role": "user", "content": "hi"}])

        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-placeholder"


class TestCoerceLlmJson:
    def test_numeric_id_coerced_to_str(self) -> None:
        out = _coerce_llm_json('{"scene_id": 3, "text": "x"}')
        assert '"scene_id": "3"' in out

    def test_importance_word_mapped(self) -> None:
        out = _coerce_llm_json('{"importance": "critical"}')
        assert '"importance": 4' in out

    def test_importance_float_rounded(self) -> None:
        out = _coerce_llm_json('{"importance": 2.6}')
        assert '"importance": 3' in out

    def test_scenes_list_of_strings_becomes_dicts(self) -> None:
        out = _coerce_llm_json('{"scenes": ["a walk in the park"]}')
        assert '"visual_description": "a walk in the park"' in out
        assert '"id": "1"' in out

    def test_shots_list_of_strings_becomes_dicts(self) -> None:
        out = _coerce_llm_json('{"shots": ["hello there"]}')
        assert '"narration": "hello there"' in out

    def test_none_becomes_empty_string(self) -> None:
        out = _coerce_llm_json('{"caption": null}')
        assert '"caption": ""' in out

    def test_strips_markdown_code_fence(self) -> None:
        out = _coerce_llm_json('```json\n{"scene_id": 1}\n```')
        assert '"scene_id": "1"' in out

    def test_non_json_text_passed_through(self) -> None:
        assert _coerce_llm_json("just plain text") == "just plain text"


class TestRegister:
    def test_registers_default_and_qwen3_dashscope(self) -> None:
        from obase import ProviderRegistry

        from oprim.providers.dashscope import register

        register(replace=True)
        registry = ProviderRegistry.get()
        # NOTE: "llm" is a builtin category in obase — it's dispatched through
        # register_llm()/.llm(), NOT register_generic()/.generic(). See the
        # _default_llm() fix in omodul for the bug this distinction caught.
        assert registry.llm("default") is not None
        assert registry.llm("qwen3_dashscope") is not None
        assert registry.llm("default") is registry.llm("qwen3_dashscope")
