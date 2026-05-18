"""Tests for oprim.llm.llm_call — mocking DashScope and Anthropic APIs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.llm.llm_call import LLMResponse, llm_call
from oprim.errors import LLMError, LLMRateLimitError


def _make_dashscope_response(
    text: str = "hello",
    status_code: int = 200,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    resp = MagicMock()
    resp.status_code = status_code
    resp.output.choices[0].message.content = text
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.message = "OK"
    return resp


class TestLLMCall:
    def test_dashscope_success(self):
        resp = _make_dashscope_response("The answer is 42.")

        with patch("dashscope.Generation.call", return_value=resp):
            result = llm_call("What is the answer?", provider="qwen3_dashscope")

        assert isinstance(result, LLMResponse)
        assert result.text == "The answer is 42."
        assert result.model == "qwen-plus"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.cost_usd > 0

    def test_dashscope_custom_model(self):
        resp = _make_dashscope_response("response")

        with patch("dashscope.Generation.call", return_value=resp) as mock_call:
            llm_call("prompt", provider="qwen3_dashscope", model="qwen-max")
            # check model was passed
            call_kwargs = mock_call.call_args.kwargs
            assert call_kwargs.get("model") == "qwen-max"

    def test_dashscope_system_prompt(self):
        resp = _make_dashscope_response("ok")

        with patch("dashscope.Generation.call", return_value=resp) as mock_call:
            llm_call("hello", provider="qwen3_dashscope", system="You are helpful")
            messages = mock_call.call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful"

    def test_dashscope_rate_limit_raises(self):
        resp = _make_dashscope_response(status_code=429)
        resp.message = "Rate limit exceeded"

        with patch("dashscope.Generation.call", return_value=resp):
            with pytest.raises(LLMRateLimitError):
                llm_call("test", provider="qwen3_dashscope")

    def test_dashscope_error_status_raises(self):
        resp = _make_dashscope_response(status_code=500)
        resp.message = "Internal server error"

        with patch("dashscope.Generation.call", return_value=resp):
            with patch("time.sleep"):
                with pytest.raises(LLMError):
                    llm_call("test", provider="qwen3_dashscope")

    def test_dashscope_retry_on_exception(self):
        """Exception on first 2 calls, success on 3rd."""
        resp = _make_dashscope_response("recovered")
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient")
            return resp

        with patch("dashscope.Generation.call", side_effect=side_effect):
            with patch("time.sleep"):
                result = llm_call("prompt", provider="qwen3_dashscope")

        assert result.text == "recovered"
        assert call_count["n"] == 3

    def test_dashscope_exhausted_retries_raises(self):
        def always_fail(**kwargs):
            raise ConnectionError("always fails")

        with patch("dashscope.Generation.call", side_effect=always_fail):
            with patch("time.sleep"):
                with pytest.raises(LLMError, match="3 retries"):
                    llm_call("test", provider="qwen3_dashscope")

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            llm_call("test", provider="openai")

    def test_claude_no_anthropic_package(self):
        """If anthropic is not installed, should raise LLMError with anthropic mention."""
        # Directly test the error path via mocking _call_claude
        with patch("oprim.llm.llm_call._call_claude", side_effect=LLMError("anthropic package not installed")):
            with pytest.raises(LLMError, match="anthropic"):
                llm_call("test", provider="claude")

    def test_claude_success_mocked(self):
        import sys
        from unittest.mock import MagicMock
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Claude says hello")]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_resp
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = llm_call("hi", provider="claude", model="claude-haiku-4-5-20251001")

        assert result.text == "Claude says hello"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.input_tokens == 10
        assert result.cost_usd > 0

    def test_claude_retry_on_exception(self):
        import sys
        from unittest.mock import MagicMock
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="retry success")]
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 3
        call_n = {"n": 0}

        def side_effect(**kwargs):
            call_n["n"] += 1
            if call_n["n"] < 2:
                raise ConnectionError("transient")
            return mock_resp

        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):
                result = llm_call("hi", provider="claude")

        assert result.text == "retry success"

    def test_claude_exhausted_retries_raises(self):
        import sys
        from unittest.mock import MagicMock
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("always fails")
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):
                with pytest.raises(LLMError, match="3 retries"):
                    llm_call("hi", provider="claude")

    def test_claude_with_system_prompt(self):
        import sys
        from unittest.mock import MagicMock
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 2
        mock_client.messages.create.return_value = mock_resp
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = llm_call("hi", provider="claude", system="You are a tester")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are a tester"
        assert result.text == "ok"

