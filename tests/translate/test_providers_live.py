"""Tests for provider translate() paths — mocked API calls (no Gemini, Phase 10)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from oprim.translate.providers.deepseek import DeepSeekProvider
from oprim.translate.providers.claude import ClaudeProvider
from oprim.translate.providers.qwen3 import Qwen3Provider
from oprim.translate.protocol import TranslationRequest
from oprim.errors import LLMError, LLMRateLimitError


def _req(text: str = "Hello world.") -> TranslationRequest:
    return TranslationRequest(text=text, source_lang="en", target_lang="zh")


def _openai_resp(translated: str = "你好世界", in_tok: int = 10, out_tok: int = 8):
    resp = MagicMock()
    resp.choices[0].message.content = translated
    resp.usage.prompt_tokens = in_tok
    resp.usage.completion_tokens = out_tok
    return resp


def _anthropic_resp(translated: str = "你好世界", in_tok: int = 10, out_tok: int = 8):
    resp = MagicMock()
    resp.content[0].text = translated
    resp.usage.input_tokens = in_tok
    resp.usage.output_tokens = out_tok
    return resp


# ── DeepSeek ──────────────────────────────────────────────────────────────────

class TestDeepSeekProvider:
    def test_translate_success(self):
        prov = DeepSeekProvider()
        with patch("oprim.translate.providers.deepseek.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _openai_resp("深度求索")
            result = prov.translate(_req())
        assert result.text == "深度求索"
        assert result.provider == "deepseek"
        assert result.input_tokens == 10
        assert result.output_tokens == 8
        assert result.cost_usd > 0

    def test_translate_rate_limit(self):
        prov = DeepSeekProvider()
        with patch("oprim.translate.providers.deepseek.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("HTTP 429 rate limit")
            with pytest.raises(LLMRateLimitError):
                prov.translate(_req())

    def test_translate_retries_then_fails(self):
        prov = DeepSeekProvider()
        with patch("oprim.translate.providers.deepseek.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI, \
             patch("oprim.translate.providers.deepseek.time.sleep"):
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("network error")
            with pytest.raises(LLMError, match="3 retries"):
                prov.translate(_req())
            assert mock_client.chat.completions.create.call_count == 3

    def test_translate_model_override(self):
        prov = DeepSeekProvider()
        req = TranslationRequest(text="hello", source_lang="en", target_lang="zh", model="deepseek-reasoner")
        with patch("oprim.translate.providers.deepseek.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _openai_resp()
            prov.translate(req)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-reasoner"


# ── Claude ────────────────────────────────────────────────────────────────────

class TestClaudeProvider:
    def test_translate_success(self):
        prov = ClaudeProvider()
        import anthropic as _ant
        with patch("oprim.translate.providers.claude.cfg.get", return_value="fake-key"), \
             patch.object(_ant, "Anthropic") as MockAnt:
            mock_client = MagicMock()
            MockAnt.return_value = mock_client
            mock_client.messages.create.return_value = _anthropic_resp("克劳德")
            result = prov.translate(_req())
        assert result.text == "克劳德"
        assert result.provider == "claude"
        assert result.cost_usd > 0

    def test_translate_retries_then_fails(self):
        prov = ClaudeProvider()
        import anthropic as _ant
        with patch("oprim.translate.providers.claude.cfg.get", return_value="fake-key"), \
             patch.object(_ant, "Anthropic") as MockAnt, \
             patch("oprim.translate.providers.claude.time.sleep"):
            mock_client = MagicMock()
            MockAnt.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("server error")
            with pytest.raises(LLMError, match="3 retries"):
                prov.translate(_req())

    def test_translate_import_error(self):
        prov = ClaudeProvider()
        import sys
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(LLMError, match="anthropic package"):
                prov.translate(_req())


# ── Qwen3 ─────────────────────────────────────────────────────────────────────

class TestQwen3Provider:
    def test_translate_success(self):
        prov = Qwen3Provider()
        with patch("oprim.translate.providers.qwen3.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _openai_resp("通义千问")
            result = prov.translate(_req())
        assert result.text == "通义千问"
        assert result.provider == "qwen3"

    def test_uses_qwen_api_key_fallback(self):
        prov = Qwen3Provider()
        def _cfg_get(key: str, default=None):
            return "qwen-fallback-key" if key == "QWEN_API_KEY" else None
        with patch("oprim.translate.providers.qwen3.cfg.get", side_effect=_cfg_get), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _openai_resp()
            result = prov.translate(_req())
        assert result.text == "你好世界"

    def test_translate_rate_limit(self):
        prov = Qwen3Provider()
        with patch("oprim.translate.providers.qwen3.cfg.get", return_value="fake-key"), \
             patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("429 quota exceeded")
            with pytest.raises(LLMRateLimitError):
                prov.translate(_req())
