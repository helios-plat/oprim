"""Tests for translation providers (unit tests — no real API calls, async)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from oprim.translate.providers import get_provider, DeepSeekProvider, ClaudeProvider, Qwen3Provider
from oprim.translate.protocol import TranslationRequest
from oprim.errors import LLMError


def test_get_provider_deepseek():
    prov = get_provider("deepseek")
    assert isinstance(prov, DeepSeekProvider)
    assert prov.name == "deepseek"


def test_get_provider_claude():
    prov = get_provider("claude")
    assert isinstance(prov, ClaudeProvider)


def test_get_provider_qwen3():
    prov = get_provider("qwen3")
    assert isinstance(prov, Qwen3Provider)
    assert prov.name == "qwen3"


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown translation provider"):
        get_provider("unknown_provider")


def test_get_provider_gemini_not_available():
    with pytest.raises(ValueError, match="Unknown translation provider"):
        get_provider("gemini")


async def test_deepseek_missing_key():
    prov = DeepSeekProvider()
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh")
    with patch("oprim.translate.providers.deepseek.cfg.get", return_value=None):
        with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
            await prov.translate(req)


async def test_qwen3_missing_key():
    prov = Qwen3Provider()
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh")
    with patch("oprim.translate.providers.qwen3.cfg.get", return_value=None):
        with pytest.raises(LLMError, match="DASHSCOPE_API_KEY"):
            await prov.translate(req)


def test_estimate_cost_positive():
    for name in ["deepseek", "claude", "qwen3"]:
        prov = get_provider(name)
        cost = prov.estimate_cost(1000)
        assert cost > 0, f"{name} estimate_cost should be positive"


async def test_provider_translate_is_coroutine():
    import inspect
    for cls in [DeepSeekProvider, ClaudeProvider, Qwen3Provider]:
        assert inspect.iscoroutinefunction(cls.translate), f"{cls.__name__}.translate must be async"


async def test_provider_health_check_is_coroutine():
    import inspect
    for cls in [DeepSeekProvider, ClaudeProvider, Qwen3Provider]:
        assert inspect.iscoroutinefunction(cls.health_check), f"{cls.__name__}.health_check must be async"
