"""Tests for TranslationRouter."""
import pytest
from unittest.mock import MagicMock
from oprim.translate.router import TranslationRouter
from oprim.translate.protocol import TranslationRequest


def _make_provider(name: str):
    prov = MagicMock()
    prov.name = name
    return prov


def _providers():
    return {
        "deepseek": _make_provider("deepseek"),
        "claude": _make_provider("claude"),
        "qwen3": _make_provider("qwen3"),
    }


def test_default_routes_to_deepseek():
    router = TranslationRouter(_providers())
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh")
    chosen = router.route(req)
    assert chosen.name == "deepseek"


def test_premium_routes_to_claude():
    router = TranslationRouter(_providers())
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh", quality="premium")
    chosen = router.route(req)
    assert chosen.name == "claude"


def test_domestic_routes_to_qwen3():
    router = TranslationRouter(_providers())
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh", user_preference="domestic")
    chosen = router.route(req)
    assert chosen.name == "qwen3"


def test_premium_without_claude_falls_to_deepseek():
    provs = {"deepseek": _make_provider("deepseek"), "qwen3": _make_provider("qwen3")}
    router = TranslationRouter(provs)
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh", quality="premium")
    chosen = router.route(req)
    assert chosen.name == "deepseek"


def test_domestic_without_qwen3_falls_to_deepseek():
    provs = {"deepseek": _make_provider("deepseek"), "claude": _make_provider("claude")}
    router = TranslationRouter(provs)
    req = TranslationRequest(text="hi", source_lang="en", target_lang="zh", user_preference="domestic")
    chosen = router.route(req)
    assert chosen.name == "deepseek"


def test_missing_deepseek_raises():
    provs = {"claude": _make_provider("claude")}
    with pytest.raises(ValueError, match="deepseek"):
        TranslationRouter(provs)


def test_provider_names():
    router = TranslationRouter(_providers())
    names = router.provider_names()
    assert set(names) == {"deepseek", "claude", "qwen3"}
