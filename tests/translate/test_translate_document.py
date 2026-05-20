"""Tests for oprim.translate entry points (async primary + deprecated sync)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from oprim.translate import translate_document, translate_document_async, TerminologyGlossary
from oprim.translate.protocol import TranslationResult


def _mock_translate_result(text: str = "译文") -> TranslationResult:
    return TranslationResult(
        text=text,
        provider="mock",
        model="mock",
        input_tokens=10,
        output_tokens=8,
        cost_usd=0.001,
        source_lang="en",
        target_lang="zh",
    )


def _mock_provider(reply: str = "译文"):
    prov = MagicMock()
    prov.name = "mock"
    prov.translate = AsyncMock(return_value=_mock_translate_result(reply))
    return prov


# ── async primary tests ──────────────────────────────────────────────────────

async def test_translate_document_async_with_string_provider():
    with patch("oprim.translate.get_provider") as mock_gp, \
         patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_prov = _mock_provider()
        mock_gp.return_value = mock_prov
        mock_tma.return_value = ("译文", [_mock_translate_result()])
        result, chunks = await translate_document_async("Hello.", "en", "zh", provider="deepseek")
    assert result == "译文"
    assert len(chunks) == 1
    mock_gp.assert_called_once_with("deepseek")


async def test_translate_document_async_with_provider_instance():
    prov = _mock_provider()
    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("译文", [_mock_translate_result()])
        result, chunks = await translate_document_async("Hello.", "en", "zh", provider=prov)
    assert result == "译文"


async def test_translate_document_async_with_glossary():
    glossary = TerminologyGlossary()
    glossary.add("substrate", "底层文档", "en", "zh")
    prov = _mock_provider("这是__TERM_0000__。")

    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("这是__TERM_0000__。", [_mock_translate_result()])
        result, _ = await translate_document_async(
            "This is substrate.", "en", "zh",
            provider=prov, glossary=glossary,
        )
    assert "底层文档" in result


async def test_translate_document_async_checkpoint_forwarded(tmp_path: Path):
    prov = _mock_provider()
    cp = tmp_path / "cp.json"
    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("done", [])
        await translate_document_async("Text.", "en", "zh", provider=prov, checkpoint_path=cp)
    call_kwargs = mock_tma.call_args[1]
    assert call_kwargs["checkpoint_path"] == cp


async def test_translate_document_async_aggregated_cost_logged():
    prov = _mock_provider()
    chunks = [_mock_translate_result("a"), _mock_translate_result("b")]
    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma, \
         patch("oprim.translate.log") as mock_log:
        mock_tma.return_value = ("ab", chunks)
        await translate_document_async("Hello.", "en", "zh", provider=prov)
    logged = {k: v for call in mock_log.info.call_args_list for k, v in call[1].items()}
    assert logged.get("chunks") == 2


# ── deprecated sync wrapper tests ────────────────────────────────────────────

def test_translate_document_sync_deprecated_warns():
    """Calling translate_document() emits DeprecationWarning."""
    prov = _mock_provider()
    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("sync_result", [_mock_translate_result()])
        with pytest.warns(DeprecationWarning, match="translate_document_async"):
            result, chunks = translate_document("Hello.", "en", "zh", provider=prov)
    assert result == "sync_result"


def test_translate_document_sync_behavior_unchanged():
    """Deprecated sync wrapper returns same result as async version."""
    prov = _mock_provider("一致结果")
    with patch("oprim.translate.translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("一致结果", [_mock_translate_result("一致结果")])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result, _ = translate_document("Test.", "en", "zh", provider=prov)
    assert result == "一致结果"
