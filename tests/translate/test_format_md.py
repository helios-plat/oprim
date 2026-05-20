"""Tests for markdown translation pipeline (async primary + deprecated sync)."""
import pytest
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from oprim.translate.format_md import translate_markdown, translate_markdown_async
from oprim.translate.protocol import TranslationResult


def _make_provider(reply: str = "translated"):
    prov = MagicMock()
    prov.name = "mock"
    prov.translate = AsyncMock(return_value=TranslationResult(
        text=reply,
        provider="mock",
        model="mock",
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.0,
        source_lang="en",
        target_lang="zh",
    ))
    return prov


async def test_translate_simple_markdown():
    prov = _make_provider("翻译内容")
    result, chunk_results = await translate_markdown_async("Hello world.", prov, "en", "zh")
    assert "翻译内容" in result
    assert len(chunk_results) == 1


async def test_code_blocks_pass_through():
    prov = _make_provider("translated prose")
    text = "Intro.\n\n```python\nprint('x')\n```\n\nOutro."
    result, chunk_results = await translate_markdown_async(text, prov, "en", "zh")
    assert "```python" in result
    assert "print('x')" in result
    assert len(chunk_results) == 2  # only prose chunks get translated


async def test_checkpoint_skip_completed(tmp_path: Path):
    prov = _make_provider("from_provider")
    checkpoint_path = tmp_path / "cp.json"
    from oprim.translate.checkpoint import TranslationCheckpoint
    cp = TranslationCheckpoint(checkpoint_path)
    cp.save_chunk(0, "cached_translation")

    result, chunk_results = await translate_markdown_async(
        "A single paragraph.", prov, "en", "zh", checkpoint_path=checkpoint_path
    )
    assert "cached_translation" in result
    prov.translate.assert_not_called()


async def test_checkpoint_cleared_after_done(tmp_path: Path):
    prov = _make_provider("done")
    checkpoint_path = tmp_path / "cp.json"
    await translate_markdown_async("Hello.", prov, "en", "zh", checkpoint_path=checkpoint_path)
    assert not checkpoint_path.exists()


async def test_empty_text():
    prov = _make_provider()
    result, chunks = await translate_markdown_async("", prov, "en", "zh")
    assert result == ""
    assert chunks == []


def test_translate_markdown_sync_deprecated():
    """Deprecated sync wrapper emits DeprecationWarning."""
    prov = _make_provider("sync_result")
    with pytest.warns(DeprecationWarning, match="translate_markdown_async"):
        result, chunks = translate_markdown("Hello.", prov, "en", "zh")
    assert result == "sync_result"
    assert len(chunks) == 1
