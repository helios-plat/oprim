"""Tests for markdown translation pipeline."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from oprim.translate.format_md import translate_markdown
from oprim.translate.protocol import TranslationResult


def _make_provider(reply: str = "translated"):
    prov = MagicMock()
    prov.name = "mock"
    prov.translate.return_value = TranslationResult(
        text=reply,
        provider="mock",
        model="mock",
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.0,
        source_lang="en",
        target_lang="zh",
    )
    return prov


def test_translate_simple_markdown():
    prov = _make_provider("翻译内容")
    text = "Hello world."
    result, chunk_results = translate_markdown(text, prov, "en", "zh")
    assert "翻译内容" in result
    assert len(chunk_results) == 1


def test_code_blocks_pass_through():
    prov = _make_provider("translated prose")
    text = "Intro.\n\n```python\nprint('x')\n```\n\nOutro."
    result, chunk_results = translate_markdown(text, prov, "en", "zh")
    assert "```python" in result
    assert "print('x')" in result
    assert len(chunk_results) == 2  # only prose chunks get translated


def test_checkpoint_skip_completed(tmp_path):
    prov = _make_provider("from_provider")
    checkpoint_path = tmp_path / "cp.json"
    from oprim.translate.checkpoint import TranslationCheckpoint
    cp = TranslationCheckpoint(checkpoint_path)
    cp.save_chunk(0, "cached_translation")

    text = "A single paragraph."
    result, chunk_results = translate_markdown(
        text, prov, "en", "zh", checkpoint_path=checkpoint_path
    )
    assert "cached_translation" in result
    prov.translate.assert_not_called()


def test_checkpoint_cleared_after_done(tmp_path):
    prov = _make_provider("done")
    checkpoint_path = tmp_path / "cp.json"
    translate_markdown("Hello.", prov, "en", "zh", checkpoint_path=checkpoint_path)
    assert not checkpoint_path.exists()


def test_empty_text():
    prov = _make_provider()
    result, chunks = translate_markdown("", prov, "en", "zh")
    assert result == ""
    assert chunks == []
