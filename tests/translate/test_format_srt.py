"""Tests for SRT translation format adapter."""
import pytest
from unittest.mock import MagicMock
from oprim.translate.format_srt import translate_srt, _parse_srt, _render_srt, SrtBlock
from oprim.translate.protocol import TranslationResult


_SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello world.

2
00:00:04,000 --> 00:00:06,000
This is a test.

"""


def _mock_provider(translated_text: str):
    prov = MagicMock()
    prov.name = "mock"
    result = TranslationResult(
        text=translated_text,
        provider="mock",
        model="mock",
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.001,
        source_lang="en",
        target_lang="zh",
    )
    prov.translate.return_value = result
    return prov


def test_parse_srt_basic():
    blocks = _parse_srt(_SAMPLE_SRT)
    assert len(blocks) == 2
    assert blocks[0].index == 1
    assert "Hello world" in blocks[0].text
    assert "00:00:01" in blocks[0].timecode


def test_render_srt_round_trip():
    blocks = [SrtBlock(1, "00:00:01,000 --> 00:00:03,000", "Hello")]
    rendered = _render_srt(blocks)
    assert "00:00:01" in rendered
    assert "Hello" in rendered


def test_translate_srt_preserves_timecodes():
    prov = _mock_provider("你好世界。\n---\n这是一个测试。")
    result_srt, results = translate_srt(
        _SAMPLE_SRT, prov, "en", "zh", batch_size=10
    )
    assert "00:00:01,000 --> 00:00:03,000" in result_srt
    assert "00:00:04,000 --> 00:00:06,000" in result_srt
    assert len(results) == 1


def test_translate_srt_empty():
    prov = _mock_provider("")
    result_srt, results = translate_srt("", prov, "en", "zh")
    assert result_srt == ""
    assert results == []
