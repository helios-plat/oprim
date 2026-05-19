"""Tests for oprim.translate.translate_document (main entry point)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from oprim.translate import translate_document, TerminologyGlossary
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
    prov.translate.return_value = _mock_translate_result(reply)
    return prov


def test_translate_document_with_string_provider():
    with patch("oprim.translate.get_provider") as mock_gp, \
         patch("oprim.translate.translate_markdown") as mock_tm:
        mock_prov = _mock_provider()
        mock_gp.return_value = mock_prov
        mock_tm.return_value = ("译文", [_mock_translate_result()])
        result, chunks = translate_document("Hello.", "en", "zh", provider="deepseek")
    assert result == "译文"
    assert len(chunks) == 1
    mock_gp.assert_called_once_with("deepseek")


def test_translate_document_with_provider_instance():
    prov = _mock_provider()
    with patch("oprim.translate.translate_markdown") as mock_tm:
        mock_tm.return_value = ("译文", [_mock_translate_result()])
        result, chunks = translate_document("Hello.", "en", "zh", provider=prov)
    assert result == "译文"


def test_translate_document_with_glossary():
    glossary = TerminologyGlossary()
    glossary.add("substrate", "底层文档", "en", "zh")
    prov = _mock_provider("这是底层文档。")

    with patch("oprim.translate.translate_markdown") as mock_tm:
        mock_tm.return_value = ("这是__TERM_0000__。", [_mock_translate_result()])
        result, _ = translate_document(
            "This is substrate.", "en", "zh",
            provider=prov, glossary=glossary
        )
    assert "底层文档" in result


def test_translate_document_no_glossary():
    prov = _mock_provider("翻译结果")
    with patch("oprim.translate.translate_markdown") as mock_tm:
        mock_tm.return_value = ("翻译结果", [_mock_translate_result()])
        result, chunks = translate_document("Text.", "en", "zh", provider=prov)
    assert result == "翻译结果"


def test_translate_document_checkpoint_forwarded(tmp_path):
    prov = _mock_provider()
    cp = tmp_path / "cp.json"
    with patch("oprim.translate.translate_markdown") as mock_tm:
        mock_tm.return_value = ("done", [])
        translate_document("Text.", "en", "zh", provider=prov, checkpoint_path=cp)
    call_kwargs = mock_tm.call_args[1]
    assert call_kwargs["checkpoint_path"] == cp


def test_translate_document_aggregated_cost_logged():
    prov = _mock_provider()
    chunks = [
        _mock_translate_result("a"),
        _mock_translate_result("b"),
    ]
    with patch("oprim.translate.translate_markdown") as mock_tm, \
         patch("oprim.translate.log") as mock_log:
        mock_tm.return_value = ("ab", chunks)
        translate_document("Hello.", "en", "zh", provider=prov)
    logged = {k: v for call in mock_log.info.call_args_list for k, v in call[1].items()}
    assert logged.get("chunks") == 2
