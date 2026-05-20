"""Tests for EPUB translation format adapter (async primary + deprecated sync)."""
from __future__ import annotations

import sys
import types
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from oprim.translate.protocol import TranslationResult
from oprim.errors import StratumError


def _mock_provider(reply: str = "翻译内容"):
    prov = MagicMock()
    prov.name = "mock"
    prov.translate = AsyncMock(return_value=TranslationResult(
        text=reply,
        provider="mock",
        model="mock",
        input_tokens=10,
        output_tokens=8,
        cost_usd=0.001,
        source_lang="en",
        target_lang="zh",
    ))
    return prov


def _make_epub_item(body_html: str = "<p>Hello.</p>", name: str = "ch1.xhtml"):
    item = MagicMock()
    html = f"<html><body>{body_html}</body></html>"
    item.get_content.return_value = html.encode("utf-8")
    item.get_name.return_value = name
    item.set_content = MagicMock()
    return item


def _build_epub_modules():
    """Build minimal fake ebooklib / ebooklib.epub modules for patching."""
    mock_ebooklib = types.ModuleType("ebooklib")
    mock_ebooklib.ITEM_DOCUMENT = 9

    mock_epub = types.ModuleType("ebooklib.epub")
    mock_book = MagicMock()
    mock_epub.read_epub = MagicMock(return_value=mock_book)
    mock_epub.write_epub = MagicMock()
    mock_ebooklib.epub = mock_epub

    return mock_ebooklib, mock_epub, mock_book


async def test_translate_epub_missing_deps(tmp_path: Path):
    """When ebooklib is absent, translate_epub_async raises StratumError."""
    with patch.dict(sys.modules, {"ebooklib": None, "ebooklib.epub": None, "bs4": None}):
        import importlib
        import oprim.translate.format_epub as _mod
        importlib.reload(_mod)
        prov = _mock_provider()
        with pytest.raises(StratumError, match="ebooklib"):
            await _mod.translate_epub_async(
                tmp_path / "in.epub", tmp_path / "out.epub", prov, "en", "zh"
            )
        importlib.reload(_mod)  # restore for subsequent tests


async def test_translate_epub_basic(tmp_path: Path):
    from oprim.translate import format_epub as _mod

    item = _make_epub_item("<p>Hello world.</p>")
    mock_ebooklib, mock_epub, mock_book = _build_epub_modules()
    mock_book.get_items_of_type.return_value = [item]

    with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub}), \
         patch.object(_mod, "translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("翻译段落", [
            TranslationResult("翻译段落", "mock", "mock", 10, 8, 0.001, "en", "zh")
        ])
        epub_in = tmp_path / "in.epub"
        epub_out = tmp_path / "out.epub"
        out, results = await _mod.translate_epub_async(
            epub_in, epub_out, _mock_provider(), "en", "zh"
        )

    assert out == epub_out
    assert len(results) == 1
    mock_epub.write_epub.assert_called_once_with(str(epub_out), mock_book)
    item.set_content.assert_called_once()


async def test_translate_epub_skips_empty_body(tmp_path: Path):
    from oprim.translate import format_epub as _mod

    item = _make_epub_item("")  # empty body
    mock_ebooklib, mock_epub, mock_book = _build_epub_modules()
    mock_book.get_items_of_type.return_value = [item]

    with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub}), \
         patch.object(_mod, "translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        epub_in = tmp_path / "in.epub"
        epub_out = tmp_path / "out.epub"
        out, results = await _mod.translate_epub_async(
            epub_in, epub_out, _mock_provider(), "en", "zh"
        )

    mock_tma.assert_not_called()
    assert results == []


async def test_translate_epub_no_body_tag(tmp_path: Path):
    from oprim.translate import format_epub as _mod

    item = MagicMock()
    item.get_content.return_value = b"<html></html>"
    item.get_name.return_value = "ch.xhtml"
    item.set_content = MagicMock()

    mock_ebooklib, mock_epub, mock_book = _build_epub_modules()
    mock_book.get_items_of_type.return_value = [item]

    with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub}), \
         patch.object(_mod, "translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        epub_in = tmp_path / "in.epub"
        epub_out = tmp_path / "out.epub"
        await _mod.translate_epub_async(epub_in, epub_out, _mock_provider(), "en", "zh")

    mock_tma.assert_not_called()


async def test_translate_epub_no_p_tag_appends_new(tmp_path: Path):
    """Body exists but has no <p>: translated text appended as new <p>."""
    from oprim.translate import format_epub as _mod

    # body with only a <div>, no <p>
    item = MagicMock()
    item.get_content.return_value = b"<html><body><div>Hello.</div></body></html>"
    item.get_name.return_value = "ch_nop.xhtml"
    item.set_content = MagicMock()

    mock_ebooklib, mock_epub, mock_book = _build_epub_modules()
    mock_book.get_items_of_type.return_value = [item]

    with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub}), \
         patch.object(_mod, "translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("无段落体", [
            TranslationResult("无段落体", "mock", "mock", 5, 4, 0.0, "en", "zh")
        ])
        epub_in = tmp_path / "in.epub"
        epub_out = tmp_path / "out.epub"
        out, results = await _mod.translate_epub_async(epub_in, epub_out, _mock_provider(), "en", "zh")

    assert len(results) == 1
    item.set_content.assert_called_once()


def test_translate_epub_sync_deprecated(tmp_path: Path):
    """Deprecated sync wrapper emits DeprecationWarning."""
    from oprim.translate import format_epub as _mod

    item = _make_epub_item("<p>Hello.</p>")
    mock_ebooklib, mock_epub, mock_book = _build_epub_modules()
    mock_book.get_items_of_type.return_value = [item]

    with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub}), \
         patch.object(_mod, "translate_markdown_async", new_callable=AsyncMock) as mock_tma:
        mock_tma.return_value = ("done", [])
        epub_in = tmp_path / "in.epub"
        epub_out = tmp_path / "out.epub"
        with pytest.warns(DeprecationWarning, match="translate_epub_async"):
            _mod.translate_epub(epub_in, epub_out, _mock_provider(), "en", "zh")
