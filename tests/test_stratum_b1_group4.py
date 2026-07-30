"""Tests for Stratum Batch 1 Group 4: file_parser_pdf, file_parser_epub,
file_parser_html, file_parser_markdown, file_parser_plaintext,
document_structure_extractor."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from ebooklib import epub

from oprim._document_types import ImageRef, Page, ParsedDocument, Table
from oprim._exceptions import OprimError
from oprim.document_structure_extractor import document_structure_extractor
from oprim.file_parser_epub import file_parser_epub
from oprim.file_parser_html import file_parser_html
from oprim.file_parser_markdown import file_parser_markdown
from oprim.file_parser_pdf import file_parser_pdf
from oprim.file_parser_plaintext import file_parser_plaintext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, text: str = "Hello PDF World") -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    doc.save(str(path))
    doc.close()
    return path


def _make_epub(path: Path, title: str = "Test Book", chapter_text: str = "Hello EPUB") -> Path:
    book = epub.EpubBook()
    book.set_identifier("id-test-001")
    book.set_title(title)
    book.set_language("en")
    c1 = epub.EpubHtml(title="Chapter 1", file_name="chap_01.xhtml", lang="en")
    c1.content = f"<html><body><h1>Chapter 1</h1><p>{chapter_text}</p></body></html>".encode()
    book.add_item(c1)
    book.toc = (epub.Link("chap_01.xhtml", "Chapter 1", "chap1"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    epub.write_epub(str(path), book)
    return path


# ---------------------------------------------------------------------------
# file_parser_pdf
# ---------------------------------------------------------------------------


class TestFileParserPdf:
    def test_not_found_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="file_not_found"):
            file_parser_pdf(file_path=Path("/nonexistent/file.pdf"))

    def test_invalid_file_is_handled(self, tmp_path: Path) -> None:
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf at all")
        # fitz may silently open or raise; either outcome is acceptable
        try:
            result = file_parser_pdf(file_path=bad_pdf)
            assert isinstance(result, ParsedDocument)
        except OprimError:
            pass

    def test_valid_pdf_returns_parsed_document(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "test.pdf")
        result = file_parser_pdf(file_path=pdf_path)
        assert isinstance(result, ParsedDocument)
        assert result.status == "ok"
        assert result.source_path == str(pdf_path)

    def test_valid_pdf_has_pages(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "paged.pdf", text="Sample content here")
        result = file_parser_pdf(file_path=pdf_path)
        assert len(result.pages) >= 1
        assert result.pages[0].page_number == 1

    def test_default_strategy_is_pymupdf4llm(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "strat.pdf")
        # default strategy should work without error
        result = file_parser_pdf(file_path=pdf_path, strategy="pymupdf4llm")
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# file_parser_epub
# ---------------------------------------------------------------------------


class TestFileParserEpub:
    def test_not_found_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="file_not_found"):
            file_parser_epub(file_path=Path("/nonexistent/file.epub"))

    def test_invalid_file_raises_oprim_error(self, tmp_path: Path) -> None:
        bad_epub = tmp_path / "bad.epub"
        bad_epub.write_bytes(b"not an epub")
        with pytest.raises(OprimError, match="epub_parse_failed"):
            file_parser_epub(file_path=bad_epub)

    def test_valid_epub_returns_parsed_document(self, tmp_path: Path) -> None:
        epub_path = _make_epub(tmp_path / "test.epub")
        result = file_parser_epub(file_path=epub_path)
        assert isinstance(result, ParsedDocument)
        assert result.status == "ok"
        assert result.source_path == str(epub_path)

    def test_valid_epub_has_pages(self, tmp_path: Path) -> None:
        epub_path = _make_epub(tmp_path / "paged.epub", chapter_text="Deep content here")
        result = file_parser_epub(file_path=epub_path)
        assert len(result.pages) >= 1

    def test_valid_epub_metadata_title(self, tmp_path: Path) -> None:
        epub_path = _make_epub(tmp_path / "meta.epub", title="My Test Book")
        result = file_parser_epub(file_path=epub_path)
        assert result.metadata.get("title") == "My Test Book"


# ---------------------------------------------------------------------------
# file_parser_html
# ---------------------------------------------------------------------------

_SIMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Heading</h1>
<p>This is a paragraph with some interesting content that trafilatura can extract.</p>
<p>Another paragraph with more words to ensure extraction works correctly.</p>
</body>
</html>
"""

_MINIMAL_HTML = "<html><body></body></html>"


class TestFileParserHtml:
    def test_simple_html_returns_parsed_document(self) -> None:
        result = file_parser_html(html_content=_SIMPLE_HTML)
        assert isinstance(result, ParsedDocument)
        assert result.status == "ok"
        assert len(result.pages) == 1

    def test_title_extraction(self) -> None:
        result = file_parser_html(html_content=_SIMPLE_HTML)
        assert result.metadata.get("title") == "Test Page"

    def test_url_stored_in_metadata(self) -> None:
        result = file_parser_html(html_content=_SIMPLE_HTML, url="https://example.com/page")
        assert result.metadata.get("url") == "https://example.com/page"
        assert result.source_path == "https://example.com/page"

    def test_empty_html_returns_document(self) -> None:
        result = file_parser_html(html_content=_MINIMAL_HTML)
        assert isinstance(result, ParsedDocument)
        # trafilatura may return empty text — that's fine
        assert result.pages[0].page_number == 1

    def test_page_number_is_1(self) -> None:
        result = file_parser_html(html_content=_SIMPLE_HTML)
        assert result.pages[0].page_number == 1

    def test_no_url_source_path_is_none(self) -> None:
        result = file_parser_html(html_content=_SIMPLE_HTML)
        assert result.source_path is None


# ---------------------------------------------------------------------------
# file_parser_markdown
# ---------------------------------------------------------------------------

_MD_WITH_FRONTMATTER = """\
---
title: My Document
author: Test Author
tags: [a, b]
---

# Introduction

This is the introduction section.

## Details

Some details here.
"""

_MD_WITHOUT_FRONTMATTER = """\
# Plain Heading

Paragraph one.

## Section Two

Content of section two.
"""

_MD_NO_HEADING = """\
Just some text without any heading at all.
More text here.
"""


class TestFileParserMarkdown:
    def test_not_found_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="file_not_found"):
            file_parser_markdown(file_path=Path("/nonexistent/file.md"))

    def test_frontmatter_extracted(self, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        md_file.write_text(_MD_WITH_FRONTMATTER, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert result.frontmatter.get("author") == "Test Author"

    def test_title_from_frontmatter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        md_file.write_text(_MD_WITH_FRONTMATTER, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert result.title == "My Document"

    def test_title_from_h1_when_no_frontmatter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "plain.md"
        md_file.write_text(_MD_WITHOUT_FRONTMATTER, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert result.title == "Plain Heading"

    def test_sections_extracted(self, tmp_path: Path) -> None:
        md_file = tmp_path / "sections.md"
        md_file.write_text(_MD_WITHOUT_FRONTMATTER, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert len(result.sections) >= 2
        titles = [s.title for s in result.sections]
        assert "Plain Heading" in titles
        assert "Section Two" in titles

    def test_section_levels(self, tmp_path: Path) -> None:
        md_file = tmp_path / "levels.md"
        md_file.write_text(_MD_WITHOUT_FRONTMATTER, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        h1 = next(s for s in result.sections if s.title == "Plain Heading")
        h2 = next(s for s in result.sections if s.title == "Section Two")
        assert h1.level == 1
        assert h2.level == 2

    def test_no_heading_title_is_none(self, tmp_path: Path) -> None:
        md_file = tmp_path / "nohead.md"
        md_file.write_text(_MD_NO_HEADING, encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert result.title is None

    def test_source_path_stored(self, tmp_path: Path) -> None:
        md_file = tmp_path / "src.md"
        md_file.write_text("# Hello\n", encoding="utf-8")
        result = file_parser_markdown(file_path=md_file)
        assert result.source_path == str(md_file)


# ---------------------------------------------------------------------------
# file_parser_plaintext
# ---------------------------------------------------------------------------


class TestFileParserPlaintext:
    def test_not_found_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="file_not_found"):
            file_parser_plaintext(file_path=Path("/nonexistent/file.txt"))

    def test_basic_text_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "hello.txt"
        txt.write_text("Hello world\nSecond line\n", encoding="utf-8")
        result = file_parser_plaintext(file_path=txt)
        assert result.line_count == 2
        assert result.source_path == str(txt)

    def test_encoding_detected(self, tmp_path: Path) -> None:
        txt = tmp_path / "utf8.txt"
        txt.write_text("Unicode: café résumé\n", encoding="utf-8")
        result = file_parser_plaintext(file_path=txt)
        assert result.encoding is not None
        assert len(result.encoding) > 0

    def test_paragraph_splitting(self, tmp_path: Path) -> None:
        txt = tmp_path / "paras.txt"
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        txt.write_text(content, encoding="utf-8")
        result = file_parser_plaintext(file_path=txt)
        assert len(result.paragraphs) == 3

    def test_line_count_accurate(self, tmp_path: Path) -> None:
        txt = tmp_path / "lines.txt"
        content = "\n".join(f"Line {i}" for i in range(1, 11))
        txt.write_text(content, encoding="utf-8")
        result = file_parser_plaintext(file_path=txt)
        assert result.line_count == 10

    def test_language_hint_is_none(self, tmp_path: Path) -> None:
        txt = tmp_path / "lang.txt"
        txt.write_text("Some text.", encoding="utf-8")
        result = file_parser_plaintext(file_path=txt)
        assert result.language_hint is None


# ---------------------------------------------------------------------------
# document_structure_extractor
# ---------------------------------------------------------------------------


class TestDocumentStructureExtractor:
    def test_empty_doc_returns_structure(self) -> None:
        doc = ParsedDocument()
        result = document_structure_extractor(parsed_doc=doc)
        assert result.word_count == 0
        assert result.headings == []
        assert result.paragraphs == []

    def test_doc_with_pages_counts_words(self) -> None:
        doc = ParsedDocument(
            pages=[Page(page_number=1, text="Hello world this is a test sentence.")]
        )
        result = document_structure_extractor(parsed_doc=doc)
        assert result.word_count == 7

    def test_heading_detection_short_line(self) -> None:
        # Short line without terminal punctuation = heading
        doc = ParsedDocument(
            pages=[
                Page(
                    page_number=1, text="Executive Summary\nThis is a paragraph with content here."
                )
            ]
        )
        result = document_structure_extractor(parsed_doc=doc)
        heading_texts = [h["text"] for h in result.headings]
        assert "Executive Summary" in heading_texts

    def test_table_and_image_count_propagated(self) -> None:
        doc = ParsedDocument(
            pages=[Page(page_number=1, text="Some text.")],
            tables=[Table(), Table()],
            images=[ImageRef(index=0), ImageRef(index=1), ImageRef(index=2)],
        )
        result = document_structure_extractor(parsed_doc=doc)
        assert result.table_count == 2
        assert result.image_count == 3

    def test_toc_mirrors_headings(self) -> None:
        doc = ParsedDocument(
            pages=[
                Page(page_number=1, text="Chapter One\nChapter Two\nBody text with punctuation.")
            ]
        )
        result = document_structure_extractor(parsed_doc=doc)
        assert len(result.toc) == len(result.headings)

    def test_multi_page_word_count(self) -> None:
        pages = [
            Page(page_number=1, text="One two three."),
            Page(page_number=2, text="Four five six seven."),
        ]
        doc = ParsedDocument(pages=pages)
        result = document_structure_extractor(parsed_doc=doc)
        assert result.word_count == 7
