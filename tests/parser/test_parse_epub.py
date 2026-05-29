"""Tests for oprim.parser.parse_epub."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.parser.parse_epub import parse_epub
from oprim.parser.parse_pdf import ParsedContent


class TestParseEpub:
    def test_basic_parse(self, simple_epub: Path):
        result = parse_epub(simple_epub)
        assert isinstance(result, ParsedContent)
        assert result.parser_name == "ebooklib"
        assert result.page_count >= 1
        assert len(result.markdown) > 0

    def test_chapter_extracted(self, simple_epub: Path):
        result = parse_epub(simple_epub)
        assert len(result.chapters) >= 1
        chapter = result.chapters[0]
        assert "title" in chapter
        assert "content_len" in chapter

    def test_metadata_title(self, simple_epub: Path):
        result = parse_epub(simple_epub)
        assert "title" in result.metadata

    def test_quality_score_positive(self, simple_epub: Path):
        result = parse_epub(simple_epub)
        assert result.parse_quality_score >= 0.1

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_epub(tmp_path / "missing.epub")

    def test_content_in_markdown(self, simple_epub: Path):
        result = parse_epub(simple_epub)
        assert "Chapter" in result.markdown or len(result.markdown) > 10

    def test_plaintext_no_headers(self, simple_epub: Path):
        """plaintext should not contain markdown ## headers."""
        result = parse_epub(simple_epub)
        # Plaintext strips headers
        assert "## " not in result.plaintext
