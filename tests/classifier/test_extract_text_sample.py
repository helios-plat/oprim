"""Tests for oprim.classifier.extract_text_sample."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.classifier.extract_text_sample import extract_text_sample
from oprim.errors import UnsupportedFileTypeError


class TestExtractTextSample:
    def test_pdf_extraction(self, simple_pdf: Path):
        text = extract_text_sample(simple_pdf, "application/pdf")
        assert "Hello World" in text

    def test_text_plain(self, simple_txt: Path):
        text = extract_text_sample(simple_txt, "text/plain")
        assert "plain text" in text.lower()

    def test_text_markdown(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Title\n\nSome **markdown** content here.\n")
        text = extract_text_sample(path, "text/markdown")
        assert "Title" in text or "markdown" in text

    def test_html_extraction(self, simple_html: Path):
        text = extract_text_sample(simple_html, "text/html")
        assert isinstance(text, str)

    def test_epub_extraction(self, simple_epub: Path):
        text = extract_text_sample(simple_epub, "application/epub+zip")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_unsupported_mime_raises(self, simple_pdf: Path):
        with pytest.raises(UnsupportedFileTypeError):
            extract_text_sample(simple_pdf, "application/octet-stream")

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            extract_text_sample(tmp_path / "missing.txt", "text/plain")

    def test_max_chars_respected(self, simple_txt: Path):
        text = extract_text_sample(simple_txt, "text/plain", max_chars=10)
        assert len(text) <= 10

    def test_x_markdown_mime(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("Content here")
        text = extract_text_sample(path, "text/x-markdown")
        assert "Content" in text

    def test_pdf_max_chars_triggers_break(self, simple_pdf: Path):
        text = extract_text_sample(simple_pdf, "application/pdf", max_chars=3)
        assert len(text) <= 3

    def test_pdf_fitz_exception_returns_empty(self, simple_pdf: Path, monkeypatch):
        import fitz
        monkeypatch.setattr(fitz, "open", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fitz fail")))
        result = extract_text_sample(simple_pdf, "application/pdf")
        assert result == ""

    def test_empty_text_file_returns_empty(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        text = extract_text_sample(f, "text/plain")
        assert text == ""

    def test_text_bad_encoding_falls_back_to_utf8(self, tmp_path: Path, monkeypatch):
        import chardet
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello world")
        monkeypatch.setattr(chardet, "detect", lambda x: {"encoding": "invalid-codec-xxx"})
        text = extract_text_sample(f, "text/plain")
        assert "hello" in text

    def test_empty_html_returns_empty(self, tmp_path: Path):
        f = tmp_path / "empty.html"
        f.write_bytes(b"")
        text = extract_text_sample(f, "text/html")
        assert text == ""

    def test_epub_max_chars_triggers_break(self, simple_epub: Path):
        text = extract_text_sample(simple_epub, "application/epub+zip", max_chars=5)
        assert len(text) <= 5

    def test_epub_bad_file_returns_empty(self, tmp_path: Path):
        f = tmp_path / "bad.epub"
        f.write_bytes(b"not a real epub zip")
        text = extract_text_sample(f, "application/epub+zip")
        assert text == ""
