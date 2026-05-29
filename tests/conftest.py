"""Shared test fixtures for oprim."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng):
    """Sample daily returns array (252 trading days)."""
    return rng.normal(0.0005, 0.02, size=252)


# ── Knowledge sub-package fixtures ──────────────────────────────────────────

@pytest.fixture()
def simple_pdf(tmp_path: Path) -> Path:
    import fitz
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "Hello World test PDF content for testing purposes.",
        "Second line: more content to ensure detection works properly.",
        "Third line: even more text to be well above the scanned threshold.",
        "Fourth line: making sure we have plenty of extractable text.",
    ]
    for i, line in enumerate(lines):
        page.insert_text((50, 50 + i * 20), line)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def multi_page_pdf(tmp_path: Path) -> Path:
    import fitz
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 50 + i * 20), f"Page {i + 1} content goes here.")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def simple_png(tmp_path: Path) -> Path:
    from PIL import Image
    path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(str(path))
    return path


@pytest.fixture()
def screen_size_png(tmp_path: Path) -> Path:
    from PIL import Image
    path = tmp_path / "screenshot.png"
    img = Image.new("RGB", (1920, 1080), color=(0, 128, 255))
    img.save(str(path))
    return path


@pytest.fixture()
def simple_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("Hello world! This is a plain text file.\n" * 10)
    return path


@pytest.fixture()
def simple_html(tmp_path: Path) -> Path:
    path = tmp_path / "sample.html"
    path.write_text(
        "<html><head><title>Test</title></head><body>"
        "<p>This is a test HTML document with enough content to extract.</p>"
        "<p>Second paragraph for good measure.</p>"
        "</body></html>"
    )
    return path


@pytest.fixture()
def simple_epub(tmp_path: Path) -> Path:
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier("test-id-001")
    book.set_title("Test Book")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Chapter 1", file_name="chapter1.xhtml", lang="en")
    chapter.content = (
        b"<html><body><h1>Chapter 1</h1>"
        b"<p>This is the first chapter of the test book. "
        b"It has some content to extract during testing.</p>"
        b"</body></html>"
    )
    book.add_item(chapter)
    book.toc = (epub.Link("chapter1.xhtml", "Chapter 1", "chapter1"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    path = tmp_path / "test.epub"
    epub.write_epub(str(path), book)
    return path
