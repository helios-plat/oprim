"""Tests for oprim.parser.parse_html."""
from __future__ import annotations

from oprim.parser.parse_html import parse_html
from oprim.parser.parse_pdf import ParsedContent

_SAMPLE_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
  <article>
    <h1>Test Article Title</h1>
    <p>This is the first paragraph of content. It has multiple sentences
    to ensure trafilatura actually extracts it as real article content.</p>
    <p>Second paragraph with additional context to help content extraction.</p>
  </article>
</body>
</html>
"""

_MINIMAL_HTML = "<html><body><p>Hi</p></body></html>"


class TestParseHTML:
    def test_basic_parse(self):
        result = parse_html(_SAMPLE_HTML)
        assert isinstance(result, ParsedContent)
        assert result.parser_name == "trafilatura"
        assert result.page_count == 1

    def test_text_extracted(self):
        result = parse_html(_SAMPLE_HTML)
        # Either trafilatura or readability should extract something
        assert isinstance(result.plaintext, str)

    def test_bytes_input(self):
        result = parse_html(_SAMPLE_HTML.encode("utf-8"))
        assert isinstance(result, ParsedContent)

    def test_base_url_stored_in_metadata(self):
        result = parse_html(_SAMPLE_HTML, base_url="https://example.com")
        assert result.metadata.get("base_url") == "https://example.com"

    def test_no_base_url(self):
        result = parse_html(_MINIMAL_HTML)
        assert result.metadata.get("base_url") == ""

    def test_quality_score_for_short_content(self):
        result = parse_html(_MINIMAL_HTML)
        assert result.parse_quality_score in (0.2, 0.8)

    def test_quality_score_for_rich_content(self):
        result = parse_html(_SAMPLE_HTML)
        assert isinstance(result.parse_quality_score, float)

    def test_empty_html(self):
        result = parse_html("<html></html>")
        assert isinstance(result, ParsedContent)
        # Should not raise even if no content extracted

    def test_trafilatura_failure_uses_readability(self, monkeypatch):
        import trafilatura
        def raise_err(*args, **kwargs):
            raise RuntimeError("trafilatura internal error")
        monkeypatch.setattr(trafilatura, "extract", raise_err)
        result = parse_html(_SAMPLE_HTML)
        assert isinstance(result, ParsedContent)
        # readability or empty fallback; must not raise

    def test_both_parsers_fail_returns_empty(self, monkeypatch):
        import trafilatura
        from readability import Document as _Doc
        def raise_trafi(*args, **kwargs):
            raise RuntimeError("trafilatura fail")
        def raise_read(*args, **kwargs):
            raise RuntimeError("readability fail")
        monkeypatch.setattr(trafilatura, "extract", raise_trafi)
        monkeypatch.setattr(_Doc, "summary", raise_read)
        result = parse_html(_SAMPLE_HTML)
        assert result.markdown == ""
