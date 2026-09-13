__all__ = ["parse_pdf", "ParsedContent", "PDFParser", "parse_epub", "parse_html"]


def parse_pdf(*args, **kwargs):
    """Lazy proxy for the PDF parser and its optional PyMuPDF dependency."""
    from oprim.parser.parse_pdf import parse_pdf as _parse_pdf

    return _parse_pdf(*args, **kwargs)


def __getattr__(name: str):
    """Load format-specific parsers only when explicitly requested."""
    if name == "parse_epub":
        from oprim.parser.parse_epub import parse_epub

        return parse_epub
    if name == "parse_html":
        from oprim.parser.parse_html import parse_html

        return parse_html
    if name == "ParsedContent":
        from oprim.parser._common import ParsedContent

        return ParsedContent
    if name == "PDFParser":
        from oprim.parser.parse_pdf import PDFParser

        return PDFParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
