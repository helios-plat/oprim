from oprim.parser.parse_html import parse_html


def parse_epub(*args, **kwargs):
    from oprim.parser.parse_epub import parse_epub as _parse
    return _parse(*args, **kwargs)


def parse_pdf(*args, **kwargs):
    from oprim.parser.parse_pdf import parse_pdf as _parse
    return _parse(*args, **kwargs)


def __getattr__(name: str):
    if name in {"ParsedContent", "PDFParser"}:
        from oprim.parser.parse_pdf import ParsedContent, PDFParser
        return {"ParsedContent": ParsedContent, "PDFParser": PDFParser}[name]
    raise AttributeError(name)

__all__ = ["parse_pdf", "ParsedContent", "PDFParser", "parse_epub", "parse_html"]
