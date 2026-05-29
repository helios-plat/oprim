"""EPUB parser using ebooklib."""
from __future__ import annotations

import re
from pathlib import Path

import ebooklib
from ebooklib import epub

from oprim._logging import log as olog
from oprim.parser.parse_pdf import ParsedContent


def parse_epub(path: Path) -> ParsedContent:
    """Parse an EPUB file and return structured content.

    Raises:
        FileNotFoundError: file does not exist.
        Exception: propagated from ebooklib on parse failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        book = epub.read_epub(str(path))
        chapters = []
        all_md: list[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode("utf-8", errors="replace")
            plain = re.sub(r"<[^>]+>", " ", content)
            plain = re.sub(r"\s+", " ", plain).strip()
            title = item.get_name()
            chapters.append({"title": title, "content_len": len(plain)})
            all_md.append(f"## {title}\n\n{plain}")

        markdown = "\n\n".join(all_md)
        plaintext = re.sub(r"#{1,6}\s+", "", markdown)

        return ParsedContent(
            markdown=markdown,
            plaintext=plaintext,
            page_count=len(chapters),
            chapters=chapters,
            metadata={"title": book.title or ""},
            parser_name="ebooklib",
            parse_quality_score=0.8 if chapters else 0.1,
        )
    except Exception as e:
        olog.error("parse_epub failed", path=str(path), error=str(e))
        raise
