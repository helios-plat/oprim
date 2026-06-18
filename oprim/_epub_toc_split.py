"""oprim.epub_toc_split — Split EPUB by top-level TOC into individual books."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ebooklib  # type: ignore[import-untyped]
from bs4 import BeautifulSoup
from ebooklib import epub

from oprim._exceptions import OprimError


@dataclass
class EpubBook:
    """Single book extracted from EPUB (may be one of many in a bundle)."""
    book_title: str
    toc_subtree: list[Any]       # TOC nodes belonging to this book
    content: str                 # Concatenated markdown text
    metadata: dict[str, str]     # title/author/language from DC metadata


def epub_toc_split(*, file_path: Path) -> list[EpubBook]:
    """Parse EPUB and split by top-level TOC into individual books.

    Single-book EPUB returns list of length 1.
    Bundle (e.g. collected works) returns list of length N.

    Args:
        file_path: Path to the EPUB file.

    Returns:
        List of EpubBook, one per top-level TOC entry.

    Raises:
        OprimError: File not found, DRM protected, or parse failed.

    Example:
        >>> books = epub_toc_split(file_path=Path("bundle.epub"))
        >>> for book in books:
        ...     print(book.book_title, len(book.content))
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise OprimError(f"file_not_found: {file_path}")

    try:
        book = epub.read_epub(str(file_path))
    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ("drm", "decrypt", "encrypted")):
            raise OprimError("drm_protected: EPUB is DRM-protected") from e
        raise OprimError(f"epub_parse_failed: {e}") from e

    # Global metadata
    def _dc(field: str) -> str:
        vals = book.get_metadata("DC", field)
        return str(vals[0][0]) if vals else ""

    base_meta = {
        "title": _dc("title"),
        "author": _dc("creator"),
        "language": _dc("language"),
    }

    # Build href → content map
    item_map: dict[str, str] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        item_map[item.get_name()] = text

    top_toc = book.toc
    if not top_toc:
        # No TOC — treat whole book as single entry
        full_content = "\n\n".join(item_map.values())
        return [EpubBook(
            book_title=base_meta["title"] or file_path.stem,
            toc_subtree=[],
            content=full_content,
            metadata=base_meta,
        )]

    def _collect_hrefs(nodes) -> list[str]:
        hrefs = []
        for node in nodes:
            if hasattr(node, "href"):
                hrefs.append(node.href.split("#")[0])
            if hasattr(node, "__iter__") and not hasattr(node, "href"):
                hrefs.extend(_collect_hrefs(node))
        return hrefs

    result: list[EpubBook] = []
    for top_node in top_toc:
        if hasattr(top_node, "href"):
            node_title = getattr(top_node, "title", "") or base_meta["title"]
            children = list(getattr(top_node, "__iter__", lambda: [])())
        elif isinstance(top_node, tuple) and len(top_node) == 2:
            section, children = top_node
            node_title = getattr(section, "title", "") or base_meta["title"]
        else:
            continue

        hrefs = _collect_hrefs(children) if children else []
        if hasattr(top_node, "href"):
            hrefs.insert(0, top_node.href.split("#")[0])

        parts = [item_map[h] for h in hrefs if h in item_map]
        content = "\n\n".join(parts)

        result.append(EpubBook(
            book_title=node_title,
            toc_subtree=list(children),
            content=content,
            metadata={**base_meta, "title": node_title},
        ))

    return result if result else [EpubBook(
        book_title=base_meta["title"] or file_path.stem,
        toc_subtree=[],
        content="\n\n".join(item_map.values()),
        metadata=base_meta,
    )]
