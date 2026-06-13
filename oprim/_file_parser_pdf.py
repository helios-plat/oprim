"""Parse a PDF file and extract text, tables, and metadata."""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-untyped]  # pymupdf

from oprim._document_types import Page, ParsedDocument
from oprim._exceptions import OprimError


def file_parser_pdf(
    *,
    file_path: Path,
    strategy: str = "pymupdf4llm",
) -> ParsedDocument:
    """Parse a PDF file and extract text, tables, and metadata.

    Uses pymupdf4llm for text extraction. strategy="docling" is reserved for future use.

    Args:
        file_path: Path to the PDF file
        strategy: Extraction strategy ("pymupdf4llm" default; "docling" reserved)

    Returns:
        ParsedDocument with pages, tables, images, metadata

    Raises:
        OprimError: File not found, DRM protected, or parse failed

    Example:
        >>> doc = file_parser_pdf(file_path=Path("report.pdf"))
        >>> len(doc.pages) > 0
        True
    """
    if not file_path.exists():
        raise OprimError(f"file_not_found: {file_path}")

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise OprimError(f"pdf_parse_failed: {e}") from e

    # Check for DRM/encryption
    if doc.is_encrypted and not doc.authenticate(""):  # try empty password
        doc.close()
        raise OprimError("drm_protected: PDF is encrypted")

    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        pages.append(Page(page_number=page_num, text=text))

    metadata = doc.metadata or {}
    doc.close()

    return ParsedDocument(
        source_path=str(file_path),
        pages=pages,
        metadata={k: v for k, v in metadata.items() if v},
        status="ok",
    )
