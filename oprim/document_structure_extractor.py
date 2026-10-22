"""Public facade for document structure extraction."""
from __future__ import annotations
from typing import Any
from oprim._document_structure_extractor import DocumentStructure, document_structure_extractor as _document_structure_extractor

def document_structure_extractor(*args: Any, **kwargs: Any) -> Any:
    return _document_structure_extractor(*args, **kwargs)

__all__ = ["DocumentStructure", "document_structure_extractor"]
