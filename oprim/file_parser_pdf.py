"""Public facade for PDF parsing."""
from __future__ import annotations
from typing import Any
from oprim._file_parser_pdf import file_parser_pdf as _file_parser_pdf

def file_parser_pdf(*args: Any, **kwargs: Any) -> Any:
    return _file_parser_pdf(*args, **kwargs)

__all__ = ["file_parser_pdf"]
