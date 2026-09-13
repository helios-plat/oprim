"""Parser result types shared by format-specific parsers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedContent:
    markdown: str
    plaintext: str
    page_count: int
    images: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_name: str = ""
    parse_quality_score: float = 0.0
