"""TranslationChunker — splits markdown text into translation-friendly segments."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    index: int
    text: str
    translatable: bool = True  # False for fenced code blocks


class TranslationChunker:
    """Splits markdown text into chunks suitable for sequential translation.

    Code fences (``` ... ```) are extracted as non-translatable chunks.
    Remaining prose is split on paragraph boundaries up to max_chars.
    """

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def split(self, text: str) -> list[TextChunk]:
        """Return ordered list of TextChunk."""
        # Split on fenced code blocks, preserving delimiters.
        raw_parts = re.split(r"(```[\s\S]*?```)", text)
        chunks: list[TextChunk] = []
        idx = 0

        for part in raw_parts:
            if not part:
                continue
            if part.startswith("```"):
                chunks.append(TextChunk(index=idx, text=part, translatable=False))
                idx += 1
            else:
                idx = self._split_prose(part, chunks, idx)

        return chunks

    def _split_prose(self, text: str, chunks: list[TextChunk], idx: int) -> int:
        paragraphs = re.split(r"\n\n+", text)
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > self.max_chars and current:
                chunks.append(TextChunk(index=idx, text="\n\n".join(current), translatable=True))
                idx += 1
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append(TextChunk(index=idx, text="\n\n".join(current), translatable=True))
            idx += 1

        return idx

    def join(self, chunks: list[TextChunk]) -> str:
        """Reassemble chunks (sorted by index) into a single string."""
        ordered = sorted(chunks, key=lambda c: c.index)
        return "\n\n".join(c.text for c in ordered if c.text)
