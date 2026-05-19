"""SRT subtitle translation pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass

from oprim._logging import log
from oprim.translate.protocol import TranslationProvider, TranslationRequest, TranslationResult


@dataclass
class SrtBlock:
    index: int
    timecode: str
    text: str


def _parse_srt(srt: str) -> list[SrtBlock]:
    blocks: list[SrtBlock] = []
    raw = srt.strip().replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\n+", raw):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        timecode = lines[1].strip()
        text = "\n".join(lines[2:]).strip()
        blocks.append(SrtBlock(index=idx, timecode=timecode, text=text))
    return blocks


def _render_srt(blocks: list[SrtBlock]) -> str:
    parts: list[str] = []
    for b in blocks:
        parts.append(f"{b.index}\n{b.timecode}\n{b.text}")
    return "\n\n".join(parts) + "\n"


def translate_srt(
    srt_text: str,
    provider: TranslationProvider,
    source_lang: str,
    target_lang: str,
    *,
    domain: str | None = None,
    model: str | None = None,
    batch_size: int = 20,
) -> tuple[str, list[TranslationResult]]:
    """Translate SRT subtitle content, preserving timecodes.

    Batches subtitle lines to reduce API calls.

    Returns:
        Tuple of (translated_srt_text, list[TranslationResult]).
    """
    blocks = _parse_srt(srt_text)
    if not blocks:
        return srt_text, []

    results: list[TranslationResult] = []
    translated_blocks: list[SrtBlock] = []

    for batch_start in range(0, len(blocks), batch_size):
        batch = blocks[batch_start : batch_start + batch_size]
        combined = "\n---\n".join(b.text for b in batch)
        req = TranslationRequest(
            text=combined,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            model=model,
        )
        result = provider.translate(req)
        results.append(result)

        translated_parts = result.text.split("\n---\n")
        for i, block in enumerate(batch):
            translated_text = translated_parts[i].strip() if i < len(translated_parts) else block.text
            translated_blocks.append(SrtBlock(
                index=block.index,
                timecode=block.timecode,
                text=translated_text,
            ))
        log.info(
            "translate.srt_batch_done",
            batch_start=batch_start,
            count=len(batch),
            tokens_in=result.input_tokens,
        )

    return _render_srt(translated_blocks), results
