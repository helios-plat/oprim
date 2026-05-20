"""Markdown-format translation pipeline (async primary, sync deprecated)."""
from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

from oprim._logging import log
from oprim.translate.checkpoint import TranslationCheckpoint
from oprim.translate.chunker import TranslationChunker
from oprim.translate.protocol import TranslationProvider, TranslationRequest, TranslationResult


async def translate_markdown_async(
    text: str,
    provider: TranslationProvider,
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: Path | None = None,
    max_chars: int = 2000,
    domain: str | None = None,
    model: str | None = None,
) -> tuple[str, list[TranslationResult]]:
    """Translate a markdown document, preserving fenced code blocks (async).

    Args:
        text: Source markdown text.
        provider: TranslationProvider instance with async translate().
        source_lang: Source language code (e.g. "en").
        target_lang: Target language code (e.g. "zh").
        checkpoint_path: Optional path for resumable progress.
        max_chars: Max chars per translatable chunk.
        domain: Optional domain hint ("academic", "literary", "technical").
        model: Optional model override for the provider.

    Returns:
        Tuple of (translated_text, list[TranslationResult]).
    """
    chunker = TranslationChunker(max_chars=max_chars)
    chunks = chunker.split(text)
    checkpoint = TranslationCheckpoint(checkpoint_path) if checkpoint_path else None
    results: list[TranslationResult] = []
    translated_chunks = list(chunks)

    for chunk in chunks:
        if not chunk.translatable:
            continue
        if checkpoint and checkpoint.is_done(chunk.index):
            cached = checkpoint.get_chunk(chunk.index)
            translated_chunks[chunk.index] = chunk.__class__(
                index=chunk.index, text=cached or chunk.text, translatable=True
            )
            log.info("translate.chunk_cached", index=chunk.index)
            continue

        req = TranslationRequest(
            text=chunk.text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            model=model,
        )
        result = await provider.translate(req)
        results.append(result)
        translated_chunks[chunk.index] = chunk.__class__(
            index=chunk.index, text=result.text, translatable=True
        )
        if checkpoint:
            checkpoint.save_chunk(chunk.index, result.text)
        log.info(
            "translate.chunk_done",
            index=chunk.index,
            tokens_in=result.input_tokens,
            tokens_out=result.output_tokens,
        )

    translated_text = chunker.join(translated_chunks)
    if checkpoint:
        checkpoint.clear()
    return translated_text, results


def translate_markdown(
    text: str,
    provider: TranslationProvider,
    source_lang: str,
    target_lang: str,
    *,
    checkpoint_path: Path | None = None,
    max_chars: int = 2000,
    domain: str | None = None,
    model: str | None = None,
) -> tuple[str, list[TranslationResult]]:
    """Deprecated sync wrapper — use translate_markdown_async."""
    warnings.warn(
        "translate_markdown (sync) is deprecated; use translate_markdown_async instead. "
        "Will be removed in oprim 3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return asyncio.run(
        translate_markdown_async(
            text,
            provider,
            source_lang,
            target_lang,
            checkpoint_path=checkpoint_path,
            max_chars=max_chars,
            domain=domain,
            model=model,
        )
    )
