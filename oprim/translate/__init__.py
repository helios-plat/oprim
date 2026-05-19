"""oprim.translate — multi-provider translation pipeline."""
from __future__ import annotations

from pathlib import Path

from oprim._logging import log
from oprim.translate.checkpoint import TranslationCheckpoint
from oprim.translate.chunker import TextChunk, TranslationChunker
from oprim.translate.format_epub import translate_epub
from oprim.translate.format_md import translate_markdown
from oprim.translate.format_srt import translate_srt
from oprim.translate.protocol import TranslationProvider, TranslationRequest, TranslationResult
from oprim.translate.providers import get_provider
from oprim.translate.terminology import GlossaryEntry, TerminologyGlossary


def translate_document(
    text: str,
    source_lang: str,
    target_lang: str,
    provider: TranslationProvider | str = "deepseek",
    *,
    checkpoint_path: Path | None = None,
    max_chars: int = 2000,
    domain: str | None = None,
    model: str | None = None,
    glossary: TerminologyGlossary | None = None,
) -> tuple[str, list[TranslationResult]]:
    """Translate a markdown document using the specified provider.

    This is the primary entry point for oprim.translate. Internally uses
    TranslationChunker + TranslationCheckpoint for reliable long-document handling.

    Args:
        text: Source markdown text.
        source_lang: ISO language code for source (e.g. "en", "zh").
        target_lang: ISO language code for target (e.g. "zh", "en").
        provider: Provider name string or TranslationProvider instance.
        checkpoint_path: Optional file path for resumable translation.
        max_chars: Max characters per chunk sent to the provider.
        domain: Optional domain hint passed to the provider.
        model: Optional model override for the provider.
        glossary: Optional TerminologyGlossary for domain-specific terms.

    Returns:
        Tuple of (translated_text, list[TranslationResult]) — one result per
        translated chunk (untranslatable code blocks are passed through as-is).
    """
    if isinstance(provider, str):
        prov = get_provider(provider)
    else:
        prov = provider

    if glossary:
        text, token_map = glossary.protect(text, source_lang, target_lang)
    else:
        token_map = {}

    translated, results = translate_markdown(
        text,
        provider=prov,
        source_lang=source_lang,
        target_lang=target_lang,
        checkpoint_path=checkpoint_path,
        max_chars=max_chars,
        domain=domain,
        model=model,
    )

    if token_map:
        translated = glossary.restore(translated, token_map)  # type: ignore[union-attr]

    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    total_cost = sum(r.cost_usd for r in results)
    log.info(
        "translate.document_done",
        provider=prov.name,
        chunks=len(results),
        tokens_in=total_in,
        tokens_out=total_out,
        cost_usd=round(total_cost, 6),
    )

    return translated, results


__all__ = [
    "translate_document",
    "translate_markdown",
    "translate_epub",
    "translate_srt",
    "TranslationRequest",
    "TranslationResult",
    "TranslationProvider",
    "TranslationChunker",
    "TextChunk",
    "TranslationCheckpoint",
    "TerminologyGlossary",
    "GlossaryEntry",
    "get_provider",
]
