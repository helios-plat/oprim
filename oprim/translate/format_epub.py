"""EPUB translation pipeline."""
from __future__ import annotations

from pathlib import Path

from oprim._logging import log
from oprim.errors import StratumError
from oprim.translate.format_md import translate_markdown
from oprim.translate.protocol import TranslationProvider, TranslationResult


def translate_epub(
    epub_path: Path,
    output_path: Path,
    provider: TranslationProvider,
    source_lang: str,
    target_lang: str,
    *,
    domain: str | None = None,
    model: str | None = None,
    max_chars: int = 2000,
) -> tuple[Path, list[TranslationResult]]:
    """Translate an EPUB file chapter by chapter.

    Reads the EPUB, translates each HTML chapter's text content,
    and writes a new EPUB to output_path.

    Returns:
        Tuple of (output_path, list[TranslationResult]).

    Raises:
        StratumError: If ebooklib is not installed or EPUB parsing fails.
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise StratumError(f"epub translation requires ebooklib + beautifulsoup4: {e}") from e

    book = epub.read_epub(str(epub_path))
    all_results: list[TranslationResult] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, "html.parser")
        body = soup.find("body")
        if not body:
            continue

        original_text = body.get_text(separator="\n\n")
        if not original_text.strip():
            continue

        translated_text, results = translate_markdown(
            original_text,
            provider=provider,
            source_lang=source_lang,
            target_lang=target_lang,
            max_chars=max_chars,
            domain=domain,
            model=model,
        )
        all_results.extend(results)

        for p in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            p.string = ""

        if body.p:
            body.p.string = translated_text
        else:
            new_p = soup.new_tag("p")
            new_p.string = translated_text
            body.append(new_p)

        item.set_content(str(soup).encode("utf-8"))
        log.info("translate.epub_chapter_done", item_name=item.get_name(), chunks=len(results))

    epub.write_epub(str(output_path), book)
    log.info("translate.epub_done", output=str(output_path), total_results=len(all_results))
    return output_path, all_results
