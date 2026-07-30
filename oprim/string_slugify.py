"""oprim.string_slugify — turn arbitrary text into a URL-safe slug."""

from __future__ import annotations

import re

_NON_WORD_RUN = re.compile(r"[^\w]+", re.UNICODE)
_EDGE_HYPHENS = re.compile(r"^-+|-+$")


def string_slugify(text: str) -> str:
    """Convert `text` into a lowercase, hyphen-separated slug.

    Non-Latin scripts (e.g. Chinese product titles) are preserved rather than
    stripped — `\\w` under `re.UNICODE` covers CJK characters, so this stays
    usable for the batch-warehouse commerce vertical's Chinese-language catalog.

    Args:
        text: Arbitrary input text.

    Returns:
        Lowercased slug with runs of non-word characters collapsed to a
        single hyphen and leading/trailing hyphens stripped.

    Example:
        >>> string_slugify("Men's Classic  T-Shirt!")
        'men-s-classic-t-shirt'
        >>> string_slugify("经典T恤")
        '经典t恤'
    """
    collapsed = _NON_WORD_RUN.sub("-", text.strip().lower())
    return _EDGE_HYPHENS.sub("", collapsed)
