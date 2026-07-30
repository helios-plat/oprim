"""oprim.ext_search_upsert — upsert a search-index document via a registered
obase.SearchProvider (category="search").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import SearchIndexOprimError


async def ext_search_upsert(provider: str, *, index: str, document: dict[str, Any]) -> bool:
    """Upsert a document into a search index.

    Args:
        provider: Provider name registered in ProviderRegistry (category="search").
        index: Index/collection name.
        document: Document to index (must be JSON-serializable).

    Returns:
        True on success.

    Raises:
        SearchIndexOprimError: Provider not registered or the provider call failed.
    """
    try:
        search_provider = ProviderRegistry.get().generic("search", provider)
    except ProviderNotFoundError as exc:
        raise SearchIndexOprimError(f"search provider not found: {provider!r}", cause=exc) from exc

    try:
        return await search_provider.upsert_doc(index=index, document=document)
    except Exception as exc:
        raise SearchIndexOprimError(f"upsert_doc failed: {exc}", cause=exc) from exc
