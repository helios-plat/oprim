"""oprim.ext_search_delete — delete a search-index document via a registered
obase.SearchProvider (category="search").
"""

from __future__ import annotations

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import SearchIndexOprimError


async def ext_search_delete(provider: str, *, index: str, doc_id: str) -> bool:
    """Delete a document from a search index.

    Args:
        provider: Provider name registered in ProviderRegistry (category="search").
        index: Index/collection name.
        doc_id: Document ID to remove.

    Returns:
        True if a document was removed, False if it did not exist.

    Raises:
        SearchIndexOprimError: Provider not registered or the provider call itself raised.
    """
    try:
        search_provider = ProviderRegistry.get().generic("search", provider)
    except ProviderNotFoundError as exc:
        raise SearchIndexOprimError(f"search provider not found: {provider!r}", cause=exc) from exc

    try:
        return await search_provider.delete_doc(index=index, doc_id=doc_id)
    except Exception as exc:
        raise SearchIndexOprimError(f"delete_doc failed: {exc}", cause=exc) from exc
