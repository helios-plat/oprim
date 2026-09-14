"""Compatibility import for the retired cache invalidation primitive."""

from obase.cache import cache_invalidate

_memory_cache: dict[str, object] = {}

__all__ = ["cache_invalidate", "_memory_cache"]
