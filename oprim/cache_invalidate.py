from __future__ import annotations

import warnings

import redis

from oprim._exceptions import OprimError


def cache_invalidate(
    *,
    key: str,
    cache_backend: str = "redis",
    redis_url: str | None = None,
) -> bool:
    """Invalidate (delete) a single cache key.

    Args:
        key: Cache key to invalidate
        cache_backend: "redis" | "memory" (in-process dict, for testing)
        redis_url: Redis connection URL. Required for "redis" backend.
                   Defaults to "redis://localhost:6379/0" if not provided.

    Returns:
        True if key existed and was deleted, False if key was not found

    Raises:
        OprimError: Connection failed or unsupported backend

    Example:
        >>> cache_invalidate(key="user:123:profile", redis_url="redis://localhost:6379/0")
        True
    """
    warnings.warn(
        "oprim.cache_invalidate is deprecated and will be removed in oprim v3.0.0. "
        "Migrate to a dedicated cache layer.",
        DeprecationWarning,
        stacklevel=2,
    )
    if cache_backend == "redis":
        try:
            url = redis_url or "redis://localhost:6379/0"
            client = redis.Redis.from_url(url)
            deleted = client.delete(key)
            return int(deleted) > 0  # type: ignore[arg-type]
        except Exception as e:
            raise OprimError(f"cache_invalidate redis failed: {e}") from e
    elif cache_backend == "memory":
        # For testing: use a module-level dict
        return _memory_cache.pop(key, None) is not None
    else:
        raise OprimError(f"unsupported cache_backend: {cache_backend}")


_memory_cache: dict[str, object] = {}
