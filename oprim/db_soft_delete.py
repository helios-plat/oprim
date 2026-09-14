"""Compatibility import for the retired database soft-delete primitive."""

from typing import Any


async def db_soft_delete(*args: Any, **kwargs: Any) -> Any:
    """Compatibility entry point; canonical CRUD soft-delete is not exposed."""
    raise NotImplementedError(
        "db_soft_delete compatibility API requires the project persistence layer"
    )


__all__ = ["db_soft_delete"]
