"""Compatibility entry point for the retired database write primitive."""

from typing import Any


async def db_write(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("db_write compatibility API requires the project persistence layer")


__all__ = ["db_write"]
