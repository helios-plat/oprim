"""Compatibility entry point for the retired database update primitive."""

from typing import Any


async def db_update(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("db_update compatibility API requires the project persistence layer")


__all__ = ["db_update"]
