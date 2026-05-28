"""Shared Protocol definitions for IO oprims.

These Protocols decouple IO oprims from specific HTTP/DB/Redis implementations,
enabling dependency injection and testability.
"""

from __future__ import annotations

from typing import Any, Protocol


class HttpClient(Protocol):
    """Protocol for async HTTP client operations."""

    async def get(self, url: str, **kwargs: Any) -> dict | list | None: ...
    async def post(self, url: str, **kwargs: Any) -> dict | list | None: ...


class DbExecutor(Protocol):
    """Protocol for async database operations."""

    async def fetch_one(self, query: str, params: dict | None = None) -> dict | None: ...
    async def fetch_all(self, query: str, params: dict | None = None) -> list[dict]: ...
    async def execute(self, query: str, params: dict | None = None) -> int: ...


class CacheClient(Protocol):
    """Protocol for async cache (Redis) operations."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...
    async def publish(self, channel: str, message: str) -> None: ...
