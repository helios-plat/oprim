"""Storage adapter protocol — defines the interface every storage provider must satisfy."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Callable, Protocol, runtime_checkable


@dataclass
class StorageFile:
    file_id: str
    name: str
    size: int
    mime_type: str
    created_at: datetime
    modified_at: datetime
    md5: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class StorageQuota:
    used_bytes: int
    total_bytes: int
    available_bytes: int


@dataclass
class UploadResult:
    file_id: str
    size: int
    md5: str


@runtime_checkable
class StorageAdapter(Protocol):
    """Unified interface for all storage providers (GDrive, local, etc.)."""

    name: str
    supports_changes_api: bool
    max_file_size: int

    async def authenticate(self) -> bool: ...

    async def upload(
        self,
        local_path: str,
        remote_path: str,
        mime_type: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> UploadResult: ...

    async def download(
        self,
        file_id: str,
        local_path: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> None: ...

    async def delete(self, file_id: str) -> None: ...

    async def list_files(
        self,
        folder: str = "/Stratum",
        recursive: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[StorageFile]: ...

    async def get_file_metadata(self, file_id: str) -> StorageFile: ...

    async def get_quota(self) -> StorageQuota: ...

    async def list_changes_since(
        self,
        page_token: str | None,
    ) -> tuple[list[dict], str]: ...

    async def health_check(self) -> bool: ...
