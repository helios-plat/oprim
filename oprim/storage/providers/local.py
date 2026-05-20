"""Local filesystem storage adapter (testing and offline use)."""
from __future__ import annotations

import asyncio
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

from oprim._logging import log
from oprim.storage.errors import FileNotFoundStorageError
from oprim.storage.protocol import StorageFile, StorageQuota, UploadResult


class LocalStorageAdapter:
    """Local filesystem storage — useful for tests and privacy-sensitive scenarios."""

    name = "local"
    supports_changes_api = False
    max_file_size = 100 * 1024**3  # 100 GB

    def __init__(
        self,
        root_dir: Path = Path.home() / ".stratum" / "storage_local",
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Auth ──────────────────────────────────────────────────────────────

    async def authenticate(self) -> bool:
        ok = self.root.exists() and self.root.is_dir()
        log.info("local_storage_authenticate", root=str(self.root), ok=ok)
        return ok

    # ── CRUD ──────────────────────────────────────────────────────────────

    async def upload(
        self,
        local_path: str,
        remote_path: str,
        mime_type: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> UploadResult:
        dest = self.root / remote_path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, local_path, str(dest))

        md5 = await asyncio.to_thread(_md5, dest)
        size = dest.stat().st_size
        if on_progress:
            on_progress(1.0)
        log.info("local_upload_ok", dest=str(dest), size=size)
        return UploadResult(file_id=str(dest), size=size, md5=md5)

    async def download(
        self,
        file_id: str,
        local_path: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        src = Path(file_id)
        if not src.exists():
            raise FileNotFoundStorageError(f"File not found: {file_id}")
        await asyncio.to_thread(shutil.copy2, str(src), local_path)
        if on_progress:
            on_progress(1.0)
        log.info("local_download_ok", file_id=file_id, local=local_path)

    async def delete(self, file_id: str) -> None:
        p = Path(file_id)
        if p.exists():
            p.unlink()
            log.info("local_delete_ok", file_id=file_id)
        else:
            log.warning("local_delete_noop", file_id=file_id, reason="file not found")

    async def list_files(
        self,
        folder: str = "/Stratum",
        recursive: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[StorageFile]:
        base = self.root / folder.lstrip("/")
        if not base.exists():
            return
        pattern = "**/*" if recursive else "*"
        for p in sorted(base.glob(pattern)):
            if p.is_file():
                st = p.stat()
                yield StorageFile(
                    file_id=str(p),
                    name=p.name,
                    size=st.st_size,
                    mime_type="application/octet-stream",
                    created_at=datetime.fromtimestamp(st.st_ctime, tz=timezone.utc),
                    modified_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                    md5=None,
                    metadata={},
                )

    async def get_file_metadata(self, file_id: str) -> StorageFile:
        p = Path(file_id)
        if not p.exists():
            raise FileNotFoundStorageError(f"File not found: {file_id}")
        st = p.stat()
        return StorageFile(
            file_id=str(p),
            name=p.name,
            size=st.st_size,
            mime_type="application/octet-stream",
            created_at=datetime.fromtimestamp(st.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            md5=await asyncio.to_thread(_md5, p),
            metadata={},
        )

    async def get_quota(self) -> StorageQuota:
        usage = await asyncio.to_thread(shutil.disk_usage, str(self.root))
        return StorageQuota(
            used_bytes=usage.used,
            total_bytes=usage.total,
            available_bytes=usage.free,
        )

    async def list_changes_since(self, page_token: str | None) -> tuple[list[dict], str]:
        raise NotImplementedError(
            "LocalStorageAdapter does not support the changes API — use polling instead."
        )

    async def health_check(self) -> bool:
        ok = self.root.exists() and self.root.is_dir()
        if not ok:
            log.warning("local_health_check_failed", root=str(self.root))
        return ok


# ── helpers ───────────────────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
