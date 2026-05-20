"""Tests for LocalStorageAdapter (no network required)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from oprim.storage.errors import FileNotFoundStorageError
from oprim.storage.protocol import StorageFile, StorageQuota, UploadResult
from oprim.storage.providers.local import LocalStorageAdapter


@pytest.fixture
def adapter(tmp_path: Path) -> LocalStorageAdapter:
    return LocalStorageAdapter(root_dir=tmp_path / "store")


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "hello.txt"
    p.write_bytes(b"hello stratum")
    return p


class TestAuthenticate:
    async def test_returns_true_when_dir_exists(self, adapter):
        ok = await adapter.authenticate()
        assert ok is True

    async def test_creates_root_on_init(self, tmp_path):
        root = tmp_path / "new_root"
        assert not root.exists()
        a = LocalStorageAdapter(root_dir=root)
        assert root.exists()
        ok = await a.authenticate()
        assert ok is True


class TestUpload:
    async def test_upload_creates_file(self, adapter, sample_file, tmp_path):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        assert isinstance(result, UploadResult)
        dest = adapter.root / "Stratum" / "hello.txt"
        assert dest.exists()
        assert dest.read_bytes() == b"hello stratum"

    async def test_upload_returns_correct_size(self, adapter, sample_file):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        assert result.size == len(b"hello stratum")

    async def test_upload_returns_md5(self, adapter, sample_file):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        import hashlib
        expected = hashlib.md5(b"hello stratum").hexdigest()
        assert result.md5 == expected

    async def test_upload_file_id_is_path(self, adapter, sample_file):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        assert result.file_id == str(adapter.root / "Stratum" / "hello.txt")

    async def test_upload_creates_nested_dirs(self, adapter, sample_file):
        await adapter.upload(str(sample_file), "Stratum/a/b/c/file.txt")
        assert (adapter.root / "Stratum" / "a" / "b" / "c" / "file.txt").exists()

    async def test_upload_calls_progress(self, adapter, sample_file):
        progress_vals = []
        await adapter.upload(str(sample_file), "Stratum/f.txt", on_progress=progress_vals.append)
        assert 1.0 in progress_vals

    async def test_upload_overwrites(self, adapter, sample_file, tmp_path):
        await adapter.upload(str(sample_file), "Stratum/hello.txt")
        new_file = tmp_path / "new.txt"
        new_file.write_bytes(b"updated content")
        await adapter.upload(str(new_file), "Stratum/hello.txt")
        dest = adapter.root / "Stratum" / "hello.txt"
        assert dest.read_bytes() == b"updated content"


class TestDownload:
    async def test_download_retrieves_file(self, adapter, sample_file, tmp_path):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        dest = tmp_path / "downloaded.txt"
        await adapter.download(result.file_id, str(dest))
        assert dest.read_bytes() == b"hello stratum"

    async def test_download_calls_progress(self, adapter, sample_file, tmp_path):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        progress_vals = []
        await adapter.download(result.file_id, str(tmp_path / "out.txt"), on_progress=progress_vals.append)
        assert 1.0 in progress_vals

    async def test_download_missing_raises(self, adapter, tmp_path):
        with pytest.raises(FileNotFoundStorageError):
            await adapter.download("/nonexistent/path/file.txt", str(tmp_path / "out.txt"))


class TestDelete:
    async def test_delete_removes_file(self, adapter, sample_file):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        path = Path(result.file_id)
        assert path.exists()
        await adapter.delete(result.file_id)
        assert not path.exists()

    async def test_delete_nonexistent_is_noop(self, adapter):
        await adapter.delete("/nonexistent/file.txt")  # must not raise


class TestListFiles:
    async def test_list_files_empty_folder(self, adapter):
        (adapter.root / "Stratum").mkdir(parents=True, exist_ok=True)
        files = [f async for f in adapter.list_files("/Stratum")]
        assert files == []

    async def test_list_files_missing_folder_empty(self, adapter):
        files = [f async for f in adapter.list_files("/NoFolder")]
        assert files == []

    async def test_list_files_returns_storage_files(self, adapter, sample_file):
        await adapter.upload(str(sample_file), "Stratum/hello.txt")
        files = [f async for f in adapter.list_files("/Stratum")]
        assert len(files) == 1
        f = files[0]
        assert isinstance(f, StorageFile)
        assert f.name == "hello.txt"
        assert f.size == len(b"hello stratum")

    async def test_list_files_multiple(self, adapter, tmp_path):
        for name in ("a.txt", "b.txt", "c.txt"):
            p = tmp_path / name
            p.write_bytes(b"x")
            await adapter.upload(str(p), f"Stratum/{name}")
        files = [f async for f in adapter.list_files("/Stratum")]
        assert len(files) == 3

    async def test_list_files_recursive(self, adapter, sample_file):
        await adapter.upload(str(sample_file), "Stratum/sub/nested.txt")
        flat = [f async for f in adapter.list_files("/Stratum", recursive=False)]
        recursive = [f async for f in adapter.list_files("/Stratum", recursive=True)]
        assert len(flat) == 0  # sub/ is a dir, no files at top level
        assert len(recursive) == 1


class TestGetFileMetadata:
    async def test_returns_storage_file(self, adapter, sample_file):
        result = await adapter.upload(str(sample_file), "Stratum/hello.txt")
        meta = await adapter.get_file_metadata(result.file_id)
        assert isinstance(meta, StorageFile)
        assert meta.name == "hello.txt"
        assert meta.size == len(b"hello stratum")
        assert meta.md5 is not None

    async def test_missing_raises(self, adapter):
        with pytest.raises(FileNotFoundStorageError):
            await adapter.get_file_metadata("/nonexistent/file.txt")


class TestGetQuota:
    async def test_returns_storage_quota(self, adapter):
        q = await adapter.get_quota()
        assert isinstance(q, StorageQuota)
        assert q.total_bytes > 0
        assert q.used_bytes >= 0
        assert q.available_bytes >= 0

    async def test_quota_consistent(self, adapter):
        q = await adapter.get_quota()
        assert q.used_bytes + q.available_bytes == q.total_bytes


class TestHealthCheck:
    async def test_healthy_when_root_exists(self, adapter):
        assert await adapter.health_check() is True

    async def test_unhealthy_when_root_removed(self, adapter):
        import shutil
        shutil.rmtree(adapter.root)
        assert await adapter.health_check() is False


class TestListChangesSince:
    async def test_raises_not_implemented(self, adapter):
        with pytest.raises(NotImplementedError):
            await adapter.list_changes_since(None)

    async def test_supports_changes_api_is_false(self, adapter):
        assert adapter.supports_changes_api is False
