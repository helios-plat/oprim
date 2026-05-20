"""Integration tests for GoogleDriveAdapter — requires live OAuth (Wiki one-time click).

Run with:
    pytest tests/storage/test_gdrive_integration.py -v -m integration
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from oprim.storage.providers.gdrive import GoogleDriveAdapter
from oprim.storage.protocol import StorageFile, StorageQuota, UploadResult

pytestmark = pytest.mark.integration


@pytest.fixture
async def gdrive():
    """Live GDrive adapter — calls real OAuth on first run."""
    adapter = GoogleDriveAdapter()
    await adapter.authenticate()
    return adapter


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    p = tmp_path / "stratum_test.txt"
    p.write_bytes(b"Stratum Phase 2 integration test " + str(time.time()).encode())
    return p


class TestGDriveIntegration:
    async def test_authenticate_succeeds(self, gdrive):
        assert gdrive._service is not None
        assert gdrive._creds is not None
        assert gdrive._creds.valid

    async def test_health_check(self, gdrive):
        assert await gdrive.health_check() is True

    async def test_upload_and_delete(self, gdrive, tmp_file):
        result = await gdrive.upload(str(tmp_file), "stratum_integration_test.txt")
        assert isinstance(result, UploadResult)
        assert result.file_id
        assert result.size > 0
        # cleanup
        await gdrive.delete(result.file_id)

    async def test_upload_download_roundtrip(self, gdrive, tmp_file, tmp_path):
        result = await gdrive.upload(str(tmp_file), "stratum_roundtrip.txt")
        dest = tmp_path / "downloaded.txt"
        await gdrive.download(result.file_id, str(dest))
        assert dest.read_bytes() == tmp_file.read_bytes()
        await gdrive.delete(result.file_id)

    async def test_list_files(self, gdrive, tmp_file):
        result = await gdrive.upload(str(tmp_file), "stratum_list_test.txt")
        files = [f async for f in gdrive.list_files("/Stratum")]
        assert any(f.file_id == result.file_id for f in files)
        await gdrive.delete(result.file_id)

    async def test_get_file_metadata(self, gdrive, tmp_file):
        result = await gdrive.upload(str(tmp_file), "stratum_meta_test.txt")
        meta = await gdrive.get_file_metadata(result.file_id)
        assert isinstance(meta, StorageFile)
        assert meta.file_id == result.file_id
        await gdrive.delete(result.file_id)

    async def test_get_quota(self, gdrive):
        q = await gdrive.get_quota()
        assert isinstance(q, StorageQuota)
        assert q.total_bytes > 0

    async def test_list_changes_since_start_token(self, gdrive):
        changes, token = await gdrive.list_changes_since(None)
        assert changes == []
        assert token

    async def test_list_changes_since_with_token(self, gdrive, tmp_file):
        _, start_token = await gdrive.list_changes_since(None)
        result = await gdrive.upload(str(tmp_file), "stratum_changes_test.txt")
        changes, next_token = await gdrive.list_changes_since(start_token)
        assert next_token
        await gdrive.delete(result.file_id)
