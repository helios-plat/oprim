"""Tests for oprim.storage protocol dataclasses."""
from __future__ import annotations

from datetime import datetime, timezone

from oprim.storage.protocol import StorageFile, StorageQuota, UploadResult


class TestStorageFile:
    def test_fields(self):
        now = datetime.now(tz=timezone.utc)
        f = StorageFile(
            file_id="abc123",
            name="test.pdf",
            size=1024,
            mime_type="application/pdf",
            created_at=now,
            modified_at=now,
            md5="deadbeef",
            metadata={"custom": True},
        )
        assert f.file_id == "abc123"
        assert f.name == "test.pdf"
        assert f.size == 1024
        assert f.mime_type == "application/pdf"
        assert f.md5 == "deadbeef"
        assert f.metadata == {"custom": True}

    def test_md5_optional(self):
        now = datetime.now(tz=timezone.utc)
        f = StorageFile("x", "x", 0, "text/plain", now, now, None)
        assert f.md5 is None

    def test_default_metadata(self):
        now = datetime.now(tz=timezone.utc)
        f = StorageFile("x", "x", 0, "text/plain", now, now, None)
        assert f.metadata == {}

    def test_metadata_not_shared(self):
        now = datetime.now(tz=timezone.utc)
        f1 = StorageFile("a", "a", 0, "text/plain", now, now, None)
        f2 = StorageFile("b", "b", 0, "text/plain", now, now, None)
        f1.metadata["key"] = "val"
        assert "key" not in f2.metadata


class TestStorageQuota:
    def test_fields(self):
        q = StorageQuota(used_bytes=500, total_bytes=1000, available_bytes=500)
        assert q.used_bytes == 500
        assert q.total_bytes == 1000
        assert q.available_bytes == 500

    def test_zero_quota(self):
        q = StorageQuota(0, 0, 0)
        assert q.used_bytes == 0


class TestUploadResult:
    def test_fields(self):
        r = UploadResult(file_id="xyz", size=2048, md5="abc")
        assert r.file_id == "xyz"
        assert r.size == 2048
        assert r.md5 == "abc"
