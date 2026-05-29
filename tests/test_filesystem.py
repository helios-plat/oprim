"""Tests for oprim filesystem operations."""

from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path

import pytest

from oprim import dir_archive_to_targz, disk_usage, file_checksum
from oprim._exceptions import OprimError, OprimNotFoundError
from oprim._filesystem import ArchiveResult, DiskUsage


# ---------------------------------------------------------------------------
# disk_usage
# ---------------------------------------------------------------------------

class TestDiskUsage:
    def test_root_path(self, tmp_path):
        result = disk_usage(path=str(tmp_path))
        assert isinstance(result, DiskUsage)
        assert result.total_bytes > 0
        assert result.used_bytes > 0
        assert result.free_bytes >= 0
        assert 0.0 <= result.used_percent <= 100.0

    def test_returns_resolved_path(self, tmp_path):
        result = disk_usage(path=str(tmp_path))
        assert result.path  # non-empty

    def test_nonexistent_path(self):
        with pytest.raises(OprimNotFoundError):
            disk_usage(path="/nonexistent/path/abcdefgh")

    def test_used_plus_free_approx_total(self, tmp_path):
        result = disk_usage(path=str(tmp_path))
        # used + free should be close to total (some slack for reserved blocks)
        assert result.used_bytes + result.free_bytes <= result.total_bytes + 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# dir_archive_to_targz
# ---------------------------------------------------------------------------

class TestDirArchiveToTargz:
    def _make_tree(self, base: Path) -> int:
        """Create a small directory tree; return count of regular files."""
        (base / "a.txt").write_text("hello")
        (base / "b.tmp").write_text("temp")
        sub = base / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("print('hi')")
        return 3  # a.txt, b.tmp, sub/c.py

    def test_basic_archive(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._make_tree(src)
        dst = str(tmp_path / "out.tar.gz")
        result = dir_archive_to_targz(src_dir=str(src), dst_path=dst)
        assert isinstance(result, ArchiveResult)
        assert result.file_count == 3
        assert result.archive_bytes > 0
        assert len(result.checksum_sha256) == 64

    def test_archive_valid_tarfile(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "hello.txt").write_text("world")
        dst = str(tmp_path / "out.tar.gz")
        dir_archive_to_targz(src_dir=str(src), dst_path=dst)
        assert tarfile.is_tarfile(dst)

    def test_exclude_patterns(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._make_tree(src)
        dst = str(tmp_path / "out.tar.gz")
        result = dir_archive_to_targz(
            src_dir=str(src),
            dst_path=dst,
            exclude_patterns=["*.tmp"],
        )
        assert result.file_count == 2  # b.tmp excluded

    def test_checksum_matches(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").write_text("content")
        dst = str(tmp_path / "out.tar.gz")
        result = dir_archive_to_targz(src_dir=str(src), dst_path=dst)
        # Verify checksum manually
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        assert result.checksum_sha256 == h.hexdigest()

    def test_nonexistent_src(self, tmp_path):
        with pytest.raises(OprimNotFoundError):
            dir_archive_to_targz(
                src_dir="/nonexistent/source",
                dst_path=str(tmp_path / "out.tar.gz"),
            )

    def test_symlinks_excluded_by_default(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        real = src / "real.txt"
        real.write_text("content")
        link = src / "link.txt"
        link.symlink_to(real)
        dst = str(tmp_path / "out.tar.gz")
        result = dir_archive_to_targz(src_dir=str(src), dst_path=dst, follow_symlinks=False)
        assert result.file_count == 1  # only real.txt counted, symlink skipped


# ---------------------------------------------------------------------------
# file_checksum
# ---------------------------------------------------------------------------

class TestFileChecksum:
    def test_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        result = file_checksum(file_path=str(f))
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    def test_md5(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        result = file_checksum(file_path=str(f), algorithm="md5")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

    def test_sha1(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        result = file_checksum(file_path=str(f), algorithm="sha1")
        expected = hashlib.sha1(b"hello world").hexdigest()
        assert result == expected

    def test_nonexistent_file(self):
        with pytest.raises(OprimNotFoundError):
            file_checksum(file_path="/nonexistent/file.txt")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = file_checksum(file_path=str(f))
        assert result == hashlib.sha256(b"").hexdigest()

    def test_large_file_streaming(self, tmp_path):
        f = tmp_path / "large.bin"
        data = b"x" * (200 * 1024)  # 200KB
        f.write_bytes(data)
        result = file_checksum(file_path=str(f), chunk_size=4096)
        assert result == hashlib.sha256(data).hexdigest()
