"""Tests for oprim.disk_cleanup — allowlist-constrained cleanup (aegis DESIGN §9 S2)."""

from __future__ import annotations

import pytest

from oprim import disk_cleanup
from oprim._exceptions import OprimValidationError
from oprim._filesystem import CleanupResult


class TestDiskCleanup:
    def test_deletes_file_within_allowlist(self, tmp_path):
        f = tmp_path / "junk.log"
        f.write_text("x" * 100)
        r = disk_cleanup(targets=[str(f)], allowlist=[str(tmp_path)], dry_run=False)
        assert isinstance(r, CleanupResult)
        assert r.dry_run is False
        assert r.freed_bytes == 100
        assert not f.exists()

    def test_dry_run_counts_but_keeps(self, tmp_path):
        f = tmp_path / "junk.log"
        f.write_text("y" * 50)
        r = disk_cleanup(targets=[str(f)], allowlist=[str(tmp_path)], dry_run=True)
        assert r.dry_run is True
        assert r.freed_bytes == 50
        assert str(f.resolve()) in r.touched_paths
        assert f.exists()  # 未删

    def test_target_outside_allowlist_rejected_and_nothing_deleted(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        inside = allowed / "keep.log"
        inside.write_text("data")
        outside = tmp_path / "outside.log"
        outside.write_text("secret")
        # inside 合法在前、outside 越界在后:整批应拒绝,inside 也不许被删
        with pytest.raises(OprimValidationError):
            disk_cleanup(
                targets=[str(inside), str(outside)],
                allowlist=[str(allowed)],
                dry_run=False,
            )
        assert inside.exists() and outside.exists()

    def test_dotdot_escape_rejected(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (tmp_path / "escape.log").write_text("nope")
        with pytest.raises(OprimValidationError):
            disk_cleanup(
                targets=[str(allowed / ".." / "escape.log")],
                allowlist=[str(allowed)],
                dry_run=True,
            )

    def test_empty_allowlist_rejects_everything(self, tmp_path):
        f = tmp_path / "x.log"
        f.write_text("z")
        with pytest.raises(OprimValidationError):
            disk_cleanup(targets=[str(f)], allowlist=[], dry_run=True)

    def test_nonexistent_target_skipped(self, tmp_path):
        missing = tmp_path / "gone.log"
        r = disk_cleanup(targets=[str(missing)], allowlist=[str(tmp_path)], dry_run=False)
        assert r.freed_bytes == 0
        assert r.touched_paths == []

    def test_directory_deleted_and_size_summed(self, tmp_path):
        d = tmp_path / "cache"
        d.mkdir()
        (d / "a.bin").write_bytes(b"a" * 30)
        (d / "b.bin").write_bytes(b"b" * 70)
        r = disk_cleanup(targets=[str(d)], allowlist=[str(tmp_path)], dry_run=False)
        assert r.freed_bytes == 100
        assert not d.exists()
