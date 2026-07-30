"""B2 Filesystem tests — fs_disk_usage alias + fs_inode_check."""

from __future__ import annotations

import pytest

from oprim import fs_disk_usage, fs_inode_check, disk_usage
from oprim._exceptions import OprimNotFoundError, OprimError


def test_fs_disk_usage_is_alias():
    assert fs_disk_usage is disk_usage


def test_fs_disk_usage_returns_disk_usage(tmp_path):
    result = fs_disk_usage(path=str(tmp_path))
    assert result.path == str(tmp_path)
    assert result.total_bytes > 0
    assert result.free_bytes >= 0
    assert 0.0 <= result.used_percent <= 100.0


def test_fs_disk_usage_has_same_signature():
    import inspect

    assert (
        inspect.signature(fs_disk_usage).parameters.keys()
        == inspect.signature(disk_usage).parameters.keys()
    )


def test_fs_inode_check_returns_dict(tmp_path):
    result = fs_inode_check(path=str(tmp_path))
    assert isinstance(result, dict)
    assert "inodes_total" in result
    assert "inodes_used" in result
    assert "inodes_free" in result
    assert "inodes_used_percent" in result
    assert result["path"] == str(tmp_path)


def test_fs_inode_check_nonzero_totals(tmp_path):
    result = fs_inode_check(path=str(tmp_path))
    assert result["inodes_total"] > 0
    assert result["inodes_free"] >= 0


def test_fs_inode_check_percent_in_range(tmp_path):
    result = fs_inode_check(path=str(tmp_path))
    assert 0.0 <= result["inodes_used_percent"] <= 100.0


def test_fs_inode_check_path_not_found():
    with pytest.raises(OprimNotFoundError, match="not found"):
        fs_inode_check(path="/nonexistent/path/xyz/abc")


def test_fs_inode_check_used_plus_free_lte_total(tmp_path):
    result = fs_inode_check(path=str(tmp_path))
    # used + free may not exactly equal total due to reserved inodes
    assert result["inodes_used"] + result["inodes_free"] <= result["inodes_total"] + 1


def test_fs_inode_check_root_path():
    """Root path / should always be checkable on Linux."""
    result = fs_inode_check(path="/")
    assert result["inodes_total"] > 0
