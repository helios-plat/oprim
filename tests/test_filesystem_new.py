import os
import tarfile
from pathlib import Path
from unittest.mock import patch
import pytest
from oprim import archive_to_targz, dir_archive_to_targz
from oprim._exceptions import OprimNotFoundError, OprimError

def test_archive_to_targz_single_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("hello")
    (src / "subdir").mkdir()
    (src / "subdir" / "file2.txt").write_text("world")

    dst = tmp_path / "archive.tar.gz"
    res = archive_to_targz(sources=[str(src)], dst_path=str(dst))

    assert res.file_count == 2
    assert Path(dst).exists()
    assert res.sources == [str(src)]

    # Verify contents
    with tarfile.open(dst, "r:gz") as tar:
        names = tar.getnames()
        assert "file1.txt" in names
        assert "subdir/file2.txt" in names

def test_archive_to_targz_multi(tmp_path):
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "f1.txt").write_text("1")
    
    f2 = tmp_path / "f2.txt"
    f2.write_text("2")

    dst = tmp_path / "multi.tar.gz"
    res = archive_to_targz(sources=[str(d1), str(f2)], dst_path=str(dst))

    assert res.file_count == 2
    with tarfile.open(dst, "r:gz") as tar:
        names = tar.getnames()
        assert "f1.txt" in names
        assert "f2.txt" in names

def test_archive_to_targz_exclude(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("keep")
    (src / "skip.tmp").write_text("skip")

    dst = tmp_path / "excluded.tar.gz"
    res = archive_to_targz(sources=[str(src)], dst_path=str(dst), exclude_patterns=["*.tmp"])

    assert res.file_count == 1
    with tarfile.open(dst, "r:gz") as tar:
        assert "keep.txt" in tar.getnames()
        assert "skip.tmp" not in tar.getnames()

def test_archive_to_targz_not_found():
    with pytest.raises(OprimNotFoundError):
        archive_to_targz(sources=["/nonexistent"], dst_path="out.tar.gz")

def test_archive_to_targz_empty_sources():
    with pytest.raises(OprimError, match="No sources provided"):
        archive_to_targz(sources=[], dst_path="out.tar.gz")

def test_archive_to_targz_file_error(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = src / "f.txt"
    f.write_text("ok")
    
    dst = tmp_path / "out.tar.gz"
    
    # Mock open to fail during archiving
    with patch("builtins.open", side_effect=OSError("denied")):
        with pytest.raises(OprimError, match="Failed to create archive"):
            archive_to_targz(sources=[str(src)], dst_path=str(dst))

def test_archive_to_targz_checksum_error(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("ok")
    dst = tmp_path / "out.tar.gz"
    
    # First create it normally so it exists
    archive_to_targz(sources=[str(src)], dst_path=str(dst))
    
    # Now mock open to fail only during the second open (for checksum)
    original_open = open
    def side_effect(path, mode="r", *args, **kwargs):
        if str(path) == str(dst) and mode == "rb":
            raise OSError("read error")
        return original_open(path, mode, *args, **kwargs)
    
    with patch("builtins.open", side_effect=side_effect):
        with pytest.raises(OprimError, match="Failed to read archive for checksum"):
            archive_to_targz(sources=[str(src)], dst_path=str(dst))

def test_dir_archive_to_targz_legacy(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("content")
    dst = tmp_path / "legacy.tar.gz"

    with pytest.warns(DeprecationWarning, match="deprecated"):
        res = dir_archive_to_targz(src_dir=str(src), dst_path=str(dst))

    assert res.file_count == 1
    assert res.src_dir == str(src) # Test property compatibility
