"""Tests — H-B A组: file IO extensions (ensure_parent_dir / file_read_bytes / image_to_base64 / atomic_write / backup_before_overwrite)."""
from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import pytest

from oprim._hb_file_io import (
    atomic_write,
    backup_before_overwrite,
    ensure_parent_dir,
    file_read_bytes,
    image_to_base64,
)


# ---------------------------------------------------------------------------
# ensure_parent_dir
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_parent_dir_creates_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "file.txt"
    await ensure_parent_dir(target)
    assert target.parent.is_dir()


@pytest.mark.asyncio
async def test_ensure_parent_dir_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "x" / "y" / "z.txt"
    await ensure_parent_dir(target)
    await ensure_parent_dir(target)  # second call — no error
    assert target.parent.is_dir()


@pytest.mark.asyncio
async def test_ensure_parent_dir_existing(tmp_path: Path) -> None:
    # Parent already exists — no error
    await ensure_parent_dir(tmp_path / "file.txt")
    assert tmp_path.is_dir()


@pytest.mark.asyncio
async def test_ensure_parent_dir_parent_is_file(tmp_path: Path) -> None:
    (tmp_path / "blocker").write_text("x")
    with pytest.raises((NotADirectoryError, OSError)):
        await ensure_parent_dir(tmp_path / "blocker" / "child.txt")


@pytest.mark.asyncio
async def test_ensure_parent_dir_root(tmp_path: Path) -> None:
    # root dir itself — no-op
    file_in_root = tmp_path / "file.txt"
    await ensure_parent_dir(file_in_root)  # parent = tmp_path (already exists)
    assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# file_read_bytes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_bytes_full(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\x03")
    result = await file_read_bytes(f)
    assert result == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_file_read_bytes_range(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"ABCDEFGH")
    result = await file_read_bytes(f, offset=2, length=3)
    assert result == b"CDE"


@pytest.mark.asyncio
async def test_file_read_bytes_offset_beyond_size(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"AB")
    result = await file_read_bytes(f, offset=100)
    assert result == b""


@pytest.mark.asyncio
async def test_file_read_bytes_negative_offset(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"AB")
    with pytest.raises(ValueError, match="offset"):
        await file_read_bytes(f, offset=-1)


@pytest.mark.asyncio
async def test_file_read_bytes_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await file_read_bytes(tmp_path / "missing.bin")


@pytest.mark.asyncio
async def test_file_read_bytes_binary_roundtrip(tmp_path: Path) -> None:
    binary = bytes(range(256))
    f = tmp_path / "all_bytes.bin"
    f.write_bytes(binary)
    result = await file_read_bytes(f)
    assert result == binary


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------

# Minimal valid PNG (1x1 pixel)
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.mark.asyncio
async def test_image_to_base64_png(tmp_path: Path) -> None:
    img = tmp_path / "test.png"
    img.write_bytes(_PNG_1x1)
    b64 = await image_to_base64(img)
    assert isinstance(b64, str)
    decoded = base64.b64decode(b64)
    assert decoded == _PNG_1x1


@pytest.mark.asyncio
async def test_image_to_base64_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await image_to_base64(tmp_path / "missing.png")


@pytest.mark.asyncio
async def test_image_to_base64_roundtrip(tmp_path: Path) -> None:
    data = bytes(range(200))
    f = tmp_path / "fake.jpg"
    f.write_bytes(data)
    b64 = await image_to_base64(f)
    assert base64.b64decode(b64) == data


@pytest.mark.asyncio
async def test_image_to_base64_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.png"
    f.write_bytes(b"")
    b64 = await image_to_base64(f)
    assert b64 == ""


@pytest.mark.asyncio
async def test_image_to_base64_large_file_warning(tmp_path: Path) -> None:
    import warnings
    f = tmp_path / "big.png"
    f.write_bytes(b"X" * (21 * 1024 * 1024))  # 21 MB
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await image_to_base64(f)
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_atomic_write_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    await atomic_write(p, content="hello world")
    assert p.read_text() == "hello world"


@pytest.mark.asyncio
async def test_atomic_write_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    p.write_text("old")
    await atomic_write(p, content="new")
    assert p.read_text() == "new"


@pytest.mark.asyncio
async def test_atomic_write_creates_parent(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "out.txt"
    await atomic_write(p, content="data")
    assert p.read_text() == "data"


@pytest.mark.asyncio
async def test_atomic_write_no_temp_leftover(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    await atomic_write(p, content="hello")
    leftovers = list(tmp_path.glob(".atomic_*"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_atomic_write_unicode(tmp_path: Path) -> None:
    p = tmp_path / "unicode.txt"
    content = "你好世界\nHello World\n"
    await atomic_write(p, content=content)
    assert p.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# backup_before_overwrite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backup_returns_path(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("original")
    bak = await backup_before_overwrite(p)
    assert bak is not None
    assert bak.exists()
    assert bak.read_text() == "original"


@pytest.mark.asyncio
async def test_backup_not_exist_returns_none(tmp_path: Path) -> None:
    result = await backup_before_overwrite(tmp_path / "missing.txt")
    assert result is None


@pytest.mark.asyncio
async def test_backup_collision_sequence(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("v1")
    bak1 = await backup_before_overwrite(p)
    p.write_text("v2")
    bak2 = await backup_before_overwrite(p)
    assert bak1 != bak2
    assert bak1 is not None and bak1.exists()
    assert bak2 is not None and bak2.exists()


@pytest.mark.asyncio
async def test_backup_large_file(tmp_path: Path) -> None:
    p = tmp_path / "large.bin"
    p.write_bytes(b"X" * (1024 * 1024))  # 1 MB
    bak = await backup_before_overwrite(p)
    assert bak is not None
    assert bak.stat().st_size == 1024 * 1024


@pytest.mark.asyncio
async def test_backup_returns_correct_path(tmp_path: Path) -> None:
    p = tmp_path / "myfile.txt"
    p.write_text("data")
    bak = await backup_before_overwrite(p)
    assert bak is not None
    # bak should be under the same directory
    assert bak.parent == p.parent
