import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path

from oprim._crop_video_to_portrait import CropVideoToPortraitError, crop_video_to_portrait


@pytest.mark.asyncio
async def test_missing_input_raises(tmp_path):
    with pytest.raises(CropVideoToPortraitError):
        await crop_video_to_portrait(input_path=tmp_path / "nope.mp4")


@pytest.mark.asyncio
async def test_no_ffmpeg_raises(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    with patch("shutil.which", return_value=None):
        with pytest.raises(CropVideoToPortraitError):
            await crop_video_to_portrait(input_path=f)


@pytest.mark.asyncio
async def test_ffmpeg_success(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    out = tmp_path / "out.mp4"
    fake_proc = AsyncMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"", b""))
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_proc)):
            res = await crop_video_to_portrait(input_path=f, output_path=out)
            assert res == out
