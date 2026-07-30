"""Tests for oprim.media_probe and oprim.video_thumbnail."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from oprim import media_probe, video_thumbnail
from oprim._exceptions import OprimError, OprimNotFoundError, OprimValidationError
from oprim._media_probe import MediaInfo

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

_FAKE_PROBE = {
    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "3.5", "size": "12345"},
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480},
        {"codec_type": "audio", "codec_name": "aac"},
    ],
}


class TestMediaProbeParse:
    def test_injected_json(self) -> None:
        r = media_probe(ffprobe_json=json.dumps(_FAKE_PROBE))
        assert isinstance(r, MediaInfo)
        assert r.is_video is True and r.is_audio is True
        assert r.width == 640 and r.height == 480
        assert r.duration_seconds == 3.5
        assert r.size_bytes == 12345
        assert len(r.streams) == 2

    def test_requires_path_or_json(self) -> None:
        with pytest.raises(OprimNotFoundError):
            media_probe()

    def test_bad_json(self) -> None:
        with pytest.raises(OprimError, match="not valid JSON"):
            media_probe(ffprobe_json="{bad")

    def test_audio_only(self) -> None:
        r = media_probe(
            ffprobe_json=json.dumps(
                {
                    "format": {"duration": "10"},
                    "streams": [{"codec_type": "audio", "codec_name": "mp3"}],
                }
            )
        )
        assert r.is_audio is True and r.is_video is False and r.width is None


class TestVideoThumbnailValidation:
    def test_empty_path(self) -> None:
        with pytest.raises(OprimValidationError, match="path"):
            video_thumbnail(path="")

    def test_bad_max_size(self) -> None:
        with pytest.raises(OprimValidationError, match="max_size"):
            video_thumbnail(path="/x.mp4", max_size=0)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestReal:
    @pytest.fixture
    def sample_video(self, tmp_path: Path) -> str:
        out = tmp_path / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "quiet",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=640x480:rate=5",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=True,
            timeout=30,
        )
        return str(out)

    def test_probe_real_video(self, sample_video: str) -> None:
        r = media_probe(path=sample_video)
        assert r.is_video is True
        assert r.width == 640 and r.height == 480
        assert r.duration_seconds is not None and r.duration_seconds > 1.5

    def test_thumbnail_real_video_jpeg(self, sample_video: str) -> None:
        r = video_thumbnail(path=sample_video, max_size=160, fmt="jpeg")
        assert r.format == "jpeg"
        assert r.data[:2] == b"\xff\xd8"  # JPEG SOI marker
        # Longest side should be capped at ~160.
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(r.data))
        assert max(img.size) <= 160

    def test_thumbnail_missing_file_raises(self) -> None:
        with pytest.raises(OprimError):
            video_thumbnail(path="/nonexistent/x.mp4")
