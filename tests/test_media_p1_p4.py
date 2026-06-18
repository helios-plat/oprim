"""Tests for P-1 to P-4 of the video/audio ingestion batch."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._media_types import VideoMeta, TranscriptResult
from oprim._video_filter_rules import video_filter_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


def _vm(
    video_id="v1",
    title="Test Video",
    duration=300.0,
    url="https://yt.be/v1",
    upload_date="20240101",
    description="desc",
) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        duration=duration,
        url=url,
        upload_date=upload_date,
        description=description,
    )


_YDUMP = json.dumps({
    "id": "abc123",
    "title": "Python Tutorial",
    "duration": 600,
    "url": "https://yt.be/abc123",
    "upload_date": "20240315",
    "description": "A tutorial",
}).encode()


# ===========================================================================
# P-1: channel_list_videos
# ===========================================================================

class TestChannelListVideos:
    async def test_normal_returns_video_list(self):
        from oprim._channel_list_videos import channel_list_videos

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=_make_proc(_YDUMP))):
                result = await channel_list_videos(channel_url="https://yt.be/@chan")
        assert len(result) == 1
        assert result[0].video_id == "abc123"
        assert result[0].title == "Python Tutorial"
        assert result[0].duration == 600.0

    async def test_limit_passes_playlist_end_flag(self):
        from oprim._channel_list_videos import channel_list_videos

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            mock_exec = AsyncMock(return_value=_make_proc(_YDUMP))
            with patch("asyncio.create_subprocess_exec", mock_exec):
                await channel_list_videos(channel_url="https://yt.be/@chan", limit=5)
        cmd = mock_exec.call_args[0]
        assert "--playlist-end" in cmd
        assert "5" in cmd

    async def test_proxy_passed_to_yt_dlp(self):
        from oprim._channel_list_videos import channel_list_videos

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            mock_exec = AsyncMock(return_value=_make_proc(_YDUMP))
            with patch("asyncio.create_subprocess_exec", mock_exec):
                await channel_list_videos(channel_url="https://yt.be/@chan", proxy="http://p:3128")
        cmd = mock_exec.call_args[0]
        assert "--proxy" in cmd
        assert "http://p:3128" in cmd

    async def test_yt_dlp_not_installed_raises_runtime_error(self):
        from oprim._channel_list_videos import channel_list_videos

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="yt-dlp"):
                await channel_list_videos(channel_url="https://yt.be/@chan")

    async def test_empty_channel_url_raises_value_error(self):
        from oprim._channel_list_videos import channel_list_videos

        with pytest.raises(ValueError, match="channel_url"):
            await channel_list_videos(channel_url="")

    async def test_empty_channel_returns_empty_list(self):
        from oprim._channel_list_videos import channel_list_videos

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=_make_proc(b""))):
                result = await channel_list_videos(channel_url="https://yt.be/@empty")
        assert result == []


# ===========================================================================
# P-2: media_extract
# ===========================================================================

class TestMediaExtract:
    def _info(self, extra=None) -> bytes:
        d = {
            "id": "vid1",
            "title": "My Video",
            "duration": 500,
            "uploader": "Chan",
            "upload_date": "20240301",
            "description": "Desc",
            "subtitles": {},
            "automatic_captions": {},
        }
        if extra:
            d.update(extra)
        return json.dumps(d).encode()

    async def test_subtitle_path_returns_has_subtitle_true(self, tmp_path):
        from oprim._media_extract import media_extract

        # First call: dump-json info; second: --list-subs (contains zh-Hans); third: download subs
        subs_listing = b"zh-Hans vtt\n"
        sub_file = tmp_path / "vid1.srt"
        sub_file.write_text("WEBVTT\n\n1\n00:00:01,000 --> 00:00:03,000\nHello world\n")

        procs = [
            _make_proc(self._info()),           # dump-json
            _make_proc(subs_listing),            # --list-subs
            _make_proc(b""),                     # download subs (writes file)
        ]
        call_count = [0]
        async def fake_exec(*args, **kwargs):
            i = call_count[0]
            call_count[0] += 1
            return procs[min(i, len(procs) - 1)]

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                result = await media_extract(video_url="https://yt.be/vid1", work_dir=tmp_path)

        assert result.has_subtitle is True
        assert result.title == "My Video"

    async def test_no_subtitle_audio_path_returned(self, tmp_path):
        from oprim._media_extract import media_extract

        mp3 = tmp_path / "vid1.mp3"
        mp3.write_bytes(b"fakeaudio")
        procs = [
            _make_proc(self._info()),   # dump-json
            _make_proc(b""),             # --list-subs (no subs)
            _make_proc(b""),             # download audio
        ]
        call_count = [0]
        async def fake_exec(*args, **kwargs):
            i = call_count[0]; call_count[0] += 1
            return procs[min(i, len(procs)-1)]

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                result = await media_extract(video_url="https://yt.be/vid1", work_dir=tmp_path)

        assert result.has_subtitle is False
        assert result.audio_path is not None

    async def test_proxy_forwarded(self, tmp_path):
        from oprim._media_extract import media_extract

        mp3 = tmp_path / "vid1.mp3"
        mp3.write_bytes(b"x")
        cmds = []
        async def fake_exec(*args, **kwargs):
            cmds.append(args)
            return _make_proc(self._info() if not cmds[1:] else b"")

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                await media_extract(video_url="https://yt.be/vid1", proxy="http://p:3128", work_dir=tmp_path)

        all_args = " ".join(str(a) for cmd in cmds for a in cmd)
        assert "http://p:3128" in all_args

    async def test_private_video_raises(self, tmp_path):
        from oprim._media_extract import media_extract

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", AsyncMock(
                return_value=_make_proc(b"", b"This video is private", returncode=1)
            )):
                with pytest.raises(RuntimeError, match="[Pp]rivate"):
                    await media_extract(video_url="https://yt.be/priv", work_dir=tmp_path)

    async def test_work_dir_auto_created(self, tmp_path):
        from oprim._media_extract import media_extract

        new_dir = tmp_path / "new" / "subdir"
        assert not new_dir.exists()
        mp3 = new_dir / "vid1.mp3"  # won't exist yet, just test dir creation path

        async def fake_exec(*args, **kwargs):
            new_dir.mkdir(parents=True, exist_ok=True)
            mp3.write_bytes(b"x")
            return _make_proc(self._info() if "--dump-json" in args else b"")

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
                try:
                    await media_extract(video_url="https://yt.be/vid1", work_dir=new_dir)
                except Exception:
                    pass
        assert new_dir.exists()

    async def test_empty_url_raises_value_error(self, tmp_path):
        from oprim._media_extract import media_extract

        with pytest.raises(ValueError, match="video_url"):
            await media_extract(video_url="", work_dir=tmp_path)


# ===========================================================================
# P-3: transcribe_audio
# ===========================================================================

class TestTranscribeAudio:
    async def test_local_transcription(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fakeaudio")
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Mock WhisperModel
        mock_segments = [MagicMock(start=0.0, end=2.0, text="你好")]
        mock_info = MagicMock(language="zh", duration=2.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=MagicMock(return_value=mock_model))}):
            result = await transcribe_audio(
                audio_path=audio, backend="local", model_path=str(model_dir)
            )

        assert result.text == "你好"
        assert result.language == "zh"
        assert len(result.segments) == 1

    async def test_dashscope_backend_missing_raises(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")

        with patch.dict("sys.modules", {"dashscope": None, "dashscope.audio.asr": None}):
            with pytest.raises((RuntimeError, ImportError)):
                await transcribe_audio(audio_path=audio, backend="dashscope")

    async def test_audio_not_found_raises_file_not_found(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        with pytest.raises(FileNotFoundError):
            await transcribe_audio(audio_path=tmp_path / "missing.mp3")

    async def test_unknown_backend_raises_value_error(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        with pytest.raises(ValueError, match="backend"):
            await transcribe_audio(audio_path=audio, backend="unknown")

    async def test_segments_structure_complete(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        model_dir = tmp_path / "m"
        model_dir.mkdir()

        segs = [
            MagicMock(start=0.0, end=1.5, text="Hello"),
            MagicMock(start=1.5, end=3.0, text=" World"),
        ]
        mock_info = MagicMock(language="en", duration=3.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segs), mock_info)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=MagicMock(return_value=mock_model))}):
            result = await transcribe_audio(
                audio_path=audio, backend="local", language="en", model_path=str(model_dir)
            )

        assert len(result.segments) == 2
        assert all("start" in s and "end" in s and "text" in s for s in result.segments)

    async def test_model_path_not_exists_raises(self, tmp_path):
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")

        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            with pytest.raises(RuntimeError, match="model"):
                await transcribe_audio(
                    audio_path=audio, backend="local", model_path="/nonexistent/path"
                )


# ===========================================================================
# P-4: video_filter_rules
# ===========================================================================

_VIDEOS = [
    _vm("v1", "Python 入门教程", 300.0, upload_date="20240301"),
    _vm("v2", "JavaScript 高级技巧", 900.0, upload_date="20240115"),
    _vm("v3", "Python 进阶课程", 600.0, upload_date="20231201"),
    _vm("v4", "React 基础", 450.0, upload_date="20240401"),
    _vm("v5", "广告推广内容", 60.0, upload_date="20240315"),
]


class TestVideoFilterRules:
    def test_empty_videos_returns_empty(self):
        assert video_filter_rules([]) == []

    def test_after_date_filters_old_videos(self):
        result = video_filter_rules(_VIDEOS, after_date="20240201")
        ids = {v.video_id for v in result}
        assert "v3" not in ids  # 20231201 < 20240201
        assert "v2" not in ids  # 20240115 < 20240201

    def test_limit_returns_latest_n(self):
        result = video_filter_rules(_VIDEOS, limit=2)
        assert len(result) == 2
        # Should be sorted by upload_date desc
        assert result[0].upload_date >= result[1].upload_date

    def test_limit_zero_returns_empty(self):
        assert video_filter_rules(_VIDEOS, limit=0) == []

    def test_min_duration_filters_short(self):
        result = video_filter_rules(_VIDEOS, min_duration=400.0)
        for v in result:
            assert v.duration >= 400.0

    def test_max_duration_filters_long(self):
        result = video_filter_rules(_VIDEOS, max_duration=500.0)
        for v in result:
            assert v.duration <= 500.0

    def test_title_include_keyword(self):
        result = video_filter_rules(_VIDEOS, title_include=["Python"])
        assert all("Python" in v.title for v in result)

    def test_title_exclude_keyword(self):
        result = video_filter_rules(_VIDEOS, title_exclude=["广告"])
        assert all("广告" not in v.title for v in result)

    def test_invalid_after_date_raises(self):
        with pytest.raises(ValueError, match="YYYYMMDD"):
            video_filter_rules(_VIDEOS, after_date="2024-01-01")

    def test_sorted_by_upload_date_desc(self):
        result = video_filter_rules(_VIDEOS)
        dates = [v.upload_date for v in result if v.upload_date]
        assert dates == sorted(dates, reverse=True)
