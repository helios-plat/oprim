"""P0 media canonicalization — tests for the canonical media OPrim set.

Covers SPEC §9 Test Contract for every new/收敛 element:
happy path, invalid input, provider failure, missing artifact, timeout, and
output contract — plus the single-source invariants from §8.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim._audio_mix import AudioMixError, audio_mix, mix_audio_tracks
from oprim._encode_voice_reference import EncodeVoiceReferenceError, encode_voice_reference
from oprim._exceptions import OprimError, OprimNotFoundError
from oprim._extract_audio_waveform import (
    ExtractAudioWaveformError,
    _downsample,
    _parse_volume,
    extract_audio_waveform,
)
from oprim._extract_media_segment import (
    ExtractMediaSegmentError,
    extract_media_segment,
)
from oprim._extract_video_frames import ExtractVideoFramesError, extract_video_frames
from oprim._media_probe import MediaInfo, media_probe, probe_media
from oprim._probe_duration import ProbeDurationError, probe_duration
from oprim._render_media import RenderMediaError, render_media
from oprim._segment_media import SegmentMediaError, segment_media
from oprim._subtitle_burn import SubtitleBurnError, burn_subtitles, subtitle_burn
from oprim._transcribe_audio import transcribe_audio, transcribe_media

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

_RICH_PROBE = {
    "format": {
        "format_name": "mov,mp4,m4a",
        "duration": "3.5",
        "size": "12345",
        "bit_rate": "1500000",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "bit_rate": "1400000",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "128000",
        },
    ],
}


def _video(tmp_path: Path, name: str = "in.mp4") -> Path:
    p = tmp_path / name
    p.write_bytes(b"not-really-a-video")
    return p


# ---------------------------------------------------------------------------
# §8.1 / §4.1 — probe_media is the single canonical inspection element
# ---------------------------------------------------------------------------


class TestProbeMediaCanonical:
    def test_parses_full_field_set(self) -> None:
        info = media_probe(ffprobe_json=json.dumps(_RICH_PROBE))
        assert isinstance(info, MediaInfo)
        assert info.duration_seconds == 3.5
        assert info.size_bytes == 12345
        assert info.bitrate == 1500000
        assert info.fps == pytest.approx(30000 / 1001, rel=1e-6)
        assert info.width == 1920 and info.height == 1080
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.audio_sample_rate == 48000
        assert info.audio_channels == 2

    def test_rational_and_garbage_fps_are_tolerated(self) -> None:
        payload = json.loads(json.dumps(_RICH_PROBE))
        payload["streams"][0]["avg_frame_rate"] = "0/0"
        payload["streams"][0]["r_frame_rate"] = "N/A"
        assert media_probe(ffprobe_json=json.dumps(payload)).fps is None

    def test_falls_back_to_r_frame_rate(self) -> None:
        payload = json.loads(json.dumps(_RICH_PROBE))
        payload["streams"][0]["avg_frame_rate"] = "0/0"
        payload["streams"][0]["r_frame_rate"] = "25/1"
        assert media_probe(ffprobe_json=json.dumps(payload)).fps == 25.0

    def test_duration_falls_back_to_stream_duration(self) -> None:
        payload = {"format": {}, "streams": [{"codec_type": "video", "duration": "7.25"}]}
        assert media_probe(ffprobe_json=json.dumps(payload)).duration_seconds == 7.25

    def test_probe_media_is_same_implementation(self) -> None:
        payload = json.dumps(_RICH_PROBE)
        assert probe_media(ffprobe_json=payload) == media_probe(ffprobe_json=payload)

    def test_probe_media_shares_signature(self) -> None:
        assert inspect.signature(probe_media).parameters.keys() == inspect.signature(
            media_probe
        ).parameters.keys()

    def test_requires_path_or_json(self) -> None:
        with pytest.raises(OprimNotFoundError):
            probe_media()

    def test_bad_json(self) -> None:
        with pytest.raises(OprimError, match="not valid JSON"):
            probe_media(ffprobe_json="{bad")

    def test_non_object_json_rejected(self) -> None:
        with pytest.raises(OprimError, match="not a JSON object"):
            probe_media(ffprobe_json="[1, 2, 3]")

    def test_ffprobe_missing_raises(self, tmp_path: Path) -> None:
        with (
            patch("oprim._ffprobe.shutil.which", return_value=None),
            pytest.raises(OprimError, match="not found on PATH"),
        ):
            probe_media(path=_video(tmp_path))

    def test_ffprobe_timeout_raises(self, tmp_path: Path) -> None:
        with (
            patch(
                "oprim._ffprobe.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffprobe", 30),
            ),
            pytest.raises(OprimError, match="timed out"),
        ):
            probe_media(path=_video(tmp_path))

    def test_empty_output_raises(self, tmp_path: Path) -> None:
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="boom")
        with (
            patch("oprim._ffprobe.subprocess.run", return_value=proc),
            pytest.raises(OprimError, match="produced no output"),
        ):
            probe_media(path=_video(tmp_path))

    def test_probe_duration_wraps_canonical_executor(self, tmp_path: Path) -> None:
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="1.5\n", stderr="")
        with patch("oprim._ffprobe.subprocess.run", return_value=proc) as run:
            assert probe_duration(_video(tmp_path)) == pytest.approx(1.5)
        assert run.call_args.args[0][0] == "ffprobe"

    def test_probe_duration_maps_errors(self, tmp_path: Path) -> None:
        with (
            patch("oprim._ffprobe.subprocess.run", side_effect=OSError("no ffprobe")),
            pytest.raises(ProbeDurationError, match="ffprobe failed"),
        ):
            probe_duration(_video(tmp_path))

    def test_probe_duration_rejects_unparsable(self, tmp_path: Path) -> None:
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="N/A", stderr="")
        with (
            patch("oprim._ffprobe.subprocess.run", return_value=proc),
            pytest.raises(ProbeDurationError, match="unparsable"),
        ):
            probe_duration(_video(tmp_path))


# ---------------------------------------------------------------------------
# §4.3 — extract_video_frames
# ---------------------------------------------------------------------------


class TestExtractVideoFrames:
    async def test_missing_video(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="not found"):
            await extract_video_frames(video_path=tmp_path / "nope.mp4", output_dir=tmp_path)

    async def test_unknown_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="Unknown mode"):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path, mode="bogus"  # type: ignore[arg-type]
            )

    async def test_mode_requires_its_selector(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="requires its selector"):
            await extract_video_frames(video_path=_video(tmp_path), output_dir=tmp_path)

    async def test_two_selectors_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="Exactly one selector"):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path, fps=2, count=3
            )

    async def test_fps_mode_writes_sequence(self, tmp_path: Path) -> None:
        out = tmp_path / "frames"

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            pattern = args[-1]
            base = Path(pattern).parent
            for i in range(1, 4):
                (base / Path(pattern).name.replace("%06d", f"{i:06d}")).write_bytes(b"jpg")

        with patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_video_frames(
                video_path=_video(tmp_path), output_dir=out, mode="fps", fps=2.0
            )
        assert len(result.frames) == 3
        assert result.frames[0].timestamp == 0.0
        assert result.frames[1].timestamp == pytest.approx(0.5)
        assert all(f.path.exists() for f in result.frames)

    async def test_fps_mode_rejects_non_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="fps must be"):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path / "f", mode="fps", fps=0
            )

    async def test_timestamps_mode_uses_seek_args(self, tmp_path: Path) -> None:
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"jpg")

        with patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="timestamps",
                timestamps=[0.0, 1.25],
            )
        assert [f.timestamp for f in result.frames] == [0.0, 1.25]
        assert all("-ss" in a for a in seen)
        assert seen[1][seen[1].index("-ss") + 1] == "1.250000"

    async def test_negative_timestamp_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match=">= 0"):
            await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="timestamps",
                timestamps=[-1.0],
            )

    async def test_empty_timestamps_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="must not be empty"):
            await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="timestamps",
                timestamps=[],
            )

    async def test_count_requires_duration(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractVideoFramesError, match="requires a positive duration_s"):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path / "f", mode="count", count=4
            )

    async def test_count_spreads_across_duration(self, tmp_path: Path) -> None:
        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            Path(args[-1]).write_bytes(b"jpg")

        with patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="count",
                count=4,
                duration_s=8.0,
            )
        assert [f.timestamp for f in result.frames] == [1.0, 3.0, 5.0, 7.0]

    async def test_scene_boundaries_are_hints_not_detection(self, tmp_path: Path) -> None:
        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            Path(args[-1]).write_bytes(b"jpg")

        with patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="scene_boundaries",
                scene_boundaries=[0.5, 2.0],
            )
        assert [f.timestamp for f in result.frames] == [0.5, 2.0]

    async def test_ffmpeg_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        with (
            patch(
                "oprim._extract_video_frames.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(ExtractVideoFramesError, match="FFmpeg frame extraction failed"),
        ):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path / "f", mode="fps", fps=1
            )

    async def test_missing_artifact_detected(self, tmp_path: Path) -> None:
        with (
            patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(return_value="")),
            pytest.raises(ExtractVideoFramesError, match="No frames written"),
        ):
            await extract_video_frames(
                video_path=_video(tmp_path), output_dir=tmp_path / "f", mode="fps", fps=1
            )

    async def test_output_contract(self, tmp_path: Path) -> None:
        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            base = Path(args[-1]).parent
            (base / "in_f000001.jpg").write_bytes(b"jpg")

        with patch("oprim._extract_video_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_video_frames(
                video_path=_video(tmp_path),
                output_dir=tmp_path / "f",
                mode="fps",
                fps=1,
                width=640,
                height=480,
            )
        dumped = result.model_dump()
        assert set(dumped) == {"source", "width", "height", "frames"}
        assert set(dumped["frames"][0]) == {"index", "timestamp", "path"}


# ---------------------------------------------------------------------------
# §4.6 — extract_media_segment
# ---------------------------------------------------------------------------


class TestExtractMediaSegment:
    async def test_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractMediaSegmentError, match="not found"):
            await extract_media_segment(
                media_path=tmp_path / "no.mp4", start=0, end=1, output_path=tmp_path / "o.mp4"
            )

    async def test_negative_start(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractMediaSegmentError, match="start must be"):
            await extract_media_segment(
                media_path=_video(tmp_path), start=-1, end=1, output_path=tmp_path / "o.mp4"
            )

    async def test_end_before_start(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractMediaSegmentError, match="end must be > start"):
            await extract_media_segment(
                media_path=_video(tmp_path), start=2, end=2, output_path=tmp_path / "o.mp4"
            )

    async def test_stream_copy_path(self, tmp_path: Path) -> None:
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"mp4")

        out = tmp_path / "nested" / "seg.mp4"
        with patch("oprim._extract_media_segment.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            result = await extract_media_segment(
                media_path=_video(tmp_path), start=1.0, end=3.5, output_path=out
            )
        assert result == out and out.exists()
        assert seen[0][seen[0].index("-ss") + 1] == "1.000000"
        assert seen[0][seen[0].index("-t") + 1] == "2.500000"
        assert "-c" in seen[0] and "copy" in seen[0]

    async def test_reencode_path(self, tmp_path: Path) -> None:
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"mp4")

        with patch("oprim._extract_media_segment.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            await extract_media_segment(
                media_path=_video(tmp_path),
                start=0,
                end=1,
                output_path=tmp_path / "o.mp4",
                reencode=True,
            )
        assert "libx264" in seen[0] and "yuv420p" in seen[0]

    async def test_ffmpeg_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        with (
            patch(
                "oprim._extract_media_segment.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(ExtractMediaSegmentError, match="segment extraction failed"),
        ):
            await extract_media_segment(
                media_path=_video(tmp_path), start=0, end=1, output_path=tmp_path / "o.mp4"
            )


# ---------------------------------------------------------------------------
# §4.5 — segment_media
# ---------------------------------------------------------------------------


class TestSegmentMedia:
    async def test_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentMediaError, match="not found"):
            await segment_media(media_path=tmp_path / "no.mp4")

    async def test_threshold_validated(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentMediaError, match="threshold"):
            await segment_media(media_path=_video(tmp_path), threshold=0)
        with pytest.raises(SegmentMediaError, match="threshold"):
            await segment_media(media_path=_video(tmp_path), threshold=101)

    async def test_min_duration_validated(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentMediaError, match="min_duration"):
            await segment_media(media_path=_video(tmp_path), min_duration=0)

    async def test_hints_requires_hints(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentMediaError, match="requires boundary_hints"):
            await segment_media(media_path=_video(tmp_path), method="hints")

    async def test_hints_build_contiguous_spans(self, tmp_path: Path) -> None:
        result = await segment_media(
            media_path=_video(tmp_path),
            method="hints",
            boundary_hints=[2.0, 1.0, 2.0],
            total_duration=5.0,
        )
        assert [(s.start, s.end) for s in result.segments] == [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]
        assert [s.index for s in result.segments] == [0, 1, 2]
        assert result.segments[0].duration == 1.0

    async def test_min_duration_drops_slivers(self, tmp_path: Path) -> None:
        result = await segment_media(
            media_path=_video(tmp_path),
            method="hints",
            boundary_hints=[0.1, 4.0],
            min_duration=1.0,
            total_duration=5.0,
        )
        assert [(s.start, s.end) for s in result.segments] == [(0.1, 4.0), (4.0, 5.0)]

    async def test_scene_mode_parses_scdet(self, tmp_path: Path) -> None:
        stderr = (
            "[scdet @ 0x1] lavfi.scd.score: 0.000, lavfi.scd.time: 0\n"
            "[scdet @ 0x1] lavfi.scd.score: 42.500, lavfi.scd.time: 2.5\n"
            "[scdet @ 0x1] lavfi.scd.score: 61.250, lavfi.scd.time: 4.25\n"
        )
        with patch(
            "oprim._segment_media.ffmpeg_run", new=AsyncMock(return_value=stderr)
        ):
            result = await segment_media(
                media_path=_video(tmp_path), method="scene", total_duration=6.0
            )
        assert [(s.start, s.end) for s in result.segments] == [
            (0.0, 2.5),
            (2.5, 4.25),
            (4.25, 6.0),
        ]

    async def test_scene_mode_forwards_native_threshold_scale(self, tmp_path: Path) -> None:
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float):  # noqa: ANN001
            seen.append(args)
            return ""

        with patch("oprim._segment_media.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            await segment_media(media_path=_video(tmp_path), method="scene", threshold=25.0)
        assert "scdet=threshold=25.0" in seen[0]

    async def test_silence_mode_parses_silencedetect(self, tmp_path: Path) -> None:
        stderr = (
            "[silencedetect] silence_start: 1.5\n"
            "[silencedetect] silence_end: 3.25 | silence_duration: 1.75\n"
        )
        with patch(
            "oprim._segment_media.ffmpeg_run", new=AsyncMock(return_value=stderr)
        ):
            result = await segment_media(
                media_path=_video(tmp_path), method="silence", total_duration=5.0
            )
        assert [(s.start, s.end) for s in result.segments] == [(0.0, 3.25), (3.25, 5.0)]

    async def test_silence_falls_back_to_start_negation(self, tmp_path: Path) -> None:
        stderr = "[silencedetect] silence_start: -2.0\n"
        with patch(
            "oprim._segment_media.ffmpeg_run", new=AsyncMock(return_value=stderr)
        ):
            result = await segment_media(
                media_path=_video(tmp_path), method="silence", total_duration=4.0
            )
        assert [(s.start, s.end) for s in result.segments] == [(0.0, 2.0), (2.0, 4.0)]

    async def test_unknown_method(self, tmp_path: Path) -> None:
        with pytest.raises(SegmentMediaError, match="Unknown method"):
            await segment_media(media_path=_video(tmp_path), method="bogus")  # type: ignore[arg-type]

    async def test_ffmpeg_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        with (
            patch(
                "oprim._segment_media.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(SegmentMediaError, match="segmentation failed"),
        ):
            await segment_media(media_path=_video(tmp_path), method="scene")

    async def test_output_contract(self, tmp_path: Path) -> None:
        result = await segment_media(
            media_path=_video(tmp_path), method="hints", boundary_hints=[1.0], total_duration=2.0
        )
        assert set(result.model_dump()) == {"source", "method", "total_duration", "segments"}


# ---------------------------------------------------------------------------
# §4.4 — extract_audio_waveform
# ---------------------------------------------------------------------------


class TestExtractAudioWaveform:
    def test_parse_volume_reads_last_reporting(self) -> None:
        stderr = (
            "mean_volume: -21.0dB\n[Parsed_volumedetect_0] max_volume: -3.0dB\n"
            "mean_volume: -18.5dB\n[Parsed_volumedetect_0] max_volume: -1.2dB\n"
        )
        assert _parse_volume(stderr) == (-18.5, -1.2)

    def test_parse_volume_absent(self) -> None:
        assert _parse_volume("nothing here") == (None, None)

    def test_downsample_normalizes_to_unit_peak(self) -> None:
        pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in (-1000, 2000, -3000))
        samples = _downsample(pcm, target_samples=10)
        assert max(abs(s) for s in samples) == pytest.approx(1.0)
        assert samples[0] == pytest.approx(-1000 / 3000, rel=1e-3)

    def test_downsample_buckets_peaks(self) -> None:
        pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in (0, 100, 0, 50))
        assert _downsample(pcm, target_samples=2) == [pytest.approx(100 / 100), pytest.approx(0.5)]

    def test_downsample_handles_silence(self) -> None:
        assert _downsample(b"\x00\x00" * 4, target_samples=10) == [0.0] * 4

    def test_downsample_empty(self) -> None:
        assert _downsample(b"", target_samples=10) == []

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractAudioWaveformError, match="not found"):
            extract_audio_waveform(audio_path=tmp_path / "no.wav")

    def test_invalid_sample_rate(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractAudioWaveformError, match="sample_rate"):
            extract_audio_waveform(audio_path=_video(tmp_path), sample_rate=0)

    def test_invalid_target_samples(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractAudioWaveformError, match="target_samples"):
            extract_audio_waveform(audio_path=_video(tmp_path), target_samples=0)

    def test_happy_path_mocked(self, tmp_path: Path) -> None:
        pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in range(-500, 500))
        calls: list[list[str]] = []

        def _fake_run(cmd, **kw):  # noqa: ANN001, ANN202
            calls.append(list(cmd))
            if "s16le" in cmd:
                return subprocess.CompletedProcess(cmd, 0, pcm, "")
            return subprocess.CompletedProcess(
                cmd, 0, "", "mean_volume: -20.0dB\nmax_volume: -1.0dB\n"
            )

        with patch("oprim._extract_audio_waveform.subprocess.run", side_effect=_fake_run):
            wf = extract_audio_waveform(
                audio_path=_video(tmp_path), sample_rate=8000, target_samples=100
            )
        assert wf.sample_rate == 8000
        assert wf.duration == pytest.approx(1000 / 8000, rel=1e-6)
        assert wf.rms_db == -20.0 and wf.peak_db == -1.0
        assert len(wf.samples) == 100
        assert calls[0][0] == "ffmpeg"

    def test_timeout_wrapped(self, tmp_path: Path) -> None:
        with (
            patch(
                "oprim._extract_audio_waveform.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 1),
            ),
            pytest.raises(ExtractAudioWaveformError, match="timed out"),
        ):
            extract_audio_waveform(audio_path=_video(tmp_path))

    def test_missing_binary(self, tmp_path: Path) -> None:
        with (
            patch("oprim._extract_audio_waveform.shutil.which", return_value=None),
            pytest.raises(ExtractAudioWaveformError, match="not found on PATH"),
        ):
            extract_audio_waveform(audio_path=_video(tmp_path))

    def test_output_contract(self, tmp_path: Path) -> None:
        pcm = b"\x00\x01" * 8
        with patch(
            "oprim._extract_audio_waveform.subprocess.run",
            side_effect=lambda cmd, **kw: subprocess.CompletedProcess(  # noqa: ANN001
                cmd, 0, pcm if "s16le" in cmd else "", ""
            ),
        ):
            wf = extract_audio_waveform(audio_path=_video(tmp_path), target_samples=4)
        assert set(wf.model_dump()) == {
            "source",
            "sample_rate",
            "duration",
            "peak_db",
            "rms_db",
            "samples",
        }


# ---------------------------------------------------------------------------
# §4.11 — encode_voice_reference
# ---------------------------------------------------------------------------


class TestEncodeVoiceReference:
    async def test_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(EncodeVoiceReferenceError, match="not found"):
            await encode_voice_reference(
                audio_path=tmp_path / "no.wav", output_path=tmp_path / "ref.wav"
            )

    async def test_invalid_sample_rate(self, tmp_path: Path) -> None:
        with pytest.raises(EncodeVoiceReferenceError, match="sample_rate"):
            await encode_voice_reference(
                audio_path=_video(tmp_path), output_path=tmp_path / "r.wav", sample_rate=0
            )

    async def test_invalid_max_duration(self, tmp_path: Path) -> None:
        with pytest.raises(EncodeVoiceReferenceError, match="max_duration_s"):
            await encode_voice_reference(
                audio_path=_video(tmp_path), output_path=tmp_path / "r.wav", max_duration_s=0
            )

    async def test_happy_path_is_content_addressed(self, tmp_path: Path) -> None:
        src = tmp_path / "ref.wav"
        src.write_bytes(b"audio-bytes")
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"wav")

        out = tmp_path / "nested" / "voice.wav"
        with patch(
            "oprim._encode_voice_reference.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)
        ):
            ref = await encode_voice_reference(audio_path=src, output_path=out)

        digest = hashlib.sha256(b"audio-bytes").hexdigest()
        assert ref.reference_id == f"vref_{digest[:24]}"
        assert ref.sha256 == digest
        assert ref.sample_rate == 24000 and ref.channels == 1
        assert out.exists()
        joined = " ".join(seen[0])
        assert "-vn" in seen[0] and "loudnorm=I=-20.0" in joined

    async def test_same_input_yields_same_id(self, tmp_path: Path) -> None:
        src = tmp_path / "ref.wav"
        src.write_bytes(b"same")

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            Path(args[-1]).write_bytes(b"wav")

        with patch(
            "oprim._encode_voice_reference.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)
        ):
            a = await encode_voice_reference(audio_path=src, output_path=tmp_path / "a.wav")
            b = await encode_voice_reference(audio_path=src, output_path=tmp_path / "b.wav")
        assert a.reference_id == b.reference_id

    async def test_ffmpeg_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        src = tmp_path / "ref.wav"
        src.write_bytes(b"x")
        with (
            patch(
                "oprim._encode_voice_reference.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(EncodeVoiceReferenceError, match="voice reference encode failed"),
        ):
            await encode_voice_reference(audio_path=src, output_path=tmp_path / "r.wav")

    async def test_missing_artifact_detected(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        src = tmp_path / "ref.wav"
        src.write_bytes(b"x")
        with (
            patch(
                "oprim._encode_voice_reference.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("Expected output file not found")),
            ),
            pytest.raises(EncodeVoiceReferenceError),
        ):
            await encode_voice_reference(audio_path=src, output_path=tmp_path / "r.wav")


# ---------------------------------------------------------------------------
# §4.7 — render_media
# ---------------------------------------------------------------------------


class TestRenderMedia:
    async def test_empty_pattern(self, tmp_path: Path) -> None:
        with pytest.raises(RenderMediaError, match="frames_pattern is required"):
            await render_media(frames_pattern="", output_path=tmp_path / "o.mp4")

    async def test_invalid_fps(self, tmp_path: Path) -> None:
        with pytest.raises(RenderMediaError, match="fps must be"):
            await render_media(
                frames_pattern=str(tmp_path / "f_%06d.png"), output_path=tmp_path / "o.mp4", fps=0
            )

    async def test_invalid_dimensions(self, tmp_path: Path) -> None:
        with pytest.raises(RenderMediaError, match="width must be"):
            await render_media(
                frames_pattern=str(tmp_path / "f_%06d.png"),
                output_path=tmp_path / "o.mp4",
                width=0,
            )

    async def test_incomplete_sequence_detected(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "frame_000001.png").write_bytes(b"p")
        with pytest.raises(RenderMediaError, match="Incomplete frame sequence"):
            await render_media(
                frames_pattern=str(frames / "frame_%06d.png"),
                output_path=tmp_path / "o.mp4",
                frame_count=5,
            )

    async def test_no_frames_detected(self, tmp_path: Path) -> None:
        (tmp_path / "frames").mkdir()
        with pytest.raises(RenderMediaError, match="No frames match pattern"):
            await render_media(
                frames_pattern=str(tmp_path / "frames" / "frame_%06d.png"),
                output_path=tmp_path / "o.mp4",
                frame_count=2,
            )

    async def test_happy_path_encodes(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        for i in (1, 2, 3):
            (frames / f"frame_{i:06d}.png").write_bytes(b"p")

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            Path(args[-1]).write_bytes(b"mp4")

        out = tmp_path / "o.mp4"
        with patch(
            "oprim._encode_frames.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)
        ):
            result = await render_media(
                frames_pattern=str(frames / "frame_%06d.png"),
                output_path=out,
                fps=24,
                frame_count=3,
                width=1280,
                height=720,
            )
        assert result == out and out.exists()

    async def test_encode_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        with (
            patch(
                "oprim._encode_frames.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(RenderMediaError, match="frame encode failed"),
        ):
            await render_media(
                frames_pattern=str(tmp_path / "f_%06d.png"), output_path=tmp_path / "o.mp4"
            )

    async def test_encode_rejects_bad_fps(self, tmp_path: Path) -> None:
        from oprim._encode_frames import EncodeFramesError, encode_frames_to_mp4

        with pytest.raises(EncodeFramesError, match="fps must be"):
            await encode_frames_to_mp4(
                input_pattern=str(tmp_path / "f_%06d.png"),
                output_path=tmp_path / "o.mp4",
                fps=0,
            )


# ---------------------------------------------------------------------------
# §4.8 — audio_mix enrichment + mix_audio_tracks alias
# ---------------------------------------------------------------------------


class TestAudioMixEnrichment:
    async def test_empty_inputs(self, tmp_path: Path) -> None:
        with pytest.raises(AudioMixError, match="must not be empty"):
            await audio_mix(inputs=[], output_path=tmp_path / "o.wav")

    async def test_missing_input_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioMixError, match="not found"):
            await audio_mix(inputs=[tmp_path / "no.wav"], output_path=tmp_path / "o.wav")

    async def test_weights_length_mismatch(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        with pytest.raises(AudioMixError, match="weights length"):
            await audio_mix(
                inputs=[a, b], weights=[1.0], output_path=tmp_path / "o.wav"
            )

    async def test_offsets_length_mismatch(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        with pytest.raises(AudioMixError, match="offsets length"):
            await audio_mix(inputs=[a, b], offsets=[0.0], output_path=tmp_path / "o.wav")

    async def test_negative_offset_rejected(self, tmp_path: Path) -> None:
        a = _video(tmp_path, "a.wav")
        with pytest.raises(AudioMixError, match="offsets must be >= 0"):
            await audio_mix(inputs=[a], offsets=[-0.5], output_path=tmp_path / "o.wav")

    async def test_fade_length_mismatch(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        with pytest.raises(AudioMixError, match="fade_in_s length"):
            await audio_mix(
                inputs=[a, b], fade_in_s=[1.0], output_path=tmp_path / "o.wav"
            )

    async def test_negative_fade_rejected(self, tmp_path: Path) -> None:
        a = _video(tmp_path, "a.wav")
        with pytest.raises(AudioMixError, match="fade_out_s must be >= 0"):
            await audio_mix(inputs=[a], fade_out_s=-1.0, output_path=tmp_path / "o.wav")

    async def test_unknown_duration_policy(self, tmp_path: Path) -> None:
        a = _video(tmp_path, "a.wav")
        with pytest.raises(AudioMixError, match="Unknown duration policy"):
            await audio_mix(
                inputs=[a], duration="whatever", output_path=tmp_path / "o.wav"  # type: ignore[arg-type]
            )

    async def test_filter_graph_carries_offset_fades_and_policy(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"wav")

        with patch("oprim._audio_mix.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            await audio_mix(
                inputs=[a, b],
                weights=[1.0, 0.3],
                offsets=[0.0, 0.5],
                fade_in_s=0.2,
                fade_out_s=[0.1, 0.4],
                duration="shortest",
                output_path=tmp_path / "o.wav",
            )
        graph = seen[0][seen[0].index("-filter_complex") + 1]
        assert graph == (
            "[0]volume=1.0,afade=t=in:st=0:d=0.2,afade=t=out:st=-0.1:d=0.1[a0];"
            "[1]volume=0.3,adelay=delays=500:all=1,"
            "afade=t=in:st=0:d=0.2,afade=t=out:st=-0.4:d=0.4[a1];"
            "[a0][a1]amix=inputs=2:duration=shortest"
        )

    async def test_plain_mix_keeps_legacy_graph(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"wav")

        with patch("oprim._audio_mix.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            await audio_mix(inputs=[a, b], weights=[1.0, 0.3], output_path=tmp_path / "o.wav")
        graph = seen[0][seen[0].index("-filter_complex") + 1]
        assert graph == (
            "[0]volume=1.0[a0];[1]volume=0.3[a1];[a0][a1]amix=inputs=2:duration=longest"
        )

    async def test_mix_audio_tracks_alias_delegates(self, tmp_path: Path) -> None:
        a, b = _video(tmp_path, "a.wav"), _video(tmp_path, "b.wav")
        seen: list[list[str]] = []

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            seen.append(args)
            Path(args[-1]).write_bytes(b"wav")

        with patch("oprim._audio_mix.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            await mix_audio_tracks(
                inputs=[a, b], offsets=[0.0, 0.25], output_path=tmp_path / "o.wav"
            )
        graph = seen[0][seen[0].index("-filter_complex") + 1]
        assert "adelay=delays=250:all=1" in graph

    async def test_ffmpeg_failure_wrapped(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        a = _video(tmp_path, "a.wav")
        with (
            patch(
                "oprim._audio_mix.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ),
            pytest.raises(AudioMixError, match="mixing failed"),
        ):
            await audio_mix(inputs=[a], output_path=tmp_path / "o.wav")


# ---------------------------------------------------------------------------
# §4.2 / §4.9 — transcription + subtitle aliases
# ---------------------------------------------------------------------------


class TestCapabilityAliases:
    async def test_burn_subtitles_delegates(self, tmp_path: Path) -> None:
        video = _video(tmp_path)
        srt = tmp_path / "s.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

        async def _fake_run(*, args: list[str], timeout_s: float, expected_output=None):  # noqa: ANN001
            Path(args[-1]).write_bytes(b"mp4")

        out = tmp_path / "o.mp4"
        with patch("oprim._subtitle_burn.ffmpeg_run", new=AsyncMock(side_effect=_fake_run)):
            assert await burn_subtitles(video_path=video, srt_paths=[srt], output_path=out) == out
            assert callable(subtitle_burn)
            with pytest.raises(SubtitleBurnError, match="not found"):
                await burn_subtitles(
                    video_path=tmp_path / "no.mp4", srt_paths=[srt], output_path=out
                )

    async def test_transcribe_media_delegates(self, tmp_path: Path) -> None:
        audio = _video(tmp_path, "a.wav")
        sentinel = object()

        async def _fake(*, audio_path, backend, model_size, language, model_path):  # noqa: ANN001
            assert audio_path == audio
            return sentinel

        with patch("oprim._transcribe_audio.transcribe_audio", new=_fake):
            assert await transcribe_media(audio_path=audio) is sentinel
        assert callable(transcribe_audio)

    async def test_transcribe_media_propagates_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await transcribe_media(audio_path=tmp_path / "no.wav")


# ---------------------------------------------------------------------------
# §8 — single-source invariants
# ---------------------------------------------------------------------------


class TestSingleSourceInvariants:
    def _definers(self, name: str) -> list[str]:
        pkg = Path(__file__).resolve().parent.parent / "oprim"
        hits: list[str] = []
        for py in sorted(pkg.rglob("*.py")):
            if py.name == "__init__.py":
                # The package __init__ re-declares lazy wrappers for obase-dependent
                # elements (tts_synthesize, llm_complete, ...); the element contract
                # itself lives in exactly one implementation module.
                continue
            if name in _top_level_names(py):
                hits.append(str(py.relative_to(pkg)))
        return hits

    @pytest.mark.parametrize(
        ("name", "expected_module"),
        [
            ("media_probe", "_media_probe.py"),
            ("probe_media", "_media_probe.py"),
            ("probe_duration", "_probe_duration.py"),
            ("audio_mix", "_audio_mix.py"),
            ("mix_audio_tracks", "_audio_mix.py"),
            ("subtitle_burn", "_subtitle_burn.py"),
            ("burn_subtitles", "_subtitle_burn.py"),
            ("transcribe_audio", "_transcribe_audio.py"),
            ("transcribe_media", "_transcribe_audio.py"),
            ("render_media", "_render_media.py"),
            ("extract_video_frames", "_extract_video_frames.py"),
            ("extract_media_segment", "_extract_media_segment.py"),
            ("extract_audio_waveform", "_extract_audio_waveform.py"),
            ("segment_media", "_segment_media.py"),
            ("encode_voice_reference", "_encode_voice_reference.py"),
            ("video_generate", "_video_generate.py"),
            ("tts_synthesize", "tts_synthesize.py"),
            ("audio_normalize", "_audio_normalize.py"),
            ("audio_video_merge", "_audio_video_merge.py"),
        ],
    )
    def test_exactly_one_defining_module(self, name: str, expected_module: str) -> None:
        assert self._definers(name) == [expected_module]

    def test_media_elements_live_in_one_canonical_package(self) -> None:
        import oprim

        for name in (
            "probe_media",
            "extract_video_frames",
            "extract_media_segment",
            "extract_audio_waveform",
            "segment_media",
            "encode_voice_reference",
            "render_media",
            "mix_audio_tracks",
            "burn_subtitles",
            "transcribe_media",
        ):
            assert name in oprim.__all__, name

    def test_no_vendor_names_in_new_canonical_set(self) -> None:
        """SPEC §8.2/§8.3: capability names must not carry a vendor prefix."""
        forbidden = (
            "longlive",
            "wan_",
            "veo",
            "kling",
            "hailuo",
            "ltx",
            "luxtts",
            "edge_tts",
            "fal_",
            "dashscope",
        )
        canonical = (
            "probe_media",
            "extract_video_frames",
            "extract_media_segment",
            "extract_audio_waveform",
            "segment_media",
            "encode_voice_reference",
            "render_media",
            "mix_audio_tracks",
            "burn_subtitles",
            "transcribe_media",
            "video_generate",
            "tts_synthesize",
        )
        for name in canonical:
            lowered = name.lower()
            for token in forbidden:
                assert token not in lowered, f"{name} leaks vendor token {token!r}"

    def test_canonical_media_atoms_use_only_the_obase_ffmpeg_executor(self) -> None:
        """SPEC §8.4: media atoms route FFmpeg through obase.ffmpeg, never a second wrapper.

        `_ffprobe` owns the single ffprobe spawn; `_extract_audio_waveform` spawns
        ffmpeg directly because it must read decoded PCM off stdout, which
        `obase.ffmpeg.run` discards. Both are deliberate and asserted here so a
        future third spawn site has to be declared deliberately too.
        """
        pkg = Path(__file__).resolve().parent.parent / "oprim"
        declared_direct_spawners = {"_ffprobe.py", "_extract_audio_waveform.py"}
        offenders: list[str] = []
        for py in sorted(pkg.glob("_*.py")):
            if not _is_media_atom(py):
                continue
            source = py.read_text(encoding="utf-8")
            if "subprocess.run" not in source:
                continue
            if py.name in declared_direct_spawners:
                continue
            if "obase.ffmpeg" not in source:
                offenders.append(py.name)
        assert offenders == []

    def test_no_second_ffmpeg_wrapper_module(self) -> None:
        """SPEC §8.4: no video_ffmpeg / media_ffmpeg / hevi_ffmpeg style module."""
        pkg = Path(__file__).resolve().parent.parent / "oprim"
        forbidden = ("video_ffmpeg", "media_ffmpeg", "hevi_ffmpeg", "ffmpeg_wrapper")
        offenders = [
            py.name
            for py in pkg.rglob("*.py")
            if any(token in py.stem for token in forbidden)
        ]
        assert offenders == []

    def test_known_sibling_call_debt_is_pinned(self) -> None:
        """SPEC §12.C is not yet met by pre-existing code — pin the exact debt.

        `3o_lint` deliberately keeps `_video_generate.py` and
        `_render_html_to_mp4.py` out of its allowlist because both call sibling
        OPrim elements. That debt predates this work and fixing it is the
        §12.C refactor, not P0-A/B/C. This test records the current set so any
        *new* sibling edge fails loudly instead of hiding in the same files.

        When the DAG is refactored (provider dispatch moved behind an injected
        resolver, `validate_html` moved to infra), update this set to empty.
        """
        pkg = Path(__file__).resolve().parent.parent / "oprim"
        offenders: dict[str, list[str]] = {}
        for name in ("_video_generate.py", "_render_html_to_mp4.py"):
            targets: list[str] = []
            for line in (pkg / name).read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("from oprim.") or stripped.startswith("import oprim."):
                    target = stripped.split()[1]
                    if target.split(".")[1] in _MEDIA_INFRA_ALLOWLIST:
                        continue
                    targets.append(target)
            if targets:
                offenders[name] = sorted(set(targets))

        assert offenders == {
            "_render_html_to_mp4.py": ["oprim._validate_html"],
            "_video_generate.py": [
                "oprim._fal_queue_generate",
                "oprim._hailuo_generate",
                "oprim._kling_v2_generate",
                "oprim._ltx2_cloud_generate",
                "oprim._providers.wan_cloud",
                "oprim._veo3_generate",
            ],
        }

    def test_new_media_atoms_do_not_call_sibling_atoms(self) -> None:
        """SPEC §2.2: an OPrim must not import another OPrim's element."""
        pkg = Path(__file__).resolve().parent.parent / "oprim"
        media_atoms = (
            "_media_probe.py",
            "_probe_duration.py",
            "_extract_video_frames.py",
            "_extract_media_segment.py",
            "_extract_audio_waveform.py",
            "_segment_media.py",
            "_encode_voice_reference.py",
            "_render_media.py",
            "_audio_mix.py",
        )
        allowed_infra = _MEDIA_INFRA_ALLOWLIST
        offenders: list[str] = []
        for name in media_atoms:
            source = (pkg / name).read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("import ", "from ")):
                    continue
                if "obase" in stripped:
                    continue
                if not stripped.startswith("from oprim."):
                    continue
                target = stripped.split()[1]
                if target.split(".")[1] not in allowed_infra:
                    offenders.append(f"{name}: {stripped}")
        assert offenders == []


def _top_level_names(py: Path) -> set[str]:
    import ast

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


#: Modules in the canonical media set — the surface §8 single-source rules govern.
_MEDIA_ATOMS = (
    "_media_probe.py",
    "_probe_duration.py",
    "_ffprobe.py",
    "_extract_video_frames.py",
    "_extract_media_segment.py",
    "_extract_audio_waveform.py",
    "_segment_media.py",
    "_encode_voice_reference.py",
    "_encode_frames.py",
    "_render_media.py",
    "_render_html_to_mp4.py",
    "_audio_mix.py",
    "_audio_normalize.py",
    "_audio_video_merge.py",
    "_subtitle_burn.py",
    "_video_concat.py",
    "_video_recompose.py",
)


#: Same-layer infra an OPrim may import without becoming a sibling call.
#: Mirrors 3o_lint `OPRIM_INFRA_PREFIXES` for the media subset.
_MEDIA_INFRA_ALLOWLIST = {
    "_exceptions",
    "_ffprobe",
    "_encode_frames",
    "_config",
    "_types",
    "_media_types",
}


def _is_media_atom(py: Path) -> bool:
    return py.name in _MEDIA_ATOMS


# ---------------------------------------------------------------------------
# Real-ffmpeg integration (skipped when ffmpeg is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestRealFfmpeg:
    @pytest.fixture
    def real_video(self, tmp_path: Path) -> Path:
        out = tmp_path / "real.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "quiet", "-f", "lavfi",
                "-i", "testsrc=duration=4:size=320x240:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
            ],
            check=True, timeout=60,
        )
        return out

    def test_probe_media_reports_full_metadata(self, real_video: Path) -> None:
        info = probe_media(path=real_video)
        assert info.is_video and info.is_audio
        assert info.width == 320 and info.height == 240
        assert info.fps == pytest.approx(10.0, rel=1e-3)
        assert info.duration_seconds == pytest.approx(4.0, abs=0.3)
        assert info.audio_channels == 1
        assert info.bitrate is not None and info.bitrate > 0

    def test_probe_duration_agrees_with_canonical_probe(self, real_video: Path) -> None:
        assert probe_duration(real_video) == pytest.approx(
            probe_media(path=real_video).duration_seconds, abs=0.01
        )

    async def test_extract_video_frames_real(self, real_video: Path, tmp_path: Path) -> None:
        out = tmp_path / "frames"
        result = await extract_video_frames(
            video_path=real_video, output_dir=out, mode="count", count=4, duration_s=4.0
        )
        assert len(result.frames) == 4
        assert all(f.path.exists() and f.path.stat().st_size > 0 for f in result.frames)

    async def test_extract_media_segment_real(self, real_video: Path, tmp_path: Path) -> None:
        out = tmp_path / "seg.mp4"
        await extract_media_segment(
            media_path=real_video, start=1.0, end=2.0, output_path=out, reencode=True
        )
        assert probe_media(path=out).duration_seconds == pytest.approx(1.0, abs=0.35)

    async def test_segment_media_scene_real(self, real_video: Path) -> None:
        result = await segment_media(
            media_path=real_video,
            method="scene",
            threshold=0.01,
            min_duration=0.1,
            total_duration=4.0,
        )
        assert result.segments
        assert result.segments[0].start == 0.0
        for seg in result.segments:
            assert seg.end > seg.start

    def test_extract_audio_waveform_real(self, real_video: Path) -> None:
        wf = extract_audio_waveform(audio_path=real_video, sample_rate=4000, target_samples=50)
        assert wf.duration == pytest.approx(4.0, abs=0.3)
        assert len(wf.samples) == 50
        assert max(abs(s) for s in wf.samples) == pytest.approx(1.0)
        assert wf.peak_db is not None and wf.rms_db is not None

    async def test_encode_voice_reference_real(self, real_video: Path, tmp_path: Path) -> None:
        ref = await encode_voice_reference(
            audio_path=real_video, output_path=tmp_path / "voice.wav", max_duration_s=2.0
        )
        assert ref.path.exists()
        info = probe_media(path=ref.path)
        assert info.audio_sample_rate == 24000
        assert info.audio_channels == 1
        assert info.duration_seconds == pytest.approx(2.0, abs=0.2)

    async def test_render_media_real(self, real_video: Path, tmp_path: Path) -> None:
        frames = tmp_path / "seq"
        frames.mkdir()
        for i in range(1, 6):
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-y", "-i", str(real_video), "-vf",
                 f"select=eq(n\\,{i})", "-vframes", "1", str(frames / f"f_{i:06d}.png")],
                check=True, timeout=60,
            )
        out = tmp_path / "rendered.mp4"
        await render_media(
            frames_pattern=str(frames / "f_%06d.png"), output_path=out, fps=5
        )
        assert probe_media(path=out).duration_seconds == pytest.approx(1.0, abs=0.4)
