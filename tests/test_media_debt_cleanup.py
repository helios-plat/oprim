"""P0-D D4–D8 — quality-debt closure tests.

Each class is the executable form of one debt item, and each pins the *fixed*
behaviour so the debt cannot come back:

* D4  `audio_lufs` was declared but never written — a silent lie
* D5  `_manifest.py` was an orphaned, stale, media-blind element list
* D6  `external/clients/*` were empty shells advertised as public capabilities
* D7  `_vibevoice_synthesize` carried a mutable speaker map inside a closure
* D8  `transcribe_audio` reloaded the Whisper model on every single call
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "oprim"

_HAS_FFMPEG = __import__("shutil").which("ffmpeg") is not None


def _completed(stdout: bytes, stderr: bytes) -> AsyncMock:
    """A mock subprocess whose communicate() yields the given streams."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = 0
    return proc


def _probe_payload(*, with_audio: bool) -> bytes:
    """Minimal ffprobe -print_format json payload for video_quality_metrics."""
    streams = [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "25/1",
        }
    ]
    if with_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return json.dumps(
        {"format": {"duration": "10", "bit_rate": "1500000"}, "streams": streams}
    ).encode()


# ---------------------------------------------------------------------------
# D4 — audio_lufs
# ---------------------------------------------------------------------------


class TestAudioLufsIsNoLongerDead:
    def test_field_still_exists_for_compatibility(self) -> None:
        """oskill constructs VideoQualityMetrics(audio_lufs=None) in 4 places."""
        from oprim._video_quality_metrics import VideoQualityMetrics

        assert "audio_lufs" in VideoQualityMetrics.model_fields

    def test_defaults_to_not_measuring(self) -> None:
        from oprim._video_quality_metrics import video_quality_metrics

        assert "measure_loudness" in inspect.signature(video_quality_metrics).parameters
        assert inspect.signature(video_quality_metrics).parameters[
            "measure_loudness"
        ].default is False

    async def test_populated_when_measured(self, tmp_path: Path) -> None:
        from oprim._video_quality_metrics import video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")

        # Call 1 is ffprobe (JSON on stdout); call 2 is the ffmpeg volumedetect
        # pass, whose reading lands on stderr.
        procs = [
            _completed(_probe_payload(with_audio=True), b""),
            _completed(
                b"",
                b"[Parsed_volumedetect_0] mean_volume: -21.3 dB\n"
                b"[Parsed_volumedetect_0] max_volume: -3.0 dB\n",
            ),
        ]
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=procs)):
            result = await video_quality_metrics(video_path=video, measure_loudness=True)

        assert result.audio_lufs == pytest.approx(-21.3)

    async def test_none_without_audio_stream(self, tmp_path: Path) -> None:
        from oprim._video_quality_metrics import video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(_probe_payload(with_audio=False), b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            result = await video_quality_metrics(video_path=video, measure_loudness=True)

        assert result.codec_audio is None
        assert result.audio_lufs is None

    async def test_ffmpeg_failure_degrades_to_none(self, tmp_path: Path) -> None:
        """Loudness is optional QC; it must never fail the metrics call."""
        from obase.ffmpeg import FFmpegError

        from oprim._video_quality_metrics import video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(_probe_payload(with_audio=True), b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            with patch(
                "oprim._video_quality_metrics.ffmpeg_run",
                new=AsyncMock(side_effect=FFmpegError("boom", code=1)),
            ):
                result = await video_quality_metrics(video_path=video, measure_loudness=True)

        assert result.audio_lufs is None
        assert result.codec_audio == "aac"


# ---------------------------------------------------------------------------
# D5 — stale _manifest
# ---------------------------------------------------------------------------


class TestManifestRemoved:
    def test_module_is_gone(self) -> None:
        assert not (PKG / "_manifest.py").exists()

    def test_not_in_coverage_omit(self) -> None:
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert "oprim/_manifest.py" not in text

    def test_nothing_imports_it(self) -> None:
        offenders: list[str] = []
        for py in list(PKG.rglob("*.py")) + list((REPO / "tests").rglob("*.py")):
            if "worktrees" in str(py):
                continue
            if py.name == Path(__file__).name:
                continue  # this file names the module in prose
            source = py.read_text(encoding="utf-8")
            if "oprim._manifest" in source or "oprim/_manifest" in source:
                offenders.append(str(py.relative_to(REPO)))
        assert offenders == []

    def test_element_map_has_no_second_authority(self) -> None:
        """A static list can never track AST discovery; there must be exactly one."""
        import oprim

        assert hasattr(oprim, "_ELEMENT_MAP")
        assert len(oprim.__all__) == len(oprim._ELEMENT_MAP)


# ---------------------------------------------------------------------------
# D6 — empty stub clients
# ---------------------------------------------------------------------------


STUB_NAMES = ("WhisperClient", "WhisperSegment", "TtsClient", "SdClient", "SearxngClient")


class TestStubClientsUnpublished:
    def test_not_in_all(self) -> None:
        import oprim

        assert [n for n in STUB_NAMES if n in oprim.__all__] == []

    def test_modules_still_importable(self) -> None:
        """oskill.knowledge/* imports these by module path — deleting breaks oskill."""
        from oprim.external.clients.sd_client import SdClient
        from oprim.external.clients.searxng_client import SearxngClient, WebSearchResult
        from oprim.external.clients.whisper_client import WhisperClient, WhisperSegment

        assert SdClient and SearxngClient and WebSearchResult
        assert WhisperClient and WhisperSegment

    def test_each_states_it_is_unimplemented(self) -> None:
        for name in ("sd_client", "searxng_client", "tts_client", "whisper_client"):
            source = (PKG / "external" / "clients" / f"{name}.py").read_text(encoding="utf-8")
            assert "UNIMPLEMENTED PLACEHOLDER" in source, name
            assert "__oprim_layer__" in source, name

    def test_placeholder_classes_have_no_behaviour(self) -> None:
        """Pin the defect: these are still empty. When implemented, this fails."""
        from oprim.external.clients.sd_client import SdClient
        from oprim.external.clients.searxng_client import SearxngClient
        from oprim.external.clients.tts_client import TtsClient
        from oprim.external.clients.whisper_client import WhisperClient, WhisperSegment

        for cls in (WhisperClient, WhisperSegment, TtsClient, SdClient, SearxngClient):
            behavioural = {
                name
                for name in vars(cls)
                if not name.startswith("__")
                and not callable(getattr(cls, name, None))
            }
            methods = [n for n in vars(cls) if not n.startswith("__")]
            assert behavioural == set(), f"{cls.__name__} grew data members: {behavioural}"
            assert methods == [], f"{cls.__name__} grew methods: {methods}"


# ---------------------------------------------------------------------------
# D7 — vibevoice speaker map
# ---------------------------------------------------------------------------


class _Line:
    def __init__(self, speaker_id: str, text: str = "hi") -> None:
        self.speaker_id = speaker_id
        self.text = text
        self.voice_ref = None


class TestSpeakerNumberingIsDeterministic:
    def test_numbering_depends_only_on_the_roster(self) -> None:
        from oprim._vibevoice_synthesize import _speaker_numbers

        a = _speaker_numbers([_Line("bob"), _Line("ann")])
        b = _speaker_numbers([_Line("ann"), _Line("bob")])
        assert a == b == {"ann": 1, "bob": 2}

    def test_numbers_are_dense_and_one_based(self) -> None:
        from oprim._vibevoice_synthesize import _speaker_numbers

        assert _speaker_numbers([_Line("c"), _Line("a"), _Line("b")]) == {
            "a": 1,
            "b": 2,
            "c": 3,
        }

    def test_closure_no_longer_mutates_state(self) -> None:
        """The inference closure must not carry a growing dict."""
        source = (PKG / "_vibevoice_synthesize.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "_make_inference":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.FunctionDef) and inner.name == "_infer":
                    for sub in ast.walk(inner):
                        if isinstance(sub, ast.Assign):
                            targets = [
                                t.id for t in sub.targets if isinstance(t, ast.Name)
                            ]
                            if "_spk_map" in targets:
                                offenders.append("_spk_map assigned inside _infer")
                        if (
                            isinstance(sub, ast.AugAssign)
                            and isinstance(sub.target, ast.Name)
                            and sub.target.id == "_spk_map"
                        ):
                            offenders.append("_spk_map mutated inside _infer")
        assert offenders == []

    def test_unknown_speaker_fails_loudly(self) -> None:
        from oprim._vibevoice_synthesize import VibeVoiceError, _make_inference

        infer = _make_inference(object(), False, {"ann": 1})
        with pytest.raises(VibeVoiceError, match="not in script roster"):
            infer("hello", "zed", None)


# ---------------------------------------------------------------------------
# D8 — reusable ASR model
# ---------------------------------------------------------------------------


class TestAsrModelReuse:
    def test_cache_is_reusable_across_calls(self, tmp_path: Path) -> None:
        from oprim._asr_runtime import AsrModelCache

        cache = AsrModelCache()
        built: list[str] = []

        def _fake_build(model_path: Path, device: str, compute_type: str):
            built.append(str(model_path))
            return object()

        with patch.object(AsrModelCache, "_load", staticmethod(_fake_build)):
            first = cache.get(model_path=tmp_path, device="cpu", compute_type="int8")
            second = cache.get(model_path=tmp_path, device="cpu", compute_type="int8")
            third = cache.get(model_path=tmp_path, device="cpu", compute_type="int8")

        assert first is second is third
        assert built == [str(tmp_path)], "model must be constructed exactly once"
        assert cache.load_count == 1
        assert len(cache) == 1

    def test_distinct_keys_load_separately(self, tmp_path: Path) -> None:
        from oprim._asr_runtime import AsrModelCache

        cache = AsrModelCache()
        with patch.object(AsrModelCache, "_load", staticmethod(lambda *a, **k: object())):
            a = cache.get(model_path=tmp_path, device="cpu", compute_type="int8")
            b = cache.get(model_path=tmp_path, device="cpu", compute_type="float16")
        assert a is not b
        assert cache.load_count == 2

    def test_clear_drops_models(self, tmp_path: Path) -> None:
        from oprim._asr_runtime import AsrModelCache

        cache = AsrModelCache()
        with patch.object(AsrModelCache, "_load", staticmethod(lambda *a, **k: object())):
            cache.get(model_path=tmp_path, device="cpu", compute_type="int8")
            assert len(cache) == 1
            cache.clear()
            assert len(cache) == 0

    def test_module_level_default_is_clearable(self) -> None:
        from oprim._asr_runtime import clear_asr_cache, default_asr_cache

        cache = default_asr_cache()
        assert cache is default_asr_cache(), "default cache must be a stable singleton"
        clear_asr_cache()
        assert len(default_asr_cache()) == 0

    def test_element_exposes_an_injection_point_not_a_global(self) -> None:
        from oprim._transcribe_audio import transcribe_audio

        params = inspect.signature(transcribe_audio).parameters
        assert "model_cache" in params
        assert params["model_cache"].default is None

    def test_asr_runtime_is_infra_not_a_capability(self) -> None:
        import oprim

        assert oprim._MODULE_LAYERS["oprim._asr_runtime"] == "infra"
        assert "default_asr_cache" not in oprim.__all__

    async def test_transcribe_reuses_one_model_for_many_calls(self, tmp_path: Path) -> None:
        """The actual D8 regression: N calls used to mean N model loads."""
        from oprim._asr_runtime import AsrModelCache
        from oprim._transcribe_audio import transcribe_audio

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        cache = AsrModelCache()
        loads: list[str] = []

        class _Seg:
            def __init__(self, i: int) -> None:
                self.start, self.end, self.text = float(i), float(i + 1), f"w{i} "

        class _Info:
            language = "en"
            duration = 3.0

        class _Model:
            def transcribe(self, path: str, language: str):  # noqa: ANN001, ANN202
                return iter([_Seg(i) for i in range(3)]), _Info()

        def _fake_build(model_path: Path, device: str, compute_type: str):
            loads.append(str(model_path))
            return _Model()

        with patch.object(AsrModelCache, "_load", staticmethod(_fake_build)):
            for _ in range(3):
                result = await transcribe_audio(
                    audio_path=audio, model_path=str(models_dir), model_cache=cache
                )
                assert len(result.segments) == 3

        assert len(loads) == 1, f"expected 1 model load, got {len(loads)}"
        assert cache.load_count == 1


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
class TestRealFfmpegLoudness:
    def test_measure_loudness_against_a_real_tone(self, tmp_path: Path) -> None:
        import asyncio

        from oprim._video_quality_metrics import video_quality_metrics

        video = tmp_path / "tone.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "quiet", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
            ],
            check=True,
            timeout=60,
        )
        result = asyncio.run(video_quality_metrics(video_path=video, measure_loudness=True))
        assert result.codec_audio == "aac"
        assert result.audio_lufs is not None
        assert -60.0 < result.audio_lufs < 0.0
