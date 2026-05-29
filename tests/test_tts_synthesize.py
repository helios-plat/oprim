"""Tests for oprim.tts_synthesize."""

from __future__ import annotations

from pathlib import Path

import pytest

from obase import ProviderRegistry
from oprim.tts_synthesize import TTSError, tts_synthesize


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestTTSSynthesize:
    async def test_normal_success(self, tmp_path: Path) -> None:
        async def _tts(**kw: object) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 100)

        ProviderRegistry.register(category="tts", name="mock", fn=_tts)
        out = tmp_path / "speech.mp3"
        result = await tts_synthesize(
            provider="mock", text="Hello", voice="en-US-AriaNeural", output_path=out
        )
        assert result == out
        assert out.exists()

    async def test_empty_text_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="text must not be empty"):
            await tts_synthesize(
                provider="mock", text="", voice="v", output_path=tmp_path / "out.mp3"
            )

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="TTS provider not found"):
            await tts_synthesize(
                provider="nope", text="Hi", voice="v", output_path=tmp_path / "out.mp3"
            )

    async def test_provider_failure(self, tmp_path: Path) -> None:
        async def _fail(**kw: object) -> None:
            raise RuntimeError("network error")

        ProviderRegistry.register(category="tts", name="bad", fn=_fail)
        with pytest.raises(TTSError, match="TTS synthesis failed"):
            await tts_synthesize(
                provider="bad", text="Hi", voice="v", output_path=tmp_path / "out.mp3"
            )

    async def test_output_not_produced(self, tmp_path: Path) -> None:
        async def _noop(**kw: object) -> None:
            pass

        ProviderRegistry.register(category="tts", name="noop", fn=_noop)
        with pytest.raises(TTSError, match="did not produce"):
            await tts_synthesize(
                provider="noop", text="Hi", voice="v", output_path=tmp_path / "out.mp3"
            )

    async def test_rate_and_pitch_passed(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        async def _cap(**kw: object) -> None:
            captured.update(kw)
            Path(str(kw["output_path"])).write_bytes(b"\x00")

        ProviderRegistry.register(category="tts", name="cap", fn=_cap)
        out = tmp_path / "out.mp3"
        await tts_synthesize(
            provider="cap", text="Hi", voice="v", output_path=out,
            rate="+20%", pitch="+5Hz",
        )
        assert captured["rate"] == "+20%"
        assert captured["pitch"] == "+5Hz"
