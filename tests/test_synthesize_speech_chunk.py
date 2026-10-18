"""Tests for oprim.synthesize_speech_chunk."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from oprim._synthesize_speech_chunk import SynthesizeSpeechChunkError, synthesize_speech_chunk


@pytest.mark.asyncio
async def test_empty_text_raises(tmp_path):
    with pytest.raises(SynthesizeSpeechChunkError):
        await synthesize_speech_chunk(text="   ")


@pytest.mark.asyncio
async def test_runtime_unavailable_raises():
    with patch.dict("sys.modules", {"obase.kokoro_runtime": None}):
        with pytest.raises(SynthesizeSpeechChunkError):
            await synthesize_speech_chunk(text="hi")


@pytest.mark.asyncio
async def test_success_with_runtime(tmp_path):
    fake_runtime = MagicMock()
    fake_runtime.synthesize.return_value = [0.1, 0.2]
    out = tmp_path / "out.wav"
    # patch write_audio_file
    with patch("obase.kokoro_runtime.write_audio_file", return_value=out) as waf:
        # inject runtime directly
        res = await synthesize_speech_chunk(text="hi", runtime=fake_runtime, output_path=out)
        assert res == out
        fake_runtime.synthesize.assert_called_once()
