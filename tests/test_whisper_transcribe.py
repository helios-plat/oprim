import pytest
from unittest.mock import patch, AsyncMock

from oprim._whisper_transcribe import WhisperTranscribeError, whisper_transcribe


@pytest.mark.asyncio
async def test_missing_file_raises(tmp_path):
    with pytest.raises(WhisperTranscribeError):
        await whisper_transcribe(audio_path=tmp_path / "nope.wav")


@pytest.mark.asyncio
async def test_delegates_to_transcribe_audio(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    fake = {"text": "hi", "segments": []}
    with patch("oprim.transcribe_audio", new=AsyncMock(return_value=fake)) as m:
        res = await whisper_transcribe(audio_path=f)
        assert res == fake
        m.assert_awaited()
