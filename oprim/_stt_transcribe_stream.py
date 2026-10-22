"""oprim.stt_transcribe_stream — 流式语音转写.

经注入的 transcriber Protocol 对音频（文件或帧列表）执行流式转写，
返回标准化 dict（含分段）。

Example:
    >>> r = await stt_transcribe_stream("/tmp/a.wav", transcriber=my_stt)
    >>> r["text"].strip() != ""
    True
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class SttTranscribeError(OprimError):
    """语音转写失败。"""


@runtime_checkable
class TranscriberHandle(Protocol):
    """STT 转写协议（注入面）。"""

    async def transcribe(self, audio: Any, *, language: str | None = None) -> dict[str, Any]: ...


async def stt_transcribe_stream(
    audio: str | list[bytes],
    *,
    transcriber: TranscriberHandle,
    language: str | None = None,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """流式语音转写。

    Args:
        audio: 音频文件路径 或 帧列表（16bit PCM）。
        transcriber: 转写器（注入）。
        language: 语言提示（可选）。
        sample_rate: 采样率（帧列表模式）。

    Returns:
        {"status": "ok", "text": str, "segments": list[dict], "frames": int}

    Raises:
        SttTranscribeError: 转写失败。
        OprimValidationError: audio 为空 / transcriber 未注入。
    """
    if not audio:
        raise OprimValidationError("stt_transcribe_stream: audio must not be empty")
    if transcriber is None:
        raise OprimValidationError("stt_transcribe_stream: transcriber must be injected")

    try:
        result = await transcriber.transcribe(audio, language=language)
    except Exception as exc:
        raise SttTranscribeError(
            f"stt_transcribe_stream failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    if not isinstance(result, dict):
        result = {"text": str(result)}

    frames = len(audio) if isinstance(audio, list) else 0
    return {
        "status": "ok",
        "text": str(result.get("text", "")),
        "segments": result.get("segments", []),
        "frames": result.get("frames", frames),
        "sample_rate": sample_rate,
    }
