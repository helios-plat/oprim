"""oprim.tts_synthesize_stream — 流式语音合成.

经注入的 synthesizer Protocol 对文本执行流式 TTS，输出分块音频；
返回块数/字节数/格式。

Example:
    >>> r = await tts_synthesize_stream("你好", synthesizer=my_tts)
    >>> r["chunks"] >= 1
    True
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class TtsSynthError(OprimError):
    """语音合成失败。"""


@runtime_checkable
class SynthesizerHandle(Protocol):
    """TTS 合成协议（注入面）。"""

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[bytes]: ...


async def tts_synthesize_stream(
    text: str,
    *,
    synthesizer: SynthesizerHandle,
    voice: str | None = None,
    format_hint: str = "pcm_s16le",
) -> dict[str, Any]:
    """流式语音合成。

    Args:
        text: 待合成文本。
        synthesizer: 合成器（注入，须提供 synthesize_stream 异步生成器）。
        voice: 音色（可选）。
        format_hint: 音频格式提示。

    Returns:
        {"status": "ok", "chunks": int, "bytes": int, "format": str,
         "chunk_size": int, "text_chars": int}

    Raises:
        TtsSynthError: 合成失败。
        OprimValidationError: text 为空 / synthesizer 未注入。
    """
    if not text or not text.strip():
        raise OprimValidationError("tts_synthesize_stream: text must not be empty")
    if synthesizer is None:
        raise OprimValidationError("tts_synthesize_stream: synthesizer must be injected")

    stream_fn = getattr(synthesizer, "synthesize_stream", None)
    if stream_fn is None or not callable(stream_fn):
        raise TtsSynthError(
            "tts_synthesize_stream: synthesizer has no synthesize_stream()"
        )

    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in stream_fn(text, voice=voice):
            chunks.append(chunk)
            total += len(chunk)
    except Exception as exc:
        raise TtsSynthError(
            f"tts_synthesize_stream failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    return {
        "status": "ok",
        "chunks": len(chunks),
        "bytes": total,
        "format": format_hint,
        "chunk_size": len(chunks[0]) if chunks else 0,
        "text_chars": len(text),
    }
