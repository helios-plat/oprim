"""oprim.edge_tts_word_boundary — 单文本 edge-tts 合成 + 词级 WordBoundary。

3O SPEC §2 Task 2.2 原子操作:取代 hevi/explainer/voiceover.py 的自写
_synthesize(同一份 edge_tts.Communicate stream + WordBoundary 逻辑)。
返回值 start/end 已从 WordBoundary 原始 100ns 单位换算为秒。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EdgeTtsWordBoundaryError(Exception):
    """edge-tts word-boundary synthesis failed."""


async def edge_tts_word_boundary(
    text: str,
    voice: str,
    *,
    rate: str | None = None,
    pitch: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """合成单段文本并返回音频文件路径 + 词级时间戳。

    Args:
        text: 待合成文本。
        voice: edge-tts 音色 ID(如 "zh-CN-XiaoxiaoNeural")。
        rate: 语速覆盖,如 "-10%"。
        pitch: 音高覆盖,如 "+0Hz"。
        output_path: 产物落盘路径;None 时写入临时目录。

    Returns:
        {"audio_path": Path, "words": [{"text", "start", "end"}, ...]}
        (start/end 单位:秒)。

    Raises:
        EdgeTtsWordBoundaryError: edge-tts 未安装或合成无音频输出。
    """
    import edge_tts  # noqa: PLC0415

    out = Path(output_path) if output_path else None
    if out is None:
        import tempfile  # noqa: PLC0415

        out = Path(tempfile.mkdtemp(prefix="edge_tts_wb_")) / "speech.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {"boundary": "WordBoundary"}
    if rate is not None:
        kwargs["rate"] = rate
    if pitch is not None:
        kwargs["pitch"] = pitch
    communicate = edge_tts.Communicate(text, voice=voice, **kwargs)

    words: list[dict[str, Any]] = []
    audio_bytes = bytearray()
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append(
                    {
                        "text": chunk["text"],
                        "start": chunk["offset"] / 1e7,
                        "end": (chunk["offset"] + chunk["duration"]) / 1e7,
                    }
                )
    except Exception as exc:  # 网络/限流瞬时故障
        raise EdgeTtsWordBoundaryError(f"edge-tts stream failed: {exc}") from exc

    if not audio_bytes:
        raise EdgeTtsWordBoundaryError("edge-tts produced no audio data")

    out.write_bytes(bytes(audio_bytes))
    return {"audio_path": out, "words": words}
