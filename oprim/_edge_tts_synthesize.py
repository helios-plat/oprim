"""oprim.edge_tts_synthesize — 多语言云 TTS 旁白合成(microsoft edge-tts)。

edge-tts(微软 Edge 神经语音)免费、无需模型、不占 GPU、30+ 语言、中文音色自然,与
`vibevoice_synthesize`(本地推理)并列作 audio 原语。逐行合成 mp3 → ffmpeg 合并重采样为
统一 WAV(pcm_s16le / 44100 / mono,装配器期望格式)。按每行文本自动选中/英音色(含 CJK
→ 中文),天然支持多语言混排。

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.edge_tts_synthesize import edge_tts_synthesize, Line
    >>> out = asyncio.run(edge_tts_synthesize(
    ...     script=[Line(text="你好世界"), Line(text="hello world")],
    ...     output_path=Path("narration.wav"),
    ... ))

Raises:
    ValueError: script 为空。
    EdgeTtsSetupError: edge-tts 未安装(需 `pip install oprim[tts]`)。
    EdgeTtsError: 全部分段合成失败,或 ffmpeg 合并失败。
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oprim._config import cfg

logger = logging.getLogger(__name__)


class EdgeTtsError(Exception):
    """edge-tts synthesis or concat failed."""


class EdgeTtsSetupError(EdgeTtsError):
    """edge-tts package not installed."""


@runtime_checkable
class Line(Protocol):
    """Minimal protocol for a narration line (compatible with oskill SpeakerLine)."""

    text: str


# 语言 → 音色映射;zh/en 可经 env(EDGE_TTS_VOICE_ZH/EN)覆盖,未命中回退中/英。
_LANG_STATIC: dict[str, str] = {
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _voice_for(text: str, language: str | None) -> str:
    """选音色:优先显式 language;否则按文本是否含 CJK 自动判定(中/英)。"""
    voice_zh = cfg.get("EDGE_TTS_VOICE_ZH", "zh-CN-XiaoxiaoNeural")
    voice_en = cfg.get("EDGE_TTS_VOICE_EN", "en-US-AriaNeural")
    if language:
        base = language.split("-")[0].lower()
        if base == "zh":
            return voice_zh
        if base == "en":
            return voice_en
        if base in _LANG_STATIC:
            return _LANG_STATIC[base]
    return voice_zh if _CJK_RE.search(text or "") else voice_en


async def _default_synth(text: str, voice: str, out_mp3: Path) -> None:
    """默认单行合成:edge_tts.Communicate → 保存 mp3。"""
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(out_mp3))


async def edge_tts_synthesize(
    *,
    config: dict[str, Any] | None = None,
    script: list[Any],
    output_path: Path,
    language: str | None = None,
    watermark: bool = False,  # 接口兼容占位;edge-tts 无水印概念
    _synth_fn: Any = None,
    _ffmpeg_fn: Any = None,
    **_kwargs: Any,
) -> Path:
    """把 script 各行旁白合成为单个 WAV(output_path)。

    Args:
        config: 覆盖 dict;language 缺省时取 config["language"]。
        script: 每行需有 .text(必需);.speaker_id 可选(与 vibevoice 一致)。
        output_path: 产物 WAV 落盘路径。
        language: 显式语言(如 "zh"/"en");缺省按文本 CJK 自动判定。
        watermark: 接口兼容占位(edge-tts 无水印)。
        _synth_fn / _ffmpeg_fn: 测试注入钩子(默认走 edge_tts / obase.ffmpeg)。

    Returns:
        output_path。

    Raises:
        ValueError: 空 script。
        EdgeTtsSetupError: edge-tts 未安装。
        EdgeTtsError: 全段失败或 ffmpeg 合并失败。
    """
    cfg_dict = config or {}
    language = language or cfg_dict.get("language")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [getattr(ln, "text", str(ln)) for ln in script]
    lines = [t.strip() for t in lines if t and t.strip()]
    if not lines:
        raise ValueError("edge_tts_synthesize: empty script")

    synth = _synth_fn or _default_synth
    if _synth_fn is None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise EdgeTtsSetupError("edge-tts not installed — run `pip install oprim[tts]`") from e

    with tempfile.TemporaryDirectory(prefix="edge_tts_") as td:
        tmp = Path(td)
        parts: list[Path] = []
        for i, text in enumerate(lines):
            voice = _voice_for(text, language)
            seg = tmp / f"seg_{i:04d}.mp3"
            try:
                await synth(text, voice, seg)
            except Exception as e:  # 单行失败不应毁掉整段旁白
                logger.warning("edge-tts segment %d failed (voice=%s): %s", i, voice, e)
                continue
            if seg.exists() and seg.stat().st_size > 0:
                parts.append(seg)

        if not parts:
            raise EdgeTtsError("edge_tts_synthesize: all segments failed")

        # 单段直接转码;多段 concat filter 统一为 WAV(避免 mp3 边界爆音)。
        if len(parts) == 1:
            args = ["-i", str(parts[0]), "-ar", "44100", "-ac", "1", str(output_path)]
        else:
            inputs: list[str] = []
            for p in parts:
                inputs += ["-i", str(p)]
            n = len(parts)
            filt = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
            args = [
                *inputs,
                "-filter_complex",
                filt,
                "-map",
                "[a]",
                "-ar",
                "44100",
                "-ac",
                "1",
                str(output_path),
            ]

        ffmpeg_run = _ffmpeg_fn
        if ffmpeg_run is None:
            from obase.ffmpeg import FFmpegError
            from obase.ffmpeg import run as ffmpeg_run

            try:
                await ffmpeg_run(args=args, expected_output=output_path)
            except FFmpegError as e:
                raise EdgeTtsError(f"ffmpeg concat failed: {e}") from e
        else:
            await ffmpeg_run(args=args, expected_output=output_path)

    logger.info("edge-tts: saved %d segment(s) to %s", len(parts), output_path)
    return output_path
