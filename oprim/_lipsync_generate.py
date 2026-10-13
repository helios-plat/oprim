"""oprim.lipsync_generate — Unified dispatcher for portrait/video-driven lip-sync.

A4: unifies the local subprocess providers (duix / sadtalker / musetalk /
longcat_avatar) behind one interface. All four are currently
portrait-driven only (portrait_image + audio -> talking-head video); none
of them re-lip-sync an existing video (source_video + audio -> dubbed
video) — that mode has no backend yet and raises LipsyncNotImplementedError.

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.lipsync_generate import lipsync_generate
    >>> result = asyncio.run(lipsync_generate(
    ...     provider="duix", portrait_image=Path("face.png"),
    ...     audio_path=Path("speech.wav"), output_path=Path("out.mp4"),
    ... ))

Raises:
    LipsyncGenError: Generation failed.
    LipsyncSetupError: Vendor binary/model files not found.
    LipsyncNotImplementedError: Provider has no implementation yet (video-driven).
"""

from __future__ import annotations

from pathlib import Path


class LipsyncGenError(Exception):
    """Lip-sync generation failed."""


class LipsyncSetupError(LipsyncGenError):
    """Vendor binary or model files not found."""


class LipsyncNotImplementedError(LipsyncGenError):
    """Provider recognized but has no backend implementation yet."""


# 肖像驱动(portrait_image + audio -> talking head):duix 走既有 avatar_generate
# 子进程隔离路径;sadtalker/musetalk/longcat_avatar 走 oprim._providers 子进程包装。
_PORTRAIT_DRIVEN = ("duix", "sadtalker", "musetalk", "longcat_avatar")
# 视频驱动(source_video + audio -> 重对口型):无本地/云后端实现,占位标记。
_VIDEO_DRIVEN = ("sync", "latentsync")


async def lipsync_generate(
    *,
    provider: str,
    audio_path: Path,
    output_path: Path,
    portrait_image: Path | None = None,
    source_video: Path | None = None,
    vendor_dir: Path | None = None,
    fps: int = 25,
    timeout_s: float = 600.0,
) -> Path:
    """Generate a lip-synced video via a portrait or video-driven provider.

    Args:
        provider: One of "duix", "sadtalker", "musetalk", "longcat_avatar"
            (portrait-driven) or "sync", "latentsync" (video-driven, not
            yet implemented).
        audio_path: Driving audio file to lip-sync to.
        output_path: Destination video file.
        portrait_image: Face/portrait image (required for portrait-driven providers).
        source_video: Existing video to re-lip-sync (required for video-driven providers).
        vendor_dir: Local model/inference-script directory (sadtalker/musetalk/longcat_avatar).
        fps: Output video frame rate.
        timeout_s: Timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        LipsyncGenError: On validation failure or generation error.
        LipsyncSetupError: Vendor binary/model files not available.
        LipsyncNotImplementedError: Video-driven providers have no backend yet.
    """
    if not audio_path.exists():
        raise LipsyncGenError(f"Audio file not found: {audio_path}")

    if provider in _VIDEO_DRIVEN:
        raise LipsyncNotImplementedError(
            f"Video-driven lip-sync provider {provider!r} has no implementation yet "
            "(no local/cloud backend exists for source_video re-lip-sync)."
        )

    if provider not in _PORTRAIT_DRIVEN:
        raise LipsyncGenError(f"Unknown lip-sync provider: {provider!r}")

    if portrait_image is None:
        raise LipsyncGenError(f"portrait_image is required for provider {provider!r}")
    if not portrait_image.exists():
        raise LipsyncGenError(f"Portrait image not found: {portrait_image}")

    if provider == "duix":
        from oprim._avatar_generate import AvatarGenError, AvatarSetupError, avatar_generate

        try:
            return await avatar_generate(
                provider="duix",
                portrait_image=portrait_image,
                audio_path=audio_path,
                output_path=output_path,
                fps=fps,
                timeout_s=timeout_s,
            )
        except AvatarSetupError as exc:
            raise LipsyncSetupError(str(exc)) from exc
        except AvatarGenError as exc:
            raise LipsyncGenError(str(exc)) from exc

    if vendor_dir is None:
        raise LipsyncGenError(f"vendor_dir is required for provider {provider!r}")

    if provider == "sadtalker":
        from oprim._providers.sadtalker import SadTalkerError, SadTalkerSetupError
        from oprim._providers.sadtalker import invoke as _invoke

        setup_err, gen_err = SadTalkerSetupError, SadTalkerError
    elif provider == "musetalk":
        from oprim._providers.musetalk import MuseTalkError, MuseTalkSetupError
        from oprim._providers.musetalk import invoke as _invoke

        setup_err, gen_err = MuseTalkSetupError, MuseTalkError
    else:  # longcat_avatar
        from oprim._providers.longcat_avatar import LongCatAvatarError, LongCatAvatarSetupError
        from oprim._providers.longcat_avatar import invoke_local as _invoke

        setup_err, gen_err = LongCatAvatarSetupError, LongCatAvatarError

    try:
        result = await _invoke(
            portrait_image=portrait_image,
            audio_path=audio_path,
            output_path=output_path,
            vendor_dir=vendor_dir,
            fps=fps,
            timeout_s=timeout_s,
        )
    except setup_err as exc:
        raise LipsyncSetupError(str(exc)) from exc
    except gen_err as exc:
        raise LipsyncGenError(str(exc)) from exc

    if not output_path.exists():
        raise LipsyncGenError(f"Provider did not produce output: {output_path}")

    return result
