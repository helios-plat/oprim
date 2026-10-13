"""oprim.video_generate — Video generation via provider injection.

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.video_generate import video_generate
    >>> result = asyncio.run(video_generate(
    ...     provider="wan2.2",
    ...     prompt="A cat walking on the moon",
    ...     output_path=Path("generated.mp4"),
    ... ))

Raises:
    VideoGenError: Generation failed.
    VideoGenProviderNotFoundError: Provider not registered.
"""

from __future__ import annotations

from pathlib import Path

from obase import ProviderRegistry


class VideoGenError(Exception):
    """Video generation failed."""


class VideoGenProviderNotFoundError(Exception):
    """Video generation provider not registered."""


async def video_generate(
    *,
    provider: str,
    prompt: str,
    reference_image: Path | None = None,
    duration_s: float = 5.0,
    width: int = 1080,
    height: int = 1920,
    output_path: Path,
    timeout_s: float = 600.0,
    fps: int = 24,
    bitrate_kbps: int | None = None,
) -> Path:
    """Generate video using a registered provider.

    Args:
        provider: Provider name registered in obase.ProviderRegistry (category='video_gen').
        prompt: Text prompt for generation.
        reference_image: Optional reference image for guided generation.
        duration_s: Target video duration in seconds.
        width: Output width in pixels.
        height: Output height in pixels.
        output_path: Destination file path.
        timeout_s: Timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        VideoGenProviderNotFoundError: Provider not registered.
        VideoGenError: Generation failed or output not produced.

    Example:
        >>> await video_generate(provider="stub", prompt="test", output_path=Path("out.mp4"))
    """
    # Built-in wan_cloud dispatch — no ProviderRegistry registration needed
    if provider == "wan_cloud":
        from oprim._config import cfg
        from oprim._providers.wan_cloud import WanCloudError
        from oprim._providers.wan_cloud import invoke as _wan_invoke

        api_key: str = cfg.get("DASHSCOPE_API_KEY", "")  # type: ignore[assignment]
        if not api_key:
            raise VideoGenError("DASHSCOPE_API_KEY not configured for wan_cloud")
        _mode = "i2v" if reference_image is not None else "t2v"
        try:
            return await _wan_invoke(
                mode=_mode,
                prompt=prompt,
                reference_image=reference_image,
                output_path=output_path,
                api_key=api_key,
                fps=fps,
                bitrate_kbps=bitrate_kbps,
            )
        except WanCloudError as exc:
            raise VideoGenError(f"wan_cloud generation failed: {exc}") from exc

    # B6: 内建 fal 高写实 providers(veo3/kling_v2/hailuo)—— 与 wan_cloud 并列,免 registry
    # 注册,使 A 组 fal 视频原语可经 video_generate 统一到达(t2v;size→朝向推导)。
    if provider in ("veo3", "kling_v2", "hailuo"):
        from oprim._fal_queue_generate import FalQueueError
        from oprim._hailuo_generate import hailuo_generate
        from oprim._kling_v2_generate import kling_v2_generate
        from oprim._veo3_generate import veo3_generate

        _fal = {
            "veo3": veo3_generate,
            "kling_v2": kling_v2_generate,
            "hailuo": hailuo_generate,
        }
        try:
            return await _fal[provider](
                prompt=prompt,
                output_path=output_path,
                duration_s=duration_s,
                size=(width, height),
            )
        except FalQueueError as exc:
            raise VideoGenError(f"{provider} generation failed: {exc}") from exc

    if not ProviderRegistry.has("video_gen", provider):
        raise VideoGenProviderNotFoundError(f"Video generation provider not found: {provider!r}")
    gen_fn = ProviderRegistry.get().generic("video_gen", provider)

    try:
        await gen_fn(
            prompt=prompt,
            reference_image=reference_image,
            duration_s=duration_s,
            width=width,
            height=height,
            output_path=output_path,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        if isinstance(exc, VideoGenProviderNotFoundError):
            raise
        raise VideoGenError(f"Video generation failed: {exc}") from exc

    if not output_path.exists():
        raise VideoGenError(f"Provider did not produce output: {output_path}")

    return output_path
