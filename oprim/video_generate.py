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
from obase.exceptions import ProviderNotFoundError


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
            )
        except WanCloudError as exc:
            raise VideoGenError(f"wan_cloud generation failed: {exc}") from exc

    try:
        gen_fn = ProviderRegistry.get(category="video_gen", name=provider)
    except ProviderNotFoundError as exc:
        raise VideoGenProviderNotFoundError(
            f"Video generation provider not found: {provider!r}"
        ) from exc

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
