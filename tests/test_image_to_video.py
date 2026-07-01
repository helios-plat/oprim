"""Tests for oprim._image_to_video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from obase import ProviderRegistry
from oprim._image_to_video import (
    ImageToVideoError,
    ImageToVideoProviderNotFoundError,
    image_to_video,
)


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


@pytest.fixture()
def ref_image(tmp_path: Path) -> Path:
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\x00" * 20)
    return img


class TestImageToVideo:
    async def test_success(self, tmp_path: Path, ref_image: Path) -> None:
        out = tmp_path / "out.mp4"

        async def _gen(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 256)

        ProviderRegistry.register(category="image_to_video", name="mock_wan", fn=_gen)
        result = await image_to_video(
            provider="mock_wan",
            reference_image=ref_image,
            motion_prompt="pan left",
            output_path=out,
        )
        assert result == out
        assert out.exists()

    async def test_reference_image_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ImageToVideoError, match="Reference image not found"):
            await image_to_video(
                provider="mock",
                reference_image=tmp_path / "missing.png",
                motion_prompt="zoom",
                output_path=tmp_path / "out.mp4",
            )

    async def test_no_output_produced(self, tmp_path: Path, ref_image: Path) -> None:
        async def _noop(**kw: Any) -> None:
            pass

        ProviderRegistry.register(category="image_to_video", name="noop", fn=_noop)
        with pytest.raises(ImageToVideoError, match="did not produce"):
            await image_to_video(
                provider="noop",
                reference_image=ref_image,
                motion_prompt="zoom",
                output_path=tmp_path / "out.mp4",
            )

    async def test_provider_not_found(self, tmp_path: Path, ref_image: Path) -> None:
        """Regression: unregistered image_to_video provider raises ImageToVideoProviderNotFoundError."""
        with pytest.raises(ImageToVideoProviderNotFoundError, match="Provider not found"):
            await image_to_video(
                provider="nonexistent_i2v",
                reference_image=ref_image,
                motion_prompt="slow zoom",
                output_path=tmp_path / "out.mp4",
            )
