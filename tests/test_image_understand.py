"""Tests for oprim.image_understand."""

from __future__ import annotations

from pathlib import Path

import pytest

from obase import ProviderRegistry
from oprim.image_understand import ImageUnderstandError, image_understand


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestImageUnderstand:
    async def test_normal_success(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        async def _vlm(**kw: object) -> str:
            return "A cat sitting on a table"

        ProviderRegistry.register(category="vlm", name="mock", fn=_vlm)
        result = await image_understand(
            provider="mock", image_path=img, prompt="Describe this"
        )
        assert "cat" in result

    async def test_image_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ImageUnderstandError, match="not found"):
            await image_understand(
                provider="mock", image_path=tmp_path / "missing.jpg", prompt="Describe"
            )

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        with pytest.raises(ImageUnderstandError, match="VLM provider not found"):
            await image_understand(provider="nope", image_path=img, prompt="Describe")

    async def test_timeout_raises(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        async def _timeout(**kw: object) -> str:
            raise TimeoutError("timed out")

        ProviderRegistry.register(category="vlm", name="slow", fn=_timeout)
        with pytest.raises(ImageUnderstandError, match="VLM call failed"):
            await image_understand(provider="slow", image_path=img, prompt="Describe")

    async def test_empty_prompt_raises(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        with pytest.raises(ImageUnderstandError, match="prompt must not be empty"):
            await image_understand(provider="mock", image_path=img, prompt="")
