"""Tests for oprim.image_generate."""

from __future__ import annotations

from pathlib import Path

import pytest

from obase import ProviderRegistry
from oprim.image_generate import ImageGenError, ImageGenRateLimitError, image_generate


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestImageGenerate:
    async def test_provider_success(self, tmp_path: Path) -> None:
        async def _gen(**kw: object) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x89PNG")

        ProviderRegistry.register(category="image_gen", name="mock", fn=_gen)
        out = tmp_path / "img.png"
        result = await image_generate(provider="mock", prompt="cat", output_path=out)
        assert result == out
        assert out.exists()

    async def test_rate_limit_429(self, tmp_path: Path) -> None:
        async def _rate(**kw: object) -> None:
            raise RuntimeError("429 Too Many Requests")

        ProviderRegistry.register(category="image_gen", name="limited", fn=_rate)
        with pytest.raises(ImageGenRateLimitError, match="Rate limited"):
            await image_generate(
                provider="limited", prompt="cat", output_path=tmp_path / "out.png"
            )

    async def test_timeout_raises(self, tmp_path: Path) -> None:
        async def _timeout(**kw: object) -> None:
            raise TimeoutError("timed out")

        ProviderRegistry.register(category="image_gen", name="slow", fn=_timeout)
        with pytest.raises(ImageGenError, match="failed"):
            await image_generate(
                provider="slow", prompt="cat", output_path=tmp_path / "out.png"
            )

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ImageGenError, match="not found"):
            await image_generate(
                provider="nope", prompt="cat", output_path=tmp_path / "out.png"
            )

    async def test_seed_passed(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        async def _cap(**kw: object) -> None:
            captured.update(kw)
            Path(str(kw["output_path"])).write_bytes(b"\x00")

        ProviderRegistry.register(category="image_gen", name="s", fn=_cap)
        out = tmp_path / "seeded.png"
        await image_generate(provider="s", prompt="cat", output_path=out, seed=42)
        assert captured["seed"] == 42

    async def test_empty_prompt_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageGenError, match="prompt must not be empty"):
            await image_generate(provider="x", prompt="", output_path=tmp_path / "out.png")

    async def test_output_not_produced(self, tmp_path: Path) -> None:
        async def _noop(**kw: object) -> None:
            pass

        ProviderRegistry.register(category="image_gen", name="noop", fn=_noop)
        with pytest.raises(ImageGenError, match="did not produce"):
            await image_generate(
                provider="noop", prompt="cat", output_path=tmp_path / "out.png"
            )
