"""Tests for oprim.video_generate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from obase import ProviderRegistry
from oprim.video_generate import VideoGenError, VideoGenProviderNotFoundError, video_generate


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure clean provider registry for each test."""
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


def _register_stub_provider(tmp_path: Path) -> None:
    """Register a stub video_gen provider that creates a file."""

    async def _stub(
        *,
        prompt: str,
        reference_image: Path | None,
        duration_s: float,
        width: int,
        height: int,
        output_path: Path,
        timeout_s: float,
    ) -> None:
        output_path.write_bytes(b"\x00" * 128)

    ProviderRegistry.register(category="video_gen", name="stub", fn=_stub)


class TestVideoGenerate:
    async def test_provider_registered_success(self, tmp_path: Path) -> None:
        _register_stub_provider(tmp_path)
        out = tmp_path / "generated.mp4"
        result = await video_generate(
            provider="stub", prompt="A cat on the moon", output_path=out
        )
        assert result == out
        assert out.exists()

    async def test_provider_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VideoGenProviderNotFoundError, match="not found"):
            await video_generate(
                provider="nonexistent", prompt="test", output_path=tmp_path / "out.mp4"
            )

    async def test_mock_provider_called_with_params(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        async def _capture(**kw: object) -> None:
            captured.update(kw)
            Path(str(kw["output_path"])).write_bytes(b"\x00")

        ProviderRegistry.register(category="video_gen", name="mock", fn=_capture)
        out = tmp_path / "out.mp4"
        await video_generate(
            provider="mock",
            prompt="test prompt",
            duration_s=10.0,
            width=720,
            height=1280,
            output_path=out,
        )
        assert captured["prompt"] == "test prompt"
        assert captured["duration_s"] == 10.0
        assert captured["width"] == 720

    async def test_provider_timeout_raises(self, tmp_path: Path) -> None:
        async def _timeout(**kw: object) -> None:
            raise TimeoutError("provider timed out")

        ProviderRegistry.register(category="video_gen", name="slow", fn=_timeout)
        with pytest.raises(VideoGenError, match="failed"):
            await video_generate(
                provider="slow", prompt="test", output_path=tmp_path / "out.mp4"
            )

    async def test_output_not_produced_raises(self, tmp_path: Path) -> None:
        async def _noop(**kw: object) -> None:
            pass  # Does not create output file

        ProviderRegistry.register(category="video_gen", name="noop", fn=_noop)
        with pytest.raises(VideoGenError, match="did not produce output"):
            await video_generate(
                provider="noop", prompt="test", output_path=tmp_path / "out.mp4"
            )

    async def test_reference_image_passed(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        async def _capture(**kw: object) -> None:
            captured.update(kw)
            Path(str(kw["output_path"])).write_bytes(b"\x00")

        ProviderRegistry.register(category="video_gen", name="ref", fn=_capture)
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"\x89PNG")
        out = tmp_path / "out.mp4"
        await video_generate(
            provider="ref", prompt="test", reference_image=ref, output_path=out
        )
        assert captured["reference_image"] == ref
