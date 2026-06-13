"""Tests for oprim.video_generate wan_cloud provider (M2 — ≥4 + regression)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim.video_generate import VideoGenError, video_generate


class TestVideoGenerateWanCloud:
    async def test_wan_cloud_t2v_success(self, tmp_path: Path) -> None:
        """wan_cloud t2v: no reference_image → mode t2v, output produced."""
        out = tmp_path / "wan.mp4"

        async def _mock_invoke(*, mode: str, prompt: str, reference_image: object,
                                output_path: Path, api_key: str, **kw: object) -> Path:
            assert mode == "t2v"
            assert reference_image is None
            output_path.write_bytes(b"\x00" * 64)
            return output_path

        with (
            patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}),
            patch("oprim._providers.wan_cloud.invoke", new=AsyncMock(side_effect=_mock_invoke)),
        ):
            result = await video_generate(
                provider="wan_cloud",
                prompt="A flowing river",
                output_path=out,
            )

        assert result == out
        assert out.exists()

    async def test_wan_cloud_i2v_success(self, tmp_path: Path) -> None:
        """wan_cloud i2v: reference_image → mode i2v forwarded to invoke."""
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"\x89PNG" + b"\x00" * 60)
        out = tmp_path / "wan_i2v.mp4"
        captured: dict = {}

        async def _mock_invoke(*, mode: str, reference_image: object,
                                output_path: Path, **kw: object) -> Path:
            captured["mode"] = mode
            captured["reference_image"] = reference_image
            output_path.write_bytes(b"\x00" * 64)
            return output_path

        with (
            patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}),
            patch("oprim._providers.wan_cloud.invoke", new=AsyncMock(side_effect=_mock_invoke)),
        ):
            await video_generate(
                provider="wan_cloud",
                prompt="Animate",
                reference_image=ref,
                output_path=out,
            )

        assert captured.get("mode") == "i2v"
        assert captured.get("reference_image") == ref

    async def test_wan_cloud_poll_success(self, tmp_path: Path) -> None:
        """wan_cloud invoke called exactly once per video_generate call."""
        out = tmp_path / "poll.mp4"
        call_count = 0

        async def _mock_invoke(*, output_path: Path, **kw: object) -> Path:
            nonlocal call_count
            call_count += 1
            output_path.write_bytes(b"\x00" * 32)
            return output_path

        with (
            patch.dict("os.environ", {"DASHSCOPE_API_KEY": "k"}),
            patch("oprim._providers.wan_cloud.invoke", new=AsyncMock(side_effect=_mock_invoke)),
        ):
            await video_generate(provider="wan_cloud", prompt="poll", output_path=out)

        assert call_count == 1

    async def test_wan_cloud_api_failure_wraps_video_gen_error(self, tmp_path: Path) -> None:
        """WanCloudError → VideoGenError with descriptive message."""
        from oprim._providers.wan_cloud import WanCloudError

        async def _fail(**kw: object) -> Path:
            raise WanCloudError("API error 429: rate limited")

        with (
            patch.dict("os.environ", {"DASHSCOPE_API_KEY": "k"}),
            patch("oprim._providers.wan_cloud.invoke", new=AsyncMock(side_effect=_fail)),
        ):
            with pytest.raises(VideoGenError, match="wan_cloud generation failed"):
                await video_generate(
                    provider="wan_cloud",
                    prompt="fail",
                    output_path=tmp_path / "out.mp4",
                )

    async def test_regression_existing_provider_not_broken(self, tmp_path: Path) -> None:
        """Existing ProviderRegistry providers still work after wan_cloud addition."""
        from obase import ProviderRegistry

        ProviderRegistry.clear()

        async def _stub(**kw: object) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 32)

        ProviderRegistry.register(category="video_gen", name="stub", fn=_stub)
        out = tmp_path / "stub.mp4"

        try:
            result = await video_generate(provider="stub", prompt="test", output_path=out)
            assert result == out
        finally:
            ProviderRegistry.clear()
