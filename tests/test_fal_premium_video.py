"""Tests for oprim veo3 / kling_v2 / hailuo 高写实视频原语 (payload 构造 — ≥6 tests).

均 patch 掉底层 fal_queue_generate,只验证各原语构造的 endpoint + payload 正确
(不触网;端点连通性/产物由 fal_queue_generate 自身测试与 e2e 覆盖)。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from oprim import hailuo_generate, kling_v2_generate, veo3_generate


class _Capture:
    """记录传给 fal_queue_generate 的 endpoint/payload。"""

    def __init__(self) -> None:
        self.endpoint: str | None = None
        self.payload: dict = {}

    def patch(self, module: str) -> object:
        async def _fake(*, endpoint: str, payload: dict, output_path: Path, **_: object) -> Path:
            self.endpoint = endpoint
            self.payload = payload
            return output_path

        return patch(f"{module}.fal_queue_generate", new=AsyncMock(side_effect=_fake))


class TestVeo3:
    async def test_endpoint_and_defaults(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(prompt="a chef", output_path=tmp_path / "v.mp4")
        assert cap.endpoint == "fal-ai/veo3/fast"
        assert cap.payload["aspect_ratio"] == "9:16"
        assert cap.payload["duration"] == "8s"
        assert cap.payload["generate_audio"] is True
        assert cap.payload["resolution"] == "720p"

    async def test_negative_prompt_only_when_set(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(prompt="p", output_path=tmp_path / "v.mp4")
        assert "negative_prompt" not in cap.payload  # 空则不下发
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(
                prompt="p", output_path=tmp_path / "v.mp4", negative_prompt="blurry"
            )
        assert cap.payload["negative_prompt"] == "blurry"

    async def test_aspect_ratio_explicit_and_derived(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(prompt="p", output_path=tmp_path / "v.mp4", aspect_ratio="16:9")
        assert cap.payload["aspect_ratio"] == "16:9"
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(
                prompt="p", output_path=tmp_path / "v.mp4", aspect_ratio="bad", size=(1080, 1920)
            )
        assert cap.payload["aspect_ratio"] == "9:16"  # 竖屏推导

    async def test_duration_formatting(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._veo3_generate"):
            await veo3_generate(prompt="p", output_path=tmp_path / "v.mp4", duration_s=6)
        assert cap.payload["duration"] == "6s"


class TestKlingV2:
    async def test_endpoint_and_payload(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._kling_v2_generate"):
            await kling_v2_generate(prompt="a dancer", output_path=tmp_path / "v.mp4")
        assert cap.endpoint == "fal-ai/kling-video/v2/master/text-to-video"
        assert cap.payload["duration"] == "5"
        assert cap.payload["cfg_scale"] == 0.5
        assert cap.payload["negative_prompt"]  # 空则用内置默认

    async def test_custom_negative_prompt(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._kling_v2_generate"):
            await kling_v2_generate(
                prompt="p", output_path=tmp_path / "v.mp4", negative_prompt="ugly"
            )
        assert cap.payload["negative_prompt"] == "ugly"


class TestHailuo:
    async def test_endpoint_and_payload(self, tmp_path: Path) -> None:
        cap = _Capture()
        with cap.patch("oprim._hailuo_generate"):
            await hailuo_generate(prompt="a cat", output_path=tmp_path / "v.mp4")
        assert cap.endpoint == "fal-ai/minimax/hailuo-02/standard/text-to-video"
        assert cap.payload["duration"] == "6"
        assert cap.payload["resolution"] == "768P"
        assert cap.payload["prompt_optimizer"] is True
