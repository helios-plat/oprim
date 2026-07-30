"""B2/B3/B6: ltx2 negative_prompt + 轮询总超时 + video_generate 内建 fal dispatch。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._ltx2_cloud_generate import Ltx2CloudError, ltx2_cloud_generate


def _capturing_client(captured: dict) -> MagicMock:
    async def _post(url: str, json: dict, headers: dict) -> MagicMock:
        captured.update(json)
        r = MagicMock(status_code=200)
        r.json.return_value = {"video": {"url": "http://cdn/v.mp4"}}
        return r

    async def _get(url: str, **_: Any) -> MagicMock:
        r = MagicMock(status_code=200)
        r.content = b"\x00" * 64
        return r

    client = MagicMock()
    client.post = _post
    client.get = _get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestLtx2NegativePrompt:
    async def test_negative_prompt_in_payload(self, tmp_path: Path) -> None:
        """B2: 传 negative_prompt → 下发到 fal payload。"""
        captured: dict = {}
        with patch("httpx.AsyncClient", return_value=_capturing_client(captured)):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="t2v",
                prompt="p",
                duration_s=5.0,
                resolution=(256, 256),
                negative_prompt="blurry, extra fingers",
                output_path=tmp_path / "o.mp4",
            )
        assert captured["negative_prompt"] == "blurry, extra fingers"

    async def test_negative_prompt_omitted_when_empty(self, tmp_path: Path) -> None:
        """空 negative_prompt → 不下发(向后兼容)。"""
        captured: dict = {}
        with patch("httpx.AsyncClient", return_value=_capturing_client(captured)):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="t2v",
                prompt="p",
                duration_s=5.0,
                resolution=(256, 256),
                output_path=tmp_path / "o.mp4",
            )
        assert "negative_prompt" not in captured

    async def test_poll_timeout_raises(self, tmp_path: Path) -> None:
        """B3: 轮询超过 _POLL_TIMEOUT_S 仍未完成 → Ltx2CloudError timeout。"""
        post = MagicMock(status_code=200)
        post.json.return_value = {"request_id": "abc"}
        client = MagicMock()
        client.post = AsyncMock(return_value=post)
        client.get = AsyncMock(
            return_value=MagicMock(
                status_code=200, json=MagicMock(return_value={"status": "IN_PROGRESS"})
            )
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("oprim._ltx2_cloud_generate._POLL_TIMEOUT_S", -1.0),
            pytest.raises(Ltx2CloudError, match="timeout"),
        ):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="t2v",
                prompt="p",
                duration_s=5.0,
                resolution=(256, 256),
                output_path=tmp_path / "o.mp4",
            )


class TestVideoGenerateFalDispatch:
    async def test_dispatches_veo3_with_size_and_duration(self, tmp_path: Path) -> None:
        """B6: video_generate(provider='veo3') 内建 dispatch 到 veo3_generate,带 size/duration。"""
        from oprim._video_generate import video_generate

        captured: dict = {}

        async def _fake_veo3(*, prompt: str, output_path: Path, **kw: Any) -> Path:
            captured.update(kw)
            captured["prompt"] = prompt
            output_path.write_bytes(b"\x00" * 64)
            return output_path

        with patch("oprim._veo3_generate.veo3_generate", new=AsyncMock(side_effect=_fake_veo3)):
            res = await video_generate(
                provider="veo3",
                prompt="a chef plating",
                output_path=tmp_path / "o.mp4",
                width=1080,
                height=1920,
                duration_s=8,
            )
        assert res == tmp_path / "o.mp4"
        assert captured["size"] == (1080, 1920)
        assert captured["duration_s"] == 8
        assert captured["prompt"] == "a chef plating"
