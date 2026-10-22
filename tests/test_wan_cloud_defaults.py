"""B1: wan_cloud 修正端点/模型默认 + 吸收 video_generate 多传的 fps/bitrate。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from oprim._providers.wan_cloud import invoke

_SLEEP = "oprim._providers.wan_cloud.asyncio.sleep"


def _wan_client(captured: dict) -> MagicMock:
    async def _post(url: str, json: dict, headers: dict) -> MagicMock:
        captured["url"] = url
        captured["payload"] = json
        r = MagicMock(status_code=200)
        r.json.return_value = {"output": {"task_id": "t1"}}
        return r

    poll = MagicMock(status_code=200)
    poll.json.return_value = {
        "output": {"task_status": "SUCCEEDED", "video_url": "http://cdn/v.mp4"}
    }
    dl = MagicMock(status_code=200)
    dl.content = b"\x00" * 64

    client = MagicMock()
    client.post = _post
    client.get = AsyncMock(side_effect=[poll, dl])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestWanCloudB1:
    async def test_defaults_use_video_synthesis_and_wanx21(self, tmp_path: Path) -> None:
        captured: dict = {}
        with (
            patch("httpx.AsyncClient", return_value=_wan_client(captured)),
            patch(_SLEEP, new_callable=AsyncMock),
        ):
            await invoke(
                mode="t2v",
                prompt="p",
                reference_image=None,
                output_path=tmp_path / "o.mp4",
                api_key="k",
            )
        assert captured["url"].endswith("video-generation/video-synthesis")
        assert captured["payload"]["model"] == "wanx2.1-t2v-turbo"

    async def test_absorbs_extra_kwargs_from_video_generate(self, tmp_path: Path) -> None:
        """video_generate 会多传 fps/bitrate_kbps —— 此前 TypeError,现被 **_ignored 吸收。"""
        captured: dict = {}
        with (
            patch("httpx.AsyncClient", return_value=_wan_client(captured)),
            patch(_SLEEP, new_callable=AsyncMock),
        ):
            res: Any = await invoke(
                mode="t2v",
                prompt="p",
                reference_image=None,
                output_path=tmp_path / "o.mp4",
                api_key="k",
                fps=24,
                bitrate_kbps=1000,
            )
        assert res == tmp_path / "o.mp4"
