"""Tests for oprim.fal_queue_generate (fal 队列通用工具 — ≥6 tests)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim import FalQueueError, fal_queue_generate
from oprim._fal_queue_generate import _fal_aspect_ratio


def _resp(status_code: int = 200, *, json_val: object = None, content: bytes = b"") -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_val
    r.content = content
    r.text = str(json_val)
    return r


def _client(post_resp: MagicMock, get_responses: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=post_resp)
    client.get = AsyncMock(side_effect=get_responses)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


_SLEEP = "oprim._fal_queue_generate.asyncio.sleep"


class TestFalQueueGenerate:
    async def test_queue_lifecycle_success(self, tmp_path: Path) -> None:
        """submit → poll COMPLETED → fetch response → download → file written."""
        out = tmp_path / "v.mp4"
        post = _resp(json_val={"status_url": "http://fal/s", "response_url": "http://fal/r"})
        gets = [
            _resp(json_val={"status": "COMPLETED"}),
            _resp(json_val={"video": {"url": "http://cdn/v.mp4"}}),
            _resp(content=b"\x00" * 2048),
        ]
        with (
            patch("httpx.AsyncClient", return_value=_client(post, gets)),
            patch(_SLEEP, new_callable=AsyncMock),
        ):
            res = await fal_queue_generate(
                endpoint="fal-ai/veo3/fast",
                payload={"prompt": "x"},
                output_path=out,
                config={"FAL_API_KEY": "k"},
            )
        assert res == out
        assert out.read_bytes() == b"\x00" * 2048

    async def test_sync_response_no_status_url(self, tmp_path: Path) -> None:
        """submit 直接返回结果(无 status_url)→ 跳过轮询,直接下载。"""
        out = tmp_path / "v.mp4"
        post = _resp(json_val={"video": {"url": "http://cdn/v.mp4"}})
        gets = [_resp(content=b"\x01" * 4096)]
        with patch("httpx.AsyncClient", return_value=_client(post, gets)):
            res = await fal_queue_generate(
                endpoint="e", payload={}, output_path=out, config={"FAL_API_KEY": "k"}
            )
        assert res == out
        assert out.stat().st_size == 4096

    async def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        """无 FAL_API_KEY → FalQueueError,且不触网。"""
        with (
            patch("oprim._fal_queue_generate.cfg.get", return_value=""),
            pytest.raises(FalQueueError, match="FAL_API_KEY"),
        ):
            await fal_queue_generate(
                endpoint="e", payload={}, output_path=tmp_path / "v.mp4", config=None
            )

    async def test_job_failed_status_raises(self, tmp_path: Path) -> None:
        """轮询返回 FAILED → FalQueueError。"""
        post = _resp(json_val={"status_url": "http://fal/s", "response_url": "http://fal/r"})
        gets = [_resp(json_val={"status": "FAILED"})]
        with (
            patch("httpx.AsyncClient", return_value=_client(post, gets)),
            patch(_SLEEP, new_callable=AsyncMock),
            pytest.raises(FalQueueError, match="FAILED"),
        ):
            await fal_queue_generate(
                endpoint="e",
                payload={},
                output_path=tmp_path / "v.mp4",
                config={"FAL_API_KEY": "k"},
            )

    async def test_timeout_raises(self, tmp_path: Path) -> None:
        """超过 timeout_s 仍未完成 → FalQueueError timeout(治轮询无限挂)。"""
        post = _resp(json_val={"status_url": "http://fal/s", "response_url": "http://fal/r"})
        gets = [_resp(json_val={"status": "IN_PROGRESS"})]
        with (
            patch("httpx.AsyncClient", return_value=_client(post, gets)),
            pytest.raises(FalQueueError, match="timeout"),
        ):
            await fal_queue_generate(
                endpoint="e",
                payload={},
                output_path=tmp_path / "v.mp4",
                timeout_s=-1.0,
                config={"FAL_API_KEY": "k"},
            )

    async def test_submit_error_status_raises(self, tmp_path: Path) -> None:
        """fal submit 非 2xx → FalQueueError 带状态码。"""
        post = _resp(status_code=403, json_val=None)
        post.text = "Exhausted balance"
        with (
            patch("httpx.AsyncClient", return_value=_client(post, [])),
            pytest.raises(FalQueueError, match="403"),
        ):
            await fal_queue_generate(
                endpoint="e",
                payload={},
                output_path=tmp_path / "v.mp4",
                config={"FAL_API_KEY": "k"},
            )

    async def test_empty_output_raises(self, tmp_path: Path) -> None:
        """产物 <1024B → FalQueueError no/empty output。"""
        out = tmp_path / "v.mp4"
        post = _resp(json_val={"video": {"url": "http://cdn/v.mp4"}})
        gets = [_resp(content=b"\x00" * 10)]
        with (
            patch("httpx.AsyncClient", return_value=_client(post, gets)),
            pytest.raises(FalQueueError, match="empty"),
        ):
            await fal_queue_generate(
                endpoint="e", payload={}, output_path=out, config={"FAL_API_KEY": "k"}
            )

    def test_aspect_ratio_helper(self) -> None:
        """_fal_aspect_ratio:显式合法值直取;非法按 size 推导;缺省 9:16。"""
        assert _fal_aspect_ratio("16:9", {}) == "16:9"
        assert _fal_aspect_ratio("bogus", {"size": (1920, 1080)}) == "16:9"
        assert _fal_aspect_ratio("bogus", {"size": (512, 512)}) == "1:1"
        assert _fal_aspect_ratio(None, {}) == "9:16"
