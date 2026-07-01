"""Tests for oprim.ltx2_cloud_generate (M1 — ≥6 tests)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim.ltx2_cloud_generate import Ltx2CloudError, ltx2_cloud_generate


def _make_client(submit_resp: object, poll_resp: object | None, dl_bytes: bytes) -> MagicMock:
    """Build a mock httpx.AsyncClient context manager."""
    mock_dl = MagicMock()
    mock_dl.status_code = 200
    mock_dl.content = dl_bytes

    mock_submit = MagicMock()
    mock_submit.status_code = 200
    mock_submit.json.return_value = submit_resp

    get_responses = []
    if poll_resp is not None:
        mock_poll = MagicMock()
        mock_poll.status_code = 200
        mock_poll.json.return_value = poll_resp
        get_responses = [mock_poll, mock_dl]
    else:
        get_responses = [mock_dl]

    client = MagicMock()
    client.post = AsyncMock(return_value=mock_submit)
    client.get = AsyncMock(side_effect=get_responses)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestLtx2CloudGenerate:
    async def test_t2v_success(self, tmp_path: Path) -> None:
        """t2v: submit returns video_url directly, file written."""
        video_bytes = b"\x00" * 256
        submit_resp = {"video": {"url": "http://cdn/out.mp4"}}
        client = _make_client(submit_resp, None, video_bytes)
        out = tmp_path / "out.mp4"

        with patch("httpx.AsyncClient", return_value=client):
            result = await ltx2_cloud_generate(
                config={"FAL_API_KEY": "test-key"},
                mode="t2v",
                prompt="A sunset over the ocean",
                duration_s=5.0,
                resolution=(1280, 720),
                output_path=out,
            )

        assert result == out
        assert out.exists()
        assert out.read_bytes() == video_bytes

    async def test_i2v_success_with_reference(self, tmp_path: Path) -> None:
        """i2v: reference_image base64-encoded and sent in payload."""
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"\x89PNG\r\n" + b"\x00" * 50)
        out = tmp_path / "out.mp4"

        captured_payload: dict = {}

        async def _fake_post(url: str, json: dict, headers: dict) -> MagicMock:
            captured_payload.update(json)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"video": {"url": "http://cdn/out.mp4"}}
            return resp

        async def _fake_get(url: str, **_: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\x00" * 64
            return resp

        client = MagicMock()
        client.post = _fake_post
        client.get = _fake_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "key"},
                mode="i2v",
                prompt="Animate this image",
                reference_image=ref,
                duration_s=4.0,
                resolution=(512, 512),
                output_path=out,
            )

        assert "image_url" in captured_payload
        assert captured_payload["image_url"].startswith("data:image/png;base64,")

    async def test_audio_enabled_flag_in_payload(self, tmp_path: Path) -> None:
        """audio_enabled flag is propagated to payload."""
        captured: dict = {}

        async def _fake_post(url: str, json: dict, **_: object) -> MagicMock:
            captured.update(json)
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"video": {"url": "http://x/v.mp4"}}
            return r

        async def _fake_get(url: str, **_: object) -> MagicMock:
            r = MagicMock()
            r.status_code = 200
            r.content = b"\x00" * 64
            return r

        client = MagicMock()
        client.post = _fake_post
        client.get = _fake_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        out = tmp_path / "out.mp4"

        with patch("httpx.AsyncClient", return_value=client):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="t2v",
                prompt="test",
                duration_s=3.0,
                resolution=(256, 256),
                audio_enabled=False,
                output_path=out,
            )

        assert captured.get("enable_audio") is False

    async def test_poll_pending_then_completed(self, tmp_path: Path) -> None:
        """Async poll: PENDING → COMPLETED lifecycle."""
        out = tmp_path / "out.mp4"
        submit_resp = {"request_id": "abc123"}
        poll_pending = {"status": "IN_PROGRESS"}
        poll_done = {"status": "COMPLETED", "video": {"url": "http://cdn/v.mp4"}}

        poll_responses = [poll_pending, poll_done]

        async def _fake_get(url: str, **_: object) -> MagicMock:
            if "status" in url:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = poll_responses.pop(0)
                return r
            r = MagicMock()
            r.status_code = 200
            r.content = b"\x00" * 128
            return r

        client = MagicMock()
        client.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value=submit_resp),
        ))
        client.get = _fake_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            with patch("oprim.ltx2_cloud_generate.asyncio.sleep", new_callable=AsyncMock):
                result = await ltx2_cloud_generate(
                    config={"FAL_API_KEY": "k"},
                    mode="t2v",
                    prompt="poll test",
                    duration_s=5.0,
                    resolution=(256, 256),
                    output_path=out,
                )

        assert result == out

    async def test_api_failure_raises_ltx2_error(self, tmp_path: Path) -> None:
        """4xx/5xx from fal.ai → Ltx2CloudError with code+message."""
        client = MagicMock()
        client.post = AsyncMock(return_value=MagicMock(
            status_code=503,
            text="Service Unavailable",
        ))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(Ltx2CloudError, match="503"):
                await ltx2_cloud_generate(
                    config={"FAL_API_KEY": "k"},
                    mode="t2v",
                    prompt="fail test",
                    duration_s=5.0,
                    resolution=(256, 256),
                    output_path=tmp_path / "out.mp4",
                )

    async def test_duration_over_limit_raises_value_error(self, tmp_path: Path) -> None:
        """duration_s > 20 → ValueError before any network call."""
        with pytest.raises(ValueError, match="20s"):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="t2v",
                prompt="too long",
                duration_s=25.0,
                resolution=(256, 256),
                output_path=tmp_path / "out.mp4",
            )

    async def test_i2v_without_reference_raises_value_error(self, tmp_path: Path) -> None:
        """mode='i2v' without reference_image → ValueError."""
        with pytest.raises(ValueError, match="reference_image"):
            await ltx2_cloud_generate(
                config={"FAL_API_KEY": "k"},
                mode="i2v",
                prompt="no ref",
                reference_image=None,
                duration_s=5.0,
                resolution=(256, 256),
                output_path=tmp_path / "out.mp4",
            )
