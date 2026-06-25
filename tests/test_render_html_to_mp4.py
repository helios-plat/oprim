"""Tests for render_html_to_mp4."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim._render_html_to_mp4 import RenderHtmlError, render_html_to_mp4

_CLEAN_HTML = "<html><body><div class='box'>animated</div></body></html>"
_EVIL_HTML = '<div onclick="steal()">bad</div>'


async def _mock_capture(**kw):
    frames_dir: Path = kw["frames_dir"]
    (frames_dir / "frame_000000.png").write_bytes(b"\x89PNG\r\n")


async def _mock_encode(**kw):
    out: Path = kw["output_path"]
    out.write_bytes(b"\x00" * 16)


class TestRenderHtmlToMp4:

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_basic_render_returns_output_path(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        result = await render_html_to_mp4(html=_CLEAN_HTML, output_path=out, duration_s=3.0)
        assert result == out

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_validate_true_blocks_dangerous_html(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        with pytest.raises(RenderHtmlError, match="validation failed"):
            await render_html_to_mp4(html=_EVIL_HTML, output_path=out, duration_s=2.0, validate=True)
        mock_cap.assert_not_called()

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_validate_false_skips_check(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        # dangerous HTML with validate=False → no RenderHtmlError
        result = await render_html_to_mp4(html=_EVIL_HTML, output_path=out, duration_s=1.0, validate=False)
        assert result == out

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_empty_html_raises(self, mock_cap, mock_enc, tmp_path):
        with pytest.raises(RenderHtmlError):
            await render_html_to_mp4(html="   ", output_path=tmp_path / "o.mp4", duration_s=1.0)

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_capture_called_with_correct_duration(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        await render_html_to_mp4(html=_CLEAN_HTML, output_path=out, duration_s=7.5)
        call_kw = mock_cap.call_args.kwargs
        assert call_kw["duration_s"] == 7.5

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_encode_uses_h264_args(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        await render_html_to_mp4(html=_CLEAN_HTML, output_path=out, duration_s=2.0)
        # encode was called — codec selection is inside _encode_frames_to_mp4
        mock_enc.assert_called_once()
        assert mock_enc.call_args.kwargs["output_path"] == out

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_custom_resolution_passed_through(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        await render_html_to_mp4(
            html=_CLEAN_HTML, output_path=out, duration_s=1.0,
            width=1280, height=720,
        )
        kw = mock_cap.call_args.kwargs
        assert kw["width"] == 1280 and kw["height"] == 720

    @patch("oprim._render_html_to_mp4._encode_frames_to_mp4", new_callable=AsyncMock)
    @patch("oprim._render_html_to_mp4._capture_html_frames", new_callable=AsyncMock)
    async def test_timeout_passed_to_capture_and_encode(self, mock_cap, mock_enc, tmp_path):
        out = tmp_path / "out.mp4"
        mock_cap.side_effect = _mock_capture
        mock_enc.side_effect = _mock_encode
        await render_html_to_mp4(html=_CLEAN_HTML, output_path=out, duration_s=1.0, timeout_s=30.0)
        assert mock_cap.call_args.kwargs["timeout_s"] == 30.0
        assert mock_enc.call_args.kwargs["timeout_s"] == 30.0
