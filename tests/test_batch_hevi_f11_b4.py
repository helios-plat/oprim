"""Batch 4 backward-compat tests: ltx2_cloud_generate + video_generate fps/bitrate."""
from __future__ import annotations

import inspect
import pytest

from oprim._ltx2_cloud_generate import ltx2_cloud_generate
from oprim._video_generate import video_generate


class TestLtx2FpsBitrate:
    def test_ltx2_accepts_fps_param(self):
        sig = inspect.signature(ltx2_cloud_generate)
        assert "fps" in sig.parameters

    def test_ltx2_fps_default_24(self):
        sig = inspect.signature(ltx2_cloud_generate)
        assert sig.parameters["fps"].default == 24

    def test_ltx2_bitrate_default_none(self):
        sig = inspect.signature(ltx2_cloud_generate)
        assert sig.parameters["bitrate_kbps"].default is None

    def test_ltx2_fps_bitrate_passthrough(self):
        """fps and bitrate_kbps accepted without error (API call not made)."""
        import asyncio

        async def _run():
            try:
                await ltx2_cloud_generate(
                    mode="t2v", prompt="test",
                    duration_s=3.0, resolution=(1280, 720),
                    output_path=__import__("pathlib").Path("/tmp/out.mp4"),
                    fps=30, bitrate_kbps=4000,
                    config={"FAL_API_KEY": ""},
                )
            except Exception as e:
                # Expected: missing API key or network — not a TypeError from sig
                assert "TypeError" not in type(e).__name__

        asyncio.run(_run())

    def test_backward_compat_no_fps_arg(self):
        """Calling without fps/bitrate must not raise TypeError."""
        sig = inspect.signature(ltx2_cloud_generate)
        params = sig.parameters
        assert params["fps"].default == 24
        assert params["bitrate_kbps"].default is None


class TestVideoGenerateFpsBitrate:
    def test_video_generate_accepts_fps(self):
        sig = inspect.signature(video_generate)
        assert "fps" in sig.parameters

    def test_video_generate_fps_default_24(self):
        sig = inspect.signature(video_generate)
        assert sig.parameters["fps"].default == 24

    def test_video_generate_bitrate_default_none(self):
        sig = inspect.signature(video_generate)
        assert sig.parameters["bitrate_kbps"].default is None

    def test_wan_fps_bitrate_passthrough(self):
        """wan_cloud branch with fps/bitrate doesn't crash on sig mismatch."""
        import asyncio

        async def _run():
            try:
                await video_generate(
                    provider="wan_cloud", prompt="test",
                    output_path=__import__("pathlib").Path("/tmp/out.mp4"),
                    fps=30, bitrate_kbps=2000,
                )
            except Exception as e:
                assert "TypeError" not in type(e).__name__

        asyncio.run(_run())

    def test_backward_compat_no_fps_arg(self):
        """Existing call sites without fps/bitrate are not broken."""
        sig = inspect.signature(video_generate)
        params = sig.parameters
        assert params["fps"].default == 24
        assert params["bitrate_kbps"].default is None
