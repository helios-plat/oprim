"""Tests for oprim.video_edit_element_remove."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from obase import ProviderRegistry
from oprim.video_edit_element_remove import (
    VideoEditError,
    VideoEditProviderNotFoundError,
    video_edit_element_remove,
)


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    v = tmp_path / "input.mp4"
    v.write_bytes(b"\x00" * 256)
    return v


class TestVideoEditElementRemove:
    async def test_mask_as_path_success(self, tmp_path: Path, video: Path) -> None:
        """mask as Path: mock provider succeeds."""
        mask = tmp_path / "mask.png"
        mask.write_bytes(b"\x89PNG\x00" * 10)
        out = tmp_path / "output.mp4"

        async def _edit(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 256)

        ProviderRegistry.register(category="video_inpaint", name="mock_sam2", fn=_edit)
        result = await video_edit_element_remove(
            video_path=video,
            element_mask=mask,
            inpaint_provider="mock_sam2",
            output_path=out,
        )
        assert result == out
        assert out.exists()

    async def test_mask_as_string_description(self, tmp_path: Path, video: Path) -> None:
        """mask as str text description: mock provider succeeds."""
        out = tmp_path / "output.mp4"

        async def _edit(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 256)

        ProviderRegistry.register(category="video_inpaint", name="mock_text", fn=_edit)
        result = await video_edit_element_remove(
            video_path=video,
            element_mask="remove the watermark in the top-right corner",
            inpaint_provider="mock_text",
            output_path=out,
        )
        assert result == out

    async def test_video_not_found(self, tmp_path: Path) -> None:
        """Missing video_path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="video_path not found"):
            await video_edit_element_remove(
                video_path=tmp_path / "missing.mp4",
                element_mask="remove watermark",
                inpaint_provider="mock",
                output_path=tmp_path / "out.mp4",
            )

    async def test_provider_not_found(self, tmp_path: Path, video: Path) -> None:
        """Unregistered provider raises VideoEditProviderNotFoundError."""
        with pytest.raises(VideoEditProviderNotFoundError):
            await video_edit_element_remove(
                video_path=video,
                element_mask="remove logo",
                inpaint_provider="nonexistent",
                output_path=tmp_path / "out.mp4",
            )

    async def test_no_output_produced(self, tmp_path: Path, video: Path) -> None:
        """Provider not creating output raises VideoEditError."""

        async def _noop(**kw: Any) -> None:
            pass

        ProviderRegistry.register(category="video_inpaint", name="noop", fn=_noop)
        with pytest.raises(VideoEditError, match="did not produce"):
            await video_edit_element_remove(
                video_path=video,
                element_mask="remove text",
                inpaint_provider="noop",
                output_path=tmp_path / "out.mp4",
            )

    async def test_provider_failure_wrapped(self, tmp_path: Path, video: Path) -> None:
        """Provider RuntimeError is wrapped into VideoEditError."""

        async def _fail(**kw: Any) -> None:
            raise RuntimeError("segfault in inpaint model")

        ProviderRegistry.register(category="video_inpaint", name="crash", fn=_fail)
        with pytest.raises(VideoEditError, match="segfault"):
            await video_edit_element_remove(
                video_path=video,
                element_mask="remove element",
                inpaint_provider="crash",
                output_path=tmp_path / "out.mp4",
            )
