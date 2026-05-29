"""Tests for oprim.first_last_frame_transition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from obase import ProviderRegistry
from oprim.first_last_frame_transition import (
    FrameTransitionError,
    FrameTransitionProviderNotFoundError,
    first_last_frame_transition,
)


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


@pytest.fixture()
def frames(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"\x89PNG\x00" * 10)
    last.write_bytes(b"\x89PNG\x00" * 10)
    return first, last


class TestFirstLastFrameTransition:
    async def test_success(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Mock provider succeeds and output file is returned."""
        first, last = frames
        out = tmp_path / "transition.mp4"

        async def _gen(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 128)

        ProviderRegistry.register(category="image_to_video", name="mock_wan", fn=_gen)
        result = await first_last_frame_transition(
            first_frame=first,
            last_frame=last,
            duration_s=3.0,
            video_provider="mock_wan",
            output_path=out,
        )
        assert result == out
        assert out.exists()

    async def test_provider_not_found(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Unregistered provider raises FrameTransitionProviderNotFoundError."""
        first, last = frames
        with pytest.raises(FrameTransitionProviderNotFoundError):
            await first_last_frame_transition(
                first_frame=first,
                last_frame=last,
                duration_s=3.0,
                video_provider="nonexistent",
                output_path=tmp_path / "out.mp4",
            )

    async def test_first_frame_not_found(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Missing first_frame raises FileNotFoundError."""
        _, last = frames
        with pytest.raises(FileNotFoundError, match="first_frame not found"):
            await first_last_frame_transition(
                first_frame=tmp_path / "missing.png",
                last_frame=last,
                duration_s=3.0,
                video_provider="mock",
                output_path=tmp_path / "out.mp4",
            )

    async def test_last_frame_not_found(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Missing last_frame raises FileNotFoundError."""
        first, _ = frames
        with pytest.raises(FileNotFoundError, match="last_frame not found"):
            await first_last_frame_transition(
                first_frame=first,
                last_frame=tmp_path / "missing.png",
                duration_s=3.0,
                video_provider="mock",
                output_path=tmp_path / "out.mp4",
            )

    async def test_provider_failure(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Provider raising RuntimeError → FrameTransitionError."""
        first, last = frames

        async def _fail(**kw: Any) -> None:
            raise RuntimeError("GPU OOM")

        ProviderRegistry.register(category="image_to_video", name="bad", fn=_fail)
        with pytest.raises(FrameTransitionError, match="GPU OOM"):
            await first_last_frame_transition(
                first_frame=first,
                last_frame=last,
                duration_s=3.0,
                video_provider="bad",
                output_path=tmp_path / "out.mp4",
            )

    async def test_no_output_produced(self, tmp_path: Path, frames: tuple[Path, Path]) -> None:
        """Provider not creating output file raises FrameTransitionError."""
        first, last = frames

        async def _noop(**kw: Any) -> None:
            pass  # doesn't create output

        ProviderRegistry.register(category="image_to_video", name="noop", fn=_noop)
        with pytest.raises(FrameTransitionError, match="did not produce"):
            await first_last_frame_transition(
                first_frame=first,
                last_frame=last,
                duration_s=3.0,
                video_provider="noop",
                output_path=tmp_path / "out.mp4",
            )
