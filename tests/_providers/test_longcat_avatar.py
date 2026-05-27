"""Tests for oprim._providers.longcat_avatar."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._providers.longcat_avatar import (
    LongCatAvatarError,
    LongCatAvatarSetupError,
    invoke_cloud,
    invoke_local,
)


@pytest.fixture()
def setup(tmp_path: Path):
    """Create vendor_dir with fake inference.py, portrait, and audio."""
    vendor_dir = tmp_path / "longcat"
    vendor_dir.mkdir()
    (vendor_dir / "inference.py").write_text("# stub")
    portrait = tmp_path / "face.png"
    portrait.write_bytes(b"\x89PNG" + b"\x00" * 32)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 32)
    output = tmp_path / "avatar.mp4"
    return vendor_dir, portrait, audio, output


class TestInvokeLocal:
    async def test_vendor_dir_not_found_raises_setup_error(self, tmp_path: Path) -> None:
        """Missing vendor_dir raises LongCatAvatarSetupError."""
        with pytest.raises(LongCatAvatarSetupError, match="vendor_dir not found"):
            await invoke_local(
                portrait_image=tmp_path / "face.png",
                audio_path=tmp_path / "audio.wav",
                output_path=tmp_path / "out.mp4",
                vendor_dir=tmp_path / "nonexistent",
            )

    async def test_script_not_found_raises_setup_error(self, tmp_path: Path) -> None:
        """vendor_dir exists but no inference.py raises LongCatAvatarSetupError."""
        vendor_dir = tmp_path / "longcat"
        vendor_dir.mkdir()
        portrait = tmp_path / "face.png"
        portrait.write_bytes(b"\x89PNG" + b"\x00" * 32)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 32)
        with pytest.raises(LongCatAvatarSetupError, match="inference script not found"):
            await invoke_local(
                portrait_image=portrait,
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                vendor_dir=vendor_dir,
            )

    async def test_success_mock_subprocess(self, tmp_path: Path, setup) -> None:
        """Mock subprocess returning exit 0 → output_path returned."""
        vendor_dir, portrait, audio, output = setup

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(b"", b"")):
                result = await invoke_local(
                    portrait_image=portrait,
                    audio_path=audio,
                    output_path=output,
                    vendor_dir=vendor_dir,
                )
        assert result == output

    async def test_subprocess_nonzero_exit_raises_error(self, tmp_path: Path, setup) -> None:
        """Subprocess exit != 0 raises LongCatAvatarError."""
        vendor_dir, portrait, audio, output = setup

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"GPU OOM error"))
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(b"", b"GPU OOM error")):
                with pytest.raises(LongCatAvatarError, match="exited 1"):
                    await invoke_local(
                        portrait_image=portrait,
                        audio_path=audio,
                        output_path=output,
                        vendor_dir=vendor_dir,
                    )

    async def test_subprocess_timeout_raises_error(self, tmp_path: Path, setup) -> None:
        """asyncio.TimeoutError raises LongCatAvatarError about timeout."""
        vendor_dir, portrait, audio, output = setup

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(LongCatAvatarError, match="timed out"):
                    await invoke_local(
                        portrait_image=portrait,
                        audio_path=audio,
                        output_path=output,
                        vendor_dir=vendor_dir,
                        timeout_s=0.001,
                    )


class TestInvokeCloud:
    async def test_invoke_cloud_raises_not_implemented(self, tmp_path: Path) -> None:
        """invoke_cloud always raises NotImplementedError (TECHNICAL_DEBT stub)."""
        with pytest.raises(NotImplementedError, match="cloud API not yet implemented"):
            await invoke_cloud(
                portrait_image=tmp_path / "face.png",
                audio_path=tmp_path / "audio.wav",
                output_path=tmp_path / "out.mp4",
                api_key="test_key",
                base_url="https://api.example.com",
            )
