"""Tests for oprim.thumbnail_generate."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from oprim import thumbnail_generate
from oprim._exceptions import OprimValidationError
from oprim._thumbnail_generate import ThumbnailResult


def _png(w: int, h: int, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (w, h), color=(120, 30, 200) if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestValidation:
    def test_empty_bytes(self) -> None:
        with pytest.raises(OprimValidationError, match="empty"):
            thumbnail_generate(image_bytes=b"")

    def test_bad_max_size(self) -> None:
        with pytest.raises(OprimValidationError, match="max_size"):
            thumbnail_generate(image_bytes=_png(10, 10), max_size=0)

    def test_not_an_image(self) -> None:
        with pytest.raises(OprimValidationError, match="not a decodable image"):
            thumbnail_generate(image_bytes=b"this is not an image")


class TestThumbnail:
    def test_downscales_preserving_aspect(self) -> None:
        r = thumbnail_generate(image_bytes=_png(800, 400), max_size=256)
        assert isinstance(r, ThumbnailResult)
        assert max(r.width, r.height) == 256
        assert r.width == 256 and r.height == 128  # 2:1 preserved
        assert r.source_width == 800 and r.source_height == 400

    def test_does_not_upscale(self) -> None:
        r = thumbnail_generate(image_bytes=_png(50, 50), max_size=256)
        assert r.width == 50 and r.height == 50  # thumbnail() never enlarges

    def test_webp_default_and_decodable(self) -> None:
        r = thumbnail_generate(image_bytes=_png(300, 300))
        assert r.format == "webp"
        out = Image.open(io.BytesIO(r.data))
        assert out.format == "WEBP"

    def test_jpeg_output(self) -> None:
        r = thumbnail_generate(image_bytes=_png(300, 300), fmt="jpeg")
        out = Image.open(io.BytesIO(r.data))
        assert out.format == "JPEG"

    def test_png_output_preserves(self) -> None:
        r = thumbnail_generate(image_bytes=_png(300, 300), fmt="png")
        out = Image.open(io.BytesIO(r.data))
        assert out.format == "PNG"

    def test_rgba_source_to_jpeg(self) -> None:
        # RGBA (alpha) must be converted for JPEG without raising.
        r = thumbnail_generate(image_bytes=_png(120, 120, mode="RGBA"), fmt="jpeg")
        assert r.width == 120
