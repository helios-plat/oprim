"""Tests for oprim.classifier.detect_image_exif."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from oprim.classifier.detect_image_exif import ImageExif, detect_image_exif
from oprim.errors import UnsupportedImageError


class TestDetectImageExif:
    def test_basic_png(self, simple_png: Path):
        result = detect_image_exif(simple_png)
        assert isinstance(result, ImageExif)
        assert result.width == 100
        assert result.height == 100
        assert not result.has_exif
        assert result.camera_make is None

    def test_screenshot_heuristic(self, screen_size_png: Path):
        result = detect_image_exif(screen_size_png)
        assert result.is_screenshot_likely
        assert result.width == 1920
        assert result.height == 1080

    def test_non_screen_size_not_screenshot(self, simple_png: Path):
        result = detect_image_exif(simple_png)
        assert not result.is_screenshot_likely  # 100x100 is not a screen size

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            detect_image_exif(tmp_path / "missing.png")

    def test_invalid_file_raises(self, tmp_path: Path):
        f = tmp_path / "fake.png"
        f.write_bytes(b"not an image")
        with pytest.raises(UnsupportedImageError):
            detect_image_exif(f)

    def test_jpeg_without_exif(self, tmp_path: Path):
        path = tmp_path / "no_exif.jpg"
        img = Image.new("RGB", (640, 480), color=(0, 255, 0))
        img.save(str(path), "JPEG")
        result = detect_image_exif(path)
        assert result.width == 640
        assert result.height == 480
        assert not result.has_exif

    def test_result_fields_present(self, simple_png: Path):
        result = detect_image_exif(simple_png)
        assert hasattr(result, "has_exif")
        assert hasattr(result, "camera_make")
        assert hasattr(result, "camera_model")
        assert hasattr(result, "datetime_taken")
        assert hasattr(result, "width")
        assert hasattr(result, "height")
        assert hasattr(result, "is_screenshot_likely")

    def test_jpeg_with_camera_exif(self, tmp_path: Path):
        from unittest.mock import patch, MagicMock
        path = tmp_path / "camera.jpg"
        Image.new("RGB", (640, 480), color=(128, 64, 32)).save(str(path), "JPEG")

        fake_exif = {0x010F: "Apple", 0x0110: "iPhone 14 Pro", 0x9003: "2023:01:01 12:00:00"}
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_img.format = "JPEG"
        mock_img._getexif.return_value = fake_exif

        with patch("oprim.classifier.detect_image_exif.Image") as mock_Image:
            mock_Image.open.return_value = mock_img
            mock_Image.UnidentifiedImageError = Image.UnidentifiedImageError
            result = detect_image_exif(path)

        assert result.has_exif is True
        assert result.camera_make == "Apple"
        assert result.camera_model == "iPhone 14 Pro"
        assert result.datetime_taken is not None

    def test_exif_with_none_make(self, tmp_path: Path):
        from unittest.mock import patch, MagicMock
        path = tmp_path / "cam.jpg"
        Image.new("RGB", (100, 100)).save(str(path), "JPEG")

        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.format = "JPEG"
        mock_img._getexif.return_value = {0x9003: "2023:01:01 12:00:00"}

        with patch("oprim.classifier.detect_image_exif.Image") as mock_Image:
            mock_Image.open.return_value = mock_img
            mock_Image.UnidentifiedImageError = Image.UnidentifiedImageError
            result = detect_image_exif(path)

        assert result.has_exif is True
        assert result.camera_make is None
        assert result.camera_model is None

    def test_getexif_raises_is_silenced(self, tmp_path: Path):
        from unittest.mock import patch, MagicMock
        path = tmp_path / "weird.jpg"
        Image.new("RGB", (100, 100)).save(str(path), "JPEG")

        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_img.format = "JPEG"
        mock_img._getexif.side_effect = AttributeError("no _getexif")

        with patch("oprim.classifier.detect_image_exif.Image") as mock_Image:
            mock_Image.open.return_value = mock_img
            mock_Image.UnidentifiedImageError = Image.UnidentifiedImageError
            result = detect_image_exif(path)

        assert result.has_exif is False
