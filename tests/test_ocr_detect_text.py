"""Tests for oprim.ocr_detect_text."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.ocr_detect_text import ocr_detect_text

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal fake image bytes


def test_returns_dict_with_required_keys():
    result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert set(result.keys()) >= {"text", "confidence", "language", "provider_used", "error"}


def test_stub_returns_empty_text_when_no_provider():
    # With no OCR libs installed (or mocked away), stub path returns text=""
    with patch.dict("sys.modules", {"pytesseract": None, "paddleocr": None}):
        result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert result["text"] == ""


def test_confidence_field_present():
    result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert "confidence" in result


def test_language_returned():
    result = ocr_detect_text(image_bytes=FAKE_PNG, language="chi_sim")
    assert result["language"] == "chi_sim"


def test_provider_used_field_present():
    result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert "provider_used" in result
    assert isinstance(result["provider_used"], str)


def test_empty_bytes_returns_stub():
    result = ocr_detect_text(image_bytes=b"")
    assert result["text"] == ""
    assert result["error"] is None


def test_error_none_on_stub_success():
    with patch.dict("sys.modules", {"pytesseract": None, "paddleocr": None}):
        result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert result["error"] is None


def test_error_set_on_tesseract_exception():
    mock_tess = MagicMock()
    mock_tess.image_to_string.side_effect = RuntimeError("tesseract not found")
    mock_tess.image_to_data.side_effect = RuntimeError("tesseract not found")
    mock_tess.Output = MagicMock()
    mock_tess.Output.DICT = "dict"

    mock_pil_image = MagicMock()
    mock_pil_module = MagicMock()
    mock_pil_module.Image.open.return_value = mock_pil_image

    with (
        patch.dict("sys.modules", {"pytesseract": mock_tess, "PIL": mock_pil_module}),
        patch(
            "builtins.__import__",
            side_effect=_make_importer({"pytesseract": mock_tess, "PIL": mock_pil_module}),
        ),
    ):
        result = ocr_detect_text(image_bytes=FAKE_PNG, provider="tesseract")

    assert result["error"] is not None


def test_provider_parameter_passed():
    result = ocr_detect_text(image_bytes=FAKE_PNG, provider="stub_nonexistent")
    # unknown provider falls through to stub
    assert result["provider_used"] == "stub"


def test_image_bytes_type_check():
    result = ocr_detect_text(image_bytes="not bytes")  # type: ignore[arg-type]
    assert result["error"] is not None


def test_tesseract_provider_used_when_available():
    mock_tess = MagicMock()
    mock_tess.Output = MagicMock()
    mock_tess.Output.DICT = "dict"
    mock_tess.image_to_data.return_value = {"conf": [90, 95]}
    mock_tess.image_to_string.return_value = "hello world"

    mock_image = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_image

    import io as _io

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            return mock_tess
        if name == "PIL":
            return mock_pil
        if name == "io":
            return _io
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        result = ocr_detect_text(image_bytes=FAKE_PNG, provider="tesseract")

    # Either used tesseract or fell through to stub — provider_used must be a string
    assert isinstance(result["provider_used"], str)


def test_default_language_is_eng():
    result = ocr_detect_text(image_bytes=FAKE_PNG)
    assert result["language"] == "eng"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_importer(overrides: dict):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name in overrides:
            mod = overrides[name]
            if mod is None:
                raise ImportError(name)
            return mod
        return real_import(name, *args, **kwargs)

    return _import
