"""Tests for oprim.classifier.detect_pdf_features."""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from oprim.classifier.detect_pdf_features import PDFFeatures, detect_pdf_features
from oprim.errors import PDFParseError


class TestDetectPDFFeatures:
    def test_basic_pdf(self, simple_pdf: Path):
        features = detect_pdf_features(simple_pdf)
        assert isinstance(features, PDFFeatures)
        assert features.page_count == 1
        assert "Hello World" in features.first_page_text
        assert not features.is_scanned  # text is present
        assert not features.has_cjk

    def test_multi_page_pdf(self, multi_page_pdf: Path):
        features = detect_pdf_features(multi_page_pdf)
        assert features.page_count == 3

    def test_scanned_pdf_detection(self, tmp_path: Path):
        """A PDF with no text content should be flagged as scanned."""
        path = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page()  # completely blank page
        doc.save(str(path))
        doc.close()
        features = detect_pdf_features(path)
        assert features.is_scanned  # < 150 chars

    def test_cjk_detection_on_ascii_pdf(self, simple_pdf: Path):
        """A PDF with only ASCII content should have has_cjk=False."""
        # Note: PyMuPDF's built-in font doesn't support CJK chars (renders as dots).
        # This test verifies the has_cjk=False path for a normal ASCII PDF.
        features = detect_pdf_features(simple_pdf)
        assert not features.has_cjk

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            detect_pdf_features(tmp_path / "missing.pdf")

    def test_corrupt_pdf_raises(self, tmp_path: Path):
        path = tmp_path / "corrupt.pdf"
        path.write_bytes(b"NOT A PDF AT ALL")
        with pytest.raises(PDFParseError):
            detect_pdf_features(path)

    def test_encrypted_pdf_raises(self, tmp_path: Path):
        path = tmp_path / "enc.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "secret content")
        doc.save(
            str(path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        doc.close()
        with pytest.raises(PDFParseError, match="encrypted"):
            detect_pdf_features(path)

    def test_dataclass_fields(self, simple_pdf: Path):
        features = detect_pdf_features(simple_pdf)
        assert hasattr(features, "page_count")
        assert hasattr(features, "first_page_text")
        assert hasattr(features, "has_cjk")
        assert hasattr(features, "is_scanned")
        assert hasattr(features, "has_tables")
        assert hasattr(features, "is_two_column")
