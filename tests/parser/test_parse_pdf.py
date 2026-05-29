"""Tests for oprim.parser.parse_pdf."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from oprim.parser.parse_pdf import ParsedContent, parse_pdf
from oprim.errors import PDFParseError


class TestParsePDF:
    def test_pymupdf4llm_provider(self, simple_pdf: Path):
        result = parse_pdf(simple_pdf, provider="pymupdf4llm")
        assert isinstance(result, ParsedContent)
        assert result.parser_name == "pymupdf4llm"
        assert result.page_count == 1
        assert len(result.markdown) > 0
        assert len(result.plaintext) > 0

    def test_auto_dispatch_default_to_pymupdf4llm(self, simple_pdf: Path):
        """Simple non-scanned, non-CJK PDF should dispatch to pymupdf4llm."""
        result = parse_pdf(simple_pdf, provider="auto")
        assert result.parser_name == "pymupdf4llm"

    def test_marker_fallback_to_pymupdf4llm(self, simple_pdf: Path):
        """marker provider falls back to pymupdf4llm when not installed."""
        result = parse_pdf(simple_pdf, provider="marker")
        # marker-pdf is not installed → fallback to pymupdf4llm
        assert result.parser_name == "pymupdf4llm"

    def test_mineru_fallback_chain(self, simple_pdf: Path):
        """mineru → marker → pymupdf4llm fallback chain."""
        result = parse_pdf(simple_pdf, provider="mineru")
        assert result.parser_name == "pymupdf4llm"

    def test_unknown_provider_raises(self, simple_pdf: Path):
        with pytest.raises(PDFParseError, match="Unknown PDF provider"):
            parse_pdf(simple_pdf, provider="nonexistent_parser")

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_pdf(tmp_path / "missing.pdf")

    def test_encrypted_pdf_raises(self, tmp_path: Path):
        path = tmp_path / "enc.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "secret")
        doc.save(
            str(path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        doc.close()
        with pytest.raises(PDFParseError):
            parse_pdf(path, provider="pymupdf4llm")

    def test_parsed_content_has_required_fields(self, simple_pdf: Path):
        result = parse_pdf(simple_pdf, provider="pymupdf4llm")
        assert hasattr(result, "markdown")
        assert hasattr(result, "plaintext")
        assert hasattr(result, "page_count")
        assert hasattr(result, "images")
        assert hasattr(result, "tables")
        assert hasattr(result, "chapters")
        assert hasattr(result, "metadata")
        assert hasattr(result, "parser_name")
        assert hasattr(result, "parse_quality_score")

    def test_quality_score_between_0_and_1(self, simple_pdf: Path):
        result = parse_pdf(simple_pdf, provider="pymupdf4llm")
        assert 0.0 <= result.parse_quality_score <= 1.0

    def test_multi_page_pdf(self, multi_page_pdf: Path):
        result = parse_pdf(multi_page_pdf, provider="pymupdf4llm")
        assert result.page_count == 3

    def test_hint_zh_dispatch_to_mineru(self, tmp_path: Path):
        """CJK PDF with hint language=zh should dispatch to mineru (→ fallback chain)."""
        path = tmp_path / "zh.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "你好世界这是中文内容" * 20)
        doc.save(str(path))
        doc.close()

        # mineru not installed → falls all the way back to pymupdf4llm
        result = parse_pdf(path, provider="auto", hint={"language": "zh"})
        assert result.parser_name == "pymupdf4llm"

    def test_dispatch_mineru_with_mocked_cjk_features(self, simple_pdf: Path):
        from oprim.classifier.detect_pdf_features import PDFFeatures
        mock_features = PDFFeatures(
            page_count=1, first_page_text="测试", has_cjk=True,
            is_scanned=False, has_tables=False, is_two_column=False,
        )
        with patch("oprim.classifier.detect_pdf_features.detect_pdf_features", return_value=mock_features):
            result = parse_pdf(simple_pdf, provider="auto", hint={"language": "zh"})
        # mineru → marker → pymupdf4llm fallback chain
        assert result.parser_name == "pymupdf4llm"

    def test_dispatch_marker_with_mocked_scanned_features(self, simple_pdf: Path):
        from oprim.classifier.detect_pdf_features import PDFFeatures
        mock_features = PDFFeatures(
            page_count=1, first_page_text="", has_cjk=False,
            is_scanned=True, has_tables=False, is_two_column=False,
        )
        with patch("oprim.classifier.detect_pdf_features.detect_pdf_features", return_value=mock_features):
            result = parse_pdf(simple_pdf, provider="auto")
        # marker not installed → fallback to pymupdf4llm
        assert result.parser_name == "pymupdf4llm"

    def test_pymupdf4llm_exception_raises_pdfparseerror(self, simple_pdf: Path):
        with patch("oprim.parser.parse_pdf.fitz.open", side_effect=RuntimeError("fitz boom")):
            with pytest.raises(PDFParseError, match="pymupdf4llm failed"):
                parse_pdf(simple_pdf, provider="pymupdf4llm")

    def test_mineru_with_magic_pdf_mocked_raises_pdfparseerror(self, simple_pdf: Path):
        """When magic_pdf is importable, mineru raises NotImplementedError → PDFParseError."""
        import sys
        with patch.dict(sys.modules, {"magic_pdf": MagicMock()}):
            with pytest.raises(PDFParseError, match="mineru failed"):
                parse_pdf(simple_pdf, provider="mineru")

    def test_marker_installed_but_fails(self, simple_pdf: Path):
        import sys
        mock_marker_convert = MagicMock(side_effect=RuntimeError("marker internal error"))
        mock_marker_models = MagicMock(return_value=MagicMock())
        mock_marker_pkg = MagicMock()
        mock_marker_pkg.convert.convert_single_pdf = mock_marker_convert
        mock_marker_pkg.models.load_all_models = mock_marker_models
        with patch.dict(sys.modules, {
            "marker": MagicMock(),
            "marker.convert": mock_marker_pkg.convert,
            "marker.models": mock_marker_pkg.models,
        }):
            with pytest.raises(PDFParseError, match="marker failed"):
                parse_pdf(simple_pdf, provider="marker")

    def test_marker_installed_and_succeeds(self, simple_pdf: Path):
        import sys
        mock_full_text = "# Mocked Marker Output\nContent here."
        mock_images = []
        mock_meta = {"title": "test"}
        mock_convert_fn = MagicMock(return_value=(mock_full_text, mock_images, mock_meta))
        mock_models_fn = MagicMock(return_value=MagicMock())
        mock_marker_convert = MagicMock()
        mock_marker_convert.convert_single_pdf = mock_convert_fn
        mock_marker_models = MagicMock()
        mock_marker_models.load_all_models = mock_models_fn
        with patch.dict(sys.modules, {
            "marker": MagicMock(),
            "marker.convert": mock_marker_convert,
            "marker.models": mock_marker_models,
        }):
            result = parse_pdf(simple_pdf, provider="marker")
        assert result.parser_name == "marker"
        assert result.markdown == mock_full_text

    def test_parse_epub_error_passthrough(self, tmp_path: Path):
        """Non-EPUB file passed as EPUB raises an exception (from ebooklib)."""
        f = tmp_path / "bad.epub"
        f.write_bytes(b"not real epub data at all")
        from oprim.parser.parse_epub import parse_epub
        with pytest.raises(Exception):
            parse_epub(f)
