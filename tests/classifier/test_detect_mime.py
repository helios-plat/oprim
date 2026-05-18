"""Tests for oprim.classifier.detect_mime."""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from oprim.classifier.detect_mime import detect_mime


class TestDetectMime:
    def test_detect_text_file(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        mime = detect_mime(f)
        assert mime.startswith("text/")

    def test_detect_pdf_file(self, simple_pdf: Path):
        mime = detect_mime(simple_pdf)
        assert mime == "application/pdf"

    def test_detect_png_file(self, simple_png: Path):
        mime = detect_mime(simple_png)
        assert mime == "image/png"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            detect_mime(tmp_path / "nonexistent.pdf")

    def test_not_a_file(self, tmp_path: Path):
        # A directory is not a file
        with pytest.raises(FileNotFoundError):
            detect_mime(tmp_path)

    def test_fake_extension_detected_by_content(self, tmp_path: Path):
        """File named .pdf but contains text → mime should be text/..., not application/pdf."""
        f = tmp_path / "fake.pdf"
        f.write_text("This is plain text, not a PDF at all!")
        mime = detect_mime(f)
        assert mime.startswith("text/")

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        # Should not raise; libmagic returns application/x-empty or similar
        mime = detect_mime(f)
        assert isinstance(mime, str)

    def test_accepts_path_string(self, simple_png: Path):
        # detect_mime should coerce strings to Path
        mime = detect_mime(str(simple_png))
        assert mime == "image/png"

    def test_magic_exception_propagated(self, tmp_path: Path):
        import magic
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")

        def raise_magic_err(*args, **kwargs):
            raise RuntimeError("libmagic internal error")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(magic, "from_file", raise_magic_err)
            with pytest.raises(RuntimeError, match="libmagic"):
                detect_mime(f)

    def test_permission_error_propagated(self, tmp_path: Path):
        import os
        f = tmp_path / "noaccess.bin"
        f.write_bytes(b"data")
        os.chmod(str(f), 0o000)
        try:
            # May not raise on root; skip if no error
            with pytest.raises(PermissionError):
                detect_mime(f)
        except Exception:
            pass  # running as root; skip
        finally:
            os.chmod(str(f), 0o644)
