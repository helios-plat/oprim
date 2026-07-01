"""Tests for Stratum Batch 1 Group 5:
llm_summarize, cache_invalidate, file_upload_handler,
temp_file_manager, push_email, otp_generate."""

from __future__ import annotations

import hashlib
import io
import smtplib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprim._exceptions import OprimError
from oprim.cache_invalidate import _memory_cache, cache_invalidate
from oprim.file_upload_handler import UploadResult, file_upload_handler
from oprim.llm_summarize import SummarizeResult, llm_summarize
from oprim.otp_generate import OTPResult, otp_generate, otp_verify
from oprim.push_email import EmailResult, push_email
from oprim.temp_file_manager import TempFileResult, _temp_registry, temp_file_manager


# ---------------------------------------------------------------------------
# llm_summarize
# ---------------------------------------------------------------------------


class TestLlmSummarize:
    def test_empty_text_returns_empty_summary(self) -> None:
        result = llm_summarize(text="   ", provider="qwen3", model="qwen3-max")
        assert result.summary == ""
        assert result.tokens_used == 0
        assert result.provider == "qwen3"

    def test_mock_provider_returns_string(self) -> None:
        mock_caller = MagicMock(return_value="Short summary.")
        with patch("oprim.llm_summarize.ProviderRegistry.get_caller", return_value=mock_caller):
            result = llm_summarize(
                text="A long text to summarize.", provider="qwen3", model="qwen3-max"
            )
        assert result.summary == "Short summary."
        assert result.provider == "qwen3"
        assert result.tokens_used == 2  # "Short summary." → 2 words

    def test_mock_provider_returns_dict_with_content(self) -> None:
        dict_response = {"content": "Dict summary here.", "usage": {"total_tokens": 42}}
        mock_caller = MagicMock(return_value=dict_response)
        with patch("oprim.llm_summarize.ProviderRegistry.get_caller", return_value=mock_caller):
            result = llm_summarize(text="Some input text.", provider="qwen3", model="qwen3-max")
        assert result.summary == "Dict summary here."
        assert result.tokens_used == 42

    def test_provider_error_raises_oprim_error(self) -> None:
        with patch(
            "oprim.llm_summarize.ProviderRegistry.get_caller",
            side_effect=RuntimeError("provider unavailable"),
        ):
            with pytest.raises(OprimError, match="llm_summarize failed"):
                llm_summarize(text="Some text.", provider="bad_provider", model="x")

    def test_style_param_passed_in_prompt(self) -> None:
        captured: list[list[dict[str, str]]] = []

        def capturing_caller(messages: list[dict[str, str]]) -> str:
            captured.append(messages)
            return "Bullet summary."

        with patch(
            "oprim.llm_summarize.ProviderRegistry.get_caller", return_value=capturing_caller
        ):
            llm_summarize(
                text="Input text.", provider="qwen3", model="qwen3-max", style="bullet_points"
            )

        assert len(captured) == 1
        prompt_content = captured[0][0]["content"]
        assert "bullet" in prompt_content.lower()

    def test_result_is_summarize_result_model(self) -> None:
        mock_caller = MagicMock(return_value="Summary text.")
        with patch("oprim.llm_summarize.ProviderRegistry.get_caller", return_value=mock_caller):
            result = llm_summarize(
                text="Something to summarize.", provider="qwen3", model="qwen3-max"
            )
        assert isinstance(result, SummarizeResult)


# ---------------------------------------------------------------------------
# cache_invalidate
# ---------------------------------------------------------------------------


class TestCacheInvalidate:
    def setup_method(self) -> None:
        _memory_cache.clear()

    def test_memory_backend_returns_true_when_key_exists(self) -> None:
        _memory_cache["user:42"] = "cached_value"
        result = cache_invalidate(key="user:42", cache_backend="memory")
        assert result is True
        assert "user:42" not in _memory_cache

    def test_memory_backend_returns_false_when_key_missing(self) -> None:
        result = cache_invalidate(key="nonexistent:key", cache_backend="memory")
        assert result is False

    def test_redis_mock_returns_true_when_delete_returns_one(self) -> None:
        mock_client = MagicMock()
        mock_client.delete.return_value = 1
        with patch("oprim.cache_invalidate.redis.Redis.from_url", return_value=mock_client):
            result = cache_invalidate(
                key="session:abc", cache_backend="redis", redis_url="redis://localhost:6379/0"
            )
        assert result is True
        mock_client.delete.assert_called_once_with("session:abc")

    def test_unsupported_backend_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="unsupported cache_backend"):
            cache_invalidate(key="k", cache_backend="memcached")

    def test_redis_connection_error_raises_oprim_error(self) -> None:
        with patch(
            "oprim.cache_invalidate.redis.Redis.from_url",
            side_effect=ConnectionError("refused"),
        ):
            with pytest.raises(OprimError, match="cache_invalidate redis failed"):
                cache_invalidate(key="k", cache_backend="redis", redis_url="redis://bad:6379/0")


# ---------------------------------------------------------------------------
# file_upload_handler
# ---------------------------------------------------------------------------


class TestFileUploadHandler:
    def test_basic_upload_sha256_correct(self, tmp_path: Path) -> None:
        content = b"hello world"
        stream = io.BytesIO(content)
        result = file_upload_handler(
            upload_stream=stream,
            filename="test.txt",
            total_size=len(content),
            dest_dir=tmp_path,
        )
        expected_sha256 = hashlib.sha256(content).hexdigest()
        assert result.sha256 == expected_sha256
        assert result.size_bytes == len(content)
        assert result.filename == "test.txt"

    def test_chunked_upload_assembles_correctly(self, tmp_path: Path) -> None:
        # Force multiple chunks by using small chunk_size
        content = b"A" * 100
        stream = io.BytesIO(content)
        result = file_upload_handler(
            upload_stream=stream,
            filename="big.bin",
            total_size=len(content),
            dest_dir=tmp_path,
            chunk_size=30,
        )
        assert result.chunks_written == 4  # ceil(100/30) = 4
        assert result.size_bytes == 100
        written = (tmp_path / "big.bin").read_bytes()
        assert written == content

    def test_filename_sanitized_strips_path_components(self, tmp_path: Path) -> None:
        stream = io.BytesIO(b"data")
        result = file_upload_handler(
            upload_stream=stream,
            filename="../../etc/passwd",
            total_size=4,
            dest_dir=tmp_path,
        )
        # Should only keep the final filename part
        assert result.filename == "passwd"
        assert result.dest_path == str(tmp_path / "passwd")

    def test_invalid_empty_filename_raises_oprim_error(self, tmp_path: Path) -> None:
        stream = io.BytesIO(b"data")
        with pytest.raises(OprimError, match="invalid filename"):
            file_upload_handler(
                upload_stream=stream,
                filename="/",
                total_size=4,
                dest_dir=tmp_path,
            )

    def test_small_file_single_chunk(self, tmp_path: Path) -> None:
        content = b"tiny"
        stream = io.BytesIO(content)
        result = file_upload_handler(
            upload_stream=stream,
            filename="tiny.txt",
            total_size=len(content),
            dest_dir=tmp_path,
        )
        assert result.chunks_written == 1
        assert result.size_bytes == 4

    def test_result_is_upload_result_model(self, tmp_path: Path) -> None:
        stream = io.BytesIO(b"x")
        result = file_upload_handler(
            upload_stream=stream, filename="x.bin", total_size=1, dest_dir=tmp_path
        )
        assert isinstance(result, UploadResult)


# ---------------------------------------------------------------------------
# temp_file_manager
# ---------------------------------------------------------------------------


class TestTempFileManager:
    def setup_method(self) -> None:
        _temp_registry.clear()

    def test_create_returns_file_path(self) -> None:
        result = temp_file_manager(action="create")
        assert result.success is True
        assert result.file_path is not None
        assert Path(result.file_path).exists()
        # cleanup
        Path(result.file_path).unlink(missing_ok=True)

    def test_get_existing_file_returns_path(self) -> None:
        created = temp_file_manager(action="create")
        assert created.file_path is not None
        got = temp_file_manager(action="get", file_path=Path(created.file_path))
        assert got.success is True
        assert got.file_path == created.file_path
        Path(created.file_path).unlink(missing_ok=True)

    def test_get_expired_file_returns_none(self) -> None:
        created = temp_file_manager(action="create")
        assert created.file_path is not None
        path_str = created.file_path
        # Manually backdate the registry entry
        _temp_registry[path_str] = (0.0, None)  # epoch 0 → definitely expired
        got = temp_file_manager(action="get", file_path=Path(path_str))
        assert got.success is False
        assert got.message == "expired"
        Path(path_str).unlink(missing_ok=True)

    def test_get_unknown_file_returns_not_found(self) -> None:
        got = temp_file_manager(action="get", file_path=Path("/tmp/never_registered.tmp"))
        assert got.success is False
        assert got.message == "not_found"

    def test_cleanup_expired_removes_old_entries(self) -> None:
        created = temp_file_manager(action="create")
        assert created.file_path is not None
        # Backdate to expired
        _temp_registry[created.file_path] = (0.0, None)
        result = temp_file_manager(action="cleanup_expired")
        assert result.cleaned_count >= 1
        assert created.file_path not in _temp_registry

    def test_cleanup_user_removes_user_files(self) -> None:
        r1 = temp_file_manager(action="create", user_key_hash="user_abc")
        r2 = temp_file_manager(action="create", user_key_hash="user_abc")
        r3 = temp_file_manager(action="create", user_key_hash="user_xyz")
        result = temp_file_manager(action="cleanup_user", user_key_hash="user_abc")
        assert result.cleaned_count == 2
        assert r3.file_path is not None
        assert r3.file_path in _temp_registry
        # cleanup remaining
        if r3.file_path:
            Path(r3.file_path).unlink(missing_ok=True)

    def test_unknown_action_raises_oprim_error(self) -> None:
        with pytest.raises(OprimError, match="unknown action"):
            temp_file_manager(action="delete")  # type: ignore[arg-type]

    def test_result_is_temp_file_result_model(self) -> None:
        result = temp_file_manager(action="create")
        assert isinstance(result, TempFileResult)
        if result.file_path:
            Path(result.file_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# push_email
# ---------------------------------------------------------------------------


class TestPushEmail:
    def test_success_plain_text(self) -> None:
        with patch("oprim.push_email.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = push_email(
                to="test@example.com",
                subject="Test",
                body="Hello",
                from_addr="sender@example.com",
                smtp_host="smtp.example.com",
            )
        assert result.success is True
        assert result.to == "test@example.com"
        assert result.subject == "Test"

    def test_smtp_exception_raises_oprim_error(self) -> None:
        with patch("oprim.push_email.smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.return_value.__enter__ = MagicMock(
                side_effect=smtplib.SMTPException("auth failed")
            )
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(OprimError, match="push_email SMTP error"):
                push_email(
                    to="a@b.com",
                    subject="x",
                    body="y",
                    from_addr="c@d.com",
                    smtp_host="smtp.bad.com",
                )

    def test_html_body_sends_multipart(self) -> None:
        with patch("oprim.push_email.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = push_email(
                to="user@example.com",
                subject="HTML Test",
                body="Plain fallback",
                from_addr="noreply@example.com",
                smtp_host="smtp.example.com",
                html_body="<h1>Hello</h1>",
            )
        assert result.success is True
        # Verify sendmail was called with multipart content
        call_args = mock_server.sendmail.call_args
        assert call_args is not None
        raw_msg: str = call_args[0][2]
        assert "multipart/alternative" in raw_msg

    def test_no_auth_skips_login(self) -> None:
        with patch("oprim.push_email.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            push_email(
                to="r@r.com",
                subject="s",
                body="b",
                from_addr="f@f.com",
                smtp_host="smtp.r.com",
                smtp_user=None,
                smtp_password=None,
            )
        mock_server.login.assert_not_called()

    def test_os_error_raises_oprim_error(self) -> None:
        with patch(
            "oprim.push_email.smtplib.SMTP",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(OprimError, match="push_email connection failed"):
                push_email(
                    to="a@b.com",
                    subject="x",
                    body="y",
                    from_addr="c@d.com",
                    smtp_host="smtp.unreachable.com",
                )

    def test_result_is_email_result_model(self) -> None:
        with patch("oprim.push_email.smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = push_email(
                to="a@b.com",
                subject="s",
                body="b",
                from_addr="f@f.com",
                smtp_host="smtp.x.com",
            )
        assert isinstance(result, EmailResult)


# ---------------------------------------------------------------------------
# otp_generate
# ---------------------------------------------------------------------------


class TestOtpGenerate:
    def test_code_is_six_digits(self) -> None:
        result = otp_generate()
        assert len(result.code) == 6
        assert result.code.isdigit()

    def test_with_known_secret_produces_consistent_code(self) -> None:
        import pyotp

        secret = pyotp.random_base32()
        result = otp_generate(secret=secret)
        # Verify the code matches what pyotp produces
        expected = pyotp.TOTP(secret).now()
        assert result.code == expected
        assert result.secret == secret

    def test_code_is_digits_only(self) -> None:
        result = otp_generate(digits=6)
        assert result.code.isdigit()

    def test_otp_verify_valid_code(self) -> None:
        result = otp_generate()
        assert otp_verify(secret=result.secret, code=result.code) is True

    def test_otp_verify_invalid_code_returns_false(self) -> None:
        result = otp_generate()
        assert otp_verify(secret=result.secret, code="000000") is False

    def test_expires_at_is_future(self) -> None:
        from datetime import UTC, datetime

        result = otp_generate()
        assert result.expires_at > datetime.now(tz=UTC)

    def test_result_is_otp_result_model(self) -> None:
        result = otp_generate()
        assert isinstance(result, OTPResult)
