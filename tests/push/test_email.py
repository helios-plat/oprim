"""Tests for EmailPushChannel (mock SMTP)."""
from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from oprim.push.channels.email import EmailPushChannel
from oprim.push.errors import PushConfigError


def _make_channel(**kwargs):
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_port=465,
        smtp_user="user@example.com",
        smtp_password="pass",
        from_address="noreply@example.com",
        use_tls=True,
    )
    defaults.update(kwargs)
    return EmailPushChannel(**defaults)


class TestEmailPushChannelInit:
    def test_raises_without_host(self):
        with pytest.raises(PushConfigError, match="smtp_host"):
            _make_channel(smtp_host="")

    def test_raises_without_from(self):
        with pytest.raises(PushConfigError, match="from_address"):
            _make_channel(from_address="")

    def test_valid_init(self):
        ch = _make_channel()
        assert ch.name == "email"


class TestEmailPushSend:
    async def test_send_success(self):
        ch = _make_channel()
        mock_server = MagicMock()

        with (
            patch("smtplib.SMTP_SSL", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("user@example.com", "Title", "Body text")

        assert result.success is True
        assert result.channel == "email"
        mock_server.sendmail.assert_called_once()

    async def test_send_with_deep_link(self):
        ch = _make_channel()
        mock_server = MagicMock()

        with (
            patch("smtplib.SMTP_SSL", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("u@e.com", "T", "B", deep_link="stratum://open/sub_1")
        assert result.success is True

    async def test_send_smtp_exception_returns_failure(self):
        ch = _make_channel()

        with (
            patch("smtplib.SMTP_SSL", side_effect=smtplib.SMTPException("connection refused")),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("u@e.com", "T", "B")

        assert result.success is False
        assert "connection refused" in result.error_message

    async def test_send_unexpected_exception_returns_failure(self):
        ch = _make_channel()

        with (
            patch("smtplib.SMTP_SSL", side_effect=OSError("network down")),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("u@e.com", "T", "B")

        assert result.success is False

    async def test_send_starttls_mode(self):
        ch = _make_channel(use_tls=False)
        mock_server = MagicMock()

        with (
            patch("smtplib.SMTP", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("u@e.com", "T", "B")

        assert result.success is True
        mock_server.starttls.assert_called_once()

    async def test_send_no_auth(self):
        ch = _make_channel(smtp_user="", smtp_password="")
        mock_server = MagicMock()

        with (
            patch("smtplib.SMTP_SSL", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            result = await ch.send("u@e.com", "T", "B")

        assert result.success is True
        mock_server.login.assert_not_called()


class TestEmailHealthCheck:
    async def test_healthy_when_smtp_ok(self):
        ch = _make_channel()
        mock_server = MagicMock()
        mock_server.noop.return_value = (250, b"OK")

        with (
            patch("smtplib.SMTP_SSL", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            assert await ch.health_check() is True

    async def test_unhealthy_on_exception(self):
        ch = _make_channel()

        with (
            patch("smtplib.SMTP_SSL", side_effect=OSError("refused")),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            assert await ch.health_check() is False

    async def test_healthy_starttls(self):
        ch = _make_channel(use_tls=False)
        mock_server = MagicMock()

        with (
            patch("smtplib.SMTP", return_value=mock_server),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
        ):
            assert await ch.health_check() is True
