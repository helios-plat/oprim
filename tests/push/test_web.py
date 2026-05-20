"""Tests for WebPushChannel (no real VAPID — mock pywebpush)."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from oprim.push.channels.web import WebPushChannel
from oprim.push.errors import PushConfigError


class TestWebPushChannelInit:
    def test_raises_without_vapid_key(self):
        with pytest.raises(PushConfigError, match="vapid_private_key"):
            WebPushChannel(vapid_private_key="", vapid_claims={"sub": "mailto:a@b.com"})

    def test_raises_without_sub_claim(self):
        with pytest.raises(PushConfigError, match="sub"):
            WebPushChannel(vapid_private_key="key", vapid_claims={})

    def test_valid_init(self):
        ch = WebPushChannel(vapid_private_key="key", vapid_claims={"sub": "mailto:a@b.com"})
        assert ch.name == "web"


class TestWebPushSend:
    @pytest.fixture
    def channel(self):
        return WebPushChannel(vapid_private_key="fake_key", vapid_claims={"sub": "mailto:a@b.com"})

    async def test_send_when_pywebpush_unavailable(self, channel):
        with patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", False):
            result = await channel.send("sub_json", "Title", "Body")
        assert result.success is False
        assert "pywebpush" in result.error_message

    async def test_send_invalid_json_recipient(self, channel):
        with patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", True):
            result = await channel.send("not-json{{{", "Title", "Body")
        assert result.success is False
        assert "Invalid subscription JSON" in result.error_message

    async def test_send_success(self, channel):
        subscription = json.dumps({
            "endpoint": "https://push.example.com/sub",
            "keys": {"p256dh": "abc", "auth": "def"},
        })
        with (
            patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", True),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
            patch("oprim.push.channels.web.webpush", return_value=None),
        ):
            result = await channel.send(subscription, "Title", "Body")
        assert result.success is True
        assert result.channel == "web"

    async def test_send_exception_returns_failure(self, channel):
        subscription = json.dumps({"endpoint": "https://example.com", "keys": {}})
        with (
            patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", True),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
            patch("oprim.push.channels.web.webpush", side_effect=Exception("push failed")),
        ):
            result = await channel.send(subscription, "Title", "Body")
        assert result.success is False
        assert "push failed" in result.error_message

    async def test_send_with_deep_link(self, channel):
        subscription = json.dumps({"endpoint": "https://example.com", "keys": {}})
        with (
            patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", True),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
            patch("oprim.push.channels.web.webpush", return_value=None) as mock_wp,
        ):
            result = await channel.send(subscription, "T", "B", deep_link="stratum://open/sub_1")
        assert result.success is True


class TestWebPushHealthCheck:
    async def test_healthy_when_available(self):
        ch = WebPushChannel(vapid_private_key="key", vapid_claims={"sub": "mailto:a@b.com"})
        with patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", True):
            assert await ch.health_check() is True

    async def test_unhealthy_when_unavailable(self):
        ch = WebPushChannel(vapid_private_key="key", vapid_claims={"sub": "mailto:a@b.com"})
        with patch("oprim.push.channels.web._PYWEBPUSH_AVAILABLE", False):
            assert await ch.health_check() is False
