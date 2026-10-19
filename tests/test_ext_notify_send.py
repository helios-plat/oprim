"""Tests for oprim.ext_notify_send."""

from __future__ import annotations

import pytest
from obase import ProviderRegistry
from obase.notification_providers import LogNotificationProvider

from oprim._exceptions import NotifyOprimError
from oprim.ext_notify_send import ext_notify_send


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestExtNotifySendEmail:
    async def test_renders_template_and_sends_email(self):
        provider = LogNotificationProvider()
        ProviderRegistry.get().register_generic("notification", "log", provider)
        ok = await ext_notify_send(
            "log",
            channel="email",
            template="Hello {{ name }}, your order {{ order_id }} shipped.",
            data={"to": "a@b.com", "subject": "Shipped!", "name": "Ada", "order_id": "o1"},
        )
        assert ok is True
        assert provider.sent[0]["body"] == "Hello Ada, your order o1 shipped."
        assert provider.sent[0]["subject"] == "Shipped!"
        assert provider.sent[0]["to"] == "a@b.com"

    async def test_missing_subject_rejected(self):
        ProviderRegistry.get().register_generic("notification", "log", LogNotificationProvider())
        with pytest.raises(NotifyOprimError, match="subject"):
            await ext_notify_send("log", channel="email", template="hi", data={"to": "a@b.com"})


class TestExtNotifySendSms:
    async def test_renders_template_and_sends_sms(self):
        provider = LogNotificationProvider()
        ProviderRegistry.get().register_generic("notification", "log", provider)
        ok = await ext_notify_send(
            "log",
            channel="sms",
            template="Code: {{ code }}",
            data={"to": "+1555", "code": "123456"},
        )
        assert ok is True
        assert provider.sent[0]["message"] == "Code: 123456"


class TestExtNotifySendValidation:
    async def test_unknown_channel_rejected(self):
        ProviderRegistry.get().register_generic("notification", "log", LogNotificationProvider())
        with pytest.raises(NotifyOprimError, match="unknown channel"):
            await ext_notify_send("log", channel="carrier_pigeon", template="hi", data={"to": "x"})

    async def test_missing_to_rejected(self):
        ProviderRegistry.get().register_generic("notification", "log", LogNotificationProvider())
        with pytest.raises(NotifyOprimError, match="'to'"):
            await ext_notify_send("log", channel="sms", template="hi", data={})

    async def test_provider_not_found(self):
        with pytest.raises(NotifyOprimError, match="not found"):
            await ext_notify_send("ghost", channel="sms", template="hi", data={"to": "x"})
