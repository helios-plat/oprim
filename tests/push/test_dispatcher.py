"""Tests for PushDispatcher."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from oprim.meta_db.duckdb import open_meta_db
from oprim.push.dispatcher import PushDispatcher
from oprim.push.errors import PushError
from oprim.push.protocol import PushResult

_MIGRATIONS_DIR = (
    Path(__file__).parent.parent.parent / "oprim" / "meta_db" / "migrations"
)


@pytest.fixture()
def db(tmp_path: Path):
    m = open_meta_db(tmp_path / "meta.duckdb")
    m.migrate(_MIGRATIONS_DIR)
    yield m
    m.close()


def _make_channel(name: str, success: bool = True, error: str | None = None):
    ch = MagicMock()
    ch.name = name
    result = PushResult(channel=name, success=success, recipient="r", error_message=error)
    ch.send = AsyncMock(return_value=result)
    return ch


def _seed_subscription(db, user_id: str, channel: str, recipient: str):
    db.execute(
        "INSERT INTO push_subscriptions (id, user_id, channel, recipient) VALUES (?, ?, ?, ?)",
        [f"{user_id}_{channel}", user_id, channel, recipient],
    )


class TestDispatcherPush:
    async def test_push_sends_to_first_available_channel(self, db):
        _seed_subscription(db, "u1", "web", "endpoint_url")
        web_ch = _make_channel("web", success=True)
        d = PushDispatcher({"web": web_ch}, db=db)
        results = await d.push("u1", "Title", "Body")
        assert len(results) == 1
        assert results[0].success is True
        web_ch.send.assert_called_once()

    async def test_push_stops_after_first_success(self, db):
        _seed_subscription(db, "u2", "web", "ep")
        _seed_subscription(db, "u2", "email", "u2@example.com")
        web_ch = _make_channel("web", success=True)
        email_ch = _make_channel("email", success=True)
        d = PushDispatcher({"web": web_ch, "email": email_ch}, db=db)
        results = await d.push("u2", "T", "B", channels_preference=["web", "email"])
        web_ch.send.assert_called_once()
        email_ch.send.assert_not_called()

    async def test_push_falls_through_on_failure(self, db):
        _seed_subscription(db, "u3", "web", "ep")
        _seed_subscription(db, "u3", "email", "u3@e.com")
        web_ch = _make_channel("web", success=False)
        email_ch = _make_channel("email", success=True)
        d = PushDispatcher({"web": web_ch, "email": email_ch}, db=db)
        results = await d.push("u3", "T", "B", channels_preference=["web", "email"])
        assert len(results) == 2
        assert results[1].success is True

    async def test_push_unknown_channel_skipped(self, db):
        d = PushDispatcher({}, db=db)
        results = await d.push("u4", "T", "B", channels_preference=["web"])
        assert results == []

    async def test_push_no_recipient_skips(self, db):
        web_ch = _make_channel("web")
        d = PushDispatcher({"web": web_ch}, db=db)
        # No subscription registered
        results = await d.push("u_no_sub", "T", "B")
        web_ch.send.assert_not_called()

    async def test_push_channel_exception_returns_failure(self, db):
        _seed_subscription(db, "u5", "email", "u5@e.com")
        email_ch = MagicMock()
        email_ch.name = "email"
        email_ch.send = AsyncMock(side_effect=PushError("smtp down"))
        d = PushDispatcher({"email": email_ch}, db=db)
        results = await d.push("u5", "T", "B", channels_preference=["email"])
        assert len(results) == 1
        assert results[0].success is False

    async def test_default_channel_preference(self, db):
        _seed_subscription(db, "u6", "web", "ep")
        web_ch = _make_channel("web", success=True)
        d = PushDispatcher({"web": web_ch}, db=db)
        results = await d.push("u6", "T", "B")  # no channels_preference arg
        assert results[0].success is True

    async def test_push_with_deep_link(self, db):
        _seed_subscription(db, "u7", "email", "u7@e.com")
        email_ch = _make_channel("email", success=True)
        d = PushDispatcher({"email": email_ch}, db=db)
        await d.push("u7", "T", "B", channels_preference=["email"], deep_link="stratum://x")
        email_ch.send.assert_called_once_with(
            recipient="u7@e.com", title="T", body="B",
            deep_link="stratum://x", metadata=None,
        )


class TestDispatcherNoDb:
    async def test_push_without_db_returns_empty(self):
        web_ch = _make_channel("web", success=True)
        d = PushDispatcher({"web": web_ch})  # no db
        results = await d.push("u", "T", "B")
        assert results == []

    async def test_get_recipient_without_db_returns_none(self):
        d = PushDispatcher({})
        r = await d._get_recipient("u", "web")
        assert r is None


class TestDispatcherRegisterChannel:
    async def test_register_and_use(self, db):
        d = PushDispatcher({}, db=db)
        ch = _make_channel("sms")
        d.register_channel(ch)
        assert "sms" in d._channels

    async def test_unregister(self):
        ch = _make_channel("sms")
        d = PushDispatcher({"sms": ch})
        d.unregister_channel("sms")
        assert "sms" not in d._channels

    async def test_unregister_nonexistent_noop(self):
        d = PushDispatcher({})
        d.unregister_channel("nonexistent")  # must not raise
