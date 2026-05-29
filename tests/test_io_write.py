"""Tests for io_write oprims."""

from __future__ import annotations

import pytest

from oprim.io_write import (
    WriteError,
    clear_event,
    is_deduped,
    refresh_view,
    send_alert,
    store_news,
    upsert_canonical_metric,
    upsert_equity_series,
    upsert_events,
    write_event,
    write_rows,
)


class MockDb:
    def __init__(self, one=None, exec_=1):
        self._one = one
        self._exec = exec_
        self.calls = []

    async def fetch_one(self, q, p=None):
        self.calls.append(("fetch_one", q, p))
        return self._one

    async def fetch_all(self, q, p=None):
        self.calls.append(("fetch_all", q, p))
        return []

    async def execute(self, q, p=None):
        self.calls.append(("execute", q, p))
        return self._exec


class FailDb:
    async def fetch_one(self, q, p=None):
        raise RuntimeError("db down")

    async def fetch_all(self, q, p=None):
        raise RuntimeError("db down")

    async def execute(self, q, p=None):
        raise RuntimeError("db down")


class MockCache:
    def __init__(self):
        self.published = []

    async def get(self, key):
        return None

    async def set(self, key, value, ex=None):
        pass

    async def publish(self, channel, message):
        self.published.append((channel, message))


# --- write_rows ---


@pytest.mark.asyncio
async def test_write_rows_success():
    db = MockDb()
    r = await write_rows(db=db, rows=[("2024", "BTC", "ETH", 30, None, 0.85, 28, "ok")])
    assert r == 1


@pytest.mark.asyncio
async def test_write_rows_empty():
    r = await write_rows(db=MockDb(), rows=[])
    assert r == 0


@pytest.mark.asyncio
async def test_write_rows_error():
    with pytest.raises(WriteError):
        await write_rows(db=FailDb(), rows=[("x",)])


@pytest.mark.asyncio
async def test_write_rows_multiple():
    r = await write_rows(db=MockDb(), rows=[("a",), ("b",), ("c",)])
    assert r == 3


@pytest.mark.asyncio
async def test_write_rows_custom_table():
    db = MockDb()
    await write_rows(db=db, rows=[("x",)], table="my_table")
    assert "my_table" in db.calls[0][1]


# --- store_news ---


@pytest.mark.asyncio
async def test_store_news_success():
    r = await store_news(
        db=MockDb(exec_=1),
        items=[{"title": "t", "url": "u", "source": "s", "published_at": "p", "summary": "x"}],
    )
    assert r == 1


@pytest.mark.asyncio
async def test_store_news_empty():
    r = await store_news(db=MockDb(), items=[])
    assert r == 0


@pytest.mark.asyncio
async def test_store_news_error():
    with pytest.raises(WriteError):
        await store_news(db=FailDb(), items=[{"title": "t"}])


@pytest.mark.asyncio
async def test_store_news_multiple():
    r = await store_news(db=MockDb(exec_=1), items=[{"title": "a"}, {"title": "b"}])
    assert r == 2


@pytest.mark.asyncio
async def test_store_news_zero_inserts():
    r = await store_news(db=MockDb(exec_=0), items=[{"title": "dup"}])
    assert r == 0


# --- upsert_events ---


@pytest.mark.asyncio
async def test_upsert_events_success():
    r = await upsert_events(
        db=MockDb(),
        rows=[
            {
                "event_date": "2024-01-01",
                "event_type": "earnings",
                "title": "AAPL",
                "impact": "high",
            }
        ],
    )
    assert r == 1


@pytest.mark.asyncio
async def test_upsert_events_empty():
    r = await upsert_events(db=MockDb(), rows=[])
    assert r == 0


@pytest.mark.asyncio
async def test_upsert_events_error():
    with pytest.raises(WriteError):
        await upsert_events(db=FailDb(), rows=[{"event_date": "x"}])


# --- upsert_canonical_metric ---


@pytest.mark.asyncio
async def test_upsert_canonical_metric_success():
    r = await upsert_canonical_metric(
        db=MockDb(one={"id": "uuid-123"}), agent="h", strategy="f", market="crypto"
    )
    assert r == "uuid-123"


@pytest.mark.asyncio
async def test_upsert_canonical_metric_error():
    with pytest.raises(WriteError):
        await upsert_canonical_metric(db=FailDb(), agent="h")


# --- upsert_equity_series ---


@pytest.mark.asyncio
async def test_upsert_equity_series_success():
    db = MockDb()
    await upsert_equity_series(
        db=db, agent="h", strategy="f", market="c", as_of_date="2024-01-01", equity=10000.0
    )
    assert len(db.calls) == 1


@pytest.mark.asyncio
async def test_upsert_equity_series_error():
    with pytest.raises(WriteError):
        await upsert_equity_series(
            db=FailDb(), agent="h", strategy="f", market="c", as_of_date="x", equity=0
        )


# --- write_event / clear_event / is_deduped ---


@pytest.mark.asyncio
async def test_write_event_success():
    r = await write_event(
        db=MockDb(one={"id": "ev-1"}), preset="crash", trigger_data={"x": 1}, summary="BTC -10%"
    )
    assert r == "ev-1"


@pytest.mark.asyncio
async def test_write_event_error():
    with pytest.raises(WriteError):
        await write_event(db=FailDb(), preset="x", trigger_data={}, summary="")


@pytest.mark.asyncio
async def test_clear_event_success():
    db = MockDb()
    await clear_event(db=db, event_id="ev-1", reason="recovered")
    assert len(db.calls) == 1


@pytest.mark.asyncio
async def test_clear_event_error():
    with pytest.raises(WriteError):
        await clear_event(db=FailDb(), event_id="x", reason="y")


@pytest.mark.asyncio
async def test_is_deduped_false():
    r = await is_deduped(db=MockDb(one=None), preset="x")
    assert r is False


@pytest.mark.asyncio
async def test_is_deduped_true():
    r = await is_deduped(db=MockDb(one={"1": 1}), preset="x")
    assert r is True


# --- refresh_view ---


@pytest.mark.asyncio
async def test_refresh_view_success():
    r = await refresh_view(db=MockDb(), view="mv_test")
    assert r["status"] == "ok"
    assert r["view"] == "mv_test"


@pytest.mark.asyncio
async def test_refresh_view_error():
    with pytest.raises(WriteError):
        await refresh_view(db=FailDb(), view="x")


# --- send_alert ---


@pytest.mark.asyncio
async def test_send_alert_success():
    c = MockCache()
    await send_alert(cache=c, level="warn", title="Stale", message="data stale")
    assert len(c.published) == 1
    assert "alerts" in c.published[0][0]


@pytest.mark.asyncio
async def test_send_alert_content():
    c = MockCache()
    await send_alert(cache=c, level="critical", title="T", message="M")
    import json

    payload = json.loads(c.published[0][1])
    assert payload["level"] == "critical"
