"""Tests for io_fetch oprims."""

from __future__ import annotations

import pytest

from oprim.io_fetch import (
    FetchError,
    fetch_btc_spy_corr,
    fetch_crypto,
    fetch_current_price,
    fetch_decision_count,
    fetch_equity_series,
    fetch_prefs,
    fetch_regime,
    fetch_regime_crisis_flips,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    get_active_event,
    get_etf_weight_modifier,
    get_previous_30d,
    get_regime_by_date,
    get_stablecoin_change_7d,
    get_symbol_funding_rate,
    get_symbol_oi_change_7d,
)


class MockHttp:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def get(self, url, **kw):
        if self._error:
            raise self._error
        return self._response

    async def post(self, url, **kw):
        if self._error:
            raise self._error
        return self._response


class MockDb:
    def __init__(self, one=None, all_=None, exec_=0):
        self._one = one
        self._all = all_ or []
        self._exec = exec_

    async def fetch_one(self, q, p=None):
        return self._one

    async def fetch_all(self, q, p=None):
        return self._all

    async def execute(self, q, p=None):
        return self._exec


class MockCache:
    def __init__(self, data=None):
        self._data = data or {}

    async def get(self, key):
        return self._data.get(key)

    async def set(self, key, value, ex=None):
        self._data[key] = value

    async def publish(self, channel, message):
        pass


# --- fetch_yahoo_history ---


@pytest.mark.asyncio
async def test_fetch_yahoo_history_success():
    c = MockHttp([{"date": "2024-01-01", "close": 42000}])
    r = await fetch_yahoo_history(client=c, symbol="BTC-USD", days=30)
    assert r == [("2024-01-01", 42000.0)]


@pytest.mark.asyncio
async def test_fetch_yahoo_history_empty():
    r = await fetch_yahoo_history(client=MockHttp([]), symbol="X", days=1)
    assert r == []


@pytest.mark.asyncio
async def test_fetch_yahoo_history_error():
    with pytest.raises(FetchError):
        await fetch_yahoo_history(client=MockHttp(error=RuntimeError("net")), symbol="X")


@pytest.mark.asyncio
async def test_fetch_yahoo_history_none_response():
    r = await fetch_yahoo_history(client=MockHttp(None), symbol="X")
    assert r == []


@pytest.mark.asyncio
async def test_fetch_yahoo_history_dict_response():
    r = await fetch_yahoo_history(client=MockHttp({"error": "bad"}), symbol="X")
    assert r == []


# --- fetch_yahoo_quote ---


@pytest.mark.asyncio
async def test_fetch_yahoo_quote_success():
    r = await fetch_yahoo_quote(client=MockHttp({"price": 450.5}), symbol="SPY")
    assert r == 450.5


@pytest.mark.asyncio
async def test_fetch_yahoo_quote_none():
    r = await fetch_yahoo_quote(client=MockHttp(None), symbol="X")
    assert r is None


@pytest.mark.asyncio
async def test_fetch_yahoo_quote_no_price():
    r = await fetch_yahoo_quote(client=MockHttp({"other": 1}), symbol="X")
    assert r is None


@pytest.mark.asyncio
async def test_fetch_yahoo_quote_error():
    with pytest.raises(FetchError):
        await fetch_yahoo_quote(client=MockHttp(error=RuntimeError()), symbol="X")


@pytest.mark.asyncio
async def test_fetch_current_price_delegates():
    r = await fetch_current_price(client=MockHttp({"price": 100.0}), asset="BTC")
    assert r == 100.0


# --- fetch_crypto ---


@pytest.mark.asyncio
async def test_fetch_crypto_success():
    r = await fetch_crypto(
        client=MockHttp({"bitcoin": {"usd": 50000, "usd_24h_change": 2.5, "usd_24h_vol": 1e9}})
    )
    assert len(r) == 1
    assert r[0]["price"] == 50000


@pytest.mark.asyncio
async def test_fetch_crypto_empty():
    r = await fetch_crypto(client=MockHttp(None))
    assert r == []


@pytest.mark.asyncio
async def test_fetch_crypto_error():
    with pytest.raises(FetchError):
        await fetch_crypto(client=MockHttp(error=RuntimeError()))


# --- cache-based reads ---


@pytest.mark.asyncio
async def test_get_stablecoin_change_7d():
    c = MockCache({"environ:flow:stablecoin_change_7d": "0.015"})
    r = await get_stablecoin_change_7d(cache=c)
    assert r == 0.015


@pytest.mark.asyncio
async def test_get_stablecoin_change_7d_miss():
    r = await get_stablecoin_change_7d(cache=MockCache())
    assert r is None


@pytest.mark.asyncio
async def test_get_etf_weight_modifier():
    c = MockCache({"environ:etf:net_flow_7d:BTC-USDT": "0.7"})
    r = await get_etf_weight_modifier(cache=c, symbol="BTC-USDT")
    assert r == 0.7


@pytest.mark.asyncio
async def test_get_etf_weight_modifier_default():
    r = await get_etf_weight_modifier(cache=MockCache())
    assert r == 1.0


@pytest.mark.asyncio
async def test_get_symbol_funding_rate():
    c = MockCache({"external:binance:funding_rate:BTC-USDT": '{"rate": 0.0001}'})
    r = await get_symbol_funding_rate(cache=c, symbol="BTC-USDT")
    assert r == 0.0001


@pytest.mark.asyncio
async def test_get_symbol_funding_rate_miss():
    r = await get_symbol_funding_rate(cache=MockCache(), symbol="X")
    assert r is None


@pytest.mark.asyncio
async def test_get_symbol_oi_change_7d():
    c = MockCache({"external:binance:oi_history:BTC-USDT": '{"oi_change_7d": 0.15}'})
    r = await get_symbol_oi_change_7d(cache=c, symbol="BTC-USDT")
    assert r == 0.15


@pytest.mark.asyncio
async def test_fetch_regime_success():
    c = MockCache({"regime:current": '{"regime": "bull_low_vol", "confidence": 0.8}'})
    r = await fetch_regime(cache=c)
    assert r["regime"] == "bull_low_vol"


@pytest.mark.asyncio
async def test_fetch_regime_miss():
    r = await fetch_regime(cache=MockCache())
    assert r is None


# --- DB reads ---


@pytest.mark.asyncio
async def test_fetch_btc_spy_corr():
    r = await fetch_btc_spy_corr(db=MockDb(one={"correlation": 0.45}))
    assert r == 0.45


@pytest.mark.asyncio
async def test_fetch_btc_spy_corr_none():
    r = await fetch_btc_spy_corr(db=MockDb())
    assert r is None


@pytest.mark.asyncio
async def test_fetch_regime_crisis_flips():
    r = await fetch_regime_crisis_flips(db=MockDb(all_=[{"time": "t", "regime": "bear_high_vol"}]))
    assert len(r) == 1


@pytest.mark.asyncio
async def test_get_active_event_found():
    r = await get_active_event(db=MockDb(one={"id": "uuid", "preset": "x"}), preset="x")
    assert r["id"] == "uuid"


@pytest.mark.asyncio
async def test_get_active_event_none():
    r = await get_active_event(db=MockDb(), preset="x")
    assert r is None


@pytest.mark.asyncio
async def test_get_regime_by_date():
    r = await get_regime_by_date(db=MockDb(all_=[{"dt": "2024-01-01", "regime": "bull_low_vol"}]))
    assert r == {"2024-01-01": "bull_low_vol"}


@pytest.mark.asyncio
async def test_get_previous_30d():
    r = await get_previous_30d(db=MockDb(one={"correlation": 0.85}), asset_a="BTC", asset_b="ETH")
    assert r == 0.85


@pytest.mark.asyncio
async def test_fetch_equity_series():
    r = await fetch_equity_series(
        db=MockDb(all_=[{"timestamp": "t", "equity_usd": 10000}]),
        account_id="a",
        since="2024-01-01",
    )
    assert len(r) == 1


@pytest.mark.asyncio
async def test_fetch_decision_count():
    r = await fetch_decision_count(
        db=MockDb(one={"cnt": 5}), account_id="a", since="2024-01-01", as_of="2024-02-01"
    )
    assert r == 5


@pytest.mark.asyncio
async def test_fetch_prefs():
    r = await fetch_prefs(db=MockDb(all_=[{"level": "info", "telegram": True}]))
    assert "info" in r
