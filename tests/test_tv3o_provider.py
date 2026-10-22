import json
from obase.tv_cache import SlidingWindowRateGate, TTLCache
from oprim._tv_transport import TVTransport
from oprim.tv_fetch_technicals import fetch_tv_technicals
from oprim.tv_fetch_crypto_screener import fetch_tv_crypto_screener
from oprim.tv_fetch_symbol_markets import fetch_tv_symbol_markets
from oprim.tv_fetch_news import fetch_tv_news
from oprim.tv_fetch_minds import fetch_tv_minds
from oprim.tv_fetch_ideas import fetch_tv_ideas
from oprim.tv_fetch_ohlcv import fetch_tv_ohlcv, parse_tv_ohlcv_packet, parse_tv_ws_frames

class Response:
    def __init__(self, data, status=200, text=""):
        self.data, self.status_code, self.text = data, status, text
    def json(self): return self.data
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(f"HTTP {self.status_code}")

class Session:
    def __init__(self, response=None, error=None): self.response, self.error, self.calls = response, error, []
    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self.error: raise self.error
        return self.response
    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.error: raise self.error
        return self.response

def transport(session, *, cache=None, ttl=10, stale=30):
    return TVTransport(session=session, cache=cache if cache is not None else TTLCache(), gate=SlidingWindowRateGate(0),
                       retries=0, ttl_seconds=ttl, stale_seconds=stale)

def test_technicals_includes_upstream_recommend_all():
    session=Session(Response({"RSI":55.2,"Recommend.All":.42}))
    out=fetch_tv_technicals("BINANCE","BTCUSDT",transport=transport(session))
    assert out["status"]=="success" and out["data"]["Recommend.All"]==.42
    assert session.calls[0][0]=="GET" and session.calls[0][2]["timeout"]==5

def test_technicals_rejects_bad_symbols():
    assert fetch_tv_technicals("BINANCE","BTC/USDT")["status"]=="failed"

def test_screener_transforms_provider_rows():
    data={"totalCount":1,"data":[{"s":"BINANCE:BTCUSDT","d":["BTCUSDT",80000,1.2,123,"Binance","USDT","BTC",.5]}]}
    session=Session(Response(data)); out=fetch_tv_crypto_screener(limit=1,transport=transport(session))
    assert out["data"]["rows"][0]["close"]==80000 and out["data"]["rows"][0]["Recommend.All"]==.5
    assert session.calls[0][0]=="POST"

def test_symbol_markets_filters_pair():
    data={"totalCount":1,"data":[{"s":"BINANCE:BTCUSDT","d":["BTCUSDT",80000,1.2,123,"Binance","USDT","BTC","spot"]}]}
    session=Session(Response(data)); out=fetch_tv_symbol_markets("BTC",quote="USDT",transport=transport(session))
    assert out["data"]["rows"][0]["symbol"]=="BINANCE:BTCUSDT"
    assert {f["left"] for f in session.calls[0][2]["json"]["filter"]}=={"base_currency_desc","currency"}

def test_news_is_sorted_newest_first():
    data={"items":[{"title":"old","published":1},{"title":"new","published":3}]}
    out=fetch_tv_news("BINANCE","BTCUSDT",limit=1,transport=transport(Session(Response(data))))
    assert out["data"]["items"][0]["title"]=="new"

def test_minds_requires_exchange_prefix_without_request():
    session=Session(Response({})); out=fetch_tv_minds("BTCUSDT",transport=transport(session))
    assert out["status"]=="failed" and not session.calls

def test_minds_forwards_exchange_prefixed_symbol():
    session=Session(Response({"results":[{"text":"raw content"}]}))
    out=fetch_tv_minds("BINANCE:BTCUSDT",transport=transport(session))
    assert out["status"]=="success" and session.calls[0][2]["params"]["symbol"]=="BINANCE:BTCUSDT"

def test_ideas_challenge_is_failure():
    out=fetch_tv_ideas("BTCUSD",transport=transport(Session(Response({},text="<title>Captcha Challenge</title>"))))
    assert out["status"]=="failed" and "CAPTCHA" in out["error"]["message"]

def test_ideas_response_shape():
    data={"data":{"ideas":{"data":{"items":[{"name":"idea"}]}}}}
    out=fetch_tv_ideas("BTCUSD",transport=transport(Session(Response(data))))
    assert out["data"]["items"][0]["name"]=="idea"

def test_network_failure_is_a_structured_failure():
    out=fetch_tv_crypto_screener(transport=transport(Session(error=OSError("offline"))))
    assert out["status"]=="failed" and out["data"] is None and out["error"]["message"]=="offline"

def test_stale_fallback_is_explicit():
    now=[0.0]; cache=TTLCache(clock=lambda:now[0]); session=Session(Response({"RSI":50}))
    trans=transport(session,cache=cache,ttl=1,stale=20)
    fetch_tv_technicals("BINANCE","BTCUSDT",transport=trans); now[0]=11; session.error=OSError("offline")
    out=fetch_tv_technicals("BINANCE","BTCUSDT",transport=trans)
    assert out["status"]=="degraded" and out["stale"] is True and out["data"]["RSI"]==50

def test_websocket_frame_parser():
    packet={"m":"timescale_update","p":["cs",{"sds_1":{"s":[{"i":0,"v":[100,1,2,.5,1.5,7]}]}}]}
    raw=json.dumps(packet,separators=(",",":"))
    assert parse_tv_ws_frames(f"~m~{len(raw)}~m~{raw}")==[packet]

def test_ohlcv_parser_maps_timestamp_and_prices():
    packet={"m":"timescale_update","p":["cs",{"sds_1":{"s":[{"v":[100,1,2,.5,1.5,7]}]}}]}
    out=parse_tv_ohlcv_packet(packet,"BINANCE:BTCUSDT","1")[0]
    assert out=={"symbol":"BINANCE:BTCUSDT","timeframe":"1","timestamp":100,"open":1,"high":2,"low":.5,"close":1.5,"volume":7}

def test_ohlcv_rejects_unqualified_symbol_without_network():
    assert fetch_tv_ohlcv("BTCUSDT")["status"]=="failed"
