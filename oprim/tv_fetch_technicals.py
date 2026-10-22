"""Atomic OPrim: fetch scanner technicals and upstream Recommend.All."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_technicals"]
from oprim._tv_transport import TVTransport
DEFAULT_FIELDS = ("Recommend.All", "Recommend.Other", "Recommend.MA", "RSI", "MACD.macd", "MACD.signal", "ADX")

def fetch_tv_technicals(exchange: str, symbol: str, *, fields=DEFAULT_FIELDS, transport: TVTransport | None = None) -> dict:
    exchange = exchange.strip().upper() if isinstance(exchange, str) else ""
    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not exchange or not symbol or any(c in exchange + symbol for c in ":/?# ") or not fields:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid exchange, symbol or fields"})
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Fields must be non-empty strings"})
    params = {"symbol": f"{exchange}:{symbol}", "fields": ",".join(fields).replace("+", "%2B"), "no_404": "true"}
    out = (transport or TVTransport()).request_json("GET", "https://scanner.tradingview.com/symbol", params=params, ttl_seconds=10)
    if out["status"] in {"success", "degraded"}:
        data = out.get("data")
        out = dict(out)
        out["data"] = {str(k).split("|", 1)[0]: v for k, v in data.items()} if isinstance(data, dict) else {}
    return out
