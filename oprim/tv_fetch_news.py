"""Atomic OPrim: fetch and sort the public TV headlines for one symbol."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_news"]
from oprim._tv_transport import TVTransport

def fetch_tv_news(exchange: str, symbol: str, *, limit: int = 20, language: str = "en", transport: TVTransport | None = None) -> dict:
    exchange = exchange.strip().upper() if isinstance(exchange, str) else ""
    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not exchange or not symbol or any(c in exchange + symbol for c in ":/?# ") or not 1 <= limit <= 100:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid symbol or limit"})
    params = {"client": "web", "lang": language, "area": "", "provider": "", "section": "", "streaming": "", "symbol": f"{exchange}:{symbol}"}
    out = (transport or TVTransport()).request_json("GET", "https://news-headlines.tradingview.com/v2/view/headlines/symbol", params=params, ttl_seconds=30)
    if out.get("status") not in {"success", "degraded"}: return out
    data = out.get("data"); items = data.get("items", []) if isinstance(data, dict) else []
    items = sorted((i for i in items if isinstance(i, dict)), key=lambda i: i.get("published", 0) or 0, reverse=True)
    out = dict(out); out["data"] = {"items": items[:limit], "total_count": len(items)}
    return out
