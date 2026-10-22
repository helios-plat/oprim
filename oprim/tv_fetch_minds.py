"""Atomic OPrim: fetch raw Minds community content without sentiment scoring."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_minds"]
from oprim._tv_transport import TVTransport

def fetch_tv_minds(symbol: str, *, sort: str = "recent", limit: int = 20, transport: TVTransport | None = None) -> dict:
    if not isinstance(symbol, str) or ":" not in symbol:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "symbol must include exchange prefix"})
    exchange, ticker = (part.strip().upper() for part in symbol.split(":", 1))
    if not exchange or not ticker or any(c in exchange + ticker for c in "/?# ") or sort not in {"recent", "popular", "trending"} or not 1 <= limit <= 100:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid symbol, sort or limit"})
    return (transport or TVTransport()).request_json("GET", "https://www.tradingview.com/api/v1/minds/",
        params={"symbol": f"{exchange}:{ticker}", "limit": limit, "sort": sort}, ttl_seconds=10)
