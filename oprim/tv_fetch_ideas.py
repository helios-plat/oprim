"""Atomic OPrim: fetch one page of public TradingView Ideas."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_ideas"]
from oprim._tv_transport import TVTransport

def fetch_tv_ideas(symbol: str, *, sort: str = "recent", page: int = 1, transport: TVTransport | None = None) -> dict:
    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or any(c in symbol for c in ":/?# ") or sort not in {"popular", "recent"} or not 1 <= page <= 20:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid symbol, sort or page"})
    path = f"{symbol}/ideas/" if page == 1 else f"{symbol}/ideas/page-{page}/"
    params = {"component-data-only": "1", "sort": sort}
    out = (transport or TVTransport()).request_json("GET", "https://www.tradingview.com/symbols/" + path, params=params, ttl_seconds=60)
    if out.get("status") not in {"success", "degraded"}: return out
    data = out.get("data"); data = data.get("data", {}) if isinstance(data, dict) else {}
    area = data.get("ideas", {}).get("data", {}) if isinstance(data, dict) else {}
    items = area.get("items", []) if isinstance(area, dict) else []
    out = dict(out); out["data"] = {"items": [item for item in items if isinstance(item, dict)], "total_count": len(items) if isinstance(items, list) else 0}
    return out
