"""Atomic OPrim: scan listings for one crypto base/quote pair across exchanges."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_symbol_markets"]
from oprim._tv_transport import TVTransport
COLUMNS = ("name", "close", "change", "volume", "exchange", "currency", "base_currency_desc", "type")

def fetch_tv_symbol_markets(base: str, *, quote: str = "USDT", limit: int = 100, transport: TVTransport | None = None) -> dict:
    base = base.strip().upper() if isinstance(base, str) else ""
    quote = quote.strip().upper() if isinstance(quote, str) else ""
    if not base or not quote or any(c in base + quote for c in ":/?# ") or not 1 <= limit <= 200:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid base, quote or limit"})
    body = {"columns": list(COLUMNS), "filter": [{"left": "base_currency_desc", "operation": "equal", "right": base},
             {"left": "currency", "operation": "equal", "right": quote}], "ignore_unknown_fields": True,
             "options": {"lang": "en"}, "range": [0, limit], "sort": {"sortBy": "volume", "sortOrder": "desc"},
             "symbols": {"query": {"types": []}, "tickers": []}}
    out = (transport or TVTransport()).request_json("POST", "https://scanner.tradingview.com/crypto/scan", body=body, ttl_seconds=12)
    if out.get("status") not in {"success", "degraded"}: return out
    payload = out.get("data"); rows = []
    for row in payload.get("data", []) if isinstance(payload, dict) else []:
        values = row.get("d", []) if isinstance(row, dict) else []
        if isinstance(values, list):
            item = {"symbol": row.get("s", "")}; item.update({col: values[i] for i, col in enumerate(COLUMNS) if i < len(values)}); rows.append(item)
    out = dict(out); out["data"] = {"total_count": payload.get("totalCount") if isinstance(payload, dict) else None, "rows": rows}
    return out
