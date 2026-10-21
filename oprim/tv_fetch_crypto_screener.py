"""Atomic OPrim: fetch crypto screener rows."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_crypto_screener"]
from oprim._tv_transport import TVTransport
COLUMNS = ("name", "close", "change", "volume", "exchange", "currency", "base_currency_desc", "Recommend.All")

def fetch_tv_crypto_screener(*, limit: int = 20, sort_by: str = "volume", sort_order: str = "desc", transport: TVTransport | None = None) -> dict:
    if not 1 <= limit <= 200 or not sort_by or sort_order not in {"asc", "desc"}:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid screener bounds or sort"})
    body = {"columns": list(COLUMNS), "filter": [], "ignore_unknown_fields": True, "options": {"lang": "en"},
            "range": [0, limit], "sort": {"sortBy": sort_by, "sortOrder": sort_order},
            "symbols": {"query": {"types": []}, "tickers": []}}
    out = (transport or TVTransport()).request_json("POST", "https://scanner.tradingview.com/crypto/scan", body=body, ttl_seconds=12)
    return _normalise(out, COLUMNS)

def _normalise(out: dict, columns: tuple[str, ...]) -> dict:
    if out.get("status") not in {"success", "degraded"}: return out
    payload = out.get("data"); rows = []
    for row in payload.get("data", []) if isinstance(payload, dict) else []:
        values = row.get("d", []) if isinstance(row, dict) else []
        if isinstance(values, list):
            item = {"symbol": row.get("s", "")}; item.update({col: values[i] for i, col in enumerate(columns) if i < len(values)}); rows.append(item)
    result = dict(out); result["data"] = {"total_count": payload.get("totalCount") if isinstance(payload, dict) else None, "rows": rows}
    return result
