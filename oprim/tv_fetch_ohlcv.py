"""Atomic OPrim: return actual TradingView websocket OHLCV bars."""
__oprim_layer__ = "provider"
__oprim_exports__ = ["fetch_tv_ohlcv", "parse_tv_ws_frames", "parse_tv_ohlcv_packet"]
import json
import re
import secrets
import time
from oprim._tv_transport import TVTransport

_URL = "wss://data.tradingview.com/socket.io/websocket?from=chart%2F&type=chart"

def parse_tv_ws_frames(message: str) -> list[dict]:
    """Parse TradingView ~m~length~m~ frames; silently ignore heartbeat/non-JSON frames."""
    packets = []
    for raw in re.split(r"~m~\d+~m~", message):
        if not raw or raw.startswith("~h~"):
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            packets.append(payload)
    return packets

def parse_tv_ohlcv_packet(packet: dict, symbol: str, timeframe: str) -> list[dict]:
    """Decode timescale_update bars using the protocol's timestamp/OHLCV ordering."""
    if packet.get("m") != "timescale_update":
        return []
    params = packet.get("p")
    if not isinstance(params, list) or len(params) < 2 or not isinstance(params[1], dict):
        return []
    series = params[1].get("sds_1", {})
    rows = series.get("s", []) if isinstance(series, dict) else []
    bars = []
    for row in rows:
        values = row.get("v") if isinstance(row, dict) else None
        if not isinstance(values, list) or len(values) < 6:
            continue
        bars.append({"symbol": symbol, "timeframe": timeframe, "timestamp": int(values[0]),
                     "open": values[1], "high": values[2], "low": values[3],
                     "close": values[4], "volume": values[5]})
    return bars

def fetch_tv_ohlcv(exchange_symbol: str, *, timeframe: str = "1", limit: int = 10,
                   timeout: float = 8, websocket_factory=None) -> dict:
    """Read a live packet. Public unauthenticated token is used; access failures stay failures."""
    if not isinstance(exchange_symbol, str) or ":" not in exchange_symbol:
        return TVTransport.result("failed", error={"type": "ValueError", "message": "symbol must include exchange prefix"})
    exchange, ticker = (v.strip().upper() for v in exchange_symbol.split(":", 1))
    if (not exchange or not ticker or any(c in exchange + ticker for c in "/?# ") or
            timeframe not in {"1", "5", "15", "30", "60", "120", "240", "1D", "1W"} or
            not 1 <= limit <= 500 or timeout <= 0):
        return TVTransport.result("failed", error={"type": "ValueError", "message": "Invalid symbol/timeframe/limit/timeout"})
    socket = None
    symbol = f"{exchange}:{ticker}"
    try:
        if websocket_factory is None:
            import websocket
            websocket_factory = websocket.create_connection
        socket = websocket_factory(_URL, timeout=timeout, header=[
            "Origin: https://www.tradingview.com",
            "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/155.0 Safari/537.36"])
        def send(method: str, params: list) -> None:
            raw = json.dumps({"m": method, "p": params}, separators=(",", ":"))
            socket.send(f"~m~{len(raw)}~m~{raw}")
        chart, quote = "cs_" + secrets.token_hex(6), "qs_" + secrets.token_hex(6)
        send("set_auth_token", ["unauthorized_user_token"])
        send("set_locale", ["en", "US"])
        send("chart_create_session", [chart, ""])
        send("quote_create_session", [quote])
        send("quote_set_fields", [quote, "lp", "ch", "chp", "volume", "lp_time", "description", "exchange"])
        send("resolve_symbol", [chart, "sds_sym_1", "=" + json.dumps(
            {"symbol": symbol, "adjustment": "splits"}, separators=(",", ":"))])
        send("create_series", [chart, "sds_1", "s1", "sds_sym_1", timeframe, limit, ""])
        send("quote_add_symbols", [quote, symbol])
        send("quote_fast_symbols", [quote, symbol])
        deadline, bars = time.monotonic() + timeout, []
        while time.monotonic() < deadline:
            frame = socket.recv()
            if isinstance(frame, str) and re.fullmatch(r"~m~\d+~m~~h~\d+", frame):
                try:
                    socket.send(frame)
                except Exception:
                    pass
                continue
            for packet in parse_tv_ws_frames(frame):
                bars = parse_tv_ohlcv_packet(packet, symbol, timeframe)
                if bars:
                    break
            if bars:
                break
        if not bars:
            return TVTransport.result("failed", error={"type": "TimeoutError", "message": "No OHLCV bars received before timeout"})
        return TVTransport.result("success", {"symbol": symbol, "timeframe": timeframe, "bars": bars})
    except Exception as exc:
        return TVTransport.result("failed", error={"type": type(exc).__name__, "message": str(exc)[:500]})
    finally:
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
