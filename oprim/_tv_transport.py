"""Shared provider infrastructure; no TradingView business policy."""
from __future__ import annotations
__oprim_layer__ = "infra"
__oprim_exports__ = ["TVTransport"]
import json, os, threading, time
from datetime import datetime, timezone
from typing import Any
import requests
from obase.tv_cache import SlidingWindowRateGate, TTLCache

class TVTransport:
    """Timeout-bounded GET/POST transport with cache, pacing, retries and stale fallback."""
    def __init__(self, *, timeout: float = 5, ttl_seconds: float = 15, stale_seconds: float = 300,
                 retries: int = 1, session: Any = None, cache: TTLCache | None = None,
                 gate: SlidingWindowRateGate | None = None, sleeper=time.sleep, cookie: str | None = None):
        if timeout <= 0 or ttl_seconds < 0 or stale_seconds < 0 or retries < 0:
            raise ValueError("Invalid timeout/cache/retry configuration")
        self.timeout, self.ttl_seconds, self.stale_seconds, self.retries = timeout, ttl_seconds, stale_seconds, retries
        self.session = session if session is not None else requests.Session()
        self.cache = cache if cache is not None else TTLCache(max_entries=512)
        self.gate, self.sleeper = gate or SlidingWindowRateGate(.15), sleeper
        self._lock = threading.RLock()
        self.headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/155.0 Safari/537.36",
                        "Accept": "application/json"}
        value = cookie if cookie is not None else os.getenv("TRADINGVIEW_COOKIE", "")
        if value and value.strip(): self.headers["Cookie"] = value.strip()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def result(status: str, data=None, *, error=None, cached=False, stale=False, observed_at=None) -> dict:
        return {"status": status, "source": "tradingview", "data": data, "error": error,
                "cached": cached, "stale": stale, "observed_at": observed_at or TVTransport.now()}

    def request_json(self, method: str, url: str, *, params: dict | None = None, body: dict | None = None,
                     ttl_seconds: float | None = None) -> dict:
        method = method.upper()
        if method not in {"GET", "POST"}:
            return self.result("failed", error={"type": "ValueError", "message": "Only GET and POST are allowed"})
        key = json.dumps({"method": method, "url": url, "params": params, "body": body}, sort_keys=True, default=str)
        found, packed, _ = self.cache.lookup(key)
        if found: return self.result("success", packed["data"], cached=True, observed_at=packed["observed_at"])
        error = None
        for attempt in range(self.retries + 1):
            try:
                self.gate.acquire()
                with self._lock:
                    if method == "GET": response = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
                    else: response = self.session.post(url, params=params, json=body, headers=self.headers, timeout=self.timeout)
                page = getattr(response, "text", "") or ""
                if "captcha challenge" in page.lower():
                    raise RuntimeError("TradingView returned a CAPTCHA challenge")
                if hasattr(response, "raise_for_status"): response.raise_for_status()
                elif not 200 <= int(getattr(response, "status_code", 200)) < 300:
                    raise RuntimeError(f"TradingView HTTP {getattr(response, 'status_code', 'unknown')}")
                data, observed = response.json(), self.now()
                self.cache.store(key, {"data": data, "observed_at": observed},
                                 self.ttl_seconds if ttl_seconds is None else ttl_seconds, self.stale_seconds)
                return self.result("success", data, observed_at=observed)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                if attempt < self.retries: self.sleeper(min(.2 * (2 ** attempt), 1.0))
        found, packed, is_stale = self.cache.lookup(key, allow_stale=True)
        if found and is_stale:
            return self.result("degraded", packed["data"], error=error, cached=True, stale=True, observed_at=packed["observed_at"])
        return self.result("failed", error=error)
