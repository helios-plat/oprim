"""Caddy oprim — 3 atomic Caddy Admin API operations."""

from __future__ import annotations

import time
from datetime import UTC
from typing import Any

import httpx
from pydantic import BaseModel

from oprim._exceptions import (
    OprimConnectionError,
    OprimValidationError,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ReloadResult(BaseModel):
    success: bool
    elapsed_ms: int
    config_id: str | None


class Route(BaseModel):
    id: str | None
    matchers: list[dict[str, Any]]
    handlers: list[dict[str, Any]]
    target_upstream: str | None


class CertStatus(BaseModel):
    domain: str
    issued: bool
    issuer: str | None
    not_before: str | None
    not_after: str | None
    days_until_expiry: int | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _admin_request(
    method: str,
    admin_url: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    timeout_sec: int = 10,
) -> httpx.Response:
    url = admin_url.rstrip("/") + "/" + path.lstrip("/")
    try:
        if method == "GET":
            resp = httpx.get(url, timeout=timeout_sec)
        elif method == "POST":
            resp = httpx.post(url, json=json_body, timeout=timeout_sec)
        else:
            resp = httpx.request(method, url, json=json_body, timeout=timeout_sec)
    except httpx.ConnectError as exc:
        raise OprimConnectionError(f"Cannot reach Caddy admin API at {admin_url}: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise OprimConnectionError(f"Caddy admin API request timed out: {exc}") from exc
    return resp


def _extract_upstream(handlers: list[dict[str, Any]]) -> str | None:
    """Extract first reverse_proxy upstream from handler list."""
    for handler in handlers:
        if handler.get("handler") == "reverse_proxy":
            upstreams = handler.get("upstreams", [])
            if upstreams:
                return str(upstreams[0].get("dial")) if upstreams[0].get("dial") is not None else None
    return None


# ---------------------------------------------------------------------------
# 5.2 caddy_admin_reload
# ---------------------------------------------------------------------------

def caddy_admin_reload(
    *,
    admin_url: str,
    new_config: dict[str, Any],
    timeout_sec: int = 10,
) -> ReloadResult:
    """加载新的 Caddy 配置 (整体替换, atomic).

    Args:
        admin_url: Caddy admin API URL (e.g. "http://localhost:2019")
        new_config: 完整 Caddy JSON config
        timeout_sec: 请求超时

    Returns:
        ReloadResult

    Raises:
        OprimValidationError: config 格式无效 (Caddy 返 400)
        OprimConnectionError
    """
    t0 = time.monotonic()
    resp = _admin_request("POST", admin_url, "/load", json_body=new_config, timeout_sec=timeout_sec)
    elapsed = int((time.monotonic() - t0) * 1000)

    if resp.status_code == 400 or resp.status_code == 422:
        raise OprimValidationError(
            f"Invalid Caddy config (HTTP {resp.status_code}): {resp.text[:200]}"
        )
    if not resp.is_success:
        raise OprimConnectionError(f"Caddy /load returned {resp.status_code}: {resp.text[:200]}")

    config_id = resp.headers.get("Etag") or resp.headers.get("X-Config-Id")

    return ReloadResult(
        success=True,
        elapsed_ms=elapsed,
        config_id=config_id,
    )


# ---------------------------------------------------------------------------
# 5.3 caddy_routes_list
# ---------------------------------------------------------------------------

def caddy_routes_list(
    *,
    admin_url: str,
    server_name: str = "srv0",
    timeout_sec: int = 5,
) -> list[Route]:
    """列出当前 Caddy 所有路由 (从 config tree 提取).

    Args:
        admin_url: Caddy admin API URL
        server_name: Caddy server block name (default "srv0")
        timeout_sec: 请求超时

    Returns:
        Route 列表

    Raises:
        OprimConnectionError
    """
    path = f"/config/apps/http/servers/{server_name}/routes"
    resp = _admin_request("GET", admin_url, path, timeout_sec=timeout_sec)

    if resp.status_code == 404:
        return []
    if not resp.is_success:
        raise OprimConnectionError(
            f"Caddy config API returned {resp.status_code}: {resp.text[:200]}"
        )

    raw_routes = resp.json()
    if not isinstance(raw_routes, list):
        return []

    result = []
    for r in raw_routes:
        handlers = r.get("handle", [])
        result.append(Route(
            id=r.get("@id"),
            matchers=r.get("match", []),
            handlers=handlers,
            target_upstream=_extract_upstream(handlers),
        ))
    return result


# ---------------------------------------------------------------------------
# 5.4 caddy_certificates_status
# ---------------------------------------------------------------------------

def caddy_certificates_status(
    *,
    admin_url: str,
    domain: str,
    timeout_sec: int = 5,
) -> CertStatus:
    """查 Caddy 为指定域名管理的证书状态.

    Note:
        Caddy does not expose a direct per-domain certificate query endpoint.
        This implementation queries /pki/ca/local and the certificates list
        endpoint. If Caddy does not manage TLS for the domain, returns
        issued=False with null fields.

    Args:
        admin_url: Caddy admin API URL
        domain: 域名
        timeout_sec: 请求超时

    Returns:
        CertStatus

    Raises:
        OprimConnectionError
    """
    from datetime import datetime

    resp = _admin_request("GET", admin_url, "/certificates", timeout_sec=timeout_sec)

    if resp.status_code == 404:
        # Caddy version without /certificates endpoint
        return CertStatus(
            domain=domain,
            issued=False,
            issuer=None,
            not_before=None,
            not_after=None,
            days_until_expiry=None,
        )
    if not resp.is_success:
        raise OprimConnectionError(f"Caddy /certificates returned {resp.status_code}")

    certs = resp.json()
    if not isinstance(certs, list):
        certs = []

    # Find certificate matching domain
    for cert in certs:
        names = cert.get("names", [])
        if domain in names or f"*.{'.'.join(domain.split('.')[1:])}" in names:
            not_after_raw = cert.get("not_after") or cert.get("expiry")
            not_before_raw = cert.get("not_before") or cert.get("issued")
            days_left = None
            if not_after_raw:
                try:
                    exp = datetime.fromisoformat(not_after_raw.replace("Z", "+00:00"))
                    delta = exp - datetime.now(UTC)
                    days_left = delta.days
                except (ValueError, TypeError):
                    pass
            return CertStatus(
                domain=domain,
                issued=True,
                issuer=cert.get("issuer"),
                not_before=not_before_raw,
                not_after=not_after_raw,
                days_until_expiry=days_left,
            )

    return CertStatus(
        domain=domain,
        issued=False,
        issuer=None,
        not_before=None,
        not_after=None,
        days_until_expiry=None,
    )
