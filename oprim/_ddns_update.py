"""Dynamic DNS update oprim — 多 provider DDNS 记录更新(R1 幂等)."""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from oprim._exceptions import OprimError, OprimValidationError

# dyndns2 是被 No-IP / DynDNS / 及大量路由器固件复用的事实标准协议。
_DEFAULT_DYNDNS2_URL = "https://dynupdate.no-ip.com/nic/update"
_DUCKDNS_URL = "https://www.duckdns.org/update"

Provider = Literal["duckdns", "dyndns2"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DdnsResult(BaseModel):
    provider: str
    hostname: str
    success: bool
    changed: bool  # True=记录被改, False=本就是该值(nochg)
    ip: str | None = None  # 生效的 IP(能从响应解析时)
    status: str  # 归一化状态: ok / nochg / badauth / nohost / abuse / error
    raw: str  # provider 原始响应(截断)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_get(
    url: str, *, params: dict[str, str], auth: tuple[str, str] | None, timeout: float
) -> httpx.Response:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            return client.get(
                url, params=params, auth=auth, headers={"User-Agent": "oprim-ddns/1.0"}
            )
    except httpx.HTTPError as e:
        raise OprimError(f"DDNS request failed: {e}", cause=e) from e


def _parse_duckdns(body: str, ip: str | None) -> DdnsResult:
    b = body.strip()
    ok = b.upper().startswith("OK")
    return DdnsResult(
        provider="duckdns",
        hostname="",  # 调用方填(duckdns 用 domains 概念)
        success=ok,
        changed=ok,  # duckdns 不区分 changed/nochg,OK 即视为已置位
        ip=ip,
        status="ok" if ok else "error",
        raw=b[:200],
    )


def _parse_dyndns2(body: str) -> DdnsResult:
    b = body.strip()
    first = b.split()[0].lower() if b else ""
    # 成功码: good <ip> / nochg <ip>; 其余为失败码
    ip = None
    parts = b.split()
    if first in ("good", "nochg") and len(parts) >= 2:
        ip = parts[1]
    status_map = {
        "good": "ok",
        "nochg": "nochg",
        "badauth": "badauth",
        "nohost": "nohost",
        "abuse": "abuse",
        "notfqdn": "notfqdn",
        "!donator": "error",
        "911": "error",
    }
    status = status_map.get(first, "error")
    return DdnsResult(
        provider="dyndns2",
        hostname="",
        success=first in ("good", "nochg"),
        changed=first == "good",
        ip=ip,
        status=status,
        raw=b[:200],
    )


# ---------------------------------------------------------------------------
# ddns_update
# ---------------------------------------------------------------------------


def ddns_update(
    *,
    provider: Provider,
    hostname: str,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
    ip: str | None = None,
    base_url: str | None = None,
    timeout: float = 15.0,
) -> DdnsResult:
    """更新一个动态 DNS 记录(A 记录). R1 幂等(重复更新到同值返回 nochg).

    支持两种协议:
      - **duckdns**: 需 token + hostname(子域名, 不含 .duckdns.org). ip 可省(provider 自测).
      - **dyndns2**: No-IP / DynDNS 等复用的协议, 需 username + password + hostname;
        base_url 默认 No-IP, 换 provider 传其 /nic/update 端点.

    Args:
        provider: "duckdns" | "dyndns2".
        hostname: 要更新的主机名/子域.
        token: duckdns 的 token.
        username/password: dyndns2 的 HTTP Basic 凭据.
        ip: 目标 IPv4;省略则由 provider 按请求来源自测.
        base_url: dyndns2 的更新端点(覆盖默认 No-IP).
        timeout: HTTP 超时秒.

    Returns:
        DdnsResult: success/changed/status/ip/raw. hostname 回填入参.

    Raises:
        OprimValidationError: 缺 provider 必需的凭据.
        OprimError: HTTP 传输失败.
    """
    if not hostname:
        raise OprimValidationError("hostname is required")

    if provider == "duckdns":
        if not token:
            raise OprimValidationError("duckdns requires token")
        # duckdns 用不含后缀的子域名; 允许传完整域, 取第一段.
        domains = hostname.split(".")[0]
        params = {"domains": domains, "token": token}
        if ip:
            params["ip"] = ip
        resp = _run_get(_DUCKDNS_URL, params=params, auth=None, timeout=timeout)
        result = _parse_duckdns(resp.text, ip)
    elif provider == "dyndns2":
        if not (username and password):
            raise OprimValidationError("dyndns2 requires username and password")
        params = {"hostname": hostname}
        if ip:
            params["myip"] = ip
        url = base_url or _DEFAULT_DYNDNS2_URL
        resp = _run_get(url, params=params, auth=(username, password), timeout=timeout)
        result = _parse_dyndns2(resp.text)
    else:  # pragma: no cover — Literal 已约束
        raise OprimValidationError(f"unknown provider: {provider}")

    result.hostname = hostname
    return result
