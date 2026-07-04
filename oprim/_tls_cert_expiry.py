"""oprim._tls_cert_expiry — TLS 叶证书到期探针 (aegis DESIGN §3.1 拨测顺带)."""

from __future__ import annotations

import socket
import ssl
from datetime import UTC, datetime

from pydantic import BaseModel

from oprim._exceptions import OprimConnectionError

# openssl notAfter 格式: 'Jun  1 12:00:00 2035 GMT'
_CERT_TIME_FMT = "%b %d %H:%M:%S %Y %Z"


class CertExpiry(BaseModel):
    host: str
    port: int
    not_after: str  # ISO8601
    days_remaining: int
    expired: bool
    warn: bool  # days_remaining <= warn_days


def _parse_cert_datetime(raw: str) -> datetime:
    """解析证书 notAfter 字符串为 UTC datetime。"""
    return datetime.strptime(raw, _CERT_TIME_FMT).replace(tzinfo=UTC)


def _fetch_peer_cert(*, host: str, port: int, timeout_sec: float) -> dict:
    """握手取叶证书解析字典。失败抛 OprimConnectionError。"""
    ctx = ssl.create_default_context()
    try:
        with (
            socket.create_connection((host, port), timeout=timeout_sec) as sock,
            ctx.wrap_socket(sock, server_hostname=host) as ssock,
        ):
            return ssock.getpeercert() or {}
    except (OSError, ssl.SSLError) as exc:
        raise OprimConnectionError(f"TLS probe failed for {host}:{port}: {exc}") from exc


def tls_cert_expiry_probe(
    *,
    host: str,
    port: int = 443,
    warn_days: int = 14,
    timeout_sec: float = 10.0,
) -> CertExpiry:
    """握手取叶证书 not_after,计算剩余天数与是否临期/过期。

    Args:
        host: 目标主机(SNI)。
        port: TLS 端口。
        warn_days: 临期阈值(剩余天数 <= 此值 → warn=True)。
        timeout_sec: 连接超时。

    Returns:
        CertExpiry(not_after / days_remaining / expired / warn)。

    Raises:
        OprimConnectionError: 连接/握手失败,或证书无 notAfter。
    """
    cert = _fetch_peer_cert(host=host, port=port, timeout_sec=timeout_sec)
    raw = cert.get("notAfter")
    if not raw:
        raise OprimConnectionError(f"peer cert for {host}:{port} has no notAfter")

    not_after = _parse_cert_datetime(raw)
    now = datetime.now(UTC)
    days_remaining = (not_after - now).days
    return CertExpiry(
        host=host,
        port=port,
        not_after=not_after.isoformat(),
        days_remaining=days_remaining,
        expired=now >= not_after,
        warn=days_remaining <= warn_days,
    )
