"""Tailscale status oprim — 解析 `tailscale status --json`(只读 R0)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from pydantic import BaseModel

from oprim._exceptions import OprimError


class TailscalePeer(BaseModel):
    hostname: str | None = None
    dns_name: str | None = None
    os: str | None = None
    ips: list[str] = []
    online: bool | None = None


class TailscaleStatus(BaseModel):
    installed: bool
    running: bool  # BackendState == "Running"
    backend_state: str | None = None  # Running / Stopped / NeedsLogin / NoState ...
    self_ips: list[str] = []
    self_hostname: str | None = None
    tailnet: str | None = None  # MagicDNSSuffix
    peers: list[TailscalePeer] = []
    peer_count: int = 0
    message: str | None = None  # 降级/未装说明


def _run_tailscale() -> str:
    """跑 `tailscale status --json` 返回 stdout. 供测试 monkeypatch."""
    if shutil.which("tailscale") is None:
        raise OprimError("tailscale not found on host")
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,  # 未登录时非零退出但仍给 JSON
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError("tailscale status timed out", cause=e) from e
    if not proc.stdout.strip():
        raise OprimError(f"tailscale status produced no output: {proc.stderr.strip()}")
    return proc.stdout


def _parse(data: dict[str, Any]) -> TailscaleStatus:
    backend = data.get("BackendState")
    self_node = data.get("Self") or {}
    magic = data.get("MagicDNSSuffix")
    peers: list[TailscalePeer] = []
    for p in (data.get("Peer") or {}).values():
        peers.append(
            TailscalePeer(
                hostname=p.get("HostName"),
                dns_name=(p.get("DNSName") or "").rstrip(".") or None,
                os=p.get("OS"),
                ips=p.get("TailscaleIPs") or [],
                online=p.get("Online"),
            )
        )
    return TailscaleStatus(
        installed=True,
        running=backend == "Running",
        backend_state=backend,
        self_ips=self_node.get("TailscaleIPs") or [],
        self_hostname=self_node.get("HostName"),
        tailnet=magic,
        peers=peers,
        peer_count=len(peers),
    )


def tailscale_status(*, status_json: str | None = None) -> TailscaleStatus:
    """读 Tailscale 网状 VPN 状态,解析 `tailscale status --json`.

    只读(R0). 执行位置由调用方决定:不传 status_json 本地跑 tailscale;传入则
    只解析(调用方可经特权 host-shell / 远端节点取到 JSON 再交本原语).

    Args:
        status_json: 可选. 预取的 `tailscale status --json` 原始输出.

    Returns:
        TailscaleStatus: installed/running/backend_state/self_ips/tailnet/peers.
            tailscale 未安装且走本地执行时返回 installed=False(不抛).

    Raises:
        OprimError: 传入的 status_json 非法 JSON;或本地执行超时/无输出.
    """
    if status_json is not None:
        try:
            data = json.loads(status_json)
        except json.JSONDecodeError as e:
            raise OprimError("status_json is not valid JSON", cause=e) from e
        return _parse(data)

    if shutil.which("tailscale") is None:
        return TailscaleStatus(installed=False, running=False, message="tailscale not installed")
    raw = _run_tailscale()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise OprimError("tailscale produced invalid JSON", cause=e) from e
    return _parse(data)
