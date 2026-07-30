"""Tests for oprim.tailscale_status."""

from __future__ import annotations

import json

import pytest

from oprim import tailscale_status
from oprim._exceptions import OprimError
from oprim._tailscale_status import TailscaleStatus, _parse

_FAKE = {
    "BackendState": "Running",
    "MagicDNSSuffix": "tail1234.ts.net",
    "Self": {"HostName": "aegis-box", "TailscaleIPs": ["100.64.0.1", "fd7a::1"]},
    "Peer": {
        "abc": {
            "HostName": "laptop",
            "DNSName": "laptop.tail1234.ts.net.",
            "OS": "macOS",
            "TailscaleIPs": ["100.64.0.2"],
            "Online": True,
        },
        "def": {
            "HostName": "phone",
            "DNSName": "phone.tail1234.ts.net.",
            "OS": "iOS",
            "TailscaleIPs": ["100.64.0.3"],
            "Online": False,
        },
    },
}


class TestParseInjected:
    def test_running_with_peers(self) -> None:
        r = tailscale_status(status_json=json.dumps(_FAKE))
        assert isinstance(r, TailscaleStatus)
        assert r.installed is True and r.running is True
        assert r.backend_state == "Running"
        assert r.self_ips == ["100.64.0.1", "fd7a::1"]
        assert r.self_hostname == "aegis-box"
        assert r.tailnet == "tail1234.ts.net"
        assert r.peer_count == 2
        laptop = next(p for p in r.peers if p.hostname == "laptop")
        assert laptop.online is True
        assert laptop.dns_name == "laptop.tail1234.ts.net"  # trailing dot stripped

    def test_needs_login_not_running(self) -> None:
        r = tailscale_status(status_json=json.dumps({"BackendState": "NeedsLogin", "Self": {}}))
        assert r.running is False and r.backend_state == "NeedsLogin"
        assert r.peer_count == 0

    def test_bad_json_raises(self) -> None:
        with pytest.raises(OprimError, match="not valid JSON"):
            tailscale_status(status_json="{nope")

    def test_empty_peer_map(self) -> None:
        r = _parse({"BackendState": "Running", "Self": {}, "Peer": None})
        assert r.peer_count == 0


class TestLocalDegrade:
    def test_not_installed_returns_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: None)
        r = tailscale_status()
        assert r.installed is False and r.running is False
        assert "not installed" in (r.message or "")
