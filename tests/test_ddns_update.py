"""Tests for oprim.ddns_update."""

from __future__ import annotations

import httpx
import pytest
import respx

from oprim import ddns_update
from oprim._ddns_update import DdnsResult, _parse_dyndns2
from oprim._exceptions import OprimError, OprimValidationError


class TestValidation:
    def test_missing_hostname(self):
        with pytest.raises(OprimValidationError, match="hostname"):
            ddns_update(provider="duckdns", hostname="", token="t")

    def test_duckdns_requires_token(self):
        with pytest.raises(OprimValidationError, match="token"):
            ddns_update(provider="duckdns", hostname="foo")

    def test_dyndns2_requires_creds(self):
        with pytest.raises(OprimValidationError, match="username and password"):
            ddns_update(provider="dyndns2", hostname="foo.example.com", username="u")


class TestDuckDns:
    @respx.mock
    def test_ok(self):
        route = respx.get("https://www.duckdns.org/update").mock(
            return_value=httpx.Response(200, text="OK")
        )
        r = ddns_update(
            provider="duckdns", hostname="myhost.duckdns.org", token="tok", ip="1.2.3.4"
        )
        assert isinstance(r, DdnsResult)
        assert r.success is True and r.changed is True
        assert r.hostname == "myhost.duckdns.org"
        assert r.ip == "1.2.3.4"
        # duckdns 用不含后缀的子域名
        assert route.calls.last.request.url.params["domains"] == "myhost"

    @respx.mock
    def test_ko(self):
        respx.get("https://www.duckdns.org/update").mock(
            return_value=httpx.Response(200, text="KO")
        )
        r = ddns_update(provider="duckdns", hostname="bad", token="tok")
        assert r.success is False and r.status == "error"


class TestDynDns2:
    @respx.mock
    def test_good(self):
        respx.get("https://dynupdate.no-ip.com/nic/update").mock(
            return_value=httpx.Response(200, text="good 203.0.113.5")
        )
        r = ddns_update(provider="dyndns2", hostname="h.example.com", username="u", password="p")
        assert r.success is True and r.changed is True
        assert r.ip == "203.0.113.5" and r.status == "ok"

    @respx.mock
    def test_nochg(self):
        respx.get("https://dynupdate.no-ip.com/nic/update").mock(
            return_value=httpx.Response(200, text="nochg 203.0.113.5")
        )
        r = ddns_update(provider="dyndns2", hostname="h.example.com", username="u", password="p")
        assert r.success is True and r.changed is False and r.status == "nochg"

    @respx.mock
    def test_badauth(self):
        respx.get("https://dynupdate.no-ip.com/nic/update").mock(
            return_value=httpx.Response(200, text="badauth")
        )
        r = ddns_update(provider="dyndns2", hostname="h.example.com", username="u", password="bad")
        assert r.success is False and r.status == "badauth"

    @respx.mock
    def test_custom_base_url_and_auth(self):
        route = respx.get("https://dynupdate.example.net/nic/update").mock(
            return_value=httpx.Response(200, text="good 1.1.1.1")
        )
        r = ddns_update(
            provider="dyndns2",
            hostname="h.example.com",
            username="u",
            password="p",
            ip="1.1.1.1",
            base_url="https://dynupdate.example.net/nic/update",
        )
        assert r.success is True
        req = route.calls.last.request
        assert req.url.params["myip"] == "1.1.1.1"
        assert "Authorization" in req.headers  # basic auth sent


class TestParseUnit:
    def test_nohost(self):
        assert _parse_dyndns2("nohost").status == "nohost"

    def test_unknown_code(self):
        r = _parse_dyndns2("weirdstuff")
        assert r.success is False and r.status == "error"


class TestTransportError:
    @respx.mock
    def test_connect_error_wrapped(self):
        respx.get("https://www.duckdns.org/update").mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(OprimError, match="DDNS request failed"):
            ddns_update(provider="duckdns", hostname="h", token="t")
