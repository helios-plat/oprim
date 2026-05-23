"""Tests for oprim network probe operations."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from oprim import dns_resolve, http_health_probe, http_request_once, tcp_port_check
from oprim._exceptions import OprimConnectionError, OprimTimeoutError, OprimValidationError
from oprim._network import DNSResolveResult, HealthProbeResult, HttpResponse, PortCheckResult


# ---------------------------------------------------------------------------
# tcp_port_check
# ---------------------------------------------------------------------------

class TestTcpPortCheck:
    def test_reachable(self):
        mock_sock = MagicMock()
        with patch("oprim._network.socket.create_connection", return_value=mock_sock):
            result = tcp_port_check(host="localhost", port=80)
        assert isinstance(result, PortCheckResult)
        assert result.reachable is True
        assert result.error is None

    def test_connection_refused(self):
        with patch("oprim._network.socket.create_connection", side_effect=ConnectionRefusedError()):
            result = tcp_port_check(host="localhost", port=9999)
        assert result.reachable is False
        assert "refused" in (result.error or "")

    def test_timeout(self):
        with patch("oprim._network.socket.create_connection", side_effect=socket.timeout()):
            result = tcp_port_check(host="10.255.255.1", port=80, timeout_sec=1)
        assert result.reachable is False
        assert "timeout" in (result.error or "")

    def test_dns_failure(self):
        with patch("oprim._network.socket.create_connection", side_effect=socket.gaierror(-2, "Name not found")):
            result = tcp_port_check(host="nonexistent.example.invalid", port=80)
        assert result.reachable is False
        assert "dns" in (result.error or "").lower()

    def test_oserror_returns_not_reachable(self):
        with patch("oprim._network.socket.create_connection", side_effect=OSError("network unreachable")):
            result = tcp_port_check(host="localhost", port=80)
        assert result.reachable is False
        assert result.error is not None

    def test_invalid_port_raises(self):
        with pytest.raises(OprimValidationError):
            tcp_port_check(host="localhost", port=0)

    def test_invalid_port_too_large(self):
        with pytest.raises(OprimValidationError):
            tcp_port_check(host="localhost", port=65536)


# ---------------------------------------------------------------------------
# http_health_probe
# ---------------------------------------------------------------------------

class TestHttpHealthProbe:
    def _make_resp(self, status_code=200, text="OK"):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.text = text
        return resp

    def test_healthy_200(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(200, "OK")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://localhost:8080/health")
        assert isinstance(result, HealthProbeResult)
        assert result.healthy is True
        assert result.status_code == 200

    def test_unhealthy_503(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(503, "Service Unavailable")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://localhost:8080/health")
        assert result.healthy is False
        assert result.status_code == 503

    def test_connection_failure_no_raise(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://nonexistent:9999/health")
        assert result.healthy is False
        assert result.status_code is None
        assert result.error is not None

    def test_timeout_no_raise(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.TimeoutException("timeout")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://localhost:8080/health")
        assert result.healthy is False
        assert result.error is not None

    def test_custom_expected_status(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(204, "")
            mock_client_cls.return_value = ctx
            result = http_health_probe(
                url="http://localhost:8080/health",
                expected_status=204,
            )
        assert result.healthy is True

    def test_head_method_no_body_preview(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(200, "")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://localhost:8080/health", method="HEAD")
        assert result.healthy is True
        assert result.response_body_preview is None

    def test_http_error_no_raise(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.HTTPError("generic http error")
            mock_client_cls.return_value = ctx
            result = http_health_probe(url="http://localhost:8080/health")
        assert result.healthy is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# dns_resolve
# ---------------------------------------------------------------------------

class TestDnsResolve:
    def test_successful_a_record(self):
        mock_answers = MagicMock()
        mock_rr = MagicMock()
        mock_rr.to_text.return_value = "93.184.216.34"
        mock_answers.__iter__ = MagicMock(return_value=iter([mock_rr]))
        mock_answers.rrset = MagicMock()
        mock_answers.rrset.ttl = 300

        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.return_value = mock_answers
            mock_dns.resolver.NXDOMAIN = type("NXDOMAIN", (Exception,), {})
            mock_dns.resolver.NoAnswer = type("NoAnswer", (Exception,), {})
            mock_dns.resolver.Timeout = type("Timeout", (Exception,), {})
            mock_dns.exception.DNSException = type("DNSException", (Exception,), {})
            result = dns_resolve(hostname="example.com", record_type="A")

        assert isinstance(result, DNSResolveResult)
        assert "93.184.216.34" in result.records
        assert result.error is None

    def _dns_mock_setup(self):
        nxdomain_cls = type("NXDOMAIN", (Exception,), {})
        noanswer_cls = type("NoAnswer", (Exception,), {})
        timeout_cls = type("Timeout", (Exception,), {})
        dnsexc_cls = type("DNSException", (Exception,), {})
        return nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls

    def test_nxdomain(self):
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()
        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.side_effect = nxdomain_cls()
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="totally.invalid.nonexistent.domain")
        assert result.records == []
        assert result.error is not None
        assert "NXDOMAIN" in (result.error or "")

    def test_timeout(self):
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()
        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.side_effect = timeout_cls()
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="slow.example.com")
        assert result.records == []
        assert result.error is not None
        assert "timed out" in (result.error or "")

    def test_no_answer(self):
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()
        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.side_effect = noanswer_cls()
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="example.com", record_type="MX")
        assert result.records == []

    def test_dns_exception_generic(self):
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()
        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.side_effect = dnsexc_cls("generic dns error")
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="example.com")
        assert result.records == []
        assert result.error is not None

    def test_dns_with_nameserver(self):
        mock_answers = MagicMock()
        mock_rr = MagicMock()
        mock_rr.to_text.return_value = "1.2.3.4"
        mock_answers.__iter__ = MagicMock(return_value=iter([mock_rr]))
        mock_answers.rrset = MagicMock()
        mock_answers.rrset.ttl = 60
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()

        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.return_value = mock_answers
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="example.com", nameserver="8.8.8.8")

        assert result.error is None
        assert mock_dns.resolver.Resolver.return_value.nameservers == ["8.8.8.8"]

    def test_dns_not_installed_returns_error(self):
        with patch("oprim._network.dns", None):
            result = dns_resolve(hostname="example.com")
        assert result.error is not None
        assert "dnspython" in result.error.lower() or "not installed" in result.error.lower()

    def test_dns_generic_exception_caught(self):
        nxdomain_cls, noanswer_cls, timeout_cls, dnsexc_cls = self._dns_mock_setup()
        with patch("oprim._network.dns") as mock_dns:
            mock_dns.resolver.Resolver.return_value.resolve.side_effect = RuntimeError("unexpected")
            mock_dns.resolver.NXDOMAIN = nxdomain_cls
            mock_dns.resolver.NoAnswer = noanswer_cls
            mock_dns.resolver.Timeout = timeout_cls
            mock_dns.exception.DNSException = dnsexc_cls
            result = dns_resolve(hostname="example.com")
        assert result.records == []
        assert result.error is not None


# ---------------------------------------------------------------------------
# http_request_once
# ---------------------------------------------------------------------------

class TestHttpRequestOnce:
    def _make_resp(self, status_code=200, content=b"body", headers=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.content = content
        resp.headers = headers or {"content-type": "text/plain"}
        return resp

    def test_get_request(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(200, b"OK")
            mock_client_cls.return_value = ctx
            result = http_request_once(method="GET", url="http://example.com/api")
        assert isinstance(result, HttpResponse)
        assert result.status_code == 200
        assert result.body == b"OK"

    def test_post_with_body(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(201, b"created")
            mock_client_cls.return_value = ctx
            result = http_request_once(
                method="POST",
                url="http://example.com/api",
                body=b'{"key": "val"}',
                headers={"content-type": "application/json"},
            )
        assert result.status_code == 201

    def test_timeout_raises(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.TimeoutException("timeout")
            mock_client_cls.return_value = ctx
            with pytest.raises(OprimTimeoutError):
                http_request_once(method="GET", url="http://slow.example.com/api")

    def test_connection_error_raises(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value = ctx
            with pytest.raises(OprimConnectionError):
                http_request_once(method="GET", url="http://nonexistent.example.com/api")

    def test_elapsed_ms_positive(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.return_value = self._make_resp(200)
            mock_client_cls.return_value = ctx
            result = http_request_once(method="GET", url="http://example.com")
        assert result.elapsed_ms >= 0

    def test_http_error_raises_connection_error(self):
        with patch("oprim._network.httpx.Client") as mock_client_cls:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.request.side_effect = httpx.HTTPError("generic http error")
            mock_client_cls.return_value = ctx
            with pytest.raises(OprimConnectionError):
                http_request_once(method="GET", url="http://example.com")
