"""Tests for oprim.url_safety_check."""

import socket
from unittest.mock import patch

import pytest

from oprim.url_safety_check import URLSafetyError, URLSafetyResult, url_safety_check


class TestUrlSafetyCheckLoopback:
    def test_loopback_ipv4_blocked(self) -> None:
        result = url_safety_check(url="http://127.0.0.1")
        assert result.is_safe is False
        assert result.failed_check == "is_loopback"

    def test_loopback_ipv6_blocked(self) -> None:
        result = url_safety_check(url="http://[::1]")
        assert result.is_safe is False
        assert result.failed_check == "is_loopback"


class TestUrlSafetyCheckPrivate:
    def test_private_rfc1918_blocked(self) -> None:
        result = url_safety_check(url="http://10.0.0.1")
        assert result.is_safe is False
        assert result.failed_check == "is_private"

    def test_link_local_aws_metadata_blocked(self) -> None:
        result = url_safety_check(url="http://169.254.169.254")
        assert result.is_safe is False
        assert result.failed_check == "is_link_local"

    def test_cgn_reserved_blocked(self) -> None:
        # 100.64.0.1 is in the CGN range (RFC 6598).
        # is_private=False for this address; is_reserved=True catches it.
        result = url_safety_check(url="http://100.64.0.1")
        assert result.is_safe is False
        assert result.failed_check == "is_reserved"

    def test_multicast_blocked(self) -> None:
        result = url_safety_check(url="http://224.0.0.1")
        assert result.is_safe is False
        assert result.failed_check == "is_multicast"


class TestUrlSafetyCheckScheme:
    def test_ftp_scheme_blocked(self) -> None:
        result = url_safety_check(url="ftp://example.com")
        assert result.is_safe is False
        assert result.reason == "scheme_not_allowed"

    def test_custom_allowed_schemes_rejects_https(self) -> None:
        result = url_safety_check(url="https://example.com", allowed_schemes=["http"])
        assert result.is_safe is False
        assert result.reason == "scheme_not_allowed"

    def test_no_hostname_blocked(self) -> None:
        result = url_safety_check(url="http://")
        assert result.is_safe is False
        assert result.reason == "no_hostname"


class TestUrlSafetyCheckDns:
    def test_nonexistent_domain_fails(self) -> None:
        result = url_safety_check(url="http://nonexistent-domain-xyz-12345.invalid")
        assert result.is_safe is False
        assert result.reason == "dns_resolution_failed"


class TestUrlSafetyCheckBlockFlags:
    def test_block_private_false_allows_private_ip(self) -> None:
        # 10.0.0.1: is_private=True, is_loopback=False, is_link_local=False,
        # is_reserved=False, is_multicast=False — so disabling block_private makes it safe.
        result = url_safety_check(url="http://10.0.0.1", block_private=False)
        assert result.is_safe is True

    def test_custom_allowed_schemes_permits_http(self) -> None:
        result = url_safety_check(url="http://127.0.0.1", allowed_schemes=["http", "https"])
        # Still blocked by loopback even with scheme allowed
        assert result.is_safe is False
        assert result.failed_check == "is_loopback"


class TestUrlSafetyCheckResult:
    def test_result_is_url_safety_result_instance(self) -> None:
        result = url_safety_check(url="http://127.0.0.1")
        assert isinstance(result, URLSafetyResult)

    def test_resolved_ips_populated_on_blocked_ip(self) -> None:
        result = url_safety_check(url="http://127.0.0.1")
        assert "127.0.0.1" in result.resolved_ips

    def test_resolved_ips_empty_on_dns_failure(self) -> None:
        result = url_safety_check(url="http://nonexistent-xyz-12345.invalid")
        assert result.resolved_ips == []

    def test_safe_result_has_none_reason_and_none_failed_check(self) -> None:
        # Use loopback with block_loopback=False + block_private=False to get is_safe=True
        result = url_safety_check(
            url="http://127.0.0.1",
            block_loopback=False,
            block_private=False,
            block_link_local=False,
            block_reserved=False,
            block_multicast=False,
        )
        assert result.is_safe is True
        assert result.reason is None
        assert result.failed_check is None


class TestUrlSafetyCheckErrorPaths:
    """Cover technical-error branches (URLSafetyError) via mocking."""

    def test_urlparse_exception_raises_url_safety_error(self) -> None:
        with (
            patch("oprim.url_safety_check.urlparse", side_effect=ValueError("bad")),
            pytest.raises(URLSafetyError, match="url parse failed"),
        ):
            url_safety_check(url="http://example.com")

    def test_getaddrinfo_unexpected_error_raises_url_safety_error(self) -> None:
        with (
            patch(
                "oprim.url_safety_check.socket.getaddrinfo",
                side_effect=OSError("unexpected"),
            ),
            pytest.raises(URLSafetyError, match="getaddrinfo unexpected error"),
        ):
            url_safety_check(url="http://example.com")

    def test_all_sockaddr_malformed_returns_dns_failed(self) -> None:
        # addrinfo returns entries but all sockaddr are empty tuples (IndexError)
        bad_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ())]
        with patch("oprim.url_safety_check.socket.getaddrinfo", return_value=bad_addrinfo):
            result = url_safety_check(url="http://example.com")
        assert result.is_safe is False
        assert result.reason == "dns_resolution_failed"

    def test_invalid_ip_string_in_resolved_ips_is_skipped(self) -> None:
        # getaddrinfo returns a valid-looking entry but with an unparseable IP
        bad_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))]
        with patch("oprim.url_safety_check.socket.getaddrinfo", return_value=bad_addrinfo):
            result = url_safety_check(url="http://example.com")
        # "not-an-ip" fails ip_address() → skipped → all IPs pass → is_safe=True
        assert result.is_safe is True
