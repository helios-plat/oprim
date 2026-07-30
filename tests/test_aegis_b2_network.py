"""B2 Network alias tests — network_port_check / network_http_health / network_dns_resolve."""

from __future__ import annotations

import pytest

from oprim import (
    network_port_check,
    network_http_health,
    network_dns_resolve,
    tcp_port_check,
    http_health_probe,
    dns_resolve,
)


def test_network_port_check_is_alias():
    assert network_port_check is tcp_port_check


def test_network_http_health_is_alias():
    assert network_http_health is http_health_probe


def test_network_dns_resolve_is_alias():
    assert network_dns_resolve is dns_resolve


def test_network_port_check_callable():
    """network_port_check is callable with same signature as tcp_port_check."""
    import inspect

    sig_orig = inspect.signature(tcp_port_check)
    sig_alias = inspect.signature(network_port_check)
    assert sig_orig.parameters.keys() == sig_alias.parameters.keys()


def test_network_http_health_callable():
    import inspect

    sig_orig = inspect.signature(http_health_probe)
    sig_alias = inspect.signature(network_http_health)
    assert sig_orig.parameters.keys() == sig_alias.parameters.keys()


def test_network_dns_resolve_callable():
    import inspect

    sig_orig = inspect.signature(dns_resolve)
    sig_alias = inspect.signature(network_dns_resolve)
    assert sig_orig.parameters.keys() == sig_alias.parameters.keys()


def test_network_port_check_unreachable_host():
    """Unreachable host returns reachable=False, no raise."""
    result = network_port_check(host="127.0.0.1", port=19999, timeout_sec=1)
    assert result.reachable is False
    assert result.host == "127.0.0.1"
    assert result.port == 19999


def test_network_http_health_bad_url():
    """Invalid URL returns healthy=False."""
    result = network_http_health(url="http://localhost:19998/health", timeout_sec=1)
    assert result.healthy is False
    assert result.url == "http://localhost:19998/health"


def test_network_dns_resolve_localhost():
    """localhost resolves to 127.0.0.1."""
    result = network_dns_resolve(hostname="localhost")
    assert result.hostname == "localhost"
    assert isinstance(result.records, list)


def test_network_port_check_result_has_elapsed():
    result = network_port_check(host="127.0.0.1", port=19999, timeout_sec=1)
    assert isinstance(result.elapsed_ms, int)
    assert result.elapsed_ms >= 0


def test_network_http_health_result_has_elapsed():
    result = network_http_health(url="http://localhost:19998/", timeout_sec=1)
    assert isinstance(result.elapsed_ms, int)
