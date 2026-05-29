"""Tests for oprim Caddy Admin API operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from oprim import caddy_admin_reload, caddy_certificates_status, caddy_routes_list
from oprim._caddy import CertStatus, ReloadResult, Route
from oprim._exceptions import OprimConnectionError, OprimValidationError


def _mock_resp(status_code: int = 200, json_data=None, headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data
    resp.text = str(json_data)
    resp.headers = headers or {}
    return resp


VALID_CONFIG = {
    "apps": {
        "http": {
            "servers": {
                "srv0": {
                    "listen": [":443"],
                    "routes": []
                }
            }
        }
    }
}


# ---------------------------------------------------------------------------
# caddy_admin_reload
# ---------------------------------------------------------------------------

class TestCaddyAdminReload:
    def test_successful_reload(self):
        resp = _mock_resp(200, headers={"Etag": "etag-abc"})
        with patch("oprim._caddy.httpx.post", return_value=resp):
            result = caddy_admin_reload(
                admin_url="http://localhost:2019",
                new_config=VALID_CONFIG,
            )
        assert isinstance(result, ReloadResult)
        assert result.success is True
        assert result.elapsed_ms >= 0

    def test_reload_returns_config_id(self):
        resp = _mock_resp(200, headers={"Etag": "v2"})
        with patch("oprim._caddy.httpx.post", return_value=resp):
            result = caddy_admin_reload(
                admin_url="http://localhost:2019",
                new_config=VALID_CONFIG,
            )
        assert result.config_id == "v2"

    def test_invalid_config_raises_validation_error(self):
        resp = _mock_resp(400)
        with patch("oprim._caddy.httpx.post", return_value=resp):
            with pytest.raises(OprimValidationError):
                caddy_admin_reload(
                    admin_url="http://localhost:2019",
                    new_config={"invalid": True},
                )

    def test_connection_error(self):
        with patch("oprim._caddy.httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                caddy_admin_reload(
                    admin_url="http://nonexistent:2019",
                    new_config=VALID_CONFIG,
                )

    def test_caddy_server_error(self):
        resp = _mock_resp(500)
        with patch("oprim._caddy.httpx.post", return_value=resp):
            with pytest.raises(OprimConnectionError):
                caddy_admin_reload(
                    admin_url="http://localhost:2019",
                    new_config=VALID_CONFIG,
                )

    def test_timeout_error(self):
        with patch("oprim._caddy.httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(OprimConnectionError):
                caddy_admin_reload(
                    admin_url="http://localhost:2019",
                    new_config=VALID_CONFIG,
                )


# ---------------------------------------------------------------------------
# caddy_routes_list
# ---------------------------------------------------------------------------

class TestCaddyRoutesList:
    def _routes_data(self):
        return [
            {
                "@id": "route1",
                "match": [{"host": ["example.com"]}],
                "handle": [
                    {
                        "handler": "reverse_proxy",
                        "upstreams": [{"dial": "backend:8080"}],
                    }
                ],
            }
        ]

    def test_list_routes(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, self._routes_data())):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert len(result) == 1
        assert isinstance(result[0], Route)
        assert result[0].id == "route1"
        assert result[0].target_upstream == "backend:8080"

    def test_empty_routes(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, [])):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert result == []

    def test_route_not_found_returns_empty(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(404)):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert result == []

    def test_non_reverse_proxy_handler(self):
        data = [{"match": [], "handle": [{"handler": "file_server"}]}]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert result[0].target_upstream is None

    def test_connection_error(self):
        with patch("oprim._caddy.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                caddy_routes_list(admin_url="http://nonexistent:2019")

    def test_server_error(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(500)):
            with pytest.raises(OprimConnectionError):
                caddy_routes_list(admin_url="http://localhost:2019")

    def test_non_list_response_returns_empty(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, {"routes": []})):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert result == []

    def test_reverse_proxy_empty_upstreams(self):
        data = [{"match": [], "handle": [{"handler": "reverse_proxy", "upstreams": []}]}]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_routes_list(admin_url="http://localhost:2019")
        assert result[0].target_upstream is None


# ---------------------------------------------------------------------------
# caddy_certificates_status
# ---------------------------------------------------------------------------

class TestCaddyCertificatesStatus:
    def _cert_data(self, domain="example.com"):
        return [
            {
                "names": [domain],
                "issuer": "Let's Encrypt",
                "not_before": "2026-01-01T00:00:00Z",
                "not_after": "2026-07-01T00:00:00Z",
            }
        ]

    def test_domain_found(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, self._cert_data())):
            result = caddy_certificates_status(
                admin_url="http://localhost:2019",
                domain="example.com",
            )
        assert isinstance(result, CertStatus)
        assert result.issued is True
        assert result.domain == "example.com"
        assert result.issuer == "Let's Encrypt"

    def test_domain_not_found(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, [])):
            result = caddy_certificates_status(
                admin_url="http://localhost:2019",
                domain="missing.com",
            )
        assert result.issued is False
        assert result.not_after is None

    def test_no_certificates_endpoint(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(404)):
            result = caddy_certificates_status(
                admin_url="http://localhost:2019",
                domain="example.com",
            )
        assert result.issued is False

    def test_expiry_days_computed(self):
        from datetime import datetime, timezone, timedelta
        future_date = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        data = [{"names": ["example.com"], "issuer": "LE", "not_after": future_date}]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_certificates_status(
                admin_url="http://localhost:2019",
                domain="example.com",
            )
        assert result.days_until_expiry is not None
        assert result.days_until_expiry > 0

    def test_connection_error(self):
        with patch("oprim._caddy.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                caddy_certificates_status(
                    admin_url="http://nonexistent:2019",
                    domain="example.com",
                )

    def test_server_error(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(500)):
            with pytest.raises(OprimConnectionError):
                caddy_certificates_status(
                    admin_url="http://localhost:2019",
                    domain="example.com",
                )

    def test_non_list_response_returns_not_issued(self):
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, {"not": "a list"})):
            result = caddy_certificates_status(admin_url="http://localhost:2019", domain="example.com")
        assert result.issued is False

    def test_wildcard_domain_match(self):
        data = [{"names": ["*.example.com"], "issuer": "LE", "not_after": None}]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_certificates_status(admin_url="http://localhost:2019", domain="api.example.com")
        assert result.issued is True

    def test_invalid_expiry_date_no_error(self):
        data = [{"names": ["example.com"], "issuer": "LE", "not_after": "not-a-date"}]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_certificates_status(admin_url="http://localhost:2019", domain="example.com")
        assert result.issued is True
        assert result.days_until_expiry is None

    def test_multiple_certs_second_matches(self):
        data = [
            {"names": ["other.com"], "issuer": "LE"},
            {"names": ["example.com"], "issuer": "LE", "not_after": None},
        ]
        with patch("oprim._caddy.httpx.get", return_value=_mock_resp(200, data)):
            result = caddy_certificates_status(admin_url="http://localhost:2019", domain="example.com")
        assert result.issued is True
