"""B2 AppStore catalog fetch tests — appstore_catalog_fetch."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from oprim import appstore_catalog_fetch
from oprim.appstore_catalog_fetch import AppCatalogEntry
from oprim._exceptions import OprimNotFoundError, OprimConnectionError, OprimTimeoutError


def _mock_response(status_code=200, data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = status_code < 400
    resp.json.return_value = data or {}
    resp.text = text
    return resp


_CATALOG_DATA = {
    "app_id": "myapp",
    "name": "My App",
    "version": "1.2.3",
    "image": "registry.example.com/myapp:1.2.3",
    "compose_file": "/apps/myapp/docker-compose.yml",
    "routes": [{"host": "myapp.example.com", "upstream": "localhost:8080"}],
    "env_vars": {"APP_ENV": "production"},
    "service_url": "http://localhost:8080/health",
    "description": "A great app",
    "tags": ["web", "prod"],
}


def test_appstore_catalog_fetch_success():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)):
        result = appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1",
            app_id="myapp",
        )
    assert isinstance(result, AppCatalogEntry)
    assert result.app_id == "myapp"
    assert result.version == "1.2.3"
    assert result.image == "registry.example.com/myapp:1.2.3"


def test_appstore_catalog_fetch_routes():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)):
        result = appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1",
            app_id="myapp",
        )
    assert len(result.routes) == 1
    assert result.routes[0]["host"] == "myapp.example.com"


def test_appstore_catalog_fetch_env_vars():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)):
        result = appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1",
            app_id="myapp",
        )
    assert result.env_vars["APP_ENV"] == "production"


def test_appstore_catalog_fetch_404_raises_not_found():
    with patch("httpx.get", return_value=_mock_response(404, text="not found")):
        with pytest.raises(OprimNotFoundError, match="myapp"):
            appstore_catalog_fetch(
                catalog_url="http://appstore.internal/api/v1",
                app_id="myapp",
            )


def test_appstore_catalog_fetch_500_raises_connection_error():
    with patch("httpx.get", return_value=_mock_response(500, text="server error")):
        with pytest.raises(OprimConnectionError):
            appstore_catalog_fetch(
                catalog_url="http://appstore.internal/api/v1",
                app_id="myapp",
            )


def test_appstore_catalog_fetch_timeout():
    import httpx

    with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(OprimTimeoutError):
            appstore_catalog_fetch(
                catalog_url="http://appstore.internal/api/v1",
                app_id="myapp",
            )


def test_appstore_catalog_fetch_connect_error():
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("no route to host")):
        with pytest.raises(OprimConnectionError):
            appstore_catalog_fetch(
                catalog_url="http://appstore.internal/api/v1",
                app_id="myapp",
            )


def test_appstore_catalog_fetch_sends_auth_token():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)) as mock_get:
        appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1",
            app_id="myapp",
            auth_token="secret-token",
        )
    headers = mock_get.call_args.kwargs.get("headers", {})
    assert headers.get("Authorization") == "Bearer secret-token"


def test_appstore_catalog_fetch_constructs_correct_url():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)) as mock_get:
        appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1",
            app_id="myapp",
        )
    called_url = mock_get.call_args[0][0]
    assert called_url == "http://appstore.internal/api/v1/apps/myapp"


def test_appstore_catalog_fetch_trailing_slash_normalized():
    with patch("httpx.get", return_value=_mock_response(200, _CATALOG_DATA)) as mock_get:
        appstore_catalog_fetch(
            catalog_url="http://appstore.internal/api/v1/",
            app_id="myapp",
        )
    called_url = mock_get.call_args[0][0]
    assert not called_url.endswith("//apps/myapp")


def test_appstore_catalog_fetch_empty_optional_fields():
    minimal = {"app_id": "x", "name": "x", "version": "1.0", "image": "img"}
    with patch("httpx.get", return_value=_mock_response(200, minimal)):
        result = appstore_catalog_fetch(
            catalog_url="http://localhost/api",
            app_id="x",
        )
    assert result.routes == []
    assert result.env_vars == {}
    assert result.tags == []
