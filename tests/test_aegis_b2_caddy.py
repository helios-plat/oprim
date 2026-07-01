"""B2 Caddy tests — caddy_admin_config / caddy_admin_routes alias /
caddy_route_add_atomic / caddy_route_remove_atomic."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from oprim import (
    caddy_admin_config,
    caddy_admin_routes,
    caddy_route_add_atomic,
    caddy_route_remove_atomic,
    caddy_routes_list,
)
from oprim._exceptions import OprimConnectionError, OprimValidationError


def test_caddy_admin_routes_is_alias():
    assert caddy_admin_routes is caddy_routes_list


# ===== caddy_admin_config =====


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = status_code < 400
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_caddy_admin_config_success():
    full_config = {"apps": {"http": {"servers": {}}}}
    with patch("oprim._caddy._admin_request", return_value=_mock_response(200, full_config)):
        result = caddy_admin_config(admin_url="http://localhost:2019")
    assert result == full_config


def test_caddy_admin_config_connection_error():
    with patch(
        "oprim._caddy._admin_request",
        return_value=_mock_response(500, text="internal error"),
    ):
        with pytest.raises(OprimConnectionError):
            caddy_admin_config(admin_url="http://localhost:2019")


def test_caddy_admin_config_passes_timeout():
    with patch("oprim._caddy._admin_request", return_value=_mock_response(200, {})) as mock_req:
        caddy_admin_config(admin_url="http://localhost:2019", timeout_sec=3)
    _, kwargs = mock_req.call_args
    assert kwargs.get("timeout_sec") == 3 or mock_req.call_args[0][-1] == 3


# ===== caddy_route_add_atomic =====

_EXISTING_ROUTES = [
    {"@id": "route-1", "match": [{"host": ["a.example.com"]}], "handle": []},
]

_NEW_ROUTE = {
    "@id": "route-2",
    "match": [{"host": ["b.example.com"]}],
    "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "localhost:8080"}]}],
}


def _setup_route_mocks(get_routes=None, put_status=200):
    def fake_admin_request(method, admin_url, path, json_body=None, timeout_sec=10):
        if method == "GET":
            resp = _mock_response(200, get_routes or _EXISTING_ROUTES)
        else:
            resp = _mock_response(put_status, {"status": "ok"})
        return resp

    return fake_admin_request


def test_caddy_route_add_atomic_appends_route():
    calls = []

    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        calls.append((method, json_body))
        if method == "GET":
            return _mock_response(200, list(_EXISTING_ROUTES))
        return _mock_response(200, {})

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        result = caddy_route_add_atomic(
            admin_url="http://localhost:2019",
            route=_NEW_ROUTE,
        )

    assert result["status"] == "ok"
    assert result["routes_total"] == 2
    put_body = calls[1][1]
    assert any(r.get("@id") == "route-2" for r in put_body)


def test_caddy_route_add_atomic_insert_position():
    calls = []

    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        calls.append((method, json_body))
        if method == "GET":
            return _mock_response(200, list(_EXISTING_ROUTES))
        return _mock_response(200, {})

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        result = caddy_route_add_atomic(
            admin_url="http://localhost:2019",
            route=_NEW_ROUTE,
            position=0,
        )

    put_body = calls[1][1]
    assert put_body[0].get("@id") == "route-2"


def test_caddy_route_add_atomic_validation_error():
    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        if method == "GET":
            return _mock_response(200, [])
        return _mock_response(400, text="invalid route")

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        with pytest.raises(OprimValidationError):
            caddy_route_add_atomic(
                admin_url="http://localhost:2019",
                route=_NEW_ROUTE,
            )


# ===== caddy_route_remove_atomic =====


def test_caddy_route_remove_atomic_removes_by_id():
    routes = [
        {"@id": "route-keep", "match": []},
        {"@id": "route-remove", "match": []},
    ]
    calls = []

    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        calls.append((method, json_body))
        if method == "GET":
            return _mock_response(200, list(routes))
        return _mock_response(200, {})

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        result = caddy_route_remove_atomic(
            admin_url="http://localhost:2019",
            route_id="route-remove",
        )

    assert result["removed"] is True
    assert result["routes_total"] == 1
    put_body = calls[1][1]
    assert all(r.get("@id") != "route-remove" for r in put_body)


def test_caddy_route_remove_atomic_missing_id():
    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        if method == "GET":
            return _mock_response(200, [{"@id": "route-a"}])
        return _mock_response(200, {})

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        result = caddy_route_remove_atomic(
            admin_url="http://localhost:2019",
            route_id="nonexistent",
        )

    assert result["removed"] is False
    assert result["routes_total"] == 1


def test_caddy_route_remove_atomic_validation_error():
    def fake_req(method, admin_url, path, json_body=None, timeout_sec=10):
        if method == "GET":
            return _mock_response(200, [{"@id": "route-x"}])
        return _mock_response(422, text="bad config")

    with patch("oprim._caddy._admin_request", side_effect=fake_req):
        with pytest.raises(OprimValidationError):
            caddy_route_remove_atomic(
                admin_url="http://localhost:2019",
                route_id="route-x",
            )
