"""Tests for oprim.heartbeat_emit — external dead-man heartbeat (aegis DESIGN §6 L1)."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from oprim import heartbeat_emit
from oprim._heartbeat_emit import HeartbeatResult

URL = "https://hc-ping.example/uuid"


def _resp(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, text="ok")


def _patch_request(return_value=None, side_effect=None):
    p = patch("oprim._heartbeat_emit.httpx.Client")
    m = p.start()
    req = m.return_value.__enter__.return_value.request
    if side_effect is not None:
        req.side_effect = side_effect
    else:
        req.return_value = return_value
    return p


class TestHeartbeatEmit:
    def test_200_delivered(self):
        p = _patch_request(return_value=_resp(200))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert isinstance(r, HeartbeatResult)
        assert r.delivered is True
        assert r.status_code == 200
        assert r.error is None
        assert r.elapsed_ms >= 0

    def test_404_not_delivered(self):
        p = _patch_request(return_value=_resp(404))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert r.delivered is False
        assert r.status_code == 404
        assert r.error == "http_4xx"

    def test_500_http_5xx(self):
        p = _patch_request(return_value=_resp(503))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert r.delivered is False
        assert r.error == "http_5xx"

    def test_timeout_never_raises(self):
        p = _patch_request(side_effect=httpx.TimeoutException("t"))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert r.delivered is False
        assert r.status_code is None
        assert r.error == "timeout"

    def test_connect_error(self):
        p = _patch_request(side_effect=httpx.ConnectError("refused"))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert r.delivered is False
        assert r.error.startswith("connect_failed")

    def test_unexpected_never_raises(self):
        p = _patch_request(side_effect=RuntimeError("boom"))
        try:
            r = heartbeat_emit(url=URL)
        finally:
            p.stop()
        assert r.delivered is False
        assert r.error.startswith("unexpected")
