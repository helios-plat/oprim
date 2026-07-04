"""Tests for oprim.tls_cert_expiry_probe (aegis DESIGN §3.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from oprim import tls_cert_expiry_probe
from oprim._exceptions import OprimConnectionError
from oprim._tls_cert_expiry import CertExpiry, _parse_cert_datetime

_FMT = "%b %d %H:%M:%S %Y GMT"


def _cert_not_after(delta_days: float) -> dict:
    dt = datetime.now(UTC) + timedelta(days=delta_days)
    return {"notAfter": dt.strftime(_FMT)}


def _patch_cert(cert: dict):
    p = patch("oprim._tls_cert_expiry._fetch_peer_cert", return_value=cert)
    p.start()
    return p


class TestTlsCertExpiryProbe:
    def test_healthy_far_future(self):
        p = _patch_cert(_cert_not_after(400))
        try:
            r = tls_cert_expiry_probe(host="example.com")
        finally:
            p.stop()
        assert isinstance(r, CertExpiry)
        assert r.expired is False
        assert r.warn is False
        assert r.days_remaining >= 390

    def test_warn_near_expiry(self):
        p = _patch_cert(_cert_not_after(5))
        try:
            r = tls_cert_expiry_probe(host="example.com", warn_days=14)
        finally:
            p.stop()
        assert r.expired is False
        assert r.warn is True

    def test_expired(self):
        p = _patch_cert(_cert_not_after(-2))
        try:
            r = tls_cert_expiry_probe(host="example.com")
        finally:
            p.stop()
        assert r.expired is True
        assert r.warn is True
        assert r.days_remaining < 0

    def test_missing_not_after_raises(self):
        p = _patch_cert({})
        try:
            with pytest.raises(OprimConnectionError):
                tls_cert_expiry_probe(host="example.com")
        finally:
            p.stop()

    def test_parse_openssl_style_string(self):
        # openssl 空格补位的单位日
        dt = _parse_cert_datetime("Jun  1 12:00:00 2035 GMT")
        assert dt.year == 2035 and dt.month == 6 and dt.day == 1
        assert dt.tzinfo is not None
