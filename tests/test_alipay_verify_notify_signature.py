"""Tests for oprim.alipay_verify_notify_signature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.alipay_create_qr_order import AlipayConfig
from oprim.alipay_verify_notify_signature import (
    AlipayInvalidSignatureError,
    alipay_verify_notify_signature,
)

CONFIG = AlipayConfig(
    app_id="2021000000000000",
    app_private_key="FAKE_PRIVATE_KEY",
    alipay_public_key="FAKE_PUBLIC_KEY",
    notify_url="https://example.com/notify",
    sandbox=False,
)

VALID_NOTIFY = {
    "trade_no": "2024010112345678",
    "out_trade_no": "ORDER-001",
    "trade_status": "TRADE_SUCCESS",
    "total_amount": "10.00",
    "sign_type": "RSA2",
    "sign": "VALID_SIGNATURE_BASE64",
}


def test_valid_signature_returns_true() -> None:
    with patch("oprim.alipay_verify_notify_signature._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.verify.return_value = True
        mock_factory.return_value = mock_client

        result = alipay_verify_notify_signature(config=CONFIG, notify_data=VALID_NOTIFY)

    assert result is True


def test_invalid_signature_raises() -> None:
    with patch("oprim.alipay_verify_notify_signature._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.verify.return_value = False
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayInvalidSignatureError, match="Signature verification failed"):
            alipay_verify_notify_signature(config=CONFIG, notify_data=VALID_NOTIFY)


def test_missing_sign_field_raises() -> None:
    notify_data = {
        "trade_no": "2024010112345678",
        "out_trade_no": "ORDER-001",
        "trade_status": "TRADE_SUCCESS",
    }
    with pytest.raises(AlipayInvalidSignatureError, match="Missing 'sign' field"):
        alipay_verify_notify_signature(config=CONFIG, notify_data=notify_data)


def test_sign_and_sign_type_excluded_from_verify() -> None:
    """sign and sign_type are stripped before calling verify."""
    with patch("oprim.alipay_verify_notify_signature._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.verify.return_value = True
        mock_factory.return_value = mock_client

        alipay_verify_notify_signature(config=CONFIG, notify_data=VALID_NOTIFY)

    verify_call_args = mock_client.verify.call_args
    data_passed = verify_call_args.args[0]
    assert "sign" not in data_passed
    assert "sign_type" not in data_passed


def test_chinese_values_handled() -> None:
    notify_data = {
        "trade_no": "2024010112345678",
        "out_trade_no": "ORDER-CN-001",
        "subject": "商品名称",  # Chinese: 商品名称
        "sign_type": "RSA2",
        "sign": "SOME_VALID_SIG",
    }
    with patch("oprim.alipay_verify_notify_signature._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.verify.return_value = True
        mock_factory.return_value = mock_client

        result = alipay_verify_notify_signature(config=CONFIG, notify_data=notify_data)

    assert result is True


def test_verify_exception_wrapped_as_invalid_signature_error() -> None:
    with patch("oprim.alipay_verify_notify_signature._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.verify.side_effect = RuntimeError("Crypto error")
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayInvalidSignatureError, match="Signature verification error"):
            alipay_verify_notify_signature(config=CONFIG, notify_data=VALID_NOTIFY)
