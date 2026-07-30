"""Tests for oprim.alipay_create_qr_order."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from oprim.alipay_create_qr_order import (
    AlipayAPIError,
    AlipayConfig,
    AlipayQRCode,
    alipay_create_qr_order,
)

CONFIG = AlipayConfig(
    app_id="2021000000000000",
    app_private_key="FAKE_PRIVATE_KEY",
    alipay_public_key="FAKE_PUBLIC_KEY",
    notify_url="https://example.com/notify",
    sandbox=False,
)

SANDBOX_CONFIG = AlipayConfig(
    app_id="2021000000000000",
    app_private_key="FAKE_PRIVATE_KEY",
    alipay_public_key="FAKE_PUBLIC_KEY",
    notify_url="https://example.com/notify",
    sandbox=True,
)

SUCCESS_RESPONSE = {
    "qr_code": "https://qr.alipay.com/bax03431liymlvcrx70r",
    "out_trade_no": "ORDER-001",
}

API_ERROR_RESPONSE = {
    "sub_code": "ACQ.TRADE_HAS_CLOSED",
    "sub_msg": "Trade already closed",
}


@pytest.mark.asyncio
async def test_create_qr_order_success() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        result = await alipay_create_qr_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            total_amount=Decimal("10.00"),
            subject="Test Product",
        )

    assert isinstance(result, AlipayQRCode)
    assert result.qr_code_url == "https://qr.alipay.com/bax03431liymlvcrx70r"
    assert result.out_trade_no == "ORDER-001"


@pytest.mark.asyncio
async def test_create_qr_order_api_error_raises() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = API_ERROR_RESPONSE
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayAPIError, match="ACQ.TRADE_HAS_CLOSED"):
            await alipay_create_qr_order(
                config=CONFIG,
                out_trade_no="ORDER-001",
                total_amount=Decimal("10.00"),
                subject="Test Product",
            )


@pytest.mark.asyncio
async def test_sandbox_uses_sandbox_url() -> None:
    """sandbox=True passes debug=True to AliPay constructor."""
    with patch("oprim.alipay_create_qr_order.AliPay") as mock_alipay_cls:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = SUCCESS_RESPONSE
        mock_alipay_cls.return_value = mock_client

        await alipay_create_qr_order(
            config=SANDBOX_CONFIG,
            out_trade_no="ORDER-SANDBOX",
            total_amount=Decimal("1.00"),
            subject="Sandbox Test",
        )

    _, kwargs = mock_alipay_cls.call_args
    assert kwargs["debug"] is True


@pytest.mark.asyncio
async def test_minimum_amount_001() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = {
            "qr_code": "https://qr.alipay.com/min",
            "out_trade_no": "ORDER-MIN",
        }
        mock_factory.return_value = mock_client

        result = await alipay_create_qr_order(
            config=CONFIG,
            out_trade_no="ORDER-MIN",
            total_amount=Decimal("0.01"),
            subject="Min Amount",
        )

    assert result.out_trade_no == "ORDER-MIN"
    # Verify 0.01 passed to SDK as string
    call_kwargs = mock_client.api_alipay_trade_precreate.call_args
    assert call_kwargs.kwargs["total_amount"] == "0.01"


@pytest.mark.asyncio
async def test_large_amount() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = {
            "qr_code": "https://qr.alipay.com/large",
            "out_trade_no": "ORDER-LARGE",
        }
        mock_factory.return_value = mock_client

        result = await alipay_create_qr_order(
            config=CONFIG,
            out_trade_no="ORDER-LARGE",
            total_amount=Decimal("99999.00"),
            subject="Large Order",
        )

    assert result.out_trade_no == "ORDER-LARGE"
    call_kwargs = mock_client.api_alipay_trade_precreate.call_args
    assert call_kwargs.kwargs["total_amount"] == "99999.00"


@pytest.mark.asyncio
async def test_body_none_omitted() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        await alipay_create_qr_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            total_amount=Decimal("10.00"),
            subject="No Body",
            body=None,
        )

    call_kwargs = mock_client.api_alipay_trade_precreate.call_args.kwargs
    assert "body" not in call_kwargs


@pytest.mark.asyncio
async def test_body_included_when_provided() -> None:
    with patch("oprim.alipay_create_qr_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_precreate.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        await alipay_create_qr_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            total_amount=Decimal("10.00"),
            subject="With Body",
            body="Product description",
        )

    call_kwargs = mock_client.api_alipay_trade_precreate.call_args.kwargs
    assert call_kwargs["body"] == "Product description"
