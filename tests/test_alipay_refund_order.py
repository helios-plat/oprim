"""Tests for oprim.alipay_refund_order."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from oprim.alipay_create_qr_order import AlipayAPIError, AlipayConfig
from oprim.alipay_refund_order import alipay_refund_order

CONFIG = AlipayConfig(
    app_id="2021000000000000",
    app_private_key="FAKE_PRIVATE_KEY",
    alipay_public_key="FAKE_PUBLIC_KEY",
    notify_url="https://example.com/notify",
)

SUCCESS_RESPONSE: dict[str, object] = {
    "out_trade_no": "ORDER-001",
    "refund_fee": "10.00",
    "trade_no": "2024010112345678",
}


@pytest.mark.asyncio
async def test_full_refund_success() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        result = await alipay_refund_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            refund_amount=Decimal("10.00"),
        )

    assert result is True


@pytest.mark.asyncio
async def test_partial_refund_success() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = {
            **SUCCESS_RESPONSE,
            "refund_fee": "5.00",
        }
        mock_factory.return_value = mock_client

        result = await alipay_refund_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            refund_amount=Decimal("5.00"),
        )

    assert result is True
    call_kwargs = mock_client.api_alipay_trade_refund.call_args.kwargs
    assert call_kwargs["refund_amount"] == "5.00"


@pytest.mark.asyncio
async def test_duplicate_refund_raises() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = {
            "sub_code": "ACQ.TRADE_HAS_FINISHED",
            "sub_msg": "Trade already finished",
        }
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayAPIError, match="ACQ.TRADE_HAS_FINISHED"):
            await alipay_refund_order(
                config=CONFIG,
                out_trade_no="ORDER-001",
                refund_amount=Decimal("10.00"),
            )


@pytest.mark.asyncio
async def test_api_error_raises() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = {
            "sub_code": "ACQ.SYSTEM_ERROR",
            "sub_msg": "System error",
        }
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayAPIError, match="ACQ.SYSTEM_ERROR"):
            await alipay_refund_order(
                config=CONFIG,
                out_trade_no="ORDER-001",
                refund_amount=Decimal("10.00"),
            )


@pytest.mark.asyncio
async def test_refund_reason_included_in_request() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        await alipay_refund_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            refund_amount=Decimal("10.00"),
            refund_reason="Customer request",
        )

    call_kwargs = mock_client.api_alipay_trade_refund.call_args.kwargs
    assert call_kwargs["refund_reason"] == "Customer request"


@pytest.mark.asyncio
async def test_empty_refund_reason_omitted() -> None:
    with patch("oprim.alipay_refund_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_refund.return_value = SUCCESS_RESPONSE
        mock_factory.return_value = mock_client

        await alipay_refund_order(
            config=CONFIG,
            out_trade_no="ORDER-001",
            refund_amount=Decimal("10.00"),
            refund_reason="",
        )

    call_kwargs = mock_client.api_alipay_trade_refund.call_args.kwargs
    assert "refund_reason" not in call_kwargs
