"""Tests for oprim.alipay_query_order."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from oprim.alipay_create_qr_order import AlipayConfig
from oprim.alipay_query_order import AlipayOrderStatus, alipay_query_order
from oprim.alipay_create_qr_order import AlipayAPIError

CONFIG = AlipayConfig(
    app_id="2021000000000000",
    app_private_key="FAKE_PRIVATE_KEY",
    alipay_public_key="FAKE_PUBLIC_KEY",
    notify_url="https://example.com/notify",
)


def _success_response(trade_status: str) -> dict[str, object]:
    return {
        "out_trade_no": "ORDER-001",
        "trade_status": trade_status,
        "total_amount": "10.00",
        "trade_no": "2024010112345678",
    }


@pytest.mark.asyncio
async def test_query_order_wait_buyer_pay() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = _success_response("WAIT_BUYER_PAY")
        mock_factory.return_value = mock_client

        result = await alipay_query_order(config=CONFIG, out_trade_no="ORDER-001")

    assert isinstance(result, AlipayOrderStatus)
    assert result.trade_status == "WAIT_BUYER_PAY"
    assert result.out_trade_no == "ORDER-001"


@pytest.mark.asyncio
async def test_query_order_trade_success() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = _success_response("TRADE_SUCCESS")
        mock_factory.return_value = mock_client

        result = await alipay_query_order(config=CONFIG, out_trade_no="ORDER-001")

    assert result.trade_status == "TRADE_SUCCESS"
    assert result.total_amount == Decimal("10.00")
    assert result.trade_no == "2024010112345678"


@pytest.mark.asyncio
async def test_query_order_trade_closed() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = _success_response("TRADE_CLOSED")
        mock_factory.return_value = mock_client

        result = await alipay_query_order(config=CONFIG, out_trade_no="ORDER-001")

    assert result.trade_status == "TRADE_CLOSED"


@pytest.mark.asyncio
async def test_query_order_trade_finished() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = _success_response("TRADE_FINISHED")
        mock_factory.return_value = mock_client

        result = await alipay_query_order(config=CONFIG, out_trade_no="ORDER-001")

    assert result.trade_status == "TRADE_FINISHED"


@pytest.mark.asyncio
async def test_query_order_api_error_raises() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = {
            "sub_code": "ACQ.TRADE_NOT_EXIST",
            "sub_msg": "Trade not exist",
        }
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayAPIError, match="ACQ.TRADE_NOT_EXIST"):
            await alipay_query_order(config=CONFIG, out_trade_no="ORDER-MISSING")


@pytest.mark.asyncio
async def test_query_order_missing_trade_status_raises() -> None:
    with patch("oprim.alipay_query_order._make_alipay_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.api_alipay_trade_query.return_value = {}
        mock_factory.return_value = mock_client

        with pytest.raises(AlipayAPIError, match="Missing trade_status"):
            await alipay_query_order(config=CONFIG, out_trade_no="ORDER-001")
