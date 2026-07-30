"""Tests for oprim.stripe_refund_payment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import stripe as stripe_sdk

from oprim.stripe_create_payment_intent import StripeAPIError, StripeConfig
from oprim.stripe_refund_payment import stripe_refund_payment

CONFIG = StripeConfig(
    api_key="sk_test_fake_key",
    webhook_secret="whsec_fake_secret",
)


@pytest.mark.asyncio
async def test_full_refund_returns_true() -> None:
    mock_refund = MagicMock()
    with patch("stripe.Refund.create", return_value=mock_refund) as mock_create:
        result = await stripe_refund_payment(
            config=CONFIG,
            intent_id="pi_test_001",
        )

    assert result is True
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["payment_intent"] == "pi_test_001"
    assert "amount" not in call_kwargs


@pytest.mark.asyncio
async def test_partial_refund_returns_true() -> None:
    mock_refund = MagicMock()
    with patch("stripe.Refund.create", return_value=mock_refund) as mock_create:
        result = await stripe_refund_payment(
            config=CONFIG,
            intent_id="pi_test_001",
            amount=500,
        )

    assert result is True
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["amount"] == 500


@pytest.mark.asyncio
async def test_reason_duplicate() -> None:
    mock_refund = MagicMock()
    with patch("stripe.Refund.create", return_value=mock_refund) as mock_create:
        result = await stripe_refund_payment(
            config=CONFIG,
            intent_id="pi_test_001",
            reason="duplicate",
        )

    assert result is True
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reason"] == "duplicate"


@pytest.mark.asyncio
async def test_reason_fraudulent() -> None:
    mock_refund = MagicMock()
    with patch("stripe.Refund.create", return_value=mock_refund) as mock_create:
        result = await stripe_refund_payment(
            config=CONFIG,
            intent_id="pi_test_001",
            reason="fraudulent",
        )

    assert result is True
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reason"] == "fraudulent"


@pytest.mark.asyncio
async def test_api_error_raises_stripe_api_error() -> None:
    with patch(
        "stripe.Refund.create",
        side_effect=stripe_sdk.StripeError("Charge already refunded"),
    ):
        with pytest.raises(StripeAPIError, match="Charge already refunded"):
            await stripe_refund_payment(
                config=CONFIG,
                intent_id="pi_test_001",
            )
