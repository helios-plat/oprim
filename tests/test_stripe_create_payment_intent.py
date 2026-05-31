"""Tests for oprim.stripe_create_payment_intent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import stripe as stripe_sdk

from oprim.stripe_create_payment_intent import (
    StripeAPIError,
    StripeConfig,
    StripePaymentIntent,
    stripe_create_payment_intent,
)

CONFIG = StripeConfig(
    api_key="sk_test_fake_key",
    webhook_secret="whsec_fake_secret",
)


def _make_mock_intent(
    *,
    intent_id: str = "pi_test_001",
    client_secret: str = "pi_test_001_secret_abc",
    amount: int = 1000,
    currency: str = "usd",
    status: str = "requires_payment_method",
    metadata: dict[str, str] | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.id = intent_id
    mock.client_secret = client_secret
    mock.amount = amount
    mock.currency = currency
    mock.status = status
    mock.metadata = metadata or {}
    return mock


@pytest.mark.asyncio
async def test_create_payment_intent_success() -> None:
    mock_intent = _make_mock_intent()
    with patch("stripe.PaymentIntent.create", return_value=mock_intent):
        result = await stripe_create_payment_intent(
            config=CONFIG,
            amount=1000,
            currency="usd",
        )

    assert isinstance(result, StripePaymentIntent)
    assert result.intent_id == "pi_test_001"
    assert result.amount == 1000
    assert result.currency == "usd"
    assert result.status == "requires_payment_method"


@pytest.mark.asyncio
async def test_create_payment_intent_eur_currency() -> None:
    mock_intent = _make_mock_intent(amount=500, currency="eur", status="requires_payment_method")
    with patch("stripe.PaymentIntent.create", return_value=mock_intent):
        result = await stripe_create_payment_intent(
            config=CONFIG,
            amount=500,
            currency="eur",
        )

    assert result.currency == "eur"
    assert result.amount == 500


@pytest.mark.asyncio
async def test_metadata_injected() -> None:
    metadata = {"order_id": "ORDER-001", "user_id": "USER-123"}
    mock_intent = _make_mock_intent(metadata=metadata)
    with patch("stripe.PaymentIntent.create", return_value=mock_intent) as mock_create:
        result = await stripe_create_payment_intent(
            config=CONFIG,
            amount=1000,
            currency="usd",
            metadata=metadata,
        )

    assert result.metadata == metadata
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["metadata"] == metadata


@pytest.mark.asyncio
async def test_api_error_raises_stripe_api_error() -> None:
    with patch(
        "stripe.PaymentIntent.create",
        side_effect=stripe_sdk.StripeError("Invalid API key"),
    ):
        with pytest.raises(StripeAPIError, match="Invalid API key"):
            await stripe_create_payment_intent(
                config=CONFIG,
                amount=1000,
                currency="usd",
            )


@pytest.mark.asyncio
async def test_minimum_amount_one_cent() -> None:
    mock_intent = _make_mock_intent(amount=1, currency="usd")
    with patch("stripe.PaymentIntent.create", return_value=mock_intent) as mock_create:
        result = await stripe_create_payment_intent(
            config=CONFIG,
            amount=1,
            currency="usd",
        )

    assert result.amount == 1
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["amount"] == 1
