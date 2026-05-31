"""Tests for oprim.stripe_retrieve_payment_intent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import stripe as stripe_sdk

from oprim.stripe_create_payment_intent import StripeAPIError, StripeConfig, StripePaymentIntent
from oprim.stripe_retrieve_payment_intent import stripe_retrieve_payment_intent

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
async def test_retrieve_requires_payment_method_status() -> None:
    mock_intent = _make_mock_intent(status="requires_payment_method")
    with patch("stripe.PaymentIntent.retrieve", return_value=mock_intent):
        result = await stripe_retrieve_payment_intent(
            config=CONFIG,
            intent_id="pi_test_001",
        )

    assert isinstance(result, StripePaymentIntent)
    assert result.status == "requires_payment_method"
    assert result.intent_id == "pi_test_001"


@pytest.mark.asyncio
async def test_retrieve_processing_status() -> None:
    mock_intent = _make_mock_intent(status="processing")
    with patch("stripe.PaymentIntent.retrieve", return_value=mock_intent):
        result = await stripe_retrieve_payment_intent(
            config=CONFIG,
            intent_id="pi_test_001",
        )

    assert result.status == "processing"


@pytest.mark.asyncio
async def test_retrieve_succeeded_status() -> None:
    mock_intent = _make_mock_intent(status="succeeded")
    with patch("stripe.PaymentIntent.retrieve", return_value=mock_intent):
        result = await stripe_retrieve_payment_intent(
            config=CONFIG,
            intent_id="pi_test_001",
        )

    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_retrieve_not_found_raises_api_error() -> None:
    with patch(
        "stripe.PaymentIntent.retrieve",
        side_effect=stripe_sdk.StripeError("No such payment_intent: pi_unknown"),
    ):
        with pytest.raises(StripeAPIError, match="No such payment_intent"):
            await stripe_retrieve_payment_intent(
                config=CONFIG,
                intent_id="pi_unknown",
            )


@pytest.mark.asyncio
async def test_retrieve_api_error_raises_stripe_api_error() -> None:
    with patch(
        "stripe.PaymentIntent.retrieve",
        side_effect=stripe_sdk.StripeError("Connection error"),
    ):
        with pytest.raises(StripeAPIError, match="Connection error"):
            await stripe_retrieve_payment_intent(
                config=CONFIG,
                intent_id="pi_test_001",
            )
