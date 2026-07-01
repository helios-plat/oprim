"""Tests for oprim.stripe_verify_webhook_signature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import stripe as stripe_sdk

from oprim.stripe_create_payment_intent import StripeConfig
from oprim.stripe_verify_webhook_signature import (
    StripeInvalidSignatureError,
    stripe_verify_webhook_signature,
)

CONFIG = StripeConfig(
    api_key="sk_test_fake_key",
    webhook_secret="whsec_fake_secret",
)

CONFIG_NO_SECRET = StripeConfig(
    api_key="sk_test_fake_key",
    webhook_secret=None,
)

PAYLOAD = b'{"id": "evt_001", "type": "payment_intent.succeeded", "object": "event"}'
SIG_HEADER = "t=1234567890,v1=abc123"


def test_valid_webhook_returns_event_dict() -> None:
    mock_event = {"id": "evt_001", "type": "payment_intent.succeeded", "object": "event"}
    with patch("stripe.Webhook.construct_event", return_value=mock_event):
        result = stripe_verify_webhook_signature(
            config=CONFIG,
            payload=PAYLOAD,
            signature=SIG_HEADER,
        )

    assert isinstance(result, dict)
    assert result["type"] == "payment_intent.succeeded"


def test_event_has_type_key() -> None:
    mock_event = {"id": "evt_002", "type": "charge.refunded", "object": "event"}
    with patch("stripe.Webhook.construct_event", return_value=mock_event):
        result = stripe_verify_webhook_signature(
            config=CONFIG,
            payload=PAYLOAD,
            signature=SIG_HEADER,
        )

    assert "type" in result
    assert result["type"] == "charge.refunded"


def test_invalid_signature_raises() -> None:
    with patch(
        "stripe.Webhook.construct_event",
        side_effect=stripe_sdk.error.SignatureVerificationError(
            "No signatures found matching the expected signature for payload",
            sig_header=SIG_HEADER,
        ),
    ):
        with pytest.raises(StripeInvalidSignatureError):
            stripe_verify_webhook_signature(
                config=CONFIG,
                payload=PAYLOAD,
                signature=SIG_HEADER,
            )


def test_webhook_secret_none_raises_value_error() -> None:
    with pytest.raises(ValueError, match="webhook_secret not configured"):
        stripe_verify_webhook_signature(
            config=CONFIG_NO_SECRET,
            payload=PAYLOAD,
            signature=SIG_HEADER,
        )


def test_tampered_payload_raises_invalid_signature_error() -> None:
    tampered = b'{"id": "evt_TAMPERED", "type": "payment_intent.succeeded"}'
    with patch(
        "stripe.Webhook.construct_event",
        side_effect=stripe_sdk.error.SignatureVerificationError(
            "Signature mismatch",
            sig_header=SIG_HEADER,
        ),
    ):
        with pytest.raises(StripeInvalidSignatureError):
            stripe_verify_webhook_signature(
                config=CONFIG,
                payload=tampered,
                signature=SIG_HEADER,
            )
