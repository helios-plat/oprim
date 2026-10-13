"""Tests for oprim.ext_pay_authorize/capture/refund/cancel."""

from __future__ import annotations

import pytest
from obase import ProviderRegistry

from oprim._exceptions import PaymentOprimError
from oprim.ext_pay_authorize import ext_pay_authorize
from oprim.ext_pay_cancel import ext_pay_cancel
from oprim.ext_pay_capture import ext_pay_capture
from oprim.ext_pay_refund import ext_pay_refund


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class _FakePaymentProvider:
    def __init__(self) -> None:
        self.intents: dict[str, dict] = {}

    async def authorize(self, *, amount, currency, meta=None):
        intent_id = f"fake_{len(self.intents)}"
        self.intents[intent_id] = {"status": "authorized", "amount": amount}
        return {"intent_id": intent_id, "status": "authorized"}

    async def capture(self, *, intent_id):
        if self.intents[intent_id]["status"] != "authorized":
            raise ValueError("not authorized")
        self.intents[intent_id]["status"] = "captured"
        return {"intent_id": intent_id, "status": "captured"}

    async def refund(self, *, intent_id, amount):
        if self.intents[intent_id]["status"] != "captured":
            raise ValueError("not captured")
        self.intents[intent_id]["status"] = "refunded"
        return {"intent_id": intent_id, "status": "refunded"}

    async def cancel(self, *, intent_id):
        if self.intents[intent_id]["status"] != "authorized":
            raise ValueError("cannot cancel")
        self.intents[intent_id]["status"] = "canceled"
        return {"intent_id": intent_id, "status": "canceled"}


class TestExtPayAuthorize:
    async def test_authorize_success(self):
        ProviderRegistry.get().register_generic("payment", "fake", _FakePaymentProvider())
        result = await ext_pay_authorize("fake", amount=1000, currency="USD")
        assert result["status"] == "authorized"

    async def test_provider_not_found(self):
        with pytest.raises(PaymentOprimError, match="not found"):
            await ext_pay_authorize("ghost", amount=1000, currency="USD")


class TestExtPayCapture:
    async def test_capture_success(self):
        provider = _FakePaymentProvider()
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        auth = await ext_pay_authorize("fake", amount=1000, currency="USD")
        result = await ext_pay_capture("fake", intent_id=auth["intent_id"])
        assert result["status"] == "captured"

    async def test_capture_unauthorized_intent_raises(self):
        provider = _FakePaymentProvider()
        provider.intents["x"] = {"status": "captured", "amount": 100}
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        with pytest.raises(PaymentOprimError, match="capture failed"):
            await ext_pay_capture("fake", intent_id="x")

    async def test_provider_not_found(self):
        with pytest.raises(PaymentOprimError, match="not found"):
            await ext_pay_capture("ghost", intent_id="x")


class TestExtPayRefund:
    async def test_refund_success(self):
        provider = _FakePaymentProvider()
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        auth = await ext_pay_authorize("fake", amount=1000, currency="USD")
        await ext_pay_capture("fake", intent_id=auth["intent_id"])
        result = await ext_pay_refund("fake", intent_id=auth["intent_id"], amount=1000)
        assert result["status"] == "refunded"

    async def test_refund_uncaptured_raises(self):
        provider = _FakePaymentProvider()
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        auth = await ext_pay_authorize("fake", amount=1000, currency="USD")
        with pytest.raises(PaymentOprimError, match="refund failed"):
            await ext_pay_refund("fake", intent_id=auth["intent_id"], amount=1000)

    async def test_provider_not_found(self):
        with pytest.raises(PaymentOprimError, match="not found"):
            await ext_pay_refund("ghost", intent_id="x", amount=1000)


class TestExtPayCancel:
    async def test_cancel_success(self):
        provider = _FakePaymentProvider()
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        auth = await ext_pay_authorize("fake", amount=1000, currency="USD")
        result = await ext_pay_cancel("fake", intent_id=auth["intent_id"])
        assert result["status"] == "canceled"

    async def test_cancel_captured_raises(self):
        provider = _FakePaymentProvider()
        ProviderRegistry.get().register_generic("payment", "fake", provider)
        auth = await ext_pay_authorize("fake", amount=1000, currency="USD")
        await ext_pay_capture("fake", intent_id=auth["intent_id"])
        with pytest.raises(PaymentOprimError, match="cancel failed"):
            await ext_pay_cancel("fake", intent_id=auth["intent_id"])

    async def test_provider_not_found(self):
        with pytest.raises(PaymentOprimError, match="not found"):
            await ext_pay_cancel("ghost", intent_id="x")


class TestManualPaymentProviderIntegration:
    """Confirms the atoms work against the real ManualPaymentProvider, not just a fake."""

    async def test_full_authorize_capture_refund_cycle(self):
        from obase.payment_providers import ManualPaymentProvider

        ProviderRegistry.get().register_generic("payment", "manual", ManualPaymentProvider())
        auth = await ext_pay_authorize("manual", amount=500, currency="CNY")
        captured = await ext_pay_capture("manual", intent_id=auth["intent_id"])
        assert captured["status"] == "captured"
        refunded = await ext_pay_refund("manual", intent_id=auth["intent_id"], amount=500)
        assert refunded["status"] == "refunded"
