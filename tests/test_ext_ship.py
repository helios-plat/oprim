"""Tests for oprim.ext_ship_get_rates/create_label/cancel_label."""

from __future__ import annotations

import pytest
from obase import ProviderRegistry
from obase.fulfillment_providers import ManualFulfillmentProvider

from oprim._exceptions import ShippingOprimError
from oprim.ext_ship_cancel_label import ext_ship_cancel_label
from oprim.ext_ship_create_label import ext_ship_create_label
from oprim.ext_ship_get_rates import ext_ship_get_rates


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestExtShipGetRates:
    async def test_get_rates_success(self):
        ProviderRegistry.get().register_generic(
            "fulfillment", "manual", ManualFulfillmentProvider(flat_rate_cents=750)
        )
        rates = await ext_ship_get_rates(
            "manual", package={"weight_g": 500}, address={"country": "CN"}
        )
        assert rates == [{"carrier": "manual", "service": "standard", "rate_cents": 750}]

    async def test_provider_not_found(self):
        with pytest.raises(ShippingOprimError, match="not found"):
            await ext_ship_get_rates("ghost", package={}, address={})


class TestExtShipCreateLabel:
    async def test_create_label_success(self):
        ProviderRegistry.get().register_generic(
            "fulfillment", "manual", ManualFulfillmentProvider()
        )
        result = await ext_ship_create_label("manual", shipment_info={"to": "somewhere"})
        assert result["status"] == "created"
        assert "tracking_number" in result

    async def test_provider_not_found(self):
        with pytest.raises(ShippingOprimError, match="not found"):
            await ext_ship_create_label("ghost", shipment_info={})

    async def test_provider_error_wrapped(self):
        class _Bad:
            async def create_label(self, *, shipment_info):
                raise RuntimeError("carrier down")

        ProviderRegistry.get().register_generic("fulfillment", "bad", _Bad())
        with pytest.raises(ShippingOprimError, match="create_label failed"):
            await ext_ship_create_label("bad", shipment_info={})


class TestExtShipCancelLabel:
    async def test_cancel_label_success(self):
        provider = ManualFulfillmentProvider()
        ProviderRegistry.get().register_generic("fulfillment", "manual", provider)
        label = await ext_ship_create_label("manual", shipment_info={"to": "x"})
        ok = await ext_ship_cancel_label("manual", tracking_number=label["tracking_number"])
        assert ok is True

    async def test_cancel_unknown_label_returns_false(self):
        ProviderRegistry.get().register_generic(
            "fulfillment", "manual", ManualFulfillmentProvider()
        )
        ok = await ext_ship_cancel_label("manual", tracking_number="ghost")
        assert ok is False

    async def test_provider_not_found(self):
        with pytest.raises(ShippingOprimError, match="not found"):
            await ext_ship_cancel_label("ghost", tracking_number="x")
