"""Tests for oprim.ext_tax_calculate."""

from __future__ import annotations

import pytest
from obase import ProviderRegistry
from obase.tax_providers import FlatRateTaxProvider

from oprim._exceptions import TaxOprimError
from oprim.ext_tax_calculate import ext_tax_calculate


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestExtTaxCalculate:
    async def test_calculate_success(self):
        ProviderRegistry.get().register_generic(
            "tax", "flat", FlatRateTaxProvider(rate_percent=8.0)
        )
        result = await ext_tax_calculate(
            "flat", address={"country": "US"}, items=[{"amount_cents": 1000}]
        )
        assert result["tax_cents"] == 80

    async def test_provider_not_found(self):
        with pytest.raises(TaxOprimError, match="not found"):
            await ext_tax_calculate("ghost", address={}, items=[])

    async def test_provider_error_wrapped(self):
        class _Bad:
            async def calculate(self, *, address, items):
                raise RuntimeError("tax service down")

        ProviderRegistry.get().register_generic("tax", "bad", _Bad())
        with pytest.raises(TaxOprimError, match="calculate failed"):
            await ext_tax_calculate("bad", address={}, items=[])
