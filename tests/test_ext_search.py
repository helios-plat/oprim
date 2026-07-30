"""Tests for oprim.ext_search_upsert/delete."""

from __future__ import annotations

import pytest
from obase import ProviderRegistry
from obase.search_providers import LogSearchProvider

from oprim._exceptions import SearchIndexOprimError
from oprim.ext_search_delete import ext_search_delete
from oprim.ext_search_upsert import ext_search_upsert


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


class TestExtSearchUpsert:
    async def test_upsert_success(self):
        provider = LogSearchProvider()
        ProviderRegistry.get().register_generic("search", "log", provider)
        ok = await ext_search_upsert("log", index="products", document={"id": "p1", "title": "T恤"})
        assert ok is True
        assert provider.indexed["products"]["p1"]["title"] == "T恤"

    async def test_provider_not_found(self):
        with pytest.raises(SearchIndexOprimError, match="not found"):
            await ext_search_upsert("ghost", index="products", document={"id": "p1"})


class TestExtSearchDelete:
    async def test_delete_success(self):
        provider = LogSearchProvider()
        ProviderRegistry.get().register_generic("search", "log", provider)
        await ext_search_upsert("log", index="products", document={"id": "p1"})
        ok = await ext_search_delete("log", index="products", doc_id="p1")
        assert ok is True

    async def test_delete_missing_doc_returns_false(self):
        provider = LogSearchProvider()
        ProviderRegistry.get().register_generic("search", "log", provider)
        ok = await ext_search_delete("log", index="products", doc_id="ghost")
        assert ok is False

    async def test_provider_not_found(self):
        with pytest.raises(SearchIndexOprimError, match="not found"):
            await ext_search_delete("ghost", index="products", doc_id="p1")
