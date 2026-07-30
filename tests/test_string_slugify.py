"""Tests for oprim.string_slugify."""

from __future__ import annotations

from oprim.string_slugify import string_slugify


class TestStringSlugify:
    def test_basic_lowercase_hyphenate(self):
        assert string_slugify("Men's Classic  T-Shirt!") == "men-s-classic-t-shirt"

    def test_preserves_chinese_characters(self):
        assert string_slugify("经典T恤") == "经典t恤"

    def test_strips_edge_hyphens(self):
        assert string_slugify("  --hello--  ") == "hello"

    def test_collapses_multiple_separators(self):
        assert string_slugify("a   b---c") == "a-b-c"

    def test_empty_string(self):
        assert string_slugify("") == ""

    def test_idempotent(self):
        once = string_slugify("Hello World")
        assert string_slugify(once) == once
