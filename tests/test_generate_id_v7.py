"""Tests for oprim.generate_id_v7."""

from __future__ import annotations

import pytest

from oprim.generate_id_v7 import generate_id_v7


class TestGenerateIdV7:
    def test_has_prefix(self):
        result = generate_id_v7("cust")
        assert result.startswith("cust_")

    def test_unique_across_calls(self):
        ids = {generate_id_v7("prod") for _ in range(50)}
        assert len(ids) == 50

    def test_empty_prefix_rejected(self):
        with pytest.raises(ValueError, match="prefix"):
            generate_id_v7("")

    def test_lexicographic_time_order(self):
        import time

        first = generate_id_v7("x")
        time.sleep(0.01)
        second = generate_id_v7("x")
        assert first < second
