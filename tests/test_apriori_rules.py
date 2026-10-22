"""Tests for tide._platform_ops.apriori_rules."""

import pytest

from oprim.apriori_rules import apriori_rules


class TestAprioriBasic:
    def test_beer_diaper_classic(self):
        """Classic beer-diaper association rule dataset."""
        transactions = [
            ["beer", "diaper", "nuts"],
            ["beer", "diaper"],
            ["beer", "diaper", "milk"],
            ["diaper", "milk"],
            ["beer", "milk"],
            ["beer", "diaper", "nuts", "milk"],
        ]
        result = apriori_rules(transactions, min_support=0.3, min_confidence=0.5, min_lift=1.0)
        assert "rules" in result
        assert result["n_transactions"] == 6
        assert result["n_unique_items"] == 4
        # Beer -> Diaper should be found
        rules = result["rules"]
        beer_diaper = [
            r
            for r in rules
            if frozenset(["beer"]) <= r["antecedent"] and frozenset(["diaper"]) <= r["consequent"]
        ]
        assert len(beer_diaper) > 0 or any(
            frozenset(["diaper"]) <= r["antecedent"] and frozenset(["beer"]) <= r["consequent"]
            for r in rules
        )

    def test_return_itemsets(self):
        transactions = [["a", "b"], ["a", "b", "c"], ["a", "c"], ["b", "c"]]
        result = apriori_rules(transactions, min_support=0.4, return_itemsets=True)
        assert "frequent_itemsets" in result
        assert len(result["frequent_itemsets"]) > 0

    def test_max_length_1_no_rules(self):
        transactions = [["a", "b"], ["a", "c"], ["b", "c"]]
        result = apriori_rules(transactions, min_support=0.3, max_length=1)
        assert len(result["rules"]) == 0

    def test_apriori_property(self):
        """All subsets of frequent itemsets must also be frequent."""
        transactions = [
            ["a", "b", "c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
            ["a", "b", "c", "d"],
        ]
        result = apriori_rules(transactions, min_support=0.3, return_itemsets=True)
        freq_sets = {entry["itemset"] for entry in result["frequent_itemsets"]}
        for fs in list(freq_sets):
            if len(fs) >= 2:
                for item in fs:
                    sub = fs - frozenset([item])
                    assert sub in freq_sets, (
                        f"Apriori property violated: {sub} not frequent but {fs} is"
                    )

    def test_confidence_lift_calculation(self):
        transactions = [["a", "b"]] * 8 + [["a"]] * 2
        result = apriori_rules(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0)
        rules = result["rules"]
        for r in rules:
            assert 0 < r["confidence"] <= 1.0
            assert r["lift"] >= 0


class TestAprioriValidation:
    def test_invalid_min_support(self):
        with pytest.raises(ValueError, match="min_support"):
            apriori_rules([["a", "b"]], min_support=0.0)

    def test_invalid_min_confidence(self):
        with pytest.raises(ValueError, match="min_confidence"):
            apriori_rules([["a", "b"]], min_confidence=1.5)

    def test_too_few_transactions(self):
        with pytest.raises(ValueError, match="at least 2"):
            apriori_rules([["a", "b"]])

    def test_invalid_max_length(self):
        with pytest.raises(ValueError, match="max_length"):
            apriori_rules([["a"], ["b"]], max_length=0)
