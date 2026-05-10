"""Tests for oprim.regime module."""

import numpy as np
import pandas as pd
import pytest

from oprim.regime import regime_filter_data, regime_label_align, regime_transition_matrix


# ============================================================
# regime_filter_data
# ============================================================
class TestRegimeFilterData:
    def test_hard_single(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({"val": range(10)}, index=idx)
        labels = pd.Series(["bull"] * 5 + ["bear"] * 5, index=idx)
        result = regime_filter_data(data, labels, "bull")
        assert len(result) == 5

    def test_hard_multi(self):
        idx = pd.date_range("2024-01-01", periods=9, freq="D")
        data = pd.DataFrame({"val": range(9)}, index=idx)
        labels = pd.Series(["bull"] * 3 + ["bear"] * 3 + ["neutral"] * 3, index=idx)
        result = regime_filter_data(data, labels, ["bull", "neutral"])
        assert len(result) == 6

    def test_soft_mode(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame({"val": [1, 2, 3, 4]}, index=idx)
        labels = pd.Series([
            {"bull": 0.8, "bear": 0.2},
            {"bull": 0.3, "bear": 0.7},
            {"bull": 0.6, "bear": 0.4},
            {"bull": 0.1, "bear": 0.9},
        ], index=idx)
        result = regime_filter_data(data, labels, "bull", mode="soft", min_probability=0.5)
        assert len(result) == 2  # indices 0 and 2

    def test_empty_result(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"val": range(5)}, index=idx)
        labels = pd.Series(["bear", "bear", "bull", "bear", "bear"], index=idx)
        result = regime_filter_data(data, labels, "bull")
        assert len(result) == 1

    def test_target_not_found_raises(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"val": range(5)}, index=idx)
        labels = pd.Series(["bear"] * 5, index=idx)
        with pytest.raises(ValueError, match="not found"):
            regime_filter_data(data, labels, "crash")

    def test_no_overlap_raises(self):
        idx1 = pd.date_range("2024-01-01", periods=5, freq="D")
        idx2 = pd.date_range("2025-01-01", periods=5, freq="D")
        data = pd.DataFrame({"val": range(5)}, index=idx1)
        labels = pd.Series(["bull"] * 5, index=idx2)
        with pytest.raises(ValueError, match="overlapping"):
            regime_filter_data(data, labels, "bull")


# ============================================================
# regime_transition_matrix
# ============================================================
class TestRegimeTransitionMatrix:
    def test_known_chain(self):
        """Two-state chain with known transitions."""
        labels = pd.Series(["A", "A", "B", "B", "A", "B", "A", "A", "B"])
        result = regime_transition_matrix(labels)
        assert result["transition_matrix"].shape == (2, 2)
        # Each row sums to 1
        np.testing.assert_allclose(
            result["transition_matrix"].sum(axis=1).values, [1.0, 1.0], rtol=1e-9
        )

    def test_sticky_state(self):
        labels = pd.Series(["A"] * 50 + ["B"] * 50)
        result = regime_transition_matrix(labels)
        # A→A should be high, A→B should be low (only 1 transition)
        assert result["transition_matrix"].loc["A", "A"] > 0.9

    def test_duration(self):
        labels = pd.Series(["A"] * 10 + ["B"] * 5 + ["A"] * 8)
        result = regime_transition_matrix(labels)
        assert result["duration_distribution"]["A"]["mean"] == pytest.approx(9.0)
        assert result["duration_distribution"]["B"]["mean"] == pytest.approx(5.0)

    def test_explicit_states(self):
        labels = pd.Series(["A", "B", "A"])
        result = regime_transition_matrix(labels, states=["A", "B", "C"])
        assert result["transition_matrix"].shape == (3, 3)

    def test_n_transitions(self):
        labels = pd.Series(["A", "B", "A", "B"])
        result = regime_transition_matrix(labels)
        assert result["n_transitions"] == 3

    def test_stationary_distribution(self):
        labels = pd.Series(["A", "B"] * 100)
        result = regime_transition_matrix(labels)
        # Should be approximately 50/50
        np.testing.assert_allclose(
            result["stationary_distribution"].values, [0.5, 0.5], atol=0.1
        )


# ============================================================
# regime_label_align
# ============================================================
class TestRegimeLabelAlign:
    def test_asof_basic(self):
        regime_idx = pd.DatetimeIndex(["2024-01-01", "2024-01-05", "2024-01-10"])
        labels = pd.Series(["bull", "bear", "bull"], index=regime_idx)
        target = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        result = regime_label_align(target, labels, method="asof")
        assert result.iloc[0] == "bull"
        assert result.iloc[4] == "bear"  # Jan 5
        assert result.iloc[9] == "bull"  # Jan 10

    def test_ffill(self):
        regime_idx = pd.DatetimeIndex(["2024-01-01", "2024-01-05"])
        labels = pd.Series(["bull", "bear"], index=regime_idx)
        target = pd.date_range("2024-01-01", "2024-01-07", freq="D")
        result = regime_label_align(target, labels, method="ffill")
        assert result.iloc[0] == "bull"
        assert result.iloc[4] == "bear"
        assert result.iloc[6] == "bear"  # forward-filled

    def test_tolerance(self):
        regime_idx = pd.DatetimeIndex(["2024-01-01", "2024-01-10"])
        labels = pd.Series(["bull", "bear"], index=regime_idx)
        target = pd.date_range("2024-01-01", "2024-01-15", freq="D")
        result = regime_label_align(target, labels, method="asof", tolerance=pd.Timedelta(days=3))
        # Jan 4 is within 3 days of Jan 1 → bull
        assert result.iloc[3] == "bull"

    def test_high_freq_alignment(self):
        """Daily regime → 5min ticks."""
        regime_idx = pd.date_range("2024-01-01", periods=3, freq="D")
        labels = pd.Series(["bull", "bear", "bull"], index=regime_idx)
        target = pd.date_range("2024-01-01", periods=48, freq="h")
        result = regime_label_align(target, labels, method="asof")
        assert result.iloc[0] == "bull"
        assert result.iloc[24] == "bear"


# ============================================================
# Additional coverage tests
# ============================================================
class TestRegimeExtra:
    def test_soft_mode_mixed_input_raises(self):
        """Soft mode with non-dict values should raise."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        data = pd.DataFrame({"val": [1, 2, 3]}, index=idx)
        labels = pd.Series(["bull", "bear", "bull"], index=idx)
        with pytest.raises(ValueError, match="mode='soft' requires dict values"):
            regime_filter_data(data, labels, "bull", mode="soft")

    def test_transition_no_duration(self):
        labels = pd.Series(["A", "B", "A"])
        result = regime_transition_matrix(labels, include_duration=False)
        assert result["duration_distribution"] == {}

    def test_transition_state_never_appears(self):
        labels = pd.Series(["A", "B", "A"])
        result = regime_transition_matrix(labels, states=["A", "B", "C"])
        assert result["duration_distribution"]["C"]["count"] == 0

    def test_transition_unknown_label_warns(self):
        """Unknown labels in data should warn when states is provided."""
        labels = pd.Series(["A", "B", "C", "A"])  # C is not in explicit states
        with pytest.warns(UserWarning, match="unknown labels"):
            result = regime_transition_matrix(labels, states=["A", "B"])
        # Only A->B and C->A transitions, C is skipped
        assert result["n_transitions"] == 1  # only A->B counted

    def test_ffill_with_tolerance(self):
        regime_idx = pd.DatetimeIndex(["2024-01-01"])
        labels = pd.Series(["bull"], index=regime_idx)
        target = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        result = regime_label_align(target, labels, method="ffill", tolerance=pd.Timedelta(days=3))
        # Jan 1-4 should be bull, Jan 5+ should be NaN
        assert result.iloc[0] == "bull"
        assert pd.isna(result.iloc[5])


# ============================================================
# Academic Validation Tests
# ============================================================
class TestRegimeAcademicValidation:
    def test_absorbing_state_chain(self):
        """Test transition matrix for chain with absorbing state."""
        # Chain with true absorbing state: A never leaves
        # A -> A -> A -> B -> B -> B -> B (A is absorbing because last A never transitions out)
        # But we need proper absorbing: A that only self-transitions
        # Let's use: X (absorbing) where all X transitions are X->X
        labels = pd.Series(["X", "X", "X"])  # Only X, all self-transitions
        result = regime_transition_matrix(labels)
        
        # X should have 1.0 self-transition probability
        assert result["transition_matrix"].loc["X", "X"] == 1.0
        
        # Stationary distribution should be valid (sums to 1)
        assert result["stationary_distribution"].sum() == pytest.approx(1.0)
        # All values should be non-negative
        assert (result["stationary_distribution"] >= 0).all()
        # Stationary for X should be 1.0
        assert result["stationary_distribution"]["X"] == pytest.approx(1.0)

    def test_two_state_markov_chain_known(self):
        """Test against known 2-state Markov chain parameters."""
        # Generate sequence from known transition matrix
        # P = [[0.7, 0.3], [0.4, 0.6]] (row-stochastic)
        # Expected stationary: solve pi = pi @ P -> pi = [0.571, 0.429]
        np.random.seed(42)
        P_true = np.array([[0.7, 0.3], [0.4, 0.6]])  # row-stochastic
        states = ["bull", "bear"]
        
        # Generate 1000 steps
        labels = []
        current = 0
        for _ in range(1000):
            labels.append(states[current])
            current = 1 if np.random.random() < P_true[current, 1] else 0
        
        result = regime_transition_matrix(pd.Series(labels))
        
        # Stationary should be close to theoretical
        # pi_bull = P_bear_bull / (P_bear_bull + P_bull_bear) = 0.4 / (0.4 + 0.3) = 0.571
        stat = result["stationary_distribution"]
        assert stat["bull"] == pytest.approx(0.571, abs=0.1)

    def test_transition_matrix_matches_analytical(self):
        """Compare with analytical transition count."""
        # Transitions: A->B, B->B, B->A, A->A, A->A, A->B
        labels = pd.Series(["A", "B", "B", "A", "A", "A", "B"])
        result = regime_transition_matrix(labels)
        
        # Manual count:
        # From A: A->B (1), A->A (2), A->B (1) = 4 transitions from A
        #   Actually: positions 0,3,4,5 are A
        #   0->1: A->B, 3->4: A->A, 4->5: A->A, 5->6: A->B
        #   So from A: A->B=2, A->A=2 -> P(A|A)=0.5, P(B|A)=0.5
        # From B: positions 1,2,6 are B
        #   1->2: B->B, 2->3: B->A
        #   So from B: B->B=1, B->A=1 -> P(A|B)=0.5, P(B|B)=0.5
        tm = result["transition_matrix"]
        assert tm.loc["A", "A"] == pytest.approx(0.5)
        assert tm.loc["A", "B"] == pytest.approx(0.5)
        assert tm.loc["B", "A"] == pytest.approx(0.5)
        assert tm.loc["B", "B"] == pytest.approx(0.5)

    def test_duration_calculation_accuracy(self):
        """Test duration statistics match manual calculation."""
        labels = pd.Series(["A", "A", "B", "A", "A", "A", "B", "B"])
        result = regime_transition_matrix(labels)
        
        # A durations: [2, 3], B durations: [1, 2]
        dur_a = result["duration_distribution"]["A"]
        dur_b = result["duration_distribution"]["B"]
        
        assert dur_a["count"] == 2
        assert dur_a["mean"] == 2.5
        assert dur_a["min"] == 2
        assert dur_a["max"] == 3
        
        assert dur_b["count"] == 2
        assert dur_b["mean"] == 1.5

    def test_stationary_distribution_single_state(self):
        """Test stationary distribution for single-state chain."""
        labels = pd.Series(["A", "A", "A"])
        result = regime_transition_matrix(labels)
        
        # Single state should have stationary = [1.0]
        assert result["stationary_distribution"]["A"] == pytest.approx(1.0)


# ============================================================
# Performance Tests
# ============================================================
class TestRegimePerformance:
    def test_label_align_ffill_performance(self):
        """ffill mode should handle n=10000 in < 1 second."""
        import time
        
        regime_idx = pd.date_range("2020-01-01", periods=100, freq="D")
        labels = pd.Series(["bull", "bear"] * 50, index=regime_idx)
        target = pd.date_range("2020-01-01", "2047-05-01", freq="h")  # ~10000 points
        
        start = time.perf_counter()
        result = regime_label_align(target, labels, method="ffill")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"ffill took {elapsed:.2f}s, should be < 1s"
        assert len(result) == len(target)

    def test_filter_data_soft_performance(self):
        """Soft mode should handle n=10000 in < 100ms."""
        import time
        
        n = 10000
        idx = pd.date_range("2020-01-01", periods=n, freq="min")
        data = pd.DataFrame({"val": np.random.randn(n)}, index=idx)
        
        # Create probability dicts
        probs = np.random.dirichlet([1, 1], size=n)
        labels = pd.Series(
            [{"bull": p[0], "bear": p[1]} for p in probs],
            index=idx
        )
        
        start = time.perf_counter()
        result = regime_filter_data(data, labels, "bull", mode="soft")
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"soft mode took {elapsed*1000:.1f}ms, should be < 100ms"


class TestRegimeLabelAlignEdgeCases:
    def test_label_align_list_input(self):
        """Test that list input is converted to DatetimeIndex."""
        regime_idx = pd.DatetimeIndex(["2024-01-01"])
        labels = pd.Series(["bull"], index=regime_idx)
        # Pass list instead of DatetimeIndex
        target = ["2024-01-01", "2024-01-02"]
        result = regime_label_align(target, labels, method="asof")
        assert len(result) == 2
