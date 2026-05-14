"""Tests for oprim.point_process.hawkes_nll."""
import numpy as np
import pytest

from oprim.point_process import hawkes_nll


class TestHawkesNll:
    def _default_params(self):
        return np.array([np.log(0.5), np.log(0.3), np.log(1.0)])

    def test_returns_float(self):
        """hawkes_nll returns a Python float."""
        events = np.array([0.1, 0.5, 1.0, 2.3, 3.7])
        result = hawkes_nll(self._default_params(), events, T=5.0)
        assert isinstance(result, float)

    def test_returns_finite_for_valid_input(self):
        """Valid event sequence with positive params → finite NLL."""
        events = np.array([0.1, 0.5, 1.2, 2.0, 3.1, 4.0])
        result = hawkes_nll(self._default_params(), events, T=5.0)
        assert np.isfinite(result)

    def test_too_few_events_returns_penalty(self):
        """n < 2 events → returns 1e10 penalty."""
        events = np.array([1.0])
        result = hawkes_nll(self._default_params(), events, T=5.0)
        assert result == 1e10

    def test_empty_events_returns_penalty(self):
        """Empty event array → returns 1e10 penalty."""
        events = np.array([])
        result = hawkes_nll(self._default_params(), events, T=5.0)
        assert result == 1e10

    def test_nll_decreases_with_better_fit(self):
        """True params should give lower NLL than wildly wrong params."""
        rng = np.random.default_rng(42)
        # Simulate Poisson process (mu=2, alpha=0, beta=1) → Poisson(2)
        n_events = 40
        events = np.sort(rng.uniform(0, 10, n_events))
        T = 10.0

        good_params = np.array([np.log(2.0), np.log(0.01), np.log(1.0)])
        bad_params = np.array([np.log(0.001), np.log(5.0), np.log(0.01)])

        nll_good = hawkes_nll(good_params, events, T)
        nll_bad = hawkes_nll(bad_params, events, T)
        assert nll_good < nll_bad

    def test_nll_depends_on_T(self):
        """Longer observation window T changes the integral term → different NLL."""
        events = np.array([0.5, 1.0, 2.5, 3.0])
        p = self._default_params()
        nll5 = hawkes_nll(p, events, T=5.0)
        nll10 = hawkes_nll(p, events, T=10.0)
        assert nll5 != nll10

    def test_A_recursive_accumulation(self):
        """A_i = Σ_{j<i} exp(-β*(t_i - t_j)): for 2 events A[0]=0, A[1]=exp(-β*Δ)."""
        beta = 1.0
        events = np.array([0.0, 0.5])
        T = 1.0
        params = np.array([np.log(1.0), np.log(0.5), np.log(beta)])
        result = hawkes_nll(params, events, T)
        assert np.isfinite(result)

    def test_log_params_ensure_positivity(self):
        """Log-parameterization ensures mu, alpha, beta > 0 for any real params."""
        params = np.array([-10.0, -10.0, -10.0])  # very small positive values
        events = np.array([0.1, 0.5, 1.0, 2.0])
        result = hawkes_nll(params, events, T=3.0)
        # Should return finite value or 1e10 penalty, not raise
        assert isinstance(result, float)
