"""Tests for oprim.information module."""

import numpy as np
import pytest

from oprim.information import ordinal_pattern, phase_randomize, shannon_entropy


class TestShannonEntropy:
    def test_uniform_distribution(self):
        """H = log2(n) for uniform distribution over n symbols."""
        x = np.array([0, 1, 2, 3])  # 4 equally likely symbols
        h = shannon_entropy(x)
        assert h == pytest.approx(np.log2(4), rel=1e-9)

    def test_deterministic(self):
        """Single symbol -> H = 0."""
        x = np.array([5, 5, 5, 5, 5])
        assert shannon_entropy(x) == pytest.approx(0.0, abs=1e-12)

    def test_empty_returns_zero(self):
        """Empty array -> H = 0."""
        assert shannon_entropy(np.array([])) == 0.0

    def test_base_e(self):
        """base=np.e computes entropy in nats."""
        x = np.array([0, 1, 0, 1])  # uniform over 2 symbols
        h = shannon_entropy(x, base=np.e)
        assert h == pytest.approx(np.log(2), rel=1e-9)

    def test_base_2_vs_base_e(self):
        """H_bits = H_nats / log(2)."""
        x = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        h_bits = shannon_entropy(x, base=2.0)
        h_nats = shannon_entropy(x, base=np.e)
        assert h_bits == pytest.approx(h_nats / np.log(2), rel=1e-9)

    def test_two_symbols_unequal(self):
        """H < 1 bit for non-uniform binary distribution."""
        x = np.array([0, 0, 0, 1])  # p0=0.75, p1=0.25
        h = shannon_entropy(x)
        expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
        assert h == pytest.approx(expected, rel=1e-9)

    def test_single_element(self):
        """One element -> H = 0."""
        assert shannon_entropy(np.array([42])) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.academic_reference
    def test_shannon_entropy_shannon_1948(self):
        """Shannon (1948): H([0.5, 0.5]) = 1 bit."""
        x = np.array([0, 1])  # one each -> p = 0.5, 0.5
        h = shannon_entropy(x, base=2.0)
        assert h == pytest.approx(1.0, rel=1e-9)


class TestOrdinalPattern:
    def test_shape(self):
        """d=3, len=10 -> len - d + 1 = 8 patterns."""
        x = np.arange(10, dtype=float)
        patterns = ordinal_pattern(x, d=3)
        assert len(patterns) == 8

    def test_range(self):
        """Pattern indices must be in [0, d! - 1]."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50)
        d = 3
        patterns = ordinal_pattern(x, d=d)
        from math import factorial
        assert patterns.min() >= 0
        assert patterns.max() <= factorial(d) - 1

    def test_too_short_raises(self):
        """len < d must raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            ordinal_pattern(np.array([1.0, 2.0]), d=5)

    def test_sorted_ascending(self):
        """Strictly ascending sequence -> pattern index 0 every window."""
        x = np.arange(1, 11, dtype=float)
        patterns = ordinal_pattern(x, d=3)
        assert np.all(patterns == 0)

    def test_d2_binary(self):
        """d=2: only 2 possible patterns (0=ascending, 1=descending)."""
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        patterns = ordinal_pattern(x, d=2)
        assert len(patterns) == 4
        assert set(patterns).issubset({0, 1})

    def test_integer_output_dtype(self):
        """Output dtype should be integer."""
        patterns = ordinal_pattern(np.arange(6.0), d=3)
        assert patterns.dtype.kind == "i"


class TestPhaseRandomize:
    def test_same_length(self):
        """Output length == input length."""
        x = np.random.default_rng(0).normal(size=64)
        out = phase_randomize(x, rng=np.random.default_rng(1))
        assert len(out) == len(x)

    def test_preserves_power_spectrum(self):
        """np.abs(rfft(original)) == np.abs(rfft(surrogate))."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=128)
        surrogate = phase_randomize(x, rng=np.random.default_rng(7))
        orig_amp = np.abs(np.fft.rfft(x))
        surr_amp = np.abs(np.fft.rfft(surrogate))
        np.testing.assert_allclose(orig_amp, surr_amp, rtol=1e-9)

    def test_different_phases(self):
        """Two calls with different rng produce different series."""
        x = np.random.default_rng(0).normal(size=64)
        s1 = phase_randomize(x, rng=np.random.default_rng(1))
        s2 = phase_randomize(x, rng=np.random.default_rng(2))
        assert not np.allclose(s1, s2)

    def test_default_rng(self):
        """No rng argument still works (uses default_rng internally)."""
        x = np.random.default_rng(0).normal(size=32)
        out = phase_randomize(x)
        assert len(out) == len(x)

    def test_odd_length(self):
        """Odd-length input: Nyquist not pinned -> phases different."""
        x = np.random.default_rng(0).normal(size=33)
        out = phase_randomize(x, rng=np.random.default_rng(5))
        assert len(out) == 33
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(out)), rtol=1e-9
        )
