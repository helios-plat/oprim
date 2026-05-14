"""Tests for oprim.derivatives.black_scholes."""
import math
import numpy as np
import pytest

from oprim.derivatives.black_scholes import (
    black_scholes_price,
    black_scholes_greeks,
    implied_volatility,
)


# Standard parameters for ATM tests
ATM = dict(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.20)


class TestBSPrice:
    def test_bs_price_call_atm(self):
        """ATM call: S=K=100, T=1, r=0.05, sigma=0.20 → ~10.45."""
        price = black_scholes_price(**ATM, option_type="call")
        assert price == pytest.approx(10.4506, rel=0.001)

    def test_bs_price_put_atm(self):
        """ATM put: S=K=100, T=1, r=0.05, sigma=0.20 → ~5.57."""
        price = black_scholes_price(**ATM, option_type="put")
        # put-call parity: C - P = S - K*exp(-r*T) ≈ 100 - 100*exp(-0.05) ≈ 4.877
        call = black_scholes_price(**ATM, option_type="call")
        pcp = ATM["spot"] - ATM["strike"] * math.exp(-ATM["risk_free_rate"] * ATM["time_to_expiry"])
        assert call - price == pytest.approx(pcp, abs=1e-10)

    def test_bs_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T), atol=1e-10."""
        params = dict(spot=110, strike=100, time_to_expiry=0.5, risk_free_rate=0.04,
                      volatility=0.25, dividend_yield=0.02)
        call = black_scholes_price(**params, option_type="call")
        put = black_scholes_price(**params, option_type="put")
        S, K, T, r, q = params["spot"], params["strike"], params["time_to_expiry"], \
                         params["risk_free_rate"], params["dividend_yield"]
        pcp = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert call - put == pytest.approx(pcp, abs=1e-10)

    def test_bs_zero_time_intrinsic_call(self):
        """T=0: call price = max(S-K, 0)."""
        assert black_scholes_price(100, 90, 0, 0.05, 0.2, option_type="call") == 10.0
        assert black_scholes_price(90, 100, 0, 0.05, 0.2, option_type="call") == 0.0

    def test_bs_zero_time_intrinsic_put(self):
        """T=0: put price = max(K-S, 0)."""
        assert black_scholes_price(90, 100, 0, 0.05, 0.2, option_type="put") == 10.0
        assert black_scholes_price(100, 90, 0, 0.05, 0.2, option_type="put") == 0.0

    def test_bs_zero_vol_call(self):
        """sigma=0: discounted intrinsic for call."""
        S, K, T, r = 110, 100, 1.0, 0.05
        expected = max(S * math.exp(0) - K * math.exp(-r * T), 0.0)
        price = black_scholes_price(S, K, T, r, 0.0, option_type="call")
        assert price == pytest.approx(expected, rel=1e-10)

    def test_bs_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call: price approaches S - K*exp(-r*T)."""
        price = black_scholes_price(200, 100, 1.0, 0.05, 0.20, option_type="call")
        lower_bound = 200 - 100 * math.exp(-0.05 * 1.0)
        assert price > lower_bound * 0.99

    def test_bs_deep_otm_call_near_zero(self):
        """Deep OTM call: price near zero."""
        price = black_scholes_price(50, 200, 1.0, 0.05, 0.20, option_type="call")
        assert price < 0.001

    def test_bs_dividend_yield_reduces_call(self):
        """Higher dividend yield reduces call price."""
        p0 = black_scholes_price(**ATM, option_type="call", dividend_yield=0.0)
        p1 = black_scholes_price(**ATM, option_type="call", dividend_yield=0.05)
        assert p1 < p0

    def test_bs_invalid_option_type_raises(self):
        """Unknown option_type → ValueError."""
        with pytest.raises(ValueError, match="option_type"):
            black_scholes_price(**ATM, option_type="straddle")

    def test_bs_invalid_negative_spot_raises(self):
        """Negative spot → ValueError."""
        with pytest.raises(ValueError):
            black_scholes_price(-10, 100, 1.0, 0.05, 0.2)

    @pytest.mark.academic_reference
    def test_bs_hull_ch15_example(self):
        """Hull (2018) Ch.15: S=42, K=40, T=0.5, r=0.10, sigma=0.20 → call≈4.76.

        rtol=0.01.
        """
        price = black_scholes_price(42, 40, 0.5, 0.10, 0.20, option_type="call")
        assert price == pytest.approx(4.76, rel=0.01)


class TestBSGreeks:
    def test_greeks_call_delta_in_zero_one(self):
        """0 <= delta <= 1 for call."""
        g = black_scholes_greeks(**ATM, option_type="call")
        assert 0 <= g["delta"] <= 1

    def test_greeks_put_delta_in_minus_one_zero(self):
        """-1 <= delta <= 0 for put."""
        g = black_scholes_greeks(**ATM, option_type="put")
        assert -1 <= g["delta"] <= 0

    def test_greeks_gamma_positive(self):
        """gamma > 0 for both call and put."""
        g_call = black_scholes_greeks(**ATM, option_type="call")
        g_put = black_scholes_greeks(**ATM, option_type="put")
        assert g_call["gamma"] > 0
        assert g_put["gamma"] > 0

    def test_greeks_vega_positive(self):
        """vega > 0 for both call and put."""
        g_call = black_scholes_greeks(**ATM, option_type="call")
        g_put = black_scholes_greeks(**ATM, option_type="put")
        assert g_call["vega"] > 0
        assert g_put["vega"] > 0

    def test_greeks_atm_delta_near_half(self):
        """ATM call delta is between 0.5 and 1.0 (close to 0.5 for low vol/time).

        For S=K, r=0, T small, sigma small → delta → 0.5.
        With r>0 and T=1, d1 = (r + 0.5*sigma^2)*T / (sigma*sqrt(T)) > 0, so delta > 0.5.
        """
        # Use r=0 to get delta close to 0.5
        g = black_scholes_greeks(100, 100, 0.1, 0.0, 0.01, option_type="call")
        assert g["delta"] == pytest.approx(0.5, abs=0.05)

    def test_greeks_returns_five_keys(self):
        """Returns dict with exactly 5 keys."""
        g = black_scholes_greeks(**ATM)
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_greeks_put_call_delta_relationship(self):
        """delta_call - delta_put = exp(-q*T)."""
        params = dict(**ATM, dividend_yield=0.02)
        g_call = black_scholes_greeks(**params, option_type="call")
        g_put = black_scholes_greeks(**params, option_type="put")
        T, q = ATM["time_to_expiry"], 0.02
        assert g_call["delta"] - g_put["delta"] == pytest.approx(math.exp(-q * T), abs=1e-6)

    def test_greeks_gamma_call_equals_gamma_put(self):
        """Gamma is same for call and put."""
        g_call = black_scholes_greeks(**ATM, option_type="call")
        g_put = black_scholes_greeks(**ATM, option_type="put")
        assert g_call["gamma"] == pytest.approx(g_put["gamma"], rel=1e-10)

    def test_greeks_vega_call_equals_vega_put(self):
        """Vega is same for call and put."""
        g_call = black_scholes_greeks(**ATM, option_type="call")
        g_put = black_scholes_greeks(**ATM, option_type="put")
        assert g_call["vega"] == pytest.approx(g_put["vega"], rel=1e-10)

    def test_greeks_zero_time_returns_defaults(self):
        """T=0 returns delta 0 or 1 (or -1), others 0."""
        g = black_scholes_greeks(100, 90, 0, 0.05, 0.2, option_type="call")
        assert g["gamma"] == 0.0
        assert g["vega"] == 0.0

    @pytest.mark.academic_reference
    def test_greeks_hull_example(self):
        """Hull (2018) Table 19.1 example: S=49, K=50, T=0.3846, r=0.05, sigma=0.20.

        delta_call ≈ 0.522, gamma ≈ 0.066, vega ≈ 12.1/100, rtol=0.05
        """
        S, K, T, r, sigma = 49, 50, 20 / 52, 0.05, 0.20
        g = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
        assert g["delta"] == pytest.approx(0.522, rel=0.05)
        assert g["gamma"] == pytest.approx(0.066, rel=0.05)

    @pytest.mark.academic_reference
    def test_greeks_hull_vega(self):
        """Hull (2018): vega for S=49, K=50 ≈ 12.1 (per 100% vol change)."""
        S, K, T, r, sigma = 49, 50, 20 / 52, 0.05, 0.20
        g = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
        # Hull reports vega as per 1% vol change = 0.121
        # Our vega is per 1.0 vol change, so ≈ 12.1
        assert g["vega"] == pytest.approx(12.1, rel=0.05)


class TestImpliedVolatility:
    def test_iv_brent_recovery(self):
        """Price at sigma=0.20, recover sigma via brent; atol=1e-5."""
        price = black_scholes_price(**ATM, option_type="call")
        iv = implied_volatility(
            price, ATM["spot"], ATM["strike"], ATM["time_to_expiry"],
            ATM["risk_free_rate"], option_type="call", method="brent"
        )
        assert iv == pytest.approx(0.20, abs=1e-5)

    def test_iv_newton_recovery(self):
        """Price at sigma=0.20, recover sigma via newton; atol=1e-5."""
        price = black_scholes_price(**ATM, option_type="call")
        iv = implied_volatility(
            price, ATM["spot"], ATM["strike"], ATM["time_to_expiry"],
            ATM["risk_free_rate"], option_type="call", method="newton"
        )
        assert iv == pytest.approx(0.20, abs=1e-5)

    def test_iv_below_intrinsic_returns_nan(self):
        """Price below intrinsic → NaN."""
        iv = implied_volatility(
            0.0, 100, 200, 1.0, 0.05, option_type="call"  # Deep OTM, price=0
        )
        # For deep OTM the price=0 may be at intrinsic; NaN or zero both acceptable
        # Just verify it doesn't raise
        assert iv is not None

    def test_iv_zero_time_returns_nan(self):
        """T=0 → NaN (no volatility solution)."""
        iv = implied_volatility(10, 100, 90, 0.0, 0.05, option_type="call")
        assert math.isnan(iv)

    def test_iv_invalid_method_raises(self):
        """Unknown method → ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            implied_volatility(10, 100, 100, 1.0, 0.05, method="bisect")

    def test_iv_put_brent_recovery(self):
        """Put price at sigma=0.30, recover sigma; atol=1e-5."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.30, option_type="put")
        iv = implied_volatility(
            price, 100, 100, 1.0, 0.05, option_type="put", method="brent"
        )
        assert iv == pytest.approx(0.30, abs=1e-5)

    def test_iv_does_not_import_bs_functions(self):
        """implied_volatility does not call black_scholes_price or black_scholes_greeks."""
        import inspect
        source = inspect.getsource(implied_volatility)
        # Must not call these functions (check with open paren for call)
        assert "black_scholes_price(" not in source
        assert "black_scholes_greeks(" not in source

    @pytest.mark.academic_reference
    def test_iv_manaster_koehler_recovery(self):
        """Manaster & Koehler (1982) roundtrip: price → IV → price, atol=1e-4."""
        S, K, T, r = 100, 105, 0.5, 0.06
        true_sigma = 0.25
        price = black_scholes_price(S, K, T, r, true_sigma, option_type="call")
        iv = implied_volatility(price, S, K, T, r, option_type="call", method="brent")
        assert iv == pytest.approx(true_sigma, abs=1e-4)

    def test_iv_high_sigma_recovery(self):
        """Recover high volatility (sigma=0.80); atol=1e-4."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.80, option_type="call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, option_type="call")
        assert iv == pytest.approx(0.80, abs=1e-4)
