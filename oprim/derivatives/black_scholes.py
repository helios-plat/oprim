"""Black-Scholes-Merton option pricing and Greeks.

References
----------
Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
    Liabilities. Journal of Political Economy, 81(3), 637-654.
Merton, R.C. (1973). Theory of Rational Option Pricing. Bell Journal of
    Economics and Management Science, 4(1), 141-183.
Hull, J.C. (2018). Options, Futures, and Other Derivatives (10th ed.).
    Pearson Education.
Manaster, S. & Koehler, G. (1982). The Calculation of Implied Variances from
    the Black-Scholes Model. Journal of Finance, 37(1), 227-230.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq, newton
from scipy.stats import norm

from oprim.derivatives._base import _bs_price_from_d1d2, _d1_d2


def black_scholes_price(
    spot: float, *,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Compute Black-Scholes-Merton option price.

    Parameters
    ----------
    spot : float
        Current asset price (>= 0).
    strike : float
        Strike price (>= 0).
    time_to_expiry : float
        Time to expiry in years (>= 0).
    risk_free_rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Implied volatility (>= 0).
    option_type : {"call", "put"}, optional
        Option type. Default "call".
    dividend_yield : float, optional
        Continuous dividend yield. Default 0.0.

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Black & Scholes (1973)
    Merton (1973).
    """
    if spot < 0 or strike < 0 or time_to_expiry < 0 or volatility < 0:
        raise ValueError("spot, strike, time_to_expiry, volatility must all be >= 0")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    s_val, k_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    # Handle T=0 edge case: intrinsic value
    if t_val == 0:
        if option_type == "call":
            return float(max(s_val - k_val, 0.0))
        else:
            return float(max(k_val - s_val, 0.0))

    # Handle sigma=0 edge case: discounted intrinsic
    if sigma_val == 0:
        if option_type == "call":
            return float(
                max(s_val * math.exp(-q_val * t_val) - k_val * math.exp(-r_val * t_val), 0.0)
            )
        else:
            return float(
                max(k_val * math.exp(-r_val * t_val) - s_val * math.exp(-q_val * t_val), 0.0)
            )

    d1, d2 = _d1_d2(s_val, k_val, t_val, r_val, sigma_val, q_val)
    return float(
        _bs_price_from_d1d2(s_val, k_val, t_val, r_val, sigma_val, q_val, d1, d2, option_type)
    )


def black_scholes_greeks(
    spot: float, *,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Compute Black-Scholes Greeks.

    Parameters
    ----------
    spot : float
        Current asset price.
    strike : float
        Strike price.
    time_to_expiry : float
        Time to expiry in years.
    risk_free_rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Implied volatility.
    option_type : {"call", "put"}, optional
        Option type. Default "call".
    dividend_yield : float, optional
        Continuous dividend yield. Default 0.0.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.

    References
    ----------
    Hull, J.C. (2018). Options, Futures, and Other Derivatives (10th ed.).
    Ch.19.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    s_val, k_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    phi = norm.pdf  # standard normal PDF
    cdf_fn = norm.cdf  # standard normal CDF

    if t_val == 0 or sigma_val == 0:
        delta = (
            (1.0 if s_val > k_val else 0.0)
            if option_type == "call"
            else -1.0
            if s_val < k_val
            else 0.0
        )
        return {
            "delta": float(delta),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    d1, d2 = _d1_d2(s_val, k_val, t_val, r_val, sigma_val, q_val)

    exp_qt = math.exp(-q_val * t_val)
    exp_rt = math.exp(-r_val * t_val)
    sqrt_t = math.sqrt(t_val)

    gamma = float(exp_qt * phi(d1) / (s_val * sigma_val * sqrt_t))
    vega = float(s_val * exp_qt * phi(d1) * sqrt_t)

    if option_type == "call":
        delta = float(exp_qt * cdf_fn(d1))
        theta = float(
            -(s_val * exp_qt * phi(d1) * sigma_val / (2.0 * sqrt_t))
            - r_val * k_val * exp_rt * cdf_fn(d2)
            + q_val * s_val * exp_qt * cdf_fn(d1)
        )
        rho = float(k_val * t_val * exp_rt * cdf_fn(d2))
    else:  # put
        delta = float(-exp_qt * cdf_fn(-d1))
        theta = float(
            -(s_val * exp_qt * phi(d1) * sigma_val / (2.0 * sqrt_t))
            + r_val * k_val * exp_rt * cdf_fn(-d2)
            - q_val * s_val * exp_qt * cdf_fn(-d1)
        )
        rho = float(-k_val * t_val * exp_rt * cdf_fn(-d2))

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_volatility(
    market_price: float, *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    method: str = "brent",
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Compute implied volatility from a market option price.

    Does NOT import or call ``black_scholes_price`` or ``black_scholes_greeks``.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    spot : float
        Current asset price.
    strike : float
        Strike price.
    time_to_expiry : float
        Time to expiry in years.
    risk_free_rate : float
        Continuously compounded risk-free rate.
    option_type : {"call", "put"}, optional
        Option type. Default "call".
    dividend_yield : float, optional
        Continuous dividend yield. Default 0.0.
    method : {"brent", "newton"}, optional
        Root-finding method. Default "brent".
    tolerance : float, optional
        Convergence tolerance. Default 1e-6.
    max_iter : int, optional
        Maximum iterations. Default 100.

    Returns
    -------
    float
        Implied volatility, or NaN if no solution.

    Raises
    ------
    ValueError
        If method is unknown.

    References
    ----------
    Manaster, S. & Koehler, G. (1982). The Calculation of Implied Variances
    from the Black-Scholes Model. Journal of Finance, 37(1), 227-230.
    """
    if method not in ("brent", "newton"):
        raise ValueError(f"Unknown method '{method}'. Expected 'brent' or 'newton'.")

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    s_val, k_val, t_val, r_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        dividend_yield,
    )

    # T=0: no volatility solution
    if t_val <= 0:
        return float("nan")

    # Check intrinsic value lower bound
    if option_type == "call":
        intrinsic = max(s_val * math.exp(-q_val * t_val) - k_val * math.exp(-r_val * t_val), 0.0)
    else:
        intrinsic = max(k_val * math.exp(-r_val * t_val) - s_val * math.exp(-q_val * t_val), 0.0)

    if (
        market_price <= intrinsic
        and market_price < s_val * math.exp(-q_val * t_val)
        and market_price < intrinsic - 1e-8
    ):
        return float("nan")

    def _bs_price_inline(sigma):
        """Inline BSM price computation (no import of black_scholes_price)."""
        if sigma <= 0:  # pragma: no cover
            if option_type == "call":  # pragma: no cover
                return max(s_val * math.exp(-q_val * t_val) - k_val * math.exp(-r_val * t_val), 0.0)
            else:  # pragma: no cover
                return max(k_val * math.exp(-r_val * t_val) - s_val * math.exp(-q_val * t_val), 0.0)
        log_sk = math.log(s_val / k_val)
        d1 = (log_sk + (r_val - q_val + 0.5 * sigma**2) * t_val) / (sigma * math.sqrt(t_val))
        d2 = d1 - sigma * math.sqrt(t_val)
        exp_qt = math.exp(-q_val * t_val)
        exp_rt = math.exp(-r_val * t_val)
        call = s_val * exp_qt * norm.cdf(d1) - k_val * exp_rt * norm.cdf(d2)
        if option_type == "call":
            return float(call)
        else:
            return float(call - s_val * exp_qt + k_val * exp_rt)

    def _vega_inline(sigma):
        """Inline vega computation."""
        if sigma <= 0:  # pragma: no cover
            return 0.0  # pragma: no cover
        d1 = (math.log(s_val / k_val) + (r_val - q_val + 0.5 * sigma**2) * t_val) / (
            sigma * math.sqrt(t_val)
        )
        return float(s_val * math.exp(-q_val * t_val) * norm.pdf(d1) * math.sqrt(t_val))

    objective = lambda sig: _bs_price_inline(sig) - market_price  # noqa: E731

    try:
        if method == "brent":
            iv = brentq(
                objective,
                1e-6,
                10.0,
                xtol=tolerance,
                maxiter=max_iter,
            )
            return float(iv)
        else:  # newton
            # Use vega as derivative
            x0 = 0.2  # initial guess
            iv = newton(
                objective,
                x0,
                fprime=_vega_inline,
                tol=tolerance,
                maxiter=max_iter,
            )
            if iv <= 0:  # pragma: no cover
                return float("nan")  # pragma: no cover
            return float(iv)
    except (ValueError, RuntimeError):
        return float("nan")
