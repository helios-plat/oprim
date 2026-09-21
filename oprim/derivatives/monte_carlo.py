"""Monte Carlo option pricing: European and Asian options.

References
----------
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
    Springer, New York.
Hull, J.C. (2018). Options, Futures, and Other Derivatives (10th ed.).
    Pearson Education.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def _bs_call_price(s: float, k: float, t: float, r: float, sigma: float, q: float) -> float:
    """Closed-form Black-Scholes call price for control variate."""
    from scipy.stats import norm

    if t <= 0 or sigma <= 0:
        return max(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))


def mc_european_price(
    spot: float, *,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    n_simulations: int = 10000,
    option_type: Literal["call", "put"] = "call",
    dividend_yield: float = 0.0,
    seed: int | None = None,
    antithetic: bool = True,
    control_variate: bool = False,
) -> dict[str, Any]:
    """Price a European option via Monte Carlo simulation.

    Parameters
    ----------
    spot : float
        Current asset price (> 0).
    strike : float
        Strike price (> 0).
    time_to_expiry : float
        Time to expiry in years (>= 0).
    risk_free_rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annual volatility (>= 0).
    n_simulations : int
        Number of simulated paths. Default 10000.
    option_type : {"call", "put"}
        Default "call".
    dividend_yield : float
        Continuous dividend yield. Default 0.0.
    seed : int or None
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction. Default True.
    control_variate : bool
        Use Black-Scholes as control variate. Default False.

    Returns
    -------
    dict with keys:
        price, standard_error, 95_confidence_interval, n_simulations_used, method.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Glasserman (2004). Monte Carlo Methods in Financial Engineering.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry < 0:
        raise ValueError(f"time_to_expiry must be >= 0, got {time_to_expiry}")
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    s_val, k_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    # Edge case: T=0
    if t_val == 0:
        price = (
            float(max(s_val - k_val, 0.0))
            if option_type == "call"
            else float(max(k_val - s_val, 0.0))
        )
        return {
            "price": price,
            "standard_error": 0.0,
            "95_confidence_interval": (price, price),
            "n_simulations_used": 0,
            "method": "analytic_at_expiry",
        }

    rng = np.random.default_rng(seed)

    drift = (r_val - q_val - 0.5 * sigma_val**2) * t_val
    vol_sqrt_t = sigma_val * np.sqrt(t_val)

    method_parts = []

    if antithetic:
        half_n = n_simulations // 2 if n_simulations > 1 else 1
        z_val = rng.standard_normal(half_n)
        z_full = np.concatenate([z_val, -z_val])
        method_parts.append("antithetic")
    else:
        z_full = rng.standard_normal(n_simulations)

    n_used = len(z_full)

    if sigma_val == 0:
        st_val = s_val * np.exp(drift * np.ones(n_used))
    else:
        st_val = s_val * np.exp(drift + vol_sqrt_t * z_full)

    # Discounted payoff
    disc = np.exp(-r_val * t_val)
    if option_type == "call":
        payoffs = disc * np.maximum(st_val - k_val, 0.0)
    else:
        payoffs = disc * np.maximum(k_val - st_val, 0.0)

    if control_variate and sigma_val > 0:
        # Control variate: use log(ST/S) as normal control
        # E[ST] = S * exp((r-q)*T)
        st_mean_analytic = s_val * np.exp((r_val - q_val) * t_val)
        beta = -np.cov(payoffs, st_val)[0, 1] / np.var(st_val)
        payoffs_cv = payoffs + beta * (st_val - st_mean_analytic)
        payoffs = payoffs_cv
        method_parts.append("control_variate")

    price_est = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(n_used))
    ci_low = price_est - 1.96 * se
    ci_high = price_est + 1.96 * se

    method_str = "monte_carlo[" + ",".join(method_parts) + "]" if method_parts else "monte_carlo"

    return {
        "price": price_est,
        "standard_error": se,
        "95_confidence_interval": (ci_low, ci_high),
        "n_simulations_used": n_used,
        "method": method_str,
    }


def mc_asian_price(
    spot: float, *,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    n_simulations: int = 10000,
    n_averaging_dates: int = 252,
    option_type: Literal["call", "put"] = "call",
    averaging: Literal["arithmetic", "geometric"] = "arithmetic",
    strike_type: Literal["fixed", "floating"] = "fixed",
    dividend_yield: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Price an Asian option via Monte Carlo simulation.

    Parameters
    ----------
    spot : float
        Current asset price (> 0).
    strike : float
        Strike price for fixed-strike options (> 0).
    time_to_expiry : float
        Time to expiry in years (>= 0).
    risk_free_rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annual volatility (>= 0).
    n_simulations : int
        Number of simulated paths. Default 10000.
    n_averaging_dates : int
        Number of averaging dates. Default 252.
    option_type : {"call", "put"}
        Default "call".
    averaging : {"arithmetic", "geometric"}
        Averaging method. Default "arithmetic".
    strike_type : {"fixed", "floating"}
        Fixed: payoff based on avg vs K. Floating: payoff based on S_T vs avg.
    dividend_yield : float
        Continuous dividend yield. Default 0.0.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        price, standard_error, 95_confidence_interval, averaging.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Glasserman (2004). Monte Carlo Methods in Financial Engineering. Ch. 4.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry < 0:
        raise ValueError(f"time_to_expiry must be >= 0, got {time_to_expiry}")
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
    if n_averaging_dates < 1:
        raise ValueError(f"n_averaging_dates must be >= 1, got {n_averaging_dates}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if averaging not in ("arithmetic", "geometric"):
        raise ValueError(f"averaging must be 'arithmetic' or 'geometric', got {averaging!r}")
    if strike_type not in ("fixed", "floating"):
        raise ValueError(f"strike_type must be 'fixed' or 'floating', got {strike_type!r}")

    s_val, k_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    if t_val == 0:
        price = (
            float(max(s_val - k_val, 0.0))
            if option_type == "call"
            else float(max(k_val - s_val, 0.0))
        )
        return {
            "price": price,
            "standard_error": 0.0,
            "95_confidence_interval": (price, price),
            "averaging": averaging,
        }

    rng = np.random.default_rng(seed)
    dt = t_val / n_averaging_dates
    drift = (r_val - q_val - 0.5 * sigma_val**2) * dt
    vol_sqrt_dt = sigma_val * np.sqrt(dt)

    # Simulate paths: shape (n_simulations, n_averaging_dates)
    z_val = rng.standard_normal((n_simulations, n_averaging_dates))
    # Log increments
    log_increments = drift + vol_sqrt_dt * z_val
    # Cumulative log paths → asset prices at each date
    log_paths = np.cumsum(log_increments, axis=1)
    paths = s_val * np.exp(log_paths)  # shape (n_sims, n_dates)

    # Terminal price
    st_val = paths[:, -1]

    # Compute average
    if averaging == "arithmetic":
        avg = np.mean(paths, axis=1)
    else:  # geometric
        avg = np.exp(np.mean(np.log(paths), axis=1))

    # Payoff
    disc = np.exp(-r_val * t_val)
    if strike_type == "fixed":
        if option_type == "call":
            payoffs = disc * np.maximum(avg - k_val, 0.0)
        else:
            payoffs = disc * np.maximum(k_val - avg, 0.0)
    else:  # floating
        if option_type == "call":
            payoffs = disc * np.maximum(st_val - avg, 0.0)
        else:
            payoffs = disc * np.maximum(avg - st_val, 0.0)

    price_est = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(n_simulations))
    ci_low = price_est - 1.96 * se
    ci_high = price_est + 1.96 * se

    return {
        "price": price_est,
        "standard_error": se,
        "95_confidence_interval": (ci_low, ci_high),
        "averaging": averaging,
    }
