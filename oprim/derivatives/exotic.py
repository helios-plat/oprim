"""Exotic option pricing: barrier and lookback options.

References
----------
Reiner, E. & Rubinstein, M. (1991). Breaking Down the Barriers.
    Risk Magazine, 4(8), 28-35.
Goldman, M.B., Sosin, H.B. & Gatto, M.A. (1979). Path Dependent Options:
    Buy at the Low, Sell at the High. Journal of Finance, 34(5), 1111-1127.
Haug, E.G. (2007). The Complete Guide to Option Pricing Formulas (2nd ed.).
    McGraw-Hill.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.stats import norm


def _bs_vanilla(
    s: float, k: float, t: float, r: float, sigma: float, q: float, option_type: str
) -> float:
    """Black-Scholes vanilla price for parity relations."""
    if t <= 0:
        if option_type == "call":
            return max(s - k, 0.0)
        return max(k - s, 0.0)
    if sigma <= 0:
        if option_type == "call":
            return max(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
        return max(k * np.exp(-r * t) - s * np.exp(-q * t), 0.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    if option_type == "call":
        return float(call)
    return float(call - s * np.exp(-q * t) + k * np.exp(-r * t))


def _barrier_cf(
    s: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    barrier_type: str,
    option_type: str,
    rebate: float,
) -> float:
    """Closed-form barrier option price using Reiner-Rubinstein (1991) formulas."""
    if t <= 0:
        intrinsic = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
        if "out" in barrier_type:
            if "down" in barrier_type and s <= h:
                return rebate
            if "up" in barrier_type and s >= h:
                return rebate
            return intrinsic
        else:  # in
            if "down" in barrier_type and s <= h:
                return intrinsic
            if "up" in barrier_type and s >= h:
                return intrinsic
            return rebate

    if sigma <= 0:
        # Zero vol: deterministic path
        return _barrier_zero_vol(s, k, h, t, r, q, barrier_type, option_type, rebate)

    sqrt_t = np.sqrt(t)
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    lambda_val = np.sqrt(mu**2 + 2.0 * r / sigma**2)

    x1 = np.log(s / k) / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t
    x2 = np.log(s / h) / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t
    y1 = np.log(h**2 / (s * k)) / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t
    y2 = np.log(h / s) / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t
    z = np.log(h / s) / (sigma * sqrt_t) + lambda_val * sigma * sqrt_t

    phi = 1.0 if option_type == "call" else -1.0
    eta: float  # direction multiplier: +1 for down barriers, -1 for up barriers

    cdf_fn = norm.cdf
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)

    # Component functions from Reiner-Rubinstein (1991)
    def a(phi_: float, x_: float) -> float:
        return phi_ * (
            s * disc_q * cdf_fn(phi_ * x_) - k * disc_r * cdf_fn(phi_ * (x_ - sigma * sqrt_t))
        )

    def b(phi_: float, x_: float) -> float:
        return phi_ * (
            s * disc_q * cdf_fn(phi_ * x_) - k * disc_r * cdf_fn(phi_ * (x_ - sigma * sqrt_t))
        )

    def c(phi_: float, eta_: float, y_: float) -> float:
        return phi_ * (
            s * disc_q * (h / s) ** (2.0 * (mu + 1.0)) * cdf_fn(eta_ * y_)
            - k * disc_r * (h / s) ** (2.0 * mu) * cdf_fn(eta_ * (y_ - sigma * sqrt_t))
        )

    def d(phi_: float, eta_: float, y_: float) -> float:
        return phi_ * (
            s * disc_q * (h / s) ** (2.0 * (mu + 1.0)) * cdf_fn(eta_ * y_)
            - k * disc_r * (h / s) ** (2.0 * mu) * cdf_fn(eta_ * (y_ - sigma * sqrt_t))
        )

    def e_rebate(eta_: float) -> float:
        return (
            rebate
            * disc_r
            * (
                cdf_fn(eta_ * (x2 - sigma * sqrt_t))
                - (h / s) ** (2.0 * mu) * cdf_fn(eta_ * (y2 - sigma * sqrt_t))
            )
        )

    def f_rebate(eta_: float) -> float:
        return rebate * (
            (h / s) ** (mu + lambda_val) * cdf_fn(eta_ * z)
            + (h / s) ** (mu - lambda_val) * cdf_fn(eta_ * (z - 2.0 * lambda_val * sigma * sqrt_t))
        )

    # Use parity: out + in = vanilla; derive all 8 cases
    vanilla = _bs_vanilla(s, k, t, r, sigma, q, option_type)

    if barrier_type == "down_and_out":
        if s <= h:
            return rebate  # already knocked out
        eta = 1.0
        if option_type == "call":
            if k >= h:
                price = a(phi, x1) - c(phi, eta, y1) + e_rebate(eta)
            else:
                price = b(phi, x2) - c(phi, eta, y1) + d(phi, eta, y2) + e_rebate(eta)
        else:  # put
            if k >= h:
                price = a(phi, x1) - b(phi, x2) + c(phi, eta, y1) - d(phi, eta, y2) + e_rebate(eta)
            else:
                price = e_rebate(eta)
        return float(max(price, 0.0))

    elif barrier_type == "down_and_in":
        if s <= h:
            return vanilla  # already knocked in
        out_price = barrier_option_price(
            s,
            strike=k,
            barrier=h,
            time_to_expiry=t,
            risk_free_rate=r,
            volatility=sigma,
            barrier_type="down_and_out",
            option_type=option_type,
            rebate=0.0,
            dividend_yield=q,
            method="closed_form",
        )["price"]
        _barrier_cf(s, k, h, t, r, sigma, q, "down_and_out", option_type, rebate)
        # in + out = vanilla + rebate (the rebate is paid by the out)
        return float(vanilla - out_price + rebate * np.exp(-r * t))

    elif barrier_type == "up_and_out":
        if s >= h:
            return rebate  # already knocked out
        eta = -1.0
        if option_type == "call":
            if k >= h:
                price = e_rebate(eta)
            else:
                price = a(phi, x1) - b(phi, x2) + c(phi, eta, y1) - d(phi, eta, y2) + e_rebate(eta)
        else:  # put
            if k >= h:
                price = a(phi, x1) - c(phi, eta, y1) + e_rebate(eta)
            else:
                price = b(phi, x2) - d(phi, eta, y2) + e_rebate(eta)
        return float(max(price, 0.0))

    elif barrier_type == "up_and_in":
        if s >= h:
            return vanilla
        out_price = barrier_option_price(
            s,
            strike=k,
            barrier=h,
            time_to_expiry=t,
            risk_free_rate=r,
            volatility=sigma,
            barrier_type="up_and_out",
            option_type=option_type,
            rebate=0.0,
            dividend_yield=q,
            method="closed_form",
        )["price"]
        return float(vanilla - out_price + rebate * np.exp(-r * t))

    return float("nan")  # unreachable


def _barrier_zero_vol(s, k, h, t, r, q, barrier_type, option_type, rebate):
    """Barrier price under zero volatility: deterministic path."""
    # Forward price
    fwd = s * np.exp((r - q) * t)
    disc = np.exp(-r * t)
    # Check if barrier is breached deterministically
    breached = fwd <= h or s <= h if "down" in barrier_type else fwd >= h or s >= h

    if "out" in barrier_type:
        if breached:
            return float(rebate * disc)
        if option_type == "call":
            return float(max(fwd - k, 0.0) * disc)
        return float(max(k - fwd, 0.0) * disc)
    else:  # in
        if breached:
            if option_type == "call":
                return float(max(fwd - k, 0.0) * disc)
            return float(max(k - fwd, 0.0) * disc)
        return float(rebate * disc)


def barrier_option_price(
    spot: float, *,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    barrier_type: Literal["up_and_in", "up_and_out", "down_and_in", "down_and_out"] = "up_and_out",
    option_type: Literal["call", "put"] = "call",
    rebate: float = 0.0,
    dividend_yield: float = 0.0,
    method: Literal["closed_form", "monte_carlo"] = "closed_form",
    n_simulations: int | None = None,
) -> dict[str, Any]:
    """Price a barrier option.

    Parameters
    ----------
    spot : float
        Current asset price (> 0).
    strike : float
        Strike price (> 0).
    barrier : float
        Barrier level (> 0).
    time_to_expiry : float
        Time to expiry in years (>= 0).
    risk_free_rate : float
        Continuously compounded risk-free rate.
    volatility : float
        Annual volatility (>= 0).
    barrier_type : {"up_and_in", "up_and_out", "down_and_in", "down_and_out"}
        Barrier type. Default "up_and_out".
    option_type : {"call", "put"}
        Default "call".
    rebate : float
        Rebate paid if knocked out. Default 0.0.
    dividend_yield : float
        Continuous dividend yield. Default 0.0.
    method : {"closed_form", "monte_carlo"}
        Pricing method. Default "closed_form".
    n_simulations : int or None
        Number of MC simulations (required for monte_carlo method).

    Returns
    -------
    dict with keys:
        price, method, barrier_type, and (MC only) barrier_breached_pct.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Reiner & Rubinstein (1991). Risk Magazine, 4(8), 28-35.
    Haug (2007). The Complete Guide to Option Pricing Formulas.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if barrier <= 0:
        raise ValueError(f"barrier must be > 0, got {barrier}")
    if time_to_expiry < 0:
        raise ValueError(f"time_to_expiry must be >= 0, got {time_to_expiry}")
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if barrier_type not in ("up_and_in", "up_and_out", "down_and_in", "down_and_out"):
        raise ValueError(f"Invalid barrier_type: {barrier_type!r}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if method not in ("closed_form", "monte_carlo"):
        raise ValueError(f"method must be 'closed_form' or 'monte_carlo', got {method!r}")

    s_val, k_val, h_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        barrier,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    if method == "closed_form":
        price = _barrier_cf(
            s_val, k_val, h_val, t_val, r_val, sigma_val, q_val, barrier_type, option_type, rebate
        )
        return {
            "price": float(max(price, 0.0)),
            "method": "closed_form",
            "barrier_type": barrier_type,
        }

    # Monte Carlo
    n_sims = n_simulations if n_simulations is not None else 10000
    if n_sims < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_sims}")

    n_steps = max(int(252 * t_val), 1)
    dt = t_val / n_steps
    drift = (r_val - q_val - 0.5 * sigma_val**2) * dt
    vol_sqrt_dt = sigma_val * np.sqrt(dt)

    rng = np.random.default_rng(None)
    z_val = rng.standard_normal((n_sims, n_steps))
    log_inc = drift + vol_sqrt_dt * z_val
    log_paths = np.cumsum(log_inc, axis=1)
    paths = s_val * np.exp(log_paths)  # (n_sims, n_steps)

    # Check barrier breach
    if "down" in barrier_type:
        breached = np.any(paths <= h_val, axis=1)  # (n_sims,)
    else:
        breached = np.any(paths >= h_val, axis=1)

    st_val = paths[:, -1]
    disc = np.exp(-r_val * t_val)

    intrinsic = (
        np.maximum(st_val - k_val, 0.0)
        if option_type == "call"
        else np.maximum(k_val - st_val, 0.0)
    )

    if "out" in barrier_type:
        payoffs = np.where(breached, rebate, intrinsic) * disc
    else:  # in
        payoffs = np.where(breached, intrinsic, rebate) * disc

    price_est = float(np.mean(payoffs))
    breached_pct = float(np.mean(breached))

    return {
        "price": float(max(price_est, 0.0)),
        "method": "monte_carlo",
        "barrier_type": barrier_type,
        "barrier_breached_pct": breached_pct,
    }


# ---------------------------------------------------------------------------
# Lookback option pricing
# ---------------------------------------------------------------------------


def _lookback_cf(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    strike_type: str,
) -> float:
    """Closed-form lookback using Goldman-Sosin-Gatto (1979) / Conze-Viswanathan (1991)."""
    if t <= 0:
        if strike_type == "floating":
            # At expiry, min = max = s; payoff = 0
            return 0.0
        if option_type == "call":
            return max(s - k, 0.0)
        return max(k - s, 0.0)

    if sigma <= 0:
        # Deterministic: S_T = s * exp((r-q)*t)
        st_val = s * np.exp((r - q) * t)
        disc = np.exp(-r * t)
        if strike_type == "floating":
            if option_type == "call":
                return float(max(st_val - s, 0.0) * disc)  # max(S_T - min, 0) ~ max(ST - s, 0)
            return float(max(s - st_val, 0.0) * disc)
        if option_type == "call":
            return float(max(st_val - k, 0.0) * disc)
        return float(max(k - st_val, 0.0) * disc)

    sqrt_t = np.sqrt(t)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    b = r - q  # cost of carry

    def _phi(x: float) -> float:
        return float(norm.cdf(x))

    if strike_type == "floating":
        # Floating strike lookback
        # Call: E[e^{-rT} * (S_T - min_{0,t} S_t)]
        # Put:  E[e^{-rT} * (max_{0,t} S_t - S_T)]
        # Conze & Viswanathan (1991) formulas
        a1 = (np.log(s / s) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        # Since m_0 = s (current price is the initial minimum/maximum)
        # Use simplified form with M_0 = m_0 = s
        a1_c = (b + 0.5 * sigma**2) * sqrt_t / sigma
        a2_c = a1_c - sigma * sqrt_t

        if option_type == "call":
            # Floating call = s*e^{-qT}*N(a1) - s*e^{-rT}*N(a2) + sigma^2/(2b) terms
            if abs(b) > 1e-8:
                s * disc_q * _phi(a1_c)
                s * disc_r * _phi(a2_c)
                (sigma**2 / (2.0 * b)) * s * (
                    disc_r * _phi(a2_c) - disc_q * (b * t + 1.0) * _phi(-a1_c)
                    # Actually use correct GSG formula:
                )
                # GSG (1979) floating call formula:
                # C_float = s*e^{-qT}*N(a1) - s*e^{-rT}*(sigma^2/(2b))*N(-a1)
                #         - s*e^{-rT}*N(a2) + s*e^{-rT}*(sigma^2/(2b))*N(-a2) [not exactly right]
                # Use Haug formulas directly:
                # d1 = (ln(s/m) + (b+σ²/2)t)/(σ√t), with m=s → d1 = (b+σ²/2)√t/σ
                d1 = (b + 0.5 * sigma**2) * sqrt_t / sigma
                d2 = d1 - sigma * sqrt_t
                price = (
                    s * disc_q * _phi(d1)
                    - s * disc_r * _phi(d2)
                    + s
                    * disc_r
                    * sigma**2
                    / (2.0 * b)
                    * (
                        -_phi(-d1) * (s / s) ** (-2.0 * b / sigma**2)  # s/m = 1
                        + np.exp(0) * _phi(d2)
                    )
                )
                # Simplify with s/m = 1 → (s/m)^{-2b/σ²} = 1
                d1 = (b + 0.5 * sigma**2) * sqrt_t / sigma
                d2 = d1 - sigma * sqrt_t
                price = (
                    s * disc_q * _phi(d1)
                    - s * disc_r * _phi(d2)
                    + s * disc_r * (sigma**2 / (2.0 * b)) * (_phi(d2) - _phi(-d1))
                )
            else:
                # b ≈ 0: use limit
                d1 = 0.5 * sigma * sqrt_t
                d2 = -0.5 * sigma * sqrt_t
                price = (
                    s * _phi(d1)
                    - s * disc_r * _phi(d2)
                    + s * disc_r * sigma * sqrt_t * norm.pdf(d1)
                )
            return float(max(price, 0.0))

        else:  # put
            if abs(b) > 1e-8:
                d1 = (b + 0.5 * sigma**2) * sqrt_t / sigma
                d2 = d1 - sigma * sqrt_t
                price = (
                    s * disc_r * _phi(-d2)
                    - s * disc_q * _phi(-d1)
                    + s * disc_r * (sigma**2 / (2.0 * b)) * (_phi(d2) - _phi(d1))
                )
            else:
                d1 = 0.5 * sigma * sqrt_t
                d2 = -0.5 * sigma * sqrt_t
                price = (
                    s * disc_r * _phi(d2)
                    - s * _phi(-d1)
                    + s * disc_r * sigma * sqrt_t * norm.pdf(d1)
                )
            return float(max(price, 0.0))

    else:  # fixed strike
        # Fixed strike lookback call: E[e^{-rT} * max(max_{0,t} S_t - k, 0)]
        # Fixed strike lookback put:  E[e^{-rT} * max(k - min_{0,t} S_t, 0)]
        if option_type == "call":
            # Conze & Viswanathan (1991) fixed call:
            # Uses max M_0 = s (current price is initial running max)
            m0 = s
            if m0 > k:
                # Already in-the-money for the running max
                d1 = (np.log(m0 / k) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                a1 = (np.log(m0 / s) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                a2 = a1 - sigma * sqrt_t
                if abs(b) > 1e-8:
                    part1 = s * disc_q * _phi(a1) - k * disc_r * _phi(a2)
                    part2 = (
                        (sigma**2 / (2.0 * b))
                        * s
                        * disc_q
                        * (
                            -((m0 / s) ** (-2.0 * b / sigma**2)) * _phi(-a1)
                            + np.exp(b * t) * _phi(a1 - sigma * sqrt_t * (2.0 * b / sigma**2 + 1.0))
                        )
                    )
                    price = part1 + part2
                else:
                    price = s * disc_q * _phi(a1) - k * disc_r * _phi(a2)
            else:
                # Standard formula
                d1 = (np.log(s / k) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                if abs(b) > 1e-8:
                    price = (
                        s * disc_q * _phi(d1)
                        - k * disc_r * _phi(d2)
                        + s
                        * disc_r
                        * (sigma**2 / (2.0 * b))
                        * (
                            -((s / k) ** (-2.0 * b / sigma**2))
                            * _phi(-d1 + 2.0 * b * sqrt_t / sigma)
                            + np.exp(b * t) * _phi(d1)
                        )
                    )
                else:
                    price = (
                        s * disc_q * _phi(d1)
                        - k * disc_r * _phi(d2)
                        + s * disc_r * sigma * sqrt_t * norm.pdf(d1)
                    )
            return float(max(price, 0.0))

        else:  # put
            m0 = s  # current running minimum
            if m0 < k:
                d1 = (np.log(m0 / k) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                a1 = (np.log(s / m0) - (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                a2 = a1 + sigma * sqrt_t
                if abs(b) > 1e-8:
                    part1 = k * disc_r * _phi(-d2) - m0 * disc_q * _phi(-d1)
                    price = part1
                else:
                    price = k * disc_r * _phi(-d2) - m0 * disc_q * _phi(-d1)
            else:
                # k <= m0: fixed put, standard formula
                d1 = (np.log(s / k) + (b + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                if abs(b) > 1e-8:
                    price = (
                        k * disc_r * _phi(-d2)
                        - s * disc_q * _phi(-d1)
                        + s
                        * disc_r
                        * (sigma**2 / (2.0 * b))
                        * (
                            (s / k) ** (-2.0 * b / sigma**2) * _phi(d1 - 2.0 * b * sqrt_t / sigma)
                            - np.exp(b * t) * _phi(-d1)
                        )
                    )
                else:
                    price = (
                        k * disc_r * _phi(-d2)
                        - s * disc_q * _phi(-d1)
                        + s * disc_r * sigma * sqrt_t * norm.pdf(d1)
                    )
            return float(max(price, 0.0))


def lookback_option_price(
    spot: float, *,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: Literal["call", "put"] = "call",
    strike_type: Literal["fixed", "floating"] = "floating",
    dividend_yield: float = 0.0,
    method: Literal["closed_form", "monte_carlo"] = "closed_form",
    n_simulations: int | None = None,
) -> dict[str, Any]:
    """Price a lookback option.

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
    option_type : {"call", "put"}
        Default "call".
    strike_type : {"fixed", "floating"}
        Floating: payoff uses path min/max as effective strike.
        Fixed: payoff uses path max/min vs fixed k.
    dividend_yield : float
        Continuous dividend yield. Default 0.0.
    method : {"closed_form", "monte_carlo"}
        Pricing method. Default "closed_form".
    n_simulations : int or None
        MC simulations (for monte_carlo method).

    Returns
    -------
    dict with keys: price, method, strike_type.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Goldman, Sosin & Gatto (1979). Journal of Finance, 34(5), 1111-1127.
    Haug (2007). The Complete Guide to Option Pricing Formulas.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry < 0:
        raise ValueError(f"time_to_expiry must be >= 0, got {time_to_expiry}")
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if strike_type not in ("fixed", "floating"):
        raise ValueError(f"strike_type must be 'fixed' or 'floating', got {strike_type!r}")
    if method not in ("closed_form", "monte_carlo"):
        raise ValueError(f"method must be 'closed_form' or 'monte_carlo', got {method!r}")

    s_val, k_val, t_val, r_val, sigma_val, q_val = (
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    if method == "closed_form":
        price = _lookback_cf(s_val, k_val, t_val, r_val, sigma_val, q_val, option_type, strike_type)
        return {
            "price": float(price),
            "method": "closed_form",
            "strike_type": strike_type,
        }

    # Monte Carlo
    n_sims = n_simulations if n_simulations is not None else 10000
    if n_sims < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_sims}")

    n_steps = max(int(252 * t_val), 1)
    dt = t_val / n_steps
    drift = (r_val - q_val - 0.5 * sigma_val**2) * dt
    vol_sqrt_dt = sigma_val * np.sqrt(dt)

    rng = np.random.default_rng(None)
    z_val = rng.standard_normal((n_sims, n_steps))
    log_inc = drift + vol_sqrt_dt * z_val
    log_paths = np.cumsum(log_inc, axis=1)
    paths = s_val * np.exp(log_paths)  # (n_sims, n_steps)

    st_val = paths[:, -1]
    path_max = np.max(paths, axis=1)
    path_min = np.min(paths, axis=1)

    disc = np.exp(-r_val * t_val)

    if strike_type == "floating":
        if option_type == "call":
            payoffs = disc * np.maximum(st_val - path_min, 0.0)
        else:
            payoffs = disc * np.maximum(path_max - st_val, 0.0)
    else:  # fixed
        if option_type == "call":
            payoffs = disc * np.maximum(path_max - k_val, 0.0)
        else:
            payoffs = disc * np.maximum(k_val - path_min, 0.0)

    price_est = float(np.mean(payoffs))

    return {
        "price": float(max(price_est, 0.0)),
        "method": "monte_carlo",
        "strike_type": strike_type,
    }
