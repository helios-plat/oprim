"""Shared Black-Scholes helpers."""


def _d1_d2(s, k, t, r, sigma, q=0.0):
    """Compute d1, d2 for Black-Scholes formula."""
    import numpy as np

    if t <= 0 or sigma <= 0:
        return None, None
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def _bs_price_from_d1d2(s, k, t, r, sigma, q, d1, d2, option_type):
    """Compute BS price given d1, d2."""
    import numpy as np
    from scipy.stats import norm

    call = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    if option_type == "call":
        return call
    else:  # put
        return call - s * np.exp(-q * t) + k * np.exp(-r * t)  # put-call parity
