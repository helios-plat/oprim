"""Private helpers shared by oprim/timeseries submodules (H2 exempt from H1)."""

from __future__ import annotations

import numpy as np


def _ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS via numpy lstsq.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Dependent variable.
    x : np.ndarray, shape (n, k)
        Design matrix (should include constant column if desired).

    Returns
    -------
    coefficients : np.ndarray, shape (k,)
    residuals : np.ndarray, shape (n,)
    """
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coeffs
    return coeffs, residuals


def _adf_statistic(
    y: np.ndarray,
    lags: int = 0,
    regression: str = "c",
) -> tuple[float, np.ndarray]:
    """Compute the ADF test statistic (t-ratio for the lagged level coefficient).

    Builds the OLS regression matrix for:
        Delta_y_t = [deterministics] + rho * y_{t-1} + sum(beta_i * Delta_y_{t-i}) + eps

    Used by both adf_test and engle_granger_cointegration without H1 violation
    (this is the H2-exempt _base.py).

    Parameters
    ----------
    y : np.ndarray
        Time series (levels).
    lags : int
        Number of augmenting lags of Delta_y.
    regression : str
        Deterministic component: 'nc', 'c', 'ct', 'ctt'.

    Returns
    -------
    t_stat : float
        t-statistic for rho (unit root coefficient).
    residuals : np.ndarray
        OLS residuals.
    """
    n = len(y)
    dy = np.diff(y)  # Delta_y, length n-1

    # Need at least lags+1 observations of dy
    # Regression uses obs from index lags onwards in dy (i.e., dy[lags:])
    t_val = len(dy) - lags  # effective sample size
    if t_val <= 0:  # pragma: no cover
        return np.nan, np.array([])

    # Dependent variable: dy[lags:]
    endog = dy[lags:]

    # Lagged level: y[lags : n-1] (i.e., y_{t-1} for each delta_y)
    y_lag = y[lags : n - 1]

    # Build design matrix
    cols = [y_lag]

    # Deterministics
    if regression in ("c", "ct", "ctt"):
        cols.append(np.ones(t_val))
    if regression in ("ct", "ctt"):
        trend = np.arange(1, t_val + 1, dtype=float)
        cols.append(trend)
    if regression == "ctt":
        cols.append(np.arange(1, t_val + 1, dtype=float) ** 2)

    # Augmenting lags
    for lag_i in range(1, lags + 1):
        # Delta_y_{t-lag_i}: dy[lags-lag_i : n-1-lag_i]
        cols.append(dy[lags - lag_i : n - 1 - lag_i])

    x_mat = np.column_stack(cols)  # shape (T, k)

    # OLS
    coeffs, residuals = _ols_fit(endog, x_mat)
    rho_coeff = coeffs[0]

    # Standard error of rho
    sigma2 = np.sum(residuals**2) / (t_val - x_mat.shape[1])
    if sigma2 <= 0:
        return np.nan, residuals  # pragma: no cover

    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se_rho = np.sqrt(sigma2 * xtx_inv[0, 0])

    if se_rho == 0:
        return np.nan, residuals  # pragma: no cover

    t_stat = float(rho_coeff / se_rho)
    return t_stat, residuals
