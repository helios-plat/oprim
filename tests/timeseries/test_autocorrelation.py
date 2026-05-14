"""Tests for oprim.timeseries.autocorrelation: ljung_box_test, durbin_watson."""
import numpy as np
import pytest

from oprim.timeseries.autocorrelation import durbin_watson, ljung_box_test


def _white_noise(n=200, seed=42):
    return np.random.default_rng(seed).standard_normal(n)


def _ar1_residuals(phi=0.7, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


# ---------- Ljung-Box tests ----------

def test_lb_white_noise_high_pvalue():
    e = _white_noise(200)
    r = ljung_box_test(e, lags=10)
    # White noise should NOT reject H0
    assert r["p_value"] > 0.05


def test_lb_autocorrelated_low_pvalue():
    e = _ar1_residuals(phi=0.8, n=300)
    r = ljung_box_test(e, lags=10)
    # Strong autocorrelation: reject H0
    assert r["p_value"] < 0.05


def test_lb_returns_expected_keys():
    e = _white_noise(100)
    r = ljung_box_test(e, lags=5)
    assert "statistic" in r
    assert "p_value" in r
    assert "lags_tested" in r
    assert "n_obs" in r


def test_lb_list_of_lags():
    e = _white_noise(200)
    r = ljung_box_test(e, lags=[5, 10, 20])
    assert len(r["statistics"]) == 3
    assert len(r["p_values"]) == 3


def test_lb_boxpierce():
    e = _white_noise(200)
    r_lb = ljung_box_test(e, lags=10)
    r_bp = ljung_box_test(e, lags=10, boxpierce=True)
    # Box-Pierce stat should be smaller (different formula)
    assert r_bp["statistic"] < r_lb["statistic"]


def test_lb_empty_raises():
    with pytest.raises(ValueError):
        ljung_box_test(np.array([]))


def test_lb_nonpositive_lags_raises():
    with pytest.raises(ValueError, match="lags"):
        ljung_box_test(_white_noise(50), lags=0)


def test_lb_p_values_in_range():
    e = _white_noise(200)
    r = ljung_box_test(e, lags=10)
    for pv in r["p_values"]:
        assert 0 <= pv <= 1


@pytest.mark.academic_reference
def test_lb_matches_statsmodels():
    """Ljung-Box stat should be within 2% of statsmodels."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        pytest.skip("statsmodels not installed")
    e = _white_noise(200)
    r = ljung_box_test(e, lags=10)
    sm_r = acorr_ljungbox(e, lags=[10], return_df=True)
    sm_stat = float(sm_r["lb_stat"].iloc[0])
    assert abs(r["statistic"] - sm_stat) < abs(sm_stat) * 0.02 + 0.01


# ---------- Durbin-Watson tests ----------

def test_dw_no_autocorrelation():
    e = _white_noise(200)
    dw = durbin_watson(e)
    # Should be near 2
    assert 1.5 < dw < 2.5


def test_dw_positive_autocorrelation():
    e = _ar1_residuals(phi=0.9, n=300)
    dw = durbin_watson(e)
    assert dw < 1.5


def test_dw_returns_float():
    e = _white_noise(100)
    dw = durbin_watson(e)
    assert isinstance(dw, float)


def test_dw_range():
    e = _white_noise(200)
    dw = durbin_watson(e)
    assert 0 <= dw <= 4


def test_dw_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        durbin_watson(np.array([1.0]))


def test_lb_series_input():
    import pandas as pd
    e = pd.Series(_white_noise(100))
    r = ljung_box_test(e, lags=5)
    assert "statistic" in r


def test_lb_list_lags_negative_raises():
    e = _white_noise(100)
    with pytest.raises(ValueError, match="positive"):
        ljung_box_test(e, lags=[-1, 5])


def test_lb_zero_variance():
    """Constant residuals: no autocorrelation."""
    e = np.ones(100)
    r = ljung_box_test(e, lags=5)
    assert r["p_value"] >= 0


def test_dw_series_input():
    import pandas as pd
    e = pd.Series(_white_noise(100))
    dw = durbin_watson(e)
    assert 0 <= dw <= 4


def test_dw_zero_variance():
    e = np.zeros(50)
    dw = durbin_watson(e)
    assert dw == pytest.approx(2.0)
