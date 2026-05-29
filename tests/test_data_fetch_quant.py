"""Tests for data_fetch and quant_analysis oprims."""
import pytest
from oprim.data_fetch import *  # noqa: F403, F405
from oprim.quant_analysis import *  # noqa: F403, F405


# --- data_fetch tests ---

class MockHttp:
    def __init__(self, resp=None):
        self._resp = resp
    async def get(self, url, **kw):
        return self._resp
    async def post(self, url, **kw):
        return self._resp

@pytest.mark.asyncio
async def test_fetch_order_book_depth():
    r = await fetch_order_book_depth(client=MockHttp({"bids": [], "asks": []}), exchange="binance", symbol="BTC")
    assert isinstance(r, dict)

@pytest.mark.asyncio
async def test_compute_spread():
    r = await compute_spread(bids=[[50000, 1]], asks=[[50010, 1]])
    assert r["spread"] == 10
    assert r["spread_bps"] == pytest.approx(2.0, abs=0.1)

@pytest.mark.asyncio
async def test_compute_slippage():
    r = await compute_slippage_estimate(order_size=5, bids=[], asks=[[100, 10], [101, 10]], side="buy")
    assert r == 0.0  # all filled at first level

@pytest.mark.asyncio
async def test_nlp_sentiment():
    r = await nlp_sentiment_analysis(text="bull pump rally up gain")
    assert r["label"] == "positive"

@pytest.mark.asyncio
async def test_nlp_sentiment_negative():
    r = await nlp_sentiment_analysis(text="crash dump sell everything")
    assert r["label"] == "negative"

@pytest.mark.asyncio
async def test_compute_option_skew():
    r = await compute_option_skew(call_iv=0.5, put_iv=0.65)
    assert r == 15.0

@pytest.mark.asyncio
async def test_compute_term_structure_contango():
    r = await compute_term_structure(futures_prices={"1M": 50100, "3M": 50500}, spot=50000)
    assert r["shape"] == "contango"

@pytest.mark.asyncio
async def test_compute_term_structure_backwardation():
    r = await compute_term_structure(futures_prices={"1M": 49900}, spot=50000)
    assert r["shape"] == "backwardation"

@pytest.mark.asyncio
async def test_merge_exchange_ohlcv():
    r = await merge_exchange_ohlcv(sources=[[{"close": 100}]])
    assert len(r) == 1

@pytest.mark.asyncio
async def test_clean_ohlcv_outliers():
    bars = [{"close": 100}] * 10 + [{"close": 999}]
    r = await clean_ohlcv_outliers(bars=bars)
    assert len(r) < len(bars)

@pytest.mark.asyncio
async def test_compute_vwap():
    r = await compute_volume_weighted_price(bars=[{"close": 100, "volume": 10}, {"close": 200, "volume": 10}])
    assert r == 150.0

@pytest.mark.asyncio
async def test_compute_microstructure():
    r = await compute_microstructure_features(ticks=[{"price": 100, "qty": 1, "side": "buy"}, {"price": 101, "qty": 2, "side": "sell"}])
    assert r["buy_ratio"] == 0.5
    assert r["tick_count"] == 2

@pytest.mark.asyncio
async def test_cross_exchange_funding():
    class MultiHttp:
        async def get(self, url, **kw):
            return {"rate": 0.0001}
    r = await cross_exchange_funding_diff(client=MultiHttp(), symbol="BTC-USDT", exchanges=["a", "b"])
    assert r["diff"] == 0.0

# --- quant_analysis tests ---

def test_compute_pnl():
    r = compute_pnl_from_trades(trades=[{"entry": 100, "exit": 110, "size": 2}])
    assert r == [20.0]

def test_compute_equity_curve():
    r = compute_equity_curve(initial_capital=1000, pnl_series=[100, -50])
    assert r == [1000, 1100, 1050]

def test_compute_drawdown_distribution():
    r = compute_drawdown_distribution(equity_curve=[100, 110, 100, 120])
    assert r["max_dd"] > 0
    assert r["dd_count"] >= 1

def test_compute_market_impact():
    r = compute_market_impact(order_size=1000, avg_daily_volume=100000, volatility=0.02)
    assert r > 0

def test_generate_bootstrap():
    r = generate_bootstrap_samples(data=[1, 2, 3, 4, 5], n_samples=10)
    assert len(r) == 10

def test_monte_carlo():
    r = compute_monte_carlo_simulation(mean_return=0.001, std_return=0.02, n_periods=10, n_paths=50)
    assert len(r["terminal_values"]) == 50

def test_benchmark_metrics():
    r = compute_benchmark_metrics(strategy_returns=[0.01, 0.02, -0.01], benchmark_returns=[0.005, 0.01, -0.005])
    assert "alpha" in r and "beta" in r

def test_relative_performance():
    r = compute_relative_performance(strategy_curve=[100, 110], benchmark_curve=[100, 105])
    assert r[1] > 1.0

def test_split_train_test():
    train, test = split_train_test_time_series(data=list(range(10)), train_ratio=0.7)
    assert len(train) == 7 and len(test) == 3

def test_portfolio_turnover():
    r = compute_portfolio_turnover(weights_before={"A": 0.5, "B": 0.5}, weights_after={"A": 0.7, "B": 0.3})
    assert r == 0.2

def test_position_churn():
    r = compute_position_churn(position_history=[{"A": 0.5}, {"A": 0.7}, {"A": 0.4}])
    assert r > 0

def test_risk_exposure():
    r = compute_risk_exposure(positions={"BTC": 0.6, "ETH": 0.4}, factor_loadings={"BTC": {"market": 1.2}, "ETH": {"market": 1.5}})
    assert r["market"] == pytest.approx(1.32)

def test_position_risk():
    r = compute_position_risk(position_size=10000, volatility=0.02)
    assert r["var_95"] > 0

def test_mcmc_sample():
    r = compute_mcmc_sample(log_posterior_fn=lambda x: -sum(xi**2 for xi in x), initial=[0.0], n_samples=50)
    assert len(r) == 50

def test_shapley():
    r = compute_shapley_decomposition(contributions={"A": 30, "B": 20}, total=72)
    assert "residual" in r

def test_herfindahl():
    r = compute_herfindahl_index(weights={"A": 0.5, "B": 0.3, "C": 0.2})
    assert 0 < r < 1

def test_signal_crowding():
    r = compute_signal_crowding(signal_counts={"buy": 80, "sell": 20}, total_participants=100)
    assert r["dominant"] == "buy"
    assert r["crowding_ratio"] == 0.8

def test_uncertainty_threshold():
    r = compute_uncertainty_threshold(uncertainties=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert r > 0.4
