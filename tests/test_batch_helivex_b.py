"""Batch B tests: 13 new oprim elements."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────── okx_rest_call ────────────────────────────


class TestOkxRestCall:
    @pytest.mark.asyncio
    async def test_successful_get(self):
        from oprim.okx_rest_call import okx_rest_call

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": "0", "data": [{"instId": "BTC-USDT"}]}

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value.request = AsyncMock(return_value=mock_resp)
            result = await okx_rest_call("/api/v5/market/tickers", params={"instType": "SPOT"})

        assert result["code"] == "0"
        assert "data" in result

    @pytest.mark.asyncio
    async def test_non_200_raises(self):
        from oprim.okx_rest_call import OkxRestError, okx_rest_call

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value.request = AsyncMock(return_value=mock_resp)
            with pytest.raises(OkxRestError):
                await okx_rest_call("/api/v5/bad")

    @pytest.mark.asyncio
    async def test_okx_error_code_raises(self):
        from oprim.okx_rest_call import OkxRestError, okx_rest_call

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": "50001", "msg": "instrument not found"}

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value.request = AsyncMock(return_value=mock_resp)
            with pytest.raises(OkxRestError):
                await okx_rest_call("/api/v5/market/candles")

    @pytest.mark.asyncio
    async def test_uses_correct_url(self):
        from oprim.okx_rest_call import okx_rest_call

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": "0", "data": []}

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value.request = AsyncMock(return_value=mock_resp)
            await okx_rest_call("/api/v5/test", base_url="https://sandbox.okx.com")
            call_args = MockClient.return_value.request.call_args
            assert "sandbox.okx.com" in call_args.args[1]

    @pytest.mark.asyncio
    async def test_post_method(self):
        from oprim.okx_rest_call import okx_rest_call

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": "0", "data": []}

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value.request = AsyncMock(return_value=mock_resp)
            await okx_rest_call("/api/v5/trade/order", method="POST", body={"instId": "BTC"})
            call_args = MockClient.return_value.request.call_args
            assert call_args.args[0].upper() == "POST"


# ─────────────────────────── ohlcv_fetch ────────────────────────────


class TestOhlcvFetch:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        from oprim.ohlcv_fetch import ohlcv_fetch

        fake_data = [["1700000000000", "42000", "43000", "41000", "42500", "100", "4200000"]]
        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": fake_data}
            result = await ohlcv_fetch("BTC-USDT-SWAP")
        assert isinstance(result, list)
        assert result[0]["close"] == 42500.0

    @pytest.mark.asyncio
    async def test_ohlcv_keys_present(self):
        from oprim.ohlcv_fetch import ohlcv_fetch

        fake_data = [["1700000000000", "1", "2", "0.5", "1.5", "10", "15"]]
        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": fake_data}
            result = await ohlcv_fetch("ETH-USDT")
        row = result[0]
        for key in ("ts", "open", "high", "low", "close", "vol"):
            assert key in row

    @pytest.mark.asyncio
    async def test_unsupported_venue_raises(self):
        from oprim.ohlcv_fetch import OhlcvFetchError, ohlcv_fetch

        with pytest.raises(OhlcvFetchError, match="venue"):
            await ohlcv_fetch("BTC-USDT", venue="binance")

    @pytest.mark.asyncio
    async def test_api_error_raises_fetch_error(self):
        from oprim.ohlcv_fetch import OhlcvFetchError, ohlcv_fetch
        from oprim.okx_rest_call import OkxRestError

        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.side_effect = OkxRestError("api down")
            with pytest.raises(OhlcvFetchError):
                await ohlcv_fetch("BTC-USDT-SWAP")

    @pytest.mark.asyncio
    async def test_empty_data_returns_empty_list(self):
        from oprim.ohlcv_fetch import ohlcv_fetch

        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": []}
            result = await ohlcv_fetch("BTC-USDT-SWAP")
        assert result == []


# ─────────────────────────── funding_rate_fetch ────────────────────────────


class TestFundingRateFetch:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        from oprim.funding_rate_fetch import funding_rate_fetch

        fake = [{"fundingTime": "1700000000000", "fundingRate": "0.0001",
                 "realizedRate": "0.00009", "nextFundingTime": "1700028800000"}]
        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": fake}
            result = await funding_rate_fetch("BTC-USDT-SWAP")
        assert result[0]["funding_rate"] == pytest.approx(0.0001)

    @pytest.mark.asyncio
    async def test_funding_keys_present(self):
        from oprim.funding_rate_fetch import funding_rate_fetch

        fake = [{"fundingTime": "1700000000000", "fundingRate": "0.0002",
                 "realizedRate": "0.0002", "nextFundingTime": "1700028800000"}]
        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": fake}
            result = await funding_rate_fetch("BTC-USDT-SWAP")
        for key in ("ts", "funding_rate", "realized_rate", "next_funding_time"):
            assert key in result[0]

    @pytest.mark.asyncio
    async def test_unsupported_venue_raises(self):
        from oprim.funding_rate_fetch import FundingRateFetchError, funding_rate_fetch

        with pytest.raises(FundingRateFetchError, match="venue"):
            await funding_rate_fetch("BTC-USDT-SWAP", venue="binance")

    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self):
        from oprim.funding_rate_fetch import funding_rate_fetch

        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.return_value = {"code": "0", "data": []}
            result = await funding_rate_fetch("BTC-USDT-SWAP")
        assert result == []

    @pytest.mark.asyncio
    async def test_api_error_propagates(self):
        from oprim.funding_rate_fetch import FundingRateFetchError, funding_rate_fetch
        from oprim.okx_rest_call import OkxRestError

        with patch("oprim.okx_rest_call.okx_rest_call", new_callable=AsyncMock) as m:
            m.side_effect = OkxRestError("err")
            with pytest.raises(FundingRateFetchError):
                await funding_rate_fetch("BTC-USDT-SWAP")


# ─────────────────────────── okx_ws_msg ────────────────────────────


class TestOkxWsMsg:
    @pytest.mark.asyncio
    async def test_callback_called_with_data(self):
        from oprim.okx_ws_msg import okx_ws_msg

        received = []

        push = json.dumps({"arg": {"channel": "tickers"}, "data": [{"last": "42000"}]})
        sub_ack = json.dumps({"event": "subscribe", "arg": {"channel": "tickers"}})

        mock_ws = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[sub_ack, push, TimeoutError()])

        with patch("websockets.connect", return_value=mock_ws):
            await okx_ws_msg("tickers", inst_id="BTC-USDT", callback=received.append, timeout=0.1)

        assert len(received) >= 1
        assert "data" in received[0]

    @pytest.mark.asyncio
    async def test_subscribe_message_sent(self):
        from oprim.okx_ws_msg import okx_ws_msg

        mock_ws = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=TimeoutError())

        with patch("websockets.connect", return_value=mock_ws):
            await okx_ws_msg("books5", inst_id="ETH-USDT-SWAP",
                             callback=lambda _: None, timeout=0.05)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["op"] == "subscribe"
        assert sent["args"][0]["channel"] == "books5"

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        from oprim.okx_ws_msg import OkxWsError, okx_ws_msg

        with patch("websockets.connect", side_effect=OSError("refused")):
            with pytest.raises(OkxWsError):
                await okx_ws_msg("tickers", inst_id="BTC", callback=lambda _: None, timeout=0.05)

    @pytest.mark.asyncio
    async def test_error_event_raises(self):
        from oprim.okx_ws_msg import OkxWsError, okx_ws_msg

        err_msg = json.dumps({"event": "error", "code": "60001", "msg": "bad channel"})

        mock_ws = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[err_msg])

        with patch("websockets.connect", return_value=mock_ws):
            with pytest.raises(OkxWsError):
                await okx_ws_msg("bad", inst_id="X", callback=lambda _: None, timeout=0.1)

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self):
        from oprim.okx_ws_msg import okx_ws_msg

        received = []

        async def async_cb(msg):
            received.append(msg)

        push = json.dumps({"data": [{"price": "100"}]})

        mock_ws = AsyncMock()
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[push, TimeoutError()])

        with patch("websockets.connect", return_value=mock_ws):
            await okx_ws_msg("tickers", inst_id="BTC", callback=async_cb, timeout=0.1)

        assert len(received) >= 1


# ─────────────────────────── ed25519_sign ────────────────────────────


class TestEd25519Sign:
    def _keygen(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key = Ed25519PrivateKey.generate()
        private_bytes = key.private_bytes_raw()
        public_key = key.public_key()
        return private_bytes, public_key

    def test_signature_is_64_bytes(self):
        from oprim.ed25519_sign import ed25519_sign
        priv, _ = self._keygen()
        sig = ed25519_sign(b"hello", private_key=priv)
        assert len(sig) == 64

    def test_signature_verifies(self):
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from oprim.ed25519_sign import ed25519_sign

        priv, pub = self._keygen()
        msg = b"sign this message"
        sig = ed25519_sign(msg, private_key=priv)
        pub.verify(sig, msg)  # raises InvalidSignature on failure

    def test_wrong_message_fails_verification(self):
        from cryptography.exceptions import InvalidSignature
        from oprim.ed25519_sign import ed25519_sign

        priv, pub = self._keygen()
        sig = ed25519_sign(b"original", private_key=priv)
        with pytest.raises(InvalidSignature):
            pub.verify(sig, b"tampered")

    def test_invalid_key_length_raises(self):
        from oprim.ed25519_sign import ed25519_sign
        with pytest.raises(ValueError, match="32 bytes"):
            ed25519_sign(b"msg", private_key=b"tooshort")

    def test_different_keys_different_signatures(self):
        from oprim.ed25519_sign import ed25519_sign
        priv1, _ = self._keygen()
        priv2, _ = self._keygen()
        sig1 = ed25519_sign(b"msg", private_key=priv1)
        sig2 = ed25519_sign(b"msg", private_key=priv2)
        assert sig1 != sig2

    def test_returns_bytes(self):
        from oprim.ed25519_sign import ed25519_sign
        priv, _ = self._keygen()
        assert isinstance(ed25519_sign(b"x", private_key=priv), bytes)


# ─────────────────────────── hmm_baum_welch / hmm_viterbi ────────────────────────────


class TestHmmBaumWelch:
    def test_returns_model_dict(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        obs = [0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15]
        result = hmm_baum_welch(obs, n_states=2)
        assert "transmat" in result and "_model" in result

    def test_n_states_respected(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        obs = list(range(20))
        result = hmm_baum_welch(obs, n_states=3)
        assert len(result["transmat"]) == 3

    def test_delegates_to_hmmlearn_runtime(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        with patch("obase.hmmlearn_runtime.HmmlearnRuntime.fit") as mock_fit:
            mock_fit.return_value = {"n_states": 2, "_model": None, "transmat": []}
            hmm_baum_welch([1, 2, 3], n_states=2)
            mock_fit.assert_called_once()

    def test_returns_all_expected_keys(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        obs = [float(i % 3) for i in range(20)]
        result = hmm_baum_welch(obs, n_states=2)
        for k in ("n_states", "transmat", "means", "covars", "startprob", "_model"):
            assert k in result

    def test_fit_predict_roundtrip(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        from oprim.hmm_viterbi import hmm_viterbi
        obs = [0.0, 1.0, 0.0, 1.0, 0.5] * 4
        model = hmm_baum_welch(obs, n_states=2)
        states = hmm_viterbi(obs, model=model)
        assert len(states) == len(obs)
        assert all(0 <= s < 2 for s in states)


class TestHmmViterbi:
    def test_returns_list_of_ints(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        from oprim.hmm_viterbi import hmm_viterbi
        obs = [float(i % 5) for i in range(20)]
        model = hmm_baum_welch(obs, n_states=2)
        result = hmm_viterbi(obs, model=model)
        assert isinstance(result, list)
        assert all(isinstance(s, int) for s in result)

    def test_invalid_model_raises(self):
        from oprim.hmm_viterbi import hmm_viterbi
        with pytest.raises(ValueError):
            hmm_viterbi([1.0, 2.0], model={"n_states": 2})

    def test_length_matches(self):
        from oprim.hmm_baum_welch import hmm_baum_welch
        from oprim.hmm_viterbi import hmm_viterbi
        obs = [0.2, 0.8, 0.3, 0.7] * 5
        model = hmm_baum_welch(obs, n_states=2)
        assert len(hmm_viterbi(obs, model=model)) == len(obs)


# ─────────────────────────── cointegration_test ────────────────────────────


class TestCointegrationTest:
    def _make_cointegrated(self, n=60):
        import numpy as np
        x = np.cumsum(np.random.default_rng(42).normal(0, 1, n))
        noise = np.random.default_rng(99).normal(0, 0.1, n)
        y = 2.0 * x + noise
        return x.tolist(), y.tolist()

    def test_returns_expected_keys(self):
        from oprim.cointegration_test import cointegration_test
        a, b = self._make_cointegrated()
        result = cointegration_test(a, b)
        for k in ("t_stat", "p_value", "crit_values", "cointegrated", "hedge_ratio"):
            assert k in result

    def test_cointegrated_series_detected(self):
        from oprim.cointegration_test import cointegration_test
        a, b = self._make_cointegrated(80)
        result = cointegration_test(a, b)
        assert result["cointegrated"] is True

    def test_unequal_length_raises(self):
        from oprim.cointegration_test import cointegration_test
        with pytest.raises(ValueError, match="equal length"):
            cointegration_test([1, 2, 3], [1, 2])

    def test_too_short_raises(self):
        from oprim.cointegration_test import cointegration_test
        with pytest.raises(ValueError, match="short"):
            cointegration_test([1, 2], [1, 2])

    def test_crit_values_keys(self):
        from oprim.cointegration_test import cointegration_test
        a, b = self._make_cointegrated()
        result = cointegration_test(a, b)
        assert "1%" in result["crit_values"]
        assert "5%" in result["crit_values"]


# ─────────────────────────── zscore_signal ────────────────────────────


class TestZscoreSignal:
    def test_known_zscore(self):
        from oprim.zscore_signal import zscore_signal
        # Series of all zeros except last = 2: z = (2-0)/std
        series = [0.0, 0.0, 0.0, 0.0, 2.0]
        result = zscore_signal(series, lookback=4)
        assert result["zscore"] > 1.0
        assert result["signal"] == "short"

    def test_returns_all_keys(self):
        from oprim.zscore_signal import zscore_signal
        result = zscore_signal([1.0, 2.0, 3.0, 4.0, 5.0], lookback=3)
        for k in ("zscore", "zscores", "mean", "std", "signal"):
            assert k in result

    def test_flat_signal(self):
        from oprim.zscore_signal import zscore_signal
        result = zscore_signal([1.0, 1.0, 1.0, 1.0], lookback=3)
        # All same → std = 0 → z = 0 → flat
        assert result["signal"] == "flat"

    def test_zscores_length(self):
        from oprim.zscore_signal import zscore_signal
        series = list(range(10))
        result = zscore_signal(series, lookback=3)
        assert len(result["zscores"]) == 10 - 3 + 1

    def test_too_short_raises(self):
        from oprim.zscore_signal import zscore_signal
        with pytest.raises(ValueError):
            zscore_signal([1.0, 2.0], lookback=5)

    def test_invalid_lookback_raises(self):
        from oprim.zscore_signal import zscore_signal
        with pytest.raises(ValueError):
            zscore_signal([1.0, 2.0, 3.0], lookback=1)


# ─────────────────────────── pbo_compute ────────────────────────────


class TestPboCompute:
    def test_no_overfitting(self):
        from oprim.pbo_compute import pbo_compute
        # IS best is also OOS best
        is_sr = [0.5, 1.5, 0.3]
        oos_sr = [0.4, 1.4, 0.2]
        result = pbo_compute(is_sr, oos_sr)
        assert result["pbo"] < 0.5
        assert result["best_is_idx"] == 1

    def test_overfitting_detected(self):
        from oprim.pbo_compute import pbo_compute
        # IS best is OOS worst
        is_sr = [0.1, 0.2, 2.0]
        oos_sr = [1.0, 0.8, -0.5]
        result = pbo_compute(is_sr, oos_sr)
        assert result["pbo"] > 0.5

    def test_returns_expected_keys(self):
        from oprim.pbo_compute import pbo_compute
        result = pbo_compute([1.0, 2.0], [1.5, 0.5])
        for k in ("pbo", "best_is_idx", "oos_normalized_rank", "lambda_"):
            assert k in result

    def test_empty_raises(self):
        from oprim.pbo_compute import pbo_compute
        with pytest.raises(ValueError):
            pbo_compute([], [])

    def test_unequal_length_raises(self):
        from oprim.pbo_compute import pbo_compute
        with pytest.raises(ValueError):
            pbo_compute([1.0, 2.0], [1.0])

    def test_pbo_in_unit_interval(self):
        from oprim.pbo_compute import pbo_compute
        result = pbo_compute([0.5, 1.0, 1.5, 2.0], [0.3, 0.8, 1.2, 0.1])
        assert 0.0 <= result["pbo"] <= 1.0


# ─────────────────────────── deflated_sharpe ────────────────────────────


class TestDeflatedSharpe:
    def test_returns_expected_keys(self):
        from oprim.deflated_sharpe import deflated_sharpe
        result = deflated_sharpe(1.5, n_trials=20)
        for k in ("deflated_sharpe", "dsr_probability", "e_max_sr", "significant"):
            assert k in result

    def test_high_sharpe_many_trials_may_not_be_significant(self):
        from oprim.deflated_sharpe import deflated_sharpe
        # 1000 trials, modest sharpe — may not pass the hurdle
        result = deflated_sharpe(0.5, n_trials=1000)
        assert "deflated_sharpe" in result

    def test_single_trial_zero_hurdle(self):
        from oprim.deflated_sharpe import deflated_sharpe
        result = deflated_sharpe(1.0, n_trials=1)
        assert result["e_max_sr"] == pytest.approx(0.0)
        assert result["deflated_sharpe"] == pytest.approx(1.0)
        assert result["significant"] is True

    def test_invalid_n_trials_raises(self):
        from oprim.deflated_sharpe import deflated_sharpe
        with pytest.raises(ValueError):
            deflated_sharpe(1.0, n_trials=0)

    def test_with_returns_uses_skewness(self):
        from oprim.deflated_sharpe import deflated_sharpe
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 20
        result = deflated_sharpe(1.2, n_trials=10, returns=returns)
        assert isinstance(result["deflated_sharpe"], float)

    def test_dsr_probability_in_unit_interval(self):
        from oprim.deflated_sharpe import deflated_sharpe
        result = deflated_sharpe(2.0, n_trials=5)
        assert 0.0 <= result["dsr_probability"] <= 1.0


# ─────────────────────────── cpcv_split ────────────────────────────


class TestCpcvSplit:
    def test_returns_n_splits_folds(self):
        from oprim.cpcv_split import cpcv_split
        data = list(range(100))
        splits = cpcv_split(data, n_splits=5)
        assert len(splits) == 5

    def test_fold_keys(self):
        from oprim.cpcv_split import cpcv_split
        splits = cpcv_split(list(range(20)), n_splits=4)
        for s in splits:
            assert "train_idx" in s and "test_idx" in s and "fold" in s

    def test_no_overlap_train_test(self):
        from oprim.cpcv_split import cpcv_split
        splits = cpcv_split(list(range(30)), n_splits=3)
        for s in splits:
            assert not set(s["train_idx"]) & set(s["test_idx"])

    def test_embargo_removes_boundary_samples(self):
        from oprim.cpcv_split import cpcv_split
        splits = cpcv_split(list(range(20)), n_splits=4, embargo=2)
        # train should be smaller with embargo
        no_embargo = cpcv_split(list(range(20)), n_splits=4, embargo=0)
        total_train_emb = sum(len(s["train_idx"]) for s in splits)
        total_train_no = sum(len(s["train_idx"]) for s in no_embargo)
        assert total_train_emb <= total_train_no

    def test_n_splits_less_than_2_raises(self):
        from oprim.cpcv_split import cpcv_split
        with pytest.raises(ValueError):
            cpcv_split(list(range(10)), n_splits=1)

    def test_data_too_short_raises(self):
        from oprim.cpcv_split import cpcv_split
        with pytest.raises(ValueError):
            cpcv_split([1, 2], n_splits=5)

    def test_all_indices_covered(self):
        from oprim.cpcv_split import cpcv_split
        n = 20
        splits = cpcv_split(list(range(n)), n_splits=4)
        all_test = sorted(idx for s in splits for idx in s["test_idx"])
        assert all_test == list(range(n))


# ─────────────────────────── risk_limit_check ────────────────────────────


class TestRiskLimitCheck:
    def test_passes_when_within_limits(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(50_000.0, max_position=100_000.0)
        assert result["pass"] is True
        assert result["violated_rule"] is None

    def test_fails_on_position_breach(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(150_000.0, max_position=100_000.0)
        assert result["pass"] is False
        assert result["violated_rule"] == "max_position"

    def test_fails_on_drawdown_breach(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(
            10_000.0, max_position=100_000.0,
            max_drawdown=0.05, current_drawdown=0.08,
        )
        assert result["pass"] is False
        assert result["violated_rule"] == "max_drawdown"

    def test_drawdown_within_limit_passes(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(
            10_000.0, max_position=100_000.0,
            max_drawdown=0.10, current_drawdown=0.05,
        )
        assert result["pass"] is True

    def test_custom_rule_above_fails(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(
            1_000.0, max_position=100_000.0,
            rules=[{"name": "vol_limit", "limit": 0.15, "value": 0.20, "direction": "above"}],
        )
        assert result["pass"] is False
        assert result["violated_rule"] == "vol_limit"

    def test_custom_rule_below_fails(self):
        from oprim.risk_limit_check import risk_limit_check
        result = risk_limit_check(
            1_000.0, max_position=100_000.0,
            rules=[{"name": "liquidity_min", "limit": 1_000_000.0,
                    "value": 500_000.0, "direction": "below"}],
        )
        assert result["pass"] is False
        assert result["violated_rule"] == "liquidity_min"

    def test_invalid_max_position_raises(self):
        from oprim.risk_limit_check import risk_limit_check
        with pytest.raises(ValueError):
            risk_limit_check(100.0, max_position=0.0)

    def test_missing_max_drawdown_raises(self):
        from oprim.risk_limit_check import risk_limit_check
        with pytest.raises(ValueError):
            risk_limit_check(100.0, max_position=1000.0, current_drawdown=0.05)
