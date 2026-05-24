# oprim Coverage Gap Report — feat/v1.5.0-phase2

**Current:** 81.11% (327 missing lines / 1866 statements)  
**Target:** 95%  
**Gap to close:** ~14 pp ≈ ~261 lines must become covered (A-class)

---

## 分类汇总

| 类别 | 行数 | 策略 |
|------|------|------|
| A — 须补测 | ~220 | 新增测试用例 |
| B — 可豁免 | ~82 | 加 `# pragma: no cover` 或 docstring 注释 |
| C — 可删/标 | ~25 | `# pragma: no cover` (死代码/防御包装) |

---

## 未覆盖文件逐一分析

### 1. `oprim/_base.py` (0%, 5 lines: 3-9)

```python
3: import numpy as np
4: from typing import Union
5: ...
9: ArrayLike = Union[np.ndarray, list, pd.Series]
```

**类别: B**  
纯类型别名定义，无可执行分支，无业务逻辑。  
**豁免方式:** 文件顶部加 `# pragma: no cover`

---

### 2. `oprim/_manifest.py` (0%, 5 lines: 3-156)

```python
3: VERSION = "1.5.0"
4: ELEMENTS = [...]
```

**类别: B**  
纯数据文件，无可执行逻辑。  
**豁免方式:** 文件顶部加 `# pragma: no cover`

---

### 3. `oprim/_validation.py` (0%, 19 lines: 3-31)

```python
 8: def validate_positive_array(data, name="data"):
18: def validate_no_nan(data, name="data"):
26: def validate_min_length(data, min_len, name="data"):
```

**类别: C**  
`grep` 确认：整个 oprim 包内没有任何模块 `import _validation`，三个 validator 函数是**孤儿代码**，从未被调用。  
**处理方式:** 文件顶部加 `# pragma: no cover`，或直接删除（建议删除以消除误导）

---

### 4. `oprim/information.py` (8%, 38 lines: 33-127)

三个公开 API 函数，全部暴露在 `oprim.__init__` 的 `__all__` 中，完全没有测试。

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 33-41 | `shannon_entropy` 函数体 | **A** | 公开 API，被 `oskill.causal` 间接依赖 |
| 69-89 | `ordinal_pattern` 函数体 | **A** | 公开 API，被 `oskill.causal` 直接调用 |
| 118-127 | `phase_randomize` 函数体 | **A** | 公开 API，surrogate significance testing |

**建议新增测试:**
- `test_shannon_entropy_uniform` — 均匀分布熵 = log2(n)
- `test_shannon_entropy_deterministic` — 单值序列熵 = 0
- `test_ordinal_pattern_basic` — d=3，验证模式索引范围 [0, 5]
- `test_ordinal_pattern_too_short_raises`
- `test_phase_randomize_preserves_power_spectrum` — `np.abs(rfft(original))` ≈ `np.abs(rfft(surrogate))`

---

### 5. `oprim/signal_processing.py` (7%, 74 lines: 30-259)

五个公开函数，全部在 `__all__` 中，零测试。

| 行 | 函数 | 类别 | 理由 |
|----|------|------|------|
| 30-39 | `linear_slope` 函数体 | **A** | 公开 API，简单纯函数，≥2 bars |
| 67-81 | `atr` Wilder 平滑循环 | **A** | `atr` 主路径（平滑部分），需要 >period+1 bars 才触发 |
| 111-145 | `hurst_exponent` R/S 循环 | **A** | 公开 API，量化因子 |
| 175-190 | `compute_dwt` 函数体 | **B** | 依赖可选库 `pywt`，不在 CI 依赖中 |
| 221-225 | `H_change_rate_std` 函数体 | **A** | 公开 API，简单纯函数 |
| 251-259 | `orderbook_entropy` 函数体 | **A** | 公开 API，简单纯函数 |

**备注:** `atr` (lines 42-66) 已有覆盖，缺的是 period 较大时触发的 Wilder 平滑迭代 (lines 67-81)。

**建议新增测试 (在 `tests/test_signal_processing.py`):**
- `test_linear_slope_basic` — 斜线序列验证斜率
- `test_linear_slope_normalized` — normalize=False 返回原始斜率
- `test_atr_wilder_smoothing` — period=3，len>=10，验证平滑迭代被执行
- `test_hurst_exponent_random_walk` — H ≈ 0.5
- `test_hurst_exponent_trending` — 单调序列 H > 0.5
- `test_H_change_rate_std_basic`
- `test_orderbook_entropy_uniform` — 均匀分布熵最大

**`compute_dwt` 豁免:** 文件中该函数加 `# pragma: no cover` 注释（`pywt` 未列入 pyproject.toml 依赖）

---

### 6. `oprim/topology.py` (12%, 18 lines: 36-92)

两个公开 API 函数（TDA），零测试。

| 行 | 函数 | 类别 | 理由 |
|----|------|------|------|
| 36-40 | `takens_embed` 函数体 | **A** | 公开 API，入参简单可测试 |
| 76-92 | `persistence_landscape` 函数体 | **A** | 公开 API，验证输出形状和 tent 函数 |

**建议新增测试 (在 `tests/test_topology.py`):**
- `test_takens_embed_shape` — x=[1..10], d=3, tau=1 → shape (8, 3)
- `test_takens_embed_too_short_raises`
- `test_persistence_landscape_empty_dgm` — 输入空矩阵 → zeros
- `test_persistence_landscape_single_point` — 验证 tent 函数峰值
- `@academic_reference test_persistence_landscape_bubenik_2015` — 手算 tent 函数值

---

### 7. `oprim/point_process.py` (12%, 17 lines: 40-61)

`hawkes_nll` 函数体从未被直接测试（`oskill.fit_hawkes` 测试覆盖率也低）。

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 40-61 | `hawkes_nll` 递归 A[] 计算 + NLL | **A** | 公开 API，被 `oskill.fit_hawkes` 调用；直接测试 NLL 数值 |

**建议新增测试 (在 `tests/test_point_process.py` 新建):**
- `test_hawkes_nll_finite_for_valid_params` — log_mu=-2, log_alpha=-2, log_beta=0
- `test_hawkes_nll_returns_large_for_less_than_2_events`
- `test_hawkes_nll_decreases_with_better_params` — 用已知模拟数据

---

### 8. `oprim/derivatives/black_scholes.py` (81%, ~15 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 82, 86 | `sigma=0` put 分支 (BS price) | **A** | 用户可触发：sigma=0 时返回折现内在价值 |
| 132 | invalid `option_type` for Greeks | **A** | 验证分支，其他函数已测 call 分支 |
| 145 | T==0 put delta | **A** | 到期日 delta 边界 |
| 248 | IV: invalid `option_type` | **A** | 验证分支 |
| 266 | IV: price below intrinsic → NaN | **A** | 不可解情形，用户可触发 |
| 271-274 | IV: inline put price (sigma≤0) | **B** | Newton 迭代内部 sigma→0 边界，极少触发 |
| 284 | IV: inline put price return | **A** | put 路径，与 call 对称，应测 |
| 288-291 | IV: vega when sigma≤0 → 0 | **B** | 数值保护分支，非用户可控 |
| 316, 318-319 | IV Newton 失败/exception | **C** | 异常兜底，`except (ValueError, RuntimeError): return nan` |

**建议新增测试:**
- `test_bs_price_sigma_zero_call` — S=110, K=100, sigma=0 → intrinsic
- `test_bs_price_sigma_zero_put` — S=90, K=100, sigma=0 → intrinsic
- `test_bs_greeks_invalid_option_type_raises`
- `test_bs_greeks_T_zero_put_delta` — delta=-1 when S<K, T=0
- `test_iv_below_intrinsic_returns_nan`
- `test_iv_put_option` — 对称测试 put IV

---

### 9. `oprim/finance.py` (69%, ~57 lines)

**最大 A 类缺口：两个完全未覆盖的公开函数**

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 39 | `drawdown_curve`: empty series after dropna | **A** | 用户可触发的 ValueError |
| 78 | `drawdown_curve`: recovery == dd_end edge | **B** | 极端恢复边界，基本不可能从用户侧触发 |
| 87, 90 | `drawdown_curve`: int vs Timestamp underwater | **A** | int 索引路径未测 |
| 183 | `factor_attribution`: non-Series input | **A** | 允许 ndarray 输入 |
| 203-205 | `factor_attribution`: HAC lags auto-compute | **A** | HAC 路径整段未测 |
| 259 | `value_at_risk_and_es`: non-Series input | **A** | 允许 ndarray 输入 |
| 298 | `value_at_risk_and_es`: empty tail ES warning | **A** | 罕见但可测：VaR 在第一分位时尾部为空 |
| 330-358 | **`nelson_siegel_fit` 整个函数** | **A** | 公开 API，利率曲线分析，完全零覆盖 |
| 386-414 | **`futures_term_structure` 整个函数** | **A** | 公开 API，期货曲线分析，完全零覆盖 |

**建议新增测试 (在 `tests/test_finance.py`):**
- `test_drawdown_curve_all_nan_raises`
- `test_drawdown_curve_integer_index_underwater`
- `test_factor_attribution_ndarray_input`
- `test_factor_attribution_hac_mode`
- `test_var_ndarray_input`
- `test_var_empty_tail_warning`
- `test_nelson_siegel_fit_basic` — 3 tenor, 3 yields → 4 params
- `test_nelson_siegel_fit_monotone_yields` — 平坦曲线
- `@academic_reference test_nelson_siegel_svensson_fit` — Nelson & Siegel 1987 基准值
- `test_futures_term_structure_basic` — contango/backwardation 检测
- `test_futures_term_structure_empty_raises`

---

### 10. `oprim/statistics.py` (84%, ~31 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 51 | BCa 大 n 警告 | **B** | n>1000 才触发，性能警告 |
| 64 | bootstrap >50% NaN raises | **A** | 用户可触发（statistic_fn 总返回 NaN） |
| 98 | unknown bootstrap method raises | **A** | 验证分支 |
| 166-169 | percentiles=None 默认填充 | **B** | trivial default path |
| 222-225 | `nan_policy="omit"` 路径 | **A** | 三个函数均有此参数，omit 路径未测 |
| 275 | one_sample KS test | **A** | 单样本 KS，双样本有测试但单样本没有 |
| 324 | Mann-Kendall ties 校正 | **B** | 需要大量 tie 数据才触发 |
| 341, 348, 353 | Mann-Kendall z: s>0, s<0, s=0 | **A** | z-score 三分支，s=0 和 s<0 未测 |
| 408-411 | `bayesian_proportion_test` quantiles 默认 | **B** | trivial default |
| 493-494 | binless brier reliability/resolution | **A** | binless 分支未测 |
| 533-537 | `spearman_correlation`: nan_policy omit | **A** | omit 路径未测 |
| 610-617 | `correlation_matrix` 函数体 | **A** | 公开 API，完全零覆盖 |
| 652-670 | `rolling_quantile` 函数体 | **A** | 公开 API，完全零覆盖 |

**建议新增测试:**
- `test_bootstrap_ci_statistic_all_nan_raises`
- `test_bootstrap_ci_unknown_method_raises`
- `test_nan_policy_omit_removes_nan` — 适用于含 omit 的函数
- `test_mann_kendall_s_negative` — 下降序列
- `test_mann_kendall_s_zero` — 常数序列
- `test_one_sample_ks_test`
- `test_brier_score_binless`
- `test_correlation_matrix_pearson`
- `test_rolling_quantile_basic`

---

### 11. `oprim/time_series.py` (91%, ~21 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 185-191 | `forward_fill`: DataFrame 多列 strict check | **A** | DataFrame 输入路径 |
| 191-205 | `forward_fill`: Timedelta max_gap 转换 | **A** | pd.Timedelta 限制路径 |
| 214-238 | `_check_gap` 私有 helper | **C** | 私有函数，由 forward_fill 调用；间接覆盖足够 |
| 289-293 | `percentile_rank`: unknown method raises | **A** | 验证分支 |
| 300-329 | `_rolling_rank`, `_expanding_rank` 私有 | **B** | 由 percentile_rank 间接调用，需先覆盖 method 分支 |
| 445, 449 | `historical_volatility`: unknown estimator | **A** | 验证分支 |
| 523-539 | `detect_data_gaps` 主体分支 | **A** | 公开 API，edge branches: len<2, non-DatetimeIndex, expected_interval=None |
| 657, 661 | `purge_embargo_split` embargo_pct 验证 | **A** | embed_pct 边界验证，非 DatetimeIndex 转换 |

**建议新增测试:**
- `test_forward_fill_dataframe_input`
- `test_forward_fill_timedelta_max_gap`
- `test_percentile_rank_unknown_method_raises`
- `test_historical_volatility_unknown_estimator_raises`
- `test_detect_data_gaps_short_series`
- `test_detect_data_gaps_non_datetime_index`
- `test_purge_embargo_split_invalid_embargo_pct_raises`

---

### 12. `oprim/mean_reversion/ornstein_uhlenbeck.py` (85%, 6 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 65-66 | `ornstein_uhlenbeck_fit`: theta≤0 → NaN dict | **A** | 随机游走时 rho>0 但 theta≤0 |
| 123 | `ornstein_uhlenbeck_half_life` regression: var==0 → NaN | **A** | 常数序列 |
| 127 | regression: theta≤0 → inf | **A** | 正漂移序列（theta 为负） |
| 136, 139 | MLE: rho≤0 or theta≤0 → inf | **A** | 随机游走 / 正漂移 |

**建议新增测试:**
- `test_ou_fit_non_mean_reverting_returns_nan_theta` — 强趋势序列 (theta≤0)
- `test_ou_half_life_constant_series_regression` — 常数序列 var==0
- `test_ou_half_life_trending_series_returns_inf` — 单调递增序列

---

### 13. `oprim/volatility/garch.py` (95%, 2 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 29 | `_garch_nll`: sigma²≤0 → return 1e10 | **B** | 优化器内部数值保护，不可从用户侧直接触发 |
| 134 | `garch_fit`: persistence≥1 → unconditional_var=NaN | **A** | IGARCH 情形，alpha+beta≥1 时分母为零 |

**建议新增测试:**
- `test_garch_fit_unit_root_persistence_is_nan` — 强制返回 alpha+beta≥1 的参数，验证 unconditional_variance=NaN

---

### 14. `oprim/technical/*` (93-95%, 10 lines)

| 文件 | 行 | 内容 | 类别 |
|------|----|------|------|
| `bands.py` | 57, 127 | empty prices/highs raises | **A** |
| `exits.py` | 58 | empty closes raises | **A** |
| `moving_averages.py` | 148, 157 | empty prices / window>n raises | **A** |
| `moving_averages.py` | 168-172 | VWAP rolling loop + sv==0 guard | **A** |
| `moving_averages.py` | 230 | MACD empty prices raises | **A** |
| `oscillators.py` | 61 | RSI series too short → return nans | **A** |

**建议:** 每个验证函数补 1 个 `test_xxx_empty_input_raises` + 1 个边界测试；VWAP 补 `test_vwap_rolling_window_path`。

---

### 15. `oprim/distance.py` (90%, 16 lines)

| 行 | 内容 | 类别 | 理由 |
|----|------|------|------|
| 44, 46 | wasserstein 2D reshape | **A** | 2D 模式从未测试 |
| 97, 103-116 | DTW 2D 验证 + multivariate 两路径 | **A** | independent/dependent 两路均未测 |
| 177 | DTW `manhattan` 距离 | **A** | 仅测了 euclidean |
| 218, 224 | cosine/DB zero-vector warnings | **B** | 极端边界，警告路径 |
| 266, 270 | RBF kernel `Y=None` + 1D reshape | **A** | 自距离矩阵模式未测 |

**建议新增测试:**
- `test_wasserstein_2d_mode`
- `test_dtw_multivariate_independent`
- `test_dtw_multivariate_dependent`
- `test_dtw_manhattan_metric`
- `test_rbf_kernel_self_distance` — Y=None

---

### 16. 其他小缺口

| 文件 | 行 | 内容 | 类别 | 豁免方式 |
|------|----|------|------|----------|
| `regime.py` | 137, 140-141 | stationary dist. 归一化 / except 块 | **B/C** | except 块 `# pragma: no cover` |
| `risk/cvar.py` | 74 | empty tail fallback | **B** | `# pragma: no cover`（极端边界） |
| `crypto/merkle.py` | 135 | Merkle proof sn==0 break | **B** | `# pragma: no cover` |
| `derivatives/_base.py` | 8 | `_d1_d2` T≤0/sigma≤0 → None | **A** | 应通过 implied_volatility 内部测试触发 |

---

## 行动优先级

| 优先级 | 目标 | 覆盖率增量估算 |
|--------|------|---------------|
| P0 — 立即补 | `information.py` + `signal_processing.py` + `topology.py` (全部零覆盖公开 API) | +4.5 pp |
| P0 — 立即补 | `finance.py` 两个完整未覆函数 (`nelson_siegel_fit`, `futures_term_structure`) | +3.1 pp |
| P1 | `point_process.py` hawkes_nll 直接测试 | +0.9 pp |
| P1 | `statistics.py` 缺失分支 (correlation_matrix, rolling_quantile, omit paths) | +1.6 pp |
| P2 | Technical + distance.py 边界验证 | +1.3 pp |
| P2 | OU / GARCH / BS 边界 | +0.8 pp |
| B 豁免 | `_base.py`, `_manifest.py`, `compute_dwt`, 防御 guards | −2.5 pp 需求 |

**估算达标路径:** 补全 P0+P1 (~10.9 pp) + B 豁免 (−2.5 pp 需求) → 约 **93-94%**，接近 95% 目标。
