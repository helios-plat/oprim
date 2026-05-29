# Helios 元实现层最终清单

**目标**：列出 Helios 22 章 Spec 中所有**真正的元实现**——能独立完成一项任务、不依赖其它元实现的最小单元。

**清单状态**：最终版（在 v1 元实现层启动前，作为唯一权威清单）

**与前次提案的区别**：
- 严格按"元实现 = 能独立完成任务、不依赖其它元实现"定义筛选
- 移除所有 Layer 2 元 skill（依赖元实现的组合层）
- 移除所有优先级排序——按你的判断"都是必须的"
- 补扫遗漏，新增 7 个候选

---

## 一、元实现的严格定义（你定义的）

```yaml
atomic_ops_definition:

  inclusion:
    ✅ 能独立完成一项任务（一个具体的数学 / 算法操作）
    ✅ 仅依赖标准库（numpy / scipy / pandas 等）
    ✅ 不调用其它 helios.ops.* 元实现
    ✅ 在 Helios Spec 22 章中 ≥ 2 章使用

  exclusion:
    ❌ 内部需要调用其它 helios.ops.* → 这属于 Layer 2 元 skill, 不在本清单
    ❌ 业务逻辑专用（e.g. Fusion Score 公式）→ 属于业务模块
    ❌ 仅 1 章使用 → 不抽
    ❌ 完全是标准库的简单包装无加值 → 不抽

  positive_examples:
    ops.bootstrap_ci         独立完成 bootstrap CI 计算
    ops.zscore_normalize     独立完成 Z-score 标准化
    ops.wasserstein_distance 独立完成距离计算
    ops.regime_filter_data   独立完成 regime 过滤

  negative_examples_excluded:
    bootstrap_sharpe   依赖 bootstrap_ci → Layer 2 元 skill
    psr_dsr           依赖 bootstrap_ci + skew_kurt → Layer 2 元 skill
    walk_forward_split 依赖 purge_embargo_split → Layer 2 元 skill
    regime_aware_rolling 依赖 regime_filter_data → Layer 2 元 skill
```

---

## 二、最终元实现清单（共 31 个）

按功能分组，**组内无优先级**，**组间无优先级**——按你的判断"都是必须的，不用考虑先后"。

---

### 组 1：时间序列基础（11 个）

#### **ops.log_returns**
- 用途：从价格序列计算对数回报率（支持 1/5/20/60 多周期，含跨资产的 calendar 处理）
- 依赖：numpy / pandas
- Spec 引用：§4 HMM features（log_return_1d）/ §6 panel core fields / §13 portfolio returns / §17 OOS / §18.4 因子归因
- 踩坑：第一行 NaN 处理；跨周末 / 跨假期 gap；复权处理（dividend / split）；加密 24/7 vs 美股周末

#### **ops.realized_vol**
- 用途：实现波动率计算（5d / 20d / 60d 多窗口；可选 close-to-close / Garman-Klass / Parkinson）
- 依赖：numpy
- Spec 引用：§4 HMM observation features / §5.3 Volatility 子分数 / §6 panel field / §9.7 TCA
- 踩坑：annualization factor（√252 vs √365 vs √8760 加密 24/7）；Garman-Klass 需要 OHLC 全字段

#### **ops.zscore_normalize**
- 用途：Z-score 标准化（参数化的 mean / std 估计窗口：rolling / expanding / fixed）
- 依赖：numpy / pandas
- Spec 引用：§4 HMM volume_zscore / §5.3 子分数标准化 / §6 panel 字段标准化 / §7.2 cross_asset_corr Z / §15 alert threshold / §18.6 异常检测 / §19.8 content score
- 踩坑：rolling vs expanding（业务语义不同）；NaN 传播；极端值掩盖（标准差爆炸）

#### **ops.ewma_smooth**
- 用途：Exponentially Weighted Moving Average（参数化 half-life / span / alpha）
- 依赖：pandas.ewm 包装
- Spec 引用：§4 SVI online update / §7 DCC-GARCH conditional variance / §9.7 TCA decay analysis / Helixa（已用）
- 踩坑：初始化（cold start）；half-life vs alpha 的数学等价转换易写错

#### **ops.rolling_window_split**
- 用途：基础滚动窗口切分，输入 series + window_size + step → 输出 (start, end) 对列表
- 依赖：numpy / pandas
- Spec 引用：§4.4.2 SVI mini-batch 30 天滑动 / §6 panel 字段计算 / §8 因子 rolling / §17.2 OOS replay / §18.4 因子归因 rolling 252-day
- 踩坑：边界（数据头尾不足一窗时怎么办）；与 dataframe index 对齐；NaN 处理

#### **ops.purge_embargo_split**
- 用途：金融时序专用切分（避免 look-ahead bias 和 serial correlation leakage）
- 依赖：numpy / pandas
- Spec 引用：§9.2.1 CPCV / §17.2 OOS / §4 HMM walk-forward retrain / §8 因子验证
- 踩坑：embargo 长度（与 label horizon 一致）；跨 fold 时间重叠检测；按 López de Prado 2018 严格实现
- 学术参考：López de Prado 2018《Advances in Financial Machine Learning》第 7 章

#### **ops.gap_detect**
- 用途：时序数据 gap 检测（缺失日期 / 不等距 / 异常长 gap），返回 gap 列表 + 严重度分类
- 依赖：pandas
- Spec 引用：§14.5 Gap Detection A18（核心）/ §22.3 数据保留 monitoring / §6 panel field 完整性检查
- 踩坑：自适应预期间隔（日 / 小时 / 加密 24/7 不同）；holiday calendar；4 类分级（no_gap / short / medium / long）

#### **ops.resample_align**
- 用途：跨资产时间对齐（不同 trading hours / timezone / frequency 对齐到统一频率）
- 依赖：pandas.resample 基础上封装
- Spec 引用：§7 跨资产相关性（前置）/ §6 panel cross-sector / §14 multi-source feed alignment
- 踩坑：BTC 24/7 vs 美股 6.5h vs 黄金 23h 对齐；forward-fill vs interpolate；timezone 处理

#### **ops.lag_forward_fill**
- 用途：时序滞后 / 前向填充（含 max_gap 限制 + 业务规则）
- 依赖：pandas
- Spec 引用：§6 panel data 填充 / §13 持仓 mark-to-market / §14 实时数据 / §17 OOS / §22 数据保留 / §19 news timestamp
- 踩坑：max_gap 限制（避免无限填充）；不同业务规则（持仓用最后已知 vs panel 用计算窗口）

#### **ops.percentile_rank**  ⭐ **新增**
- 用途：批量计算每个数据点在历史窗口中的 percentile rank（用于因子构建 + 信号生成）
- 依赖：scipy.stats / numpy
- Spec 引用：§6 panel 字段 / §7 跨资产 / §8 因子 / §18.4 因子归因 / §19 news ranking
- 踩坑：rolling rank vs cross-sectional rank（语义不同）；ties 处理（average / min / max）

#### **ops.cumulative_returns**  ⭐ **新增**
- 用途：从 returns 序列计算累计 return / equity curve / drawdown 时序
- 依赖：numpy / pandas
- Spec 引用：§9 backtest equity curve / §13 持仓 P&L / §17 OOS forward-test
- 踩坑：log return 累计（求和）vs simple return 累计（连乘）；起始资金归一化

---

### 组 2：统计推断（10 个）

#### **ops.bootstrap_ci**
- 用途：非参数 bootstrap 置信区间（resample with replacement → 计算 percentile CI）
- 依赖：numpy
- Spec 引用：§9.4 DSR / §13.9 Position Regime Beta（明确写了 1000 次 bootstrap）/ §18.4 因子归因 / §20 SLA / §17 OOS Sharpe distribution
- 踩坑：n_bootstrap 默认值（1000 vs 5000）；percentile 法 vs BCa 法；paired bootstrap（成对样本）；矢量化性能（numpy 比 python loop 快 100x）

#### **ops.percentile_ci**
- 用途：从 sample 数组计算 percentile-based CI（5% / 95% / median / 任意 quantile）
- 依赖：numpy
- Spec 引用：§9 backtest path Sharpe distribution / §11 Scenario percentile / §13 Regime Beta / §17 analogy similarity / §18 valuation distribution
- 踩坑：interpolation method（linear / lower / higher / midpoint）；NaN 处理

#### **ops.distribution_summary**
- 用途：统一分布描述（mean / median / std / skew / kurt + 5/25/50/75/95 percentiles）
- 依赖：scipy.stats
- Spec 引用：§9 backtest output / §11 Scenario / §13 Regime Beta CI / §17 path distribution
- 踩坑：NaN 处理；scipy.stats.describe 不含 percentile；统一返回 schema

#### **ops.skew_kurt_robust**
- 用途：稳健 skewness + kurtosis 估计（含 Fisher-Pearson 校正）
- 依赖：scipy.stats
- Spec 引用：§9.3 PSR 调整公式（必须）/ §12 EVT λ-distribution / §11 Scenario distribution analysis / §4 HMM emission
- 踩坑：scipy.stats.skew / kurtosis 默认 bias=True，PSR 需要 bias=False（Fisher-Pearson）

#### **ops.kolmogorov_smirnov_test**
- 用途：KS test for distribution similarity（单样本 vs 双样本）
- 依赖：scipy.stats.kstest / ks_2samp
- Spec 引用：§12 EVT 阈值选择 4 方法投票 / §11 Scenario validation / §17 OOS distribution check
- 踩坑：scipy 接口分单 / 双样本，统一封装；ties 处理

#### **ops.mann_kendall_trend**
- 用途：Mann-Kendall 单调趋势检验（含 Hamed-Rao 自相关修正）
- 依赖：自实现（pymannkendall 包可选）
- Spec 引用：§9.8 Strategy Decay / §17 historical pattern / §18 ESG trend / §6 panel trend monitoring
- 踩坑：tied data 处理；Hamed-Rao 修正（自相关）

#### **ops.bayes_beta_update**
- 用途：Beta(α, β) posterior update with binary feedback（Thompson Sampling 用）
- 依赖：scipy.stats.beta
- Spec 引用：§15.3 Bandit / §16.6 Brier Score（间接）/ §18 estimation update / §19 source historical credibility update
- 踩坑：α / β prior 选择；hierarchical bandit 跨用户共享时的 prior 聚合规则

#### **ops.brier_score_decomposed**
- 用途：Brier Score Murphy 1973 三分量分解（reliability + resolution + uncertainty）
- 依赖：numpy
- Spec 引用：§15 Bandit feedback / §16.6 thesis Brier Score（核心）/ §17 BOCPD post-mortem / §19 prediction quality
- 踩坑：分箱（默认 10 bin）；binless 替代；sample size 不足时 reliability 项不稳
- 学术参考：Murphy 1973《A New Vector Partition of the Probability Score》

#### **ops.pearson_spearman_corr**  ⭐ **新增**
- 用途：相关性计算（Pearson + Spearman 双方法），含 p-value + 样本数充足性检查
- 依赖：scipy.stats
- Spec 引用：§7.2.1 Pearson / §7.2.2 Spearman / §6 panel 跨字段相关性 / §18 因子相关性 / §13 portfolio correlation
- 踩坑：scipy 不返回统一格式；p-value 大样本几乎都显著（要看 effect size）；NaN 处理

#### **ops.kde_density**  ⭐ **新增**
- 用途：核密度估计（用于分布平滑展示 + Wasserstein 距离前置）
- 依赖：scipy.stats.gaussian_kde
- Spec 引用：§17 类比检索 distribution comparison（前置）/ §11 Scenario distribution / §19 sentiment distribution
- 踩坑：bandwidth 选择（Silverman vs Scott）；多维 KDE 性能；边界效应

---

### 组 3：距离 / 相似度（5 个）

#### **ops.wasserstein_distance**
- 用途：Wasserstein 距离（1D 简化 + Sliced multi-D for high-dim）
- 依赖：scipy.stats.wasserstein_distance（1D 用），自实现 multi-D
- Spec 引用：§17.3.1 类比检索（核心）/ §11.x Scenario distribution comparison / §18 valuation distribution similarity / §19 NLP embedding cluster comparison
- 踩坑：1D（O(N log N)）vs Sliced multi-D（O(N²) 但可并行）；scipy.stats.wasserstein_distance 仅支持 1D
- 学术参考：Bonneel et al. 2015《Sliced and Radon Wasserstein Barycenters》

#### **ops.dtw_distance**
- 用途：Dynamic Time Warping 距离（含 Sakoe-Chiba band 约束）
- 依赖：自实现 + dtaidistance 包可选
- Spec 引用：§17.3.2 类比检索（核心）/ §11 historical scenario matching / §18 stock price pattern matching
- 踩坑：O(n×m) DP 复杂度；Sakoe-Chiba band 加速；multi-variate DTW（独立 vs dependent）
- 学术参考：Berndt-Clifford 1994 / Sakoe-Chiba 1978

#### **ops.cosine_similarity_batch**
- 用途：批量余弦相似度（embedding × embedding matrix）
- 依赖：sklearn.metrics.pairwise / scipy
- Spec 引用：§17 Wasserstein 互补 / §18.5 智能同业检索 / §19.5 Event Cascade DBSCAN / §16 watchlist analogy
- 踩坑：normalize 是否预先做；大矩阵内存（batch processing）；FAISS / hnswlib 替代

#### **ops.euclidean_distance_matrix**  ⭐ **新增**
- 用途：批量欧氏距离矩阵（含 weighted Euclidean + 跨 panel 的 distance matrix）
- 依赖：scipy.spatial.distance
- Spec 引用：§17 panel signature 距离（前置）/ §6 panel cross-asset / §11 Scenario clustering / §13 portfolio similarity
- 踩坑：内存（N×N 矩阵爆炸）；weight 处理；scipy.spatial vs sklearn 接口不一致

#### **ops.symmetric_kl_divergence**
- 用途：Jensen-Shannon divergence / symmetric KL（用于分布对比）
- 依赖：scipy.special.rel_entr
- Spec 引用：§19 NLP cluster comparison / §11 distribution shift detection / §17 distribution divergence
- 踩坑：分母 0 处理；smoothing；对数底（log2 vs ln）

---

### 组 4：数值稳定性（3 个）

#### **ops.logsumexp_safe**
- 用途：log(sum(exp(x))) 数值稳定计算
- 依赖：scipy.special.logsumexp
- Spec 引用：§4.5 BOCPD log space recursion（核心）/ §4.4 SVI / §15 bandit posterior（潜在）/ §19 NLP log-prob aggregation
- 踩坑：scipy.special.logsumexp 已稳定，但 BOCPD 用法需要 axis 参数 + weight 参数

#### **ops.softmax_safe**
- 用途：数值稳定 softmax（先减 max 再 exp）
- 依赖：scipy.special.softmax
- Spec 引用：§5 Layer 1 子分数加权（潜在）/ §15 bandit policy / §19 ranking / §18 attribution weight
- 踩坑：基础实现就有；多维 axis 参数；temperature 参数

#### **ops.clip_with_warning**  ⭐ **改名**
- 用途：clip + 溢出告警（用于 Fusion Score Layer 1 ≤ 0.6 / Layer 2 ≤ 0.4 / 每项 ≤ 0.05 等业务约束）
- 依赖：numpy
- Spec 引用：§5 Fusion Score（核心约束）/ §9 backtest position size / §15 alert threshold / §13 Regime Beta CI bound / §11 Scenario value bounds
- 踩坑：np.clip 不告警；告警频率控制（避免日志爆炸）；返回值 vs raise

---

### 组 5：Regime 联动（3 个）

#### **ops.regime_filter_data**
- 用途：根据 regime label 序列过滤数据子集（per-regime statistics 必须前置）
- 依赖：pandas
- Spec 引用：§5.2 Layer 0 veto / §13.9 Position Regime Beta（核心）/ §17 historical regime replay / §18.3.2 regime-conditional valuation / §11 regime-specific scenarios
- 踩坑：regime label 与数据 index 对齐；soft probability vs hard label；跨 regime 转移时 boundary 处理

#### **ops.regime_transition_matrix**
- 用途：从 regime label 序列估计转移矩阵 + duration 分布
- 依赖：numpy
- Spec 引用：§4 HMM verification（HSMM duration distribution 对照）/ §17 historical analysis / §11 scenario calibration
- 踩坑：sticky-bias 检测；边界规则；马尔可夫性检验

#### **ops.regime_label_align**
- 用途：跨频率 regime label 对齐（daily HMM regime → 5min tick 数据 → 每条 tick 的 regime）
- 依赖：pandas.merge_asof
- Spec 引用：§13 持仓 tick × regime / §14 实时数据 regime annotation / §17 OOS replay / §15 alert with regime context
- 踩坑：as-of join；forward-fill regime 边界；timezone

---

### 组 6：金融指标（4 个，全是新增的）

#### **ops.drawdown_curve**  ⭐ **新增**
- 用途：从 equity curve 或 returns 计算 drawdown 时序 + max drawdown + drawdown duration
- 依赖：numpy / pandas
- Spec 引用：§5 Veto trigger（drawdown threshold）/ §9 backtest TCA / §11 Scenario stress / §13 portfolio risk / §17 forward-test / §15 alert / §18 stock max drawdown
- 踩坑：peak detection；underwater duration；recovery time

#### **ops.sharpe_ratio**  ⭐ **新增**
- 用途：Sharpe ratio 计算（含 risk-free rate 调整 + annualization）
- 依赖：numpy
- Spec 引用：§9 backtest（核心）/ §11 Scenario / §13 portfolio / §17 OOS / §6 panel field / §18 stock 评估
- 踩坑：annualization factor（trading days 252 vs calendar 365）；risk-free rate 时序处理；负 Sharpe 含义
- 注意：与 `ops.bootstrap_sharpe`（Layer 2）区分——这是 atomic Sharpe 计算

#### **ops.beta_alpha_ols**  ⭐ **新增**
- 用途：CAPM 风格 OLS 回归（输出 α + β + R² + std error），用于 Position Regime Beta + 因子归因
- 依赖：statsmodels.OLS / scipy.stats.linregress
- Spec 引用：§13.9 Position Regime Beta（核心）/ §18.4 Fama-French 因子归因 / §17 portfolio alpha 分析
- 踩坑：单因子 vs 多因子；HAC standard error（Newey-West）；样本数最低要求

#### **ops.value_at_risk**  ⭐ **新增**
- 用途：VaR 计算（historical / parametric / Cornish-Fisher 三方法）+ 对应 ES
- 依赖：numpy / scipy
- Spec 引用：§12 EVT（基础对照）/ §13 portfolio VaR / §11 Scenario VaR / §9 backtest TCA risk
- 踩坑：confidence level 边界；historical VaR 样本依赖；Cornish-Fisher 极端尾部失效
- 注意：完整 EVT-VaR（GARCH-EVT + GPD）属于 Layer 2 元 skill，这里仅 atomic VaR

---

## 三、最终汇总

```yaml
total_atomic_ops: 31

distribution_by_group:
  组 1 时间序列基础: 11 个
  组 2 统计推断:     10 个
  组 3 距离 / 相似度: 5 个
  组 4 数值稳定性:    3 个
  组 5 Regime 联动:   3 个
  组 6 金融指标:      4 个（新增组，主要是补扫遗漏）

new_in_this_revision:
  - ops.percentile_rank      （组 1）
  - ops.cumulative_returns   （组 1）
  - ops.pearson_spearman_corr （组 2）
  - ops.kde_density          （组 2）
  - ops.euclidean_distance_matrix （组 3）
  - ops.drawdown_curve       （组 6 新组）
  - ops.sharpe_ratio         （组 6 新组）
  - ops.beta_alpha_ols       （组 6 新组）
  - ops.value_at_risk        （组 6 新组）

removed_from_previous_proposal (因为是 Layer 2 元 skill, 不是 atomic):
  - bootstrap_sharpe         → 依赖 bootstrap_ci
  - psr_dsr                  → 依赖 bootstrap_ci + skew_kurt
  - walk_forward_split       → 依赖 purge_embargo_split
  - regime_aware_rolling     → 依赖 regime_filter_data + rolling_window_split
  - detect_outliers          → 依赖 zscore_normalize
  - bootstrap_ci 之上的所有组合方法
  → 这些将作为 Layer 2 元 skill, 在元实现层 1.0 完工后单独立 ADR

removed_as_too_trivial:
  - ops.safe_divide          → numpy.divide(where=...) 已够用
  - ops.expanding_window_apply → pandas.expanding 已够用
  - ops.rolling_apply_parallel → 性能优化，等真有需要再做
  - ops.dataframe_to_typed_dict → Pydantic 已够用
  - ops.timeindex_normalize    → pandas tz 已够用
  - ops.align_to_business_calendar → pandas-market-calendars 包装即可
  - ops.hausdorff_distance     → scipy 已够用且用得少
```

---

## 四、与 22 章的调用关系

```yaml
chapter_to_atomic_ops_call_graph:

  §4 HMM 调用:
    ops.zscore_normalize        # volume_zscore feature
    ops.log_returns             # log_return_1d feature
    ops.realized_vol            # 5d / 20d / 60d
    ops.skew_kurt_robust        # emission distribution
    ops.logsumexp_safe          # forward-backward + BOCPD
    ops.purge_embargo_split     # walk-forward retrain
    ops.rolling_window_split    # SVI mini-batch

  §5 Fusion Score 调用:
    ops.regime_filter_data      # Layer 0 veto data filter
    ops.zscore_normalize        # Layer 1 子分数标准化
    ops.clip_with_warning       # Layer 1/2 budget enforcement
    ops.drawdown_curve          # veto trigger threshold

  §6 panel 调用:
    ops.zscore_normalize        # 字段标准化
    ops.realized_vol            # vol fields
    ops.log_returns             # return fields
    ops.gap_detect              # 数据完整性
    ops.lag_forward_fill        # 数据填充
    ops.percentile_rank         # rank-based fields
    ops.pearson_spearman_corr   # cross-field correlations

  §7 跨资产 调用:
    ops.pearson_spearman_corr   # 8 方法基础（Pearson + Spearman）
    ops.resample_align          # 跨资产对齐
    ops.cosine_similarity_batch # 高维相似度
    ops.regime_filter_data      # regime-conditional correlation

  §8 自定义因子 调用:
    ops.bootstrap_ci            # 因子稳定性
    ops.zscore_normalize        # 因子标准化
    ops.percentile_rank         # 因子分位
    ops.purge_embargo_split     # 因子验证
    ops.rolling_window_split    # 因子 rolling

  §9 Backtest 调用:
    ops.purge_embargo_split     # CPCV
    ops.bootstrap_ci            # CI 估计
    ops.skew_kurt_robust        # PSR 调整
    ops.kolmogorov_smirnov_test # 残差检验
    ops.mann_kendall_trend      # decay
    ops.sharpe_ratio            # Sharpe 计算
    ops.drawdown_curve          # max drawdown
    ops.cumulative_returns      # equity curve
    ops.distribution_summary    # path distribution

  §11 Scenario 调用:
    ops.percentile_ci           # scenario percentile
    ops.distribution_summary    # scenario stats
    ops.wasserstein_distance    # scenario comparison
    ops.kde_density             # smoothed distribution
    ops.value_at_risk           # scenario VaR
    ops.kolmogorov_smirnov_test # scenario validity

  §12 EVT 调用:
    ops.kolmogorov_smirnov_test # 阈值方法投票
    ops.skew_kurt_robust        # λ-dist
    ops.value_at_risk           # 基础 VaR

  §13 持仓 调用:
    ops.bootstrap_ci            # Regime Beta CI
    ops.regime_filter_data      # per-regime data
    ops.beta_alpha_ols          # Position Regime Beta
    ops.lag_forward_fill        # mark-to-market
    ops.regime_label_align      # tick × regime
    ops.cumulative_returns      # P&L
    ops.drawdown_curve          # portfolio drawdown
    ops.value_at_risk           # portfolio VaR
    ops.sharpe_ratio            # portfolio Sharpe

  §14 实时数据 调用:
    ops.gap_detect              # Gap Detection A18
    ops.regime_label_align      # tick annotation
    ops.lag_forward_fill        # missing data

  §15 Alerts 调用:
    ops.bayes_beta_update       # Bandit
    ops.zscore_normalize        # alert threshold
    ops.regime_filter_data      # regime-conditional alerts

  §16 Watchlist 调用:
    ops.brier_score_decomposed  # thesis evaluation
    ops.cosine_similarity_batch # similar assets

  §17 Regime Replay 调用:
    ops.wasserstein_distance    # 类比检索
    ops.dtw_distance            # 类比检索
    ops.regime_filter_data      # OOS regime
    ops.regime_transition_matrix # historical regime stats
    ops.euclidean_distance_matrix # panel signature distance
    ops.bootstrap_ci            # OOS Sharpe path CI
    ops.cosine_similarity_batch # embedding similarity
    ops.kde_density             # distribution comparison

  §18 Stock Intel 调用:
    ops.bootstrap_ci            # 因子归因 CI
    ops.beta_alpha_ols          # Fama-French OLS
    ops.regime_filter_data      # regime-conditional valuation
    ops.percentile_rank         # stock ranking
    ops.dtw_distance            # price pattern matching
    ops.cumulative_returns      # stock returns
    ops.sharpe_ratio            # stock Sharpe

  §19 News NLP 调用:
    ops.cosine_similarity_batch # Event Cascade DBSCAN
    ops.zscore_normalize        # content score
    ops.bayes_beta_update       # source historical credibility
    ops.percentile_rank         # news ranking
    ops.symmetric_kl_divergence # cluster comparison
```

**调用强度统计**：

```yaml
most_called_atomic_ops:

  ≥ 7 章使用:
    ops.zscore_normalize        # 7 章
    ops.regime_filter_data      # 7 章
    ops.bootstrap_ci            # 7 章

  ≥ 5 章使用:
    ops.cumulative_returns
    ops.sharpe_ratio
    ops.drawdown_curve
    ops.cosine_similarity_batch
    ops.distribution_summary
    ops.percentile_rank
    ops.lag_forward_fill

  共 31 个 ops, 平均每个被 3-4 章调用
```

---

## 五、执行规范（一旦启动元实现层 1.0）

### 5.1 模块结构

```
helios/
  ops/                            # 元实现层
    __init__.py                   # 显式 export 全部 31 个 op
    time_series.py                # 组 1: 11 个
    statistics.py                 # 组 2: 10 个
    distance.py                   # 组 3: 5 个
    numerics.py                   # 组 4: 3 个
    regime.py                     # 组 5: 3 个
    finance.py                    # 组 6: 4 个
    _base.py                      # 共用 base class + type hints
  tests/
    ops/
      test_time_series.py
      test_statistics.py
      test_distance.py
      test_numerics.py
      test_regime.py
      test_finance.py
  docs/
    ops/
      INDEX.md                    # 31 个 op 索引
      time_series.md
      ...
```

### 5.2 每个 op 的实施标准

```yaml
per_op_implementation_standard:

  code_quality:
    - 完整 type hints (Pydantic 或 dataclass)
    - docstring 含数学定义 + 用法示例 ≥ 3 个
    - 无副作用 (pure functions for stateless ops)
    - 边界 case 显式处理 + 文档化
    - NaN / inf / 空数据策略明确

  test_coverage:
    - 单元测试覆盖 ≥ 90%
    - happy path + 边界 + 异常 各 ≥ 3 用例
    - 与已知库对比验证 (e.g. ops.bootstrap_ci 与 scipy.stats.bootstrap)
    - 性能基准 (pytest-benchmark)

  documentation:
    每个 op 一份 markdown:
      用途 (一句话)
      数学定义 (公式)
      参考文献 (如有)
      API 签名 + 类型
      用法示例 ≥ 3 个
      边界 case 说明
      已知限制
      性能特征

  versioning:
    helios.ops 独立 semantic versioning
    breaking changes 在 deprecation period 后才生效
```

### 5.3 实施工作量预估

```yaml
work_estimate:

  per_op:
    简单 op (e.g. log_returns / cumulative_returns / sharpe_ratio):    1 天
    中等 op (e.g. bootstrap_ci / brier_score_decomposed / dtw):         2 天
    复杂 op (e.g. wasserstein multi-D / purge_embargo_split):          3 天

  31 op 总计:
    简单 op (~12 个) × 1 天 = 12 天
    中等 op (~15 个) × 2 天 = 30 天
    复杂 op (~4 个) × 3 天  = 12 天
    总计: 54 天 = 约 11 周（单人全职）

  并行化降低周期:
    多个 op 之间相互独立 → 可并行
    Wiki + 1-2 CC instance 并行: 5-7 周

  含测试 + 文档 + review:
    实际周期 8-10 周

  注意: 每个 op 是一个 PR, 不要一次合并多个 op
```

### 5.4 与业务模块的关系

```yaml
relationship_with_22_chapters:

  独立性:
    helios.ops 完全独立, 不依赖任何 22 章业务模块
    可以独立 ship 一个 helios.ops 1.0 release

  调用方:
    22 章业务模块 import helios.ops
    现有 BTC 板块代码可以选择 retrofit (新功能必须用 helios.ops)
    新 8 板块直接调用 helios.ops

  retrofit_strategy:
    BTC 板块 retrofit 不强制
    优先级低于"新功能开发"
    在 BTC 板块需要重构 / 修 bug 时顺便迁移
```

---

## 六、Layer 2 元 skill（不在本清单，未来 ADR）

```yaml
future_layer_2_meta_skills:

  status: 不在本次元实现层 1.0 范围
  trigger: 元实现层 1.0 完工后, 立独立 ADR (假设 ADR-062)

  candidates_already_identified:
    - bootstrap_sharpe          = bootstrap_ci + sharpe_ratio
    - psr_dsr                   = bootstrap_ci + skew_kurt_robust + 自定义 N_eff 估计
    - walk_forward_split        = purge_embargo_split 反复调用
    - regime_aware_rolling      = regime_filter_data + rolling_window_split
    - detect_outliers           = zscore_normalize + 阈值规则
    - cpcv_full_pipeline        = purge_embargo_split + bootstrap_ci + 路径重组
    - garch_evt_pipeline        = arch.GARCH + 残差提取 + GPD fit
    - factor_attribution_full   = beta_alpha_ols + bootstrap_ci + factor library
    - 其它 ...

  这些都是元实现的组合, 等元实现层 1.0 稳定后再做
```

---

## 七、清单状态

```yaml
status: FINAL
classification: 真元实现完整清单
total_count: 31
confidence: 高 (基于 22 章 spec 文本系统扫描 + 9 个关键词类别全扫)

approval_required:
  Wiki 确认本清单 → 开始 ADR-061 (Helios atomic ops library 1.0)
  Wiki 不确认 → 列出疑问, 修订清单

next_steps_after_approval:
  1. 写 ADR-061 (含本清单 + 实施标准 + 工作量分解)
  2. 写 31 个 op 的 CC 实施 prompt (每个 op 一个独立 prompt)
  3. CC 自主跑 (FULL AUTO 模式, 每个 op 一个 PR)
  4. Wiki review PR + merge
  5. 持续 8-10 周完成全部 31 op
```
