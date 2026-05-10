# Helios Atomic Ops 实施 CC Prompt Pack

**用途**：本文档包含 1 份**通用模板** + 31 份**op 专用 spec**。把模板与某 op 的 spec 拼起来，复制粘贴给 CC instance 跑。

**使用方法**：
```
[GENERIC_TEMPLATE 整段] + [SPEC: ops.<具体 op 名>] = 完整 prompt
```

**并行执行**：
- 5 CC instance 推荐
- 每个 instance 跑 1 个 lane (1 lane = 1 组)
- 每个 instance 完成自己 lane 后等待 Wiki review
- 不同 instance 之间 0 依赖

---

# Part 1: GENERIC_TEMPLATE（通用模板，所有 op 共用）

```
======================================================================
HELIOS ATOMIC OPS 实施任务

FULL AUTO MODE
- 不要中途问问题
- 自行决策推进
- 失败才停下来报告
- 完成后只汇报一次
======================================================================

# 你的角色
你是 Helios 量化决策辅助平台的工程师, 负责实施 helios.ops 元实现层
(详见 ADR-061).

# 项目位置
~/projects/helios/
仓库: helios-plat/helios

# 红线 (must not violate)
1. 你实施的 op 内部 禁止 import 任何其它 helios.ops.* 模块
   (违反就是把 atomic op 退化成 Layer 2 元 skill)
2. 仅允许依赖: numpy, scipy, pandas, sklearn, pydantic
   其它依赖必须 ADR 中声明（e.g. dtaidistance, statsmodels）
3. 每个 op 一个 PR, 不要批量 merge
4. 测试覆盖率 ≥ 90%, 否则 reject
5. 必须含 学术对照测试 (与 scipy/statsmodels 等参考库对比)
6. 不要修改其它 op 文件, 仅写自己负责的 op
7. 不要修改业务模块 (§4 ~ §22 章 业务代码)

# 工程标准

## 包结构
~/projects/helios/helios/
  ops/
    __init__.py             # 完成时在这里 export 你的 op
    time_series.py          # 组 1 ops 写在这里
    statistics.py           # 组 2
    distance.py             # 组 3
    numerics.py             # 组 4
    regime.py               # 组 5
    finance.py              # 组 6
    _base.py                # 共用 base class (可读不可改, 已写好)
    _validation.py          # 共用输入验证 (可读不可改)
  tests/
    ops/
      test_<your_module>.py # 在这里写你的 op 的测试
      conftest.py           # fixtures
  docs/
    ops/
      <your_op>.md          # 你的 op 文档

## API 输入输出
- 用 Pydantic 做输入参数 schema (除非性能临界, prompt 中会明示)
- 输出统一用 dict 或 Pydantic model
- type hints 完整 (mypy strict 通过)

## 测试要求
- 单元测试覆盖 ≥ 90%
- 必须含 ≥ 3 个 happy path 用例
- 必须含 ≥ 3 个 边界 case 用例 (空 / 单元素 / NaN / extreme values)
- 必须含 ≥ 2 个 异常 case 用例 (错误输入 / 类型不匹配)
- 必须含 ≥ 1 个 学术对照测试 (与参考库对比, 数值容差 rtol=1e-6)
- 必须含 ≥ 1 个 性能基准 (pytest-benchmark, 测试声明的性能目标)

## 文档要求
docs/ops/<your_op>.md 必须包含:
- 用途 (一句话)
- 数学定义 (LaTeX 公式)
- 学术参考 (如有)
- API 签名 + 类型
- 用法示例 ≥ 3 个
- 边界 case 说明
- 已知限制
- 性能特征 (Big-O)

## PR 标准
- PR 标题: feat(ops): add ops.<op_name>
- PR description 必须含:
  * 数学正确性证明 (test 输出截图 / 数据)
  * 性能基准结果 (pytest-benchmark 输出)
  * 边界 case 覆盖说明
- 单 PR 仅含一个 op + 它的测试 + 它的文档

# 任务流程

## Step 1: 确认环境
- cat ~/projects/helios/helios/ops/_base.py (读 base class)
- cat ~/projects/helios/helios/ops/_validation.py (读 validation utils)
- 如果文件不存在, 先创建包骨架

## Step 2: 读你的 op spec
- 在本任务的 [SPEC] 部分有完整规范
- 仔细读 用途 / API / 数学 / 边界 case / 学术参考

## Step 3: 实施
- 在指定 module 文件中添加你的 op
- 严格按 [SPEC] 中的 API 签名
- 严格遵守 红线

## Step 4: 测试
- 在 tests/ops/ 中写测试
- 覆盖率检查: pytest --cov=helios.ops.<module> --cov-fail-under=90
- 学术对照测试必须通过

## Step 5: 文档
- 在 docs/ops/ 中写文档
- 必须包含上面列出的所有部分

## Step 6: 性能基准
- benchmarks/ops/bench_<op_name>.py
- 测试 [SPEC] 中声明的性能目标

## Step 7: 提交 PR
- git checkout -b feat/ops-<op_name>
- 单 PR 仅含: ops/<module>.py 改动 + tests/ + docs/ + benchmarks/
- PR description 按上面 PR 标准写
- 不要 merge, 等 Wiki review

## Step 8: 报告
完成后输出一行总结到 stdout:
"ops.<op_name> 实施完成, PR #<num> 待 review"

# 失败处理
- 学术对照测试失败 → 检查数学实现, 不要靠调容差混过
- 测试覆盖率不达标 → 补测试, 不要降标准
- 性能不达标 → 优化或在 PR description 中说明 + flag
- 任何不确定 → 在 PR description 中 flag, 不要自己拍脑袋决策

======================================================================
[SPEC]
[此处粘贴你负责的 op 的 spec, 见 Part 2]
======================================================================
```

---

# Part 2: 31 个 op 的专用 spec

每段以 `### SPEC: ops.<name>` 开头，把这段拼到通用模板的 `[SPEC]` 位置。

---

## 组 1：时间序列基础（11 个）

### SPEC: ops.log_returns

```yaml
op_name: ops.log_returns
module: helios.ops.time_series
group: 时间序列基础
estimated_days: 1
size: simple

purpose: 从价格序列计算对数回报率, 支持多周期 + gap 处理

api:
  signature: |
    def log_returns(
        prices: pd.Series,
        periods: list[int] = [1, 5, 20, 60],
        handle_gaps: Literal["skip", "interpolate", "raise"] = "skip"
    ) -> pd.DataFrame
  
  output_schema: |
    返回 DataFrame, columns = [f"log_ret_{p}d" for p in periods]
    索引与 prices 对齐, 头部不足 period 行 = NaN

math:
  r_t = log(P_t / P_{t-period})
  对数回报率 = ln(P_t) - ln(P_{t-period})

edge_cases:
  - 第 1 行至 period 行: NaN
  - prices 含 NaN: handle_gaps 决定 (skip = 跳过, interpolate = 线性插值, raise = 抛错)
  - prices 含 0 或负数: raise ValueError ("Prices must be positive")
  - prices 长度 < max(periods): warning + 全 NaN

validation_against:
  reference: 自实现 + manual ground truth
  test_data: |
    prices = [100, 102, 99, 105, 110]
    expected log_ret_1d = [NaN, ln(102/100), ln(99/102), ln(105/99), ln(110/105)]

performance_target:
  n=10000, periods=4: < 1ms
  benchmark with pytest-benchmark

spec_references_in_helios:
  - §4.2.1 HMM observation features (log_return_1d)
  - §6 panel core fields
  - §13 portfolio returns
  - §17 OOS replay
  - §18.4 因子归因 returns

required_tests:
  happy_path:
    - 简单递增价格序列
    - 简单递减价格序列
    - 真实 BTC 历史样本 (2020-01 to 2024-12)
  
  edge_cases:
    - 单元素 prices (输出全 NaN)
    - 含 NaN prices, handle_gaps="skip"
    - 含 NaN prices, handle_gaps="interpolate"
    - 全 NaN prices
    - 长度 < max(periods)
  
  exceptions:
    - prices 含 0 (raise ValueError)
    - prices 含 负数 (raise ValueError)
    - periods 含 0 或负数 (raise ValueError)
    - periods 为空 list (raise ValueError)
  
  academic_validation:
    与 numpy.diff(np.log(prices)) 对比 1d log returns
    rtol=1e-9
```

---

### SPEC: ops.realized_vol

```yaml
op_name: ops.realized_vol
module: helios.ops.time_series
group: 时间序列基础
estimated_days: 2
size: medium

purpose: 实现波动率多 estimator 多窗口

api:
  signature: |
    def realized_vol(
        returns: pd.Series,
        window: int = 20,
        estimator: Literal["close_to_close", "garman_klass", "parkinson"] = "close_to_close",
        annualization_factor: int = 252,
        ohlc: pd.DataFrame | None = None
    ) -> pd.Series
  
  output_schema: |
    返回 pd.Series, 索引与 returns / ohlc 对齐
    头部 window-1 行 = NaN

math:
  close_to_close: σ = std(r) × √annualization_factor
  garman_klass: 见 Garman-Klass 1980
    σ² = 0.5 × (ln(H/L))² - (2 ln 2 - 1) × (ln(C/O))²
  parkinson: 见 Parkinson 1980
    σ² = (1 / (4 × ln 2)) × (ln(H/L))²

edge_cases:
  - estimator != "close_to_close" 时 ohlc 必须提供, 否则 raise
  - annualization 跨资产: equity=252, crypto=365, crypto_hourly=8760
  - window < 2 raise
  - returns 全 NaN 返回全 NaN series

validation_against:
  reference: 
    - close_to_close: numpy std × √annualization
    - garman_klass: 论文公式自构 ground truth
    - parkinson: 论文公式自构 ground truth

performance_target:
  n=10000, window=20: < 5ms

spec_references:
  - §4.2.1 HMM observation features (realized_volatility 5d/20d/60d)
  - §5.3.x Volatility 子分数
  - §6 panel field
  - §9.7 TCA

required_tests:
  happy_path:
    - close_to_close on synthetic returns (annualized)
    - garman_klass on synthetic OHLC
    - parkinson on synthetic OHLC
  
  edge_cases:
    - window=2 (最小窗口)
    - annualization_factor=8760 (crypto hourly)
    - returns 含 NaN
    - ohlc 缺失列
  
  exceptions:
    - estimator="garman_klass" + ohlc=None (raise)
    - window < 2 (raise)
    - annualization_factor <= 0 (raise)
  
  academic_validation:
    close_to_close 与 numpy.std(returns) × √252 对比
    garman_klass 与 论文 example data 对比
```

---

### SPEC: ops.zscore_normalize

```yaml
op_name: ops.zscore_normalize
module: helios.ops.time_series
group: 时间序列基础
estimated_days: 2
size: medium

purpose: Z-score 标准化, 参数化窗口策略

api:
  signature: |
    def zscore_normalize(
        data: pd.Series | pd.DataFrame,
        window: int | None = None,
        min_periods: int = 20,
        clip_extreme: float | None = 5.0
    ) -> pd.Series | pd.DataFrame
  
  output_schema: |
    返回与输入相同类型 (Series / DataFrame), 同索引

math:
  rolling: z_t = (x_t - μ_t) / σ_t, 其中 μ_t, σ_t 是 rolling window 计算
  expanding: z_t = (x_t - μ_t) / σ_t, μ_t, σ_t 是 expanding window
  其中 window=None 表示 expanding

edge_cases:
  - window < min_periods (warning)
  - σ → 0 时 NaN
  - clip_extreme=None 不限制
  - DataFrame 时各 column 独立标准化

performance_target:
  n=10000, rolling window=60: < 5ms

spec_references:
  - §4.2.1 HMM volume_zscore feature
  - §5.3 子分数标准化
  - §6 panel 字段标准化
  - §7.2 cross_asset_corr Z
  - §15 alert threshold
  - §18.6 异常检测
  - §19.8 content score

required_tests:
  happy_path:
    - rolling window=20 on synthetic data
    - expanding (window=None) on synthetic data
    - DataFrame 多 column
  
  edge_cases:
    - window=min_periods=2 (最小)
    - σ → 0 (常数序列)
    - clip_extreme=None
    - 含 NaN
  
  academic_validation:
    与 scipy.stats.zscore 对比 (no rolling case)
```

---

### SPEC: ops.ewma_smooth

```yaml
op_name: ops.ewma_smooth
module: helios.ops.time_series
estimated_days: 1
size: simple

purpose: Exponentially Weighted Moving Average

api:
  signature: |
    def ewma_smooth(
        data: pd.Series,
        half_life: float | None = None,
        span: int | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        ignore_na: bool = False
    ) -> pd.Series

math: y_t = α × x_t + (1 - α) × y_{t-1}
math_conversions:
  α from half_life: α = 1 - exp(ln(0.5) / half_life)
  α from span: α = 2 / (span + 1)

edge_cases:
  - half_life / span / alpha 互斥 (恰好一个不为 None, 否则 raise)
  - cold start (前几期 estimator 不稳)
  - data 全 NaN 返回全 NaN

performance_target:
  n=10000: < 5ms

spec_references:
  - §4.4 SVI online update
  - §7 DCC-GARCH conditional variance
  - §9.7 TCA decay analysis

required_tests:
  happy_path: half_life / span / alpha 三种参数化
  edge_cases: 
    - 同时指定多个参数 (raise)
    - 都不指定 (raise)
    - half_life=0 (raise)
  academic_validation:
    与 pandas.Series.ewm 对比, rtol=1e-9
```

---

### SPEC: ops.rolling_window_split

```yaml
op_name: ops.rolling_window_split
module: helios.ops.time_series
estimated_days: 1
size: simple

purpose: 基础滚动窗口切分, 输出 (start, end) tuple list

api:
  signature: |
    def rolling_window_split(
        n_samples: int,
        window_size: int,
        step: int = 1,
        include_partial: bool = False
    ) -> list[tuple[int, int]]
  
  output: list of (start_index, end_index) tuples, end_index inclusive

edge_cases:
  - n_samples < window_size 时 include_partial=False → 返回空 list
  - n_samples < window_size 时 include_partial=True → 返回单个 (0, n_samples-1)
  - step > window_size → 不重叠
  - step < 1 raise

performance_target:
  n_samples=100000, window_size=252, step=21: < 10ms

spec_references:
  - §4.4.2 SVI mini-batch 30 天滑动
  - §6 panel 字段计算
  - §8 因子 rolling
  - §17.2 OOS replay
  - §18.4 因子归因 rolling 252-day

required_tests:
  happy_path:
    - 标准 rolling (window=20, step=1)
    - 步进 rolling (window=20, step=5)
    - 不重叠 rolling (window=20, step=20)
  
  edge_cases:
    - n_samples = window_size (exactly one window)
    - n_samples < window_size, include_partial=False (empty)
    - include_partial=True
  
  exceptions:
    - window_size < 1 (raise)
    - step < 1 (raise)
    - n_samples < 0 (raise)
```

---

### SPEC: ops.purge_embargo_split

```yaml
op_name: ops.purge_embargo_split
module: helios.ops.time_series
estimated_days: 3
size: complex

purpose: |
  金融时序专用切分, 用于 CPCV / WFO. 避免 look-ahead bias 和
  serial correlation leakage. 严格按 López de Prado 2018 第 7 章实现.

api:
  signature: |
    def purge_embargo_split(
        times: pd.DatetimeIndex,
        n_splits: int,
        embargo_pct: float = 0.01,
        label_horizon: int | pd.Timedelta = 0
    ) -> list[dict[str, np.ndarray]]
  
  output: |
    list[{
      "train": np.ndarray (training indices),
      "test": np.ndarray (testing indices),
      "embargo": np.ndarray (embargo indices, excluded from train)
    }]

math_reference: |
  López de Prado, M. (2018). 《Advances in Financial Machine Learning》Ch 7.
  Purge: train 中所有 label 时间与 test 重叠的样本删除
  Embargo: test 块两端再加 embargo period, train 数据避免 embargo 内样本

edge_cases:
  - embargo_pct ∈ [0, 0.1] 范围检查
  - label_horizon 与 embargo 一致性
  - 时序索引必须排序
  - n_splits >= 2

performance_target:
  n=10000, n_splits=10: < 100ms

spec_references:
  - §9.2.1 CPCV
  - §17.2 OOS
  - §4 HMM walk-forward retrain
  - §8 因子验证

required_tests:
  happy_path:
    - 标准 purge_embargo (n=1000, n_splits=5, embargo=0.01)
    - label_horizon=5 days
  
  edge_cases:
    - 时序索引未排序 (raise)
    - n_splits=2 (最小)
    - embargo_pct=0 (退化为简单切分)
  
  exceptions:
    - n_splits < 2 (raise)
    - embargo_pct < 0 or > 0.1 (raise)
  
  academic_validation:
    与 López de Prado 论文 example 对比
    手工构造 ground truth (n=20, n_splits=4)
```

---

### SPEC: ops.gap_detect

```yaml
op_name: ops.gap_detect
module: helios.ops.time_series
estimated_days: 2
size: medium

purpose: 时序数据 gap 检测含 4 类分级 (per §14.5 Gap Detection A18)

api:
  signature: |
    def gap_detect(
        times: pd.DatetimeIndex,
        expected_interval: pd.Timedelta | None = None,
        asset_class: Literal["equity", "crypto", "fx", "commodity"] = "equity",
        severity_thresholds: dict | None = None
    ) -> pd.DataFrame
  
  output_schema: |
    DataFrame columns: [start_time, end_time, gap_duration, severity]
    severity ∈ ["no_gap", "short", "medium", "long"]

severity_default_thresholds:
  equity:
    short: gap < 1 trading day
    medium: 1-3 trading days
    long: > 3 trading days
  crypto: # 24/7
    short: gap < 6 hours
    medium: 6-24 hours
    long: > 24 hours

edge_cases:
  - expected_interval=None: 自动从 times 推断 (median diff)
  - asset_class 决定 holiday calendar
  - 严重 case: 全部时间点为单一 timestamp (raise)

performance_target:
  n=10000: < 50ms

spec_references:
  - §14.5 Gap Detection A18 (核心)
  - §22.3 数据保留 monitoring
  - §6 panel field 完整性

required_tests:
  happy_path:
    - 无 gap 数据
    - 短 gap (周末)
    - 中 gap (节假日)
    - 长 gap (停市)
  
  edge_cases:
    - n=1 (无法检测)
    - 全部相同 timestamp (raise)
    - asset_class="crypto" 24/7
```

---

### SPEC: ops.resample_align

```yaml
op_name: ops.resample_align
module: helios.ops.time_series
estimated_days: 2
size: medium

purpose: 跨资产时间对齐

api:
  signature: |
    def resample_align(
        data_dict: dict[str, pd.DataFrame],
        target_freq: str = "D",
        method: Literal["last", "mean", "ohlc"] = "last",
        timezone: str = "UTC",
        forward_fill_limit: int | None = 5
    ) -> pd.DataFrame
  
  output: 统一时间索引 + concat 后的 DataFrame, columns = "{key}_{column}"

edge_cases:
  - data_dict 各 frame 频率不同
  - timezone 不一致
  - target_freq 比所有原始频率粗 (downsampling)
  - target_freq 比某些原始频率细 (upsampling, forward_fill_limit)

performance_target:
  3 frames each n=10000: < 50ms

spec_references:
  - §7 跨资产相关性 (前置)
  - §6 panel cross-sector
  - §14 multi-source feed alignment

required_tests:
  happy_path:
    - 美股 (1d) + crypto (1d) 对齐
    - 美股 (1d) + crypto (1h) 对齐到 1d
    - 跨 timezone 对齐
  
  edge_cases:
    - 单 frame 输入
    - method="ohlc" (要求 frame 是 OHLC 格式)
    - forward_fill_limit=0
```

---

### SPEC: ops.lag_forward_fill

```yaml
op_name: ops.lag_forward_fill
module: helios.ops.time_series
estimated_days: 1
size: simple

purpose: 时序滞后 + 前向填充, 含 max_gap 限制

api:
  signature: |
    def lag_forward_fill(
        data: pd.Series | pd.DataFrame,
        max_gap: int | pd.Timedelta = 5,
        lag: int = 0,
        strict: bool = False
    ) -> pd.Series | pd.DataFrame

edge_cases:
  - max_gap 超出: strict=True raise, strict=False 保留 NaN
  - lag != 0 与 forward-fill 顺序: 先 fill 再 shift
  - DataFrame multi-column 各自独立

spec_references:
  - §6 panel data 填充
  - §13 持仓 mark-to-market
  - §14 实时数据
  - §17 OOS
  - §22 数据保留
  - §19 news timestamp

required_tests:
  happy_path:
    - 简单 forward fill, max_gap=5
    - lag=1 + forward fill
    - DataFrame multi-column
  
  edge_cases:
    - 全 NaN
    - max_gap 超出, strict=True (raise)
    - max_gap 超出, strict=False (保留)
```

---

### SPEC: ops.percentile_rank

```yaml
op_name: ops.percentile_rank
module: helios.ops.time_series
estimated_days: 1
size: simple

purpose: 计算每数据点在历史窗口中的 percentile rank

api:
  signature: |
    def percentile_rank(
        data: pd.Series | pd.DataFrame,
        window: int | None = None,
        method: Literal["rolling", "cross_sectional", "expanding"] = "rolling",
        ties: Literal["average", "min", "max"] = "average"
    ) -> pd.Series | pd.DataFrame
  
  output: percentile rank ∈ [0, 1], 与输入同 shape

method_explained:
  rolling: 每个时间点的 rank 在 trailing window 中计算
  cross_sectional: 每个时间点的 rank 在所有 columns 中计算 (DataFrame only)
  expanding: 用从开始到当前的所有数据计算

edge_cases:
  - window 不足: NaN
  - 全 NaN 窗口: NaN
  - ties 处理 (average / min / max)

spec_references:
  - §6 panel 字段
  - §7 跨资产
  - §8 因子
  - §18.4 因子归因
  - §19 news ranking

required_tests:
  happy_path:
    - rolling window=252 on synthetic
    - cross_sectional on DataFrame
    - expanding mode
  
  edge_cases:
    - 单元素 (rank = 0.5 by convention)
    - 全相同值 (ties)
  
  academic_validation:
    与 scipy.stats.rankdata 对比 (cross-sectional case)
```

---

### SPEC: ops.cumulative_returns

```yaml
op_name: ops.cumulative_returns
module: helios.ops.time_series
estimated_days: 1
size: simple

purpose: 从 returns 计算累计 return / equity curve

api:
  signature: |
    def cumulative_returns(
        returns: pd.Series,
        return_type: Literal["log", "simple"] = "log",
        initial_capital: float = 1.0,
        compound: bool = True
    ) -> pd.Series

math:
  log return: cumsum(r), 然后 exp(累计) × initial_capital
  simple compound: initial_capital × cumprod(1 + r)
  simple non-compound: initial_capital × (1 + cumsum(r))

edge_cases:
  - returns 含 NaN: 跳过 (cumulative 时 NaN propagation)
  - initial_capital <= 0 raise
  - simple return 含 < -1 (大于 100% 亏损): warning + 处理为 0

spec_references:
  - §9 backtest equity curve
  - §13 持仓 P&L
  - §17 OOS forward-test

required_tests:
  happy_path:
    - log returns to equity curve
    - simple compound
    - simple non-compound
  
  edge_cases:
    - returns=[0, 0, 0] (equity = initial)
    - simple return = -1.5 (warning)
```

---

## 组 2：统计推断（10 个）

### SPEC: ops.bootstrap_ci

```yaml
op_name: ops.bootstrap_ci
module: helios.ops.statistics
estimated_days: 2
size: medium

purpose: 非参数 bootstrap 置信区间

api:
  signature: |
    def bootstrap_ci(
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: Literal["percentile", "bca", "basic"] = "percentile",
        random_state: int | None = None
    ) -> dict[str, float]
  
  output: |
    {
      "point_estimate": float,
      "ci_lower": float,
      "ci_upper": float,
      "se": float (bootstrap standard error),
      "n_bootstrap": int,
      "method": str
    }

implementation:
  必须用 numpy.random.choice 矢量化 (不用 python loop)
  正确 random_state 处理

edge_cases:
  - n_bootstrap < 100 warning
  - confidence_level ∉ (0, 1) raise
  - statistic_fn 返回 NaN 时跳过 (但 > 50% NaN raise)
  - 空 data raise
  - data 含 NaN: 自动 omit

performance_target:
  n=10000, n_bootstrap=1000: < 100ms (这是硬性目标)

spec_references:
  - §9.4 DSR (内部用)
  - §13.9 Position Regime Beta (1000 次 bootstrap, 明确)
  - §18.4 因子归因
  - §20 SLA
  - §17 OOS Sharpe distribution

required_tests:
  happy_path:
    - mean of normal distribution
    - sharpe ratio of synthetic returns
    - median 
  
  edge_cases:
    - n_bootstrap=100 (warning)
    - method="bca" (Bias-Corrected and accelerated)
    - method="basic"
    - random_state 重现性
  
  academic_validation:
    与 scipy.stats.bootstrap 对比 (percentile method)
    rtol=0.05 (因为 random sampling 有方差)
  
  performance:
    benchmark: < 100ms for n=10000, n_bootstrap=1000
```

---

### SPEC: ops.percentile_ci

```yaml
op_name: ops.percentile_ci
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: 从 sample 数组计算 percentile CI

api:
  signature: |
    def percentile_ci(
        samples: np.ndarray,
        quantiles: list[float] = [0.05, 0.5, 0.95],
        interpolation: str = "linear"
    ) -> dict[str, float]
  
  output: {f"q_{q}": float for q in quantiles}

edge_cases:
  - quantiles ∉ [0, 1] raise
  - 空 samples 返回 NaN dict
  - 全 NaN samples 返回 NaN dict

spec_references:
  - §9 backtest path Sharpe distribution
  - §11 Scenario percentile
  - §13 Regime Beta
  - §17 analogy similarity
  - §18 valuation distribution

required_tests:
  happy_path / edge_cases / academic_validation 与 numpy.percentile 对比
```

---

### SPEC: ops.distribution_summary

```yaml
op_name: ops.distribution_summary
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: 统一分布描述统计

api:
  signature: |
    def distribution_summary(
        data: np.ndarray,
        percentiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
    ) -> dict[str, float]
  
  output: |
    {"mean", "median", "std", "skew", "kurtosis_excess",
     "n", "n_nan", "min", "max",
     "q_0.05", "q_0.25", ...}

note:
  kurtosis 用 excess (Fisher-Pearson, bias=False)
  scipy.stats.kurtosis(x, fisher=True, bias=False)

spec_references:
  - §9 backtest output
  - §11 Scenario
  - §13 Regime Beta CI
  - §17 path distribution

required_tests:
  与 scipy.stats.describe 部分字段对比 + 自加 percentile
```

---

### SPEC: ops.skew_kurt_robust

```yaml
op_name: ops.skew_kurt_robust
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: 稳健 skewness + kurtosis (Fisher-Pearson 校正)

api:
  signature: |
    def skew_kurt_robust(
        data: np.ndarray,
        bias: bool = False,
        nan_policy: Literal["propagate", "raise", "omit"] = "omit"
    ) -> dict[str, float]
  
  output: {"skewness": float, "kurtosis_excess": float}

CRITICAL_NOTE: |
  scipy.stats.skew/kurtosis 默认 bias=True
  PSR 公式需要 bias=False (Fisher-Pearson)
  常见 bug 来源, 测试要 explicit cover

spec_references:
  - §9.3 PSR 调整公式 (必须)
  - §12 EVT λ-distribution
  - §11 Scenario distribution analysis
  - §4 HMM emission

required_tests:
  - bias=True vs bias=False 对比
  - n < 4 时返回 NaN
  - 学术对照: 与 scipy.stats.skew/kurtosis 对比 (含 bias 参数)
```

---

### SPEC: ops.kolmogorov_smirnov_test

```yaml
op_name: ops.kolmogorov_smirnov_test
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: KS test for distribution similarity

api:
  signature: |
    def kolmogorov_smirnov_test(
        sample_a: np.ndarray,
        sample_b: np.ndarray | str | Callable | None = None,
        mode: Literal["one_sample", "two_sample"] = "two_sample",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    ) -> dict[str, float]
  
  output: {"statistic": float, "p_value": float, "n_a": int, "n_b": int}

note:
  one_sample: sample_b 是分布名 "norm" / "uniform" 等
  two_sample: sample_b 是 array

spec_references:
  - §12 EVT 阈值方法
  - §11 Scenario validation
  - §17 OOS distribution check

required_tests:
  与 scipy.stats.kstest / ks_2samp 对比
```

---

### SPEC: ops.mann_kendall_trend

```yaml
op_name: ops.mann_kendall_trend
module: helios.ops.statistics
estimated_days: 2
size: medium

purpose: Mann-Kendall 单调趋势检验 + Hamed-Rao 自相关修正

api:
  signature: |
    def mann_kendall_trend(
        data: np.ndarray,
        alpha: float = 0.05,
        hamed_rao_correction: bool = True
    ) -> dict[str, Any]
  
  output: |
    {"trend": "increasing" | "decreasing" | "no_trend",
     "p_value": float,
     "tau": float,
     "z_score": float,
     "slope": float,
     "n": int}

spec_references:
  - §9.8 Strategy Decay
  - §17 historical pattern
  - §18 ESG trend
  - §6 panel trend monitoring

required_tests:
  - 增长序列 (expect "increasing")
  - 下降序列 (expect "decreasing")
  - 平稳序列 (expect "no_trend")
  - 自相关序列 (test Hamed-Rao 修正效果)
  
  academic_validation:
    与 pymannkendall 对比 (rtol=1e-9)
```

---

### SPEC: ops.bayes_beta_update

```yaml
op_name: ops.bayes_beta_update
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: Beta(α, β) posterior update

api:
  signature: |
    def bayes_beta_update(
        prior_alpha: float,
        prior_beta: float,
        successes: int,
        failures: int,
        posterior_quantiles: list[float] = [0.05, 0.5, 0.95]
    ) -> dict[str, float]
  
  output: |
    {"posterior_alpha", "posterior_beta",
     "posterior_mean", "posterior_mode",
     "q_0.05", "q_0.5", "q_0.95"}

math:
  α_post = α_prior + successes
  β_post = β_prior + failures

spec_references:
  - §15.3 Bandit
  - §16.6 Brier Score
  - §18 estimation update
  - §19 source historical credibility

required_tests:
  - prior=Beta(1,1), 观察 5 successes, 0 failures
  - prior=Beta(10, 5), 观察 3 successes, 2 failures
  - posterior_mode 边界 (α<1 或 β<1)
  
  academic_validation:
    与 scipy.stats.beta 对比 quantile, mean, mode
```

---

### SPEC: ops.brier_score_decomposed

```yaml
op_name: ops.brier_score_decomposed
module: helios.ops.statistics
estimated_days: 2
size: medium

purpose: Brier Score Murphy 1973 三分量分解

api:
  signature: |
    def brier_score_decomposed(
        forecasts: np.ndarray,  # ∈ [0, 1]
        outcomes: np.ndarray,   # binary {0, 1}
        n_bins: int = 10,
        method: Literal["binned", "binless"] = "binned"
    ) -> dict[str, float]
  
  output: |
    {"brier_score": float,
     "reliability": float,
     "resolution": float,
     "uncertainty": float,
     "skill": float}  # = (resolution - reliability) / uncertainty

math:
  Murphy 1973 三分量分解:
  BS = reliability - resolution + uncertainty
  reliability = E[(p_k - obar_k)^2]
  resolution = E[(obar_k - obar)^2]
  uncertainty = obar × (1 - obar)

reference: 
  Murphy, A. H. (1973). A New Vector Partition of the Probability Score.
  Journal of Applied Meteorology.

edge_cases:
  - forecasts ∉ [0, 1]: clip + warning
  - outcomes ∉ {0, 1}: raise
  - n_bins > n: warning (some bins empty)

spec_references:
  - §15 Bandit feedback
  - §16.6 thesis Brier Score (核心)
  - §17 BOCPD post-mortem
  - §19 prediction quality

required_tests:
  - perfect forecast (BS=0)
  - constant forecast at obar (resolution=0)
  - constant forecast at 0.5 with mixed outcomes
  
  academic_validation:
    Murphy 1973 论文 example data 自构 ground truth
    手算 expected reliability / resolution / uncertainty
```

---

### SPEC: ops.pearson_spearman_corr

```yaml
op_name: ops.pearson_spearman_corr
module: helios.ops.statistics
estimated_days: 1
size: simple

purpose: Pearson + Spearman 相关性

api:
  signature: |
    def pearson_spearman_corr(
        x: np.ndarray,
        y: np.ndarray,
        min_samples: int = 30,
        nan_policy: Literal["propagate", "raise", "omit"] = "omit"
    ) -> dict[str, float]
  
  output: |
    {"pearson_r": float, "pearson_p": float,
     "spearman_r": float, "spearman_p": float,
     "n_samples": int}

spec_references:
  - §7.2.1 Pearson
  - §7.2.2 Spearman
  - §6 panel 跨字段相关性
  - §18 因子相关性
  - §13 portfolio correlation

required_tests:
  - 线性相关数据 (Pearson 高, Spearman 高)
  - 单调非线性 (Pearson 中, Spearman 高)
  - 无相关 (两者都接近 0)
  
  academic_validation:
    与 scipy.stats.pearsonr / spearmanr 对比
```

---

### SPEC: ops.kde_density

```yaml
op_name: ops.kde_density
module: helios.ops.statistics
estimated_days: 2
size: medium

purpose: 核密度估计

api:
  signature: |
    def kde_density(
        data: np.ndarray,
        bandwidth: Literal["silverman", "scott"] | float = "silverman",
        eval_points: np.ndarray | None = None,
        reflect_boundary: bool = False
    ) -> dict[str, np.ndarray]
  
  output: {"x": np.ndarray, "density": np.ndarray}

spec_references:
  - §17 类比检索 distribution comparison
  - §11 Scenario distribution
  - §19 sentiment distribution

required_tests:
  与 scipy.stats.gaussian_kde 对比
```

---

## 组 3：距离 / 相似度（5 个）

### SPEC: ops.wasserstein_distance

```yaml
op_name: ops.wasserstein_distance
module: helios.ops.distance
estimated_days: 3
size: complex

purpose: Wasserstein 距离 1D + Sliced multi-D

api:
  signature: |
    def wasserstein_distance(
        u: np.ndarray,
        v: np.ndarray,
        mode: Literal["1d", "sliced_multi_d"] = "1d",
        n_projections: int = 100,
        random_state: int | None = None
    ) -> float

reference:
  1D: scipy.stats.wasserstein_distance
  Multi-D Sliced: Bonneel et al. 2015

implementation_notes:
  1D: 包装 scipy
  Sliced multi-D: 自实现, 用 numpy 矢量化
    随机采样 n_projections 个方向
    每个方向投影到 1D 算 Wasserstein
    取平均

spec_references:
  - §17.3.1 类比检索 (核心)
  - §11 Scenario distribution comparison
  - §18 valuation distribution
  - §19 NLP embedding cluster

required_tests:
  - 相同分布 → 距离 0
  - 平移分布 → 距离 = 平移量
  - 高斯 vs 高斯 (均值不同) → 已知公式
  - multi-D Gaussian vs Gaussian
  
  academic_validation:
    1D: 与 scipy.stats.wasserstein_distance 对比 (rtol=1e-9)
    Multi-D: 与论文 example 对比
  
  performance:
    1D, n=1000: < 5ms
    Sliced, d=10, n=1000, n_projections=100: < 500ms
```

---

### SPEC: ops.dtw_distance

```yaml
op_name: ops.dtw_distance
module: helios.ops.distance
estimated_days: 3
size: complex

purpose: Dynamic Time Warping 距离

api:
  signature: |
    def dtw_distance(
        x: np.ndarray,
        y: np.ndarray,
        window: int | None = None,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
        multivariate_mode: Literal["independent", "dependent"] | None = None
    ) -> dict[str, Any]
  
  output: {"distance": float, "path": list[tuple[int, int]] | None}

reference:
  Berndt-Clifford 1994 / Sakoe-Chiba 1978

implementation_notes:
  必须用 numpy 矢量化 DP, 不用 python nested loop
  Sakoe-Chiba band 加速: O(n × window)

spec_references:
  - §17.3.2 类比检索 (核心)
  - §11 historical scenario matching
  - §18 stock price pattern matching

required_tests:
  - 相同序列 → 距离 0
  - 同形状不同长度
  - multivariate independent vs dependent
  - 学术对照: 与 dtaidistance.dtw 对比
  
  performance:
    n=100, m=2520, window=20: < 100ms
```

---

### SPEC: ops.cosine_similarity_batch

```yaml
op_name: ops.cosine_similarity_batch
module: helios.ops.distance
estimated_days: 1
size: simple

purpose: 批量余弦相似度

api:
  signature: |
    def cosine_similarity_batch(
        query: np.ndarray,
        database: np.ndarray,
        pre_normalize: bool = False,
        top_k: int | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]

spec_references:
  - §17 类比检索
  - §18.5 智能同业检索
  - §19.5 Event Cascade DBSCAN
  - §16 watchlist analogy

required_tests:
  与 sklearn.metrics.pairwise.cosine_similarity 对比
```

---

### SPEC: ops.euclidean_distance_matrix

```yaml
op_name: ops.euclidean_distance_matrix
module: helios.ops.distance
estimated_days: 1
size: simple

purpose: 批量欧氏距离矩阵

api:
  signature: |
    def euclidean_distance_matrix(
        X: np.ndarray,
        Y: np.ndarray | None = None,
        weights: np.ndarray | None = None
    ) -> np.ndarray

spec_references:
  - §17 panel signature 距离
  - §6 panel cross-asset
  - §11 Scenario clustering
  - §13 portfolio similarity

required_tests:
  与 scipy.spatial.distance.cdist (Euclidean) 对比
```

---

### SPEC: ops.symmetric_kl_divergence

```yaml
op_name: ops.symmetric_kl_divergence
module: helios.ops.distance
estimated_days: 1
size: simple

purpose: Jensen-Shannon divergence / symmetric KL

api:
  signature: |
    def symmetric_kl_divergence(
        p: np.ndarray,
        q: np.ndarray,
        mode: Literal["js", "symmetric_kl"] = "js",
        base: Literal["e", "2"] = "e",
        epsilon: float = 1e-12
    ) -> float

spec_references:
  - §19 NLP cluster comparison
  - §11 distribution shift detection
  - §17 distribution divergence

required_tests:
  - p == q → divergence 0
  - JS vs Symmetric KL 对比
  - 含 0 概率 (epsilon smoothing)
```

---

## 组 4：数值稳定性（3 个）

### SPEC: ops.logsumexp_safe

```yaml
op_name: ops.logsumexp_safe
module: helios.ops.numerics
estimated_days: 1
size: simple

purpose: log(sum(exp(x))) 数值稳定

api:
  signature: |
    def logsumexp_safe(
        x: np.ndarray,
        axis: int | None = None,
        weights: np.ndarray | None = None,
        keepdims: bool = False
    ) -> np.ndarray

spec_references:
  - §4.5 BOCPD log space recursion (核心)
  - §4.4 SVI
  - §15 bandit posterior
  - §19 NLP log-prob aggregation

required_tests:
  - 防 overflow (large x)
  - 防 underflow (very small x)
  - all -inf → -inf
  - weights 参数
  
  academic_validation:
    与 scipy.special.logsumexp 对比 rtol=1e-9
```

---

### SPEC: ops.softmax_safe

```yaml
op_name: ops.softmax_safe
module: helios.ops.numerics
estimated_days: 1
size: simple

purpose: 数值稳定 softmax

api:
  signature: |
    def softmax_safe(
        x: np.ndarray,
        axis: int = -1,
        temperature: float = 1.0
    ) -> np.ndarray

spec_references:
  - §5 Layer 1 子分数加权
  - §15 bandit policy
  - §19 ranking
  - §18 attribution weight

required_tests:
  - sum to 1
  - 高温 → uniform
  - 低温 → one-hot
  - 学术对照 scipy.special.softmax
```

---

### SPEC: ops.clip_with_warning

```yaml
op_name: ops.clip_with_warning
module: helios.ops.numerics
estimated_days: 1
size: simple

purpose: clip + 溢出告警

api:
  signature: |
    def clip_with_warning(
        x: np.ndarray | float,
        lower: float | None = None,
        upper: float | None = None,
        warning_threshold_pct: float = 0.05,
        logger: logging.Logger | None = None
    ) -> np.ndarray | float

note: 用于业务约束 e.g. Fusion Score Layer 1 ≤ 0.6

spec_references:
  - §5 Fusion Score (核心约束)
  - §9 backtest position size
  - §15 alert threshold
  - §13 Regime Beta CI bound
  - §11 Scenario value bounds

required_tests:
  - 无 clip 触发 (no warning)
  - 1% clip 触发 (no warning, < threshold)
  - 10% clip 触发 (warning logged)
  - logger=None silent
```

---

## 组 5：Regime 联动（3 个）

### SPEC: ops.regime_filter_data

```yaml
op_name: ops.regime_filter_data
module: helios.ops.regime
estimated_days: 2
size: medium

purpose: 根据 regime label 过滤数据

api:
  signature: |
    def regime_filter_data(
        data: pd.DataFrame,
        regime_labels: pd.Series,
        target_regime: str | list[str],
        mode: Literal["hard", "soft"] = "hard",
        min_probability: float = 0.5
    ) -> pd.DataFrame

note:
  hard mode: regime_labels 是 string label
  soft mode: regime_labels 是 dict[regime_name -> probability], 需要不同 schema

edge_cases:
  - regime_labels 与 data index 必须对齐
  - 过滤后空 DataFrame 时正确返回
  - target_regime 不在 labels 中 raise

spec_references:
  - §5.2 Layer 0 veto
  - §13.9 Position Regime Beta (核心)
  - §17 historical regime replay
  - §18.3.2 regime-conditional valuation
  - §11 regime-specific scenarios
  - §15 regime alerts
  - §6 regime panel filter

required_tests:
  - hard mode 单 regime
  - hard mode 多 regime
  - soft mode min_probability=0.7
  - 过滤后空
```

---

### SPEC: ops.regime_transition_matrix

```yaml
op_name: ops.regime_transition_matrix
module: helios.ops.regime
estimated_days: 2
size: medium

purpose: 估计转移矩阵 + duration 分布

api:
  signature: |
    def regime_transition_matrix(
        regime_labels: pd.Series,
        states: list[str] | None = None,
        include_duration: bool = True
    ) -> dict[str, Any]
  
  output: |
    {"transition_matrix": pd.DataFrame K x K,
     "duration_distribution": dict[str, dict],
     "stationary_distribution": pd.Series,
     "n_transitions": int}

spec_references:
  - §4 HMM verification (HSMM duration)
  - §17 historical analysis
  - §11 scenario calibration

required_tests:
  - 已知 markov chain 输出对比已知矩阵
  - sticky state 检测
  - duration 估计正确性
```

---

### SPEC: ops.regime_label_align

```yaml
op_name: ops.regime_label_align
module: helios.ops.regime
estimated_days: 1
size: simple

purpose: 跨频率 regime label 对齐

api:
  signature: |
    def regime_label_align(
        target_index: pd.DatetimeIndex,
        regime_labels: pd.Series,
        method: Literal["asof", "ffill"] = "asof",
        tolerance: pd.Timedelta | None = None
    ) -> pd.Series

spec_references:
  - §13 持仓 tick × regime
  - §14 实时数据 regime annotation
  - §17 OOS replay
  - §15 alert with regime context

required_tests:
  - daily regime → 5min ticks alignment
  - method='asof' vs 'ffill'
  - tolerance 限制
```

---

## 组 6：金融指标（4 个）

### SPEC: ops.drawdown_curve

```yaml
op_name: ops.drawdown_curve
module: helios.ops.finance
estimated_days: 2
size: medium

purpose: drawdown 时序 + max drawdown + duration + recovery

api:
  signature: |
    def drawdown_curve(
        equity_or_returns: pd.Series,
        input_type: Literal["equity", "returns"] = "equity",
        compound: bool = True
    ) -> dict[str, Any]
  
  output: |
    {"drawdown_series": pd.Series,
     "max_drawdown": float,
     "max_drawdown_start": pd.Timestamp,
     "max_drawdown_end": pd.Timestamp,
     "max_drawdown_recovery": pd.Timestamp | None,
     "underwater_duration_days": int}

math:
  drawdown_t = (equity_t - peak_t) / peak_t
  peak_t = max(equity_0, ..., equity_t)

spec_references:
  - §5 Veto trigger
  - §9 backtest TCA
  - §11 Scenario stress
  - §13 portfolio risk
  - §17 forward-test
  - §15 alert
  - §18 stock max drawdown
  - §6 panel field

required_tests:
  - 无回撤 (max_drawdown=0)
  - 单峰单谷
  - 多峰多谷 (找到 max)
  - 永不恢复 (recovery=None)
  - input_type='returns'
```

---

### SPEC: ops.sharpe_ratio

```yaml
op_name: ops.sharpe_ratio
module: helios.ops.finance
estimated_days: 1
size: simple

purpose: Sharpe ratio (atomic, 不含 bootstrap)

api:
  signature: |
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float | pd.Series = 0.0,
        annualization_factor: int = 252,
        ddof: int = 1
    ) -> float

math: SR = mean(r - r_f) / std(r - r_f) × √annualization

note: |
  与 ops.bootstrap_sharpe (Layer 2) 区分
  这是 atomic Sharpe 计算

spec_references:
  - §9 backtest (核心)
  - §11 Scenario
  - §13 portfolio
  - §17 OOS
  - §6 panel field
  - §18 stock 评估

required_tests:
  - 已知 Sharpe 数据集对比
  - annualization 不同 (252 / 365 / 8760)
  - risk_free_rate 时序
  - std=0 → NaN
```

---

### SPEC: ops.beta_alpha_ols

```yaml
op_name: ops.beta_alpha_ols
module: helios.ops.finance
estimated_days: 2
size: medium

purpose: CAPM / 多因子 OLS 回归

api:
  signature: |
    def beta_alpha_ols(
        asset_returns: pd.Series,
        market_returns: pd.Series | pd.DataFrame,
        use_hac: bool = False,
        hac_lags: int | None = None,
        min_samples: int = 30
    ) -> dict[str, Any]
  
  output: |
    {"alpha": float,
     "beta": float | dict[str, float],
     "alpha_se": float, "beta_se": float | dict,
     "r_squared": float,
     "adj_r_squared": float,
     "n_samples": int,
     "p_values": dict}

note:
  market_returns 是 Series → 单因子 (CAPM)
  market_returns 是 DataFrame → 多因子 (e.g. Fama-French)

spec_references:
  - §13.9 Position Regime Beta (核心)
  - §18.4 Fama-French 因子归因
  - §17 portfolio alpha 分析

required_tests:
  - 单因子 CAPM (synthetic data)
  - 多因子 FF3 (synthetic data)
  - HAC standard error
  - n < min_samples raise
  
  academic_validation:
    与 statsmodels.regression.linear_model.OLS 对比
    rtol=1e-9
```

---

### SPEC: ops.value_at_risk

```yaml
op_name: ops.value_at_risk
module: helios.ops.finance
estimated_days: 2
size: medium

purpose: VaR 三方法 + ES (atomic, 完整 GARCH-EVT 在 Layer 2)

api:
  signature: |
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: Literal["historical", "parametric", "cornish_fisher"] = "historical",
        include_es: bool = True
    ) -> dict[str, float]
  
  output: |
    {"var": float, "es": float | None, 
     "method": str, "confidence_level": float}

math:
  historical: empirical (1-α) percentile
  parametric: μ + σ × Φ^{-1}(1-α) (assume normal)
  cornish_fisher: μ + σ × CF_quantile(skew, kurt)

spec_references:
  - §12 EVT (基础对照)
  - §13 portfolio VaR
  - §11 Scenario VaR
  - §9 backtest TCA risk

required_tests:
  - historical VaR on synthetic
  - parametric VaR (normal assumption)
  - cornish_fisher VaR (with skew/kurt)
  - 三方法在正态分布下应一致
  - ES > VaR (always)
```

---

# Part 3: 执行清单（给 Wiki 用）

## 5 lane 分配

```yaml
lane_1_time_series:
  CC instance: 1
  ops: 11 个 (组 1)
  estimated_duration: 17 天 (单 CC 串行) / 5-7 天 (含并行 review)
  
  顺序建议:
    1. ops.log_returns           (1d, simple)
    2. ops.cumulative_returns    (1d, simple)
    3. ops.rolling_window_split  (1d, simple)
    4. ops.lag_forward_fill      (1d, simple)
    5. ops.percentile_rank       (1d, simple)
    6. ops.ewma_smooth           (1d, simple)
    7. ops.realized_vol          (2d, medium)
    8. ops.zscore_normalize      (2d, medium)
    9. ops.gap_detect            (2d, medium)
    10. ops.resample_align       (2d, medium)
    11. ops.purge_embargo_split  (3d, complex)

lane_2_statistics:
  CC instance: 1
  ops: 10 个 (组 2)
  estimated_duration: 14 天 / 5-7 天
  
  顺序建议:
    简单 → 复杂

lane_3_distance:
  CC instance: 1
  ops: 5 个 (组 3)
  estimated_duration: 9 天 / 4-6 天

lane_4_numerics_regime:
  CC instance: 1
  ops: 6 个 (组 4 + 组 5)
  estimated_duration: 8 天 / 3-5 天

lane_5_finance:
  CC instance: 1
  ops: 4 个 (组 6)
  estimated_duration: 7 天 / 3-5 天

total:
  serial: 55 天
  5 parallel: 7-8 天净时间
  含 review buffer: 5-7 周

  实际时间 (Wiki review pace 限制): 6-8 周
```

## 启动步骤

```yaml
step_1_pre_flight (1-2 天):
  - 创建 helios.ops 包骨架 (Wiki 一人, 不分 CC)
  - 写 _base.py (Pydantic base classes)
  - 写 _validation.py (输入验证 utils)
  - 设置 CI/CD (lint + test + benchmark)
  - merge, 作为后续 CC PR 的基础

step_2_launch_lanes (Day 1):
  - 准备 5 份完整 prompt (template + spec)
  - 启动 5 CC instance, 各跑一个 lane
  - Wiki 每天 review 5-10 PR

step_3_review_cycle (Day 2 - 完工):
  - 每个 PR Wiki review:
    * 学术对照测试通过
    * 性能基准达标
    * 测试覆盖率 ≥ 90%
    * lint check 通过
  - 通过 → merge
  - 不通过 → 反馈给 CC, 修订
  
step_4_release (Day 完工 + 1 周):
  - oprim 1.0.0 release
  - 文档站点上线
  - 内部宣布

step_5_layer_2_准备:
  - ADR-062 Layer 2 元 skill 启动
  - 8-10 个元 skill 设计
  - 单独后续工作
```

---

# 结束

**总览**：
- 1 份通用模板 (~150 行)
- 31 份 op spec (~50-100 行 each)
- 1 份执行清单
- 总计本文档 ~3000 行

**使用方法**：
- `[GENERIC_TEMPLATE 整段]` + `[SPEC: ops.<name>]` = 完整 prompt
- 复制粘贴给 CC instance
- 每个 CC instance 跑 1 个 lane (1 lane = 1 组)
- 5 CC instance 并行

**预计完工时间**: 5-7 周（含 Wiki review buffer）
