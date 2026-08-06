"""oprim._optimize_loop — 多目标效用优化循环 (train 搜索 + OOS 硬门禁 + 评价缓存)。

协议:
  - 搜索只在 train_window 上最大化多目标效用 (U = Σ wᵢ·mᵢ);
  - 是否采纳看 gate_window (默认 OOS) + RiskGateConfig 硬门禁;
  - 评价缓存: 键 = fingerprint(params, window, meta) — 适配器保证同一键同一结果,
    因此缓存命中可安全跳过 evaluate (含跨进程 disk_path 持久化)。

BO 内核复用本包 bayesian_optimize (RBF-GP + EI), 不重造轮子。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ._bayes_opt_plan import bayesian_optimize


@dataclass(frozen=True)
class ParamSpec:
    """参数规格: 连续 / 整数 / 对数。

    kind:
      continuous — BO 在原空间优化;
      integer    — BO 连续空间优化, 评估时取整;
      log        — BO 在 log 空间优化, 评估时 exp 还原 (要求 low > 0)。
    """

    low: float
    high: float
    kind: str = "continuous"

    def __post_init__(self) -> None:
        if not (self.low < self.high):
            raise ValueError(f"ParamSpec 要求 low < high, 收到 ({self.low}, {self.high})")
        if self.kind not in ("continuous", "integer", "log"):
            raise ValueError(f"kind 必须是 continuous|integer|log, 收到 {self.kind!r}")
        if self.kind == "log" and self.low <= 0:
            raise ValueError(f"log 参数要求 low > 0, 收到 low={self.low}")

    def opt_bounds(self) -> tuple[float, float]:
        """BO 优化空间边界。"""
        if self.kind == "log":
            return (math.log(self.low), math.log(self.high))
        return (self.low, self.high)

    def decode(self, x: float) -> float:
        """优化空间值 → 真实参数值。"""
        if self.kind == "log":
            return float(math.exp(x))
        if self.kind == "integer":
            return float(round(x))
        return float(x)


def _coerce_spec(v: Any) -> ParamSpec:
    """(low, high) 元组向后兼容 → ParamSpec。"""
    if isinstance(v, ParamSpec):
        return v
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return ParamSpec(float(v[0]), float(v[1]), kind="continuous")
    raise TypeError(f"search_space 值必须是 ParamSpec 或 (low, high), 收到 {v!r}")

# 默认效用权重: U ≈ 1.0·sharpe + 0.25·total_return − 1.0·max_drawdown
#                  − 0.05·turnover − 0.5·cost_drag
DEFAULT_UTILITY_WEIGHTS: dict[str, float] = {
    "sharpe": 1.0,
    "total_return": 0.25,
    "max_drawdown": -1.0,
    "turnover": -0.05,
    "cost_drag": -0.5,
}

# 风险门禁默认值 (全开 = 只挡极端值)
DEFAULT_RISK_GATE: dict[str, float] = {
    "min_sharpe": 0.0,
    "max_drawdown": 1.0,
    "min_trades": 0,
}


# =========================================================================
# 评价区间 / 多目标效用
# =========================================================================

@dataclass(frozen=True)
class EvalWindow:
    """评价区间。start/end 为 ISO 日期字符串 (字符串比较即时间序)。"""

    start: str
    end: str
    label: str = ""

    def key(self) -> str:
        return f"{self.label}|{self.start}|{self.end}"

    def contains(self, day: str) -> bool:
        return self.start <= day <= self.end


@dataclass
class MultiObjectiveConfig:
    """多目标效用配置。weights 覆盖默认权重 (未列出的保持默认)。"""

    weights: dict[str, float] = field(default_factory=dict)

    def utility(self, metrics: dict[str, float]) -> float:
        return multi_objective_utility(metrics, weights=self.weights)


def multi_objective_utility(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """U = Σ wᵢ·mᵢ。缺省指标按 0 处理 (不惩罚也不奖励)。"""
    w = dict(DEFAULT_UTILITY_WEIGHTS)
    if weights:
        w.update(weights)
    return float(sum(w.get(k, 0.0) * float(metrics.get(k, 0.0)) for k in w))


# =========================================================================
# 评价缓存 (指纹键 + 内存/磁盘双后端)
# =========================================================================

def fingerprint_eval(
    params: dict[str, Any],
    window: EvalWindow,
    meta: dict[str, Any] | None = None,
) -> str:
    """稳定指纹: params + window + meta → sha256 前 16 位。

    确定性要求: dict 键排序序列化; 适配器保证同一键同一评价结果。
    """
    payload = json.dumps(
        {"params": params, "window": window.key(), "meta": meta or {}},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class EvalCache:
    """评价结果缓存。disk_path 提供跨进程持久化 (JSONL 追加, 启动加载)。"""

    def __init__(self, disk_path: str | Path | None = None):
        self._mem: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.disk_path = Path(disk_path) if disk_path else None
        if self.disk_path:
            self.disk_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_disk()

    def _load_disk(self) -> None:
        if not self.disk_path or not self.disk_path.exists():
            return
        for line in self.disk_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                self._mem[rec["fingerprint"]] = rec["metrics"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue   # 损坏行跳过, 不阻断

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        val = self._mem.get(fingerprint)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, fingerprint: str, metrics: dict[str, Any]) -> None:
        self._mem[fingerprint] = metrics
        if self.disk_path:
            rec = {"fingerprint": fingerprint, "metrics": metrics}
            with self.disk_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "size": len(self._mem)}


# =========================================================================
# 风险硬门禁 (OOS 采纳判定)
# =========================================================================

@dataclass
class RiskGateConfig:
    """硬门禁: 全部通过才 accepted。缺指标 = 该检查 fail (安全默认)。

    max_turnover / max_cost_drag 为 None 时不检查该项。
    """

    min_sharpe: float = 0.0
    max_drawdown: float = 1.0
    min_trades: int = 0
    max_turnover: float | None = None
    max_cost_drag: float | None = None

    def evaluate(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """返回 (通过?, 原因)。全部通过时 reason="ok"。"""
        failures: list[str] = []
        checks: list[tuple[str, str, Any]] = [
            ("sharpe", "gte", self.min_sharpe),
            ("max_drawdown", "lte", self.max_drawdown),
            ("n_trades", "gte", self.min_trades),
            ("turnover", "lte", self.max_turnover),
            ("cost_drag", "lte", self.max_cost_drag),
        ]
        for metric, op, bound in checks:
            if bound is None:
                continue
            if metric not in metrics:
                failures.append(f"{metric}=缺失")
                continue
            v = float(metrics[metric])
            ok = v >= bound if op == "gte" else v <= bound
            if not ok:
                failures.append(f"{metric}={v:.4g} 未达 {'≥' if op == 'gte' else '≤'} {bound:g}")
        return (not failures, "ok" if not failures else "; ".join(failures))


# =========================================================================
# 主循环
# =========================================================================

@dataclass
class OptimizeLoopResult:
    accepted: bool
    best_params: dict[str, Any] | None
    best_train_metrics: dict[str, float] | None
    gate_metrics: dict[str, float] | None
    gate_reason: str = ""
    utility: float = 0.0
    n_evals: int = 0
    cache_hits: int = 0
    cache_miss: int = 0
    gate_window_label: str = ""
    early_stopped: bool = False


def optimize_loop(
    search_space: dict[str, Any],
    evaluate: Callable[[dict[str, Any], EvalWindow], dict[str, float]],
    train_window: EvalWindow,
    gate_window: EvalWindow | None = None,
    *,
    utility_config: MultiObjectiveConfig | None = None,
    risk_gate: RiskGateConfig | None = None,
    cache: EvalCache | None = None,
    n_init: int = 4,
    n_iter: int = 12,
    seed: int = 0,
    gate_on: str = "gate",
    eval_meta: dict[str, Any] | None = None,
    early_stop_rounds: int = 0,
    ei_stop: float | None = None,
) -> OptimizeLoopResult:
    """train 上 BO 最大化多目标效用 → gate 区间硬门禁 → 采纳判定。

    search_space: {参数名: ParamSpec | (low, high)} — 连续/整数/对数 box 约束。
    gate_on: "gate" 用 gate_window (默认, OOS); "train" 用 train_window 门禁。
    gate_window 缺失时降级为 train_window 门禁。
    early_stop_rounds / ei_stop: 透传 bayesian_optimize 早停。
    eval_meta: 进入指纹 (如 data_version/engine_version) — 升级自动打穿缓存。
    """
    if not search_space:
        raise ValueError("search_space 不能为空")
    cfg = utility_config or MultiObjectiveConfig()
    gate = risk_gate or RiskGateConfig()
    cache = cache or EvalCache()
    gate_win = gate_window or train_window
    if gate_on == "train":
        gate_win = train_window

    names = sorted(search_space)
    specs = {n: _coerce_spec(search_space[n]) for n in names}
    bounds = [specs[n].opt_bounds() for n in names]
    n_evals = 0

    def decode_x(x: list[float]) -> dict[str, float]:
        return {n: specs[n].decode(float(v)) for n, v in zip(names, x)}

    def neg_utility(x: list[float]) -> float:
        """BO 目标: −U(train)。缓存命中跳过 evaluate。

        重复点 (BO 收敛后反复采样同一最优点属正常) 返回缓存值而非极端惩罚,
        否则 -1e9 会污染 GP 后验, 把 EI 带偏。
        """
        nonlocal n_evals
        params = decode_x(x)
        fp = fingerprint_eval(params, train_window, eval_meta)
        metrics = cache.get(fp)
        if metrics is None:
            metrics = evaluate(params, train_window)
            cache.put(fp, metrics)
            n_evals += 1
        return -cfg.utility(metrics)

    res = bayesian_optimize(
        neg_utility, bounds,
        n_init=n_init, n_iter=n_iter, seed=seed,
        early_stop_rounds=early_stop_rounds, ei_stop=ei_stop,
    )   # minimize=True (默认): 最小化 −U = 最大化 U
    best_params = decode_x(res["best_x"])
    fp = fingerprint_eval(best_params, train_window, eval_meta)
    train_metrics = cache.get(fp)
    if train_metrics is None:
        train_metrics = evaluate(best_params, train_window)
        cache.put(fp, train_metrics)
        n_evals += 1

    utility = cfg.utility(train_metrics)

    # ── 硬门禁: OOS (或 gate_on 指定区间) ──
    gate_fp = fingerprint_eval(best_params, gate_win, eval_meta)
    gate_metrics = cache.get(gate_fp)
    if gate_metrics is None:
        gate_metrics = evaluate(best_params, gate_win)
        cache.put(gate_fp, gate_metrics)
        n_evals += 1
    passed, reason = gate.evaluate(gate_metrics)

    stats = cache.stats()
    return OptimizeLoopResult(
        accepted=passed,
        best_params=best_params,
        best_train_metrics=dict(train_metrics),
        gate_metrics=dict(gate_metrics),
        gate_reason=reason,
        utility=round(utility, 6),
        n_evals=n_evals,
        cache_hits=stats["hits"],
        cache_miss=stats["misses"],
        gate_window_label=gate_win.key(),
        early_stopped=bool(res.get("early_stopped", False)),
    )


# =========================================================================
# Walk-forward: 多折 OOS 滚动验证
# =========================================================================

@dataclass
class FoldResult:
    """单折结果: train/test 区间 + 该折 optimize_loop 结果。"""

    train: EvalWindow
    test: EvalWindow
    result: OptimizeLoopResult

    def oos_utility(self, cfg: MultiObjectiveConfig) -> float:
        return cfg.utility(self.result.gate_metrics or {})


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    accept_rate: float
    oos_utility_mean: float
    oos_utility_std: float
    metric_summaries: dict[str, dict[str, float]]
    aggregate_accepted: bool
    min_accept_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "folds": len(self.folds),
            "accept_rate": self.accept_rate,
            "oos_utility_mean": self.oos_utility_mean,
            "oos_utility_std": self.oos_utility_std,
            "metric_summaries": self.metric_summaries,
            "aggregate_accepted": self.aggregate_accepted,
            "min_accept_rate": self.min_accept_rate,
        }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        idx = p * (n - 1)
        lo_i, hi_i = int(math.floor(idx)), int(math.ceil(idx))
        if lo_i == hi_i:
            return float(s[lo_i])
        frac = idx - lo_i
        return s[lo_i] * (1 - frac) + s[hi_i] * frac

    return {"mean": round(sum(values) / n, 6),
            "p25": round(q(0.25), 6), "p50": round(q(0.50), 6),
            "p75": round(q(0.75), 6)}


def walk_forward(
    folds: Sequence[tuple[EvalWindow, EvalWindow]],
    search_space: dict[str, Any],
    evaluate: Callable[[dict[str, Any], EvalWindow], dict[str, float]],
    *,
    utility_config: MultiObjectiveConfig | None = None,
    risk_gate: RiskGateConfig | None = None,
    cache: EvalCache | None = None,
    min_accept_rate: float = 0.5,
    eval_meta: dict[str, Any] | None = None,
    n_init: int = 4,
    n_iter: int = 12,
    seed: int = 0,
    early_stop_rounds: int = 0,
    ei_stop: float | None = None,
    **optimize_kwargs: Any,
) -> WalkForwardResult:
    """每折独立 optimize_loop, 共享 cache; 聚合 OOS 效用与指标分位数。

    folds: [(train_i, test_i), ...] — 滚动窗口序列。
    aggregate_accepted = accept_rate ≥ min_accept_rate。
    """
    if not folds:
        raise ValueError("folds 不能为空")
    cfg = utility_config or MultiObjectiveConfig()
    shared = cache or EvalCache()

    fold_results: list[FoldResult] = []
    for fold_i, (train_win, test_win) in enumerate(folds):
        fold_seed = seed + fold_i * 1000          # 每折不同种子, 整体可复现
        result = optimize_loop(
            search_space, evaluate,
            train_window=train_win, gate_window=test_win,
            utility_config=cfg, risk_gate=risk_gate, cache=shared,
            n_init=n_init, n_iter=n_iter, seed=fold_seed,
            eval_meta=eval_meta,
            early_stop_rounds=early_stop_rounds, ei_stop=ei_stop,
            **optimize_kwargs,
        )
        fold_results.append(FoldResult(train_win, test_win, result))

    oos_utils = [f.oos_utility(cfg) for f in fold_results]
    n_accept = sum(1 for f in fold_results if f.result.accepted)
    accept_rate = n_accept / len(fold_results)

    # 指标分位数: 跨折 gate 指标
    metric_names: set[str] = set()
    for f in fold_results:
        metric_names.update((f.result.gate_metrics or {}).keys())
    summaries: dict[str, dict[str, float]] = {}
    for m in sorted(metric_names):
        vals = [float((f.result.gate_metrics or {}).get(m, 0.0)) for f in fold_results]
        summaries[m] = _quantiles(vals)

    mean = sum(oos_utils) / len(oos_utils) if oos_utils else 0.0
    std = (sum((u - mean) ** 2 for u in oos_utils) / len(oos_utils)) ** 0.5 if oos_utils else 0.0
    return WalkForwardResult(
        folds=fold_results,
        accept_rate=round(accept_rate, 4),
        oos_utility_mean=round(mean, 6),
        oos_utility_std=round(std, 6),
        metric_summaries=summaries,
        aggregate_accepted=accept_rate >= min_accept_rate,
        min_accept_rate=min_accept_rate,
    )


# =========================================================================
# 策略生命周期: research → candidate → paper → degraded → retired
# =========================================================================

PHASES = ("research", "candidate", "paper", "degraded", "retired")


@dataclass
class LifecycleEvent:
    """生命周期审计事件。"""

    ts: float
    event: str
    phase_from: str
    phase_to: str
    reason: str
    gate_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ts": round(self.ts, 3), "event": self.event,
                "phase_from": self.phase_from, "phase_to": self.phase_to,
                "reason": self.reason, "gate_reason": self.gate_reason}


@dataclass
class LifecycleRecord:
    strategy_id: str
    phase: str = "research"
    gate_failures: int = 0
    history: list[LifecycleEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "phase": self.phase,
                "gate_failures": self.gate_failures,
                "history": [e.as_dict() for e in self.history],
                "created_at": round(self.created_at, 3)}


class StrategyLifecycle:
    """策略生命周期: 优化结果 → 升降级 (轻量, 审计留痕)。

    规则 (apply_optimize_result):
      accepted: research → candidate; degraded → research (可恢复);
                candidate/paper 保持;
      rejected: gate_failures += 1; 连续失败 ≥ max_gate_failures:
                research/candidate/paper → degraded;
                degraded → retired (再失败可退休)。
    """

    def __init__(self, strategy_id: str, *, max_gate_failures: int = 2):
        if max_gate_failures < 1:
            raise ValueError("max_gate_failures 必须 ≥ 1")
        self.rec = LifecycleRecord(strategy_id=strategy_id)
        self.max_gate_failures = max_gate_failures

    def apply_optimize_result(self, result: OptimizeLoopResult) -> LifecycleEvent | None:
        """根据优化结果升降级, 返回触发的事件 (无变化返回 None)。"""
        phase = self.rec.phase
        if result.accepted:
            self.rec.gate_failures = 0
            if phase == "research":
                return self._transition("promote_to_candidate", "candidate",
                                        "gate 通过")
            if phase == "degraded":
                return self._transition("recover_from_degraded", "research",
                                        "gate 恢复通过")
            return None                                   # candidate/paper 保持

        # rejected
        self.rec.gate_failures += 1
        if phase == "degraded":
            if self.rec.gate_failures >= self.max_gate_failures:
                return self._transition("retire", "retired", "连续 gate 失败",
                                        result.gate_reason)
            return None
        if self.rec.gate_failures >= self.max_gate_failures:
            return self._transition("degrade", "degraded", "连续 gate 失败",
                                    result.gate_reason)
        return None

    def promote_to_paper(self, reason: str) -> LifecycleEvent:
        """candidate → paper (人工/聚合验证后晋升)。"""
        if self.rec.phase != "candidate":
            raise ValueError(f"只有 candidate 可晋升 paper, 当前 {self.rec.phase}")
        return self._transition("promote_to_paper", "paper", reason)

    def retire(self, reason: str) -> LifecycleEvent:
        """任意阶段 → retired (人工退役)。"""
        if self.rec.phase == "retired":
            return self._transition("retire", "retired", reason)   # 幂等留痕
        return self._transition("retire", "retired", reason)

    def snapshot(self) -> dict[str, Any]:
        """审计快照: 完整状态 + 事件史。"""
        return self.rec.snapshot()

    def _transition(self, event: str, to: str, reason: str,
                    gate_reason: str = "") -> LifecycleEvent:
        ev = LifecycleEvent(time.time(), event, self.rec.phase, to, reason, gate_reason)
        self.rec.phase = to
        self.rec.history.append(ev)
        return ev


__all__ = [
    "DEFAULT_UTILITY_WEIGHTS", "EvalCache", "EvalWindow",
    "FoldResult", "LifecycleEvent", "LifecycleRecord",
    "MultiObjectiveConfig", "OptimizeLoopResult", "PHASES", "ParamSpec",
    "RiskGateConfig", "StrategyLifecycle", "WalkForwardResult",
    "fingerprint_eval", "multi_objective_utility", "optimize_loop",
    "walk_forward",
]
