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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ._bayes_opt_plan import bayesian_optimize

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


def optimize_loop(
    search_space: dict[str, tuple[float, float]],
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
) -> OptimizeLoopResult:
    """train 上 BO 最大化多目标效用 → gate 区间硬门禁 → 采纳判定。

    search_space: {参数名: (lo, hi)} — 连续 box 约束。
    gate_on: "gate" 用 gate_window (默认, OOS); "train" 用 train_window 门禁。
    gate_window 缺失时降级为 train_window 门禁。
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
    bounds = [(search_space[n][0], search_space[n][1]) for n in names]
    n_evals = 0

    def neg_utility(x: list[float]) -> float:
        """BO 目标: −U(train)。缓存命中跳过 evaluate。

        重复点 (BO 收敛后反复采样同一最优点属正常) 返回缓存值而非极端惩罚,
        否则 -1e9 会污染 GP 后验, 把 EI 带偏。
        """
        nonlocal n_evals
        params = dict(zip(names, [float(v) for v in x]))
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
    )   # minimize=True (默认): 最小化 −U = 最大化 U
    best_x = res["best_x"]
    best_params = dict(zip(names, [float(v) for v in best_x]))
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
    )


__all__ = [
    "DEFAULT_UTILITY_WEIGHTS", "EvalCache", "EvalWindow",
    "MultiObjectiveConfig", "OptimizeLoopResult", "RiskGateConfig",
    "fingerprint_eval", "multi_objective_utility", "optimize_loop",
]
