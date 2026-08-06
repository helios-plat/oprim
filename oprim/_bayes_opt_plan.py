"""oprim._bayes_opt_plan — 贝叶斯优化规划 (numpy RBF-GP + EI)。

连续规划从网格 (grid_search) / beam 枚举升级到贝叶斯优化:
  - RBF-GP: K(x,x') = σ_f²·exp(−‖x−x'‖²/2ℓ²) + σ_n²·δ, 闭式后验均值/方差
  - EI 采集: EI(x) = (μ−f*−ξ)·Φ(z) + σ·φ(z), z = (μ−f*−ξ)/σ
  - bayesian_optimize: 通用 BO 循环 (n_init 随机 + n_iter EI 选点)
  - continuous_plan_with_hybrid_bo: 在 hybrid SCM 上找使 query_node
    最小的干预值 (L3 反事实口径, 锚定 factual 噪声)

纯 numpy, 无额外依赖。d ≤ 4 时 RBF-GP 解析解代价可忽略。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# RBF-GP 回归
# ---------------------------------------------------------------------------

class RBFGP:
    """RBF 核高斯过程回归 (零均值先验, 闭式后验)。"""

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0,
                 noise: float = 1e-3):
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.noise = float(noise)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] \
            - 2.0 * A @ B.T
        return self.sigma_f ** 2 * np.exp(-sq / (2.0 * self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFGP":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        K = self._kernel(X, X) + self.noise ** 2 * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self.X, self.y = X, y
        return self

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (后验均值, 后验方差)。"""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        if self.X is None or len(self.X) == 0:
            return np.zeros(len(Xs)), np.full(len(Xs), self.sigma_f ** 2)
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = self.sigma_f ** 2 - np.sum(v ** 2, axis=0)
        return mu, np.maximum(var, 1e-12)

    # ── EI 采集 ─────────────────────────────────────────────────────
    def ei(self, Xs: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
        """期望改进: E[max(f*−ξ − f(x), 0)]。"""
        mu, var = self.predict(Xs)
        sigma = np.sqrt(var)
        imp = (best - xi) - mu
        with np.errstate(divide="ignore", invalid="ignore"):
            z = imp / sigma
            ei = imp * _Phi(z) + sigma * _Phi_prime(z)
        return np.maximum(ei, 0.0)


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))


def _Phi_prime(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# 通用 BO
# ---------------------------------------------------------------------------

def bayesian_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    n_init: int = 3,
    n_iter: int = 8,
    seed: int = 0,
    length_scale: float = 1.0,
    noise: float = 1e-3,
    minimize: bool = True,
    early_stop_rounds: int = 0,
    ei_stop: float | None = None,
) -> Dict[str, Any]:
    """在 box 约束上最小化/最大化 objective (RBF-GP + EI)。

    early_stop_rounds: 连续 N 轮无改进 → 提前停止 (>0 启用)。
    ei_stop: 最大 EI 改进期望 < 阈值 → 提前停止 (None 禁用)。

    Returns: {best_x, best_y, xs, ys, n_calls, early_stopped, rounds_done}。
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    def _norm(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - lo) / np.maximum(hi - lo, 1e-12)

    def _denorm(u: np.ndarray) -> np.ndarray:
        return lo + u * np.maximum(hi - lo, 1e-12)

    sign = 1.0 if minimize else -1.0

    # 初始点: 拉丁超立方近似 (分层均匀)
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for i in range(n_init):
        u = (rng.random(dim) + i) / n_init
        x = _denorm(u)
        xs.append(x)
        ys.append(sign * float(objective(x)))

    gp = RBFGP(length_scale=length_scale, noise=noise)
    early_stopped = False
    no_improve = 0
    rounds_done = 0
    for _ in range(n_iter):
        gp.fit(np.vstack([_norm(x) for x in xs]), np.array(ys))
        best = float(min(ys))
        # 候选: 当前最优附近 + 均匀网格 + 随机
        cand = [xs[int(np.argmin(ys))]]
        grid_pts = np.linspace(0.0, 1.0, 11)
        if dim == 1:
            cand += [_denorm(np.array([g])) for g in grid_pts]
        else:
            cand += [_denorm(rng.random(dim)) for _ in range(50)]
        vals = gp.ei(np.vstack([_norm(c) for c in cand]), best, xi=0.01)
        best_ei = float(np.max(vals))
        x_next = cand[int(np.argmax(vals))]
        y_next = sign * float(objective(x_next))
        xs.append(x_next)
        ys.append(y_next)
        rounds_done += 1

        if y_next < best - 1e-12:
            no_improve = 0
        else:
            no_improve += 1
        if ei_stop is not None and best_ei < ei_stop:
            early_stopped = True
            break
        if early_stop_rounds > 0 and no_improve >= early_stop_rounds:
            early_stopped = True
            break

    best_i = int(np.argmin(ys))
    return {"best_x": xs[best_i], "best_y": sign * ys[best_i],
            "xs": xs, "ys": [sign * v for v in ys], "n_calls": len(xs),
            "early_stopped": early_stopped, "rounds_done": rounds_done}


# ---------------------------------------------------------------------------
# Hybrid SCM 上的连续规划
# ---------------------------------------------------------------------------

def continuous_plan_with_hybrid_bo(
    scm: Any,
    factual: Dict[str, np.ndarray],
    query_node: str,
    bounds_map: Dict[str, Tuple[float, float]],
    n_init: int = 3,
    n_iter: int = 8,
    seed: int = 0,
    minimize: bool = True,
    length_scale: float = 1.0,
) -> Dict[str, Any]:
    """在 hybrid SCM 上做 BO: 找使 query_node 最优的干预值。

    目标函数 = query_node 的 L3 反事实 (锚定 factual 噪声, do(干预节点=x)):
      objective(x) = scm.l3_counterfactual(factual, {node: [x_i]}, query_node)[0]

    bounds_map: {干预节点: (lo, hi)} — 多节点联合优化 (维度 = len(bounds_map))。
    query_node 取第 0 维作为标量目标 (dim≥2 时注明)。

    Returns: {best_intervention: {node: value}, best_value, trace, n_calls}。
    """
    nodes = list(bounds_map.keys())
    bounds = [bounds_map[k] for k in nodes]

    def objective(x: np.ndarray) -> float:
        intervened = {nd: [float(x[i])] for i, nd in enumerate(nodes)}
        cf = scm.l3_counterfactual(factual, intervened, query_node)
        return float(np.asarray(cf, dtype=float)[0])

    res = bayesian_optimize(objective, bounds, n_init=n_init, n_iter=n_iter,
                            seed=seed, length_scale=length_scale,
                            minimize=minimize)
    best_intervention = {nd: float(res["best_x"][i]) for i, nd in enumerate(nodes)}
    return {
        "best_intervention": best_intervention,
        "best_value": res["best_y"],
        "trace": [{"x": dict(zip(nodes, map(float, xv))), "y": float(yv)}
                  for xv, yv in zip(res["xs"], res["ys"])],
        "n_calls": res["n_calls"],
        "query_node": query_node,
        "bounds": bounds_map,
    }


__all__ = ["RBFGP", "bayesian_optimize", "continuous_plan_with_hybrid_bo"]
