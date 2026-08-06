"""oprim._hybrid_scm — 混合离散–连续 SCM (NodeSpec / build / fit / 溯因 / 仿真 / CF)。

连续节点机制 (continuous_mech):
  "diagonal" — 条件对角仿射 X = μ(PA) + σ(PA)⊙U (逐维, 无残差相关)
  "cholesky" — 条件 Cholesky 仿射 X = μ(PA) + L(PA)·U (残差相关;
               diag_floor / offdiag_scale / κ 正则见 _cholesky_flow)

离散父节点: 拟合时 one-hot 编码进 PA; 仿真时按经验分布采样。
与 do-calculus / 二进制故障 SCM 互补: 本模块管连续指标向量上的
拟合 → 溯因 (u = L⁻¹(x−μ)) → 反事实 (do + 夹持 U) 全链路。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from oprim._cholesky_flow import CholeskyMechanism


@dataclass
class NodeSpec:
    """混合 SCM 节点规格。"""

    name: str
    kind: str = "continuous"              # continuous | discrete
    dim: int = 1
    continuous_mech: str = "diagonal"     # diagonal | cholesky
    diag_floor: float = 1e-4
    offdiag_scale: float = 0.5


class HybridSCM:
    """混合离散–连续 SCM: DAG 序传播 + 逐节点条件机制。"""

    def __init__(self, dag: nx.DiGraph, specs: dict[str, NodeSpec]):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Hybrid SCM 必须是 DAG")
        self.dag = dag
        self.specs = dict(specs)
        self.order = list(nx.topological_sort(dag))
        self.mechanisms: dict[str, CholeskyMechanism | None] = {
            n: None for n in self.order}
        # 离散父节点: 类别表 (fit 时从数据解析) + 经验分布 (仿真用)
        self._categories: dict[str, list[Any]] = {}
        self._pmf: dict[str, np.ndarray] = {}

    # ── 离散父编码 ──────────────────────────────────────────────────
    @staticmethod
    def _scalar(x: Any) -> Any:
        """标量归一: size-1 数组/标量统一成标量 (np.array_equal 要求同形状)。"""
        a = np.asarray(x)
        return a.item() if a.size == 1 else a

    def _is_discrete(self, node: str) -> bool:
        spec = self.specs.get(node)
        return spec is not None and spec.kind == "discrete"

    def _encode_parents(self, node: str, values: dict[str, np.ndarray]) -> np.ndarray:
        """PA 向量: 连续父取值拼接 + 离散父 one-hot。"""
        parts: list[np.ndarray] = []
        for p in self.dag.predecessors(node):
            if self._is_discrete(p):
                cats = self._categories.get(p, [])
                v = values[p]
                row = np.zeros(len(cats))
                if cats:
                    vv = self._scalar(v)
                    for ci, c in enumerate(cats):
                        if np.array_equal(vv, self._scalar(c)) or str(vv) == str(c):
                            row[ci] = 1.0
                            break
                parts.append(row)
            else:
                parts.append(np.asarray(values[p], dtype=float).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    # ── 拟合 ────────────────────────────────────────────────────────
    def fit(self, data: dict[str, np.ndarray], *,
            epochs: int = 100, kappa_reg: float | None = None,
            kappa_max: float = 50.0, lr: float = 0.05, seed: int = 0,
            hidden: int = 8, shrinkage: str = "sample",
            return_stats: bool = False,
            exact_kappa_every: int = 10,
            lr_schedule: str = "none", grad_clip: float = 0.0):
        """逐节点拟合条件机制 (拓扑序)。

        data[node] = (n, dim) 观测; 离散节点自动解析类别 + 经验分布。
        连续节点: cholesky → fit_mlp (epochs>0) 或 fit_linear;
                  diagonal → fit_mlp(diag_only) 或 fit_linear(diag_only)。

        kappa_reg=None → 自动: 存在 Cholesky 节点时 0.1, 否则 0.0。
        shrinkage: "sample" | "lw" (Ledoit–Wolf, 闭式路径小样本稳定化)。
        return_stats=True → (self, losses, stats): losses 逐 epoch 平均 NLL
        (跨节点对齐), stats = {kappa_reg, min_diag, mean_kappa_proxy,
        mean_kappa_penalty} (研究摘要 §6 训练监控)。
        """
        has_chol = any(s.continuous_mech == "cholesky"
                       for s in self.specs.values() if s.kind == "continuous")
        if kappa_reg is None:
            kappa_reg = 0.1 if has_chol else 0.0

        # 离散父: 解析类别
        for node in self.order:
            if self._is_discrete(node):
                vals = np.asarray(data[node])
                cats: list[Any] = []
                for v in vals:
                    if not any(np.array_equal(v, c) for c in cats):
                        cats.append(v)
                self._categories[node] = cats
                if len(cats):
                    counts = np.zeros(len(cats))
                    for v in vals:
                        for ci, c in enumerate(cats):
                            if np.array_equal(v, c):
                                counts[ci] += 1
                                break
                    self._pmf[node] = counts / counts.sum()

        # 连续节点: 拟合机制
        all_losses: list[Optional[list[float]]] = [None] * len(self.order)
        all_stats: list[Optional[dict[str, float]]] = [None] * len(self.order)
        for ni, node in enumerate(self.order):
            spec = self.specs.get(node)
            if spec is None or spec.kind != "continuous":
                continue
            x = np.atleast_2d(np.asarray(data[node], dtype=float))
            n = x.shape[0]
            # PA 矩阵: 逐行编码 (离散父 one-hot)
            pa_rows = []
            for i in range(n):
                vals = {p: np.asarray(data[p])[i]
                        for p in self.dag.predecessors(node)}
                pa_rows.append(self._encode_parents(node, vals))
            pa = np.vstack(pa_rows) if pa_rows else np.zeros((n, 0))

            cholesky = spec.continuous_mech == "cholesky"
            coupling = spec.continuous_mech == "coupling"
            if coupling and spec.dim < 2:
                raise ValueError(f"耦合流需要 dim ≥ 2, {node} 的 dim={spec.dim}")
            if coupling:
                # 条件仿射耦合机制 (dim ≥ 2; 单层解析梯度拟合)
                from oprim._coupling_flow import ConditionalCouplingMechanism

                mech = ConditionalCouplingMechanism(
                    spec.dim, pa.shape[1], hidden=hidden, seed=seed)
                if epochs and epochs > 0:
                    mech.fit_mlp(pa, x, epochs=epochs, lr=lr * 0.4, seed=seed,
                                 grad_clip=grad_clip if grad_clip > 0 else 1.0)
                    all_losses[ni] = [mech._train_nll]
                    all_stats[ni] = {"min_diag": 0.0, "mean_kappa_proxy": 0.0,
                                     "mean_kappa_penalty": 0.0,
                                     "mean_kappa_exact": 0.0,
                                     "mean_exact_kappa": 0.0}
            elif epochs and epochs > 0:
                mech, losses, stats = CholeskyMechanism.fit_mlp(
                    pa, x, epochs=epochs, lr=lr, seed=seed, hidden=hidden,
                    diag_floor=spec.diag_floor, offdiag_scale=spec.offdiag_scale,
                    diag_only=not cholesky, kappa_reg=kappa_reg, kappa_max=kappa_max,
                    return_stats=True, exact_kappa_every=exact_kappa_every,
                    lr_schedule=lr_schedule, grad_clip=grad_clip)
                all_losses[ni] = losses
                all_stats[ni] = stats
            else:
                mech = CholeskyMechanism.fit_linear(
                    pa, x, diag_only=not cholesky, diag_floor=spec.diag_floor,
                    shrinkage=shrinkage)
            self.mechanisms[node] = mech

        if not return_stats:
            return self
        # 跨节点逐 epoch 平均 (无 MLP 的节点跳过)
        n_epochs = max((len(l) for l in all_losses if l), default=0)
        losses: list[float] = []
        for e in range(n_epochs):
            vals = [l[e] for l in all_losses if l and e < len(l)]
            losses.append(float(np.mean(vals)) if vals else 0.0)
        stats = {
            "kappa_reg": float(kappa_reg),
            "min_diag": min((st["min_diag"] for st in all_stats if st),
                            default=float("inf")),
            "mean_kappa_proxy": float(np.mean([st["mean_kappa_proxy"]
                                               for st in all_stats if st]))
            if any(all_stats) else 0.0,
            "mean_kappa_penalty": float(np.mean([st["mean_kappa_penalty"]
                                                 for st in all_stats if st]))
            if any(all_stats) else 0.0,
            "mean_kappa_exact": float(np.mean([st["mean_kappa_exact"]
                                               for st in all_stats if st]))
            if any(all_stats) else 0.0,
            "mean_exact_kappa": float(np.mean([st["mean_kappa_exact"]
                                               for st in all_stats if st]))
            if any(all_stats) else 0.0,
        }
        return self, losses, stats

    # ── 1. Abduction: u = L⁻¹(x − μ(PA)) ────────────────────────────
    def abduct(self, evidence: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """连续观测节点反演噪声; 未观测连续节点 u=0 (MAP) 并传播期望。"""
        values: dict[str, np.ndarray] = {}
        u_map: dict[str, np.ndarray] = {}
        for node in self.order:
            spec = self.specs.get(node)
            if spec is not None and spec.kind == "discrete":
                values[node] = np.asarray(evidence.get(node, self._pmf_mode(node)))
                continue
            mech = self.mechanisms.get(node)
            pa_vals = self._encode_parents(node, values) if mech is not None else np.zeros(0)
            if node in evidence:
                x = np.asarray(evidence[node], dtype=float)
                values[node] = x
                u_map[node] = _mech_invert(mech, pa_vals, x) if mech is not None \
                    else np.zeros(spec.dim)
            else:
                values[node] = _mech_typical(mech, pa_vals) if mech is not None \
                    else np.zeros(spec.dim)
                u_map[node] = np.zeros(spec.dim)
        return u_map

    def _pmf_mode(self, node: str) -> np.ndarray:
        cats = self._categories.get(node, [])
        if not cats:
            return np.zeros(1)
        return np.asarray(cats[int(np.argmax(self._pmf.get(node, [1.0] * len(cats))))])

    # ── 2+3. Action + Prediction / Simulation ───────────────────────
    def predict(self, node: str, *,
                intervened: dict[str, Sequence[float]] | None = None,
                u_map: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """夹持溯因 U 拓扑传播: 干预节点取固定值, 其余 x = μ(PA) + L(PA)·u。"""
        intervened = intervened or {}
        u_map = u_map or {n: np.zeros(self.specs[n].dim)
                          for n in self.order if n in self.specs}
        values: dict[str, np.ndarray] = {}
        for name in self.order:
            spec = self.specs.get(name)
            if name in intervened:
                values[name] = np.asarray(intervened[name], dtype=float)
                continue
            if spec is not None and spec.kind == "discrete":
                values[name] = np.asarray(intervened.get(name) or self._pmf_mode(name))
                continue
            mech = self.mechanisms.get(name)
            if mech is None:
                continue
            pa_vals = self._encode_parents(name, values)
            u = u_map.get(name, np.zeros(spec.dim))
            values[name] = _mech_apply(mech, pa_vals, u)
        return np.asarray(values[node], dtype=float)

    def l3_counterfactual(self, evidence: dict[str, np.ndarray],
                          intervened: dict[str, Sequence[float]],
                          node: str) -> np.ndarray:
        """L3: 锚定本次观测噪声, 问"若当时 do(X=v), node 会是什么"。"""
        u_map = self.abduct(evidence)
        return self.predict(node, intervened=intervened, u_map=u_map)

    def simulate(self, *, n_samples: int = 1, seed: int = 0,
                 intervened: dict[str, Sequence[float]] | None = None
                 ) -> dict[str, np.ndarray]:
        """前向采样: 离散节点按经验分布, 连续节点 x = μ + L·u, u~N(0,I)。"""
        rng = np.random.default_rng(seed)
        intervened = intervened or {}
        samples: dict[str, np.ndarray] = {}
        for name in self.order:
            spec = self.specs.get(name)
            if name in intervened:
                v = np.asarray(intervened[name], dtype=float)
                samples[name] = np.tile(v, (n_samples, 1)) if np.ndim(v) else \
                    np.full((n_samples, 1), v)
                continue
            if spec is not None and spec.kind == "discrete":
                cats = self._categories.get(name, [])
                pmf = self._pmf.get(name)
                idx = rng.choice(len(cats), size=n_samples, p=pmf) if cats and pmf is not None \
                    else np.zeros(n_samples, dtype=int)
                samples[name] = np.array([np.asarray(cats[i], dtype=float) for i in idx])
                continue
            mech = self.mechanisms.get(name)
            if mech is None:
                continue
            out = np.zeros((n_samples, spec.dim))
            for i in range(n_samples):
                pa_vals = self._encode_parents(
                    name, {p: samples[p][i] for p in self.dag.predecessors(name)})
                out[i] = mech.sample(pa_vals, rng=rng)
            samples[name] = out
        return samples


def _mech_invert(mech: Any, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
    """机制反演: CholeskyMechanism.invert / ConditionalCouplingMechanism.inverse。"""
    return mech.invert(pa, x)


def _mech_typical(mech: Any, pa: np.ndarray) -> np.ndarray:
    """机制的典型值: Cholesky → mean; 耦合流 → u=0 映射。"""
    return mech.mean(pa)


def _mech_apply(mech: Any, pa: np.ndarray, u: np.ndarray) -> np.ndarray:
    """机制正演: Cholesky → μ+Lu; 耦合流 → forward(pa, u)。"""
    if hasattr(mech, "forward"):
        return mech.forward(pa, u)[0]
    return mech.mean(pa) + mech.chol(pa) @ u


def build_hybrid_scm(dag: nx.DiGraph, specs: dict[str, NodeSpec]) -> HybridSCM:
    """按规格构建混合 SCM (机制槽位待 fit)。"""
    return HybridSCM(dag, specs)


def fit_hybrid_scm(scm: HybridSCM, data: dict[str, np.ndarray], *,
                   epochs: int = 100, kappa_reg: float | None = None,
                   kappa_max: float = 50.0, lr: float = 0.05, seed: int = 0,
                   hidden: int = 8, shrinkage: str = "sample",
                   return_stats: bool = False,
                   exact_kappa_every: int = 10,
                   lr_schedule: str = "none", grad_clip: float = 0.0):
    """拟合混合 SCM。

    kappa_reg=None → 自动: 存在 Cholesky 节点时 0.1, 否则 0.0 (仅 Cholesky 节点受罚)。
    shrinkage: "sample" | "lw" (Ledoit–Wolf, 闭式路径)。
    exact_kappa_every: 真 κ (SVD) 低频监控间隔 (return_stats 时生效)。
    lr_schedule / grad_clip: 深度训练课表 (cosine|step|none) 与梯度裁剪。
    return_stats=True → (scm, losses, stats) 训练监控。
    """
    return scm.fit(data, epochs=epochs, kappa_reg=kappa_reg, kappa_max=kappa_max,
                   lr=lr, seed=seed, hidden=hidden, shrinkage=shrinkage,
                   return_stats=return_stats, exact_kappa_every=exact_kappa_every,
                   lr_schedule=lr_schedule, grad_clip=grad_clip)


__all__ = ["HybridSCM", "NodeSpec", "build_hybrid_scm", "fit_hybrid_scm"]
