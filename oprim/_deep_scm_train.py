"""oprim._deep_scm_train — 深度 SCM 训练课表与温度校准。

fit_deep_scm:
  HybridSCM.fit 的深度训练包装 — lr_schedule (cosine|step|none) 与
  grad_clip (全局范数裁剪) 透传给每个连续节点机制的 fit_mlp。

calibrate_deep_scm_temperature:
  经验温度校准 — 对每个连续节点在验证集上网格搜 T, 最小化
  NLL_T = mean(log p_T(x|pa)), log p_T = log p / T。
  T<1 修正过自信 (峰值过高), T>1 修正欠自信。返回最优 T 与前后 NLL。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from oprim._hybrid_scm import HybridSCM


def fit_deep_scm(scm: HybridSCM, data: Dict[str, np.ndarray], *,
                 epochs: int = 100, lr: float = 0.05,
                 lr_schedule: str = "cosine", grad_clip: float = 1.0,
                 kappa_reg: Optional[float] = None, kappa_max: float = 50.0,
                 seed: int = 0, hidden: int = 8,
                 return_stats: bool = False):
    """深度 SCM 训练: 学习率课表 + 梯度裁剪 (透传所有节点)。"""
    return scm.fit(data, epochs=epochs, lr=lr, seed=seed, hidden=hidden,
                   kappa_reg=kappa_reg, kappa_max=kappa_max,
                   lr_schedule=lr_schedule, grad_clip=grad_clip,
                   return_stats=return_stats)


def calibrate_deep_scm_temperature(
    scm: HybridSCM,
    data: Dict[str, np.ndarray],
    *,
    grid: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    seed: int = 0,
) -> Dict[str, Any]:
    """温度校准 (矩校准): T* = mean‖u‖²/d, u = L⁻¹(x−μ)。

    对连续密度, "缩放 NLL" 的网格搜索退化 (T→∞ 恒最优);
    正确的校准量是预测分布的方差矩: 良好校准时 E‖u‖²/d ≈ 1。
      T* > 1 → 欠自信 (残差比模型预期大), 需放大协方差
      T* < 1 → 过自信 (模型峰值过尖)
    T* 吸附到 grid 最近点 (保持温度档位语义)。

    Returns: {temperature, nll_before, nll_after, per_node, grid}。
    """
    per_node: Dict[str, Dict[str, float]] = {}
    chi_sum = 0.0
    chi_w = 0.0

    for node, mech in scm.mechanisms.items():
        if mech is None:
            continue
        spec = scm.specs.get(node)
        if spec is None or spec.kind != "continuous":
            continue
        x = np.atleast_2d(np.asarray(data[node], dtype=float))
        n = x.shape[0]
        pa_rows = []
        for i in range(n):
            vals = {p: np.asarray(data[p])[i]
                    for p in scm.dag.predecessors(node)}
            pa_rows.append(scm._encode_parents(node, vals))
        pa = np.vstack(pa_rows) if pa_rows else np.zeros((n, 0))

        # 残差白化: u = L⁻¹(x−μ); 矩统计 χ = mean‖u‖²/d
        us = np.stack([mech.invert(pa[i], x[i]) for i in range(n)])
        chi = float(np.mean(np.sum(us ** 2, axis=1))) / spec.dim
        t_raw = max(0.1, chi)
        t = float(min(grid, key=lambda g: abs(g - t_raw)))   # 吸附网格
        nll_before = float(-np.mean([mech.log_prob(pa[i], x[i]) for i in range(n)]))
        per_node[node] = {"temperature": t, "chi2_per_dim": round(chi, 4),
                          "nll_before": round(nll_before, 4),
                          "nll_after": round(nll_before / t, 4)}
        chi_sum += chi * n
        chi_w += n

    global_chi = chi_sum / max(chi_w, 1.0)
    t_global = float(min(grid, key=lambda g: abs(g - global_chi)))
    return {
        "temperature": t_global,
        "chi2_per_dim": round(global_chi, 4),
        "nll_before": round(sum(p["nll_before"] for p in per_node.values()), 4),
        "nll_after": round(sum(p["nll_after"] for p in per_node.values()), 4),
        "per_node": per_node,
        "grid": list(grid),
    }


__all__ = ["calibrate_deep_scm_temperature", "fit_deep_scm"]
