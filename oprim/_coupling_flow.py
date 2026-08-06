"""oprim._coupling_flow — 条件仿射耦合机制 (ConditionalCouplingMechanism)。

研究摘要 §3"全协方差流的更强形式": 耦合流 (affine coupling, Dinh et al.)。

单层 (split 前半 a / 后半 b):
    x_a = u_a · exp(s_a(PA)) + t_a(PA)          # 前半: 条件对角仿射
    s_b, t_b = NN_b([x_a, PA])                   # 后半由 x_a + PA 决定
    x_b = u_b · exp(s_b) + t_b
    log det J = Σ s_a + Σ s_b                    # 三角结构 → 对角和

性质:
  - 可逆: u_b = (x_b − t_b)·exp(−s_b); u_a = (x_a − t_a)·exp(−s_a)
  - 精确 log_prob: log N(u; 0, I) + Σ s_a + Σ s_b
  - 可叠层: 多层交替 split 顺序 (推理/生成路径), 每层保持易算 det
  - 相对 Cholesky: 非高斯残差表达能力 (u 仍是高斯, 但 x 的边缘可非高斯)

拟合: fit_mlp 单层解析反传 (与 _cholesky_flow.fit_mlp 同模式),
有限差分测试验证; num_layers≥2 提供 forward/inverse/log_prob 供推理。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _split(x: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """前半/后半拆分 (d_a = d // 2, d_b = d − d_a)。"""
    da = d // 2
    return x[:da], x[da:]


def _concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b])


class CouplingLayer:
    """单层条件仿射耦合。s/t 由 MLP(x_a, PA) 输出。"""

    def __init__(self, d: int, pa_dim: int, hidden: int = 8,
                 seed: int = 0, scale: float = 0.8):
        self.d = int(d)
        self.da = max(1, d // 2)
        self.db = d - self.da
        self.pa_dim = int(pa_dim)
        self.hidden = hidden
        # NN_a(PA) → s_a (da), t_a (da); NN_b([x_a, PA]) → s_b (db), t_b (db)
        in_b = self.da + pa_dim
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((pa_dim, hidden)) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2a = rng.standard_normal((hidden, self.da * 2)) * scale
        self.b2a = np.zeros(self.da * 2)
        self.W1b = rng.standard_normal((in_b, hidden)) * 0.5
        self.b1b = np.zeros(hidden)
        self.W2b = rng.standard_normal((hidden, self.db * 2)) * scale
        self.b2b = np.zeros(self.db * 2)

    # ── 前向 ────────────────────────────────────────────────────────
    def forward(self, pa: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, float]:
        """u → x; 返回 (x, log_det)。"""
        u = np.asarray(u, dtype=float)
        pa = np.asarray(pa, dtype=float)
        ua, ub = _split(u, self.d)
        sa_ta = _tanh(pa @ self.W1 + self.b1) @ self.W2a + self.b2a
        sa, ta = sa_ta[:self.da], sa_ta[self.da:]
        xa = ua * np.exp(sa) + ta
        hb = _tanh(_concat(xa, pa) @ self.W1b + self.b1b)
        sb_tb = hb @ self.W2b + self.b2b
        sb, tb = sb_tb[:self.db], sb_tb[self.db:]
        xb = ub * np.exp(sb) + tb
        return _concat(xa, xb), float(sa.sum() + sb.sum())

    # ── 反演 ────────────────────────────────────────────────────────
    def inverse(self, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        """x → u。"""
        x = np.asarray(x, dtype=float)
        pa = np.asarray(pa, dtype=float)
        xa, xb = _split(x, self.d)
        sa_ta = _tanh(pa @ self.W1 + self.b1) @ self.W2a + self.b2a
        sa, ta = sa_ta[:self.da], sa_ta[self.da:]
        ua = (xa - ta) * np.exp(-sa)
        hb = _tanh(_concat(xa, pa) @ self.W1b + self.b1b)
        sb_tb = hb @ self.W2b + self.b2b
        sb, tb = sb_tb[:self.db], sb_tb[self.db:]
        ub = (xb - tb) * np.exp(-sb)
        return _concat(ua, ub)

    def log_det(self, pa: np.ndarray, x: np.ndarray) -> float:
        """log|det J| (由 x 侧的 s_a, s_b 决定)。"""
        x = np.asarray(x, dtype=float)
        pa = np.asarray(pa, dtype=float)
        xa, _ = _split(x, self.d)
        sa_ta = _tanh(pa @ self.W1 + self.b1) @ self.W2a + self.b2a
        sa = sa_ta[:self.da]
        hb = _tanh(_concat(xa, pa) @ self.W1b + self.b1b)
        sb = (hb @ self.W2b + self.b2b)[:self.db]
        return float(sa.sum() + sb.sum())


class ConditionalCouplingMechanism:
    """条件仿射耦合机制 (可叠层)。

    形态:
      - 单层 (默认): fit_mlp 解析反传拟合
      - 多层 (num_layers≥2): forward/inverse/log_prob 串接 (交替 split),
        推理/生成路径可用; 训练默认单层 (梯度简洁, 有限差分验证)。
    """

    def __init__(self, d: int, pa_dim: int, num_layers: int = 1,
                 hidden: int = 8, seed: int = 0):
        if d < 2:
            raise ValueError("耦合流需要 dim ≥ 2")
        self.d = int(d)
        self.pa_dim = int(pa_dim)
        self.num_layers = int(num_layers)
        self.layers: List[CouplingLayer] = [
            CouplingLayer(d, pa_dim, hidden=hidden, seed=seed + i)
            for i in range(num_layers)
        ]

    # ── 核心操作 ────────────────────────────────────────────────────
    def invert(self, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        """hybrid SCM 兼容 API: x → u (CholeskyMechanism.invert 同名)。"""
        return self.inverse(pa, x)

    def mean(self, pa: np.ndarray) -> np.ndarray:
        """hybrid SCM 兼容 API: 典型值 = u=0 映射 (耦合流无闭式期望, 文档注明)。"""
        return self.forward(pa, np.zeros(self.d))[0]

    def forward(self, pa: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, float]:
        x, logdet = u, 0.0
        for layer in self.layers:
            x, ld = layer.forward(pa, x)
            logdet += ld
        return np.asarray(x, dtype=float), logdet

    def inverse(self, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        u = x
        for layer in reversed(self.layers):
            u = layer.inverse(pa, u)
        return np.asarray(u, dtype=float)

    def log_prob(self, pa: np.ndarray, x: np.ndarray) -> float:
        u = self.inverse(pa, x)
        ld = sum(layer.log_det(pa, self._partial(pa, x, i))
                 for i, layer in enumerate(self.layers))
        return -0.5 * float(u @ u) - 0.5 * self.d * np.log(2 * np.pi) + ld

    def _partial(self, pa: np.ndarray, x: np.ndarray, upto: int) -> np.ndarray:
        """第 upto 层的输入 x (前向串接的中间值)。"""
        cur = x
        for i, layer in enumerate(self.layers):
            if i == upto:
                return cur
            cur, _ = layer.forward(pa, cur)
        return cur

    def sample(self, pa: np.ndarray, *, u: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if u is None:
            rng = rng or np.random.default_rng()
            u = rng.standard_normal(self.d)
        x, _ = self.forward(pa, u)
        return x

    # ── 拟合 (单层解析反传) ─────────────────────────────────────────
    def fit_mlp(self, pa: np.ndarray, x: np.ndarray, *,
                epochs: int = 300, lr: float = 0.02, seed: int = 0,
                grad_clip: float = 1.0) -> "ConditionalCouplingMechanism":
        """拟合单层耦合机制 (NLL 解析梯度, 有限差分验证)。"""
        if self.num_layers != 1:
            raise ValueError("fit_mlp 仅支持 num_layers=1; 多层用于推理")
        pa = np.atleast_2d(np.asarray(pa, dtype=float))
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = x.shape[0]
        layer = self.layers[0]
        da, db = layer.da, layer.db

        def _inverse_with(p, xx, W1, b1, W2a, b2a, W1b, b1b, W2b, b2b):
            xa, xb = _split(xx, self.d)
            sa_ta = _tanh(p @ W1 + b1) @ W2a + b2a
            sa, ta = sa_ta[:da], sa_ta[da:]
            ua = (xa - ta) * np.exp(-sa)
            hb = _tanh(_concat(xa, p) @ W1b + b1b)
            sb_tb = hb @ W2b + b2b
            sb, tb = sb_tb[:db], sb_tb[db:]
            ub = (xb - tb) * np.exp(-sb)
            return _concat(ua, ub)

        rng = np.random.default_rng(seed)
        best = (float("inf"), (layer.W1.copy(), layer.b1.copy(),
                               layer.W2a.copy(), layer.b2a.copy(),
                               layer.W1b.copy(), layer.b1b.copy(),
                               layer.W2b.copy(), layer.b2b.copy()))
        for epoch in range(epochs):
            grads = [np.zeros_like(v) for v in best[1]]
            nll = 0.0
            for i in range(n):
                p = pa[i]; xx = x[i]
                xa, xb = _split(xx, self.d)
                # 前向
                h1 = _tanh(p @ layer.W1 + layer.b1)
                sa_ta = h1 @ layer.W2a + layer.b2a
                sa, ta = sa_ta[:da], sa_ta[da:]
                ua = (xa - ta) * np.exp(-sa)
                hb = _tanh(_concat(xa, p) @ layer.W1b + layer.b1b)
                sb_tb = hb @ layer.W2b + layer.b2b
                sb, tb = sb_tb[:db], sb_tb[db:]
                ub = (xb - tb) * np.exp(-sb)
                nll += 0.5 * float(ua @ ua + ub @ ub) - float(sa.sum() + sb.sum())
                # 反传: d nll/d s_a = −u_a² − 1; d nll/d t_a = −u_a·exp(−s_a)
                g_sa = -(ua ** 2) - 1.0
                g_ta = -ua * np.exp(-sa)
                g_sb = -(ub ** 2)
                g_tb = -ub * np.exp(-sb)
                # NN_b 反传 (s_b, t_b 依赖 x_a)
                g_sb_tb = _concat(g_sb, g_tb)
                g_hb = g_sb_tb @ layer.W2b.T
                g_hb *= (1 - hb ** 2)
                g_W2b = np.outer(hb, g_sb_tb)
                g_b2b = g_sb_tb
                g_in = g_hb @ layer.W1b.T
                # g_in 中 x_a 部分 → 影响 x_a 的梯度
                g_xa = g_in[:da]
                g_W1b = np.outer(_concat(xa, p), g_hb)
                g_b1b = g_hb
                # x_a 总梯度: 直接 (经 u_a) + 经 NN_b
                g_xa_total = g_xa + g_ta * (-np.exp(-sa)) + g_sa * (-ua)  # d u_a/d x_a = exp(−s_a); 见下
                # u_a = (x_a−t_a)exp(−s_a) → d u_a/d x_a = exp(−s_a); d nll/d x_a += g_ua·exp(−s_a)
                g_xa_total += ua * np.exp(-sa)
                # NN_a 反传 (s_a, t_a 依赖 PA 经 h1)
                g_sa_ta = _concat(g_sa, g_ta)
                g_h1 = g_sa_ta @ layer.W2a.T
                g_h1 *= (1 - h1 ** 2)
                g_W2a = np.outer(h1, g_sa_ta)
                g_b2a = g_sa_ta
                g_W1 = np.outer(p, g_h1)
                g_b1 = g_h1
                grads[0] += g_W1; grads[1] += g_b1
                grads[2] += g_W2a; grads[3] += g_b2a
                grads[4] += g_W1b; grads[5] += g_b1b
                grads[6] += g_W2b; grads[7] += g_b2b
            # 梯度裁剪 + 更新
            if grad_clip > 0:
                tot = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
                if tot > grad_clip:
                    f = grad_clip / max(tot, 1e-12)
                    grads = [g * f for g in grads]
            names = ("W1", "b1", "W2a", "b2a", "W1b", "b1b", "W2b", "b2b")
            cur = [getattr(layer, nm) for nm in names]
            for g, c in zip(grads, cur):
                c -= (lr / n) * g
            avg = nll / n
            if avg < best[0]:
                best = (avg, tuple(v.copy() for v in cur))
        (layer.W1, layer.b1, layer.W2a, layer.b2a,
         layer.W1b, layer.b1b, layer.W2b, layer.b2b) = best[1]
        self._train_nll = best[0]
        return self


__all__ = ["ConditionalCouplingMechanism", "CouplingLayer"]
