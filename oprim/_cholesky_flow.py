"""oprim._cholesky_flow — 条件 Cholesky 仿射机制 (Cholesky 流 / 条件高斯升级版)。

机制:   X = μ(PA) + L(PA)·U,   U ~ N(0, I)
        L(PA) 下三角且对角 > 0

性质 (研究摘要 §4-5):
  - 采样:   一次三角矩阵-向量乘
  - 反演:   前代法解 L·u = x − μ (不显式求逆, O(d²))
  - 似然:   log p(x|PA) = log N(u; 0, I) − Σ_i log L_ii   (det L = Π L_ii)
  - 条件化: vec(L) 与 μ 都由 PA 网络给出; PA 含离散时 one-hot 即可
  - do():   干预 = 确定性赋值 (割裂父依赖)

数值稳定性 (研究摘要 §4, 结论: 三角求解本身稳定, 不稳定来自 L 病态):
  - diag_floor:   L_ii = softplus(raw) + floor (对角下界, 默认 1e-4)
  - offdiag_scale: 严格下三角 = scale·tanh(off) (限幅, 默认 0.5, 防列尺度失衡)
  - κ 正则:       condition_number_proxy (对角比代理) + kappa_penalty
                  (relu(log κ̂ − log κ_max)), 拟合时可选挂进 loss (子梯度可导)

两种拟合:
  - fit_linear: 闭式 MLE (条件高斯: μ=线性回归, Σ=残差协方差, L=chol Σ)
  - fit_mlp:    单隐层 MLP 输出 (μ(PA), L(PA)), numpy 手写反向传播
                (经 Σ 参数化的精确梯度 + κ 正则子梯度; 测试有限差分验证)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 三角算子 (O(d²), 避免显式求逆)
# ---------------------------------------------------------------------------

def _softplus(v: float) -> float:
    return float(np.log1p(np.exp(min(v, 50.0))))


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-min(v, 50.0)))


def project_lower_triangular(raw: np.ndarray, d: int) -> np.ndarray:
    """unconstrained (d(d+1)/2,) → 下三角 L: 对角 softplus(>0), 严格下三角自由。"""
    L = np.zeros((d, d))
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            v = float(raw[idx])
            idx += 1
            L[i, j] = _softplus(v) if i == j else v
    return L


def project_conditioned_lower_triangular(
    raw: np.ndarray, d: int, *,
    diag_floor: float = 1e-4, offdiag_scale: float = 0.5,
    diag_only: bool = False,
) -> np.ndarray:
    """稳定化投影: L_ii = softplus + floor; 严格下三角 = scale·tanh (限幅)。

    diag_only=True → 对角退化 (条件对角仿射的 L 头, nL = d)。
    """
    L = np.zeros((d, d))
    if diag_only:
        for i in range(d):
            L[i, i] = _softplus(float(raw[i])) + diag_floor
        return L
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            v = float(raw[idx])
            idx += 1
            if i == j:
                L[i, j] = _softplus(v) + diag_floor
            else:
                L[i, j] = offdiag_scale * np.tanh(v)
    return L


def forward_substitute(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """前代法解 L·u = b (L 下三角, 对角 > 0)。"""
    d = len(b)
    u = np.zeros(d)
    for i in range(d):
        u[i] = (b[i] - float(L[i, :i] @ u[:i])) / L[i, i]
    return u


def back_substitute(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """回代法解 Lᵀ·w = b (梯度计算用)。"""
    d = len(b)
    w = np.zeros(d)
    for i in reversed(range(d)):
        w[i] = (b[i] - float(L[i + 1:, i] @ w[i + 1:])) / L[i, i]
    return w


def log_det_lower(L: np.ndarray) -> float:
    """三角矩阵: log|det L| = Σ log L_ii。"""
    return float(np.sum(np.log(np.diag(L))))


def ledoit_wolf_covariance(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf 收缩协方差 (2004): Σ̂ = (1−α)·S + α·μI, α 由风险公式自动选定。

    公式移植自 sklearn.covariance.ledoit_wolf (单块版, 供 p ≤ 1000 使用):
      beta_  = Σ_ij Σ_k x_ki²x_kj²;  delta_ = Σ_ij s_ij² / n²
      beta   = 1/(p·n)·(beta_/n − delta_);  delta = (‖S − μI‖²_F)/p
      α      = min(beta, delta)/delta ∈ [0,1]

    大维/小样本下同时改善统计精度与条件数 (抬最小特征值、压 κ)。
    Returns: (Σ̂, α̂)。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("Ledoit-Wolf 需要 n ≥ 2 且 p ≥ 1")
    Xc = X - X.mean(axis=0)
    X2 = Xc ** 2
    S = Xc.T @ Xc / n
    emp_cov_trace = np.sum(X2, axis=0) / n          # 每维 E[x²] (对角)
    mu = float(np.sum(emp_cov_trace) / p)           # μ = tr(S)/p
    beta_ = float(np.sum(X2.T @ X2))                # Σ_ij Σ_k x_ki²x_kj²
    delta_ = float(np.sum((Xc.T @ Xc) ** 2)) / (n * n)   # Σ_ij s_ij²
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = delta_ - 2.0 * mu * float(emp_cov_trace.sum()) + p * mu ** 2
    delta /= p
    beta = min(beta, delta)                         # 防收缩超过 1 (协方差反号)
    alpha = 0.0 if beta == 0 else float(beta / delta)
    alpha = max(0.0, min(1.0, alpha))
    Sigma = (1.0 - alpha) * S + alpha * mu * np.eye(p)
    return Sigma, alpha


# ---------------------------------------------------------------------------
# 机制
# ---------------------------------------------------------------------------

class CholeskyMechanism:
    """条件 Cholesky 仿射机制 f(PA, U) = μ(PA) + L(PA)·U。

    形态:
      - 函数式 (MLP): mean_fn(pa)→μ, chol_fn(pa)→L
      - 线性闭式:     mean_coef (回归系数) + chol_fixed (常数 L)
      - 确定性干预:   do(value) 后返回退化机制 (X ≡ value)

    稳定化参数 (研究摘要 §4):
      diag_floor / offdiag_scale 控制 L 的良态性; 训练期可用
      condition_number_proxy / kappa_penalty 做条件数监控与正则。
    """

    def __init__(
        self,
        d: int,
        mean_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        chol_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        mean_coef: np.ndarray | None = None,
        chol_fixed: np.ndarray | None = None,
        diag_only: bool = False,
        diag_floor: float = 1e-4,
        offdiag_scale: float = 0.5,
    ):
        self.d = int(d)
        self.mean_fn = mean_fn
        self.chol_fn = chol_fn
        self.mean_coef = mean_coef
        self.chol_fixed = chol_fixed
        self.diag_only = diag_only            # 对角退化 (对照用)
        self.diag_floor = float(diag_floor)
        self.offdiag_scale = float(offdiag_scale)
        self.intervened_value: np.ndarray | None = None

    # ── 参数查询 ────────────────────────────────────────────────────
    def mean(self, pa: np.ndarray) -> np.ndarray:
        if self.intervened_value is not None:
            return np.asarray(self.intervened_value, dtype=float)
        if self.mean_fn is not None:
            return np.asarray(self.mean_fn(np.asarray(pa, dtype=float)), dtype=float)
        Phi = np.concatenate([[1.0], np.asarray(pa, dtype=float)])
        return Phi @ self.mean_coef

    def chol(self, pa: np.ndarray) -> np.ndarray:
        if self.intervened_value is not None:
            return np.zeros((self.d, self.d))
        if self.chol_fn is not None:
            L = np.asarray(self.chol_fn(np.asarray(pa, dtype=float)), dtype=float)
            if self.diag_only:
                L = np.diag(np.diag(L))
            return L
        L = self.chol_fixed
        if self.diag_only:
            L = np.diag(np.diag(L))
        return L

    # ── 条件数监控 / 正则 (研究摘要 §4B) ─────────────────────────────
    def condition_number_proxy(self, pa: np.ndarray | None = None) -> float:
        """廉价 κ 代理 (训练用, 可微):

            κ̂ = (max L_ii / min L_ii) × (1 + mean|L_off| / mean L_ii)

        对角比捕捉尺度失衡, 非对角项捕捉相关贡献 — 避免"两对角同小"漏检。
        """
        L = self.chol(np.zeros(0) if pa is None else pa)
        diag = np.maximum(np.diag(L), self.diag_floor)
        ratio = float(np.max(diag) / max(np.min(diag), 1e-12))
        if self.d > 1:
            off = np.abs(L - np.diag(np.diag(L)))
            off_mean = float(off.sum() / (self.d * (self.d - 1))) if self.d > 1 else 0.0
            ratio *= 1.0 + off_mean / max(float(diag.mean()), 1e-12)
        return ratio

    def exact_condition_number(self, pa: np.ndarray | None = None) -> float:
        """精确 κ₂ (SVD) — 诊断/日志用, 训练不用 (每步 SVD 偏贵)。"""
        L = self.chol(np.zeros(0) if pa is None else pa)
        s = np.linalg.svd(L, compute_uv=False)
        return float(s[0] / max(s[-1], 1e-12))

    def kappa_penalty(self, pa: np.ndarray | None = None,
                      kappa_max: float = 50.0) -> float:
        """relu(log κ̂ − log κ_max): κ̂ 超上限时 > 0, 否则 0 (hinge/软门槛)。"""
        k = self.condition_number_proxy(pa)
        return max(0.0, float(np.log(k) - np.log(kappa_max)))

    # ── 核心操作 ────────────────────────────────────────────────────
    def sample(self, pa: np.ndarray, *, u: np.ndarray | None = None,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """X = μ + L·u; u 缺省采样 N(0, I)。"""
        if self.intervened_value is not None:
            return np.asarray(self.intervened_value, dtype=float)
        if u is None:
            rng = rng or np.random.default_rng()
            u = rng.standard_normal(self.d)
        return self.mean(pa) + self.chol(pa) @ np.asarray(u, dtype=float)

    def invert(self, pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        """u = L⁻¹(x − μ) — 一次三角求解, 不显式求逆。"""
        if self.intervened_value is not None:
            raise ValueError("确定性干预机制不可反演")
        return forward_substitute(self.chol(pa), np.asarray(x, dtype=float) - self.mean(pa))

    def log_prob(self, pa: np.ndarray, x: np.ndarray) -> float:
        """log p(x|PA) = log N(u; 0, I) − Σ log L_ii。"""
        if self.intervened_value is not None:
            return 0.0 if np.allclose(x, self.intervened_value) else -np.inf
        u = self.invert(pa, x)
        return -0.5 * float(u @ u) - 0.5 * self.d * np.log(2 * np.pi) \
            - log_det_lower(self.chol(pa))

    def do(self, value: Sequence[float]) -> CholeskyMechanism:
        """do(X=value): 返回确定性机制 (割裂父依赖, 无法反演)。"""
        out = CholeskyMechanism(self.d, mean_fn=self.mean_fn, chol_fn=self.chol_fn,
                                mean_coef=self.mean_coef, chol_fixed=self.chol_fixed,
                                diag_only=self.diag_only,
                                diag_floor=self.diag_floor,
                                offdiag_scale=self.offdiag_scale)
        out.intervened_value = np.asarray(value, dtype=float)
        return out

    # ── 闭式拟合: 线性条件高斯 ───────────────────────────────────────
    @classmethod
    def fit_linear(cls, pa: np.ndarray, x: np.ndarray, *,
                   diag_only: bool = False,
                   diag_floor: float = 1e-4,
                   shrinkage: str = "sample") -> CholeskyMechanism:
        """条件高斯 MLE: μ(PA) = [1|PA]·B (最小二乘), Σ = 残差协方差, L = chol(Σ)。

        diag_only=True → 对角退化 (残差相关置零), 作对照流。
        shrinkage: "sample" (裸样本协方差) | "lw" (Ledoit–Wolf 收缩,
        小样本/大维下抬最小特征值、压 κ)。
        """
        pa = np.atleast_2d(np.asarray(pa, dtype=float))
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n, k = pa.shape
        d = x.shape[1]
        Phi = np.hstack([np.ones((n, 1)), pa])
        B, *_ = np.linalg.lstsq(Phi, x, rcond=None)
        resid = x - Phi @ B
        if shrinkage == "lw":
            Sigma, _ = ledoit_wolf_covariance(resid)
        else:
            Sigma = resid.T @ resid / max(n - (k + 1), 1)
        if diag_only:
            Sigma = np.diag(np.diag(Sigma))
        L = np.linalg.cholesky(Sigma)
        if diag_floor > 0:                        # 对角下界 (防训练/数据退化)
            L = L.copy()
            idx = np.diag_indices_from(L)
            L[idx] = np.maximum(L[idx], diag_floor)
        return cls(d, mean_coef=B, chol_fixed=L, diag_only=diag_only,
                   diag_floor=diag_floor)

    # ── 非线性拟合: MLP 输出 (μ(PA), L(PA)) ──────────────────────────
    @classmethod
    def fit_mlp(cls, pa: np.ndarray, x: np.ndarray, *,
                hidden: int = 16, epochs: int = 400, lr: float = 0.05,
                seed: int = 0, diag_floor: float = 1e-4,
                offdiag_scale: float = 0.5, diag_only: bool = False,
                kappa_reg: float = 0.0, kappa_max: float = 50.0,
                return_stats: bool = False,
                exact_kappa_every: int = 10,
                svd_monitor_every: int | None = None,
                lr_schedule: str = "none", grad_clip: float = 0.0) -> Any:
        """单隐层 MLP (tanh) 拟合条件 Cholesky 机制。

        反向传播经 Σ 参数化: g_μ = −Σ⁻¹s 与 G_Σ = ½(Σ⁻¹ − Σ⁻¹ssᵀΣ⁻¹),
        再链式回传到 L 的 unconstrained 参数与 μ 头。
        κ 正则 (kappa_reg>0, 仅非对角退化时): loss += kappa_reg·relu(log κ̂ − log κ_max),
        代理 κ̂ = 对角比 × (1 + mean|off|/mean_diag), 子梯度经对角 argmax/argmin
        与非对角符号项回传。

        return_stats=True → (mech, losses, stats), stats = {min_diag,
        mean_kappa_proxy, mean_kappa_penalty, mean_exact_kappa}。
        真 κ (SVD) 低频监控: 每 exact_kappa_every 个 epoch 对 ≤32 个样本
        算一次精确 κ₂ (训练不用, 只作日志/诊断, 研究摘要 §4D)。
        lr_schedule: "none"|"cosine"|"step" (学习率课表); grad_clip: 全局范数
        裁剪阈值 (>0 生效)。
        """
        pa = np.atleast_2d(np.asarray(pa, dtype=float))
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n, k = pa.shape
        d = x.shape[1]
        nL = d if diag_only else d * (d + 1) // 2
        rng = np.random.default_rng(seed)

        W1 = rng.standard_normal((k, hidden)) * 0.5
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((hidden, d)) * 0.5
        b2 = np.zeros(d)
        W3 = rng.standard_normal((hidden, nL)) * 0.3
        b3 = np.zeros(nL)

        def forward(pa_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            h = np.tanh(pa_batch @ W1 + b1)
            return h, h @ W2 + b2, h @ W3 + b3

        def project(raw_i: np.ndarray) -> np.ndarray:
            return project_conditioned_lower_triangular(
                raw_i, d, diag_floor=diag_floor, offdiag_scale=offdiag_scale,
                diag_only=diag_only)

        best_nll = float("inf")
        best_params: tuple | None = None
        losses: list[float] = []
        stats_log: dict[str, list[float]] = {"min_diag": [], "mean_kappa_proxy": [],
                                             "mean_kappa_penalty": [],
                                             "mean_exact_kappa": []}
        use_kappa = (kappa_reg > 0) and not diag_only
        if svd_monitor_every is not None:          # 兼容旧名
            exact_kappa_every = svd_monitor_every
        svd_idx = list(range(min(n, 32)))          # 低频 SVD 监控的样本子集
        lr_eff = float(lr)
        for epoch in range(epochs):
            if lr_schedule == "cosine":
                lr_eff = float(lr) * 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs, 1)))
            elif lr_schedule == "step":
                lr_eff = float(lr) * (0.5 ** (epoch // max(epochs // 3, 1)))
            h, mu, raw = forward(pa)
            gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)
            gW2 = np.zeros_like(W2); gb2 = np.zeros_like(b2)
            gW3 = np.zeros_like(W3); gb3 = np.zeros_like(b3)
            nll = 0.0
            ep_min_diag = float("inf")
            ep_kappa_sum = 0.0
            ep_pen_sum = 0.0
            for i in range(n):
                L = project(raw[i])
                s = x[i] - mu[i]
                u = forward_substitute(L, s)
                nll += 0.5 * float(u @ u) + log_det_lower(L)
                ep_min_diag = min(ep_min_diag, float(np.min(np.diag(L))))
                # 真实梯度 (nll 对参数的导数):
                #   d nll/dμ = −Σ⁻¹s = −L⁻ᵀu;  d nll/dΣ = ½(Σ⁻¹ − Σ⁻¹ssᵀΣ⁻¹)
                g_mu = -back_substitute(L, u)
                Sigma_inv = np.linalg.inv(L @ L.T)
                z = Sigma_inv @ s
                G_sigma = 0.5 * (Sigma_inv - np.outer(z, z))
                # ∂logp/∂L_ij = tr(G_Σ · ∂Σ/∂L_ij),  ∂Σ/∂L_ij = E_ij Lᵀ + L E_ji
                grad_raw = np.zeros(nL)
                if diag_only:
                    for ii in range(d):
                        dS = np.zeros((d, d)); dS[ii, ii] = 1.0
                        dS = dS @ L.T + L @ dS.T
                        grad_raw[ii] = float(np.sum(G_sigma * dS)) * _sigmoid(raw[i][ii])
                else:
                    idx = 0
                    for ii in range(d):
                        for jj in range(ii + 1):
                            dS = np.zeros((d, d)); dS[ii, jj] = 1.0
                            dS = dS @ L.T + L @ dS.T
                            grad_L = float(np.sum(G_sigma * dS))
                            if ii == jj:
                                grad_raw[idx] = grad_L * _sigmoid(raw[i][idx])
                            else:
                                t = np.tanh(raw[i][idx])
                                grad_raw[idx] = grad_L * offdiag_scale * (1.0 - t * t)
                            idx += 1
                # κ 正则 (廉价代理 κ̂ = 对角比 × (1+mean|off|/mean_diag), 仅非对角退化)
                if use_kappa:
                    diag = np.maximum(np.diag(L), diag_floor)
                    diag_mean = float(diag.mean())
                    ratio = float(np.max(diag) / max(np.min(diag), 1e-12))
                    off = np.abs(L - np.diag(np.diag(L)))
                    n_off = d * (d - 1)
                    off_mean = float(off.sum() / n_off) if n_off else 0.0
                    k = ratio * (1.0 + off_mean / max(diag_mean, 1e-12))
                    pen = max(0.0, float(np.log(k) - np.log(kappa_max)))
                    ep_kappa_sum += k
                    ep_pen_sum += pen
                    if pen > 0:
                        i_max = int(np.argmax(diag)); i_min = int(np.argmin(diag))
                        # 对角子梯度: argmax/argmin 经 softplus
                        for ii in (i_max, i_min):
                            sign = 1.0 if ii == i_max else -1.0
                            off = ii * (ii + 1) // 2 + ii   # 对角在 nL 布局中的位置
                            grad_raw[off] += kappa_reg * sign * _sigmoid(raw[i][off])
                        # 非对角子梯度: d log(1+O)/d L_ij ≈ sign(L_ij)/((1+O)·n_off·mean_diag)
                        if n_off:
                            g_off = kappa_reg * (1.0 / (1.0 + off_mean / max(diag_mean, 1e-12))) \
                                * (1.0 / (n_off * max(diag_mean, 1e-12)))
                            idx = 0
                            for ii in range(d):
                                for jj in range(ii):
                                    if abs(L[ii, jj]) > 1e-12:
                                        t = np.tanh(raw[i][idx])
                                        grad_raw[idx] += g_off * np.sign(L[ii, jj]) \
                                            * offdiag_scale * (1.0 - t * t)
                                    idx += 1
                    nll += kappa_reg * pen
                # 回传: μ 头与 L 头
                gW2 += np.outer(h[i], g_mu); gb2 += g_mu
                gW3 += np.outer(h[i], grad_raw); gb3 += grad_raw
                grad_h = W2 @ g_mu + W3 @ grad_raw
                grad_h *= (1.0 - h[i] ** 2)
                gW1 += np.outer(pa[i], grad_h); gb1 += grad_h

            if grad_clip > 0:                     # 全局范数裁剪
                total_norm = float(np.sqrt(sum(np.sum(g * g) for g in
                                               (gW1, gb1, gW2, gb2, gW3, gb3))))
                if total_norm > grad_clip:
                    factor = grad_clip / max(total_norm, 1e-12)
                    gW1 *= factor; gb1 *= factor
                    gW2 *= factor; gb2 *= factor
                    gW3 *= factor; gb3 *= factor
            scale = lr_eff / n
            W1 -= scale * gW1; b1 -= scale * gb1
            W2 -= scale * gW2; b2 -= scale * gb2
            W3 -= scale * gW3; b3 -= scale * gb3

            avg_nll = nll / n
            losses.append(avg_nll)
            if return_stats:
                stats_log["min_diag"].append(ep_min_diag)
                stats_log["mean_kappa_proxy"].append(ep_kappa_sum / n)
                stats_log["mean_kappa_penalty"].append(ep_pen_sum / n)
                # 低频真 κ (SVD): 每 svd_monitor_every 轮对子集样本算一次
                if epoch % max(1, exact_kappa_every) == 0 or epoch == epochs - 1:
                    kappas = []
                    for si in svd_idx:
                        Ls = project(raw[si])
                        s_vals = np.linalg.svd(Ls, compute_uv=False)
                        kappas.append(float(s_vals[0] / max(s_vals[-1], 1e-12)))
                    stats_log["mean_exact_kappa"].append(float(np.mean(kappas)))
            if avg_nll < best_nll - 1e-9:
                best_nll = avg_nll
                best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                               W3.copy(), b3.copy())

        (W1, b1, W2, b2, W3, b3) = best_params

        def mean_fn(pa_v: np.ndarray) -> np.ndarray:
            h = np.tanh(np.asarray(pa_v, dtype=float) @ W1 + b1)
            return h @ W2 + b2

        def chol_fn(pa_v: np.ndarray) -> np.ndarray:
            h = np.tanh(np.asarray(pa_v, dtype=float) @ W1 + b1)
            return project(h @ W3 + b3)

        mech = cls(d, mean_fn=mean_fn, chol_fn=chol_fn, diag_only=diag_only,
                   diag_floor=diag_floor, offdiag_scale=offdiag_scale)
        mech._train_nll = best_nll
        if return_stats:
            exact = (float(np.mean(stats_log["mean_exact_kappa"]))
                     if stats_log["mean_exact_kappa"] else 0.0)
            stats = {"kappa_reg": kappa_reg if use_kappa else 0.0,
                     "kappa_max": kappa_max,
                     "min_diag": min(stats_log["min_diag"]),
                     "mean_kappa_proxy": float(np.mean(stats_log["mean_kappa_proxy"])),
                     "mean_kappa_penalty": float(np.mean(stats_log["mean_kappa_penalty"])),
                     "mean_kappa_exact": exact,
                     "mean_exact_kappa": exact}   # 兼容旧名
            return mech, losses, stats
        return mech


__all__ = ["CholeskyMechanism", "back_substitute", "forward_substitute",
           "ledoit_wolf_covariance", "log_det_lower",
           "project_conditioned_lower_triangular", "project_lower_triangular"]
