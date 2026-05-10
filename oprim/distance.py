"""Distance and similarity atomic operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def wasserstein_distance(
    u: np.ndarray,
    v: np.ndarray,
    mode: Literal["1d", "sliced_multi_d"] = "1d",
    n_projections: int = 100,
    random_state: int | None = None,
) -> float:
    """Wasserstein distance (1D exact or sliced multi-D approximation).

    Parameters
    ----------
    u, v : np.ndarray
        Input distributions. For 1D: shape (n,). For multi-D: shape (n, d).
    mode : {"1d", "sliced_multi_d"}
        Computation mode.
    n_projections : int
        Number of random projections for sliced mode.
    random_state : int | None
        Random seed for sliced mode.

    Returns
    -------
    float
        Wasserstein distance.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if mode == "1d":
        return float(stats.wasserstein_distance(u.ravel(), v.ravel()))
    else:
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        rng = np.random.default_rng(random_state)
        d = u.shape[1]
        # Random unit vectors on sphere
        directions = rng.standard_normal((n_projections, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        distances = np.zeros(n_projections)
        for i, direction in enumerate(directions):
            u_proj = u @ direction
            v_proj = v @ direction
            distances[i] = stats.wasserstein_distance(u_proj, v_proj)

        return float(distances.mean())


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: int | None = None,
    distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    multivariate_mode: Literal["independent", "dependent"] | None = None,
) -> dict[str, Any]:
    """Dynamic Time Warping distance with Sakoe-Chiba band constraint.

    Parameters
    ----------
    x, y : np.ndarray
        Input sequences. Shape (n,) or (n, d) for multivariate.
    window : int | None
        Sakoe-Chiba band width. None = no constraint.
    distance_metric : {"euclidean", "manhattan"}
        Point-wise distance metric.
    multivariate_mode : {"independent", "dependent"} | None
        For multivariate: independent computes per-dimension then sums,
        dependent uses full vector distance.

    Returns
    -------
    dict with distance and path.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        dist, path = _dtw_1d(x, y, window, distance_metric)
    elif multivariate_mode == "independent":
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        total_dist = 0.0
        for dim in range(x.shape[1]):
            d, _ = _dtw_1d(x[:, dim], y[:, dim], window, distance_metric)
            total_dist += d
        dist = total_dist
        path = None
    else:  # dependent
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        dist, path = _dtw_multi(x, y, window, distance_metric)

    return {"distance": float(dist), "path": path}


def _dtw_1d(x, y, window, metric):
    n, m = len(x), len(y)
    w = window if window is not None else max(n, m)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            if metric == "euclidean":
                d = abs(x[i - 1] - y[j - 1])
            else:
                d = abs(x[i - 1] - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Traceback
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(cost[i - 1, j - 1], i - 1, j - 1),
                      (cost[i - 1, j], i - 1, j),
                      (cost[i, j - 1], i, j - 1)]
        _, i, j = min(candidates, key=lambda c: c[0])
    path.reverse()
    return cost[n, m], path


def _dtw_multi(x, y, window, metric):
    n, m = len(x), len(y)
    w = window if window is not None else max(n, m)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            if metric == "euclidean":
                d = np.sqrt(np.sum((x[i - 1] - y[j - 1]) ** 2))
            else:
                d = np.sum(np.abs(x[i - 1] - y[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[n, m], None


def cosine_similarity_batch(
    query: np.ndarray,
    database: np.ndarray,
    pre_normalize: bool = False,
    top_k: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Batch cosine similarity between query and database vectors.

    Parameters
    ----------
    query : np.ndarray
        Query vector(s). Shape (d,) or (n_q, d).
    database : np.ndarray
        Database vectors. Shape (n_db, d).
    pre_normalize : bool
        If True, assume inputs are already L2-normalized.
    top_k : int | None
        If set, return only top-k most similar (scores, indices).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Similarity scores, or (scores, indices) if top_k is set.
    """
    query = np.asarray(query, dtype=np.float64)
    database = np.asarray(database, dtype=np.float64)

    if query.ndim == 1:
        query = query.reshape(1, -1)

    if not pre_normalize:
        q_norm = np.linalg.norm(query, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        query = query / q_norm

        db_norm = np.linalg.norm(database, axis=1, keepdims=True)
        db_norm[db_norm == 0] = 1.0
        database = database / db_norm

    similarities = query @ database.T

    if top_k is not None:
        top_k = min(top_k, similarities.shape[1])
        indices = np.argsort(-similarities, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores.squeeze(), indices.squeeze()

    return similarities.squeeze()


def euclidean_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d).
    Y : np.ndarray | None
        Shape (m, d). If None, compute X vs X.
    weights : np.ndarray | None
        Feature weights. Shape (d,).

    Returns
    -------
    np.ndarray
        Distance matrix shape (n, m).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        w_sqrt = np.sqrt(weights)
        X = X * w_sqrt
        Y = Y * w_sqrt

    return cdist(X, Y, metric="euclidean")


def symmetric_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    mode: Literal["js", "symmetric_kl"] = "js",
    base: Literal["e", "2"] = "e",
    epsilon: float = 1e-12,
) -> float:
    """Jensen-Shannon divergence or symmetric KL divergence.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (must sum to ~1).
    mode : {"js", "symmetric_kl"}
        JS divergence or symmetric KL.
    base : {"e", "2"}
        Logarithm base.
    epsilon : float
        Smoothing for zero probabilities.

    Returns
    -------
    float
        Divergence value.
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    log_fn = np.log if base == "e" else np.log2

    if mode == "js":
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * log_fn(p / m)) + 0.5 * np.sum(q * log_fn(q / m))
        return float(js)
    else:  # symmetric_kl
        kl_pq = np.sum(p * log_fn(p / q))
        kl_qp = np.sum(q * log_fn(q / p))
        return float(0.5 * (kl_pq + kl_qp))
