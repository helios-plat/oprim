"""oprim 原子操作层：单次执行全局最优指派。签名严格遵循 <=1 位置参数。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _scipy_linear_assign(
    cost_matrix: list[list[float]],
    *,
    workers: list[str],
    tasks: list[str],
) -> dict[str, Any]:
    """单次调用 scipy.optimize 进行二分图最大权匹配/最小代价分配.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        cost_matrix: n×m 代价矩阵
        workers: worker ID 列表
        tasks: task ID 列表

    Returns:
        {
            "status": "success" | "error",
            "matches": [(w, t), ...],
            "total_cost": float,
            "unassigned_workers": [...],
            "unassigned_tasks": [...],
        }
    """
    c_mat = np.array(cost_matrix, dtype=float)
    if c_mat.shape != (len(workers), len(tasks)):
        return {"status": "error", "error": "矩阵形状与实体数量不符"}

    row_ind, col_ind = linear_sum_assignment(c_mat)

    matches = []
    total_cost = 0.0
    assigned_w, assigned_t = set(), set()

    for r, c in zip(row_ind, col_ind, strict=False):
        cost = c_mat[r, c]
        if np.isinf(cost):
            continue

        w_id, t_id = workers[r], tasks[c]
        matches.append((w_id, t_id))
        total_cost += cost
        assigned_w.add(w_id)
        assigned_t.add(t_id)

    return {
        "status": "success",
        "matches": matches,
        "total_cost": float(total_cost),
        "unassigned_workers": [w for w in workers if w not in assigned_w],
        "unassigned_tasks": [t for t in tasks if t not in assigned_t],
    }
