"""oprim._grid_search — 网格搜索并行映射机制 (ProcessPool Map).

物理层纯机制, 业务全注入:
- expand_param_grid:  笛卡尔展开参数空间
- run_grid_search:    ProcessPoolExecutor 并发映射 → 进度播报 → 异常隔离
- reduce_best:        物理层规约 (按指标取最优, 跳过失败)
- build_heatmap_payload: ECharts 热力图数据 (参数 × 指标 三维可视化)

设计动机 (CPU-Bound 与大模型 I/O 的调度冲突):
- Python GIL 决定 Pandas/Numpy 海量计算必须多进程物理核心, asyncio 无效;
- 进程池 (ProcessPoolExecutor) 同时约束并发上限 (max_workers), 防止
  网格组合过多时一次性拉起 N 个进程导致内存爆炸;
- 进度回调由编排线程触发 (线程安全由调用方负责, 典型做法:
  asyncio.run_coroutine_threadsafe 跳回事件循环再推送 SSE).

不反向依赖: 本模块只 import stdlib, 策略执行/数据加载/指标计算
全部通过 worker_fn 注入 (3O 红线 5).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# 参数空间展开
# ---------------------------------------------------------------------------

def expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """把 {'window': [10, 20], 'risk': [0.01]} 展开为全组合列表.

    >>> expand_param_grid({"a": [1, 2], "b": ["x", "y"]})
    [{'a': 1, 'b': 'x'}, {'a': 1, 'b': 'y'}, {'a': 2, 'b': 'x'}, {'a': 2, 'b': 'y'}]

    空网格 → [{}] (单次执行语义).
    """
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


# ---------------------------------------------------------------------------
# ProcessPool 并发映射
# ---------------------------------------------------------------------------

def run_grid_search(
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    combos: list[dict[str, Any]],
    *,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """将全部参数组合分发到进程池并发执行 (Map), 异步收集结果 (Reduce 前身).

    Args:
        worker_fn:         物理层工作函数, 契约 ``worker_fn(params: dict) -> dict``.
                           返回值建议含 ``sharpe``/``total_return`` 等指标键;
                           抛异常的组合会被机制捕获并记为 ``{"params": ..., "error": ...}``,
                           不拖垮整个网格.
        combos:            ``expand_param_grid`` 的输出.
        max_workers:       进程池大小, 缺省 = ``os.cpu_count()``.
        progress_callback: 每完成一个组合回调 ``(done, total, result)``,
                           在编排线程 (提交者线程) 触发; 线程安全由调用方保证.

    Returns:
        与 combos 一一对应 (顺序不保证) 的结果列表, 每条含 ``params`` 键.
    """
    import concurrent.futures

    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)
    total = len(combos)
    results: list[dict[str, Any]] = []

    if total == 0:
        return results

    # 单组合特例: 免去进程池调度开销 (快路径)
    if total == 1:
        res = _safe_call(worker_fn, combos[0])
        results.append(res)
        if progress_callback is not None:
            progress_callback(1, total, res)
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, combo): combo for combo in combos}
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            combo = futures[future]
            try:
                res = future.result()
                if not isinstance(res, dict):
                    res = {"params": combo, "result": res}
                elif "params" not in res:
                    res["params"] = combo
            except Exception as exc:  # 异常隔离: 单个组合失败不影响整体
                res = {"params": combo, "error": str(exc)}
            results.append(res)
            if progress_callback is not None:
                progress_callback(done, total, res)

    return results


def _safe_call(worker_fn: Callable, combo: dict[str, Any]) -> dict[str, Any]:
    try:
        res = worker_fn(combo)
        if not isinstance(res, dict):
            res = {"params": combo, "result": res}
        elif "params" not in res:
            res["params"] = combo
        return res
    except Exception as exc:
        return {"params": combo, "error": str(exc)}


# ---------------------------------------------------------------------------
# 物理层规约 (Reduce)
# ---------------------------------------------------------------------------

def reduce_best(results: list[dict[str, Any]], key: str = "sharpe") -> dict[str, Any] | None:
    """从结果集中选指标最优者 (跳过 error/缺指标条目).

    >>> reduce_best([{"params": {"w": 10}, "sharpe": 1.2}, {"params": {"w": 20}, "sharpe": 2.1}])
    {'params': {'w': 20}, 'sharpe': 2.1}
    """
    valid = [r for r in results if isinstance(r, dict) and key in r and r.get("error") is None]
    if not valid:
        return None
    return max(valid, key=lambda r: r[key])


# ---------------------------------------------------------------------------
# 可视化载荷 (ECharts 热力图)
# ---------------------------------------------------------------------------

def build_heatmap_payload(
    results: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str = "sharpe",
) -> dict[str, Any]:
    """把网格结果规约为 ECharts heatmap 载荷.

    Returns:
        {"xAxis": [...], "yAxis": [...], "data": [[x_idx, y_idx, value], ...]}

    只含有效 (有 value_key 且无 error) 的条目; 同一 (x, y) 重复时后者覆盖.
    """
    x_vals: list[Any] = []
    y_vals: list[Any] = []
    cells: dict[tuple[int, int], Any] = {}

    for r in results:
        if not isinstance(r, dict) or "error" in r or value_key not in r:
            continue
        x = r.get("params", {}).get(x_key)
        y = r.get("params", {}).get(y_key)
        if x is None or y is None:
            continue
        if x not in x_vals:
            x_vals.append(x)
        if y not in y_vals:
            y_vals.append(y)
        cells[(x_vals.index(x), y_vals.index(y))] = r[value_key]

    return {
        "xAxis": x_vals,
        "yAxis": y_vals,
        "data": [[xi, yi, v] for (xi, yi), v in cells.items()],
    }


__all__ = ["expand_param_grid", "run_grid_search", "reduce_best", "build_heatmap_payload"]
