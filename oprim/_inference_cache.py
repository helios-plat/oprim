"""oprim._inference_cache — 进程级推理缓存 (LRU) + DAG 路径计数 DP。

热点: _do_calculus_intervention 每次重建 BN + 全路径枚举 / _counterfactual_rollout
次步重复全量推理 / causal_fault_diagnose 的 all_simple_paths 指数级枚举。

对策:
  1. InferenceCache — 进程级 LRU (默认 512), 干预结果按
     (图指纹, 干预节点, 值, 结果节点集, CPD 指纹) 寻址;
  2. graph_fingerprint — 确定性图指纹 (结构 + 节点属性 + CPD + 引擎版本),
     保证"同图同版本"才命中缓存;
  3. count_simple_paths_dag / path_frequency_counts — O(V+E) 拓扑 DP,
     替代 all_simple_paths 的指数级枚举 (只计数, 不列路径)。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any


class InferenceCache:
    """进程级 LRU 缓存: 线程不安全, 调用方保证串行 (GIL 下的 dict 操作原子)。"""

    def __init__(self, capacity: int = 512):
        self.capacity = max(1, capacity)
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Any | None:
        if key in self._data:
            self._hits += 1
            self._data.move_to_end(key)
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self._hits = self._misses = 0

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / (self._hits + self._misses), 4)
            if (self._hits + self._misses) else 0.0,
            "size": len(self._data),
            "capacity": self.capacity,
        }


# 进程级单例 (诊断/规划共享)
_INTERVENTION_CACHE = InferenceCache()


def get_intervention_cache() -> InferenceCache:
    """调试入口: from veya_loop.oprim._inference_cache import get_intervention_cache
    → get_intervention_cache().stats() 看 hits/misses。"""
    return _INTERVENTION_CACHE


def set_intervention_cache_capacity(capacity: int) -> None:
    _INTERVENTION_CACHE.capacity = max(1, capacity)


# ---------------------------------------------------------------------------
# 确定性图指纹
# ---------------------------------------------------------------------------

def graph_fingerprint(dag: Any, cpd_map: dict[str, Any] | None = None) -> str:
    """结构 + 节点属性 + CPD + 引擎版本的 sha256 指纹。

    同图同版本才命中缓存 —— 版本号/结构变了指纹就变, 缓存自动失效。
    """

    h = hashlib.sha256()
    nodes = {n: dict(dag.nodes[n]) for n in dag.nodes}
    for n in sorted(nodes):
        attrs = nodes[n]
        # 节点属性里的 cond_fail 可能是 {tuple: float}, 需规范化
        canon = {k: (sorted((str(a), v) for a, v in vv.items()) if isinstance(vv, dict) else vv)
                 for k, vv in attrs.items()}
        h.update(f"N:{n}:{json.dumps(canon, sort_keys=True, default=str)}".encode())
    for u, v in sorted(dag.edges()):
        h.update(f"E:{u}->{v}".encode())

    if cpd_map:
        for node in sorted(cpd_map):
            cpd = cpd_map[node]
            try:
                values = cpd.values.tolist() if hasattr(cpd.values, "tolist") else cpd.values
                states = getattr(cpd, "state_names", None)
                h.update(f"C:{node}:{json.dumps(values, sort_keys=True, default=str)}"
                         f":{json.dumps(states, sort_keys=True, default=str)}".encode())
            except Exception:
                h.update(f"C:{node}:<unserializable>".encode())

    # 引擎版本进指纹: 同一图在不同 pgmpy 版本下推理结果可能不同
    try:
        import pgmpy  # noqa: PLC0415
        h.update(f"pgmpy:{pgmpy.__version__}".encode())
    except Exception:
        pass
    return h.hexdigest()[:24]


# ---------------------------------------------------------------------------
# DAG 路径计数 DP (O(V+E), 替代 all_simple_paths 指数级枚举)
# ---------------------------------------------------------------------------

def count_simple_paths_dag(dag: Any, source: str, target: str,
                           cutoff: int | None = None) -> int:
    """DAG 上 source → target 的简单路径数 (拓扑 DP, O(V+E))。

    DAG 中任意路径必为简单路径; 有环时退化为保守 0 (本模块服务因果 DAG)。
    cutoff 忽略 —— 计数不含长度限制 (调用方如需限制需自行剪枝)。
    """
    import networkx as nx  # noqa: PLC0415

    if source == target:
        return 1
    if not (nx.has_path(dag, source, target)):
        return 0

    # 拓扑序: 只取 source..target 之间的子图
    sub = dag.subgraph(nx.descendants(dag, source) | {source})
    topo = list(nx.topological_sort(sub))
    dp = {n: 0 for n in topo}
    dp[source] = 1
    for n in topo:
        if n == target:
            break
        for w in sub.successors(n):
            if w in dp:
                dp[w] += dp[n]
    return dp.get(target, 0)


def path_frequency_counts(dag: Any, candidates: Sequence[str],
                          failure_node: str) -> dict[str, int]:
    """每个候选节点被多少条「根 → failure_node」路径穿过 (O(V+E) 拓扑 DP)。

    替代 causal_fault_diagnose 里对每个根节点跑 all_simple_paths 的指数级枚举:
      paths_to[v]   = 根 → v 的路径数
      paths_from[v] = v → failure_node 的路径数
      path_counts[v] = paths_to[v] × paths_from[v]
    """
    import networkx as nx  # noqa: PLC0415

    out = {n: 0 for n in candidates}
    if failure_node not in dag or not dag.nodes:
        return out

    # 需要把"根"定义为无入边节点; 若候选在 failure 下游则自然为 0
    try:
        topo = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        return out  # 有环: 不是因果 DAG, 不做计数

    idx = {n: i for i, n in enumerate(topo)}
    if failure_node not in idx:
        return out

    paths_to = {n: 0 for n in topo}
    for n in topo:
        if dag.in_degree(n) == 0:
            paths_to[n] = 1
    for n in topo:
        if paths_to[n] == 0:
            continue
        for w in dag.successors(n):
            paths_to[w] += paths_to[n]

    # 逆拓扑: 从 failure 向上累计
    paths_from = {n: 0 for n in topo}
    paths_from[failure_node] = 1
    for n in reversed(topo):
        if paths_from[n] == 0:
            continue
        for p in dag.predecessors(n):
            paths_from[p] += paths_from[n]

    for n in candidates:
        if n in paths_to and n in paths_from:
            out[n] = paths_to[n] * paths_from[n]
    return out


__all__ = ["InferenceCache", "count_simple_paths_dag", "get_intervention_cache",
           "graph_fingerprint", "path_frequency_counts",
           "set_intervention_cache_capacity"]
