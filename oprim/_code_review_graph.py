"""oprim._code_review_graph — 代码审查知识图谱原语 (code-review-graph 3O 复刻)。

把 CRG CLI (持久增量知识图谱: 调用图/影响面/死代码/社区结构) 封装为结构化原语:
  status / query(callers/callees/imports/tests/inheritors/...) / impact /
  dead-code / register / build

纯 subprocess 桥 (零外部依赖), 输出 JSON 归一; 图未构建 → 结构化提示。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

QUERY_TYPES = (
    "callers_of", "callees_of", "imports_of", "importers_of",
    "children_of", "tests_for", "inheritors_of", "file_summary",
)

_CRG_BIN = "code-review-graph"


def crg_available() -> bool:
    return shutil.which(_CRG_BIN) is not None


def _run(args: list[str], *, cwd: str = "", timeout_s: float = 120.0) -> dict[str, Any]:
    """CRG CLI 调用 → 结构化 dict (JSON 解析, 非 JSON 兜底文本)。"""
    if not crg_available():
        return {"ok": False,
                "error": "code-review-graph 未安装 (安装: pip install code-review-graph)"}
    cmd = [_CRG_BIN, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           cwd=cwd or None)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"code-review-graph {args[0]} 超时 ({timeout_s:.0f}s)"}
    out = r.stdout.strip()
    try:
        data = json.loads(out) if out else {}
    except json.JSONDecodeError:
        data = {"raw": out[:4000]}
    data.setdefault("ok", r.returncode == 0)
    if r.returncode != 0:
        data["error"] = r.stderr.strip()[-500:] or f"exit={r.returncode}"
    return data


def graph_status(cwd: str = "") -> dict[str, Any]:
    """图谱统计 (nodes/edges/files)。"""
    return _run(["status"], cwd=cwd)


def graph_query(query_type: str, target: str, *, cwd: str = "") -> dict[str, Any]:
    """图谱查询: callers_of / callees_of / imports_of / tests_for / ..."""
    if query_type not in QUERY_TYPES:
        return {"ok": False, "error": f"未知查询类型: {query_type}; 可选 {QUERY_TYPES}"}
    return _run(["query", query_type, target], cwd=cwd)


def graph_impact(target: str, *, cwd: str = "") -> dict[str, Any]:
    """变更影响面: 文件/节点改动波及范围。"""
    return _run(["impact", target], cwd=cwd)


def graph_dead_code(*, cwd: str = "") -> dict[str, Any]:
    """死代码检测。"""
    return _run(["dead-code"], cwd=cwd)


def graph_communities(*, cwd: str = "") -> dict[str, Any]:
    """社区结构 (架构模块划分)。"""
    return _run(["communities"], cwd=cwd)


def graph_register(repo_path: str, alias: str = "") -> dict[str, Any]:
    """注册仓库到图谱 (多仓库 registry)。"""
    args = ["register", repo_path]
    if alias:
        args += ["--alias", alias]
    return _run(args)


def graph_build(*, cwd: str = "", incremental: bool = True) -> dict[str, Any]:
    """构建/增量更新图谱。"""
    return _run(["update" if incremental else "build"], cwd=cwd, timeout_s=600.0)


def graph_ensure(repo_path: str = "", *, cwd: str = "") -> dict[str, Any]:
    """懒构建: 图空则 register + build; 返回就绪状态。"""
    st = graph_status(cwd=cwd)
    if st.get("ok") and int(st.get("nodes", 0) or 0) > 0:
        return {"ok": True, "ready": True, **st}
    if repo_path:
        reg = graph_register(repo_path)
        if not reg.get("ok"):
            return {"ok": False, "error": f"仓库注册失败: {reg.get('error', '')}"}
    # 图空 → 全量 build (增量 update 在无基线时只建 schema 不解析文件)
    built = graph_build(cwd=cwd, incremental=False)
    return {"ok": built.get("ok", False), "ready": bool(built.get("ok")),
            "note": "图谱已构建" if built.get("ok") else f"构建失败: {built.get('error', '')}"}


def graph_search(
    query: str,
    *,
    kind: str | None = None,
    limit: int | None = None,
    cwd: str = "",
) -> dict[str, Any]:
    """图谱实体搜索 (FTS/语义混合): query → 匹配节点列表。

    kind: File/Class/Function/Type/Test; limit: 返回上限。
    输出归一: {status, search_mode, summary, results: [{name, qualified_name,
    kind, file_path, line_start, line_end, language}]}
    """
    if not query.strip():
        return {"ok": False, "error": "query 不能为空"}
    args = ["search", query]
    if kind:
        args += ["--kind", kind]
    if limit is not None:
        args += ["--limit", str(int(limit))]
    return _run(args, cwd=cwd)


def graph_embed(*, cwd: str = "", provider: str = "local") -> dict[str, Any]:
    """计算向量嵌入 (语义搜索增强): provider=local/openai/google/minimax。

    未安装 embeddings 依赖 → 结构化错误 (语义搜索回退 FTS)。
    """
    return _run(["embed", "--provider", provider], cwd=cwd, timeout_s=300.0)


def graph_semantic_search(
    query: str,
    *,
    cwd: str = "",
    kind: str | None = None,
    limit: int | None = None,
    ensure_embed: bool = False,
) -> dict[str, Any]:
    """语义搜索原语: 确保图就绪 → search (FTS 打底, 嵌入可用时语义增强)。

    ensure_embed=True 时先尝试 embed (向量索引), 失败不阻断 (回退 FTS)。
    """
    st = graph_ensure(cwd=cwd)
    if not st.get("ok"):
        return {"ok": False, "error": f"图谱未就绪: {st.get('error', '')}"}
    if ensure_embed:
        graph_embed(cwd=cwd)  # 嵌入不可用 → 继续 FTS (不阻断)
    return graph_search(query, kind=kind, limit=limit, cwd=cwd)


__all__ = ["QUERY_TYPES", "crg_available", "graph_status", "graph_query",
           "graph_impact", "graph_dead_code", "graph_communities",
           "graph_register", "graph_build", "graph_ensure",
           "graph_search", "graph_embed", "graph_semantic_search"]
