"""Oprim — atomic operations library (Layer 1 meta-primitives). Lazy-loaded."""

from __future__ import annotations
import ast
import importlib
from pathlib import Path
from typing import Any
from oprim._version import __version__

_ELEMENT_MAP: dict[str, str] = {}
_SUBMODULE_SET: set[str] = set()

def _build_element_map() -> None:
    pkg_dir = Path(__file__).parent
    pkg_name = __package__ or "oprim"
    for py in sorted(pkg_dir.rglob("*.py")):
        rel_path = py.relative_to(pkg_dir)
        if rel_path.parts == ("__init__.py",): continue
        mod_parts = list(rel_path.with_suffix("").parts)
        if mod_parts[-1] == "__init__": mod_parts.pop()
        if not mod_parts: continue
        mod_path = pkg_name + "." + ".".join(mod_parts)
        stem = mod_parts[-1]
        _SUBMODULE_SET.add(stem)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in tree.body:
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.ImportFrom) and rel_path.name == "__init__.py":
                    for alias in node.names:
                        if alias.name != "*": names.append(alias.asname or alias.name)
                for name in names:
                    if not name.startswith("_"):
                        if name not in _ELEMENT_MAP or (
                            not mod_path.split(".")[-1].startswith("_") and _ELEMENT_MAP[name].split(".")[-1].startswith("_")
                        ):
                            _ELEMENT_MAP[name] = mod_path
        except Exception: continue

_build_element_map()

# KCState 归 obase（经 oprim._cognitive 单源、惰性暴露）。登记到元素表以保持
# `from oprim import KCState`（oskill 兼容）与 __all__ 可达，但不在 import 时 eager-load obase：
# __getattr__ 命中后 getattr(_cognitive, "KCState") 触发其模块级 __getattr__ 才 import obase。
_ELEMENT_MAP["KCState"] = "oprim._cognitive"  # re-export for oskill compatibility

# Real heavy-SDK modules (tree-sitter / networkx / playwright / subprocess)
_ELEMENT_MAP.setdefault("code_graph_parse", "oprim.code_graph_parse")
_ELEMENT_MAP.setdefault("tdd_test_run", "oprim.tdd_test_run")
_ELEMENT_MAP.setdefault("graph_impact_analysis", "oprim.graph_impact_analysis")
_ELEMENT_MAP.setdefault("browser_element_interact", "oprim.browser_element_interact")

# Real heavy-SDK function exports (eager-import so getattr resolves the function, not the module)
from oprim.code_graph_parse import code_graph_parse  # noqa: F401
from oprim.tdd_test_run import tdd_test_run  # noqa: F401
from oprim.graph_impact_analysis import graph_impact_analysis  # noqa: F401
from oprim.browser_element_interact import browser_element_interact  # noqa: F401
from oprim.agent_codegen import agent_codegen, agent_handoff_switch  # noqa: F401

# Grid-search ProcessPool 机制 (显式导出, 与 _injection_scan 同源可发现性)
from oprim._grid_search import (  # noqa: F401
    build_heatmap_payload,
    expand_param_grid,
    reduce_best,
    run_grid_search,
)

# ── 神经符号 / 组合优化 / 沙箱推演 原子 (O1/O2/O3 机制层, 显式导出) ──────
from oprim._do_calculus_intervention import (  # noqa: F401
    _do_calculus_intervention,
    build_binary_failure_cpd_map,
    intervene_on_store,
)
from oprim._plan_ir import PlanIR, parse_ir, validate  # noqa: F401
from oprim._ir_compile import compile_expr, compile_ir, domain_constraint_meta  # noqa: F401
from oprim._mus import explain, shrink_to_mus  # noqa: F401
from oprim._ir_solve import check_feasible, install_determinism, optimize  # noqa: F401
from oprim._backtranslate import diff_all, diff_one, render  # noqa: F401
from oprim._allocate import (  # noqa: F401
    assign_one_to_one,
    assign_with_capacity,
    cost_matrix,
    welfare,
)
from oprim._payments import first_price, second_price_per_task, vcg  # noqa: F401
from oprim._truthfulness import check_strategyproof  # noqa: F401
from oprim._deadlock import LeaseManager, ResourceOrder, WaitForGraph  # noqa: F401
from oprim._games import (  # noqa: F401
    Game,
    dominant_strategies,
    nash_vs_pareto_report,
    pareto_optimal,
    prisoners_dilemma,
    pure_nash,
)
from oprim._ledger import Bid, Ledger, Problem, Task, Worker  # noqa: F401
from oprim._snapshot import (  # noqa: F401
    HardlinkBackend,
    SnapshotStore,
    atomic_write,
    tree_digest,
)
from oprim._reward import (  # noqa: F401
    DiffSizeProbe,
    FileFrozenProbe,
    py_syntax_gate,
    run_probes,
    unittest_probe,
)
from oprim._mcts import MCTS, best_path, puct  # noqa: F401
from oprim._lookahead import Divergence, lookahead, render_verdict  # noqa: F401
from oprim._actions import (  # noqa: F401
    Action,
    ActionPlan,
    Applier,
    Reversibility,
    compensation_chain,
    gate,
)
from oprim._sandbox import LocalSandbox, SandboxPool  # noqa: F401

# ── Phase 3: 最优干预选择 (期望效用, 纯数值) ──────────────────────────
from oprim._expected_utility_select import (  # noqa: F401
    InterventionCandidate,
    SelectionResult,
    expected_utility,
    from_diagnosis_report,
    select_intervention,
)

# ── 决策审计统一写出口 (AuditEmitter) ──────────────────────────────────
from oprim._audit_emit import (  # noqa: F401
    AuditEmitter,
    AuditEvent,
    CompositeSink,
    JsonlSink,
    MemorySink,
)

# ── 推理缓存 (进程级 LRU + DAG 路径 DP) ────────────────────────────────
from oprim._inference_cache import (  # noqa: F401
    InferenceCache,
    count_simple_paths_dag,
    get_intervention_cache,
    graph_fingerprint,
    path_frequency_counts,
    set_intervention_cache_capacity,
)

# ── 多目标效用优化循环 (train 搜索 + OOS 硬门禁 + 评价缓存) ───────────
from oprim._optimize_loop import (  # noqa: F401
    DEFAULT_UTILITY_WEIGHTS,
    EvalCache,
    EvalWindow,
    FoldResult,
    LifecycleEvent,
    LifecycleRecord,
    MultiObjectiveConfig,
    OptimizeLoopResult,
    PHASES,
    ParamSpec,
    RiskGateConfig,
    StrategyLifecycle,
    WalkForwardResult,
    fingerprint_eval,
    multi_objective_utility,
    optimize_loop,
    walk_forward,
)
from oprim.p2p_mailbox import P2PMailbox  # noqa: F401
from oprim.task_router import route_tasks, dispatch_decision  # noqa: F401
_ELEMENT_MAP.setdefault("p2p_mailbox", "oprim.p2p_mailbox")
_ELEMENT_MAP.setdefault("task_router", "oprim.task_router")
_ELEMENT_MAP.setdefault("agent_codegen", "oprim.agent_codegen")
_ELEMENT_MAP.setdefault("agent_handoff_switch", "oprim.agent_codegen")
# llm_summarize 惰性加载（依赖 obase，不在没有 obase 的环境 eager-load）
def llm_summarize(*args, **kwargs):
    """惰性加载 llm_summarize，调用时才 import obase 依赖。"""
    from oprim._llm_summarize import llm_summarize as _fn
    return _fn(*args, **kwargs)

def __getattr__(name: str) -> Any:
    if name == "__version__": return __version__
    if name in _ELEMENT_MAP:
        mod = importlib.import_module(_ELEMENT_MAP[name])
        return getattr(mod, name)
    if name in _SUBMODULE_SET:
        pkg_name = __package__ or "oprim"
        return importlib.import_module(f"{pkg_name}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(set(list(_ELEMENT_MAP.keys()) + list(_SUBMODULE_SET) + ["__version__"]))

__all__ = sorted(_ELEMENT_MAP.keys())

# --- Explicit re-exports (Pinning) ---
from oprim._exceptions import (
    OprimError, FileOprimError, GitOprimError, ShellOprimError,
    ParseOprimError, PathSecurityError, LLMOprimError, BudgetExceededError,
    PromptOprimError, SearchOprimError, HttpOprimError, SnapshotOprimError
)
from oprim.llm._types import (
    LLMResponse, StreamDelta, EmbedResult, ConversationSnapshot,
    ThinkingResult, SearchResult, HttpResponse
)
# llm_complete: 惰性加载（依赖 obase）
def llm_complete(*args, **kwargs):
    from oprim.llm._llm_complete import llm_complete as _fn
    return _fn(*args, **kwargs)
def llm_stream(*args, **kwargs):
    from oprim.llm._llm_stream import llm_stream as _fn
    return _fn(*args, **kwargs)
def embed_text(*args, **kwargs):
    from oprim.llm._embed_text import embed_text as _fn
    return _fn(*args, **kwargs)
from oprim.prompt import (
    build_system_prompt, truncate_messages, extract_thinking, snapshot_conversation
)
def image_generate(*args, **kwargs):
    from oprim.image_generate import image_generate as _fn
    return _fn(*args, **kwargs)
def image_understand(*args, **kwargs):
    from oprim.image_understand import image_understand as _fn
    return _fn(*args, **kwargs)
def tts_synthesize(*args, **kwargs):
    from oprim.tts_synthesize import tts_synthesize as _fn
    return _fn(*args, **kwargs)

# --- Mneme elements (M-A batch) ---
from oprim.types import (
    SolveResult, SolveStep, StepCheckResult, Plot2DData, Three3DData,
    GradeResult, PeerPercentileResult
)
from oprim.compute_peer_percentile import compute_peer_percentile, compute_percentile_batch
from oprim.recognition_update import recognition_update, recognition_update_sequence
from oprim.compute_effortful_gain import compute_effortful_gain, compute_effortful_gain_from_arrays
from oprim.compute_feedback import compute_feedback, grade_answer
from oprim.file_type_detector import file_type_detector as file_type_detector
from oprim.due_compute import due_compute
from oprim.speech_to_math import speech_to_math
from oprim.error_classify import error_classify

# File parsers + structure extractor (restored exports)
def file_parser_pdf(*args, **kwargs):
    from oprim._file_parser_pdf import file_parser_pdf as _fn
    return _fn(*args, **kwargs)
def file_parser_epub(*args, **kwargs):
    from oprim._file_parser_epub import file_parser_epub as _fn
    return _fn(*args, **kwargs)
def file_parser_html(*args, **kwargs):
    from oprim._file_parser_html import file_parser_html as _fn
    return _fn(*args, **kwargs)
# from oprim._file_parser_markdown import file_parser_markdown as file_parser_markdown
from oprim._file_parser_plaintext import file_parser_plaintext as file_parser_plaintext
from oprim._document_structure_extractor import document_structure_extractor as document_structure_extractor

def epub_toc_split(*args, **kwargs):
    from oprim._epub_toc_split import epub_toc_split as _fn
    return _fn(*args, **kwargs)
def _get_EpubBook():
    from oprim._epub_toc_split import EpubBook
    return EpubBook
from oprim._markdown_frontmatter_build import markdown_frontmatter_build
from oprim._text_clean_publish_noise import text_clean_publish_noise
from oprim._arxiv_search import arxiv_search, ArxivPaper
from oprim._http_download_file import http_download_file
from oprim._media_types import SourceResult
from oprim._gutenberg_search import gutenberg_search
from oprim._oapen_search import oapen_search
# ── AII Graph Capability (P-G1 … P-G7) ──────────────────────────────────────
# Types (shared across AII graph elements)
from oprim._aii_graph_types import (
    ConflictSignal,
    ConflictPair,
    SourceTraceResult,
    GraphRetrievalResult,
    CascadeDeleteResult,
    TwoStepIngestResult,
    ConflictDetectionInput,
)
# P-G1: conflict candidate detection (pure computation, no LLM)
from oprim._ku_conflict_detect import ku_conflict_detect
# P-G2: purpose alignment scoring (cosine + keyword, no LLM)
from oprim._purpose_alignment_score import purpose_alignment_score
# P-G3: source provenance query (single async DB call)
from oprim._source_trace import source_trace
# P-G4: direct graph link score
from oprim._direct_link_score import direct_link_score
# P-G5: shared source overlap score
from oprim._source_overlap_score import source_overlap_score
# P-G6: adamic-adar similarity score
from oprim._adamic_adar_score import adamic_adar_score
# P-G7: knowledge-type affinity score
from oprim._type_affinity_score import type_affinity_score

from oprim._quant_analysis import compute_shapley_decomposition, compute_shapley_values

from oprim.soul_config_rewrite import soul_config_rewrite  # noqa: F401
_ELEMENT_MAP.setdefault('soul_config_rewrite', 'oprim.soul_config_rewrite')

from oprim.replay_step_record import replay_step_record  # noqa: F401
_ELEMENT_MAP.setdefault('replay_step_record', 'oprim.replay_step_record')

from oprim.kanban_task_update import kanban_task_update  # noqa: F401
_ELEMENT_MAP.setdefault('kanban_task_update', 'oprim.kanban_task_update')

from oprim.tmux_pane_create import tmux_pane_create  # noqa: F401
_ELEMENT_MAP.setdefault('tmux_pane_create', 'oprim.tmux_pane_create')

# ── Phase 4: 有限视距反事实滚动规划 ──────────────────────────────────
from oprim._counterfactual_rollout import (  # noqa: F401
    RolloutAction,
    RolloutPlan,
    counterfactual_rollout,
)

# ── L3 反事实 SCM (显式外生噪声) + 条件 Cholesky 流机制 ─────────────────
from oprim._structural_counterfactual import SCMNode, StructuralSCM  # noqa: F401
from oprim._cholesky_flow import (  # noqa: F401
    CholeskyMechanism,
    back_substitute,
    forward_substitute,
    log_det_lower,
    project_lower_triangular,
)

# ── 混合离散–连续 SCM (NodeSpec / build / fit / 溯因 / 仿真 / CF) ─────────
from oprim._hybrid_scm import HybridSCM, NodeSpec, build_hybrid_scm, fit_hybrid_scm  # noqa: F401
from oprim._cholesky_flow import project_conditioned_lower_triangular  # noqa: F401

# ── 耦合流机制 (条件仿射耦合, 可叠层) ───────────────────────────────────
from oprim._coupling_flow import (  # noqa: F401
    ConditionalCouplingMechanism,
    CouplingLayer,
)

# ── 贝叶斯优化规划 (RBF-GP + EI) ───────────────────────────────────────
from oprim._bayes_opt_plan import (  # noqa: F401
    RBFGP,
    bayesian_optimize,
    continuous_plan_with_hybrid_bo,
)

# ── 深度 SCM 训练课表与温度校准 ─────────────────────────────────────────
from oprim._deep_scm_train import (  # noqa: F401
    calibrate_deep_scm_temperature,
    fit_deep_scm,
)

# G5 规范化事件管道 (Vigla 复刻)
from oprim._canonical_event_ingest import (  # noqa: E402
    canonical_event_ingest,
    compute_event_fingerprint,
    deserialize_vendor,
)

# LLM 智能路由原语 (RouteLLM 内化): 路由决策 + 并行分派
from oprim._llm_router import (  # noqa: E402
    DEFAULT_MATRIX,
    load_matrix,
    route_decision,
)
from oprim._parallel_llm import (  # noqa: E402
    aggregate_results,
    dispatch_parallel,
    split_prompt,
)

# 质量闸门 (分层路由 v2)
from oprim._quality_gate import (  # noqa: E402
    quality_check,
)

# 可执行 Spec 解析 (spec-kit 内化)
from oprim._spec_parse import (  # noqa: E402
    SECTION_ALIASES,
    parse_spec,
    validate_spec,
)

# 代码审查知识图谱 (code-review-graph 3O 复刻)
from oprim._code_review_graph import (  # noqa: E402
    QUERY_TYPES,
    crg_available,
    graph_build,
    graph_communities,
    graph_dead_code,
    graph_ensure,
    graph_impact,
    graph_query,
    graph_register,
    graph_status,
)
