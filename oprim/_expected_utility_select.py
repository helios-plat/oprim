"""oprim._expected_utility_select — 最优干预选择(期望效用, 纯数值)。

把每个候选 do(X=x) 映射为期望效用:

    U(a) = ΔP_success(a) − λ·C(a) − ρ·risk(a)

其中 ΔP_success 直接来自 Phase 2 的定量 do-calculus 结果, C(a) 是动作成本
(重启代价/限流损失/人工介入), risk(a) ∈ [0,1] 是动作固有风险(如不可逆性/副作用)。

纯机制, 业务注入: 候选从诊断报告转换(见 from_diagnosis_report), 或由调用方直接构造。
确定性: 排序键 (-utility, -delta_p, cost, action_id) —— 输入顺序不影响结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class InterventionCandidate:
    action_id: str
    delta_p: float            # ΔP(success) — 来自 do-calculus (可为负)
    cost: float = 0.0         # 动作成本(重启/限流/人工)
    risk: float = 0.0         # ∈ [0,1], 动作固有风险
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"action_id": self.action_id, "delta_p": round(self.delta_p, 6),
                "cost": round(self.cost, 6), "risk": round(self.risk, 6),
                "description": self.description}


def expected_utility(c: InterventionCandidate,
                     lambda_cost: float = 1.0,
                     risk_aversion: float = 1.0) -> float:
    """U(a) = ΔP − λ·C − ρ·risk。"""
    return c.delta_p - lambda_cost * c.cost - risk_aversion * c.risk


@dataclass
class SelectionResult:
    best: Optional[InterventionCandidate]
    ranked: List[tuple[InterventionCandidate, float]] = field(default_factory=list)
    rejected: List[tuple[InterventionCandidate, float]] = field(default_factory=list)
    lambda_cost: float = 1.0
    risk_aversion: float = 1.0

    @property
    def has_action(self) -> bool:
        return self.best is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best.as_dict() if self.best else None,
            "ranked": [{"action": c.as_dict(), "utility": round(u, 6)}
                       for c, u in self.ranked],
            "rejected": [{"action": c.as_dict(), "utility": round(u, 6),
                          "reason": "效用非正, 被丢弃"} for c, u in self.rejected],
            "lambda_cost": self.lambda_cost,
            "risk_aversion": self.risk_aversion,
        }


def select_intervention(candidates: Sequence[InterventionCandidate],
                        *,
                        lambda_cost: float = 1.0,
                        risk_aversion: float = 1.0,
                        drop_negative: bool = True) -> SelectionResult:
    """按期望效用选择最优干预。

    drop_negative=True 时, 效用 ≤ 0 的动作全部丢弃 —— 不输出 least-bad:
    没有任何正效用干预时 best=None, 调用方应回报"无需干预/升级人工"。
    """
    scored = [(c, expected_utility(c, lambda_cost, risk_aversion)) for c in candidates]
    # 确定性排序: 效用降序 → ΔP 降序 → 成本升序 → id 字典序
    scored.sort(key=lambda cu: (-cu[1], -cu[0].delta_p, cu[0].cost, cu[0].action_id))

    if drop_negative:
        ranked = [(c, u) for c, u in scored if u > 0]
        rejected = [(c, u) for c, u in scored if u <= 0]
    else:
        ranked, rejected = scored, []

    best = ranked[0][0] if ranked else None
    return SelectionResult(best, ranked, rejected, lambda_cost, risk_aversion)


def from_diagnosis_report(report: Dict[str, Any]) -> List[InterventionCandidate]:
    """把 Phase 2 诊断报告直接转换为候选干预。

    容错解析:
      - 候选列表键: interventions / candidates / actions (取第一个命中的)
      - 每个候选: action_id|id|name 作 id; delta_p|ΔP|delta_p_success|delta_success 作 ΔP;
        cost / risk / description 直接透传。
    找不到任何候选 → 返回空列表。
    """
    items: Any = None
    for key in ("interventions", "candidates", "actions"):
        if isinstance(report.get(key), list):
            items = report[key]
            break
    if items is None and isinstance(report.get("action"), dict):
        items = [report["action"]]
    if not items:
        return []

    out: List[InterventionCandidate] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        aid = it.get("action_id") or it.get("id") or it.get("name")
        if aid is None:
            continue
        dp = None
        for k in ("delta_p", "ΔP", "delta_p_success", "delta_success", "delta"):
            if k in it:
                dp = float(it[k])
                break
        if dp is None:
            continue
        out.append(InterventionCandidate(
            action_id=str(aid),
            delta_p=dp,
            cost=float(it.get("cost", 0.0)),
            risk=float(it.get("risk", 0.0)),
            description=str(it.get("description", "")),
        ))
    return out


__all__ = ["InterventionCandidate", "SelectionResult", "expected_utility",
           "from_diagnosis_report", "select_intervention"]
