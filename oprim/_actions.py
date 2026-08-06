"""oprim._actions — 动作模型 + 可逆性闸门。

O3 最重要的一条规则不在 MCTS 里, 在这个文件里:

    **只对可逆动作做树搜索。**

沙箱能告诉你"这么干在沙箱里是对的", 不能告诉你"这么干在生产上是安全的"。
对 DROP TABLE、发消息、扣款、销毁资源这类动作, 再高的沙箱胜率也不构成授权 ——
搜索给出的是置信度, 不是权限。不可逆动作走两阶段提交 + Saga 补偿 + 人审,
在闸门这里就被拦下, 根本不进 rollout。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Reversibility(str, Enum):
    REVERSIBLE = "reversible"        # 快照回滚即可撤销 → 进搜索
    COMPENSABLE = "compensable"      # 需要显式补偿动作才能撤销 → 进搜索, 必须带 compensation
    IRREVERSIBLE = "irreversible"    # 无法撤销 → 永不进搜索


ORDER = {Reversibility.REVERSIBLE: 0, Reversibility.COMPENSABLE: 1,
         Reversibility.IRREVERSIBLE: 2}


@dataclass
class Action:
    id: str
    kind: str                                   # write_file | delete_file | exec | http | sql | ...
    payload: Dict[str, Any]
    reversibility: Reversibility = Reversibility.REVERSIBLE
    compensation: Optional["Action"] = None     # COMPENSABLE 必填
    description: str = ""

    def digest_parts(self) -> str:
        import json
        return json.dumps({"id": self.id, "kind": self.kind, "payload": self.payload},
                          sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ActionPlan:
    """一个候选方案 = 一组动作。LLM 一次给 k 个, O3 挑一个。"""
    id: str
    actions: List[Action]
    prior: float = 1.0                          # LLM 先验(PUCT 用); 深度1 只用来排序
    rationale: str = ""

    @property
    def risk(self) -> Reversibility:
        if not self.actions:
            return Reversibility.REVERSIBLE
        return max((a.reversibility for a in self.actions), key=lambda r: ORDER[r])

    def digest(self) -> str:
        import hashlib
        blob = "|".join(a.digest_parts() for a in self.actions)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Escalation:
    """升级给人的请求。这不是错误, 是设计路径。"""
    plan_id: str
    reason_code: str            # IRREVERSIBLE | MISSING_COMPENSATION | LOW_CONFIDENCE | UNSTABLE
    message: str
    actions: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    searchable: List[ActionPlan] = field(default_factory=list)
    escalations: List[Escalation] = field(default_factory=list)


def gate(plans: List[ActionPlan]) -> GateResult:
    """可逆性闸门。返回"可以送去 rollout 的"和"必须升级的"。"""
    out = GateResult()
    for p in plans:
        irr = [a.id for a in p.actions if a.reversibility is Reversibility.IRREVERSIBLE]
        if irr:
            out.escalations.append(Escalation(
                p.id, "IRREVERSIBLE",
                "含不可逆动作, 不进入沙箱搜索。请走两阶段提交或人工审批。", irr))
            continue
        missing = [a.id for a in p.actions
                   if a.reversibility is Reversibility.COMPENSABLE and a.compensation is None]
        if missing:
            out.escalations.append(Escalation(
                p.id, "MISSING_COMPENSATION",
                "标记为可补偿但没有给出补偿动作。Saga 缺了回滚分支, 等同不可逆。", missing))
            continue
        out.searchable.append(p)
    return out


def compensation_chain(plan: ActionPlan) -> List[Action]:
    """Saga 回滚链: 按动作逆序返回补偿动作。执行失败时用。"""
    return [a.compensation for a in reversed(plan.actions) if a.compensation is not None]


class Applier:
    """把 ActionPlan 落到 workspace 目录。只改文件, 不执行命令 ——
    执行是 rollout 的事, 混在一起你会分不清"改坏了"和"跑挂了"。"""

    SUPPORTED = ("write_file", "delete_file", "append_file")

    def apply(self, plan: ActionPlan, workspace: str) -> Tuple[bool, str]:
        import os

        from ._snapshot import atomic_write

        for a in plan.actions:
            if a.kind not in self.SUPPORTED:
                return False, f"动作 {a.id!r} 的 kind={a.kind!r} 不被 Applier 支持"
            path = os.path.join(workspace, a.payload["path"])
            if not os.path.abspath(path).startswith(os.path.abspath(workspace) + os.sep):
                return False, f"动作 {a.id!r} 试图写到 workspace 之外: {a.payload['path']!r}"
            if a.kind == "delete_file":
                if os.path.exists(path):
                    os.remove(path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = a.payload.get("content", "")
            if a.kind == "append_file" and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = f.read() + data
            atomic_write(path, data)
        return True, ""


__all__ = ["Action", "ActionPlan", "Applier", "Escalation", "GateResult",
           "Reversibility", "compensation_chain", "gate"]
