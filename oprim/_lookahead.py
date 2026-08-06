"""oprim._lookahead — 单步 lookahead: O3 的起点, 90% 场景够用的那一档。

流程:
    base 快照 → checkout → apply(候选) → 探针 → reward (并行, 每个候选独立)
    可逆性闸门 + 阈值 + 稳定性复检 → Verdict

为什么先做深度 1: 预算。每次节点扩展 = 一次 LLM 调用(秒级 + 成本),
每次 rollout = 一次沙箱启停。深度 1、k=3~8 个候选是能在一次交互延迟内跑完的规模。

"胜率 100% 就输出"是不够的: 沙箱 ≠ 生产。Verdict 强制携带 divergences
(沙箱与生产的已知差异), 由调用方**声明**而不是系统推断 ——
让决策记录里白纸黑字写着"这次沙箱验证覆盖不到什么"。
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ._actions import ActionPlan, Applier, Escalation, gate
from ._reward import Probe, Reward, run_probes
from ._sandbox import SandboxPool
from ._snapshot import SnapshotStore


@dataclass
class Divergence:
    """沙箱与生产的已知差异。由调用方声明。"""
    kind: str                    # data_scale | external_dep | concurrency | secrets | clock | hardware
    detail: str
    severity: str = "medium"     # low | medium | high


@dataclass
class Rollout:
    plan_id: str
    reward: float
    gated: bool
    gate_failed: List[str] = field(default_factory=list)
    stable: bool = True
    result_digest: str = ""
    elapsed_ms: int = 0
    error: str = ""
    detail: Optional[Reward] = None


@dataclass
class Verdict:
    decision_id: str
    chosen: Optional[Rollout]
    ranked: List[Rollout]
    escalations: List[Escalation]
    divergences: List[Divergence]
    base_digest: str
    budget: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """有可执行结论。ok=True 不代表没有升级项 ——
        某个候选不可逆而升级, 不妨碍另一个候选被选中执行。"""
        return self.chosen is not None

    def summary(self) -> str:
        if self.chosen is None:
            return f"无可用方案({len(self.escalations)} 项已升级)"
        c = self.chosen
        return f"选中 {c.plan_id}, reward={c.reward:.3f}, stable={c.stable}"


def _rollout_one(plan: ActionPlan, base_digest: str, store: SnapshotStore,
                 pool: SandboxPool, probes: Sequence[Probe], applier: Applier,
                 stability_check: bool) -> Rollout:
    t0 = time.time()
    sb = pool.acquire()
    try:
        store.checkout(base_digest, sb.workspace)
        ok, err = applier.apply(plan, sb.workspace)
        if not ok:
            return Rollout(plan.id, 0.0, True, ["apply"], error=err,
                           elapsed_ms=int((time.time() - t0) * 1000))

        rw = run_probes(probes, sb)
        stable = True
        if stability_check and not rw.gated:
            rw2 = run_probes(probes, sb)
            stable = abs(rw2.value - rw.value) < 1e-6
            if not stable:                      # 取更保守的一次
                rw = rw if rw.value < rw2.value else rw2

        digest = store.commit(sb.workspace)
        return Rollout(plan.id, rw.value, rw.gated, rw.gate_failed, stable, digest,
                       int((time.time() - t0) * 1000), detail=rw)
    except Exception as e:                       # rollout 崩溃 = 该候选失败, 不是系统失败
        return Rollout(plan.id, 0.0, True, ["exception"], error=f"{type(e).__name__}: {e}",
                       elapsed_ms=int((time.time() - t0) * 1000))
    finally:
        pool.release(sb)


def lookahead(plans: Sequence[ActionPlan],
              base_dir: str,
              store: SnapshotStore,
              pool: SandboxPool,
              probes: Sequence[Probe],
              *,
              applier: Optional[Applier] = None,
              min_reward: float = 0.999,
              stability_check: bool = False,
              max_parallel: int = 4,
              divergences: Sequence[Divergence] = (),
              seed: int = 0) -> Verdict:
    """跑一轮单步 lookahead。

    min_reward: 低于此值一律升级,**不输出"least bad"**。搜索的产物是置信度,
        沙箱没给出足够置信度时, 正确动作是交给人, 不是把最不烂的推上去。
    stability_check: 每个候选跑两遍探针, 不一致则标 unstable 并取保守值。
        flaky 测试、时序依赖、隐藏随机性都在这里现形。代价是翻倍执行时间。
    """
    applier = applier or Applier()
    t0 = time.time()

    g = gate(list(plans))                        # ① 可逆性闸门: 不可逆的根本不进沙箱
    base_digest = store.commit(base_dir)

    rollouts: List[Rollout] = []
    if g.searchable:
        workers = max(1, min(max_parallel, len(g.searchable), pool.size))
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_rollout_one, p, base_digest, store, pool,
                              probes, applier, stability_check): p for p in g.searchable}
            for f in cf.as_completed(futs):
                rollouts.append(f.result())

    # ② 确定性排序: 奖励降序 → 耗时升序 → id 字典序。完成顺序不影响结果。
    ranked = sorted(rollouts, key=lambda r: (-r.reward, r.elapsed_ms, r.plan_id))

    escalations = list(g.escalations)
    chosen: Optional[Rollout] = None
    if ranked:
        best = ranked[0]
        if best.gated or best.reward < min_reward:
            escalations.append(Escalation(
                best.plan_id, "LOW_CONFIDENCE",
                f"最佳候选 reward={best.reward:.3f} 未达阈值 {min_reward}"
                + (f"(gate 未过: {best.gate_failed})" if best.gated else "")))
        elif not best.stable:
            escalations.append(Escalation(
                best.plan_id, "UNSTABLE",
                "两次 rollout 结果不一致, 沙箱行为非确定, 结论不可信"))
        else:
            chosen = best

    blob = "|".join([base_digest, str(seed),
                     ",".join(sorted(p.id + ":" + p.digest() for p in plans)),
                     ",".join(sorted(getattr(p, "name", "?") for p in probes))])
    decision_id = "dec_" + hashlib.sha256(blob.encode()).hexdigest()[:20]

    return Verdict(decision_id, chosen, ranked, escalations, list(divergences),
                   base_digest,
                   budget={"wall_ms": int((time.time() - t0) * 1000),
                           "rollouts": len(rollouts),
                           "sandbox_acquires": pool.stats.acquires,
                           "avg_acquire_ms": round(pool.stats.avg_wait_ms, 2),
                           "sandboxes_created": pool.stats.created})


def render_verdict(v: Verdict) -> str:
    lines = [f"decision_id : {v.decision_id}",
             f"base        : {v.base_digest}",
             f"结论        : {v.summary()}", "", "候选排名: "]
    for i, r in enumerate(v.ranked, 1):
        tag = "✔" if v.chosen and r.plan_id == v.chosen.plan_id else " "
        extra = f"  gate未过={r.gate_failed}" if r.gated else ""
        extra += f"  error={r.error}" if r.error else ""
        lines.append(f" {tag} {i}. {r.plan_id:<14} reward={r.reward:.3f}  "
                     f"stable={str(r.stable):<5} {r.elapsed_ms:>5}ms{extra}")
        if r.detail:
            lines.append(f"        └─ {r.detail.breakdown}")
    if v.escalations:
        lines += ["", "升级给人: "]
        for e in v.escalations:
            lines.append(f"  [{e.reason_code}] {e.plan_id}: {e.message}")
    if v.divergences:
        lines += ["", "沙箱未覆盖的生产差异(调用方声明): "]
        for d in v.divergences:
            lines.append(f"  ({d.severity}) {d.kind}: {d.detail}")
    lines += ["", f"预算: {v.budget}"]
    return "\n".join(lines)


__all__ = ["Divergence", "Rollout", "Verdict", "lookahead", "render_verdict"]
