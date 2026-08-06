"""oprim._reward — 稠密奖励。

为什么不能只用 exit code: 它是**稀疏二值**信号, 在二值奖励下 MCTS 退化成随机搜索;
更分不出"过了但很烂"和"过了且干净"(两个候选 exit code 一样, 一个只改 3 行,
一个重写整个文件 —— 稠密奖励一眼分开)。

设计约束:
1. 每个探针 score 必须 ∈ [0,1], 否则 UCT 里 c=√2 的推导前提不成立(断言强制)。
2. gate 探针(语法/不变量/冻结文件)不参与加权, 只做否决: 任一不过奖励直接归零。
   混进加权平均会出现"语法错但测试跑了一半也拿 0.3 分"的荒谬排名。
3. 探针要便宜的先跑。gate 通常最便宜, 先跑完 gate 再跑测试, 省掉无效执行。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from ._sandbox import ExecResult, LocalSandbox


@dataclass
class ProbeResult:
    name: str
    score: float                  # 必须 ∈ [0,1]
    weight: float
    gate: bool
    detail: str = ""
    duration_ms: int = 0
    exit_code: Optional[int] = None

    def __post_init__(self):
        if not (0.0 - 1e-9 <= self.score <= 1.0 + 1e-9):
            raise ValueError(f"探针 {self.name!r} 返回 score={self.score}, 必须归一到 [0,1]")
        self.score = min(1.0, max(0.0, self.score))


class Probe(Protocol):
    name: str
    weight: float
    gate: bool
    def run(self, sandbox: LocalSandbox) -> ProbeResult: ...


Scorer = Callable[[ExecResult], "tuple[float, str]"]


def exit_zero(r: ExecResult) -> "tuple[float, str]":
    if r.timed_out:
        return 0.0, "超时"
    return (1.0, "ok") if r.exit_code == 0 else (0.0, f"exit={r.exit_code}")


def unittest_ratio(r: ExecResult) -> "tuple[float, str]":
    """从 stdlib unittest 输出解析通过率 —— 把二值信号变稠密的核心一步。"""
    text = (r.stderr or "") + (r.stdout or "")
    m = re.search(r"Ran (\d+) test", text)
    if not m:
        return (0.0, "无法解析 unittest 输出(可能是导入期就崩了)")
    total = int(m.group(1))
    if total == 0:
        return 0.0, "没有测试被收集到"
    fails = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", text))
    passed = max(0, total - fails)
    return passed / total, f"{passed}/{total} 通过"


def pytest_ratio(r: ExecResult) -> "tuple[float, str]":
    text = (r.stdout or "") + (r.stderr or "")
    p = sum(int(x) for x in re.findall(r"(\d+) passed", text))
    f = sum(int(x) for x in re.findall(r"(\d+) (?:failed|error)", text))
    tot = p + f
    return (p / tot, f"{p}/{tot} 通过") if tot else (0.0, "未采集到 pytest 结果")


def count_decay(tau: float) -> Callable[[int], float]:
    """把"越少越好"的计数映射到 [0,1]: 0 个 → 1.0, tau 个 → ~0.37。"""
    return lambda n: math.exp(-max(0, n) / tau)


@dataclass
class CommandProbe:
    name: str
    argv: Sequence[str]
    weight: float = 1.0
    gate: bool = False
    scorer: Scorer = exit_zero
    timeout_s: float = 30.0

    def run(self, sandbox: LocalSandbox) -> ProbeResult:
        r = sandbox.run(self.argv, timeout_s=self.timeout_s)
        score, detail = self.scorer(r)
        return ProbeResult(self.name, score, self.weight, self.gate,
                           detail, r.duration_ms, r.exit_code)


def py_syntax_gate(files: Sequence[str], name: str = "syntax") -> CommandProbe:
    """语法门。最便宜的探针, 永远第一个跑。"""
    return CommandProbe(name, ["python3", "-m", "py_compile", *files],
                        weight=0.0, gate=True, timeout_s=15)


def unittest_probe(target: str = "discover", weight: float = 3.0,
                   timeout_s: float = 60.0) -> CommandProbe:
    argv = ["python3", "-m", "unittest"] + (
        ["discover", "-s", ".", "-p", "test_*.py"] if target == "discover" else [target])
    return CommandProbe("tests", argv, weight=weight, gate=False,
                        scorer=unittest_ratio, timeout_s=timeout_s)


@dataclass
class DiffSizeProbe:
    """改动越小越好 —— 把"过了但很烂"和"过了且干净"分开的关键探针。"""
    baseline_files: dict                        # relpath -> 内容
    weight: float = 1.0
    gate: bool = False
    tau: float = 40.0
    name: str = "diff_size"

    def run(self, sandbox: LocalSandbox) -> ProbeResult:
        import difflib
        import os

        changed = 0
        for rel, base in self.baseline_files.items():
            p = os.path.join(sandbox.workspace, rel)
            cur = ""
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    cur = f.read()
            for line in difflib.unified_diff(base.splitlines(), cur.splitlines(), n=0):
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                    changed += 1
        return ProbeResult(self.name, count_decay(self.tau)(changed), self.weight,
                           self.gate, f"{changed} 行变更", 0)


@dataclass
class InvariantProbe:
    """领域不变量。默认 gate=True —— 业务不变量被破坏时, 测试全绿也不该放行。"""
    name: str
    script: str                                  # workspace 内的可执行脚本路径
    weight: float = 0.0
    gate: bool = True
    timeout_s: float = 20.0

    def run(self, sandbox: LocalSandbox) -> ProbeResult:
        r = sandbox.run(["python3", self.script], timeout_s=self.timeout_s)
        score, detail = exit_zero(r)
        return ProbeResult(self.name, score, self.weight, self.gate,
                           detail + (f" | {r.stdout.strip()[:120]}" if r.stdout else ""),
                           r.duration_ms, r.exit_code)


@dataclass
class FileFrozenProbe:
    """冻结文件门。防奖励劫持(reward hacking)。

    候选想让测试全绿, 最省事的办法是把失败的测试删掉 —— 这在 exit code 和
    通过率上都完美, 只有这个门能拦住。
    **从宿主侧读文件拿基线哈希比对, 不是在沙箱里跑校验脚本** ——
    校验逻辑一旦放进沙箱的可写面, 被改的就会是校验逻辑本身。
    """
    baseline_hashes: dict                        # relpath -> sha256
    name: str = "frozen"
    weight: float = 0.0
    gate: bool = True

    def run(self, sandbox: LocalSandbox) -> ProbeResult:
        import hashlib
        import os

        bad = []
        for rel, want in self.baseline_hashes.items():
            p = os.path.join(sandbox.workspace, rel)
            if not os.path.exists(p):
                bad.append(f"{rel}(已删除)")
                continue
            with open(p, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != want:
                bad.append(f"{rel}(被修改)")
        return ProbeResult(self.name, 0.0 if bad else 1.0, self.weight, self.gate,
                           "受保护文件被改动: " + ", ".join(bad) if bad else "受保护文件完好")


@dataclass
class Reward:
    value: float                                 # ∈ [0,1]
    gated: bool = False
    gate_failed: List[str] = field(default_factory=list)
    probes: List[ProbeResult] = field(default_factory=list)

    @property
    def breakdown(self) -> str:
        return "  ".join(
            f"{p.name}={p.score:.2f}" + ("*" if p.gate else "") for p in self.probes)


def run_probes(probes: Sequence[Probe], sandbox: LocalSandbox,
               short_circuit: bool = True) -> Reward:
    """gate 先跑、失败即短路。省掉的是最贵的那部分执行时间。"""
    ordered = sorted(probes, key=lambda p: (not p.gate,))
    results: List[ProbeResult] = []
    failed_gates: List[str] = []

    for p in ordered:
        res = p.run(sandbox)
        results.append(res)
        if res.gate and res.score < 1.0:
            failed_gates.append(res.name)
            if short_circuit:
                break

    if failed_gates:
        return Reward(0.0, True, failed_gates, results)

    scored = [r for r in results if not r.gate and r.weight > 0]
    if not scored:
        return Reward(1.0, False, [], results)
    total_w = sum(r.weight for r in scored)
    value = sum(r.score * r.weight for r in scored) / total_w
    return Reward(round(value, 6), False, [], results)


__all__ = ["CommandProbe", "DiffSizeProbe", "FileFrozenProbe", "InvariantProbe",
           "Probe", "ProbeResult", "Reward", "count_decay", "exit_zero",
           "py_syntax_gate", "pytest_ratio", "run_probes", "unittest_probe",
           "unittest_ratio"]
