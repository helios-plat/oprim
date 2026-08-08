"""Agentic 参数优化循环 (Agentic HPO) — 语义化提议 + 测量迭代原语。

从 optim-agent (AgentSampler / Study / space 机制) 提炼的领域无关核心:
编码 agent 读参数语义 + trial 历史 → 提议下一组配置; 测量目标值; 记录;
agent 只提议, 空间校验兜底 (bounded execution)。

规则 (与 optim-agent bounded-execution 设计对齐):
1. 提议必须过 dist.validate() (范围/类型/离散) — 非法 → 重试 1 次 →
   安全随机采样兜底 (fail_closed=True 时抛错);
2. history 驱动: prompt 组装 best/promising/recent/weak 四区摘要 + 语义
   context 先验; 避免重复已评估点;
3. _note 跨 trial 传递观察 (agent 经验积累, 下轮 prompt 回显);
4. JSON 持久化 → 跨会话恢复 (create_hp_study(storage=...));
5. sampler 注入 (Callable[[str], str]) — 纯逻辑可 stub 测试, 零 token 依赖;
   backend="mock" 时 hill-climbing 离线兜底 (测试/演示不烧 token)。

与 oprim 既有闭环的定位差异:
- _quality_gate / _silhouette_gate / _vlm_consensus: 判定"产出是否合格"
- _hp_search: 提议"下一组参数" — 参数空间上的语义化搜索 (HPO)
纯 stdlib, 同步 API (host 侧可用 asyncio.to_thread 包异步调用)。
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

HP_SCHEMA_VERSION = 1


class HPSpaceError(ValueError):
    """参数空间定义或提议校验错误。"""


# ── 参数分布 ────────────────────────────────────────────────────────────


class HPDist:
    """参数分布基类: describe / sample / validate 三件套。"""

    kind = "base"

    def __init__(self, context: str | None = None) -> None:
        self.context = context

    def describe(self) -> str:
        raise NotImplementedError

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def validate(self, value: Any) -> Any:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPDist:
        raise NotImplementedError


class FloatDist(HPDist):
    """连续参数: [low, high], log=True 时 log-uniform 采样。"""

    kind = "float"

    def __init__(
        self, low: float, high: float, log: bool = False,
        context: str | None = None,
    ) -> None:
        super().__init__(context)
        if low > high:
            raise HPSpaceError(f"FloatDist low={low} > high={high}")
        self.low = float(low)
        self.high = float(high)
        self.log = log

    def describe(self) -> str:
        base = f"float in [{self.low:g}, {self.high:g}]"
        if self.log:
            base = f"log-uniform {base}"
        if self.context:
            base += f"; {self.context}"
        return base

    def sample(self, rng: random.Random) -> float:
        if self.log:
            lo, hi = math.log(max(self.low, 1e-12)), math.log(self.high)
            return math.exp(rng.uniform(lo, hi))
        return rng.uniform(self.low, self.high)

    def validate(self, value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"not a float: {value!r}") from None
        if not (self.low <= v <= self.high):
            raise ValueError(f"{v:g} out of range [{self.low:g}, {self.high:g}]")
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind, "low": self.low, "high": self.high,
            "log": self.log, "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FloatDist:
        return cls(data["low"], data["high"], log=data.get("log", False),
                   context=data.get("context"))


class IntDist(HPDist):
    """离散整数参数: [low, high] (含端点)。"""

    kind = "int"

    def __init__(
        self, low: int, high: int, log: bool = False,
        context: str | None = None,
    ) -> None:
        super().__init__(context)
        if low > high:
            raise HPSpaceError(f"IntDist low={low} > high={high}")
        self.low = int(low)
        self.high = int(high)
        self.log = log

    def describe(self) -> str:
        base = f"int in [{self.low}, {self.high}]"
        if self.log:
            base = f"log-uniform {base}"
        if self.context:
            base += f"; {self.context}"
        return base

    def sample(self, rng: random.Random) -> int:
        if self.log:
            lo, hi = math.log(max(self.low, 1)), math.log(self.high)
            return int(round(math.exp(rng.uniform(lo, hi))))
        return rng.randint(self.low, self.high)

    def validate(self, value: Any) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"not an int: {value!r}") from None
        if not (self.low <= v <= self.high):
            raise ValueError(f"{v} out of range [{self.low}, {self.high}]")
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind, "low": self.low, "high": self.high,
            "log": self.log, "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntDist:
        return cls(data["low"], data["high"], log=data.get("log", False),
                   context=data.get("context"))


class CategoricalDist(HPDist):
    """类别参数: 有限离散选择。"""

    kind = "categorical"

    def __init__(self, choices: list[Any], context: str | None = None) -> None:
        super().__init__(context)
        if not choices:
            raise HPSpaceError("CategoricalDist needs >= 1 choice")
        self.choices = list(choices)

    def describe(self) -> str:
        base = f"categorical in {self.choices}"
        if self.context:
            base += f"; {self.context}"
        return base

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def validate(self, value: Any) -> Any:
        if value not in self.choices:
            raise ValueError(f"{value!r} not in {self.choices}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind, "choices": self.choices, "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoricalDist:
        return cls(data["choices"], context=data.get("context"))


DIST_FACTORIES: dict[str, Callable[[dict[str, Any]], HPDist]] = {
    "float": FloatDist.from_dict,
    "int": IntDist.from_dict,
    "categorical": CategoricalDist.from_dict,
}


def dist_from_dict(data: dict[str, Any]) -> HPDist:
    factory = DIST_FACTORIES.get(data.get("type", ""))
    if factory is None:
        raise HPSpaceError(f"unknown dist type: {data.get('type')!r}")
    return factory(data)


# ── 参数空间 ────────────────────────────────────────────────────────────


class HPSpace:
    """有序参数空间: name -> HPDist。"""

    def __init__(self) -> None:
        self._dists: dict[str, HPDist] = {}

    def define(
        self, name: str, dist: HPDist,
    ) -> HPSpace:
        if not name or not name.isidentifier():
            raise HPSpaceError(f"bad parameter name: {name!r}")
        self._dists[name] = dist
        return self

    def define_float(
        self, name: str, low: float, high: float, log: bool = False,
        context: str | None = None,
    ) -> HPSpace:
        return self.define(name, FloatDist(low, high, log=log, context=context))

    def define_int(
        self, name: str, low: int, high: int, log: bool = False,
        context: str | None = None,
    ) -> HPSpace:
        return self.define(name, IntDist(low, high, log=log, context=context))

    def define_categorical(
        self, name: str, choices: list[Any], context: str | None = None,
    ) -> HPSpace:
        return self.define(name, CategoricalDist(choices, context=context))

    def __contains__(self, name: str) -> bool:
        return name in self._dists

    def __len__(self) -> int:
        return len(self._dists)

    def names(self) -> list[str]:
        return list(self._dists)

    def dist(self, name: str) -> HPDist:
        return self._dists[name]

    def items(self) -> list[tuple[str, HPDist]]:
        return list(self._dists.items())

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {n: d.sample(rng) for n, d in self._dists.items()}

    def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        """校验完整提议; 缺参/非法均抛 HPSpaceError。"""
        missing = [n for n in self._dists if n not in params]
        if missing:
            raise HPSpaceError(f"missing params: {missing}")
        out: dict[str, Any] = {}
        try:
            for n, d in self._dists.items():
                out[n] = d.validate(params[n])
        except ValueError as exc:
            raise HPSpaceError(f"param {n!r}: {exc}") from None
        return out

    def to_dict(self) -> dict[str, Any]:
        return {n: d.to_dict() for n, d in self._dists.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPSpace:
        space = cls()
        for name, spec in data.items():
            space.define(name, dist_from_dict(spec))
        return space


def space_from_json(text: str) -> HPSpace:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise HPSpaceError("space JSON must be an object")
    return HPSpace.from_dict(data)


# ── Trial / Study ───────────────────────────────────────────────────────


class HPTrial:
    """一次参数评估: 提议 → 测量 → 记录。"""

    def __init__(
        self, number: int, params: dict[str, Any],
        value: float | None = None, state: str = "complete",
        note: str | None = None,
    ) -> None:
        self.number = number
        self.params = dict(params)
        self.value = value
        self.state = state
        self.note = note

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "params": self.params, "value": self.value,
            "state": self.state, "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPTrial:
        return cls(data["number"], data["params"], data.get("value"),
                   data.get("state", "complete"), data.get("note"))


class HPStudy:
    """HPO 主循环: 空间 + 历史 + 提议 + 持久化。

    optimize() 同步执行 (sampler 同步注入); host 侧可用
    ``asyncio.to_thread`` 包异步调用。
    """

    def __init__(
        self, direction: str = "maximize", space: HPSpace | None = None,
        storage: str | Path | None = None, seed: int = 0,
    ) -> None:
        if direction not in ("maximize", "minimize"):
            raise HPSpaceError(f"direction must be maximize|minimize, got {direction!r}")
        self.direction = direction
        self.space = space if space is not None else HPSpace()
        self.storage = Path(storage) if storage else None
        self.seed = seed
        self.trials: list[HPTrial] = []
        self.note: str | None = None
        self._rng = random.Random(seed)

    # -- 记录 -----------------------------------------------------------

    def ask(self, params: dict[str, Any] | None = None) -> HPTrial:
        """显式提议 (skill 模式: host agent 读代码后自己选点)。"""
        params = (
            self.space.sample(self._rng)
            if params is None
            else self.space.validate(params)
        )
        trial = HPTrial(len(self.trials), params)
        self.trials.append(trial)
        return trial

    def tell(
        self, trial: HPTrial, value: float | None = None,
        state: str = "complete",
    ) -> None:
        """记录测量值; 失败/提前终止用 state='failed'|'pruned'。"""
        trial.value = value
        trial.state = state
        if self.storage is not None:
            self.save(self.storage)

    # -- 查询 -----------------------------------------------------------

    @property
    def completed(self) -> list[HPTrial]:
        return [t for t in self.trials
                if t.state == "complete" and t.value is not None]

    @property
    def best_trial(self) -> HPTrial | None:
        done = self.completed
        if not done:
            return None
        return min(done, key=lambda t: t.value) if self.direction == "minimize" \
            else max(done, key=lambda t: t.value)

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "minimize" else a > b

    # -- 主循环 ---------------------------------------------------------

    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
        n_trials: int = 5,
        sampler: HPSampler | None = None,
        on_trial: Callable[[HPTrial], None] | None = None,
    ) -> HPStudy:
        """跑 n_trials 轮: sampler.propose(study) → objective(params) →
        tell(value)。objective 抛异常 → 记为 failed, 循环继续。"""
        if sampler is None:
            sampler = RandomSampler(seed=self.seed)
        for _ in range(n_trials):
            params = sampler.propose(self)
            if not params:
                params = self.space.sample(self._rng)
            trial = HPTrial(len(self.trials), params)
            self.trials.append(trial)
            try:
                value = objective(trial.params)
                trial.value = float(value)
                trial.state = "complete"
            except Exception as exc:  # objective 崩溃 → failed, 不中断研究
                trial.state = "failed"
                trial.note = f"objective error: {exc}"
            if on_trial is not None:
                on_trial(trial)
            if self.storage is not None:
                self.save(self.storage)
        return self

    # -- 持久化 ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": HP_SCHEMA_VERSION,
            "direction": self.direction,
            "seed": self.seed,
            "space": self.space.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPStudy:
        study = cls(
            direction=data.get("direction", "maximize"),
            space=HPSpace.from_dict(data.get("space", {})),
            seed=data.get("seed", 0),
        )
        study.trials = [HPTrial.from_dict(t) for t in data.get("trials", [])]
        study.note = data.get("note")
        return study

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.storage
        if target is None:
            raise HPSpaceError("no storage path given")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        self.storage = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> HPStudy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        study = cls.from_dict(data)
        study.storage = Path(path)
        return study


def create_hp_study(
    direction: str = "maximize", storage: str | Path | None = None,
    seed: int = 0,
) -> HPStudy:
    """创建 HPStudy (对齐 optim-agent create_study 语义)。"""
    return HPStudy(direction=direction, storage=storage, seed=seed)


# ── Samplers ────────────────────────────────────────────────────────────


class HPSampler:
    """采样器基类: propose(study) -> dict | None (None → study 随机兜底)。"""

    def propose(self, study: HPStudy) -> dict[str, Any] | None:
        raise NotImplementedError


class RandomSampler(HPSampler):
    """纯随机采样 (基线/兜底)。"""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(self, study: HPStudy) -> dict[str, Any] | None:
        return study.space.sample(self._rng)


class AgentSampler(HPSampler):
    """语义化提议器: 注入 sampler (prompt -> reply), 校验兜底。

    - prompt 组装: goal/direction + 语义 context 先验 + 空间描述 + 四区
      history (best/promising/recent/weak) + 禁止重复已评估点;
    - 回复解析失败 → 重试 1 次 (提示只回 JSON) → 随机采样兜底
      (fail_closed=True 抛错);
    - _note 跨 trial 传递 (agent 对地形的观察, 下轮回显)。
    """

    def __init__(
        self,
        sampler: Callable[[str], str],
        context: str | None = None,
        history: int = 5,
        explicit_reasoning: bool = True,
        qualitative_notes: bool = True,
        fail_closed: bool = False,
        seed: int = 0,
    ) -> None:
        self._sampler = sampler
        self.context = context
        self.history = history
        self.explicit_reasoning = explicit_reasoning
        self.qualitative_notes = qualitative_notes
        self.fail_closed = fail_closed
        self.note: str | None = None
        self._rng = random.Random(seed)

    def propose(self, study: HPStudy) -> dict[str, Any] | None:
        if not study.space or not study.space.names():
            return None
        done = study.completed
        prompt = self._prompt(study, done)
        for attempt in range(2):
            try:
                reply = self._sampler(prompt)
            except Exception as exc:  # sampler 崩溃 → 随机兜底
                if attempt == 0:
                    prompt += (f"\n\n(sampler error: {exc}; retry once)")
                    continue
                if self.fail_closed:
                    raise
                return None
            params = self._validate_reply(study, reply)
            if params is not None:
                return params
            prompt += ("\n\nYour previous reply could not be parsed into valid "
                       "parameters. Reply again with ONLY the JSON object, values "
                       "inside the stated ranges.")
        if self.fail_closed:
            raise ValueError("agent reply unparseable twice; refusing Random fallback")
        return None

    # -- prompt 组装 -----------------------------------------------------

    def _prompt(self, study: HPStudy, done: list[HPTrial]) -> str:
        lines = [
            "You are an expert parameter-optimization engine. Think both "
            "qualitatively (what the trend and the meaning of each parameter "
            "suggest) and quantitatively (the numbers in the history) before "
            "choosing the next point.",
            "",
            f"Goal: {study.direction.upper()} the objective value.",
        ]
        if self.context:
            lines += ["", f"What is being tuned: {self.context}"]
            if self.explicit_reasoning:
                lines += [
                    "",
                    "Context-derived priors:",
                    "- Prefer stable, plausible settings before extreme exploration.",
                    "- Treat parameter names and descriptions as semantic hints, "
                    "not just tokens.",
                ]
        lines += ["", "Search space:"]
        lines += [f"- {n}: {d.describe()}" for n, d in study.space.items()]

        shown = done if self.history is None else (done[-self.history:] if self.history else [])
        best = study.best_trial
        if best is not None and best not in shown:
            shown = [best] + shown
        lines += ["", "History summary:"]
        if best is not None:
            lines += [f"- Best trial: #{best.number} value={best.value:.6g} "
                      f"params={best.params}"]
        ranked = sorted(shown, key=lambda t: t.value,
                        reverse=(study.direction == "maximize"))
        if ranked:
            lines += ["- Promising trials:"]
            for t in ranked[:5]:
                lines += [f"  - #{t.number}: value={t.value:.6g}, params={t.params}"]
            lines += ["- Recent trials:"]
            for t in shown[-5:]:
                lines += [f"  - #{t.number}: value={t.value:.6g}, params={t.params}"]
            weak = ranked[5:][-3:]
            if weak:
                lines += ["- Failed or weak regions to avoid:"]
                for t in weak:
                    lines += [f"  - #{t.number}: value={t.value:.6g}, params={t.params}"]
        if best is not None:
            lines += ["", f"Best so far: trial {best.number}, "
                          f"value={best.value:.6g}, params={best.params}"]
        if self.qualitative_notes and self.note:
            lines += ["", f"Your notes from previous trials: {self.note}"]

        lines += ["", "Propose the next point to evaluate. Balance exploration of "
                      "unvisited regions against exploitation around promising "
                      "ones; never repeat an already-evaluated point exactly."]
        if self.explicit_reasoning:
            lines += ["Use the task context as priors when available: prefer "
                      "choices that make sense for the described setup unless the "
                      "trial history clearly contradicts them."]
        keys = ", ".join(f'"{n}": <value>' for n in study.space.names())
        lines += [f'Reply with ONLY a JSON object: {{{keys}}}.']
        if self.explicit_reasoning:
            lines += ['Include a short "_reasoning" field explaining your choice.']
        if self.qualitative_notes:
            lines += ['Include a "_note" field: observations about the landscape '
                      'worth carrying forward to the next trial (it will be shown '
                      'back to you).']
        return "\n".join(lines)

    # -- 回复校验 ---------------------------------------------------------

    def _validate_reply(
        self, study: HPStudy, reply: str,
    ) -> dict[str, Any] | None:
        data = self._extract_json(reply)
        if data is None:
            return None
        if self.qualitative_notes and isinstance(data.get("_note"), str):
            self.note = data["_note"][:2000]
        names = study.space.names()
        if not all(n in data for n in names):
            return None
        try:
            return study.space.validate({n: data[n] for n in names})
        except HPSpaceError:
            return None

    @staticmethod
    def _extract_json(reply: str) -> dict[str, Any] | None:
        """提取 JSON 对象: 优先整体解析, 回退到首个 {...} 块。"""
        text = reply.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    # -- 离线 mock (hill climbing, 测试/演示零 token) ----------------------

    def mock(self, study: HPStudy) -> dict[str, Any]:
        best = study.best_trial
        if best is None or not study.space.names():
            return study.space.sample(self._rng)
        out: dict[str, Any] = {}
        for n, d in study.space.items():
            v = best.params[n]
            if isinstance(d, FloatDist):
                jitter = d.high - d.low
                out[n] = min(d.high, max(d.low, v + self._rng.gauss(0, 0.15 * jitter)))
            elif isinstance(d, IntDist):
                span = max(1, d.high - d.low)
                out[n] = min(d.high, max(d.low, int(round(v + self._rng.gauss(0, 0.15 * span)))))
            else:
                out[n] = v
        return out
