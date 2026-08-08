"""_hp_search Agentic HPO 原语测试 (sampler 注入, 零 token)。"""
import json

import pytest

from oprim import (
    AgentSampler,
    CategoricalDist,
    FloatDist,
    HPStudy,
    HPSpace,
    HPTrial,
    IntDist,
    RandomSampler,
    create_hp_study,
    space_from_json,
)


def _space():
    return (
        HPSpace()
        .define_float("threshold", 0.05, 0.95, context="decision threshold")
        .define_int("budget", 10, 200, log=True, context="compute budget")
        .define_categorical("mode", ["fast", "accurate"], context="tradeoff")
    )


# ── 空间定义 ────────────────────────────────────────────────────────────


def test_space_define_and_validate():
    space = _space()
    assert space.names() == ["threshold", "budget", "mode"]
    ok = space.validate({"threshold": 0.5, "budget": 100, "mode": "fast"})
    assert ok == {"threshold": 0.5, "budget": 100, "mode": "fast"}


def test_space_validate_rejects_out_of_range():
    space = _space()
    with pytest.raises(Exception):
        space.validate({"threshold": 2.0, "budget": 100, "mode": "fast"})
    with pytest.raises(Exception):
        space.validate({"threshold": 0.5, "budget": 100, "mode": "slow"})
    with pytest.raises(Exception):
        space.validate({"threshold": 0.5, "budget": 100})  # missing mode


def test_space_roundtrip_json():
    space = _space()
    data = space.to_dict()
    rebuilt = HPSpace.from_dict(data)
    assert rebuilt.names() == space.names()
    assert rebuilt.dist("threshold").low == 0.05
    assert isinstance(rebuilt.dist("budget"), IntDist)
    assert isinstance(rebuilt.dist("mode"), CategoricalDist)


def test_space_from_json():
    text = json.dumps({
        "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True,
               "context": "learning rate"},
        "batch": {"type": "int", "low": 8, "high": 256, "log": True},
        "mode": {"type": "categorical", "choices": ["a", "b"]},
    })
    space = space_from_json(text)
    assert space.names() == ["lr", "batch", "mode"]
    assert space.dist("lr").log is True


# ── Study 基础 ──────────────────────────────────────────────────────────


def test_study_ask_tell_and_best():
    study = HPStudy(direction="maximize")
    t1 = study.ask({"threshold": 0.5, "budget": 100, "mode": "fast"})
    assert t1.number == 0
    study.tell(t1, 0.7)
    t2 = study.ask({"threshold": 0.9, "budget": 50, "mode": "fast"})
    study.tell(t2, 0.9)
    assert study.best_trial.number == 1
    assert study.direction == "maximize"


def test_study_minimize_best():
    study = HPStudy(direction="minimize")
    t1 = study.ask({"threshold": 0.5, "budget": 100, "mode": "fast"})
    study.tell(t1, 0.9)
    t2 = study.ask({"threshold": 0.9, "budget": 50, "mode": "fast"})
    study.tell(t2, 0.3)
    assert study.best_trial.number == 1


def test_study_optimize_random_sampler():
    study = HPStudy(direction="minimize", seed=1)
    study.space = _space()
    hits = []

    def objective(params):
        hits.append(params)
        return abs(params["threshold"] - 0.5) + abs(params["budget"] - 80)

    study.optimize(objective, n_trials=6, sampler=RandomSampler(seed=7))
    assert len(study.trials) == 6
    assert all(t.state == "complete" for t in study.trials)
    assert study.best_trial.value is not None


def test_study_optimize_objective_crash_is_failed():
    study = HPStudy(direction="maximize", seed=1)
    study.space = _space()
    calls = {"n": 0}

    def objective(params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return 1.0

    study.optimize(objective, n_trials=3, sampler=RandomSampler(seed=7))
    states = [t.state for t in study.trials]
    assert states.count("failed") == 1
    assert study.best_trial is not None  # 其他 trial 照常


# ── 持久化 ──────────────────────────────────────────────────────────────


def test_study_save_load(tmp_path):
    study = HPStudy(direction="maximize")
    study.space = _space()
    t1 = study.ask({"threshold": 0.5, "budget": 100, "mode": "fast"})
    study.tell(t1, 0.7)
    path = tmp_path / "study.json"
    study.save(path)
    loaded = HPStudy.load(path)
    assert loaded.direction == "maximize"
    assert len(loaded.trials) == 1
    assert loaded.trials[0].value == 0.7
    assert loaded.space.names() == study.space.names()


def test_create_hp_study_storage_auto_save(tmp_path):
    study = create_hp_study(direction="minimize", storage=tmp_path / "s.json")
    study.space = _space()
    t = study.ask({"threshold": 0.5, "budget": 100, "mode": "fast"})
    study.tell(t, 0.4)
    assert (tmp_path / "s.json").exists()
    loaded = HPStudy.load(tmp_path / "s.json")
    assert loaded.trials[0].value == 0.4


# ── AgentSampler ────────────────────────────────────────────────────────


class _StubAgent:
    """可编程 stub: 按调用次数返回预置回复。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            return "{}"
        return self.replies.pop(0)


def _study_with_done():
    study = HPStudy(direction="maximize", seed=0)
    study.space = _space()
    t1 = study.ask({"threshold": 0.5, "budget": 100, "mode": "fast"})
    study.tell(t1, 0.7)
    t2 = study.ask({"threshold": 0.8, "budget": 50, "mode": "accurate"})
    study.tell(t2, 0.9)
    return study


def test_agent_sampler_prompt_contains_history_and_context():
    stub = _StubAgent(['{"threshold": 0.6, "budget": 90, "mode": "fast"}'])
    sampler = AgentSampler(stub, context="tune a classifier", history=5)
    study = _study_with_done()
    params = sampler.propose(study)
    assert params == {"threshold": 0.6, "budget": 90, "mode": "fast"}
    prompt = stub.prompts[0]
    assert "MAXIMIZE" in prompt
    assert "tune a classifier" in prompt
    assert "Best trial: #1 value=0.9" in prompt
    assert '"threshold": <value>' in prompt


def test_agent_sampler_validates_and_falls_back_to_random():
    # 非法值 (out of range) → 重试提示 → 兜底随机
    stub = _StubAgent(['{"threshold": 99, "budget": 90, "mode": "fast"}',
                       "not json at all"])
    sampler = AgentSampler(stub, fail_closed=False, seed=3)
    study = _study_with_done()
    params = sampler.propose(study)
    assert params is None  # 兜底: 由 study 随机采样
    assert len(stub.prompts) == 2
    assert "could not be parsed" in stub.prompts[1]


def test_agent_sampler_fail_closed_raises():
    stub = _StubAgent(["garbage"])
    sampler = AgentSampler(stub, fail_closed=True, seed=3)
    study = _study_with_done()
    with pytest.raises(ValueError):
        sampler.propose(study)


def test_agent_sampler_extracts_json_from_fenced_reply():
    stub = _StubAgent(['Here is my choice:\n```json\n{"threshold": 0.55, '
                       '"budget": 120, "mode": "accurate"}\n```'])
    sampler = AgentSampler(stub, seed=3)
    study = _study_with_done()
    params = sampler.propose(study)
    assert params == {"threshold": 0.55, "budget": 120, "mode": "accurate"}


def test_agent_sampler_note_carries_across_trials():
    stub = _StubAgent([
        '{"threshold": 0.6, "budget": 90, "mode": "fast", '
        '"_note": "high budget helps accuracy"}',
        '{"threshold": 0.7, "budget": 95, "mode": "fast"}',
    ])
    sampler = AgentSampler(stub, qualitative_notes=True, seed=3)
    study = _study_with_done()
    sampler.propose(study)
    assert sampler.note == "high budget helps accuracy"
    sampler.propose(study)
    assert "high budget helps accuracy" in stub.prompts[1]


def test_agent_sampler_mock_hill_climbing():
    sampler = AgentSampler(lambda p: "{}", seed=4)
    study = _study_with_done()
    params = sampler.mock(study)
    best = study.best_trial
    assert abs(params["threshold"] - best.params["threshold"]) <= 0.15 * 0.9 + 1e-9
    assert isinstance(params["budget"], int)


def test_agent_sampler_optimize_full_loop():
    stub = _StubAgent([
        '{"threshold": 0.6, "budget": 90, "mode": "fast"}',
        '{"threshold": 0.4, "budget": 110, "mode": "fast"}',
    ])
    sampler = AgentSampler(stub, seed=3)
    study = HPStudy(direction="minimize", seed=1)
    study.space = _space()

    def objective(params):
        return (params["threshold"] - 0.45) ** 2 + (params["budget"] - 100) ** 2 / 1e4

    study.optimize(objective, n_trials=2, sampler=sampler)
    assert len(study.trials) == 2
    assert study.trials[0].params["threshold"] == 0.6
    assert study.trials[1].params["threshold"] == 0.4
    assert all(t.state == "complete" for t in study.trials)
