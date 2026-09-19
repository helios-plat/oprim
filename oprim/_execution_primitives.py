"""oprim._execution_primitives — 最小执行原语：circuit_break / restart_service / rate_limit /
rollback + dispatch_intervention + execute_fn.

"执行原语接口"（不再是空回调）：
- 每个原语受 CapabilityToken 保护（obase.permission_contract 验签后的令牌对象）
- 每个原语写入可审计 ActionLog（obase.hardened_executor.ActionLog）
- dispatch_intervention 统一派发：intervention type → capability 映射 → 对应原语
- execute_fn / aexecute_fn 统一受控执行入口（包装 obase.hardened_executor.HardenedExecutor）

组合:
- obase.permission_contract  (CapabilityToken / check_capability)
- obase.hardened_executor   (HardenedExecutor / ActionLog)
- obase.circuit_breaker     (CircuitBreaker，可选注入)
- obase.rate_limit          (RateLimitRegistry，可选注入)
- obase.persistent_store    (PersistentStore，可选注入)

铁律: ≤1 位置参数，其余 keyword-only。

例:
    >>> from obase.permission_contract import issue_capability_token
    >>> tok = issue_capability_token("s", subject="op", capabilities={"executor.circuit_break"})
    >>> tok_obj = verify_capability_token("s", tok)
    >>> circuit_break("worker-a", token=tok_obj, reason="runaway")["status"]
    'ok'
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from obase.circuit_breaker import CircuitBreaker
from obase.hardened_executor import ActionLog, HardenedExecutor
from obase.permission_contract import (
    CapabilityDeniedError,
    CapabilityToken,
    check_capability,
)
from obase.persistent_store import PersistentStore
from obase.rate_limit import RateLimitRegistry
from obase.resource_limits import ResourceEnvelope

__all__ = [
    "BreakerRegistry",
    "circuit_break",
    "clear_breaker",
    "is_circuit_open",
    "restart_service",
    "rate_limit",
    "rollback",
    "dispatch_intervention",
    "execute_fn",
    "aexecute_fn",
]

# type → 所需 capability（dispatch_intervention 用）
_INTERVENTION_CAPABILITIES: dict[str, str] = {
    "circuit_break": "executor.circuit_break",
    "clear_breaker": "executor.circuit_break",
    "restart_service": "executor.restart",
    "rate_limit": "executor.rate_limit",
    "rollback": "state.rollback",
}


class BreakerRegistry:
    """手动熔断注册表（thread-safe，支持 cooldown 过期自动失效）。"""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def open(self, service_id: str, *, reason: str, cooldown_s: float) -> None:
        with self._lock:
            self._state[service_id] = {
                "reason": reason,
                "open_until": time.monotonic() + max(0.0, cooldown_s),
            }

    def close(self, service_id: str) -> bool:
        with self._lock:
            return self._state.pop(service_id, None) is not None

    def is_open(self, service_id: str) -> dict[str, Any] | None:
        """返回打开状态（含剩余秒数）；未打开或已过期返回 None。"""
        with self._lock:
            entry = self._state.get(service_id)
            if entry is None:
                return None
            remaining = entry["open_until"] - time.monotonic()
            if remaining <= 0:
                self._state.pop(service_id, None)
                return None
            return {"reason": entry["reason"], "remaining_s": round(remaining, 3)}


_DEFAULT_BREAKER_REGISTRY = BreakerRegistry()


def _guard(token: CapabilityToken, required: str) -> str | None:
    """能力校验；失败返回错误信息（原语层不裸抛安全异常）。"""
    try:
        check_capability(token, required)
        return None
    except CapabilityDeniedError as exc:
        return str(exc)


def _log(
    action_log: ActionLog | None,
    *,
    token: CapabilityToken,
    action: str,
    capability: str,
    outcome: str,
    detail: dict[str, Any] | None = None,
) -> None:
    if action_log is None:
        return
    try:
        action_log.record(
            action=action,
            subject=token.subject,
            capability=capability,
            outcome=outcome,
            detail=detail,
        )
    except Exception:  # noqa: BLE001 — 审计失败不阻断干预
        pass


def circuit_break(
    service_id: str,
    *,
    token: CapabilityToken,
    reason: str,
    cooldown_s: float = 60.0,
    breaker: CircuitBreaker | None = None,
    registry: BreakerRegistry | None = None,
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """打开 service 的熔断：registry 记录 + 可选复位注入的 CircuitBreaker 故障计数。

    Args:
        service_id: 目标服务标识。
        token: 已验签能力令牌（需 executor.circuit_break）。
        reason: 干预原因（进 ActionLog）。
        cooldown_s: 熔断冷却时长（秒）。
        breaker: 可选 CircuitBreaker 实例（复位其故障计数）。
        registry: 手动熔断注册表（缺省模块级默认）。
    """
    cap = "executor.circuit_break"
    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action="circuit_break", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "service_id": service_id}
    (registry or _DEFAULT_BREAKER_REGISTRY).open(service_id, reason=reason, cooldown_s=cooldown_s)
    if breaker is not None:
        breaker.reset()
    _log(
        action_log,
        token=token,
        action="circuit_break",
        capability=cap,
        outcome="ok",
        detail={"service_id": service_id, "reason": reason, "cooldown_s": cooldown_s},
    )
    return {"status": "ok", "service_id": service_id, "reason": reason, "cooldown_s": cooldown_s}


def clear_breaker(
    service_id: str,
    *,
    token: CapabilityToken,
    registry: BreakerRegistry | None = None,
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """手动解除 service 的熔断。"""
    cap = "executor.circuit_break"
    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action="clear_breaker", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "service_id": service_id}
    closed = (registry or _DEFAULT_BREAKER_REGISTRY).close(service_id)
    _log(
        action_log,
        token=token,
        action="clear_breaker",
        capability=cap,
        outcome="ok",
        detail={"service_id": service_id, "closed": closed},
    )
    return {"status": "ok", "service_id": service_id, "closed": closed}


def is_circuit_open(
    service_id: str,
    *,
    registry: BreakerRegistry | None = None,
) -> dict[str, Any] | None:
    """查询 service 熔断状态（不需要令牌：只读查询）。"""
    return (registry or _DEFAULT_BREAKER_REGISTRY).is_open(service_id)


def restart_service(
    service_id: str,
    *,
    token: CapabilityToken,
    reason: str = "",
    force: bool = False,
    restarter: Callable[[str], dict[str, Any]] | None = None,
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """重启 service（restarter 注入真实重启能力；缺省返回 failed，绝不做隐式动作）。"""
    cap = "executor.restart"
    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action="restart_service", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "service_id": service_id}
    if restarter is None:
        _log(
            action_log,
            token=token,
            action="restart_service",
            capability=cap,
            outcome="failed",
            detail={"service_id": service_id, "reason": "restarter not injected"},
        )
        return {"status": "failed", "error": "restarter not injected", "service_id": service_id}
    result = restarter(service_id)
    _log(
        action_log,
        token=token,
        action="restart_service",
        capability=cap,
        outcome=result.get("status", "unknown"),
        detail={"service_id": service_id, "reason": reason, "force": force, "result": result},
    )
    return {
        "status": result.get("status", "unknown"),
        "service_id": service_id,
        "reason": reason,
        "force": force,
        "result": result,
    }


def rate_limit(
    service_id: str,
    *,
    token: CapabilityToken,
    rate: int,
    period_s: float,
    registry: type[RateLimitRegistry] | RateLimitRegistry | None = None,
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """调整 service 的限流参数（re-register 覆盖既有配置）。"""
    cap = "executor.rate_limit"
    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action="rate_limit", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "service_id": service_id}
    if rate <= 0 or period_s <= 0:
        _log(
            action_log,
            token=token,
            action="rate_limit",
            capability=cap,
            outcome="failed",
            detail={"service_id": service_id, "rate": rate, "period_s": period_s},
        )
        return {
            "status": "failed",
            "error": "rate must be > 0 and period_s must be > 0",
            "service_id": service_id,
        }
    reg = registry or RateLimitRegistry
    limiter = reg.register(service_id, rate, period_s)
    _log(
        action_log,
        token=token,
        action="rate_limit",
        capability=cap,
        outcome="ok",
        detail={"service_id": service_id, "rate": rate, "period_s": period_s},
    )
    return {
        "status": "ok",
        "service_id": service_id,
        "rate": rate,
        "period_s": period_s,
        "limiter": str(limiter),
    }


def rollback(
    service_id: str,
    *,
    token: CapabilityToken,
    to_version: str,
    kind: str = "strategy_state",
    key: str | None = None,
    store: PersistentStore | None = None,
    reason: str = "",
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """把 (kind, key) 资产回滚到指定版本（store 注入 obase.persistent_store.PersistentStore）。"""
    cap = "state.rollback"
    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action="rollback", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "service_id": service_id}
    if store is None:
        _log(
            action_log,
            token=token,
            action="rollback",
            capability=cap,
            outcome="failed",
            detail={"service_id": service_id, "reason": "store not injected"},
        )
        return {"status": "failed", "error": "store not injected", "service_id": service_id}
    try:
        new_rev = store.rollback(
            kind, key or service_id, to_version, author=token.subject, reason=reason or "rollback"
        )
    except Exception as exc:  # noqa: BLE001
        _log(
            action_log,
            token=token,
            action="rollback",
            capability=cap,
            outcome="failed",
            detail={"service_id": service_id, "to_version": to_version, "error": str(exc)},
        )
        return {"status": "failed", "error": str(exc), "service_id": service_id}
    _log(
        action_log,
        token=token,
        action="rollback",
        capability=cap,
        outcome="ok",
        detail={"service_id": service_id, "to_version": to_version, "new_rev": new_rev},
    )
    return {"status": "ok", "service_id": service_id, "to_version": to_version, "new_rev": new_rev}


def dispatch_intervention(
    intervention: dict[str, Any],
    *,
    token: CapabilityToken,
    breaker: CircuitBreaker | None = None,
    registry: BreakerRegistry | None = None,
    restarter: Callable[[str], dict[str, Any]] | None = None,
    rl_registry: type[RateLimitRegistry] | RateLimitRegistry | None = None,
    store: PersistentStore | None = None,
    action_log: ActionLog | None = None,
) -> dict[str, Any]:
    """统一干预派发：intervention type → capability 校验 → 对应原语。

    intervention 形如:
        {"type": "restart_service", "service_id": "worker-a", "reason": "OOM", "force": true}
        {"type": "rate_limit", "service_id": "worker-a", "rate": 5, "period_s": 60}
        {"type": "rollback", "service_id": "strategy-x", "to_version": "rev-abc",
        "kind": "strategy_state"}
        {"type": "circuit_break", "service_id": "worker-a", "reason": "runaway", "cooldown_s": 120}
    """
    itype = intervention.get("type")
    service_id = intervention.get("service_id")
    if itype is None or service_id is None:
        return {"status": "failed", "error": "intervention requires type + service_id"}
    cap = _INTERVENTION_CAPABILITIES.get(itype)
    if cap is None:
        return {"status": "failed", "error": f"unknown intervention type: {itype!r}"}

    denied = _guard(token, cap)
    if denied is not None:
        _log(action_log, token=token, action=f"dispatch:{itype}", capability=cap, outcome="denied")
        return {"status": "denied", "error": denied, "type": itype, "service_id": service_id}

    if itype == "circuit_break":
        return circuit_break(
            service_id,
            token=token,
            reason=intervention.get("reason", ""),
            cooldown_s=float(intervention.get("cooldown_s", 60.0)),
            breaker=breaker,
            registry=registry,
            action_log=action_log,
        )
    if itype == "clear_breaker":
        return clear_breaker(service_id, token=token, registry=registry, action_log=action_log)
    if itype == "restart_service":
        return restart_service(
            service_id,
            token=token,
            reason=intervention.get("reason", ""),
            force=bool(intervention.get("force", False)),
            restarter=restarter,
            action_log=action_log,
        )
    if itype == "rate_limit":
        return rate_limit(
            service_id,
            token=token,
            rate=int(intervention["rate"]),
            period_s=float(intervention["period_s"]),
            registry=rl_registry,
            action_log=action_log,
        )
    if itype == "rollback":
        return rollback(
            service_id,
            token=token,
            to_version=str(intervention["to_version"]),
            kind=str(intervention.get("kind", "strategy_state")),
            key=intervention.get("key"),
            store=store,
            reason=intervention.get("reason", ""),
            action_log=action_log,
        )
    return {"status": "failed", "error": f"unhandled type: {itype!r}"}  # pragma: no cover


def _prepare_execute(
    token: CapabilityToken,
    executor: HardenedExecutor,
    *,
    capability: str | None,
    envelope: ResourceEnvelope | None,
) -> tuple[dict[str, Any], ResourceEnvelope | None]:
    """execute_fn 共享前置：能力校验 + 信封校验（失败返回错误 dict）。"""
    cap = capability or executor._default_capability
    denied = _guard(token, cap)
    if denied is not None:
        return {"status": "denied", "error": denied}, None
    if envelope is not None:
        from obase.resource_limits import validate_envelope

        try:
            validate_envelope(envelope)
        except ValueError as exc:
            return {"status": "failed", "error": f"invalid envelope: {exc}"}, None
    return {"status": "ok"}, envelope


def execute_fn(
    fn: Callable[..., Any],
    *,
    token_str: str,
    secret: str,
    executor: HardenedExecutor | None = None,
    capability: str | None = None,
    envelope: ResourceEnvelope | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """统一受控执行入口（同步）：验签 → 能力校验 + 资源信封 → 执行基板（熔断/限流/蜜罐/审计）。

    让 execute_fn 不再是空回调：它真正执行 fn，并在执行前后施加全部安全横切。
    """
    from obase.permission_contract import verify_capability_token

    token = verify_capability_token(secret, token_str)
    if executor is None:
        executor = HardenedExecutor(secret=secret)
    prep, env = _prepare_execute(token, executor, capability=capability, envelope=envelope)
    if prep["status"] != "ok":
        return prep
    result = executor.execute(
        token_str, fn=fn, capability=capability, envelope=env, args=args, kwargs=kwargs
    )
    return result.to_dict()


async def aexecute_fn(
    fn: Callable[..., Any],
    *,
    token_str: str,
    secret: str,
    executor: HardenedExecutor | None = None,
    capability: str | None = None,
    envelope: ResourceEnvelope | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """统一受控执行入口（异步）：验签 → 能力校验 + 资源信封 → 执行基板（强墙钟超时）。"""
    from obase.permission_contract import verify_capability_token

    token = verify_capability_token(secret, token_str)
    if executor is None:
        executor = HardenedExecutor(secret=secret)
    prep, env = _prepare_execute(token, executor, capability=capability, envelope=envelope)
    if prep["status"] != "ok":
        return prep
    result = await executor.aexecute(
        token_str,
        fn=fn,
        capability=capability,
        envelope=env,
        args=args,
        kwargs=kwargs,
        timeout=timeout,
    )
    return result.to_dict()
