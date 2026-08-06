"""oprim.agent_codegen — AutoAgent-style agent/orchestrator code generation."""

from __future__ import annotations

import textwrap
from typing import Any


def agent_codegen(spec: dict[str, Any]) -> str:
    name = str(spec.get("name") or "UnnamedAgent")
    func_name = _func_name(name)
    desc = str(spec.get("description") or "")
    tools = list(spec.get("tools") or [])
    instructions = str(spec.get("instructions") or "You are a helpful agent.")
    model = str(spec.get("model") or "claude-sonnet-4-6")
    tools_repr = "[" + ", ".join(repr(t) for t in tools) + "]"

    lines = [
        f'"""Auto-generated agent: {name} (3O oprim.agent_codegen)."""',
        'from obase.agent_registry import register_agent',
        '',
        f'def get_{func_name}(model: str = {model!r}):',
        f'    """{desc}"""',
        f'    instructions = {instructions!r}',
        '    return {',
        f'        "name": {name!r},',
        '        "model": model,',
        f'        "description": {desc!r},',
        '        "instructions": instructions,',
        f'        "tools": {tools_repr},',
        '        "handoffs": {},',
        '    }',
        '',
        f'@register_agent(name={name!r}, func_name="get_{func_name}")',
        f'def _factory(model: str = {model!r}):',
        f'    return get_{func_name}(model)',
    ]
    return "\n".join(lines) + "\n"


def workflow_codegen(spec: dict[str, Any]) -> str:
    name = str(spec.get("name") or "unnamed_workflow")
    func_name = _func_name(name)
    events = list(spec.get("events") or [])
    lines = [
        f'"""Auto-generated event workflow: {name} (3O)."""',
        'from oservi.event_workflow_engine import EventWorkflowEngine',
        'import asyncio',
        '',
        f'_engine = EventWorkflowEngine(name={name!r})',
    ]
    for idx, ev in enumerate(events):
        ev_name = str(ev.get("name") or f"event_{idx}")
        body = str(ev.get("body") or "return {'status': 'ok'}")
        deps = [str(d) for d in ev.get("depends_on") or []]
        lines.append(f"@_engine.make_event(name={ev_name!r})")
        lines.append(f"async def {ev_name}(event_input, global_ctx):")
        for line in textwrap.dedent(body).strip().splitlines():
            lines.append("    " + line)
        if deps:
            lines.append(f"_engine.listen_group([_engine.get_event({d!r}) for d in {deps!r}], name={ev_name!r})")
    if events and not any(ev.get("depends_on") for ev in events):
        first = str(events[0].get("name") or "event_0")
        lines.append(f"_engine.listen_start({first!r})")
    lines.append(f"def run_workflow(system_input: dict): return _engine.drive(system_input)")
    return "\n".join(lines) + "\n"


def agent_handoff_switch(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("value", "")
    next_agent = result.get("agent")
    ctx = dict(result.get("context_variables") or {})
    if isinstance(value, dict):
        if value.get("agent"):
            next_agent = next_agent or value["agent"]
        ctx.update(value.get("context_variables") or {})
    return {"next_agent": next_agent, "value": value, "context_variables": ctx}


def _func_name(name: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_") or "agent"
