"""quality_check: 工具调用响应 (content 空 + tool_calls) 必须直接通过。

回归: 合法工具调用被升级重试覆盖 — deepseek 返回 content=''+tool_calls
→ 旧闸门判 empty → 升级 gpt → 返回 'None' → 主脑静默空 (用户'不回复')。
"""
from oprim._quality_gate import quality_check


def test_tool_calls_response_passes_gate():
    r = {"choices": [{"message": {"role": "assistant", "content": "",
                                   "tool_calls": [{"id": "c1", "type": "function",
                                                   "function": {"name": "mcp_od_create_project",
                                                                "arguments": "{}"}}]}}],
         "usage": {}}
    gate = quality_check(r)
    assert gate["ok"] is True
    assert gate["reason"] == "tool_calls"


def test_empty_without_tool_calls_still_fails():
    r = {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": {}}
    gate = quality_check(r)
    assert gate["ok"] is False


def test_none_literal_passes_gate_here():
    # 'None' 字面量由 llm.py _single 无效响应层拦截 (换模型重试),
    # 到不了 quality_gate; gate 层对它 pass 是预期 (避免双重拦截)。
    r = {"choices": [{"message": {"role": "assistant", "content": "None"}}], "usage": {}}
    gate = quality_check(r)
    assert gate["ok"] is True


def test_error_marker_still_fails():
    r = {"choices": [{"message": {"role": "assistant",
                                   "content": "opencode-go 调用失败: boom"}}], "usage": {}}
    gate = quality_check(r)
    assert gate["ok"] is False
    assert gate["reason"] == "error_marker"
