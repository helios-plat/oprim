"""
oprim 批次 C 测试套件
======================
9 个 oprim，每个 ≥5 个测试。
LLM/Embed/Search/Persistence 全部使用 mock Protocol 实例。
http_fetch 使用 httpx 的 mock transport 或本地服务器。
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from oprim import (
    BudgetExceededError, ConversationSnapshot, EmbedResult,
    HttpOprimError, HttpResponse, LLMOprimError, LLMResponse,
    PromptOprimError, SearchOprimError, SearchResult,
    SnapshotOprimError, StreamDelta, ThinkingResult,
    build_system_prompt, embed_text, extract_thinking,
    http_fetch, llm_complete, llm_stream, snapshot_conversation,
    truncate_messages, web_search,
)


# ===========================================================================
# helpers
# ===========================================================================

def make_llm_caller(response: dict):
    """Mock LLMCaller Protocol（非流式）。"""
    async def caller(**kwargs):
        return response
    return caller


def make_streaming_caller(deltas: list[dict]):
    """Mock StreamingLLMCaller Protocol。"""
    async def caller(**kwargs) -> AsyncIterator[dict]:
        for d in deltas:
            yield d
    return caller


def make_embed_caller(vector: list[float] | None = None, raises=None):
    async def caller(*, text, model):
        if raises:
            raise raises
        return vector or [0.1] * 8
    return caller


def make_search_caller(results: list[dict] | None = None, raises=None):
    async def caller(*, query, top_k):
        if raises:
            raise raises
        return results or []
    return caller


def make_persistence(revision: str = "rev_001", raises=None):
    store = AsyncMock()
    if raises:
        store.save = AsyncMock(side_effect=raises)
    else:
        store.save = AsyncMock(return_value=revision)
    store.load = AsyncMock(return_value=None)
    return store


BASIC_MESSAGES = [{"role": "user", "content": "hello"}]
BASIC_RESPONSE = {
    "content": [{"type": "text", "text": "Hi there!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


# ===========================================================================
# llm_complete 测试
# ===========================================================================

class TestLlmComplete:
    def test_returns_llm_response(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        result = asyncio.run(llm_complete(BASIC_MESSAGES, caller=caller))
        assert isinstance(result, LLMResponse)
        assert result.text == "Hi there!"

    def test_usage_extracted(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        result = asyncio.run(llm_complete(BASIC_MESSAGES, caller=caller))
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15

    def test_cost_calculated(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        result = asyncio.run(llm_complete(BASIC_MESSAGES, caller=caller))
        assert result.cost_usd > 0

    def test_tool_calls_extracted(self):
        resp = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "bash_exec", "input": {"cmd": "ls"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        caller = make_llm_caller(resp)
        result = asyncio.run(llm_complete(BASIC_MESSAGES, caller=caller))
        assert result.has_tool_calls
        assert result.tool_calls[0]["name"] == "bash_exec"

    def test_empty_messages_raises(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        with pytest.raises(LLMOprimError, match="empty"):
            asyncio.run(llm_complete([], caller=caller))

    def test_invalid_role_raises(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        with pytest.raises(LLMOprimError, match="role"):
            asyncio.run(llm_complete(
                [{"role": "invalid_role", "content": "hi"}], caller=caller
            ))

    def test_budget_exceeded_raises(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        # 消息很长，budget 极小
        long_msgs = [{"role": "user", "content": "x" * 10000}]
        with pytest.raises(BudgetExceededError):
            asyncio.run(llm_complete(long_msgs, caller=caller, budget_tokens=10))

    def test_caller_exception_wrapped(self):
        async def bad_caller(**kwargs):
            raise RuntimeError("provider 503")
        with pytest.raises(LLMOprimError, match="failed"):
            asyncio.run(llm_complete(BASIC_MESSAGES, caller=bad_caller))

    def test_non_dict_response_raises(self):
        async def bad_caller(**kwargs):
            return "not a dict"
        with pytest.raises(LLMOprimError, match="non-dict"):
            asyncio.run(llm_complete(BASIC_MESSAGES, caller=bad_caller))

    def test_stop_reason_preserved(self):
        resp = {**BASIC_RESPONSE, "stop_reason": "max_tokens"}
        caller = make_llm_caller(resp)
        result = asyncio.run(llm_complete(BASIC_MESSAGES, caller=caller))
        assert result.stop_reason == "max_tokens"

    def test_system_param_forwarded(self):
        received = {}
        async def capturing_caller(**kwargs):
            received.update(kwargs)
            return BASIC_RESPONSE
        asyncio.run(llm_complete(BASIC_MESSAGES, caller=capturing_caller,
                                  system="You are a helper"))
        assert received.get("system") == "You are a helper"

    def test_custom_pricing(self):
        caller = make_llm_caller(BASIC_RESPONSE)
        result = asyncio.run(llm_complete(
            BASIC_MESSAGES, caller=caller,
            pricing={"in": 1e-3, "out": 2e-3}
        ))
        # 10 * 1e-3 + 5 * 2e-3 = 0.02
        assert abs(result.cost_usd - 0.02) < 0.001


# ===========================================================================
# llm_stream 测试
# ===========================================================================

class TestLlmStream:
    def test_yields_text_deltas(self):
        caller = make_streaming_caller([
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ])
        async def collect():
            deltas = []
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                deltas.append(d)
            return deltas
        deltas = asyncio.run(collect())
        texts = [d.text for d in deltas if d.type == "text"]
        assert "Hello" in texts and " world" in texts

    def test_yields_tool_use_delta(self):
        caller = make_streaming_caller([
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ])
        async def collect():
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                return d
        delta = asyncio.run(collect())
        assert delta.type == "tool_use"
        assert delta.tool_name == "bash"

    def test_yields_usage_delta(self):
        caller = make_streaming_caller([
            {"type": "usage", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ])
        async def collect():
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                return d
        delta = asyncio.run(collect())
        assert delta.type == "usage"
        assert delta.input_tokens == 10

    def test_yields_stop_delta(self):
        caller = make_streaming_caller([
            {"type": "stop", "stop_reason": "end_turn"},
        ])
        async def collect():
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                return d
        delta = asyncio.run(collect())
        assert delta.type == "stop"
        assert delta.stop_reason == "end_turn"

    def test_empty_messages_raises(self):
        caller = make_streaming_caller([])
        async def run():
            async for _ in llm_stream([], caller=caller):
                pass
        with pytest.raises(LLMOprimError, match="empty"):
            asyncio.run(run())

    def test_caller_exception_wrapped(self):
        async def bad_caller(**kwargs) -> AsyncIterator[dict]:
            raise RuntimeError("stream broken")
            yield {}  # make it a generator
        async def run():
            async for _ in llm_stream(BASIC_MESSAGES, caller=bad_caller):
                pass
        with pytest.raises(LLMOprimError):
            asyncio.run(run())

    def test_thinking_delta(self):
        caller = make_streaming_caller([
            {"type": "thinking", "thinking": "Let me think..."},
        ])
        async def collect():
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                return d
        delta = asyncio.run(collect())
        assert delta.type == "thinking"
        assert "think" in delta.text

    def test_returns_stream_delta_objects(self):
        caller = make_streaming_caller([{"type": "text", "text": "ok"}])
        async def collect():
            results = []
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                results.append(d)
            return results
        results = asyncio.run(collect())
        assert all(isinstance(d, StreamDelta) for d in results)

    def test_skips_non_dict_items(self):
        """非 dict delta 被跳过，不报错。"""
        caller = make_streaming_caller([
            "not_a_dict",
            {"type": "text", "text": "valid"},
        ])
        async def collect():
            results = []
            async for d in llm_stream(BASIC_MESSAGES, caller=caller):
                results.append(d)
            return results
        results = asyncio.run(collect())
        assert len(results) == 1
        assert results[0].text == "valid"


# ===========================================================================
# embed_text 测试
# ===========================================================================

class TestEmbedText:
    def test_returns_embed_result(self):
        caller = make_embed_caller([0.1, 0.2, 0.3])
        result = asyncio.run(embed_text("hello", caller=caller))
        assert isinstance(result, EmbedResult)

    def test_vector_correct(self):
        vec = [0.1, 0.2, 0.3]
        caller = make_embed_caller(vec)
        result = asyncio.run(embed_text("hello", caller=caller))
        assert result.vector == vec

    def test_model_preserved(self):
        caller = make_embed_caller()
        result = asyncio.run(embed_text("hi", caller=caller, model="my-embed-model"))
        assert result.model == "my-embed-model"

    def test_empty_text_raises(self):
        caller = make_embed_caller()
        with pytest.raises(LLMOprimError, match="empty"):
            asyncio.run(embed_text("", caller=caller))

    def test_whitespace_only_raises(self):
        caller = make_embed_caller()
        with pytest.raises(LLMOprimError, match="empty"):
            asyncio.run(embed_text("   ", caller=caller))

    def test_caller_error_wrapped(self):
        caller = make_embed_caller(raises=RuntimeError("api error"))
        with pytest.raises(LLMOprimError, match="failed"):
            asyncio.run(embed_text("hello", caller=caller))

    def test_invalid_vector_raises(self):
        async def bad_caller(*, text, model):
            return "not_a_list"
        with pytest.raises(LLMOprimError, match="invalid vector"):
            asyncio.run(embed_text("hello", caller=bad_caller))

    def test_token_count_estimated(self):
        caller = make_embed_caller([0.0] * 8)
        result = asyncio.run(embed_text("hello world test", caller=caller))
        assert result.token_count >= 1


# ===========================================================================
# http_fetch 测试
# ===========================================================================

class TestHttpFetch:
    def test_get_request(self):
        import httpx
        def handler(request):
            return httpx.Response(200, text="hello")
        transport = httpx.MockTransport(handler)
        async def run():
            import httpx as hx
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                resp_mock = AsyncMock()
                resp_mock.status_code = 200
                resp_mock.text = "hello"
                resp_mock.headers = {}
                resp_mock.url = "http://example.com"
                client_inst.request = AsyncMock(return_value=resp_mock)
                return await http_fetch("http://example.com")
        result = asyncio.run(run())
        assert result.status_code == 200
        assert result.text == "hello"
        assert result.ok is True

    def test_non_200_not_raises_by_default(self):
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                resp_mock = AsyncMock()
                resp_mock.status_code = 404
                resp_mock.text = "not found"
                resp_mock.headers = {}
                resp_mock.url = "http://example.com/missing"
                client_inst.request = AsyncMock(return_value=resp_mock)
                return await http_fetch("http://example.com/missing")
        result = asyncio.run(run())
        assert result.status_code == 404
        assert result.ok is False

    def test_raise_on_error(self):
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                resp_mock = AsyncMock()
                resp_mock.status_code = 500
                resp_mock.text = "internal error"
                resp_mock.headers = {}
                resp_mock.url = "http://example.com"
                client_inst.request = AsyncMock(return_value=resp_mock)
                return await http_fetch("http://example.com", raise_on_error=True)
        with pytest.raises(HttpOprimError, match="500"):
            asyncio.run(run())

    def test_timeout_raises(self):
        import httpx
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                client_inst.request = AsyncMock(
                    side_effect=httpx.TimeoutException("timeout", request=None)
                )
                return await http_fetch("http://slow.example.com", timeout=1)
        with pytest.raises(HttpOprimError, match="timed out"):
            asyncio.run(run())

    def test_returns_http_response(self):
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                resp_mock = AsyncMock()
                resp_mock.status_code = 200
                resp_mock.text = '{"key": "value"}'
                resp_mock.headers = {"content-type": "application/json"}
                resp_mock.url = "http://example.com/api"
                client_inst.request = AsyncMock(return_value=resp_mock)
                return await http_fetch("http://example.com/api")
        result = asyncio.run(run())
        assert isinstance(result, HttpResponse)
        assert result.json()["key"] == "value"

    def test_json_body_sent(self):
        sent = {}
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                resp_mock = AsyncMock()
                resp_mock.status_code = 201
                resp_mock.text = ""
                resp_mock.headers = {}
                resp_mock.url = "http://example.com"
                async def capture_request(method, url, **kwargs):
                    sent.update(kwargs)
                    return resp_mock
                client_inst.request = capture_request
                return await http_fetch("http://example.com",
                                        method="POST", body={"a": 1})
        asyncio.run(run())
        assert sent.get("json") == {"a": 1}

    def test_request_error_wrapped(self):
        import httpx
        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                client_inst = AsyncMock()
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=client_inst)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                client_inst.request = AsyncMock(
                    side_effect=httpx.RequestError("conn refused", request=None)
                )
                return await http_fetch("http://dead.example.com")
        with pytest.raises(HttpOprimError, match="failed"):
            asyncio.run(run())


# ===========================================================================
# web_search 测试
# ===========================================================================

class TestWebSearch:
    def test_returns_results(self):
        caller = make_search_caller([
            {"title": "Python docs", "url": "https://docs.python.org", "snippet": "Official docs"},
        ])
        results = asyncio.run(web_search("python", client=caller))
        assert len(results) == 1
        assert results[0].title == "Python docs"

    def test_empty_query_raises(self):
        caller = make_search_caller()
        with pytest.raises(SearchOprimError, match="empty"):
            asyncio.run(web_search("", client=caller))

    def test_top_k_limits(self):
        caller = make_search_caller([{"title": f"r{i}", "url": f"u{i}", "snippet": ""} for i in range(10)])
        results = asyncio.run(web_search("q", client=caller, top_k=3))
        assert len(results) <= 3

    def test_caller_error_wrapped(self):
        caller = make_search_caller(raises=RuntimeError("search api down"))
        with pytest.raises(SearchOprimError, match="failed"):
            asyncio.run(web_search("query", client=caller))

    def test_returns_search_result_objects(self):
        caller = make_search_caller([{"title": "t", "url": "u", "snippet": "s"}])
        results = asyncio.run(web_search("q", client=caller))
        assert all(isinstance(r, SearchResult) for r in results)

    def test_rank_assigned(self):
        caller = make_search_caller([
            {"title": "a", "url": "u1", "snippet": ""},
            {"title": "b", "url": "u2", "snippet": ""},
        ])
        results = asyncio.run(web_search("q", client=caller))
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_non_dict_items_filtered(self):
        caller = make_search_caller([
            {"title": "good", "url": "u", "snippet": ""},
            "bad_item",
        ])
        results = asyncio.run(web_search("q", client=caller))
        assert len(results) == 1


# ===========================================================================
# build_system_prompt 测试
# ===========================================================================

class TestBuildSystemPrompt:
    def test_default_build_mode(self):
        prompt = build_system_prompt()
        assert "BUILD" in prompt
        assert "hicode" in prompt

    def test_plan_mode(self):
        prompt = build_system_prompt(mode="plan")
        assert "PLAN" in prompt
        assert "Do NOT write files" in prompt

    def test_invalid_mode_raises(self):
        with pytest.raises(PromptOprimError, match="invalid mode"):
            build_system_prompt(mode="invalid")

    def test_agents_md_included(self):
        prompt = build_system_prompt(agents_md="# My Project\nPython service.")
        assert "My Project" in prompt

    def test_empty_agents_md_omitted(self):
        prompt = build_system_prompt(agents_md="")
        assert "Project Memory" not in prompt

    def test_tools_summary_included(self):
        prompt = build_system_prompt(tools_summary="bash_exec: run commands")
        assert "bash_exec" in prompt

    def test_skills_context_included(self):
        prompt = build_system_prompt(skills_context="# Refactor Skill\nDo X then Y.")
        assert "Refactor Skill" in prompt

    def test_custom_sections(self):
        prompt = build_system_prompt(custom_sections={"Rules": "Never delete files."})
        assert "Never delete files" in prompt

    def test_max_length_truncates(self):
        prompt = build_system_prompt(
            agents_md="x" * 10000,
            max_length=100,
        )
        assert len(prompt) == 100

    def test_empty_custom_section_omitted(self):
        prompt = build_system_prompt(custom_sections={"Empty": ""})
        assert "Empty" not in prompt


# ===========================================================================
# truncate_messages 测试
# ===========================================================================

class TestTruncateMessages:
    def test_short_messages_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = truncate_messages(msgs, budget=10000)
        assert result == msgs

    def test_truncates_to_budget(self):
        # 20 条消息，每条约 500 tokens（2000 chars）
        long_msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000}
            for i in range(20)
        ]
        result = truncate_messages(long_msgs, budget=2000, keep_first=1, keep_last=2)
        # 结果比原始短（确实发生了截断）
        assert len(result) < len(long_msgs)

    def test_keeps_first_and_last(self):
        msgs = [{"role": "user", "content": f"msg{i}" * 200} for i in range(10)]
        result = truncate_messages(msgs, budget=100, keep_first=1, keep_last=2)
        # 第一条和最后两条在结果里
        assert any("msg0" in m["content"] for m in result)
        assert any("msg9" in m["content"] for m in result)

    def test_empty_returns_empty(self):
        assert truncate_messages([], budget=1000) == []

    def test_invalid_budget_raises(self):
        with pytest.raises(PromptOprimError, match="budget"):
            truncate_messages([{"role": "user", "content": "x"}], budget=0)

    def test_result_is_list_of_dicts(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = truncate_messages(msgs, budget=1000)
        assert all(isinstance(m, dict) for m in result)

    def test_very_tight_budget(self):
        """极小 budget：至少保留 keep_first + keep_last。"""
        msgs = [{"role": "user", "content": f"m{i}" * 100} for i in range(5)]
        result = truncate_messages(msgs, budget=1, keep_first=1, keep_last=1)
        # 结果不为空（保留首尾）
        assert len(result) >= 1


# ===========================================================================
# extract_thinking 测试
# ===========================================================================

class TestExtractThinking:
    def test_extracts_thinking_block(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ]
        }
        result = extract_thinking(response)
        assert result.has_thinking is True
        assert "think" in result.thinking
        assert "42" in result.text

    def test_no_thinking_block(self):
        response = {"content": [{"type": "text", "text": "Simple answer."}]}
        result = extract_thinking(response)
        assert result.has_thinking is False
        assert result.thinking == ""
        assert "Simple" in result.text

    def test_multiple_thinking_blocks(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "Step 1"},
                {"type": "text", "text": "Part 1"},
                {"type": "thinking", "thinking": "Step 2"},
                {"type": "text", "text": "Part 2"},
            ]
        }
        result = extract_thinking(response)
        assert len(result.thinking_blocks) == 2
        assert len(result.text_blocks) == 2

    def test_string_content(self):
        response = {"content": "plain string"}
        result = extract_thinking(response)
        assert result.text == "plain string"
        assert not result.has_thinking

    def test_invalid_content_raises(self):
        with pytest.raises(Exception):
            extract_thinking({"content": 42})

    def test_tool_use_blocks_ignored(self):
        response = {
            "content": [
                {"type": "tool_use", "name": "bash", "input": {}},
                {"type": "text", "text": "done"},
            ]
        }
        result = extract_thinking(response)
        assert result.text == "done"
        assert not result.has_thinking

    def test_returns_thinking_result(self):
        response = {"content": [{"type": "text", "text": "x"}]}
        assert isinstance(extract_thinking(response), ThinkingResult)


# ===========================================================================
# snapshot_conversation 测試
# ===========================================================================

class TestSnapshotConversation:
    def test_returns_snapshot(self):
        store = make_persistence()
        result = asyncio.run(snapshot_conversation(
            BASIC_MESSAGES, store=store, session_id="s1"
        ))
        assert isinstance(result, ConversationSnapshot)

    def test_snapshot_id_unique(self):
        store = make_persistence()
        s1 = asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))
        s2 = asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))
        assert s1.snapshot_id != s2.snapshot_id

    def test_message_count_correct(self):
        msgs = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}]
        store = make_persistence()
        result = asyncio.run(snapshot_conversation(msgs, store=store))
        assert result.message_count == 2

    def test_session_id_in_store_key(self):
        store = make_persistence()
        result = asyncio.run(snapshot_conversation(
            BASIC_MESSAGES, store=store, session_id="my_session"
        ))
        assert "my_session" in result.store_key

    def test_store_called_with_json(self):
        store = make_persistence()
        asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))
        store.save.assert_called_once()
        _, kwargs = store.save.call_args[0], store.save.call_args[1]
        value = store.save.call_args[1]["value"]
        parsed = json.loads(value)
        assert parsed["messages"] == BASIC_MESSAGES

    def test_store_error_raises(self):
        store = make_persistence(raises=RuntimeError("db error"))
        with pytest.raises(SnapshotOprimError, match="failed"):
            asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))

    def test_non_serializable_raises(self):
        msgs = [{"role": "user", "content": object()}]  # 不可序列化
        store = make_persistence()
        with pytest.raises(SnapshotOprimError, match="serialize"):
            asyncio.run(snapshot_conversation(msgs, store=store))

    def test_revision_from_store(self):
        store = make_persistence(revision="my_rev_001")
        result = asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))
        assert result.revision == "my_rev_001"

    def test_auto_session_id_generated(self):
        store = make_persistence()
        result = asyncio.run(snapshot_conversation(BASIC_MESSAGES, store=store))
        assert result.session_id  # 有值，不为空
