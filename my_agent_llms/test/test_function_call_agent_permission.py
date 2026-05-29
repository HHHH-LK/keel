"""Agent 主循环里 on_permission_request 回调的行为。"""
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.registry import ToolRegistry


class _WriteLike(Tool):
    requires_approval = True

    def __init__(self):
        super().__init__("WriteLike", "fake write tool")
        self.run_calls: List[Dict[str, Any]] = []

    def run(self, parameters):
        self.run_calls.append(dict(parameters))
        return "ok"

    def get_parameters(self):
        return [ToolParameter(name="path", type="string", description="...", required=True)]

    def preview_for_approval(self, args):
        return f"DIFF for {args.get('path')}"


class _ReadLike(Tool):
    """不需要审批的工具,对照组。"""

    def __init__(self):
        super().__init__("ReadLike", "fake read tool")
        self.run_calls: List[Dict[str, Any]] = []

    def run(self, parameters):
        self.run_calls.append(dict(parameters))
        return "content"

    def get_parameters(self):
        return [ToolParameter(name="path", type="string", description="...", required=True)]


def _make_agent(monkeypatch, tool_calls_then_done):
    """造一个 agent,_invoke_with_tools 第一次返回 tool_calls、第二次返回纯文本。"""
    from my_agent_llms.core.llm import MyLLM
    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"
    llm.model = "stub"
    llm.client = SimpleNamespace()  # 不会被用
    llm.temperature = 0
    llm.max_tokens = 100

    registry = ToolRegistry()
    write_tool = _WriteLike()
    read_tool = _ReadLike()
    registry.register_tool(write_tool)
    registry.register_tool(read_tool)

    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "test"
    agent.llm = llm
    agent.tool_registry = registry
    agent.max_steps = 3
    agent.last_tool_call_count = 0
    agent.system_prompt = ""
    agent.config = None
    # memory 用最小桩:assemble_context 返回空列表,_finalize_turn no-op
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile
    tmpdir = tempfile.mkdtemp()
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tmpdir), cold_backend="none", vector_backend="memory"))
    # response_hooks 不调用
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""

    calls = list(tool_calls_then_done)
    captured_messages: List[List[Dict[str, Any]]] = []

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        # 深拷贝当前 messages 快照,方便测试断言每次 LLM 调用看到了什么
        captured_messages.append([dict(m) for m in messages])
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    agent._captured_messages = captured_messages  # 测试用
    return agent, write_tool, read_tool


def _make_tool_call_response(name, args_json, tc_id="call_1"):
    msg = SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(
            id=tc_id,
            type="function",
            function=SimpleNamespace(name=name, arguments=args_json),
        )],
        reasoning_content=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


def _make_text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def test_approval_allowed_executes_tool(monkeypatch):
    agent, write_tool, _ = _make_agent(monkeypatch, [
        _make_tool_call_response("WriteLike", '{"path": "a.md"}'),
        _make_text_response("done"),
    ])
    calls = []
    def cb(name, args, preview):
        calls.append((name, args, preview))
        return True
    out = agent.run("write please", on_permission_request=cb)
    assert out == "done"
    assert calls == [("WriteLike", {"path": "a.md"}, "DIFF for a.md")]
    assert write_tool.run_calls == [{"path": "a.md"}]


def test_approval_rejected_skips_tool_and_feeds_denied(monkeypatch):
    agent, write_tool, _ = _make_agent(monkeypatch, [
        _make_tool_call_response("WriteLike", '{"path": "a.md"}'),
        _make_text_response("ok, skipping"),
    ])
    out = agent.run("write please", on_permission_request=lambda *a: False)
    assert out == "ok, skipping"
    assert write_tool.run_calls == []  # 工具没被执行
    # 严格断言:第二次 LLM 调用收到的 messages 里有 denied tool_result
    second_call_msgs = agent._captured_messages[1]
    denied = [m for m in second_call_msgs
              if m.get("role") == "tool" and "拒绝" in (m.get("content") or "")]
    assert len(denied) == 1
    assert "WriteLike" in denied[0]["content"]
    assert denied[0]["tool_call_id"] == "call_1"


def test_no_callback_passes_through_for_approval_tool(monkeypatch):
    """callback 缺席 → 放行(测试场景友好)。"""
    agent, write_tool, _ = _make_agent(monkeypatch, [
        _make_tool_call_response("WriteLike", '{"path": "a.md"}'),
        _make_text_response("done"),
    ])
    out = agent.run("write please")  # no on_permission_request
    assert out == "done"
    assert write_tool.run_calls == [{"path": "a.md"}]


def test_callback_not_called_for_read_tool(monkeypatch):
    """ReadLike.requires_approval=False → callback 不应触发。"""
    agent, _, read_tool = _make_agent(monkeypatch, [
        _make_tool_call_response("ReadLike", '{"path": "a.md"}'),
        _make_text_response("done"),
    ])
    calls = []
    def cb(name, args, preview):
        calls.append(name)
        return True
    out = agent.run("read please", on_permission_request=cb)
    assert out == "done"
    assert calls == []
    assert read_tool.run_calls == [{"path": "a.md"}]


def test_callback_exception_defaults_to_reject(monkeypatch):
    agent, write_tool, _ = _make_agent(monkeypatch, [
        _make_tool_call_response("WriteLike", '{"path": "a.md"}'),
        _make_text_response("ok"),
    ])
    def cb(name, args, preview):
        raise RuntimeError("UI broke")
    out = agent.run("write", on_permission_request=cb)
    assert out == "ok"
    assert write_tool.run_calls == []  # 异常 = 安全拒绝
