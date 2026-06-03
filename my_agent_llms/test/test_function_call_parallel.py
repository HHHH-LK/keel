"""Agent 主循环里同一轮多 tool_call 的并行/串行调度行为。

设计依据(见 memory: tool-parallelism-design):
- side_effect_free 工具 → 同一轮捞出来并行
- 其余(有副作用)→ 按模型给的原顺序串行
- 不论并行还是串行,结果都按 tool_call_id 原顺序回填 messages
"""
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.registry import ToolRegistry


def _make_agent(monkeypatch, tools, tool_calls_then_done):
    from my_agent_llms.core.llm import MyLLM
    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"
    llm.model = "stub"
    llm.client = SimpleNamespace()
    llm.temperature = 0
    llm.max_tokens = 100

    registry = ToolRegistry()
    for t in tools:
        registry.register_tool(t)

    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "test"
    agent.llm = llm
    agent.tool_registry = registry
    agent.max_steps = 3
    agent.last_tool_call_count = 0
    agent.system_prompt = ""
    agent.config = None
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile
    tmpdir = tempfile.mkdtemp()
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tmpdir), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp: None

    calls = list(tool_calls_then_done)
    captured_messages: List[List[Dict[str, Any]]] = []

    def fake_invoke(messages, tools_arg, tool_choice, on_text_chunk=None, **kw):
        captured_messages.append([dict(m) for m in messages])
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    agent._captured_messages = captured_messages
    return agent


def _multi_tool_call_response(specs):
    """specs = [(name, args_json, tc_id), ...] → 一个含多个 tool_calls 的 response。"""
    tool_calls = [
        SimpleNamespace(
            id=tc_id,
            type="function",
            function=SimpleNamespace(name=name, arguments=args_json),
        )
        for name, args_json, tc_id in specs
    ]
    msg = SimpleNamespace(content="", tool_calls=tool_calls, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


def _text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class _BarrierTool(Tool):
    """run 里撞共享 barrier。并行 → 两个同时到达通过;
    串行 → 第一个等不到第二个,超时 BrokenBarrierError。side_effect_free 可配。"""

    def __init__(self, name, barrier, side_effect_free=True):
        super().__init__(name, "barrier tool")
        self.barrier = barrier
        self.side_effect_free = side_effect_free
        self.passed = None

    def run(self, parameters):
        try:
            self.barrier.wait(timeout=1.0)
            self.passed = True
        except threading.BrokenBarrierError:
            self.passed = False
        return f"{self.name}-result"

    def get_parameters(self):
        return [ToolParameter(name="path", type="string", description="...", required=False)]


class _SleepTool(Tool):
    """按 delay 睡眠后返回 name-out,用于制造"完成顺序≠提交顺序"。"""

    def __init__(self, name, delay, side_effect_free):
        super().__init__(name, "sleep tool")
        self._delay = delay
        self.side_effect_free = side_effect_free

    def run(self, parameters):
        if self._delay:
            time.sleep(self._delay)
        return f"{self.name}-out"

    def get_parameters(self):
        return []


def test_side_effect_free_tools_run_in_parallel(monkeypatch):
    barrier = threading.Barrier(2)
    a = _BarrierTool("ReadA", barrier)
    b = _BarrierTool("ReadB", barrier)
    agent = _make_agent(monkeypatch, [a, b], [
        _multi_tool_call_response([
            ("ReadA", "{}", "call_1"),
            ("ReadB", "{}", "call_2"),
        ]),
        _text_response("done"),
    ])
    out = agent.run("read both")
    assert out == "done"
    # 串行会让 barrier 超时 → passed False;只有真并行两个都 True
    assert a.passed is True
    assert b.passed is True


def test_write_tools_do_not_run_in_parallel(monkeypatch):
    """side_effect_free=False 的工具必须串行 → barrier 超时 → 两个都 passed False。"""
    barrier = threading.Barrier(2)
    a = _BarrierTool("WriteA", barrier, side_effect_free=False)
    b = _BarrierTool("WriteB", barrier, side_effect_free=False)
    agent = _make_agent(monkeypatch, [a, b], [
        _multi_tool_call_response([
            ("WriteA", "{}", "call_1"),
            ("WriteB", "{}", "call_2"),
        ]),
        _text_response("done"),
    ])
    out = agent.run("write both")
    assert out == "done"
    assert a.passed is False
    assert b.passed is False


def test_results_backfilled_in_original_order_despite_parallel(monkeypatch):
    """并行下完成顺序≠提交顺序(R1 慢/R2 快),但回填必须按原 tool_call 顺序。"""
    r1 = _SleepTool("R1", 0.3, side_effect_free=True)   # 慢,最后完成
    w1 = _SleepTool("W1", 0.0, side_effect_free=False)  # 串行写
    r2 = _SleepTool("R2", 0.0, side_effect_free=True)   # 快,先完成
    agent = _make_agent(monkeypatch, [r1, w1, r2], [
        _multi_tool_call_response([
            ("R1", "{}", "call_1"),
            ("W1", "{}", "call_2"),
            ("R2", "{}", "call_3"),
        ]),
        _text_response("done"),
    ])
    out = agent.run("go")
    assert out == "done"
    # 第二次 LLM 调用看到的 messages 里,tool 消息必须按原顺序排列
    msgs = agent._captured_messages[1]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3"]
    assert [m["content"] for m in tool_msgs] == ["R1-out", "W1-out", "R2-out"]


def test_serial_tool_timeout_returns_message(monkeypatch):
    """串行工具超过 tool_timeout → 返回超时文案而非真实结果,agent 仍正常完成。"""
    slow = _SleepTool("Slow", 0.5, side_effect_free=False)
    agent = _make_agent(monkeypatch, [slow], [
        _multi_tool_call_response([("Slow", "{}", "call_1")]),
        _text_response("done"),
    ])
    agent.tool_timeout = 0.1
    out = agent.run("go")
    assert out == "done"
    tool_msgs = [m for m in agent._captured_messages[1] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "超时" in tool_msgs[0]["content"]
    assert "Slow-out" not in tool_msgs[0]["content"]


def test_parallel_timeout_does_not_block_fast_tool(monkeypatch):
    """并行批里慢工具超时,不应阻塞快工具;快工具拿到真实结果,慢的拿超时文案。"""
    slow = _SleepTool("Slow", 1.0, side_effect_free=True)
    fast = _SleepTool("Fast", 0.0, side_effect_free=True)
    agent = _make_agent(monkeypatch, [slow, fast], [
        _multi_tool_call_response([
            ("Slow", "{}", "call_1"),
            ("Fast", "{}", "call_2"),
        ]),
        _text_response("done"),
    ])
    agent.tool_timeout = 0.3
    t = time.monotonic()
    out = agent.run("go")
    elapsed = time.monotonic() - t
    assert out == "done"
    # 不应被 slow 的 1.0s 拖住,约 0.3s 超时即返回
    assert elapsed < 0.8, f"被慢工具阻塞了: {elapsed:.2f}s"
    tool_msgs = {m["tool_call_id"]: m["content"]
                 for m in agent._captured_messages[1] if m.get("role") == "tool"}
    assert tool_msgs["call_2"] == "Fast-out"
    assert "超时" in tool_msgs["call_1"]
