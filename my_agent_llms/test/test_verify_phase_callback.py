"""A: verify-retry 轮要能被标记出来 —— run() 在进入"自证重试"轮前发 on_verify_phase
回调,让 CLI 能给这段(模型自动跑的核对动作)打一个区别于普通工具调用的归属头,
而不是混在主对话里像"答完又莫名其妙翻工具"。
"""
from types import SimpleNamespace
from typing import Any, Dict, List

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.tools.base import Tool
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.convergence import ConvergenceJudge


def _text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=None)


def _tool_call_response(name="noop", args="{}", tc_id="call_1"):
    tc = SimpleNamespace(id=tc_id, type="function",
                         function=SimpleNamespace(name=name, arguments=args))
    msg = SimpleNamespace(content="", tool_calls=[tc], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None)


class _StubTool(Tool):
    def __init__(self, name="noop", side_effect_free=False):
        super().__init__(name, "stub")
        self.side_effect_free = side_effect_free

    def run(self, parameters):
        return f"{self.name}-ok"

    def get_parameters(self):
        return []


def _make_agent(monkeypatch, responses, *, spec, tools=None):
    from my_agent_llms.core.llm import MyLLM
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile

    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"; llm.model = "stub"; llm.client = SimpleNamespace()
    llm.temperature = 0; llm.max_tokens = 100

    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "test"; agent.llm = llm
    agent.tool_registry = ToolRegistry()
    for t in (tools or []):
        agent.tool_registry.register_tool(t)
    agent.max_steps = 5; agent.last_tool_call_count = 0
    agent.system_prompt = ""; agent.config = None; agent.tool_timeout = None
    agent.workspace = None
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tempfile.mkdtemp()), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp, *, task_turn=False: None
    agent.enable_verify = True
    from my_agent_llms.verify import CheckerRunner
    agent.spec_generator = SimpleNamespace(generate=lambda task, *, tools: spec)
    agent.checker_runner = CheckerRunner(llm=None)
    agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    agent.replan_budget = 0

    calls = list(responses)

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    return agent


def test_on_verify_phase_fires_on_retry(monkeypatch):
    """第一答缺关键词 → 门注入反馈进入重试轮:on_verify_phase 应被触发(带轮次)。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"), _text_response("还在想"), _text_response("最终结论")],
        spec=spec, tools=[_StubTool("noop", side_effect_free=False)])
    seen: List[int] = []
    agent.run("t", on_verify_phase=lambda rnd: seen.append(rnd))
    assert seen == [1]                 # 恰好一次重试 → 触发一次,轮次=1


def test_on_verify_phase_silent_when_first_answer_passes(monkeypatch):
    """第一答就过验收 → 不进重试轮 → 回调一次都不发。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "ok"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"), _text_response("ok done")],
        spec=spec, tools=[_StubTool("noop", side_effect_free=False)])
    seen: List[int] = []
    agent.run("t", on_verify_phase=lambda rnd: seen.append(rnd))
    assert seen == []


def test_run_without_callback_still_works(monkeypatch):
    """不传 on_verify_phase 时旧行为不变(回调为 None 不崩)。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"), _text_response("没词"), _text_response("结论到")],
        spec=spec, tools=[_StubTool("noop", side_effect_free=False)])
    assert agent.run("t") == "结论到"
