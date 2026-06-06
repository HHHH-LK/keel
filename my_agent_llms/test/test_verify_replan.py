"""阶段2a:STUCK/OSCILLATING 触发重新规划。"""
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
    def __init__(self, name="noop"):
        super().__init__(name, "stub")
        self.side_effect_free = True

    def run(self, parameters):
        return f"{self.name}-ok"

    def get_parameters(self):
        return []


def _make_agent(monkeypatch, responses, *, spec, tools=None, replan_budget=1):
    from my_agent_llms.core.llm import MyLLM
    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"; llm.model = "stub"; llm.client = SimpleNamespace()
    llm.temperature = 0; llm.max_tokens = 100

    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "t"; agent.llm = llm
    agent.tool_registry = ToolRegistry()
    for t in (tools or []):
        agent.tool_registry.register_tool(t)
    agent.max_steps = 10
    agent.last_tool_call_count = 0
    agent.system_prompt = ""; agent.config = None; agent.tool_timeout = None
    agent.workspace = None
    agent.replan_budget = replan_budget
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tempfile.mkdtemp()), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp: None

    agent.enable_verify = True
    from my_agent_llms.verify import CheckerRunner
    agent.spec_generator = SimpleNamespace(generate=lambda task, *, tools: spec)
    agent.checker_runner = CheckerRunner(llm=None)
    agent.convergence_judge = ConvergenceJudge(hard_cap=10, K=2)

    calls = list(responses)
    captured: List[List[Dict[str, Any]]] = []

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        captured.append([dict(m) for m in messages])
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    agent._captured = captured
    return agent


def test_make_plan_builds_prompt_and_returns_output():
    from my_agent_llms.verify.replan import make_plan
    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["content"] = messages[0]["content"]
            return "新计划:1. 先 X 2. 再 Y"

    out = make_plan(FakeLLM(), task="完成任务X", stuck_feedback="验收项A反复没过")
    assert out == "新计划:1. 先 X 2. 再 Y"
    assert "完成任务X" in captured["content"]      # 任务进了 prompt
    assert "验收项A反复没过" in captured["content"]  # 卡点进了 prompt


def test_make_plan_handles_empty_llm_reply():
    from my_agent_llms.verify.replan import make_plan
    out = make_plan(SimpleNamespace(invoke=lambda m: None), task="t", stuck_feedback="f")
    assert out == ""


def test_make_plan_method_delegates(monkeypatch):
    """agent._make_plan 委托到 verify.replan.make_plan,传入 self.llm。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "X"})])
    agent = _make_agent(monkeypatch, [_text_response("x")], spec=spec)
    import my_agent_llms.agents.function_call_agent as fca
    monkeypatch.setattr(fca, "make_plan",
                        lambda llm, task, stuck_feedback: f"PLAN({task}|{stuck_feedback})")
    assert agent._make_plan("任务", "卡点") == "PLAN(任务|卡点)"


def test_stuck_triggers_replan(monkeypatch):
    """连续残差不降 → STUCK → 触发一次 replan;新计划注入后第三答收敛。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "DONE"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"),        # 触发工具门控
         _text_response("try1"), _text_response("try2"),   # 两轮无 DONE → STUCK
         _text_response("try3 DONE")],       # replan 后含 DONE → 收敛
        spec=spec, tools=[_StubTool("noop")], replan_budget=1)
    called = {"n": 0}
    monkeypatch.setattr(agent, "_make_plan",
                        lambda task, fb: (called.__setitem__("n", called["n"] + 1), "换思路:输出 DONE")[1])
    out = agent.run("t")
    assert called["n"] == 1               # STUCK 触发了一次 replan
    assert out == "try3 DONE"             # replan 后收敛到含 DONE 的答
    assert any("换思路" in m.get("content", "")
               for cap in agent._captured for m in cap)   # 新计划注入了 messages


def test_replan_budget_zero_stops_on_stuck(monkeypatch):
    """budget=0 → STUCK 直接止损,不调 _make_plan,返回 best。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "DONE"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"), _text_response("try1"), _text_response("try2")],
        spec=spec, tools=[_StubTool("noop")], replan_budget=0)

    def _boom(task, fb):
        raise AssertionError("budget=0 不该调用 _make_plan")
    monkeypatch.setattr(agent, "_make_plan", _boom)
    out = agent.run("t")
    assert out in {"try1", "try2"}        # STUCK 止损,返回 best


def test_oscillating_triggers_replan(monkeypatch):
    """同一答重复 → 指纹重现 OSCILLATING(K=99 不触发 STUCK)→ replan。"""
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "DONE"})])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"),
         _text_response("same"), _text_response("same"),    # 指纹重现 → OSCILLATING
         _text_response("DONE here")],
        spec=spec, tools=[_StubTool("noop")], replan_budget=1)
    agent.convergence_judge = ConvergenceJudge(hard_cap=10, K=99)   # 排除 STUCK,只看 OSCILLATING
    called = {"n": 0}
    monkeypatch.setattr(agent, "_make_plan",
                        lambda task, fb: (called.__setitem__("n", called["n"] + 1), "换思路")[1])
    out = agent.run("t")
    assert called["n"] == 1
    assert out == "DONE here"
