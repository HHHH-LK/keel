"""MyFunctionCallAgent 主循环收尾 gate 的验证-重试集成。"""
from types import SimpleNamespace
from typing import Any, Dict, List

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.convergence import ConvergenceJudge


def _text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=None)


def _make_agent(monkeypatch, responses, *, enable_verify, spec):
    from my_agent_llms.core.llm import MyLLM
    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"
    llm.model = "stub"
    llm.client = SimpleNamespace()
    llm.temperature = 0
    llm.max_tokens = 100

    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.name = "test"
    agent.llm = llm
    agent.tool_registry = ToolRegistry()
    agent.max_steps = 5
    agent.last_tool_call_count = 0
    agent.system_prompt = ""
    agent.config = None
    agent.tool_timeout = None
    from my_agent_llms.memory import MemoryManager, MemoryConfig
    from pathlib import Path
    import tempfile
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tempfile.mkdtemp()), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp: None

    # 验证组件
    agent.enable_verify = enable_verify
    if enable_verify:
        from my_agent_llms.verify import CheckerRunner
        agent.spec_generator = SimpleNamespace(generate=lambda task, *, tools: spec)
        agent.checker_runner = CheckerRunner(llm=None)
        agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    else:
        agent.spec_generator = None
        agent.checker_runner = None
        agent.convergence_judge = None

    calls = list(responses)
    captured: List[List[Dict[str, Any]]] = []

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        captured.append([dict(m) for m in messages])
        return calls.pop(0)

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    agent._captured = captured
    return agent


def test_verify_disabled_returns_first_answer(monkeypatch):
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    agent = _make_agent(monkeypatch, [_text_response("没有那个词")],
                        enable_verify=False, spec=spec)
    out = agent.run("t")
    assert out == "没有那个词"          # 旧逻辑:第一答即返回,不验证


def test_verify_retries_until_pass(monkeypatch):
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    agent = _make_agent(
        monkeypatch,
        [_text_response("还在想"), _text_response("最终结论给出")],
        enable_verify=True, spec=spec)
    out = agent.run("t")
    assert out == "最终结论给出"        # 第一答缺"结论"→反馈喂回→第二答补上
    # 第二次 LLM 调用应看到注入的 user feedback
    feedback_msgs = [m for m in agent._captured[1]
                     if m.get("role") == "user" and "验收项" in m.get("content", "")]
    assert feedback_msgs


def test_verify_returns_best_on_max_steps(monkeypatch):
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "X"}),
        Check(id="b", type="string_contains", params={"s": "Y"}),
    ])
    # 三答都不全过;第 2 答 res 最低(含 X) → 返回 best
    agent = _make_agent(
        monkeypatch,
        [_text_response("none"), _text_response("has X"), _text_response("none2"),
         _text_response("none3"), _text_response("none4")],
        enable_verify=True, spec=spec)
    out = agent.run("t")
    assert out == "has X"


def test_verify_returns_best_when_max_steps_exhausted_without_verdict(monkeypatch):
    # max_steps=2 < hard_cap=5, K=99 → 两轮都判 CONTINUE,主循环耗尽 → 必须返回 best,
    # 不能再做无验证兜底调用。
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "X"}),
        Check(id="b", type="string_contains", params={"s": "Y"}),
    ])
    agent = _make_agent(
        monkeypatch,
        [_text_response("none"), _text_response("has X")],  # 只给两条:若触发兜底第三次调用会 IndexError
        enable_verify=True, spec=spec)
    agent.max_steps = 2
    agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    out = agent.run("t")
    assert out == "has X"                 # 轮2 含 X,残差最低 → best
    assert len(agent._captured) == 2      # 没有第三次(兜底)LLM 调用
