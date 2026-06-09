"""C: 验证未通过前不把候选答案显示给用户。

回归动机(transcript "答完→自证→又答"):候选最终答案是边生成边流式打到屏幕的,
而闸门在其后才判;被拒候选已经被用户看完了。改:enable_verify+本轮改动时,候选先缓冲,
闸门判通过/止损后再一次性 emit;被拒候选丢弃,用户永不可见。带 tool_calls 的叙述照常显示。
"""
from types import SimpleNamespace
from typing import Any, Dict, List

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.base import Tool
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.convergence import ConvergenceJudge


def _text(text):
    msg = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=None)


def _toolcall(name="noop", content="", tc_id="call_1"):
    tc = SimpleNamespace(id=tc_id, type="function",
                         function=SimpleNamespace(name=name, arguments="{}"))
    msg = SimpleNamespace(content=content, tool_calls=[tc], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None)


class _MutTool(Tool):
    """有副作用工具(side_effect_free=False)→ 置 _turn_mutated → 触发 verify 闸门。"""
    def __init__(self, name="noop"):
        super().__init__(name, "stub")
        self.side_effect_free = False

    def run(self, parameters):
        return f"{self.name}-ok"

    def get_parameters(self):
        return []


def _make_agent(monkeypatch, responses, *, enable_verify, spec):
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
    agent.tool_registry.register_tool(_MutTool("noop"))
    agent.max_steps = 6; agent.last_tool_call_count = 0
    agent.system_prompt = ""; agent.config = None; agent.tool_timeout = None
    agent.workspace = None
    agent.memory = MemoryManager(MemoryConfig(
        storage_dir=Path(tempfile.mkdtemp()), cold_backend="none", vector_backend="memory"))
    agent._run_response_hooks = lambda inp, resp, msgs: resp
    agent._apply_honesty_contract = lambda p: p or ""
    agent._finalize_turn = lambda inp, resp, *, task_turn=False: None

    agent.enable_verify = enable_verify
    if enable_verify:
        from my_agent_llms.verify import CheckerRunner
        agent.spec_generator = SimpleNamespace(generate=lambda task, *, tools: spec)
        agent.checker_runner = CheckerRunner(llm=None)
        agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    else:
        agent.spec_generator = agent.checker_runner = agent.convergence_judge = None

    calls = list(responses)

    def fake_invoke(messages, tools, tool_choice, on_text_chunk=None, **kw):
        resp = calls.pop(0)
        msg = resp.choices[0].message
        if on_text_chunk and msg.content:      # 模拟流式吐 content
            on_text_chunk(msg.content)
        return resp

    monkeypatch.setattr(agent, "_invoke_with_tools", fake_invoke)
    return agent


def _spec():
    return CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "结论"})])


# ── 关闭验证:候选照常流式,且不重复 emit ───────────────────────
def test_disabled_streams_live_once(monkeypatch):
    agent = _make_agent(monkeypatch, [_text("直接答")], enable_verify=False, spec=_spec())
    shown = []
    out = agent.run("t", on_text_chunk=shown.append)
    assert out == "直接答"
    assert shown == ["直接答"]              # 流式一次,末尾不重复


# ── 开启验证 + 一轮过:候选不流式,末尾一次性 emit ──────────────
def test_converge_round1_suppresses_then_emits_once(monkeypatch):
    agent = _make_agent(monkeypatch, [_toolcall("noop"), _text("最终结论在此")],
                        enable_verify=True, spec=_spec())
    shown, starts = [], []
    out = agent.run("t", on_text_chunk=shown.append,
                    on_verify_start=lambda: starts.append(1))
    assert out == "最终结论在此"
    assert shown == ["最终结论在此"]        # 候选未在 invoke 期间流式;末尾一次性 emit
    assert len(starts) >= 1                 # 校验中信号已发


# ── 重答轮:被拒候选永不可见,只显示最终答案 ───────────────────
def test_rejected_candidate_never_shown(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_toolcall("noop"), _text("缺那个词"), _text("这次有结论了")],
        enable_verify=True, spec=_spec())
    shown, starts = [], []
    out = agent.run("t", on_text_chunk=shown.append,
                    on_verify_start=lambda: starts.append(1))
    assert out == "这次有结论了"
    assert shown == ["这次有结论了"]        # "缺那个词" 被拒 → 从未显示
    assert len(starts) >= 2                 # 两轮都进了闸门


# ── 压制期内带 tool_calls 的叙述照常 flush ──────────────────────
def test_preamble_with_toolcalls_flushed_under_suppression(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_toolcall("noop"),                       # 第 0 轮:触发改动
         _toolcall("noop", content="我再改一处"),  # 第 1 轮(已压制):带叙述 + 工具
         _text("最终结论完成")],                   # 第 2 轮:候选 → 缓冲 → 末尾 emit
        enable_verify=True, spec=_spec())
    shown = []
    out = agent.run("t", on_text_chunk=shown.append)
    assert out == "最终结论完成"
    assert shown == ["我再改一处", "最终结论完成"]  # 叙述照常显示,最终答案一次性 emit
