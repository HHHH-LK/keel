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


from my_agent_llms.tools.base import Tool, ToolParameter


class _StubTool(Tool):
    """最小工具:可选地把 write_content 写进 write_path;side_effect_free 可配。"""

    def __init__(self, name="noop", write_path=None, write_content="",
                 side_effect_free=True):
        super().__init__(name, "stub tool")
        self.side_effect_free = side_effect_free
        self._write_path = write_path
        self._write_content = write_content

    def run(self, parameters):
        if self._write_path is not None:
            from pathlib import Path
            Path(self._write_path).write_text(self._write_content, encoding="utf-8")
        return f"{self.name}-ok"

    def get_parameters(self):
        return []


def _tool_call_response(name="noop", args="{}", tc_id="call_1"):
    tc = SimpleNamespace(id=tc_id, type="function",
                         function=SimpleNamespace(name=name, arguments=args))
    msg = SimpleNamespace(content="", tool_calls=[tc], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=None)


def _make_agent(monkeypatch, responses, *, enable_verify, spec, tools=None, workspace=None):
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
    for t in (tools or []):
        agent.tool_registry.register_tool(t)
    agent.max_steps = 5
    agent.last_tool_call_count = 0
    agent.system_prompt = ""
    agent.config = None
    agent.tool_timeout = None
    agent.workspace = workspace
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
        [_tool_call_response("noop"),
         _text_response("还在想"), _text_response("最终结论给出")],
        enable_verify=True, spec=spec, tools=[_StubTool("noop")])
    out = agent.run("t")
    assert out == "最终结论给出"        # 用过工具 → 验证生效;第一答缺"结论"→反馈喂回→第二答补上
    # 注入的 user feedback 应出现在最后一次 LLM 调用看到的 messages 里
    feedback_msgs = [m for m in agent._captured[-1]
                     if m.get("role") == "user" and "验收项" in m.get("content", "")]
    assert feedback_msgs


def test_verify_returns_best_on_max_steps(monkeypatch):
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "X"}),
        Check(id="b", type="string_contains", params={"s": "Y"}),
    ])
    # 前置工具调用触发门控;之后三答都不全过,第 2 答残差最低(含 X) → 返回 best
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"),
         _text_response("none"), _text_response("has X"),
         _text_response("none2"), _text_response("none3")],
        enable_verify=True, spec=spec, tools=[_StubTool("noop")])
    out = agent.run("t")
    assert out == "has X"


def test_verify_returns_best_when_max_steps_exhausted_without_verdict(monkeypatch):
    # 前置工具调用 + 一答;max_steps=2 < hard_cap=5, K=99 → 验证轮判 CONTINUE,主循环耗尽,
    # 必须返回 best,不能再做无验证兜底调用。
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "X"}),
        Check(id="b", type="string_contains", params={"s": "Y"}),
    ])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("noop"), _text_response("has X")],  # 只给两条:触发兜底第三次调用会 IndexError
        enable_verify=True, spec=spec, tools=[_StubTool("noop")])
    agent.max_steps = 2
    agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    out = agent.run("t")
    assert out == "has X"                 # 唯一验证过的候选 → best
    assert len(agent._captured) == 2      # 没有第三次(兜底)LLM 调用


def test_agent_stores_workspace():
    """workspace 经构造注入并存为 self.workspace;enable_verify 默认 False。"""
    from my_agent_llms.core.llm import MyLLM
    from my_agent_llms.workspace.workspace import Workspace
    from my_agent_llms.memory import MemoryConfig
    from pathlib import Path
    import tempfile

    llm = MyLLM.__new__(MyLLM)
    llm.provider = "openai"
    llm.model = "stub"
    llm.client = SimpleNamespace()
    llm.temperature = 0
    llm.max_tokens = 100

    ws = Workspace(root=tempfile.mkdtemp())
    agent = MyFunctionCallAgent(
        name="t", llm=llm, tool_registry=ToolRegistry(),
        workspace=ws,
        memory_config=MemoryConfig(
            storage_dir=Path(tempfile.mkdtemp()),
            cold_backend="none", vector_backend="memory"))
    assert agent.workspace is ws
    assert agent.enable_verify is False


def test_verify_skipped_when_no_tool_used(monkeypatch):
    """enable_verify=True 但本轮没调用工具 → 工具门控:跳过验证,首答原样返回,
    且 SpecGenerator 不被调用(零开销)。"""
    called = {"gen": False}

    def _gen(task, *, tools):
        called["gen"] = True
        return CheckSpec(task=task,
                         checks=[Check(id="a", type="string_contains", params={"s": "结论"})])

    spec = CheckSpec(task="t",
                     checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    agent = _make_agent(monkeypatch, [_text_response("纯闲聊没有那个词")],
                        enable_verify=True, spec=spec)
    agent.spec_generator = SimpleNamespace(generate=_gen)
    out = agent.run("t")
    assert out == "纯闲聊没有那个词"   # 没用工具 → 不验证,首答原样返回
    assert called["gen"] is False     # SpecGenerator 未被调用


def test_verify_workspace_hard_oracle_fires(monkeypatch, tmp_path):
    """工具写出 report.json,field_equals 读回校验 status==ok → 残差0 收敛。
    证明注入的 workspace 让硬 oracle 真正生效。"""
    import json
    from my_agent_llms.workspace.workspace import Workspace
    ws = Workspace(root=tmp_path)
    report = tmp_path / "report.json"
    writer = _StubTool("make_report", write_path=str(report),
                       write_content=json.dumps({"status": "ok"}),
                       side_effect_free=False)
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="field_equals",
              params={"path": "report.json", "key": "status", "value": "ok"},
              weight=10.0, is_hard_oracle=True)])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("make_report"), _text_response("报告已生成")],
        enable_verify=True, spec=spec, tools=[writer], workspace=ws)
    out = agent.run("t")
    assert out == "报告已生成"     # field_equals 读回文件命中 → 残差0 → 收敛返回该答


def test_verify_workspace_hard_oracle_can_fail(monkeypatch, tmp_path):
    """文件 status=bad 但 spec 要 ok → field_equals 始终失败 → 不收敛,返回 best(文本)。
    证明硬 oracle 真在读文件求值(不是被无 workspace 静默判过)。"""
    import json
    from my_agent_llms.workspace.workspace import Workspace
    ws = Workspace(root=tmp_path)
    report = tmp_path / "report.json"
    writer = _StubTool("make_report", write_path=str(report),
                       write_content=json.dumps({"status": "bad"}),
                       side_effect_free=False)
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="field_equals",
              params={"path": "report.json", "key": "status", "value": "ok"},
              weight=10.0, is_hard_oracle=True)])
    agent = _make_agent(
        monkeypatch,
        [_tool_call_response("make_report"),
         _text_response("尝试1"), _text_response("尝试2")],
        enable_verify=True, spec=spec, tools=[writer], workspace=ws)
    agent.max_steps = 3
    agent.convergence_judge = ConvergenceJudge(hard_cap=5, K=99)
    out = agent.run("t")
    # status=bad ≠ ok → 残差恒>0 → 永不收敛 → 返回 best(残差最小那轮的文本)
    assert out in {"尝试1", "尝试2"}
