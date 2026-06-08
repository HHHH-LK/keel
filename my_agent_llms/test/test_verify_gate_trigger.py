"""verify-retry 闸门只在"动过改动类工具"时触发(修问答被凭空验证→重复/凑关键词)。

背景:enable_verify=True 时,闸门原本"用过任何工具(tool_call_count>0)"就开。
但纯只读探索/问答(Read/LS/Glob/Grep)也会触发 → SpecGenerator 给开放问答凭空编
spec(string_contains/tool_called)→ 反馈回灌 → 模型凑关键词 + 重答整篇。
修法:只有本轮执行过有副作用(side_effect_free=False)的工具才开闸。
"""
from types import SimpleNamespace

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.registry import ToolRegistry


class _EchoTool(Tool):
    def __init__(self, name, side_effect_free, verify_exempt=False):
        super().__init__(name, "echo")
        self.side_effect_free = side_effect_free
        self.verify_exempt = verify_exempt

    def run(self, parameters):
        return "ok"

    def get_parameters(self):
        return [ToolParameter(name="x", type="string", description="",
                              required=False, default="")]


def _bare_agent(tools):
    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    reg = ToolRegistry()
    for t in tools:
        reg.register_tool(t)
    agent.tool_registry = reg
    agent.tool_timeout = None
    agent._turn_mutated = False
    return agent


def _tc(name, tcid):
    return SimpleNamespace(id=tcid, type="function",
                           function=SimpleNamespace(name=name, arguments="{}"))


# ── 决策:闸门开关 ────────────────────────────────────────────
def test_should_run_verify_requires_mutation_and_switch():
    agent = MyFunctionCallAgent.__new__(MyFunctionCallAgent)
    agent.enable_verify = True
    assert agent._should_run_verify(mutated=True) is True
    assert agent._should_run_verify(mutated=False) is False    # 纯只读 → 不验证
    agent.enable_verify = False
    assert agent._should_run_verify(mutated=True) is False     # 开关关 → 不验证


# ── 标记:只有有副作用工具才置 _turn_mutated ────────────────────
def test_readonly_tool_does_not_mark_turn_mutated():
    agent = _bare_agent([_EchoTool("Read", side_effect_free=True)])
    agent._turn_mutated = False
    agent._execute_tool_calls([_tc("Read", "1")], [],
                              on_tool_call=None, on_permission_request=None,
                              on_tool_result=None)
    assert agent._turn_mutated is False


def test_side_effecting_tool_marks_turn_mutated():
    agent = _bare_agent([_EchoTool("Write", side_effect_free=False)])
    agent._turn_mutated = False
    agent._execute_tool_calls([_tc("Write", "1")], [],
                              on_tool_call=None, on_permission_request=None,
                              on_tool_result=None)
    assert agent._turn_mutated is True


# ── 记账型工具(记忆/待办):有副作用但不是"值得验证的产物" → 不触发 verify ──
def test_bookkeeping_tool_does_not_mark_turn_mutated():
    # remember/write_todo 这类:side_effect_free=False(仍串行执行),
    # 但 verify_exempt=True → 没写代码就不该凭空触发事后验证。
    agent = _bare_agent([_EchoTool("remember", side_effect_free=False,
                                   verify_exempt=True)])
    agent._turn_mutated = False
    agent._execute_tool_calls([_tc("remember", "1")], [],
                              on_tool_call=None, on_permission_request=None,
                              on_tool_result=None)
    assert agent._turn_mutated is False


# ── 内置工具的标记契约 ──────────────────────────────────────────
def test_recall_is_side_effect_free():
    from my_agent_llms.tools.builtin.recall import RecallTool
    assert getattr(RecallTool, "side_effect_free", False) is True


def test_remember_is_verify_exempt():
    from my_agent_llms.tools.builtin.remember import RememberTool
    assert getattr(RememberTool, "verify_exempt", False) is True


def test_write_todo_is_verify_exempt():
    from my_agent_llms.planning.todo import WriteTodoTool
    assert getattr(WriteTodoTool, "verify_exempt", False) is True
