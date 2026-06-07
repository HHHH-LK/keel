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
    def __init__(self, name, side_effect_free):
        super().__init__(name, "echo")
        self.side_effect_free = side_effect_free

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
