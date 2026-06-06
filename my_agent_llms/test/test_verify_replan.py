"""阶段2a:STUCK/OSCILLATING 触发重新规划。"""
from types import SimpleNamespace
from typing import Any, Dict, List

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.tools.base import Tool
from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.convergence import ConvergenceJudge


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
