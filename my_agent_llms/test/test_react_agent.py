"""ReActAgent 单元测试：使用 FakeLLM，不打网络。

运行：
    python3 -m unittest my_agent_llms.test.test_react_agent -v
"""
import unittest

from my_agent_llms.agents.react_agent import MyReActAgent
from my_agent_llms.tools import CalculatorTool, ToolRegistry


class FakeLLM:
    """按脚本顺序返回预设回复，记录收到的 messages 以便断言。"""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []
        self.provider = "fake"
        self.model = "fake-model"

    def _next(self):
        if not self.scripted:
            return ""
        return self.scripted.pop(0)

    def think(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self._next()

    def invoke(self, messages, **kwargs):
        return self.think(messages, **kwargs)


class ParseToolCallsTest(unittest.TestCase):
    def setUp(self):
        self.agent = MyReActAgent(
            "react",
            llm=FakeLLM([]),
            tool_registry=ToolRegistry(),
            enable_tool_calling=True,
        )

    def test_no_tool_call(self):
        self.assertEqual(self.agent._parse_tool_calls("hello"), [])

    def test_single_tool_call(self):
        calls = self.agent._parse_tool_calls("先算 [TOOL_CALL:calculator:1+1] 看看")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool_name"], "calculator")
        self.assertEqual(calls[0]["parameters"], "1+1")
        self.assertEqual(calls[0]["original"], "[TOOL_CALL:calculator:1+1]")

    def test_multiple_tool_calls(self):
        calls = self.agent._parse_tool_calls(
            "[TOOL_CALL:calculator:1+1] 然后 [TOOL_CALL:calculator:2+2]"
        )

        self.assertEqual([c["parameters"] for c in calls], ["1+1", "2+2"])


class RunWithoutToolsTest(unittest.TestCase):
    def test_returns_llm_response_and_records_history(self):
        llm = FakeLLM(["你好，我是 ReAct agent"])
        agent = MyReActAgent(
            "react",
            llm=llm,
            tool_registry=ToolRegistry(),
            enable_tool_calling=False,
        )

        out = agent.run("你好")

        self.assertEqual(out, "你好，我是 ReAct agent")
        self.assertEqual(len(llm.calls), 1)
        history = agent.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "你好")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "你好，我是 ReAct agent")

    def test_system_prompt_included(self):
        llm = FakeLLM(["ok"])
        agent = MyReActAgent(
            "react",
            llm=llm,
            tool_registry=ToolRegistry(),
            system_prompt="你是测试助手",
            enable_tool_calling=False,
        )

        agent.run("hi")

        self.assertEqual(llm.calls[0][0]["role"], "system")
        # 基类 get_system_prompt 会把用户 prompt 放前、诚实协议(HONESTY_CONTRACT)拼后
        system_content = llm.calls[0][0]["content"]
        self.assertTrue(system_content.startswith("你是测试助手"))
        self.assertIn("关于诚实回答的协议", system_content)


class RunWithToolsTest(unittest.TestCase):
    def _build_agent(self, scripted):
        llm = FakeLLM(scripted)
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        agent = MyReActAgent(
            "react",
            llm=llm,
            tool_registry=registry,
            enable_tool_calling=True,
        )
        return agent, llm

    def test_calculator_loop(self):
        agent, llm = self._build_agent([
            "我需要计算 [TOOL_CALL:calculator:15*8+32]",
            "结果是 152",
        ])

        out = agent.run("15*8+32 等于多少")

        self.assertEqual(out, "结果是 152")
        last_messages = llm.calls[-1]
        self.assertTrue(
            any("工具执行结果" in m["content"] for m in last_messages),
            "tool result should be appended to messages",
        )
        self.assertTrue(
            any("152" in m["content"] for m in last_messages),
            "calculator output 152 should appear in messages",
        )

    def test_tool_prompt_injected_into_system(self):
        agent, llm = self._build_agent(["不需要工具直接回答"])

        agent.run("hi")

        system_message = llm.calls[0][0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("可用工具", system_message["content"])
        self.assertIn("calculator", system_message["content"])

    def test_max_iterations_breaks_loop(self):
        agent, llm = self._build_agent(["[TOOL_CALL:calculator:1+1]"] * 10)

        agent.run("test", max_tool_iterations=2)

        self.assertLessEqual(len(llm.calls), 3)

    def test_no_tool_call_returns_directly(self):
        agent, llm = self._build_agent(["这个问题不需要工具，直接回答"])

        out = agent.run("你好")

        self.assertEqual(out, "这个问题不需要工具，直接回答")
        self.assertEqual(len(llm.calls), 1)


class ExecuteToolCallTest(unittest.TestCase):
    def test_unknown_tool_returns_error_text(self):
        agent = MyReActAgent(
            "react",
            llm=FakeLLM([]),
            tool_registry=ToolRegistry(),
            enable_tool_calling=True,
        )

        result = agent._execute_tool_call("missing", "abc")

        self.assertIn("未找到工具", result)


if __name__ == "__main__":
    unittest.main()
