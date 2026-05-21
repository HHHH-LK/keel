"""SimpleAgent 单元测试：使用 FakeLLM，不打网络。

运行：
    uv run python -m unittest my_agent_llms.test.test_simple_agent -v
"""
import unittest

from my_agent_llms.agents.simple_agent import MySimpleAgent
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

    def stream_invoke(self, messages, **kwargs):
        text = self.think(messages, **kwargs)
        for ch in text:
            yield ch


class ParseToolCallsTest(unittest.TestCase):
    def setUp(self):
        self.agent = MySimpleAgent("t", llm=FakeLLM([]))

    def test_no_tool_call(self):
        self.assertEqual(self.agent._parse_tool_calls("hello"), [])

    def test_single_tool_call(self):
        calls = self.agent._parse_tool_calls("先算 [TOOL_CALL:calculator:1+1] 看看")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool_name"], "calculator")
        self.assertEqual(calls[0]["parameters"], "1+1")
        self.assertIn("[TOOL_CALL:calculator:1+1]", calls[0]["original"])

    def test_multiple_tool_calls(self):
        calls = self.agent._parse_tool_calls(
            "[TOOL_CALL:search:Python] 然后 [TOOL_CALL:memory:recall=user]"
        )
        self.assertEqual([c["tool_name"] for c in calls], ["search", "memory"])


class ParseParametersTest(unittest.TestCase):
    def setUp(self):
        self.agent = MySimpleAgent("t", llm=FakeLLM([]))

    def test_kv_pairs(self):
        self.assertEqual(
            self.agent._parse_tool_parameters("x", "action=search,query=Python"),
            {"action": "search", "query": "Python"},
        )

    def test_single_kv(self):
        self.assertEqual(
            self.agent._parse_tool_parameters("memory", "recall=user"),
            {"recall": "user"},
        )

    def test_search_shorthand(self):
        self.assertEqual(
            self.agent._parse_tool_parameters("search", "Python"),
            {"query": "Python"},
        )

    def test_unknown_tool_falls_back_to_input(self):
        self.assertEqual(
            self.agent._parse_tool_parameters("xyz", "abc"),
            {"input": "abc"},
        )


class RunWithoutToolsTest(unittest.TestCase):
    def test_returns_llm_response_and_records_history(self):
        llm = FakeLLM(["你好我是 AI"])
        agent = MySimpleAgent("a", llm=llm)
        out = agent.run("你好")
        self.assertEqual(out, "你好我是 AI")
        history = agent.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "你好")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "你好我是 AI")

    def test_system_prompt_included(self):
        llm = FakeLLM(["ok"])
        agent = MySimpleAgent("a", llm=llm, system_prompt="你是助手")
        agent.run("hi")
        first_messages = llm.calls[0]
        self.assertEqual(first_messages[0]["role"], "system")
        self.assertIn("你是助手", first_messages[0]["content"])


class RunWithToolsTest(unittest.TestCase):
    def _build_agent(self, scripted):
        llm = FakeLLM(scripted)
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        agent = MySimpleAgent("a", llm=llm, tool_registry=registry)
        return agent, llm

    def test_calculator_loop(self):
        agent, llm = self._build_agent([
            "我来算一下 [TOOL_CALL:calculator:15*8+32]",
            "结果是 152",
        ])
        out = agent.run("15*8+32 等于多少")
        self.assertEqual(out, "结果是 152")
        # 最后一次 LLM 调用的消息里应包含工具执行结果
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
        sys_msg = llm.calls[0][0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertIn("可用工具", sys_msg["content"])
        self.assertIn("calculator", sys_msg["content"])

    def test_max_iterations_breaks_loop(self):
        # LLM 一直发工具调用，确保循环不会无限
        agent, llm = self._build_agent(
            ["[TOOL_CALL:calculator:1+1]"] * 10
        )
        agent.run("test", max_tool_iterations=2)
        # 最多调用：2 次循环 think + 1 次 fallback invoke
        self.assertLessEqual(len(llm.calls), 3)

    def test_no_tool_call_returns_directly(self):
        agent, llm = self._build_agent(["这个问题不需要工具，直接回答"])
        out = agent.run("你好")
        self.assertEqual(out, "这个问题不需要工具，直接回答")
        self.assertEqual(len(llm.calls), 1)


class ToolManagementTest(unittest.TestCase):
    def test_add_tool_enables_tool_calling(self):
        agent = MySimpleAgent("a", llm=FakeLLM([]))
        self.assertFalse(agent.has_tools())
        agent.add_tool(CalculatorTool())
        self.assertTrue(agent.has_tools())
        self.assertIn("calculator", agent.list_tools())

    def test_remove_tool(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        agent = MySimpleAgent("a", llm=FakeLLM([]), tool_registry=registry)
        self.assertTrue(agent.remove_tool("calculator"))
        self.assertNotIn("calculator", agent.list_tools())


class StreamRunTest(unittest.TestCase):
    def test_stream_yields_chunks_and_records_history(self):
        llm = FakeLLM(["你好世界"])
        agent = MySimpleAgent("a", llm=llm)
        chunks = list(agent.stream_run("hi"))
        self.assertEqual("".join(chunks), "你好世界")
        history = agent.get_history()
        self.assertEqual(history[-1].role, "assistant")
        self.assertEqual(history[-1].content, "你好世界")


class CalculatorToolTest(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(CalculatorTool().run({"expression": "15*8+32"}), "152")

    def test_sqrt(self):
        self.assertEqual(CalculatorTool().run({"expression": "sqrt(16)"}), "4.0")

    def test_invalid_expression_returns_error_message(self):
        result = CalculatorTool().run({"expression": "import os"})
        self.assertTrue(result.startswith("计算失败"))

    def test_registry_execute_tool_with_string_param(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        self.assertEqual(registry.execute_tool("calculator", "1+2"), "3")


if __name__ == "__main__":
    unittest.main()
