"""ReflectionAgent 单元测试：使用 FakeLLM，不打网络。

运行：
    python3 -m unittest my_agent_llms.test.test_reflection_agent -v
"""
import unittest

from my_agent_llms.agents.reflection_agent import MyReflectionAgent
from my_agent_llms.tools import ToolRegistry


class FakeLLM:
    """按脚本顺序返回预设回复，并记录收到的 messages。"""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []
        self.provider = "fake"
        self.model = "fake-model"

    def invoke(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self.scripted:
            return ""
        return self.scripted.pop(0)


class ReflectionLoopTest(unittest.TestCase):
    def test_reflects_refines_then_reflects_again_until_good_enough(self):
        llm = FakeLLM([
            "初稿",
            "需要补充例子",
            "改进稿",
            "无需改进",
        ])
        agent = MyReflectionAgent(
            "reflection",
            llm=llm,
            tool_registry=ToolRegistry(),
            max_steps=3,
        )

        result = agent.run("解释 ReAct")

        self.assertEqual(result, "改进稿")
        self.assertEqual(len(llm.calls), 4)
        self.assertIn("请根据以下要求完成任务", llm.calls[0][0]["content"])
        self.assertIn("请仔细审查以下回答", llm.calls[1][0]["content"])
        self.assertIn("初稿", llm.calls[1][0]["content"])
        self.assertIn("请根据反馈意见改进你的回答", llm.calls[2][0]["content"])
        self.assertIn("初稿", llm.calls[2][0]["content"])
        self.assertIn("需要补充例子", llm.calls[2][0]["content"])
        self.assertIn("请仔细审查以下回答", llm.calls[3][0]["content"])
        self.assertIn("改进稿", llm.calls[3][0]["content"])

        history = agent.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "解释 ReAct")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "改进稿")

    def test_returns_last_response_when_max_steps_is_reached(self):
        llm = FakeLLM([
            "初稿",
            "还需要改",
            "改进稿",
        ])
        agent = MyReflectionAgent(
            "reflection",
            llm=llm,
            tool_registry=ToolRegistry(),
            max_steps=1,
        )

        result = agent.run("解释 Agent")

        self.assertEqual(result, "改进稿")
        self.assertEqual(len(llm.calls), 3)


if __name__ == "__main__":
    unittest.main()
