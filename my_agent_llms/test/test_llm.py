import unittest

from my_agent_llms.core.llm import MyLLM


class NormalizeMessagesTest(unittest.TestCase):
    def test_accepts_openai_style_messages(self):
        messages = [{"role": "user", "content": "你好"}]
        normalized = MyLLM._normalize_messages(messages)
        self.assertEqual(normalized, [{"role": "user", "content": "你好"}])

    def test_accepts_shorthand_role_mapping(self):
        messages = [{"user": "请介绍一下你自己"}]
        normalized = MyLLM._normalize_messages(messages)
        self.assertEqual(normalized, [{"role": "user", "content": "请介绍一下你自己"}])

if __name__ == "__main__":
    unittest.main()
