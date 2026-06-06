from my_agent_llms.tdd.test_author import author_tests, ProposedTest, AuthorResult


class _FakeLLM:
    """MyLLM.invoke(messages) -> str 的最小替身,记录最后一次 messages。"""
    def __init__(self, content): self._content = content
    def invoke(self, messages, **kw):
        self.last_messages = messages
        return self._content


_GOOD = ('{"tests": [{"relpath": "test_add.py", '
         '"content": "from add import add\\n\\ndef test_add():\\n    assert add(2,3)==5\\n"}]}')


def test_author_returns_proposed_tests():
    res = author_tests(_FakeLLM(_GOOD), "写个 add 函数")
    assert isinstance(res, AuthorResult)
    assert len(res.tests) == 1
    assert res.tests[0].relpath == "test_add.py"
    assert "def test_add" in res.tests[0].content


def test_author_context_excludes_implementation():
    # 隔离不变量:出题方的上下文里不能出现"实现"二字诱导它写实现
    llm = _FakeLLM(_GOOD)
    author_tests(llm, "写个 add 函数")
    joined = " ".join(m["content"] for m in llm.last_messages)
    assert "只写测试" in joined and "不要写实现" in joined


def test_author_feedback_is_passed_through():
    llm = _FakeLLM(_GOOD)
    author_tests(llm, "写个 add 函数", feedback="上次测试是假的,重写")
    joined = " ".join(m["content"] for m in llm.last_messages)
    assert "上次测试是假的" in joined


def test_author_bad_json_returns_empty():
    res = author_tests(_FakeLLM("抱歉我不会"), "写个 add 函数")
    assert res.tests == []
