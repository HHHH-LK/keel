from my_agent_llms.tdd.classify import classify, TddDecision


class _FakeLLM:
    """MyLLM.invoke(messages) -> str 的最小替身。"""
    def __init__(self, content):
        self._content = content

    def invoke(self, messages, **kw):
        return self._content


def test_user_override_true_wins():
    d = classify(_FakeLLM('{"use_tdd": false}'), "写函数", user_override=True)
    assert d.use_tdd is True and "override" in d.reason


def test_user_override_false_wins():
    d = classify(_FakeLLM('{"use_tdd": true}'), "写函数", user_override=False)
    assert d.use_tdd is False


def test_model_says_yes():
    d = classify(_FakeLLM('{"use_tdd": true, "reason": "可写测试的代码任务"}'), "写个加法函数")
    assert d.use_tdd is True


def test_model_says_no():
    d = classify(_FakeLLM('{"use_tdd": false, "reason": "闲聊"}'), "你好")
    assert d.use_tdd is False


def test_llm_failure_degrades_to_no_tdd():
    class _Boom:
        def invoke(self, *a, **k): raise RuntimeError("boom")
    d = classify(_Boom(), "写函数")
    assert d.use_tdd is False and "降级" in d.reason
