from my_agent_llms.tdd.classify import classify, TddDecision


class _FakeLLM:
    """MyLLM.invoke(messages) -> str 的最小替身,记录调用次数与 kwargs。"""
    def __init__(self, content):
        self._content = content
        self.calls = 0
        self.last_kwargs = None

    def invoke(self, messages, **kw):
        self.calls += 1
        self.last_kwargs = kw
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


# ── 前置过滤:常见问候/闲聊直接判 False,根本不调 LLM(省调用 + 不靠模型抽签)──
def test_greeting_short_circuits_without_llm():
    for greeting in ["hello", "Hi!", "你好", "在吗?", "嗨", "hey", "您好。"]:
        llm = _FakeLLM('{"use_tdd": true}')   # 即便模型想说 yes 也不该被调用
        d = classify(llm, greeting)
        assert d.use_tdd is False, f"{greeting!r} 应被前置过滤判 False"
        assert llm.calls == 0, f"{greeting!r} 不该触发 LLM 调用"


def test_coding_task_still_reaches_llm():
    # 极短但是真代码任务(如"写个快排"),不能被问候过滤误杀,要照常问 LLM。
    llm = _FakeLLM('{"use_tdd": true, "reason": "可写测试"}')
    d = classify(llm, "写个快排")
    assert d.use_tdd is True
    assert llm.calls == 1


# ── 确定化:classify 必须以 temperature=0 调 LLM,消除"同一输入结果飘"──
def test_classify_invokes_llm_deterministically():
    llm = _FakeLLM('{"use_tdd": false, "reason": "x"}')
    classify(llm, "写个加法函数")
    assert llm.last_kwargs.get("temperature") == 0
