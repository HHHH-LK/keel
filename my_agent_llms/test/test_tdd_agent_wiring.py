"""验证 run() 顶部 dispatch:enable_tdd + classify 命中 → 走 _run_tdd;否则老路。"""
from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent


def _bare_agent():
    a = MyFunctionCallAgent.__new__(MyFunctionCallAgent)  # 不跑 __init__,纯测 dispatch
    a.enable_tdd = True
    a._in_tdd = False
    return a


def test_dispatch_routes_to_run_tdd(monkeypatch):
    a = _bare_agent()
    monkeypatch.setattr(a, "_tdd_should_run", lambda text: True, raising=False)
    monkeypatch.setattr(a, "_run_tdd", lambda text: "TDD_RESULT", raising=False)
    assert a._maybe_run_tdd("写个 add 函数") == "TDD_RESULT"


def test_dispatch_skips_when_disabled(monkeypatch):
    a = _bare_agent()
    a.enable_tdd = False
    assert a._maybe_run_tdd("写个 add 函数") is None  # None = 不接管,走老路


def test_dispatch_skips_when_reentrant(monkeypatch):
    a = _bare_agent()
    a._in_tdd = True
    monkeypatch.setattr(a, "_tdd_should_run", lambda text: True, raising=False)
    assert a._maybe_run_tdd("写个 add 函数") is None


def test_run_tdd_finalizes_original_task_and_overrides_classify(monkeypatch):
    """#1 顶层把"原始任务↔最终结果"写进 memory;#4 不让 orchestrator 再 classify 一次。"""
    import my_agent_llms.tdd as tdd
    from my_agent_llms.tdd import TddResult
    a = _bare_agent()
    a.enable_verify = True
    a.llm = "LLM"
    a.workspace = "WS"
    finalized = []
    a._finalize_turn = lambda inp, resp, *, task_turn=False: finalized.append((inp, resp))
    captured = {}

    def fake_run_tdd(**kw):
        captured.update(kw)
        return TddResult(success=True, message="done", degraded=False)

    monkeypatch.setattr(tdd, "run_tdd", fake_run_tdd)
    out = a._run_tdd("写个 add 函数")
    assert out == "done"
    assert captured["user_override"] is True              # #4 跳过二次 classify
    assert finalized == [("写个 add 函数", "done")]        # #1 记原始任务而非子提示词
