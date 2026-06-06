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
