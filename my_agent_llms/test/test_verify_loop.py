"""verify/loop.py: VerifyRetryLoop 编排 + best 选择(fake executor + 注入组件)。"""
from types import SimpleNamespace

from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.checkers import CheckerRunner
from my_agent_llms.verify.convergence import ConvergenceJudge, Verdict
from my_agent_llms.verify.loop import VerifyRetryLoop, VerifyResult


class _FakeExecutor:
    """按预设脚本逐轮返回 result;记录收到的 feedback。"""
    def __init__(self, results):
        self._results = list(results)
        self.feedbacks = []
        self.calls = 0

    def tool_names(self):
        return ["write_file"]

    def execute(self, task, *, feedback=None):
        self.feedbacks.append(feedback)
        r = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return r, []   # trajectory 此处不参与断言


def _loop(spec, *, hard_cap=5, K=2):
    spec_gen = SimpleNamespace(generate=lambda task, *, tools: spec)
    return VerifyRetryLoop(
        spec_gen=spec_gen,
        checker_runner=CheckerRunner(llm=None),
        judge=ConvergenceJudge(hard_cap=hard_cap, K=K),
    )


def test_converges_first_round_when_all_pass():
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "ok"})])
    loop = _loop(spec)
    ex = _FakeExecutor(["ok done"])
    out = loop.run("t", ex)
    assert isinstance(out, VerifyResult)
    assert out.verdict == Verdict.CONVERGED
    assert out.residual == 0.0
    assert ex.calls == 1
    assert ex.feedbacks[0] is None      # 第一轮无 feedback


def test_feeds_back_failed_checks_then_converges():
    spec = CheckSpec(task="t", checks=[Check(id="a", type="string_contains", params={"s": "结论"})])
    loop = _loop(spec)
    ex = _FakeExecutor(["还在想", "最终结论是42"])  # 第一轮缺"结论",第二轮补上
    out = loop.run("t", ex)
    assert out.verdict == Verdict.CONVERGED
    assert ex.calls == 2
    assert ex.feedbacks[0] is None
    assert ex.feedbacks[1] is not None and "结论" in ex.feedbacks[1]


def test_returns_best_not_last_on_max_steps():
    # 永远不过,但第 2 轮残差最低 → 返回 best 那轮的 result
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="string_contains", params={"s": "X"}, weight=1.0),
        Check(id="b", type="string_contains", params={"s": "Y"}, weight=1.0),
    ])
    loop = _loop(spec, hard_cap=3, K=99)  # K 大 → 不触发 STUCK,跑满到 MAX_STEPS
    ex = _FakeExecutor(["none", "X only", "none again"])  # 轮1 res=2, 轮2 res=1(best), 轮3 res=2
    out = loop.run("t", ex)
    assert out.verdict == Verdict.MAX_STEPS
    assert out.result == "X only"
    assert out.residual == 1.0


def test_feedback_describes_command_ok_failure():
    from my_agent_llms.verify.loop import feedback_from
    spec = CheckSpec(task="t", checks=[
        Check(id="a", type="command_ok", params={"cmd": "pytest"}, is_hard_oracle=True),
    ])
    msg = feedback_from(spec, {"a": False})
    assert msg is not None
    assert "pytest" in msg
    assert "exit 0" in msg
