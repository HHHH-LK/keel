from my_agent_llms.tdd.runner import RunResult, RunOutcome
from my_agent_llms.tdd.gates import (
    red_gate, green_gate, RedVerdict, GreenVerdict)


def _r(outcome, **kw):
    return RunResult(outcome=outcome, **kw)


def test_red_gate_pass_means_fake_test():
    assert red_gate(_r(RunOutcome.PASS)) == RedVerdict.FAKE_TEST


def test_red_gate_assert_fail_proceeds():
    assert red_gate(_r(RunOutcome.ASSERT_FAIL)) == RedVerdict.PROCEED


def test_red_gate_missing_impl_proceeds():
    # 期望红:目标没写导致的导入错,放行去写实现
    assert red_gate(_r(RunOutcome.MISSING_IMPL)) == RedVerdict.PROCEED


def test_red_gate_broken_bounces():
    assert red_gate(_r(RunOutcome.BROKEN)) == RedVerdict.BROKEN


def test_green_gate_pass_converges():
    assert green_gate(_r(RunOutcome.PASS)) == GreenVerdict.CONVERGED


def test_green_gate_nonpass_still_red():
    assert green_gate(_r(RunOutcome.ASSERT_FAIL)) == GreenVerdict.STILL_RED
    assert green_gate(_r(RunOutcome.BROKEN)) == GreenVerdict.STILL_RED
