"""verify/convergence.py: 三态 + 双上限 + 指纹。喂残差/指纹序列断言 Verdict。"""
from my_agent_llms.verify.convergence import (
    Verdict, ConvergenceJudge, Round, fingerprint,
)


def test_fingerprint_deterministic_and_sensitive():
    traj = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "read_file"}}]}]
    f1 = fingerprint("answer", traj)
    f2 = fingerprint("answer", traj)
    f3 = fingerprint("different", traj)
    assert f1 == f2          # 同输入 → 同指纹
    assert f1 != f3          # result 变 → 指纹变


def test_converged_on_zero_residual():
    j = ConvergenceJudge()
    assert j.judge(0, 0.0, "fp0", []) == Verdict.CONVERGED


def test_oscillating_on_repeat_fingerprint():
    j = ConvergenceJudge()
    history = [Round(residual=2.0, fingerprint="fpA")]
    # 当前残差非 0,且指纹与历史里的 fpA 重现
    assert j.judge(1, 1.0, "fpA", history) == Verdict.OSCILLATING


def test_stuck_when_residual_plateaus_over_K():
    j = ConvergenceJudge(K=2)
    history = [Round(residual=3.0, fingerprint="f0")]
    # 当前 3.0,最近 2 轮 [3.0, 3.0] 首尾未下降 → STUCK
    assert j.judge(1, 3.0, "f1", history) == Verdict.STUCK


def test_not_stuck_when_still_descending():
    j = ConvergenceJudge(K=2)
    history = [Round(residual=5.0, fingerprint="f0")]
    # 最近 2 轮 [5.0, 2.0] 明显下降 → 不 STUCK,继续
    assert j.judge(1, 2.0, "f1", history) == Verdict.CONTINUE


def test_stuck_needs_full_K_window():
    j = ConvergenceJudge(K=3)
    history = [Round(residual=3.0, fingerprint="f0")]
    # 只有 2 轮(含当前),不足 K=3 → 还不判 STUCK
    assert j.judge(1, 3.0, "f1", history) == Verdict.CONTINUE


def test_max_steps_at_hard_cap():
    j = ConvergenceJudge(hard_cap=3, K=2)
    history = [Round(residual=5.0, fingerprint="f0"), Round(residual=4.0, fingerprint="f1")]
    # round_idx=2 == hard_cap-1,残差仍在降(不 STUCK)、指纹不重 → MAX_STEPS
    assert j.judge(2, 3.0, "f2", history) == Verdict.MAX_STEPS


def test_continue_otherwise():
    j = ConvergenceJudge(hard_cap=5, K=2)
    history = [Round(residual=5.0, fingerprint="f0")]
    assert j.judge(1, 3.0, "f1", history) == Verdict.CONTINUE


def test_converged_beats_max_steps():
    # 即使到 hard_cap,residual==0 优先判 CONVERGED
    j = ConvergenceJudge(hard_cap=3, K=2)
    history = [Round(residual=2.0, fingerprint="f0"), Round(residual=1.0, fingerprint="f1")]
    assert j.judge(2, 0.0, "f2", history) == Verdict.CONVERGED
