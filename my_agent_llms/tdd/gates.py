"""红门 / 绿门:把 runner 三态映射成决策 + 反馈文案。"""
from __future__ import annotations

from enum import Enum

from my_agent_llms.tdd.runner import RunResult, RunOutcome


class RedVerdict(Enum):
    PROCEED = "proceed"      # 期望的红 → 去写实现
    FAKE_TEST = "fake_test"  # 没实现就过 → 打回出题方
    BROKEN = "broken"        # 测试自身坏 → 打回出题方


class GreenVerdict(Enum):
    CONVERGED = "converged"  # 全绿 → 收敛
    STILL_RED = "still_red"  # 还红 → 喂回实现方


def red_gate(result: RunResult) -> RedVerdict:
    if result.outcome == RunOutcome.PASS:
        return RedVerdict.FAKE_TEST
    if result.outcome == RunOutcome.BROKEN:
        return RedVerdict.BROKEN
    return RedVerdict.PROCEED  # ASSERT_FAIL / MISSING_IMPL = 期望红


def green_gate(result: RunResult) -> GreenVerdict:
    if result.outcome == RunOutcome.PASS:
        return GreenVerdict.CONVERGED
    return GreenVerdict.STILL_RED


def author_feedback(verdict: RedVerdict, result: RunResult) -> str:
    """打回出题方时喂给 test-author 的话。"""
    if verdict == RedVerdict.FAKE_TEST:
        return ("你写的测试在实现缺失时就通过了,等于没测到目标行为。"
                "重写,让它在实现缺失/错误时失败。")
    return (f"你写的测试自身坏了({result.summary}),pytest 收集就报错。"
            "修正测试代码本身(语法/导入),确保是有效测试。")


def impl_feedback(result: RunResult) -> str:
    """喂回实现方的话。"""
    return f"测试还没全过:{result.summary}。修改实现让测试通过,不要改测试文件。"
