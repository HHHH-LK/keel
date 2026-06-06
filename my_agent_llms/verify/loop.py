"""编排器:把"跑一轮 + 拿回 result/trajectory"与具体 agent 范式解耦。

同一套 loop 能套在任何满足 Executor 协议的执行者上。验证由本编排器(确定性代码)
强制插入,不靠模型"自觉记得验证"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from my_agent_llms.verify.checkers import CheckContext, CheckerRunner
from my_agent_llms.verify.convergence import ConvergenceJudge, Round, Verdict, fingerprint
from my_agent_llms.verify.residual import residual, effective_count
from my_agent_llms.verify.spec import CheckSpec, SpecGenerator


class Executor(Protocol):
    def tool_names(self) -> List[str]: ...
    def execute(self, task: str, *, feedback: Optional[str]
                ) -> Tuple[str, List[dict]]: ...


@dataclass
class VerifyResult:
    result: str
    residual: float
    verdict: Verdict
    spec: CheckSpec
    passed: Dict[str, bool]


@dataclass
class _Best:
    residual: float
    result: str
    passed: Dict[str, bool]


def feedback_from(spec: CheckSpec, passed: Dict[str, bool]) -> Optional[str]:
    """取没过的 checks 的人话描述,组成 grounded 反思素材。全过返回 None。"""
    failed = [c for c in spec.checks if passed.get(c.id, False) is False]
    if not failed:
        return None
    lines = ["上一轮产出未通过以下验收项,请针对性修订(不要推倒重来,只补差距):"]
    for c in failed:
        lines.append(f"- {_describe_check(c)}")
    return "\n".join(lines)


def _describe_check(c) -> str:
    """把一条 check 渲染成对模型有用的一句话提示(按类型抽最相关字段)。"""
    p = c.params or {}
    t = c.type
    if t == "string_contains":
        return f"产出必须包含: {p.get('s')!r}"
    if t == "string_absent":
        return f"产出不得包含: {p.get('s')!r}"
    if t == "field_equals":
        return f"产物文件 {p.get('path')!r} 的字段 {p.get('key')!r} 应等于 {p.get('value')!r}"
    if t == "command_ok":
        return f"检查命令必须成功(exit 0): {p.get('cmd')!r}"
    if t == "tool_called":
        return f"任务要求必须调用工具: {p.get('tool')!r}"
    if t == "judge":
        return f"需满足评审标准: {p.get('rubric')}"
    if t == "semantic_support":
        return f"需在语义上支持: {p.get('claim')}"
    return f"[{t}] {p}"


class VerifyRetryLoop:
    def __init__(self, *, spec_gen: SpecGenerator, checker_runner: CheckerRunner,
                 judge: ConvergenceJudge):
        self.spec_gen = spec_gen
        self.checker_runner = checker_runner
        self.judge = judge

    def run(self, task: str, executor: Executor) -> VerifyResult:
        spec = self.spec_gen.generate(task, tools=executor.tool_names())  # 循环外,一次
        history: List[Round] = []
        best: Optional[_Best] = None
        feedback: Optional[str] = None

        for r in range(self.judge.hard_cap):
            result, traj = executor.execute(task, feedback=feedback)
            ctx = CheckContext(result=result, trajectory=traj)
            passed = self.checker_runner.run(spec, ctx)
            res = residual(spec, passed)
            if best is None or res < best.residual:   # 严格小于 → 平局保留更早
                best = _Best(residual=res, result=result, passed=passed)
            fp = fingerprint(result, traj)
            verdict = self.judge.judge(r, res, fp, history,
                                       has_effective=effective_count(spec, passed) > 0)
            history.append(Round(residual=res, fingerprint=fp))
            if verdict != Verdict.CONTINUE:
                # verdict = 为什么停;best = 返回哪一轮(全程残差最小那轮,防越修越差)
                return VerifyResult(result=best.result, residual=best.residual,
                                    verdict=verdict, spec=spec, passed=best.passed)
            feedback = feedback_from(spec, passed)

        return VerifyResult(result=best.result, residual=best.residual,
                            verdict=Verdict.MAX_STEPS, spec=spec, passed=best.passed)
