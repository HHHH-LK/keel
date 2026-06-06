"""跑 pytest 并把结果分成三态(供红门/绿门判定)。"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RunOutcome(Enum):
    PASS = "pass"                # 全部用例通过
    ASSERT_FAIL = "assert_fail"  # 断言失败(行为不对)
    MISSING_IMPL = "missing_impl"  # 目标实现缺失(Import/Module/Name/AttributeError)
    BROKEN = "broken"            # 测试自身坏(SyntaxError/收集错误/没收集到用例)


@dataclass
class RunResult:
    outcome: RunOutcome
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    summary: str = ""   # 喂回 agent 用的简短摘要
    raw: str = ""       # 完整 pytest 输出(调试/反馈用)


_PASS_LINE = re.compile(r"^(?P<id>\S+::\S+)\s+PASSED", re.M)
_FAIL_LINE = re.compile(r"^(?P<id>\S+::\S+)\s+FAILED", re.M)
# 测试文件自身坏:收集阶段的语法/缩进错误
_BROKEN_MARKERS = ("SyntaxError", "IndentationError", "TabError")
# 目标实现缺失:导入/名字类错误
_MISSING_MARKERS = ("ModuleNotFoundError", "ImportError",
                    "NameError", "AttributeError")


def run_pytest(target_dir: str, paths: Optional[List[str]] = None,
               timeout: float = 60.0) -> RunResult:
    """在 target_dir 下跑 pytest。paths=None 跑整目录。异常/超时 → BROKEN。"""
    cmd = [sys.executable, "-m", "pytest", "-v", "--no-header",
           "--color=no", "-p", "no:cacheprovider"]
    if paths:
        cmd.extend(paths)
    try:
        proc = subprocess.run(
            cmd, cwd=target_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(RunOutcome.BROKEN, summary="pytest 超时", raw="timeout")
    except Exception as exc:  # pytest 没装/环境缺 等
        return RunResult(RunOutcome.BROKEN, summary=f"pytest 起不来: {exc}", raw=str(exc))

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    passed = _PASS_LINE.findall(out)
    failed = _FAIL_LINE.findall(out)

    # 收集阶段 0 用例 / 收集错误 → 坏
    if proc.returncode == 5 or ("no tests ran" in out.lower() and not passed and not failed):
        outcome = RunOutcome.BROKEN
    # v1 限制:SyntaxError 一律判 BROKEN(打回出题方)。若是"被测实现文件"自身语法错,
    # 严格说该归实现方;但红门阶段实现通常还没写,绿门对 BROKEN 也会判 STILL_RED 喂回实现方,
    # 故实际影响仅限"改既存带语法错文件的红门"这一窄场景。精确按目标符号归因留作 v2。
    elif any(m in out for m in _BROKEN_MARKERS):
        outcome = RunOutcome.BROKEN
    # rc==0 即无失败;passed 为空(全 skip/xpass)也算通过,不判 BROKEN。
    elif proc.returncode == 0 and not failed:
        outcome = RunOutcome.PASS
    elif any(m in out for m in _MISSING_MARKERS):
        outcome = RunOutcome.MISSING_IMPL
    elif failed:
        outcome = RunOutcome.ASSERT_FAIL
    else:
        # 收集错误(returncode 2/3)等其余情况 → 坏
        outcome = RunOutcome.BROKEN

    return RunResult(
        outcome=outcome, passed=passed, failed=failed,
        summary=_summarize(outcome, passed, failed, out), raw=out)


def _summarize(outcome: RunOutcome, passed: List[str],
               failed: List[str], out: str) -> str:
    if outcome == RunOutcome.PASS:
        return f"全部通过({len(passed)} 个用例)"
    if outcome == RunOutcome.ASSERT_FAIL:
        return f"{len(passed)} 过 / {len(failed)} 没过: {', '.join(failed[:3])}"
    if outcome == RunOutcome.MISSING_IMPL:
        return "目标实现缺失(导入/名字错误)——这是期望的红"
    return "测试自身坏(语法/收集错误)"
