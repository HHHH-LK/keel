"""四阶段调度:classify→出题→红门→实现→绿门。失败优雅降级,绝不假装成功。"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List

from my_agent_llms.tdd.classify import classify as _classify
from my_agent_llms.tdd.test_author import author_tests as _author_tests
from my_agent_llms.tdd.runner import run_pytest as _run_pytest
from my_agent_llms.tdd.gates import (
    red_gate, green_gate, RedVerdict, GreenVerdict, author_feedback, impl_feedback)

logger = logging.getLogger(__name__)


@dataclass
class TddResult:
    success: bool
    message: str       # 给用户的最终回复
    degraded: bool     # True = 没走成 TDD,调用方应改走老路


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_tdd(*, llm, workspace, task: str,
           implement_fn: Callable[[str, List[str], str], None],
           classify_fn=_classify, author_fn=_author_tests, runner_fn=_run_pytest,
           user_override=None, author_budget: int = 2, impl_budget: int = 3) -> TddResult:
    """implement_fn(task, test_paths, feedback) 由主 agent 提供,跑工具循环写实现。"""
    decision = classify_fn(llm, task, user_override=user_override)
    if not decision.use_tdd:
        return TddResult(success=False, message=f"不走 TDD: {decision.reason}", degraded=True)

    feedback = ""
    for _ in range(author_budget):
        # ── 阶段1:出题 ──
        author = author_fn(llm, task, feedback=feedback)
        if not author.tests:
            feedback = "上次没产出任何测试文件,请输出有效的测试 JSON。"
            continue
        # 写盘 + 记哈希(契约锁定)。resolve() 强制工作区边界,越界抛 WorkspaceViolation。
        test_paths: List[str] = []
        hashes: Dict[str, str] = {}
        for t in author.tests:
            p = workspace.resolve(t.relpath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t.content, encoding="utf-8")
            test_paths.append(t.relpath)
            hashes[t.relpath] = _digest(t.content)

        # ── 阶段2:红门(硬)── 只跑一次 pytest,结果同时用于判定与反馈
        red_result = runner_fn(str(workspace.root), test_paths)
        red = red_gate(red_result)
        if red != RedVerdict.PROCEED:
            feedback = author_feedback(red, red_result)
            continue

        # ── 阶段3+4:实现 → 绿门(impl_budget 轮)──
        impl_fb = ""
        last = None
        for _ in range(impl_budget):
            implement_fn(task, test_paths, impl_fb)
            # 防作弊:实现方不得改测试文件
            if _tests_tampered(workspace, hashes):
                return TddResult(success=False,
                                 message="实现阶段检测到测试文件被篡改,已拒绝。", degraded=False)
            last = runner_fn(str(workspace.root), test_paths)
            if green_gate(last) == GreenVerdict.CONVERGED:
                return TddResult(success=True,
                                 message=f"TDD 完成:{last.summary}。测试与实现已留在工作区。",
                                 degraded=False)
            impl_fb = impl_feedback(last)
        # impl 预算用尽 → 如实告知
        if last is None:
            detail = "无结果"
        elif last.summary:
            detail = last.summary
        elif last.failed:
            detail = f"{len(last.failed)} 个用例没过: {', '.join(last.failed[:3])}"
        else:
            detail = f"outcome={last.outcome.value}"
        return TddResult(success=False,
                         message=f"实现没能全部转绿:{detail}。测试已留在工作区,可继续修。",
                         degraded=False)

    # author 预算用尽 → 降级
    return TddResult(success=False,
                     message="test-author 多次写不出有效测试,已降级为普通模式。", degraded=True)


def _tests_tampered(workspace, hashes: Dict[str, str]) -> bool:
    for relpath, h in hashes.items():
        try:
            content = workspace.resolve_read(relpath).read_text(encoding="utf-8")
            if _digest(content) != h:
                return True
        except OSError:
            return True
    return False
