"""判分:复用在线 verify 的 checker。residual≈0 且有有效 check → pass。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from my_agent_llms.verify.spec import Check, CheckSpec
from my_agent_llms.verify.checkers import CheckContext, CheckerRunner
from my_agent_llms.verify.residual import residual, effective_count


@dataclass
class RunResultLike:
    answer: str
    trajectory: list
    workspace_root: Optional[str]


@dataclass
class Score:
    case_id: str
    passed: bool
    residual: float
    failed: List[str]


class _WS:
    """最小 workspace:给 checker 的 field_equals/command_ok 提供 root/resolve_read。"""

    def __init__(self, root):
        self.root = root

    def resolve_read(self, rel):
        return Path(self.root) / rel


def score(case, run_result) -> Score:
    spec = CheckSpec(task=case.task, checks=[Check(**c) for c in case.checks])
    ws = _WS(run_result.workspace_root) if run_result.workspace_root else None
    ctx = CheckContext(result=run_result.answer,
                       trajectory=run_result.trajectory or [], workspace=ws)
    passed = CheckerRunner().run(spec, ctx)
    res = residual(spec, passed)
    eff = effective_count(spec, passed)
    ok = res <= 1e-9 and eff > 0
    failed = [c["type"] for c in case.checks if passed.get(c["id"]) is False]
    return Score(case.id, ok, res, failed)
