"""隔离跑一个 case:临时 workspace + 写 setup_files + 跑 agent + 收集结果。"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List


@dataclass
class RunResult:
    case_id: str
    answer: str
    trajectory: list = field(default_factory=list)
    workspace_root: str = ""


def run_case(case, agent_factory: Callable) -> RunResult:
    ws_root = tempfile.mkdtemp(prefix=f"bench_{case.id}_")
    for rel, content in (case.setup_files or {}).items():
        p = Path(ws_root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")
    agent = agent_factory(ws_root)
    traj: List = []
    try:
        answer = agent.run(case.task,
                           on_tool_call=lambda n, a: traj.append(
                               {"role": "assistant",
                                "tool_calls": [{"function": {"name": n}}]}))
    except TypeError:                       # mock agent 不收回调
        answer = agent.run(case.task)
    return RunResult(case_id=case.id, answer=str(answer),
                     trajectory=traj, workspace_root=ws_root)
