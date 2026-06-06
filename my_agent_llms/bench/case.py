"""离线用例:声明式 JSON(任务 + 初始文件 + 真 oracle 检查点)。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BenchCase:
    id: str
    task: str
    setup_files: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)


def load_cases(dir_path) -> List[BenchCase]:
    cases = []
    for p in sorted(Path(dir_path).glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        checks = list(obj.get("checks") or [])
        for i, c in enumerate(checks):
            c.setdefault("id", f"c{i}")          # 补 id 供 Check(**c) / 查表
        cases.append(BenchCase(id=str(obj["id"]), task=str(obj["task"]),
                               setup_files=dict(obj.get("setup_files") or {}),
                               checks=checks))
    return cases
