"""ListDir —— 列工作区文件,带大小、mtime。"""
from __future__ import annotations

import datetime as _dt
import fnmatch
from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class ListDir(Tool):
    side_effect_free = True  # 纯读目录/stat,无写 → 可并行

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="LS",
            description="列工作区文件,默认递归 2 层。",
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip() or "."
        pattern = str(parameters.get("pattern") or "*")

        raw_depth = parameters.get("max_depth")
        try:
            max_depth = int(raw_depth) if raw_depth is not None else 2
        except (ValueError, TypeError):
            return "❌ max_depth 必须为整数"

        try:
            base = self.ws.resolve_read(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not base.exists() or not base.is_dir():
            return f"❌ {path} 不是目录"

        lines: List[str] = []
        for p in self._walk(base, max_depth):
            rel = self.ws.relative(p)
            if not fnmatch.fnmatch(p.name, pattern):
                continue
            size = p.stat().st_size
            mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"{rel}\t{size}\t{mtime}")

        if not lines:
            return "(空)"
        return "\n".join(lines)

    def _walk(self, base: Path, max_depth: int):
        base_depth = len(base.parts)
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if len(p.parts) - base_depth > max_depth:
                continue
            yield p

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="起点(默认工作区根)", required=False, default="."),
            ToolParameter(name="pattern", type="string", description="glob 模式,如 *.md", required=False, default="*"),
            ToolParameter(name="max_depth", type="integer", description="最大递归深度", required=False, default=2),
        ]
