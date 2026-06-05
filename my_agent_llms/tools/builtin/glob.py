"""Glob —— 按 glob 模式找文件(mtime 倒序),锁项目根子树。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation

_MAX_GLOB = 100


class GlobTool(Tool):
    side_effect_free = True

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="Glob",
            description=(
                "按 glob 模式找文件(如 **/*.py、src/**/*.md、*.toml),按最近修改倒序"
                "返回相对路径。用于先定位文件再 Read。只搜当前工作区。"
            ),
        )
        self.ws = workspace

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="pattern", type="string", description="glob 模式,如 **/*.py", required=True),
            ToolParameter(name="path", type="string", description="搜索根(默认工作区根),限工作区内", required=False, default="."),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        pattern = str(parameters.get("pattern") or "").strip()
        if not pattern:
            return "❌ 缺少 pattern 参数"
        path = str(parameters.get("path") or ".").strip() or "."
        try:
            base = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"
        if not base.exists() or not base.is_dir():
            return f"❌ 不是目录: {path}"

        deny = self.ws._deny_dirs
        matched: List[Path] = []
        try:
            it = base.glob(pattern)
        except (ValueError, NotImplementedError) as e:
            return f"❌ 模式错误: {e}"
        for p in it:
            if not p.is_file():
                continue
            if any(part in deny for part in p.relative_to(base).parts):
                continue
            matched.append(p)
        if not matched:
            return "(无匹配)"
        matched.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        rels = [self.ws.relative(p) for p in matched[:_MAX_GLOB]]
        out = "\n".join(rels)
        if len(matched) > _MAX_GLOB:
            out += f"\n… +{len(matched) - _MAX_GLOB} more"
        return out
