"""ReadFile —— 读 sandbox 内文本文件,带行号,支持 offset/limit 分页。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation

DEFAULT_LIMIT = 200


class ReadFile(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="ReadFile",
            description=(
                "读 sandbox 内文本文件,返回带行号的内容。"
                "大文件默认仅显示前 200 行,需要看后续内容请传 offset/limit 分页。"
            ),
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        if not path:
            return "❌ 缺少 path 参数"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not p.exists():
            return f"❌ 文件不存在: {self._safe_rel(p)}。可用 ListDir 查看 sandbox 内文件"
        if p.is_dir():
            return f"❌ {self._safe_rel(p)} 是目录,不是文件。用 ListDir 查看其内容"

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self._safe_rel(p)},本期不支持"

        lines = text.splitlines()
        total = len(lines)

        if total == 0:
            return f"# {self._safe_rel(p)} (空文件)\n"

        try:
            offset = int(parameters.get("offset") or 0)
            limit = int(parameters.get("limit") or DEFAULT_LIMIT)
        except (ValueError, TypeError):
            return "❌ offset/limit 必须为整数"
        if offset < 0:
            offset = 0
        if limit <= 0:
            limit = DEFAULT_LIMIT

        chunk = lines[offset : offset + limit]
        numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))

        header = f"# {self._safe_rel(p)} (共 {total} 行,已显示 {offset + 1}-{offset + len(chunk)})\n"
        return header + numbered

    def _safe_rel(self, p) -> str:
        try:
            return self.ws.relative(p)
        except Exception:
            return str(p)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径", required=True),
            ToolParameter(name="offset", type="integer", description="从第几行开始(0-based)", required=False, default=0),
            ToolParameter(name="limit", type="integer", description="最多读多少行", required=False, default=DEFAULT_LIMIT),
        ]
