"""AttachFile —— 把外部文件复制进 sandbox。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class AttachFile(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="AttachFile",
            description=(
                "把外部文件复制进 sandbox,后续 Read/Edit/Write 才能操作。"
                "用户提到任何外部路径时,先用本工具拉进来。"
            ),
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        src = str(parameters.get("source_path") or "").strip()
        if not src:
            return "❌ 缺少 source_path 参数"
        try:
            dst = self.ws.attach(src)
        except FileNotFoundError as e:
            return f"❌ 源文件不存在: {src}"
        except IsADirectoryError as e:
            return f"❌ 源是目录,不是文件: {src}"
        except WorkspaceViolation as e:
            return f"❌ {e}"
        except FileExistsError as e:
            return f"❌ {e}。请先 ExportFile 或改名"
        return f"✅ 已 attach: {self.ws.relative(dst)} (来源: {src})"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="source_path", type="string", description="外部文件的绝对或相对路径", required=True),
        ]
