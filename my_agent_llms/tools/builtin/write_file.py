"""WriteFile —— 写整个文件到 sandbox 路径(覆盖或新建)。单步式,
审批由 Agent 主循环的 on_permission_request 回调统一处理。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.edit_file import _make_diff
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class WriteFile(Tool):
    requires_approval = True

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="WriteFile",
            description=(
                "写整个文件内容到 sandbox 路径(覆盖已有或新建)。传 path + content;"
                "框架会在执行前同步弹审批框给用户。"
            ),
        )
        self.ws = workspace

    def _resolve(self, parameters: Dict[str, Any]) -> tuple:
        """返回 (error, path, new_content)。"""
        path = str(parameters.get("path") or "").strip()
        content = parameters.get("content")
        if not path or content is None:
            return ("❌ 缺少参数 path / content", None, None)
        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return (f"❌ {e}", None, None)
        if p.exists() and p.is_dir():
            return (f"❌ {self.ws.relative(p)} 是目录", None, None)
        return (None, p, str(content))

    def run(self, parameters: Dict[str, Any]) -> str:
        err, p, new_content = self._resolve(parameters)
        if err:
            return err

        existed = p.exists()
        if existed:
            try:
                old = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 非 UTF-8 编码: {self.ws.relative(p)},本期不支持覆盖"
            if old == new_content:
                return "⚠️ 新内容与原文件相同,无需修改"

        tmp = p.with_name(f".{p.name}.tmp")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(new_content, encoding="utf-8")
            tmp.replace(p)
        except OSError as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        verb = "已覆盖" if existed else "已写入"
        return f"✅ {verb} {self.ws.relative(p)}"

    def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
        err, p, new_content = self._resolve(parameters)
        if err:
            return err
        rel = self.ws.relative(p)
        if p.exists():
            try:
                old = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"(目标非 UTF-8: {rel},覆盖时无法 diff)"
            if old == new_content:
                return "(新内容与原文件相同)"
            return _make_diff(rel, old, new_content)
        return f"(新建文件 {rel},{len(new_content.encode('utf-8'))} 字节)"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径"),
            ToolParameter(name="content", type="string", description="完整新文件内容"),
        ]
