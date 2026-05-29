"""ExportFile —— 把 sandbox 内文件写回外部真实路径。单步式,
审批由 Agent 主循环的 on_permission_request 回调统一处理。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.edit_file import _make_diff
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class ExportFile(Tool):
    requires_approval = True

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="ExportFile",
            description=(
                "把 sandbox 内文件写回外部真实路径。传 sandbox_path "
                "(sandbox 内新建的文件必传 dest_path)。框架会在执行前同步弹审批框。"
            ),
        )
        self.ws = workspace

    def _resolve(self, parameters: Dict[str, Any]) -> tuple:
        """返回 (error, sandbox_path_obj, dest_path_obj, new_content)。"""
        sb_path = str(parameters.get("sandbox_path") or "").strip()
        dest_arg = parameters.get("dest_path")
        if not sb_path:
            return ("❌ 缺少参数 sandbox_path", None, None, None)

        try:
            sb = self.ws.resolve(sb_path)
        except WorkspaceViolation as e:
            return (f"❌ {e}", None, None, None)
        if not sb.exists():
            return (f"❌ sandbox 文件不存在: {self.ws.relative(sb)}", None, None, None)
        if sb.is_dir():
            return (f"❌ {self.ws.relative(sb)} 是目录", None, None, None)

        if dest_arg:
            dest = Path(str(dest_arg)).expanduser().resolve()
        else:
            origin = self.ws.origin_of(sb)
            if origin is None:
                return (f"❌ {self.ws.relative(sb)} 是 sandbox 内新建文件,"
                        "MANIFEST 中无对应源路径。请显式提供 dest_path",
                        None, None, None)
            dest = origin

        try:
            self.ws.check_external_path(dest)
        except WorkspaceViolation as e:
            return (f"❌ {e}", None, None, None)

        try:
            new_content = sb.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return (f"❌ 非 UTF-8 编码: {self.ws.relative(sb)},本期不支持",
                    None, None, None)

        if dest.exists() and dest.is_dir():
            return (f"❌ dest_path 是目录,不是文件: {dest}", None, None, None)

        return (None, sb, dest, new_content)

    def run(self, parameters: Dict[str, Any]) -> str:
        err, sb, dest, new_content = self._resolve(parameters)
        if err:
            return err

        if dest.exists():
            try:
                old = dest.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 原目标文件非 UTF-8: {dest},本期不支持覆盖"
            if old == new_content:
                return "⚠️ sandbox 内容与原目标完全一致,无需导出"

        tmp = dest.with_name(f".{dest.name}.tmp")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(new_content, encoding="utf-8")
            tmp.replace(dest)
        except OSError as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        return f"✅ 已写回 {dest}"

    def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
        err, sb, dest, new_content = self._resolve(parameters)
        if err:
            return err
        if dest.exists():
            try:
                old = dest.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"(目标非 UTF-8: {dest},覆盖时无法 diff)"
            if old == new_content:
                return "(sandbox 内容与原目标完全一致)"
            return _make_diff(str(dest), old, new_content)
        # 新建目标文件:展示完整待写入内容(全 + 行,Panel 里高亮绿色)
        return _make_diff(str(dest), "", new_content)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="sandbox_path", type="string",
                          description="sandbox 内的文件路径"),
            ToolParameter(name="dest_path", type="string",
                          description="外部目标路径(新建文件必传)",
                          required=False),
        ]
