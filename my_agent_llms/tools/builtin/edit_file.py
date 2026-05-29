"""EditFile —— 精确替换 sandbox 内文件的一段文字。单步式;由 Agent 层在
执行前同步弹审批框(见 Tool.requires_approval / preview_for_approval)。"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def _make_diff(rel_path: str, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=rel_path, tofile=rel_path, n=3)
    return "".join(diff) or "(无文本差异)"


class EditFile(Tool):
    requires_approval = True

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="Edit",
            description=(
                "精确替换 sandbox 内文件的某段文字。传 path + old_string + new_string;"
                "old_string 必须在文件中唯一匹配。框架会在执行前同步弹审批框给用户。"
            ),
        )
        self.ws = workspace

    def _validate(self, parameters: Dict[str, Any]) -> tuple:
        """共享的输入校验。返回 (error_msg, path, old_content, new_content) —
        其中 error_msg 非 None 时其它为 None。"""
        path = str(parameters.get("path") or "").strip()
        old = parameters.get("old_string")
        new = parameters.get("new_string")
        if not path or old is None or new is None:
            return ("❌ 缺少参数 path / old_string / new_string", None, None, None)

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return (f"❌ {e}", None, None, None)

        if not p.exists():
            return (f"❌ 文件不存在: {self.ws.relative(p)}。可用 ListDir 查看 sandbox 内文件",
                    None, None, None)
        if p.is_dir():
            return (f"❌ {self.ws.relative(p)} 是目录", None, None, None)

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return (f"❌ 非 UTF-8 编码: {self.ws.relative(p)},本期不支持",
                    None, None, None)

        count = content.count(old)
        if count == 0:
            return (f"❌ 在 {self.ws.relative(p)} 中找不到 old_string。请先 ReadFile 确认实际内容",
                    None, None, None)
        if count > 1:
            return (f"❌ old_string 在 {self.ws.relative(p)} 匹配 {count} 处。"
                    "请扩大 old_string 的上下文使其唯一",
                    None, None, None)

        new_content = content.replace(old, new, 1)
        return (None, p, content, new_content)

    def run(self, parameters: Dict[str, Any]) -> str:
        err, p, content, new_content = self._validate(parameters)
        if err is not None:
            return err
        if new_content == content:
            return "⚠️ 新内容与原文件相同,无需修改"

        # 原子写: 先写 tmp 再 rename
        tmp_path = p.with_name(f".{p.name}.tmp")
        try:
            tmp_path.write_text(new_content, encoding="utf-8")
            tmp_path.replace(p)
        except OSError as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        return f"✅ 已修改 {self.ws.relative(p)}"

    def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
        err, p, content, new_content = self._validate(parameters)
        if err is not None:
            return err  # 让 UI 看到为什么会失败 —— 比 repr(args) 信息丰富
        return _make_diff(self.ws.relative(p), content, new_content)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径"),
            ToolParameter(name="old_string", type="string",
                          description="要被替换的原文本(必须在文件中唯一)"),
            ToolParameter(name="new_string", type="string", description="替换后的文本"),
        ]
