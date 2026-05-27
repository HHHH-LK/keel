"""EditFile —— 精确替换,两步确认。

提案模式: path + old_string + new_string → 校验唯一匹配 → 生成 diff → 存 pending
执行模式: pending_id + action(apply/cancel)        → 校验 hash → 落盘 / 丢弃
"""
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_diff(rel_path: str, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=rel_path, tofile=rel_path, n=3)
    return "".join(diff) or "(无文本差异)"


class EditFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="EditFile",
            description=(
                "精确替换 sandbox 内文件的某段文字。两步确认: "
                "第一次传 path + old_string + new_string,返回 pending_id 和 diff; "
                "用户在对话中明确确认后,再传 pending_id + action=apply 落盘。"
            ),
        )
        self.ws = workspace
        self.store = store

    def run(self, parameters: Dict[str, Any]) -> str:
        pid = parameters.get("pending_id")
        if pid:
            return self._handle_action(str(pid), str(parameters.get("action") or ""))
        return self._handle_propose(parameters)

    # ── 提案模式 ────────────────────────────────────────────
    def _handle_propose(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        old = parameters.get("old_string")
        new = parameters.get("new_string")
        if not path or old is None or new is None:
            return "❌ 缺少参数 path / old_string / new_string"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not p.exists():
            return f"❌ 文件不存在: {self.ws.relative(p)}。可用 ListDir 查看 sandbox 内文件"
        if p.is_dir():
            return f"❌ {self.ws.relative(p)} 是目录"

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self.ws.relative(p)},本期不支持"

        count = content.count(old)
        if count == 0:
            return f"❌ 在 {self.ws.relative(p)} 中找不到 old_string。请先 ReadFile 确认实际内容"
        if count > 1:
            return (
                f"❌ old_string 在 {self.ws.relative(p)} 匹配 {count} 处。"
                "请扩大 old_string 的上下文使其唯一"
            )

        new_content = content.replace(old, new, 1)
        if new_content == content:
            return "⚠️ 新内容与原文件相同,无需修改"

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="edit",
            path=p,
            new_content=new_content,
            diff_preview=_make_diff(self.ws.relative(p), content, new_content),
            source_hash=_sha256(content),
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"即将修改 {self.ws.relative(p)}:\n"
            f"{pe.diff_preview}\n"
            f"请用户回复确认后,再次调用 EditFile,传入 pending_id={pid}, action=apply (或 action=cancel 丢弃)"
        )

    # ── 执行模式 ────────────────────────────────────────────
    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"

        if action == "cancel":
            if self.store.discard(pid):
                return f"✅ 已取消 pending {pid},文件未改动"
            return f"❌ pending_id {pid} 不存在或已过期"

        # action == apply
        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)。请重新发起编辑"

        # hash 校验
        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return (
                    f"❌ 文件在确认期间被外部修改,pending 已失效。"
                    f"请重新读取并发起编辑"
                )

        # 原子写: 先写 tmp 再 rename
        tmp_path = pe.path.with_name(f".{pe.path.name}.tmp")
        try:
            tmp_path.write_text(pe.new_content, encoding="utf-8")
            tmp_path.replace(pe.path)
        except OSError as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        return f"✅ 已修改 {self.ws.relative(pe.path)}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径(提案模式)", required=False),
            ToolParameter(name="old_string", type="string", description="要被替换的原文本(必须在文件中唯一)", required=False),
            ToolParameter(name="new_string", type="string", description="替换后的文本", required=False),
            ToolParameter(name="pending_id", type="string", description="提案模式返回的 id(执行模式用)", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel(执行模式用)", required=False),
        ]
