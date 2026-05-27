"""WriteFile —— 写整文件(覆盖或新建),两步确认。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.tools.builtin.edit_file import _make_diff, _sha256
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class WriteFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="WriteFile",
            description=(
                "写整个文件内容到 sandbox 路径(覆盖已有或新建)。两步确认: "
                "第一次传 path + content,返回 pending_id 和 diff(新建文件无 diff); "
                "用户确认后再传 pending_id + action=apply。"
            ),
        )
        self.ws = workspace
        self.store = store

    def run(self, parameters: Dict[str, Any]) -> str:
        pid = parameters.get("pending_id")
        if pid:
            return self._handle_action(str(pid), str(parameters.get("action") or ""))
        return self._handle_propose(parameters)

    def _handle_propose(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        content = parameters.get("content")
        if not path or content is None:
            return "❌ 缺少参数 path / content"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if p.exists() and p.is_dir():
            return f"❌ {self.ws.relative(p)} 是目录"

        new_content = str(content)
        rel = self.ws.relative(p)

        if p.exists():
            try:
                old = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 非 UTF-8 编码: {rel},本期不支持覆盖"
            if old == new_content:
                return "⚠️ 新内容与原文件相同,无需修改"
            diff = _make_diff(rel, old, new_content)
            source_hash = _sha256(old)
        else:
            diff = f"(新建文件 {rel},{len(new_content.encode('utf-8'))} 字节)"
            source_hash = None

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="write",
            path=p,
            new_content=new_content,
            diff_preview=diff,
            source_hash=source_hash,
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"{'即将覆盖' if source_hash else '即将新建'} {rel}:\n"
            f"{diff}\n"
            f"请用户回复确认后,再次调用 WriteFile,传入 pending_id={pid}, action=apply"
        )

    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"
        if action == "cancel":
            return f"✅ 已取消 pending {pid},文件未改动" if self.store.discard(pid) \
                else f"❌ pending_id {pid} 不存在或已过期"

        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)"

        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return "❌ 文件在确认期间被外部修改,pending 已失效。请重新发起"

        tmp = pe.path.with_name(f".{pe.path.name}.tmp")
        try:
            pe.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(pe.new_content, encoding="utf-8")
            tmp.replace(pe.path)
        except OSError as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        action_word = "已写入" if pe.source_hash is None else "已覆盖"
        return f"✅ {action_word} {self.ws.relative(pe.path)}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径(提案模式)", required=False),
            ToolParameter(name="content", type="string", description="完整新文件内容(提案模式)", required=False),
            ToolParameter(name="pending_id", type="string", description="提案返回的 id(执行模式)", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel", required=False),
        ]
