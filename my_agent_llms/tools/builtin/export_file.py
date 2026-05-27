"""ExportFile —— 把 sandbox 内文件写回外部真实路径,两步确认。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.edit_file import _make_diff, _sha256
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.workspace import (
    DEFAULT_DENY_DIRS,
    DEFAULT_DENY_SUFFIXES,
    Workspace,
    WorkspaceViolation,
)


def _check_dest_deny(dest: Path) -> None:
    """对外部 dest 路径做黑名单校验(与 Workspace.resolve 内的规则一致)。"""
    for part in dest.parts:
        if part in DEFAULT_DENY_DIRS:
            raise WorkspaceViolation(f"导出目标命中黑名单目录: {dest}")
    if dest.suffix in DEFAULT_DENY_SUFFIXES:
        raise WorkspaceViolation(f"导出目标文件类型在黑名单: {dest.suffix}")


class ExportFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="ExportFile",
            description=(
                "把 sandbox 内文件写回外部真实路径。两步确认: "
                "第一次传 sandbox_path (+ dest_path 若为新建文件),返回 pending_id 和 diff vs 原文件; "
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
        sb_path = str(parameters.get("sandbox_path") or "").strip()
        dest_arg = parameters.get("dest_path")
        if not sb_path:
            return "❌ 缺少参数 sandbox_path"

        try:
            sb = self.ws.resolve(sb_path)
        except WorkspaceViolation as e:
            return f"❌ {e}"
        if not sb.exists():
            return f"❌ sandbox 文件不存在: {self.ws.relative(sb)}"
        if sb.is_dir():
            return f"❌ {self.ws.relative(sb)} 是目录"

        # 确定目标
        if dest_arg:
            dest = Path(str(dest_arg)).expanduser().resolve()
        else:
            origin = self.ws.origin_of(sb)
            if origin is None:
                return (
                    f"❌ {self.ws.relative(sb)} 是 sandbox 内新建文件,"
                    "MANIFEST 中无对应源路径。请显式提供 dest_path"
                )
            dest = origin

        try:
            _check_dest_deny(dest)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        try:
            new_content = sb.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self.ws.relative(sb)},本期不支持"

        if dest.exists():
            if dest.is_dir():
                return f"❌ dest_path 是目录,不是文件: {dest}"
            try:
                old = dest.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 原目标文件非 UTF-8: {dest},本期不支持覆盖"
            if old == new_content:
                return "⚠️ sandbox 内容与原目标完全一致,无需导出"
            diff = _make_diff(str(dest), old, new_content)
            source_hash = _sha256(old)
        else:
            diff = f"(新建文件 {dest},{len(new_content.encode('utf-8'))} 字节)"
            source_hash = None

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="export",
            path=dest,
            new_content=new_content,
            diff_preview=diff,
            source_hash=source_hash,
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"即将把 sandbox {self.ws.relative(sb)} 写回 {dest}:\n"
            f"{diff}\n"
            f"请用户回复确认后,再次调用 ExportFile,传入 pending_id={pid}, action=apply"
        )

    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"
        if action == "cancel":
            return f"✅ 已取消 pending {pid}" if self.store.discard(pid) \
                else f"❌ pending_id {pid} 不存在或已过期"

        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)"

        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 原目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return "❌ 原文件 hash 变化,导出会覆盖外部修改。请重新发起"

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

        return f"✅ 已写回 {pe.path}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="sandbox_path", type="string", description="sandbox 内的文件路径", required=False),
            ToolParameter(name="dest_path", type="string", description="外部目标路径(新建文件必传)", required=False),
            ToolParameter(name="pending_id", type="string", description="提案返回的 id", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel", required=False),
        ]
