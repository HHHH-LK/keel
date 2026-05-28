"""AttachDir —— 把外部目录递归复制进 sandbox。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation


_PREVIEW_LIMIT = 30


class AttachDir(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="AttachDir",
            description=(
                "把外部目录递归复制进 sandbox,后续 ListDir / ReadFile 才能操作。"
                "默认排除 .git/.venv/node_modules/__pycache__/dist/build/二进制等,"
                "单文件 1MB 上限、总体 50MB 上限。"
                "用户提到外部目录路径时,先用本工具一次拉进来,再列/读。"
            ),
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        src = str(parameters.get("source_path") or "").strip()
        if not src:
            return "❌ 缺少 source_path 参数"
        try:
            result = self.ws.attach_dir(src)
        except FileNotFoundError:
            return f"❌ 源目录不存在: {src}"
        except NotADirectoryError:
            return f"❌ 源是文件不是目录: {src} (请改用 AttachFile)"
        except WorkspaceViolation as e:
            return f"❌ {e}"
        except FileExistsError as e:
            return f"❌ {e}。请先 ExportFile 或改名"

        sandbox_dir = result["sandbox_dir"]
        lines = [
            f"✅ 已 attach 目录: {sandbox_dir}/ (来源: {result['source']})",
            f"   共拷贝 {result['copied_count']} 个文件 / {result['total_bytes'] // 1024} KB",
        ]
        if result["skipped_too_large"]:
            preview = ", ".join(result["skipped_too_large"][:5])
            more = "" if len(result["skipped_too_large"]) <= 5 else f" ...等 {len(result['skipped_too_large'])} 个"
            lines.append(f"   跳过超过 1MB 的文件: {preview}{more}")
        if result["skipped_ignored_count"]:
            lines.append(f"   按规则忽略 {result['skipped_ignored_count']} 个 (二进制/缓存/构建产物)")
        if result["truncated"]:
            lines.append("   ⚠️ 已达 50MB 总量上限,未全部拷贝。建议挑子目录单独 attach")

        preview_files = result["copied_files"][:_PREVIEW_LIMIT]
        if preview_files:
            lines.append("")
            lines.append(f"📂 {sandbox_dir}/ 内容预览:")
            for f in preview_files:
                lines.append(f"   {f}")
            remaining = result["copied_count"] - len(preview_files)
            if remaining > 0:
                lines.append(f"   ... 还有 {remaining} 个文件 (用 ListDir 查全集)")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="source_path",
                type="string",
                description="外部目录的绝对或相对路径",
                required=True,
            ),
        ]
