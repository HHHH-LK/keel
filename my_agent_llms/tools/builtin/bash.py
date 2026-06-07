"""Bash —— 在工作区执行 shell 命令。

设计:常规命令(ls/pytest/python/grep…)直接跑;**危险命令**(rm -rf / sudo /
管道到 sh / dd / chmod -R 等)经 `approval_required_for` 动态触发审批,弹框给用户确认。

⚠️ 安全说明:shell=True 执行任意命令,危险判定是【启发式黑名单】,不是安全边界
(可被间接构造的命令绕过)。它降低误伤概率,不能替代沙箱。
"""
from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter

# 危险命令黑名单(命中即要求审批)。保守列常见破坏性/提权/外泄模式。
_DANGER_PATTERNS = [
    r"\brm\s+-[a-z]*[rf]",                  # rm -rf / rm -r / rm -f
    r"\brmdir\b",
    r"\bsudo\b", r"\bsu\b",
    r"\bmkfs\b", r"\bdd\b", r"\bshred\b", r"\btruncate\b",
    r"\bchmod\b", r"\bchown\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bkill(all)?\b", r"\bpkill\b",
    r">\s*/dev/",                           # 写设备
    r"[|]\s*(sudo\s+)?(ba)?sh\b",           # 管道到 shell(curl ... | sh)
    r":\(\)\s*\{",                          # fork bomb
    r"\bgit\s+push\b", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\b",
    r"\b(npm\s+publish|twine\s+upload|uv\s+publish|poetry\s+publish)\b",
]
_DANGER_RE = [re.compile(p) for p in _DANGER_PATTERNS]

_MAX_OUTPUT_LINES = 100      # 回喂模型的输出上限(防 context 爆),展示再由渲染器折叠


def _is_dangerous(command: str) -> bool:
    """命令是否命中危险黑名单(命中 → 需要审批)。"""
    c = (command or "").lower()
    return any(rx.search(c) for rx in _DANGER_RE)


class BashTool(Tool):
    requires_approval = False        # 由 approval_required_for 动态决定(只对危险命令)
    side_effect_free = False         # 有副作用 → 串行执行

    def __init__(self, workspace=None, timeout: float = 120.0):
        super().__init__(
            "Bash",
            "在工作区执行 shell 命令(跑测试/脚本/构建等)。传 command;"
            "危险命令(rm -rf/sudo/管道到 sh 等)会先弹审批框给用户确认。")
        self.workspace = workspace
        self.timeout = timeout

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="command", type="string",
                              description="要执行的 shell 命令", required=True)]

    def preview_for_approval(self, parameters: Dict[str, Any]) -> str:
        return f"$ {parameters.get('command', '')}"

    def approval_required_for(self, parameters: Dict[str, Any]) -> bool:
        return _is_dangerous(str(parameters.get("command", "")))

    def run(self, parameters: Dict[str, Any]) -> str:
        command = str(parameters.get("command", "")).strip()
        if not command:
            return "❌ Bash 缺少 command 参数"
        cwd = getattr(self.workspace, "root", None)
        try:
            proc = subprocess.run(
                command, shell=True,
                cwd=str(cwd) if cwd else None,
                capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return f"⏱️ 命令超时(>{self.timeout}s),已放弃等待"
        except Exception as e:
            return f"❌ 执行失败: {e}"

        parts: List[str] = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip("\n"))
        if proc.stderr:
            parts.append("[stderr] " + proc.stderr.rstrip("\n"))
        body = "\n".join(parts) if parts else "(无输出)"

        lines = body.split("\n")
        if len(lines) > _MAX_OUTPUT_LINES:
            hidden = len(lines) - _MAX_OUTPUT_LINES
            body = "\n".join(lines[:_MAX_OUTPUT_LINES]) + f"\n… (+{hidden} 行已截断)"

        head = "✅" if proc.returncode == 0 else f"❌ exit={proc.returncode}"
        return f"{head}\n{body}"
