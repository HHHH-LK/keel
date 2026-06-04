"""会话内审批授权台账 —— 纯内存,不落盘,重启清空。

按 (工具名, 路径前缀) 记忆 ALLOW_ALWAYS。路径前缀 = 目标 path 所在目录(相对);
后续同工具、目标路径以该前缀开头者直接放行。取不到 path 的工具按整工具记忆。
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Set, Tuple

from my_agent_llms.cli.permission import PermissionDecision

PromptFn = Callable[[str, Dict[str, Any], str], PermissionDecision]


class PermissionGrants:
    def __init__(self) -> None:
        # (name, prefix) — prefix == "" 表示整工具授权
        self._grants: Set[Tuple[str, str]] = set()

    @staticmethod
    def _prefix(args: Dict[str, Any]) -> str:
        path = str(args.get("path") or "").strip()
        if not path:
            return ""
        return str(PurePosixPath(path).parent)   # "src/a.py" → "src"; "a.py" → "."

    def is_granted(self, name: str, args: Dict[str, Any]) -> bool:
        prefix = self._prefix(args)
        for gn, gp in self._grants:
            if gn != name:
                continue
            if gp == "":
                return True
            if prefix == gp or prefix.startswith(gp + "/"):
                return True
        return False

    def grant(self, name: str, args: Dict[str, Any]) -> None:
        self._grants.add((name, self._prefix(args)))


def decide(
    grants: PermissionGrants,
    prompt_fn: PromptFn,
    name: str,
    args: Dict[str, Any],
    preview: str,
) -> bool:
    """返回最终是否允许(bool,给 agent)。命中授权直接放行;否则弹三态框,
    ALLOW_ALWAYS 记进 grants。"""
    if grants.is_granted(name, args):
        return True
    decision = prompt_fn(name, args, preview)
    if decision is PermissionDecision.DENY:
        return False
    if decision is PermissionDecision.ALLOW_ALWAYS:
        grants.grant(name, args)
    return True
