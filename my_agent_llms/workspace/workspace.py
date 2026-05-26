"""Workspace —— Agent 文件工具的安全边界。

所有 file tool 构造时注入同一个 Workspace 实例。Workspace 负责:
- 决定 sandbox 根目录 (显式 root / 自动创建)
- 路径 resolve + 越界拦截 + 黑名单
- 维护 MANIFEST.json (sandbox 文件 → 原始源路径)
- attach / origin_of
"""
from __future__ import annotations

import datetime as _dt
import json
import secrets
from pathlib import Path
from typing import Iterable

DEFAULT_DENY_DIRS = frozenset({".git", ".env", "node_modules", "__pycache__", ".venv"})
DEFAULT_DENY_SUFFIXES = frozenset({".pem", ".key"})


class WorkspaceViolation(Exception):
    """路径越界 / 命中黑名单。Tool 内捕获后转字符串返回给 LLM。"""


def _auto_sandbox_name() -> str:
    """YYYYMMDD-HHMMSS-<6 位 hex>"""
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        deny_dirs: Iterable[str] = DEFAULT_DENY_DIRS,
        deny_suffixes: Iterable[str] = DEFAULT_DENY_SUFFIXES,
    ):
        if root is None:
            parent = Path.home() / ".my_agent_llms" / "workspaces"
            parent.mkdir(parents=True, exist_ok=True)
            root_path = parent / _auto_sandbox_name()
            root_path.mkdir()
        else:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                raise FileNotFoundError(f"workspace 根目录不存在: {root_path}")
            if not root_path.is_dir():
                raise NotADirectoryError(f"workspace 根不是目录: {root_path}")

        self.root: Path = root_path.resolve(strict=True)
        self.manifest_path: Path = self.root / "MANIFEST.json"
        self._deny_dirs: frozenset[str] = frozenset(deny_dirs)
        self._deny_suffixes: frozenset[str] = frozenset(deny_suffixes)
