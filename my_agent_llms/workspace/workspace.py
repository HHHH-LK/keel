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

    # ── 路径守门 ────────────────────────────────────────────
    def resolve(self, user_path: str) -> Path:
        """把 user_path 解析到 sandbox 内绝对路径。
        - 相对路径基于 self.root
        - 绝对路径直接用,但必须落在 self.root 下
        - 跟随符号链接,跟随后仍要在 self.root 下
        - 命中 deny_dirs / deny_suffixes → raise
        - 允许尚未存在的路径 (WriteFile 要建新文件)
        """
        up = Path(user_path).expanduser()
        candidate = up if up.is_absolute() else (self.root / up)

        # 必须先看是不是符号链接 —— is_symlink() 对悬空链接也有效,
        # 否则悬空链接的 exists() 返回 False 会落到拼路径分支,LLM 一旦
        # 写入,OS 仍会跟随链接到 sandbox 外,构成写入逃逸。
        if candidate.is_symlink():
            p = candidate.resolve(strict=False)
        elif candidate.exists():
            p = candidate.resolve(strict=True)
        else:
            # 新建文件:父目录存在就解析父目录,再拼回文件名
            parent = candidate.parent
            if parent.exists():
                p = parent.resolve(strict=True) / candidate.name
            else:
                # 父也不存在 —— resolve(strict=False) 尽力而为
                p = candidate.resolve()

        try:
            p.relative_to(self.root)
        except ValueError:
            raise WorkspaceViolation(f"路径越界: {p} 不在 workspace {self.root} 内")

        for part in p.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"路径命中黑名单目录: {p}")
        if p.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"文件类型在黑名单: {p.suffix}")
        return p

    def relative(self, p: Path) -> str:
        return str(p.relative_to(self.root))
