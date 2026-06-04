"""Workspace —— Agent 文件工具的安全边界(就地工作模型)。

所有 file tool 构造时注入同一个 Workspace 实例。Workspace 负责:
- 决定工作区根目录 (显式 root / 默认当前目录)
- resolve(写,严格越界拦截) / resolve_read(读,允许越界) + 黑名单
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_DENY_DIRS = frozenset({".git", ".env", "node_modules", "__pycache__", ".venv"})
DEFAULT_DENY_SUFFIXES = frozenset({".pem", ".key"})


class WorkspaceViolation(Exception):
    """路径越界 / 命中黑名单。Tool 内捕获后转字符串返回给 LLM。"""


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        deny_dirs: Iterable[str] = DEFAULT_DENY_DIRS,
        deny_suffixes: Iterable[str] = DEFAULT_DENY_SUFFIXES,
    ):
        if root is None:
            # 就地工作:默认以当前目录为工作区根(不再自动建空沙箱)
            root_path = Path.cwd()
        else:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                raise FileNotFoundError(f"workspace 根目录不存在: {root_path}")
            if not root_path.is_dir():
                raise NotADirectoryError(f"workspace 根不是目录: {root_path}")

        self.root: Path = root_path.resolve(strict=True)
        self._deny_dirs: frozenset[str] = frozenset(deny_dirs)
        self._deny_suffixes: frozenset[str] = frozenset(deny_suffixes)

    # ── 路径守门 ────────────────────────────────────────────
    def _to_abs(self, user_path: str) -> Path:
        """把 user_path 解析成绝对路径(跟随符号链接,允许尚未存在的路径)。
        不做任何越界/黑名单校验 —— 那由 resolve / resolve_read 各自决定。"""
        up = Path(user_path).expanduser()
        candidate = up if up.is_absolute() else (self.root / up)
        if candidate.is_symlink():
            return candidate.resolve(strict=False)
        if candidate.exists():
            return candidate.resolve(strict=True)
        parent = candidate.parent
        if parent.exists():
            return parent.resolve(strict=True) / candidate.name
        return candidate.resolve()

    def _check_blacklist(self, p: Path) -> None:
        for part in p.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"路径命中黑名单目录: {p}")
        if p.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"文件类型在黑名单: {p.suffix}")

    def resolve(self, user_path: str) -> Path:
        """严格(写用):解析到 sandbox 内,越界 raise,再查黑名单。"""
        p = self._to_abs(user_path)
        try:
            p.relative_to(self.root)
        except ValueError:
            raise WorkspaceViolation(f"路径越界: {p} 不在 workspace {self.root} 内")
        self._check_blacklist(p)
        return p

    def resolve_read(self, user_path: str) -> Path:
        """宽松(读用):允许 CWD 以外,只查黑名单(防把密钥/敏感目录读进上下文)。"""
        p = self._to_abs(user_path)
        self._check_blacklist(p)
        return p

    def relative(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)   # 读越界场景:p 在 root 外,展示绝对路径
