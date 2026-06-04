"""Workspace —— Agent 文件工具的安全边界。

所有 file tool 构造时注入同一个 Workspace 实例。Workspace 负责:
- 决定 sandbox 根目录 (显式 root / 自动创建)
- 路径 resolve + 越界拦截 + 黑名单
- 维护 MANIFEST.json (sandbox 文件 → 原始源路径)
- attach / origin_of
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DENY_DIRS = frozenset({".git", ".env", "node_modules", "__pycache__", ".venv"})
DEFAULT_DENY_SUFFIXES = frozenset({".pem", ".key"})

# attach_dir 专属的额外忽略规则（hard-deny 之外的"通常无用"项），
# 跟 DEFAULT_DENY_* 区分：deny 是安全/隔离边界，这里只是节省带宽。
ATTACH_DIR_EXTRA_IGNORE_DIRS = frozenset({
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", ".claude", ".omx", "target", "out", "runs",
})
ATTACH_DIR_EXTRA_IGNORE_GLOBS = (
    "*.lock", "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.class",
    "*.zip", "*.tar", "*.gz", "*.tgz", "*.bz2", "*.7z", "*.rar",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg",
    "*.pdf", "*.mp4", "*.mp3", "*.wav", "*.webm",
    "*.exe", "*.bin", "*.o", "*.a",
    ".DS_Store", "Thumbs.db",
)


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
        self.manifest_path: Path = self.root / "MANIFEST.json"
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

    def check_external_path(self, p: Path) -> None:
        """对外部路径(不要求在 sandbox 内)做黑名单校验。
        ExportFile 写回真实路径时调用,确保拿到 ExportFile 的 dest 也受
        per-Workspace 自定义 deny 集合的保护,与 resolve()/attach() 对称。
        """
        for part in p.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"导出目标命中黑名单目录: {p}")
        if p.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"导出目标文件类型在黑名单: {p.suffix}")

    # ── Manifest 管理 ───────────────────────────────────────
    def manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, data: dict[str, str]) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.manifest_path)

    def attach(self, source_path: str | Path) -> Path:
        """把外部文件复制进 sandbox 根。返回 sandbox 内绝对路径。
        - source 不存在 → FileNotFoundError
        - source 命中 deny (按目录段 / 后缀) → WorkspaceViolation
        - sandbox 已有同名文件 → FileExistsError
        """
        src = Path(source_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"源文件不存在: {src}")
        if not src.is_file():
            raise IsADirectoryError(f"源不是文件: {src}")

        src_resolved = src.resolve(strict=True)
        for part in src_resolved.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"源文件命中黑名单目录: {src_resolved}")
        if src_resolved.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"源文件类型在黑名单: {src_resolved.suffix}")

        dst = self.root / src_resolved.name
        if dst.exists():
            raise FileExistsError(f"sandbox 已有同名文件: {self.relative(dst)}")

        shutil.copy2(src_resolved, dst)
        m = self.manifest()
        m[self.relative(dst)] = str(src_resolved)
        self._write_manifest(m)
        return dst

    def origin_of(self, sandbox_path: str | Path) -> Path | None:
        """查 sandbox 文件对应的原始源路径。未在 manifest 中 → None。"""
        p = Path(sandbox_path).resolve()
        try:
            rel = str(p.relative_to(self.root))
        except ValueError:
            return None
        src = self.manifest().get(rel)
        return Path(src) if src else None

    def attach_dir(
        self,
        source_path: str | Path,
        *,
        max_file_bytes: int = 1 * 1024 * 1024,
        max_total_bytes: int = 50 * 1024 * 1024,
        extra_ignore_globs: Iterable[str] = (),
    ) -> dict[str, Any]:
        """递归把外部目录拷进 sandbox。返回统计 dict。

        规则:
        - 单文件 > max_file_bytes 跳过
        - 累计 > max_total_bytes 立即停止 (truncated=True)
        - hard deny (deny_dirs / deny_suffixes) 不可绕过,与单文件 attach 对称
        - extra ignore (build/cache/二进制) 默认开启,可加 extra_ignore_globs 追加
        - symlink 一律跳过 (避免指向 sandbox 外的逃逸)
        """
        src = Path(source_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"源目录不存在: {src}")
        if not src.is_dir():
            raise NotADirectoryError(f"源不是目录: {src}")

        src_resolved = src.resolve(strict=True)
        for part in src_resolved.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"源目录命中黑名单: {src_resolved}")

        dest_root = self.root / src_resolved.name
        if dest_root.exists():
            raise FileExistsError(f"sandbox 已有同名目录: {self.relative(dest_root)}")
        dest_root.mkdir()

        ignore_globs = list(ATTACH_DIR_EXTRA_IGNORE_GLOBS) + list(extra_ignore_globs)
        ignore_dirs = self._deny_dirs | ATTACH_DIR_EXTRA_IGNORE_DIRS

        manifest = self.manifest()
        copied: list[str] = []
        skipped_too_large: list[str] = []
        skipped_ignored = 0
        total_bytes = 0
        truncated = False

        for src_file in sorted(src_resolved.rglob("*")):
            if src_file.is_symlink() or not src_file.is_file():
                continue

            rel_path = src_file.relative_to(src_resolved)
            rel_parent_parts = rel_path.parts[:-1]

            if any(p in ignore_dirs for p in rel_parent_parts):
                skipped_ignored += 1
                continue
            if any(fnmatch.fnmatch(src_file.name, g) for g in ignore_globs):
                skipped_ignored += 1
                continue
            if src_file.suffix in self._deny_suffixes:
                skipped_ignored += 1
                continue

            size = src_file.stat().st_size
            if size > max_file_bytes:
                skipped_too_large.append(f"{rel_path} ({size // 1024}KB)")
                continue
            if total_bytes + size > max_total_bytes:
                truncated = True
                break

            dst_file = dest_root / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            manifest[self.relative(dst_file)] = str(src_file)
            copied.append(str(rel_path))
            total_bytes += size

        self._write_manifest(manifest)

        return {
            "sandbox_dir": self.relative(dest_root),
            "source": str(src_resolved),
            "copied_count": len(copied),
            "copied_files": copied,
            "skipped_too_large": skipped_too_large,
            "skipped_ignored_count": skipped_ignored,
            "total_bytes": total_bytes,
            "truncated": truncated,
        }
