"""项目根识别 + 双层存储目录路径(纯函数,无副作用)。

项目根 = 从起点向上找最近含 .git 的祖先目录;找不到则用起点(CWD)。
这样在 repo 任意子目录启动都共享同一份项目记忆。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def resolve_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if (d / ".git").exists():
            return d
    return start


def project_id(root: Path) -> str:
    """项目根绝对路径的 sha1 前 16 位,作目录名。"""
    return hashlib.sha1(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:16]


def project_storage_dir(base: Path, root: Path) -> Path:
    return Path(base) / "projects" / project_id(root)


def user_storage_dir(base: Path) -> Path:
    return Path(base) / "user"
