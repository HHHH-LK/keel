"""PendingEdit + PendingEditStore —— 两步确认机制的状态载体。

EditFile / WriteFile / ExportFile 第一次调用时不真写,而是构造一个
PendingEdit 放进 store 并返回 pending_id; 用户在对话中明确确认后,
LLM 再用 pending_id + action=apply 触发真正落盘。
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

EditKind = Literal["edit", "write", "export"]


@dataclass
class PendingEdit:
    id: str
    kind: EditKind
    path: Path              # 目标绝对路径 (edit/write: sandbox 内; export: 真实路径)
    new_content: str        # 整文件新内容
    diff_preview: str
    source_hash: Optional[str]   # 目标文件当前 SHA-256; 新建文件为 None
    created_at: float = field(default_factory=time.time)


class PendingEditStore:
    """进程级单例。MVP 用 dict + 锁,TTL 过期就静默丢弃。"""

    def __init__(self, ttl_seconds: int = 420):
        self._items: dict[str, PendingEdit] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return secrets.token_hex(4)  # 8 字符,够用且短

    def put(self, pe: PendingEdit) -> None:
        with self._lock:
            self._items[pe.id] = pe

    def pop(self, pid: str) -> Optional[PendingEdit]:
        with self._lock:
            self._evict_expired_locked()
            return self._items.pop(pid, None)

    def discard(self, pid: str) -> bool:
        with self._lock:
            return self._items.pop(pid, None) is not None

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [pid for pid, pe in self._items.items() if now - pe.created_at > self._ttl]
        for pid in expired:
            self._items.pop(pid, None)
