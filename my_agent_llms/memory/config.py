"""记忆系统的配置参数。"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


ColdBackendType = Literal["none", "jsonl", "sqlite"]
VectorBackendType = Literal["memory", "sqlite"]


class MemoryConfig(BaseModel):
    """记忆系统配置。

    阈值集中在这里，方便整体调参；不同 Agent 可以持有不同的 config。
    """

    # ── 容量上限 ────────────────────────────────────────────
    l1_max_tokens: int = 4000
    l1_recent_turns: int = 6
    l2_max_tokens: int = 1000

    # ── 热度 / 升降级 ───────────────────────────────────────
    promote_threshold: float = 0.7
    demote_threshold: float = 0.3
    promote_min_hits: int = 3
    decay_tau_days: float = 7.0
    w_access: float = 0.4
    w_recency: float = 0.4
    w_explicit: float = 0.2

    # ── 持久化 ─────────────────────────────────────────────
    storage_dir: Optional[Path] = None
    cold_backend: ColdBackendType = "jsonl"
    vector_backend: VectorBackendType = "memory"

    def cold_path(self) -> Optional[Path]:
        if self.storage_dir is None or self.cold_backend == "none":
            return None
        if self.cold_backend == "jsonl":
            return self.storage_dir / "cold.jsonl"
        if self.cold_backend == "sqlite":
            return self.storage_dir / "memory.db"
        return None

    def vector_path(self) -> Optional[Path]:
        if self.storage_dir is None or self.vector_backend != "sqlite":
            return None
        return self.storage_dir / "memory.db"  # 与冷存储共享同一文件
