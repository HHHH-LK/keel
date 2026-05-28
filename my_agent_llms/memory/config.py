"""记忆系统的配置参数。"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


ColdBackendType = Literal["none", "jsonl", "sqlite"]
VectorBackendType = Literal["memory", "sqlite"]
TickMode = Literal["sync", "async", "off"]
# off    = 不做冲突检测,所有记忆共存
# fast   = SimilarityConflictDetector,仅靠 embedding 相似度,零额外 LLM 调用
# accurate = LLMConflictDetector,LLM 精判矛盾,准确率高,每写一次 LLM 调用
# extreme = KnowledgeGraphConflictDetector,实体+关系+时态,处理 7 种冲突类型
ConflictStrength = Literal["off", "fast", "accurate", "extreme"]


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
    # importance 公式四因子权重(总和 = 1.0,保持 importance ∈ [0, 1])
    w_access: float = 0.3
    w_recency: float = 0.3
    w_explicit: float = 0.2
    w_prior: float = 0.2

    # ── 持久化 ─────────────────────────────────────────────
    storage_dir: Optional[Path] = None
    cold_backend: ColdBackendType = "jsonl"
    vector_backend: VectorBackendType = "memory"

    # ── tick 调度 ──────────────────────────────────────────
    tick_mode: TickMode = "sync"
    tick_every_n_turns: int = 1

    # ── 冲突检测强度 ───────────────────────────────────────
    conflict_strength: ConflictStrength = "off"   # 默认关:零干扰、零成本
    conflict_threshold: float = 0.75              # fast/accurate 用到

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

    def playbook_path(self) -> Optional[Path]:
        """L0 playbook 持久化路径。复用 memory.db。

        storage_dir 未设置时返回 None,Playbook 走内存模式(测试用)。
        """
        if self.storage_dir is None:
            return None
        return self.storage_dir / "memory.db"
