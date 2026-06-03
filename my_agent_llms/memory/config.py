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
    l1_max_tokens: int = 8000      # 工作记忆 token 预算(放得下典型对话活跃工作集,减少抖动)
    l1_recent_turns: int = 10      # 最近 N 条硬保护,永不驱逐
    l2_max_tokens: int = 1000

    # ── L2 分层托管(摘要纠错) ─────────────────────────────
    l2_reflect_every_n_turns: int = 5  # 每 N 轮定期反思校正摘要(0 = 关)

    # ── L3 检索缓冲 ────────────────────────────────────────
    l3_ttl_turns: int = 3          # 距上次命中超过 N 轮 → 过期移除
    l3_max_entries: int = 20       # 台账容量上限,超了淘汰最低分
    l3_promote_min_hits: int = 3   # 累计命中达此次数才够格晋升 L1
    l3_promote_min_score: float = 0.6  # 且最近命中分须 >= 此值

    # ── 热度 / 升降级 ───────────────────────────────────────
    promote_threshold: float = 0.7
    demote_threshold: float = 0.3
    promote_min_hits: int = 3
    # L0 实绩毕业: L1 项 pinned 且访问达此次数 → 晋升为 L0 卡(L1_GRADUATED)。
    # 比 promote_min_hits 高一档,因为 L0 是跨会话的更高门槛。
    l0_graduate_min_hits: int = 5
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
