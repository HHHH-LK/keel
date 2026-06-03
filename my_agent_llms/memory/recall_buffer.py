"""L3 检索缓冲层 —— "工作台上临时摊开的参考资料"。

定位:介于 L5(向量仓库,只增不删)和 L1(对话现场)之间。
被动 recall 命中的项在这里登记台账,跨轮累计命中次数:
- 反复命中且分够高 → 晋升进 L1(恢复完整原文,汇入长期流)
- 没再命中 → 按轮次 TTL 过期移除

只持引用(item_id + 元数据),真身留 L5;纯内存,重启不恢复
(本来就该重新检索)。淘汰比 L1 还激进 —— 它本质是临时上下文。
"""
from typing import Dict, List, Optional

from pydantic import BaseModel

from my_agent_llms.memory.config import MemoryConfig


class RecallEntry(BaseModel):
    """L3 台账的一条:哪个 L5 项被借出来了 + 借的热度。"""
    item_id: str
    hit_score: float          # 最近一次命中相似度
    hit_count: int            # 累计被召回次数(跨轮)
    first_recalled_turn: int
    last_recalled_turn: int


class RecallBuffer:
    """L3 检索缓冲。不是 MemoryTier —— 它存台账(RecallEntry),不存 MemoryItem。"""

    name = "L3"

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._entries: Dict[str, RecallEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    # ── 写入 ────────────────────────────────────────────────
    def record_hit(self, item_id: str, score: float, turn: int) -> None:
        """登记一次命中。已存在则累加次数、刷新分数与轮次。"""
        entry = self._entries.get(item_id)
        if entry is not None:
            entry.hit_count += 1
            entry.hit_score = score
            entry.last_recalled_turn = turn
        else:
            self._entries[item_id] = RecallEntry(
                item_id=item_id,
                hit_score=score,
                hit_count=1,
                first_recalled_turn=turn,
                last_recalled_turn=turn,
            )
        self._enforce_capacity()

    # ── 淘汰 ────────────────────────────────────────────────
    def evict_expired(self, current_turn: int) -> List[str]:
        """移除距上次命中超过 l3_ttl_turns 的条目,返回被移除的 id。"""
        ttl = self.config.l3_ttl_turns
        expired = [
            item_id
            for item_id, e in self._entries.items()
            if current_turn - e.last_recalled_turn > ttl
        ]
        for item_id in expired:
            del self._entries[item_id]
        return expired

    def _enforce_capacity(self) -> None:
        """超过 l3_max_entries 时,淘汰命中分最低的。"""
        cap = self.config.l3_max_entries
        if len(self._entries) <= cap:
            return
        # 分低的先走;同分按命中次数少的先走
        ordered = sorted(
            self._entries.values(),
            key=lambda e: (e.hit_score, e.hit_count),
        )
        for e in ordered[: len(self._entries) - cap]:
            self._entries.pop(e.item_id, None)

    def remove(self, item_id: str) -> Optional[RecallEntry]:
        return self._entries.pop(item_id, None)

    # ── 查询 ────────────────────────────────────────────────
    def get_entry(self, item_id: str) -> Optional[RecallEntry]:
        return self._entries.get(item_id)

    def entries(self) -> List[RecallEntry]:
        """全部台账,按命中分降序(注入/展示用)。"""
        return sorted(self._entries.values(), key=lambda e: -e.hit_score)

    def promotable(self, min_hits: int, min_score: float) -> List[RecallEntry]:
        """够格晋升 L1 的条目:命中次数 + 分数双达标,按分降序。"""
        return [
            e
            for e in self.entries()
            if e.hit_count >= min_hits and e.hit_score >= min_score
        ]
