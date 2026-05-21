"""L1 工作内存：永驻 context、带 token 上限、最近 N 条受保护。"""
from typing import List, Optional

from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.config import MemoryConfig
from my_agent_llms.memory.item import MemoryItem


class WorkingMemory(MemoryTier):
    """L1 —— 每次 LLM 调用都会被纳入 context 的核心区域。

    驱逐规则：
    1. token 总量超过 l1_max_tokens 时触发
    2. 最近 l1_recent_turns 条永远保留（避免破坏当前对话上下文）
    3. pinned 项不驱逐
    4. 其余按 FIFO 驱逐最老的
    """

    name = "L1"

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self._items.append(item)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        for it in self._items:
            if it.id == item_id:
                return it
        return None

    def items(self) -> List[MemoryItem]:
        return list(self._items)

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        for idx, it in enumerate(self._items):
            if it.id == item_id:
                return self._items.pop(idx)
        return None

    def total_tokens(self) -> int:
        return sum(it.token_estimate for it in self._items)

    def evict(self) -> List[MemoryItem]:
        """超 budget 时驱逐：跳过 pinned 和最近 N 条。"""
        evicted: List[MemoryItem] = []
        budget = self.config.l1_max_tokens
        recent_n = self.config.l1_recent_turns

        if self.total_tokens() <= budget:
            return evicted

        # 倒数 recent_n 条的起始索引 —— 这些不动
        protected_start = max(0, len(self._items) - recent_n)

        idx = 0
        while self.total_tokens() > budget and idx < protected_start:
            candidate = self._items[idx]
            if candidate.pinned:
                idx += 1
                continue
            evicted.append(self._items.pop(idx))
            protected_start -= 1  # 列表缩短，保护区起点跟着前移

        return evicted
