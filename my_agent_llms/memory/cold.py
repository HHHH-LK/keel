"""L4 冷存储：薄包装层，把请求委托给具体的 ColdBackend。"""
from typing import List, Optional

from my_agent_llms.memory.backends.base import ColdBackend
from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.item import MemoryItem


class ColdStorage(MemoryTier):
    """L4 —— 持久化全量历史。具体后端由构造时注入。"""

    name = "L4"

    def __init__(self, backend: Optional[ColdBackend] = None):
        # backend=None 表示禁用持久化
        self.backend = backend

    def add(self, item: MemoryItem) -> None:
        # 重复/重写交给后端处理:jsonl 走 last-write-wins,sqlite 走 REPLACE。
        # 不在这层拦截 —— 否则字段变更(如 pinned)的重写会被吞掉。
        if self.backend is not None:
            self.backend.add(item)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        if self.backend is None:
            return None
        return self.backend.get(item_id)

    def items(self) -> List[MemoryItem]:
        if self.backend is None:
            return []
        return list(self.backend.iter_all())

    def count(self) -> int:
        return 0 if self.backend is None else self.backend.count()
