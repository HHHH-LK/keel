"""L5 语义索引：薄包装层，把请求委托给具体的 VectorBackend。

按检索时机的不同：
- 没有 embedder + 内存后端 → TF-IDF
- 有 embedder + 内存后端 → 余弦相似度
- SQLite 后端 → 同上但带持久化（重启可恢复）
"""
from typing import List, Optional, Tuple

from my_agent_llms.memory.backends.base import VectorBackend
from my_agent_llms.memory.backends.inmemory import InMemoryVectorBackend
from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.item import MemoryItem


class SemanticIndex(MemoryTier):
    """L5 —— 语义检索层。"""

    name = "L5"

    def __init__(self, backend: Optional[VectorBackend] = None):
        self.backend = backend if backend is not None else InMemoryVectorBackend()

    def add(self, item: MemoryItem) -> None:
        self.backend.add(item)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self.backend.get(item_id)

    def items(self) -> List[MemoryItem]:
        return self.backend.items()

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        return self.backend.remove(item_id)

    def search(self, query: str, k: int = 5) -> List[Tuple[MemoryItem, float]]:
        return self.backend.search(query, k=k)
