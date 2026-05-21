"""存储后端抽象 —— Cold 与 Vector 各一套。"""
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple

from my_agent_llms.memory.item import MemoryItem


class ColdBackend(ABC):
    """L4 冷存储后端 —— 全量历史的持久化。

    关注点：写入 + 按 ID 取回 + 顺序遍历。
    无需检索能力（那是 VectorBackend 的事）。
    """

    @abstractmethod
    def add(self, item: MemoryItem) -> None:
        ...

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        ...

    @abstractmethod
    def iter_all(self) -> Iterator[MemoryItem]:
        ...

    def count(self) -> int:
        return sum(1 for _ in self.iter_all())

    def close(self) -> None:
        """子类有资源（DB 连接、文件句柄）时覆盖。"""


class VectorBackend(ABC):
    """L5 向量索引后端 —— 检索 + 存储。

    存储「MemoryItem 副本 + 可选的 embedding 向量」，提供按 query 检索的能力。
    实现可以是内存 TF-IDF、内存余弦、SQLite + BLOB、外部向量库（Chroma/FAISS）等。
    """

    @abstractmethod
    def add(
        self,
        item: MemoryItem,
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        ...

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        ...

    @abstractmethod
    def remove(self, item_id: str) -> Optional[MemoryItem]:
        ...

    @abstractmethod
    def items(self) -> List[MemoryItem]:
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        query_vector: Optional[Sequence[float]] = None,
        k: int = 5,
    ) -> List[Tuple[MemoryItem, float]]:
        ...

    def close(self) -> None:
        ...
