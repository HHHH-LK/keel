"""记忆层抽象基类 —— L1/L2/L4/L5 都实现这个接口。"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from my_agent_llms.memory.item import MemoryItem


class MemoryTier(ABC):
    """单层记忆存储的统一接口。

    各层差异由实现类决定（L1 内存 list、L4 JSONL 文件、L5 向量库等），
    Manager 只面向这个抽象编程。
    """

    name: str = "tier"

    @abstractmethod
    def add(self, item: MemoryItem) -> None:
        """写入一条记忆。"""

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """按 ID 取一条。找不到返回 None。"""

    @abstractmethod
    def items(self) -> List[MemoryItem]:
        """返回本层所有记忆（按写入顺序）。L4/L5 这种大容量层应分页或不实现。"""

    def evict(self) -> List[MemoryItem]:
        """触发本层的驱逐策略，返回被驱逐的条目。

        默认不驱逐；带容量上限的层（如 L1）需要覆盖。
        """
        return []

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        """删除并返回，找不到返回 None。默认不支持。"""
        raise NotImplementedError(f"{self.name} 不支持 remove")

    def extend(self, batch: Iterable[MemoryItem]) -> None:
        """批量写入。默认逐条调 add。"""
        for item in batch:
            self.add(item)
