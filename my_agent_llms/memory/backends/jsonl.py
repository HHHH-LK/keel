"""JSONL 后端 —— 简单、可读、零依赖。

不适合大数据量（按 ID 查询是 O(n) 全表扫描），适合学习和小规模场景。
"""
from pathlib import Path
from typing import Iterator, Optional

from my_agent_llms.memory.backends.base import ColdBackend
from my_agent_llms.memory.item import MemoryItem


class JSONLColdBackend(ColdBackend):
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def add(self, item: MemoryItem) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def get(self, item_id: str) -> Optional[MemoryItem]:
        for it in self.iter_all():
            if it.id == item_id:
                return it
        return None

    def iter_all(self) -> Iterator[MemoryItem]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield MemoryItem.model_validate_json(line)
