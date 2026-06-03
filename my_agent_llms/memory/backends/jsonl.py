"""JSONL 后端 —— 简单、可读、零依赖。

不适合大数据量（按 ID 查询是 O(n) 全表扫描），适合学习和小规模场景。

写入是 append-only 日志,同一 id 可出现多行(项被晋升回 L1 又 evict、
或字段变更后重写)。读取时按 id 做 last-write-wins 去重 —— 语义上
等价于 SQLite 的 INSERT OR REPLACE,既消除重复又保证拿到最新值。
"""
from pathlib import Path
from typing import Dict, Iterator, Optional

from my_agent_llms.memory.backends.base import ColdBackend
from my_agent_llms.memory.item import MemoryItem


class JSONLColdBackend(ColdBackend):
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def add(self, item: MemoryItem) -> None:
        # 总是追加:重写同一 id 即写新行,iter_all 取最后一行(最新值)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def get(self, item_id: str) -> Optional[MemoryItem]:
        # 走 iter_all 以拿到 last-write-wins 后的最新版本
        for it in self.iter_all():
            if it.id == item_id:
                return it
        return None

    def iter_all(self) -> Iterator[MemoryItem]:
        if not self.path.exists():
            return
        # last-write-wins: 同 id 保留最后一次写入,顺序按首次出现
        latest: Dict[str, MemoryItem] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    it = MemoryItem.model_validate_json(line)
                    latest[it.id] = it
        yield from latest.values()
