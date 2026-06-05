"""UserLayer —— 用户层持久知识(仅 L0/KG),跨项目共享。

与项目层(完整 MemoryManager)正交:活内存层 L1/L2/L3 + 对话日志 L4 不在用户层。
读:query_facts 从用户层 KG 取事实。写:add_card 写 L0;ingest_confirmed_fact
把提升上来的事实直写 KG 主图。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from my_agent_llms.memory.kg import KGStore, KnowledgeGraphConflictDetector
from my_agent_llms.memory.playbook.store import PlaybookStore


class UserLayer:
    def __init__(self, storage_dir: Optional[Path], *, llm=None, embedder=None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        kg_path = (self.storage_dir / "kg.db") if self.storage_dir else None
        playbook_path = (self.storage_dir / "memory.db") if self.storage_dir else None
        self.kg = KnowledgeGraphConflictDetector(llm, KGStore(kg_path), embedder=embedder)
        self.playbook = PlaybookStore(playbook_path)

    def query_facts(self, query: str, max_facts: int = 8) -> List[str]:
        return self.kg.query_facts(query, max_facts=max_facts)

    def ingest_confirmed_fact(self, rel_data: dict) -> None:
        self.kg.apply_confirmed_relation(rel_data)

    def add_card(self, card) -> None:
        self.playbook.add(card)
