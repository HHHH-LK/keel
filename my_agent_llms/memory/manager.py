"""MemoryManager —— 统一调度各层，Agent 持有一个。

支持自由配置：
- embedding：传 EmbeddingProvider 实例或裸 callable
- 冷存储：JSONL / SQLite / 关闭
- 向量库：内存 / SQLite（重启可恢复）
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from my_agent_llms.core.message import MessageRole
from my_agent_llms.memory.backends.base import ColdBackend, VectorBackend
from my_agent_llms.memory.backends.inmemory import InMemoryVectorBackend
from my_agent_llms.memory.backends.jsonl import JSONLColdBackend
from my_agent_llms.memory.backends.sqlite import (
    SQLiteColdBackend,
    SQLiteVectorBackend,
)
from my_agent_llms.memory.base import MemoryTier
from my_agent_llms.memory.cold import ColdStorage
from my_agent_llms.memory.config import MemoryConfig
from my_agent_llms.memory.embeddings import (
    EmbeddingProvider,
    coerce_embedding,
)
from my_agent_llms.memory.item import MemoryItem
from my_agent_llms.memory.semantic import SemanticIndex
from my_agent_llms.memory.summary import Summarizer, SummaryMemory
from my_agent_llms.memory.working import WorkingMemory


class MemoryManager:
    """记忆系统对外统一入口。

    构造时可注入：
    - embedding: EmbeddingProvider 或 callable（None 表示走 TF-IDF）
    - cold_backend / vector_backend: 直接传 backend 实例，覆盖 config 中的字符串配置
    - summarizer: L2 摘要器
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        embedding: Union[None, EmbeddingProvider, Callable[[str], Sequence[float]]] = None,
        cold_backend: Optional[ColdBackend] = None,
        vector_backend: Optional[VectorBackend] = None,
        summarizer: Optional[Summarizer] = None,
        summary_flush_threshold: int = 4,
    ):
        self.config = config or MemoryConfig()
        self.embedding = coerce_embedding(embedding)

        self.working = WorkingMemory(self.config)
        self.cold = ColdStorage(cold_backend or self._build_cold_backend())
        self.semantic = SemanticIndex(
            vector_backend or self._build_vector_backend()
        )
        self.summary = SummaryMemory(
            flush_threshold=summary_flush_threshold,
            summarizer=summarizer,
            max_tokens=self.config.l2_max_tokens,
        )

        self.tiers: Dict[str, MemoryTier] = {
            self.working.name: self.working,
            self.summary.name: self.summary,
            self.cold.name: self.cold,
            self.semantic.name: self.semantic,
        }

    # ── 后端工厂（按 config 字符串选择） ────────────────────
    def _build_cold_backend(self) -> Optional[ColdBackend]:
        path = self.config.cold_path()
        if path is None:
            return None
        backend_type = self.config.cold_backend
        if backend_type == "jsonl":
            return JSONLColdBackend(path)
        if backend_type == "sqlite":
            return SQLiteColdBackend(path)
        return None

    def _build_vector_backend(self) -> VectorBackend:
        backend_type = self.config.vector_backend
        if backend_type == "sqlite":
            path = self.config.vector_path()
            if path is None:
                raise ValueError("vector_backend='sqlite' 需要同时设置 storage_dir")
            return SQLiteVectorBackend(path, embedder=self.embedding)
        return InMemoryVectorBackend(embedder=self.embedding)

    # ── 写入路径 ────────────────────────────────────────────
    def write(
        self,
        content: str,
        role: MessageRole = "user",
        *,
        pinned: bool = False,
        metadata: Optional[Dict] = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content,
            role=role,
            pinned=pinned,
            metadata=metadata or {},
        )
        self.working.add(item)
        self.semantic.add(item)
        self._cascade_evict_from_l1()
        return item

    def _cascade_evict_from_l1(self) -> None:
        evicted = self.working.evict()
        for it in evicted:
            self.cold.add(it)
            self.summary.add(it)

    # ── 读取路径 ────────────────────────────────────────────
    def assemble_context(
        self,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        summary = self.summary.current_summary()
        if summary is not None:
            messages.append(summary.to_message_dict())

        for it in self.working.items():
            messages.append(it.to_message_dict())
        return messages

    # ── 检索 ────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        exclude_working: bool = True,
    ) -> List[Tuple[MemoryItem, float]]:
        results = self.semantic.search(query, k=max(k * 2, k))
        if exclude_working:
            l1_ids = {it.id for it in self.working.items()}
            results = [(it, s) for it, s in results if it.id not in l1_ids]
        results = results[:k]
        for item, _ in results:
            item.touch()
        return results

    # ── 热度 / 维护 ────────────────────────────────────────
    def record_access(self, item_id: str) -> Optional[MemoryItem]:
        for tier in self.tiers.values():
            try:
                item = tier.get(item_id)
            except NotImplementedError:
                continue
            if item is not None:
                item.touch()
                return item
        return None

    def pin(self, item_id: str) -> bool:
        item = self.working.get(item_id)
        if item is None:
            return False
        item.pinned = True
        return True

    def unpin(self, item_id: str) -> bool:
        item = self.working.get(item_id)
        if item is None:
            return False
        item.pinned = False
        return True

    def tick(self) -> Dict[str, List[str]]:
        promoted: List[str] = []
        demoted: List[str] = []
        recalled: List[str] = []

        cfg = self.config
        kw = dict(
            decay_tau_days=cfg.decay_tau_days,
            w_access=cfg.w_access,
            w_recency=cfg.w_recency,
            w_explicit=cfg.w_explicit,
        )

        for item in self.working.items():
            score = item.importance(**kw)
            score_without_pin = score - (cfg.w_explicit if item.pinned else 0.0)
            if (
                not item.pinned
                and score >= cfg.promote_threshold
                and item.access_count >= cfg.promote_min_hits
            ):
                item.pinned = True
                promoted.append(item.id)
            elif item.pinned and score_without_pin < cfg.demote_threshold:
                item.pinned = False
                demoted.append(item.id)

        l1_ids = {it.id for it in self.working.items()}
        for item in self.semantic.items():
            if item.id in l1_ids:
                continue
            score = item.importance(**kw)
            if (
                score >= cfg.promote_threshold
                and item.access_count >= cfg.promote_min_hits
            ):
                item.pinned = True
                self.working.add(item)
                recalled.append(item.id)

        if recalled:
            self._cascade_evict_from_l1()

        return {"promoted": promoted, "demoted": demoted, "recalled": recalled}

    # ── 工具方法 ───────────────────────────────────────────
    def clear(self) -> None:
        self.working = WorkingMemory(self.config)
        self.tiers[self.working.name] = self.working

    def stats(self) -> Dict[str, int]:
        summary = self.summary.current_summary()
        return {
            "l1_items": len(self.working.items()),
            "l1_tokens": self.working.total_tokens(),
            "l2_tokens": summary.token_estimate if summary else 0,
            "l4_items": self.cold.count(),
            "l5_items": len(self.semantic.items()),
        }
