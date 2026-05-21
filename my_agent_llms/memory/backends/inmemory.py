"""内存向量后端：TF-IDF（无 embedder）或 余弦（有 embedder）。"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from my_agent_llms.memory.backends.base import VectorBackend
from my_agent_llms.memory.item import MemoryItem


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens: List[str] = []
    tokens.extend(_WORD_RE.findall(text))
    chars = [c for c in text if "一" <= c <= "鿿"]
    tokens.extend(chars)
    tokens.extend("".join(pair) for pair in zip(chars, chars[1:]))
    return tokens


class InMemoryVectorBackend(VectorBackend):
    """纯内存向量后端 —— 进程退出即丢失。

    embedder 存在 → 走余弦；不存在 → 走 TF-IDF。
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._items: Dict[str, MemoryItem] = {}
        self._postings: Dict[str, set] = defaultdict(set)
        self._doc_tokens: Dict[str, Counter] = {}
        self._doc_freq: Counter = Counter()
        self._embeddings: Dict[str, List[float]] = {}

    def add(
        self,
        item: MemoryItem,
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        if item.id in self._items:
            return

        self._items[item.id] = item

        # TF-IDF 索引始终建好（即使有 embedder，留作 fallback）
        tokens = _tokenize(item.content)
        counts = Counter(tokens)
        self._doc_tokens[item.id] = counts
        for tok in counts:
            self._postings[tok].add(item.id)
            self._doc_freq[tok] += 1

        if vector is None and self.embedder is not None:
            vector = self.embedder.embed(item.content)
        if vector is not None:
            self._embeddings[item.id] = list(vector)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        item = self._items.pop(item_id, None)
        if item is None:
            return None
        for tok in self._doc_tokens.pop(item_id, {}):
            self._postings[tok].discard(item_id)
            self._doc_freq[tok] -= 1
            if self._doc_freq[tok] <= 0:
                del self._doc_freq[tok]
        self._embeddings.pop(item_id, None)
        return item

    def items(self) -> List[MemoryItem]:
        return list(self._items.values())

    def search(
        self,
        query: str,
        query_vector: Optional[Sequence[float]] = None,
        k: int = 5,
    ) -> List[Tuple[MemoryItem, float]]:
        if not query or not self._items:
            return []

        if query_vector is None and self.embedder is not None:
            query_vector = self.embedder.embed(query)

        if query_vector is not None and self._embeddings:
            scored = []
            for item_id, vec in self._embeddings.items():
                sim = _cosine(query_vector, vec)
                scored.append((self._items[item_id], sim))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            return scored[:k]

        return _tfidf_score(query, self._items, k, self._postings, self._doc_tokens)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _tfidf_score(
    query: str,
    items: Dict[str, MemoryItem],
    k: int,
    postings: Optional[Dict[str, set]] = None,
    doc_tokens: Optional[Dict[str, Counter]] = None,
) -> List[Tuple[MemoryItem, float]]:
    """TF-IDF 评分。允许动态重建索引（SQLite fallback 时用）。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # 如果没传索引（SQLite fallback 场景），动态构建一次
    if postings is None or doc_tokens is None:
        postings = defaultdict(set)
        doc_tokens = {}
        for item_id, item in items.items():
            counts = Counter(_tokenize(item.content))
            doc_tokens[item_id] = counts
            for tok in counts:
                postings[tok].add(item_id)

    total_docs = max(1, len(items))
    scores: Dict[str, float] = defaultdict(float)
    for tok in q_tokens:
        docs = postings.get(tok)
        if not docs:
            continue
        idf = math.log((total_docs + 1) / (len(docs) + 1)) + 1.0
        for doc_id in docs:
            tf = doc_tokens[doc_id].get(tok, 0)
            scores[doc_id] += tf * idf

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(items[i], s) for i, s in ranked if s > 0]
