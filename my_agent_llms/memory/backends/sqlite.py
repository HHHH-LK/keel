"""SQLite 后端 —— ColdBackend 与 VectorBackend 共享同一 DB 文件。

冷存储表 `memory_items`：按 id PRIMARY KEY 索引，O(log n) 查询。
向量表  `memory_vectors`：embedding 存 BLOB（float32 数组），启动时按需加载到内存做余弦检索。

设计取舍：
- 不用 sqlite-vec 扩展 → 零额外依赖
- 几万条以内查询走内存全扫足够快；再大就该上专用向量库（Chroma/Qdrant）
"""
import array
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from my_agent_llms.memory.backends.base import ColdBackend, VectorBackend
from my_agent_llms.memory.item import MemoryItem


_COLD_DDL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    role            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_access     TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    pinned          INTEGER NOT NULL DEFAULT 0,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT
);
"""

_VECTOR_DDL = """
CREATE TABLE IF NOT EXISTS memory_vectors (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    role        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    metadata_json TEXT,
    dim         INTEGER NOT NULL,
    vector      BLOB
);
"""


def _vector_to_blob(vec: Sequence[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _blob_to_vector(blob: bytes) -> List[float]:
    return list(array.array("f", blob))


def _serialize(item: MemoryItem) -> Tuple:
    return (
        item.id,
        item.content,
        item.role,
        item.created_at.isoformat(),
        item.last_access.isoformat(),
        item.access_count,
        1 if item.pinned else 0,
        item.token_estimate,
        json.dumps(item.metadata, ensure_ascii=False),
    )


def _deserialize_cold(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        role=row["role"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_access=datetime.fromisoformat(row["last_access"]),
        access_count=row["access_count"],
        pinned=bool(row["pinned"]),
        token_estimate=row["token_estimate"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


class _ConnectionMixin:
    """共享 SQLite 连接 —— 多个 backend 实例可指向同一 DB 文件。"""

    _conn_cache: Dict[str, sqlite3.Connection] = {}

    @classmethod
    def get_connection(cls, path: Path) -> sqlite3.Connection:
        key = str(path.resolve())
        if key not in cls._conn_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_COLD_DDL)
            conn.execute(_VECTOR_DDL)
            conn.commit()
            cls._conn_cache[key] = conn
        return cls._conn_cache[key]


class SQLiteColdBackend(_ConnectionMixin, ColdBackend):
    """L4 SQLite 后端。"""

    def __init__(self, path: Path):
        self.path = path
        self.conn = self.get_connection(path)

    def add(self, item: MemoryItem) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_items "
            "(id, content, role, created_at, last_access, access_count, pinned, token_estimate, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _serialize(item),
        )
        self.conn.commit()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        row = self.conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return _deserialize_cold(row) if row else None

    def iter_all(self) -> Iterator[MemoryItem]:
        cur = self.conn.execute(
            "SELECT * FROM memory_items ORDER BY created_at ASC"
        )
        for row in cur:
            yield _deserialize_cold(row)

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()
        return row[0]


class SQLiteVectorBackend(_ConnectionMixin, VectorBackend):
    """L5 SQLite 向量后端。

    embedding 持久化到 BLOB，启动时全量加载到内存（`_cache`）做余弦检索。
    没有 embedder 时退化为 TF-IDF（沿用 InMemoryVectorBackend 风格）—— 这种
    情况下 vector 字段为空，仍可重启恢复 items 但需要重建 TF-IDF 索引。
    """

    def __init__(
        self,
        path: Path,
        embedder=None,
    ):
        self.path = path
        self.embedder = embedder
        self.conn = self.get_connection(path)

        # 内存缓存：避免每次 search 都全表扫
        self._cache: Dict[str, MemoryItem] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._load_from_disk()

    # ── 启动恢复 ────────────────────────────────────────────
    def _load_from_disk(self) -> None:
        cur = self.conn.execute(
            "SELECT id, content, role, created_at, metadata_json, vector FROM memory_vectors"
        )
        for row in cur:
            item = MemoryItem(
                id=row["id"],
                content=row["content"],
                role=row["role"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_access=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            self._cache[item.id] = item
            if row["vector"]:
                self._vectors[item.id] = _blob_to_vector(row["vector"])

    # ── VectorBackend 接口 ──────────────────────────────────
    def add(
        self,
        item: MemoryItem,
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        if item.id in self._cache:
            return

        if vector is None and self.embedder is not None:
            vector = self.embedder.embed(item.content)

        self._cache[item.id] = item
        if vector is not None:
            self._vectors[item.id] = list(vector)

        self.conn.execute(
            "INSERT OR REPLACE INTO memory_vectors "
            "(id, content, role, created_at, metadata_json, dim, vector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.id,
                item.content,
                item.role,
                item.created_at.isoformat(),
                json.dumps(item.metadata, ensure_ascii=False),
                len(vector) if vector is not None else 0,
                _vector_to_blob(vector) if vector is not None else None,
            ),
        )
        self.conn.commit()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._cache.get(item_id)

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        item = self._cache.pop(item_id, None)
        self._vectors.pop(item_id, None)
        if item is not None:
            self.conn.execute("DELETE FROM memory_vectors WHERE id = ?", (item_id,))
            self.conn.commit()
        return item

    def items(self) -> List[MemoryItem]:
        return list(self._cache.values())

    def search(
        self,
        query: str,
        query_vector: Optional[Sequence[float]] = None,
        k: int = 5,
    ) -> List[Tuple[MemoryItem, float]]:
        if not self._cache:
            return []

        if query_vector is None and self.embedder is not None:
            query_vector = self.embedder.embed(query)

        if query_vector is not None and self._vectors:
            scored = []
            for item_id, vec in self._vectors.items():
                sim = _cosine(query_vector, vec)
                scored.append((self._cache[item_id], sim))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            return scored[:k]

        # 无 embedder 时降级 TF-IDF（按内存的方式动态算）
        from my_agent_llms.memory.backends.inmemory import (
            _tfidf_score,
            _tokenize as _toks,
        )

        return _tfidf_score(query, self._cache, k)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
