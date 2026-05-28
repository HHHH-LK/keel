"""L0 卡片的 sqlite 存储。

复用 memory.db 同一文件(L4/L5 已经在用),新增 playbook_cards 表。
读取走全量加载到内存(L0 数量通常 < 50,无需索引)。
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from my_agent_llms.memory.playbook.card import (
    L0Lifecycle,
    L0Source,
    L0Type,
    PlaybookCard,
)


_PLAYBOOK_DDL = """
CREATE TABLE IF NOT EXISTS playbook_cards (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    type            TEXT NOT NULL,
    lifecycle       TEXT NOT NULL DEFAULT 'active',
    confidence      REAL NOT NULL DEFAULT 0.8,
    created_at      TEXT NOT NULL,
    last_refresh    TEXT NOT NULL,
    last_negation   TEXT,
    source          TEXT NOT NULL,
    source_ref      TEXT,
    user_pinned     INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_playbook_lifecycle
    ON playbook_cards(lifecycle);
"""


def _serialize(card: PlaybookCard) -> tuple:
    return (
        card.id,
        card.content,
        card.type.value,
        card.lifecycle.value,
        card.confidence,
        card.created_at.isoformat(),
        card.last_refresh.isoformat(),
        card.last_negation.isoformat() if card.last_negation else None,
        card.source.value,
        card.source_ref,
        1 if card.user_pinned else 0,
        json.dumps(card.metadata, ensure_ascii=False),
    )


def _deserialize(row: sqlite3.Row) -> PlaybookCard:
    last_negation = row["last_negation"]
    return PlaybookCard(
        id=row["id"],
        content=row["content"],
        type=L0Type(row["type"]),
        lifecycle=L0Lifecycle(row["lifecycle"]),
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_refresh=datetime.fromisoformat(row["last_refresh"]),
        last_negation=datetime.fromisoformat(last_negation) if last_negation else None,
        source=L0Source(row["source"]),
        source_ref=row["source_ref"],
        user_pinned=bool(row["user_pinned"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


class PlaybookStore:
    """L0 卡片的持久化存储 + 查询。

    path=None 时使用纯内存模式(测试用 / storage_dir 未配置时)。
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._memory_cache: List[PlaybookCard] = []  # path=None 时用
        if path is not None:
            self._init_db()

    def _init_db(self) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.executescript(_PLAYBOOK_DDL)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("PlaybookStore in memory mode, no DB connection")
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── CRUD ─────────────────────────────────────

    def add(self, card: PlaybookCard) -> None:
        if self.path is None:
            # 内存模式:去重后追加
            self._memory_cache = [c for c in self._memory_cache if c.id != card.id]
            self._memory_cache.append(card)
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO playbook_cards "
                "(id, content, type, lifecycle, confidence, created_at, last_refresh, "
                " last_negation, source, source_ref, user_pinned, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _serialize(card),
            )
            conn.commit()

    def get(self, card_id: str) -> Optional[PlaybookCard]:
        if self.path is None:
            for c in self._memory_cache:
                if c.id == card_id:
                    return c
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM playbook_cards WHERE id = ?", (card_id,)
            ).fetchone()
            return _deserialize(row) if row else None

    def update(self, card: PlaybookCard) -> None:
        """全量替换写回(等价于 add)。"""
        self.add(card)

    def remove(self, card_id: str) -> None:
        if self.path is None:
            self._memory_cache = [c for c in self._memory_cache if c.id != card_id]
            return
        with self._conn() as conn:
            conn.execute("DELETE FROM playbook_cards WHERE id = ?", (card_id,))
            conn.commit()

    # ── 查询 ─────────────────────────────────────

    def all_active(self) -> List[PlaybookCard]:
        """所有 lifecycle=active 的卡片,按 type 优先级 + confidence 排序。"""
        if self.path is None:
            cards = [c for c in self._memory_cache if c.is_active()]
        else:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM playbook_cards WHERE lifecycle = ? "
                    "ORDER BY confidence DESC",
                    (L0Lifecycle.ACTIVE.value,),
                ).fetchall()
                cards = [_deserialize(r) for r in rows]
        # type 优先级排序:hard > identity > preference > state
        type_order = {
            L0Type.HARD_CONSTRAINT: 0,
            L0Type.IDENTITY: 1,
            L0Type.PREFERENCE: 2,
            L0Type.STATE: 3,
        }
        cards.sort(key=lambda c: (type_order.get(c.type, 4), -c.confidence))
        return cards

    def all_with_lifecycle(self, lifecycle: L0Lifecycle) -> List[PlaybookCard]:
        if self.path is None:
            return [c for c in self._memory_cache if c.lifecycle == lifecycle]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM playbook_cards WHERE lifecycle = ?",
                (lifecycle.value,),
            ).fetchall()
            return [_deserialize(r) for r in rows]

    def find_by_source_ref(self, source_ref: str) -> List[PlaybookCard]:
        """按 source_ref 反查卡片(用于 KG → L0 反哺)。"""
        if self.path is None:
            return [c for c in self._memory_cache if c.source_ref == source_ref]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM playbook_cards WHERE source_ref = ?",
                (source_ref,),
            ).fetchall()
            return [_deserialize(r) for r in rows]

    def count(self) -> int:
        if self.path is None:
            return len(self._memory_cache)
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM playbook_cards").fetchone()
            return row[0]

    def count_active(self) -> int:
        if self.path is None:
            return len([c for c in self._memory_cache if c.is_active()])
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM playbook_cards WHERE lifecycle = ?",
                (L0Lifecycle.ACTIVE.value,),
            ).fetchone()
            return row[0]
