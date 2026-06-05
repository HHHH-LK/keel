"""跨项目复现台账 —— 记 (triple_key, project_id),按 triple_key 统计不同项目数。

住在用户层 storage_dir 的 kg.db(与用户层 KG 同库,新增一张表)。当某 triple_key
出现在 ≥N 个不同项目 → 上层据此把该事实提升到用户层 KG。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

_DDL = """
CREATE TABLE IF NOT EXISTS promotion_ledger (
    triple_key TEXT NOT NULL,
    project_id TEXT NOT NULL,
    PRIMARY KEY (triple_key, project_id)
);
"""


class PromotionLedger:
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            self.conn = sqlite3.connect(":memory:")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path), check_same_thread=False)
            # 与用户层 KG 同文件、不同连接:WAL + busy_timeout 避免并发写 database is locked
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_DDL)
        self.conn.commit()

    def record(self, triple_key: str, project_id: str) -> int:
        """登记一条 (triple_key, project_id)。同项目重复无副作用。
        返回登记后该 triple_key 的不同项目数。"""
        self.conn.execute(
            "INSERT OR IGNORE INTO promotion_ledger (triple_key, project_id) VALUES (?,?)",
            (triple_key, project_id),
        )
        self.conn.commit()
        return self.project_count(triple_key)

    def project_count(self, triple_key: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM promotion_ledger WHERE triple_key=?", (triple_key,)
        ).fetchone()
        return int(row[0]) if row else 0
