"""冷回路 reconcile 测试(Phase 4.2)。

冷回路 = 后台定期巡检,产出"提议"再确认处置(propose→confirm→apply)。
- 确定性部分:pending GC(陈旧低证据的待确认事实清掉)
- LLM 部分(用 FakeLLM 搭骨架):CORRECT 自动判定、语义冲突提议
"""
import json
from datetime import datetime, timedelta

from my_agent_llms.memory.kg import KGStore
from my_agent_llms.memory.kg_reconcile import KGReconciler


def _rel(subj, pred, obj, scope="", subj_type="PERSON", obj_type="ITEM"):
    return {
        "subject_type": subj_type, "subject_name": subj,
        "predicate": pred,
        "object_type": obj_type, "object_name": obj,
        "scope": scope,
    }


# ─────────────────────────────────────────────
# Stage 1: pending GC(确定性)
# ─────────────────────────────────────────────

def test_reconcile_drops_stale_low_hit_pending():
    """陈旧 + 低证据的 pending → GC 清掉。"""
    store = KGStore()
    old = datetime(2025, 1, 1)
    store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", source_item_id="i1", now=old)
    r = KGReconciler(store, pending_ttl_seconds=3600, pending_promote_hits=2)
    report = r.reconcile(now=old + timedelta(days=1))
    assert store.pending_entries() == []
    assert len(report["dropped_pending"]) == 1


def test_reconcile_keeps_recent_pending():
    """最近还在出现的 pending 不动。"""
    store = KGStore()
    now = datetime(2025, 1, 1)
    store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", now=now)
    r = KGReconciler(store, pending_ttl_seconds=3600)
    r.reconcile(now=now + timedelta(seconds=10))
    assert len(store.pending_entries()) == 1


def test_reconcile_keeps_corroborated_pending():
    """证据已达晋升线的 pending 不能被当垃圾丢。"""
    store = KGStore()
    old = datetime(2025, 1, 1)
    store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", now=old)
    store.record_pending(_rel("user", "想学", "Rust"), reason="inferred", now=old)  # hit=2
    r = KGReconciler(store, pending_ttl_seconds=3600, pending_promote_hits=2)
    r.reconcile(now=old + timedelta(days=1))
    assert len(store.pending_entries()) == 1
