"""冷回路 reconcile 测试(Phase 4.2)。

冷回路 = 后台定期巡检,产出"提议"再确认处置(propose→confirm→apply)。
- 确定性部分:pending GC(陈旧低证据的待确认事实清掉)
- LLM 部分(用 FakeLLM 搭骨架):CORRECT 自动判定、语义冲突提议
"""
import json
from datetime import datetime, timedelta

from my_agent_llms.memory.kg import KGStore, Relation
from my_agent_llms.memory.kg_reconcile import KGReconciler


class FakeLLM:
    """固定返回一组冲突提议(模拟冷回路 LLM 语义判断)。"""

    def __init__(self, proposals):
        self._proposals = proposals

    def invoke(self, messages):
        return json.dumps(self._proposals, ensure_ascii=False)


def _two_conflicting_facts(store):
    """直接塞两条语义冲突但谓词不同的事实(词表抓不到):喜欢咖啡 vs 讨厌咖啡。"""
    uid = store.get_or_create_entity("PERSON", "user")
    coffee = store.get_or_create_entity("ITEM", "咖啡")
    store.add_relation(Relation(
        id="ra", subject_id=uid, predicate="喜欢", object_id=coffee,
        valid_from=datetime(2025, 1, 1), source_type="user_stated", authority=2,
    ))
    store.add_relation(Relation(
        id="rb", subject_id=uid, predicate="讨厌", object_id=coffee,
        valid_from=datetime(2025, 2, 1), source_type="user_stated", authority=2,
    ))


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


# ─────────────────────────────────────────────
# Stage 2: 语义冲突(LLM 提议 → 置信门确认 → 处置)
# ─────────────────────────────────────────────

def test_semantic_conflict_resolved_via_llm_proposal():
    """LLM 提议'喜欢咖啡 vs 讨厌咖啡'冲突且高置信 → supersede 更旧的一方。"""
    store = KGStore()
    _two_conflicting_facts(store)
    fake = FakeLLM([{"fact_a_id": "ra", "fact_b_id": "rb", "confidence": 0.9, "reason": "喜欢vs讨厌"}])
    r = KGReconciler(store, llm=fake, conflict_confidence_threshold=0.7)
    report = r.reconcile(now=datetime(2025, 3, 1))
    active = {a.id for a in store.find_active_relations_for_entity("user")}
    assert active == {"rb"}                     # 更新的(讨厌,2月)留下
    assert "ra" in report["conflicts_resolved"]  # 更旧的(喜欢,1月)被取代


def test_low_confidence_conflict_not_applied():
    """低置信提议不处置 —— 确认门挡住,两条都留。"""
    store = KGStore()
    _two_conflicting_facts(store)
    fake = FakeLLM([{"fact_a_id": "ra", "fact_b_id": "rb", "confidence": 0.4, "reason": "不确定"}])
    r = KGReconciler(store, llm=fake, conflict_confidence_threshold=0.7)
    report = r.reconcile(now=datetime(2025, 3, 1))
    active = {a.id for a in store.find_active_relations_for_entity("user")}
    assert active == {"ra", "rb"}
    assert report["conflicts_resolved"] == []


def test_no_llm_skips_semantic_pass():
    """没有 llm → 不跑语义冲突,只跑确定性 GC。"""
    store = KGStore()
    _two_conflicting_facts(store)
    r = KGReconciler(store)  # 无 llm
    report = r.reconcile(now=datetime(2025, 3, 1))
    assert "conflicts_resolved" not in report
    assert len(store.find_active_relations_for_entity("user")) == 2
