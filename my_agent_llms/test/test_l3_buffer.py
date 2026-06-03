"""L3 检索缓冲层测试。

L3 = "工作台上临时摊开的参考资料":被动 recall 命中的项在这里登记,
跨轮累计命中次数,反复命中(高阈值)才晋升进 L1,否则按 TTL 过期。
只持引用(item_id + 元数据),真身留在 L5。
"""
import tempfile
from pathlib import Path

from my_agent_llms.memory import (
    MemoryConfig,
    MemoryManager,
)
from my_agent_llms.memory.cold import ColdStorage
from my_agent_llms.memory.item import MemoryItem
from my_agent_llms.memory.backends.jsonl import JSONLColdBackend


# ─────────────────────────────────────────────
# RecallBuffer 单元
# ─────────────────────────────────────────────

def _buffer(**overrides):
    from my_agent_llms.memory.recall_buffer import RecallBuffer
    cfg = MemoryConfig(**overrides)
    return RecallBuffer(cfg)


def test_record_hit_creates_entry():
    buf = _buffer()
    buf.record_hit("item1", score=0.7, turn=0)
    assert len(buf) == 1
    e = buf.get_entry("item1")
    assert e.hit_count == 1
    assert e.hit_score == 0.7


def test_record_hit_increments_count():
    buf = _buffer()
    buf.record_hit("item1", score=0.5, turn=0)
    buf.record_hit("item1", score=0.8, turn=1)
    e = buf.get_entry("item1")
    assert e.hit_count == 2
    assert e.hit_score == 0.8          # 取最新分
    assert e.last_recalled_turn == 1
    assert e.first_recalled_turn == 0


def test_evict_expired_drops_stale():
    buf = _buffer(l3_ttl_turns=3)
    buf.record_hit("old", score=0.5, turn=0)
    buf.record_hit("fresh", score=0.5, turn=3)
    # 当前轮 5: old 距上次命中 5 轮 > 3 → 过期; fresh 距 2 轮 → 留
    expired = buf.evict_expired(current_turn=5)
    assert "old" in expired
    assert buf.get_entry("old") is None
    assert buf.get_entry("fresh") is not None


def test_capacity_evicts_lowest_score():
    buf = _buffer(l3_max_entries=2)
    buf.record_hit("low", score=0.2, turn=0)
    buf.record_hit("mid", score=0.5, turn=0)
    buf.record_hit("high", score=0.9, turn=0)  # 超容量 → 踢掉 low
    assert len(buf) == 2
    assert buf.get_entry("low") is None
    assert buf.get_entry("high") is not None


def test_promotable_respects_min_hits():
    buf = _buffer(l3_promote_min_hits=3)
    for t in range(3):
        buf.record_hit("hot", score=0.7, turn=t)
    buf.record_hit("cold", score=0.7, turn=0)
    ids = [e.item_id for e in buf.promotable(min_hits=3, min_score=0.6)]
    assert "hot" in ids
    assert "cold" not in ids


def test_promotable_respects_min_score():
    buf = _buffer()
    for t in range(5):
        buf.record_hit("lowscore", score=0.3, turn=t)
    ids = [e.item_id for e in buf.promotable(min_hits=3, min_score=0.6)]
    assert "lowscore" not in ids


# ─────────────────────────────────────────────
# 集成: 被动 recall 写入 L3
# ─────────────────────────────────────────────

def test_passive_recall_records_into_l3():
    mgr = MemoryManager(MemoryConfig())
    item = mgr.write("我之前提到我喜欢喝美式咖啡", role="user")
    # 挤出 L1,使其只活在 L5(被动 recall 默认排除 L1 项,避免重复注入)
    mgr.working.remove(item.id)
    mgr.assemble_context("sys", query="咖啡好喝吗", passive_recall_k=5)
    # L3 应登记了命中项
    assert len(mgr.recall_buffer) >= 1


def test_l3_entry_promotes_to_l1_after_repeated_recall():
    cfg = MemoryConfig(l3_promote_min_hits=2, l3_promote_min_score=0.0)
    mgr = MemoryManager(cfg)
    item = mgr.write("北京的会议定在下周三上午十点", role="user")
    # 把它挤出 L1,确保只活在 L5(否则 recall 默认排除 L1 项)
    mgr.working.remove(item.id)

    # 反复用相关 query 召回 → L3 命中累计
    for _ in range(3):
        mgr.assemble_context("sys", query="会议是什么时候", passive_recall_k=5)
        mgr.tick()

    l1_ids = {it.id for it in mgr.working.items()}
    assert item.id in l1_ids                       # 已晋升回 L1
    assert mgr.recall_buffer.get_entry(item.id) is None  # 晋升即移除


def test_stats_includes_l3():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我喜欢喝美式咖啡", role="user")
    mgr.assemble_context("sys", query="咖啡", passive_recall_k=5)
    assert "l3_entries" in mgr.stats()


# ─────────────────────────────────────────────
# L1 dup 修复: 同一项不重复落 L4
# ─────────────────────────────────────────────

def test_cold_storage_dedup_add():
    """同一 item 反复 add 到 L4,不产生重复(修 jsonl 追加 dup)。"""
    with tempfile.TemporaryDirectory() as d:
        backend = JSONLColdBackend(Path(d) / "cold.jsonl")
        cold = ColdStorage(backend)
        item = MemoryItem(content="会议纪要", role="user")
        cold.add(item)
        cold.add(item)   # 同 id 再写一次(模拟晋升回 L1 后又被 evict)
        assert cold.count() == 1


def test_cold_overwrites_mutated_item():
    """同 id 但字段变了(如 pinned)再写 → last-write-wins,不丢更新。

    回归: 旧的 _persisted_ids 跳过写入会让 pin 在重启后丢失。
    """
    with tempfile.TemporaryDirectory() as d:
        backend = JSONLColdBackend(Path(d) / "cold.jsonl")
        cold = ColdStorage(backend)
        item = MemoryItem(content="重要纪要", role="user", pinned=False)
        cold.add(item)
        item.pinned = True              # 晋升回 L1 后被 pin
        cold.add(item)                  # 再 evict 落盘
        got = cold.get(item.id)
        assert got.pinned is True       # 最新值,不是陈旧的 False
        assert cold.count() == 1


def test_l3_injection_skips_items_in_l1():
    """L3 注入不渲染已在 L1 的项(避免与 L1 原文重复)。

    场景: 一项在 L3 缓冲中,又被 importance 召回路径加回 L1(那条路径不清 L3)。
    """
    mgr = MemoryManager(MemoryConfig())
    item = mgr.write("季度会议定在下周三上午", role="user")
    mgr.working.remove(item.id)
    mgr.assemble_context("sys", query="会议什么时候", passive_recall_k=5)
    assert mgr.recall_buffer.get_entry(item.id) is not None  # 已在 L3

    # 模拟被另一条路径加回 L1(未清 L3)
    mgr.working.add(item)
    text = mgr._compose_l3_injection()
    assert item.content[:120] not in text


def test_reflection_robust_to_tick_throttle():
    """反思扳机不被 tick_every_n_turns 节流吞掉,按真实轮次推进。"""
    calls = []
    cfg = MemoryConfig(
        tick_every_n_turns=2,
        l2_reflect_every_n_turns=3,
        conflict_strength="off",
    )
    mgr = MemoryManager(cfg, reconciler=lambda c, s: calls.append(s) or c)
    mgr.summary._set_summary_text("摘要")
    mgr.write("一些最近对话", role="user")  # 让反思有 recent 内容
    for _ in range(12):
        mgr.tick()
    # _tick_impl 在偶数轮跑(2,4,6,8,10,12);距上次反思>=3轮即触发 → 3 次
    assert len(calls) == 3


if __name__ == "__main__":
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"  ✓ {f.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {f.__name__}: {exc}")
    print(f"\n{passed}/{len(funcs)} passed")
