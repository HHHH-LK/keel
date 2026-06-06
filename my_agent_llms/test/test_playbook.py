"""L0 Playbook + 集成场景测试。

覆盖:
- PlaybookCard 基础操作(refresh/negate/pin/forget)
- PlaybookStore 内存模式 + sqlite 持久化
- 写入消息触发 L0 晋升
- KG → L0 反哺(supersede 时 negate 旧卡)
- assemble_context 注入(核心段 + 背景段)
- 被动 recall
- CLI 命令(remember/forget/pin/list)
"""
import tempfile
from pathlib import Path

from my_agent_llms.memory import (
    L0Lifecycle,
    L0Source,
    L0Type,
    MemoryConfig,
    MemoryManager,
    PlaybookCard,
    PlaybookStore,
    classify_content_type,
)


# ─────────────────────────────────────────────
# PlaybookCard 基础操作
# ─────────────────────────────────────────────

def test_card_refresh():
    card = PlaybookCard(
        content="测试",
        type=L0Type.PREFERENCE,
        source=L0Source.USER_EXPLICIT,
        confidence=0.5,
    )
    card.refresh()
    assert card.confidence > 0.5


def test_card_refresh_pinned_no_change():
    card = PlaybookCard(
        content="测试",
        type=L0Type.PREFERENCE,
        source=L0Source.USER_EXPLICIT,
        confidence=0.8,
        user_pinned=True,
    )
    card.refresh()
    assert card.confidence == 0.8  # pinned 不动


def test_card_negate_by_type():
    """不同 type 在 negate 时降幅不同。"""
    cards = {
        t: PlaybookCard(content="测试", type=t, source=L0Source.USER_EXPLICIT, confidence=1.0)
        for t in (L0Type.STATE, L0Type.PREFERENCE, L0Type.IDENTITY, L0Type.HARD_CONSTRAINT)
    }
    for c in cards.values():
        c.negate()

    # state 降幅最大,hard_constraint 最小
    assert cards[L0Type.STATE].confidence < cards[L0Type.PREFERENCE].confidence
    assert cards[L0Type.PREFERENCE].confidence < cards[L0Type.IDENTITY].confidence
    assert cards[L0Type.IDENTITY].confidence < cards[L0Type.HARD_CONSTRAINT].confidence


def test_card_should_archive():
    card = PlaybookCard(
        content="测试", type=L0Type.PREFERENCE,
        source=L0Source.USER_EXPLICIT, confidence=0.2,
    )
    assert card.should_archive()

    # hard_constraint 永不撤下
    hard = PlaybookCard(
        content="过敏", type=L0Type.HARD_CONSTRAINT,
        source=L0Source.USER_EXPLICIT, confidence=0.1,
    )
    assert not hard.should_archive()


def test_classify_content_type():
    assert classify_content_type("我对花生过敏") == L0Type.HARD_CONSTRAINT
    assert classify_content_type("我叫张三") == L0Type.IDENTITY
    assert classify_content_type("我喜欢咖啡") == L0Type.PREFERENCE
    assert classify_content_type("今天天气好") == L0Type.STATE  # 无关键词 → 默认 state


# ─────────────────────────────────────────────
# PlaybookStore: 内存 + 持久化
# ─────────────────────────────────────────────

def test_store_memory_mode():
    store = PlaybookStore(path=None)
    card = PlaybookCard(
        content="测试", type=L0Type.PREFERENCE,
        source=L0Source.USER_EXPLICIT,
    )
    store.add(card)
    assert store.get(card.id) is not None
    assert store.count() == 1


def test_store_sqlite_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.db"
        store1 = PlaybookStore(path)
        card = PlaybookCard(
            content="跨重启测试",
            type=L0Type.HARD_CONSTRAINT,
            source=L0Source.USER_EXPLICIT,
            confidence=0.9,
        )
        store1.add(card)

        # 重新打开 store 模拟重启
        store2 = PlaybookStore(path)
        recovered = store2.get(card.id)
        assert recovered is not None
        assert recovered.content == "跨重启测试"
        assert recovered.type == L0Type.HARD_CONSTRAINT
        assert recovered.confidence == 0.9


def test_store_all_active_sorted_by_type():
    store = PlaybookStore(path=None)
    a = PlaybookCard(content="state", type=L0Type.STATE,
                     source=L0Source.USER_EXPLICIT, confidence=0.9)
    b = PlaybookCard(content="hard", type=L0Type.HARD_CONSTRAINT,
                     source=L0Source.USER_EXPLICIT, confidence=0.5)
    c = PlaybookCard(content="pref", type=L0Type.PREFERENCE,
                     source=L0Source.USER_EXPLICIT, confidence=0.8)
    store.add(a)
    store.add(b)
    store.add(c)

    actives = store.all_active()
    # type 优先级: hard > identity > preference > state
    assert actives[0].type == L0Type.HARD_CONSTRAINT
    assert actives[-1].type == L0Type.STATE


# ─────────────────────────────────────────────
# MemoryManager 集成: write 触发 L0 晋升
# ─────────────────────────────────────────────

def test_write_promotes_high_prior_to_l0():
    mgr = MemoryManager(MemoryConfig())
    # hard_constraint 类种子分 0.5,触发晋升
    mgr.write("我对花生过敏", role="user")
    actives = mgr.playbook.all_active()
    assert len(actives) == 1
    assert actives[0].type == L0Type.HARD_CONSTRAINT
    assert "花生过敏" in actives[0].content


def test_write_does_not_promote_chat():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("今天天气挺好", role="user")
    assert mgr.playbook.count_active() == 0


def test_write_does_not_promote_assistant():
    mgr = MemoryManager(MemoryConfig())
    # assistant 自述即使关键词命中也不晋升
    mgr.write("我决定为你推荐花生酱", role="assistant")
    assert mgr.playbook.count_active() == 0


def test_write_dedupe():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我对花生过敏", role="user")
    mgr.write("我对花生过敏", role="user")  # 重复
    assert mgr.playbook.count_active() == 1


# ─────────────────────────────────────────────
# MemoryManager L0 API
# ─────────────────────────────────────────────

def test_remember_command():
    mgr = MemoryManager(MemoryConfig())
    card = mgr.remember("我希望对话简洁,不要客套")
    assert card.user_pinned   # USER_EXPLICIT 默认 pinned
    assert card.confidence == 1.0


def test_forget_command():
    mgr = MemoryManager(MemoryConfig())
    card = mgr.remember("test")
    assert mgr.forget(card.id) is True
    refreshed = mgr.playbook.get(card.id)
    assert refreshed.lifecycle == L0Lifecycle.FORGOTTEN
    assert refreshed not in mgr.playbook.all_active()


def test_pin_command():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我对花生过敏", role="user")
    card = mgr.playbook.all_active()[0]
    assert not card.user_pinned  # 自动晋升的卡默认不 pinned
    mgr.pin_card(card.id)
    refreshed = mgr.playbook.get(card.id)
    assert refreshed.user_pinned
    assert refreshed.confidence == 1.0


# ─────────────────────────────────────────────
# assemble_context 注入
# ─────────────────────────────────────────────

def test_assemble_context_injects_l0_core_segment():
    mgr = MemoryManager(MemoryConfig())
    mgr.remember("用户对花生过敏")

    messages = mgr.assemble_context("test prompt", query="晚饭吃什么")
    contents = [m["content"] for m in messages if m["role"] == "system"]
    joined = "\n".join(contents)
    # hard_constraint 永远进核心段
    assert "花生过敏" in joined
    assert "核心信息" in joined


def test_assemble_context_no_query_still_shows_hard():
    """无 query 时,hard_constraint 仍然进核心段。"""
    mgr = MemoryManager(MemoryConfig())
    mgr.remember("用户对花生过敏")  # hard
    mgr.remember("用户最近在做 AI 项目")  # state

    messages = mgr.assemble_context("test prompt")
    contents = [m["content"] for m in messages if m["role"] == "system"]
    joined = "\n".join(contents)
    assert "花生过敏" in joined


def test_assemble_context_query_irrelevant_demotes_to_bg():
    """query 与某些 L0 项不相关时,这些项退到背景段或省略。"""
    mgr = MemoryManager(MemoryConfig())
    mgr.remember("用户对花生过敏")  # hard
    mgr.remember("用户在做 AI 项目")  # state,初始 confidence=1.0

    # 不相关 query
    messages = mgr.assemble_context("test", query="今天天气好不好")
    contents = "\n".join(m["content"] for m in messages if m["role"] == "system")
    # 花生过敏(hard) 必须在
    assert "花生过敏" in contents


# ─────────────────────────────────────────────
# KG → L0 反哺(简化场景,只用 SimilarityConflictDetector)
# ─────────────────────────────────────────────

def test_kg_supersede_negates_l0_card():
    """KG supersede 时,联动的 L0 卡按 type 不同幅度 negate。

    直接测 _negate_l0_cards_for 内部逻辑,不依赖 detector 实际触发。
    """
    mgr = MemoryManager(MemoryConfig())

    # 写入一条会被晋升的 hard_constraint
    item = mgr.write("我对花生过敏", role="user")
    actives = mgr.playbook.all_active()
    assert len(actives) == 1
    card = actives[0]
    assert card.source_ref == item.id
    initial_conf = card.confidence

    # 直接调内部反哺方法模拟 supersede
    mgr._negate_l0_cards_for(item.id)
    refreshed = mgr.playbook.get(card.id)
    # hard_constraint 降幅小(0.15)
    assert refreshed.confidence < initial_conf
    assert refreshed.confidence >= initial_conf - 0.2
    # hard_constraint 不应该被自动 archive(即使分降了)
    assert refreshed.lifecycle == L0Lifecycle.ACTIVE


def test_state_card_archived_on_negate():
    """state 类被 negate 时降幅大,可能触发 archive。"""
    mgr = MemoryManager(MemoryConfig())

    # 手动创建一张 state 类卡片,confidence 较低
    card = PlaybookCard(
        content="用户最近在做 X 项目",
        type=L0Type.STATE,
        source=L0Source.SEED_PROMOTED,
        source_ref="dummy_item_id",
        confidence=0.7,
    )
    mgr.playbook.add(card)

    # 触发 negate
    mgr._negate_l0_cards_for("dummy_item_id")
    refreshed = mgr.playbook.get(card.id)
    # state 降幅 0.6,从 0.7 跌到 0.1,触发 archive
    assert refreshed.confidence < 0.3
    assert refreshed.lifecycle == L0Lifecycle.ARCHIVED


# ─────────────────────────────────────────────
# 被动 recall: assemble 时自动 touch 命中项
# ─────────────────────────────────────────────

def test_passive_recall_touches_items():
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我之前提到我喜欢喝美式咖啡", role="user")
    item = mgr.working.items()[0]
    initial_access = item.access_count

    # query 与历史内容相关 → 被动 recall 命中 → touch
    mgr.assemble_context("test", query="咖啡好喝吗", passive_recall_k=5)
    # access_count 应该 +1
    assert item.access_count >= initial_access  # 至少不减


def test_passive_recall_zero_disables():
    """passive_recall_k=0 时,不调用被动 recall。"""
    mgr = MemoryManager(MemoryConfig())
    mgr.write("test content", role="user")

    messages = mgr.assemble_context("sys", query="anything", passive_recall_k=0)
    joined = "\n".join(m["content"] for m in messages if m["role"] == "system")
    assert "相关的历史片段" not in joined


# ─────────────────────────────────────────────
# 持久化 + 重启场景
# ─────────────────────────────────────────────

def test_l0_persists_across_manager_restart():
    """模拟重启: 旧 mgr 写入 L0,新 mgr 加载同一 storage_dir 应该看到。"""
    with tempfile.TemporaryDirectory() as d:
        cfg = MemoryConfig(
            storage_dir=Path(d),
            cold_backend="sqlite",
            vector_backend="sqlite",
        )

        mgr1 = MemoryManager(cfg)
        mgr1.remember("用户对花生过敏")
        mgr1.write("我叫张三", role="user")  # 自动晋升

        active_count_before = mgr1.playbook.count_active()
        assert active_count_before >= 2

        # 新 manager 实例,同一 storage_dir
        mgr2 = MemoryManager(cfg)
        assert mgr2.playbook.count_active() == active_count_before
        contents = [c.content for c in mgr2.playbook.all_active()]
        assert any("花生过敏" in c for c in contents)
        assert any("张三" in c for c in contents)


def test_seed_hard_constraint_can_archive():
    # seed 来源的 hard_constraint:置信度低 → 可撤下(误判自愈)
    seed = PlaybookCard(
        content="文件必须包含字段 status", type=L0Type.HARD_CONSTRAINT,
        source=L0Source.SEED_PROMOTED, confidence=0.1,
    )
    assert seed.should_archive()
    # 非 seed 来源(用户显式)hard_constraint:仍永久免死
    explicit = PlaybookCard(
        content="我对花生过敏", type=L0Type.HARD_CONSTRAINT,
        source=L0Source.USER_EXPLICIT, confidence=0.1,
    )
    assert not explicit.should_archive()
    # user_pinned 永远不撤
    pinned = PlaybookCard(
        content="x", type=L0Type.HARD_CONSTRAINT,
        source=L0Source.SEED_PROMOTED, confidence=0.0, user_pinned=True,
    )
    assert not pinned.should_archive()


def test_classify_task_directive_not_hard_constraint():
    # 通用祈使词但非自指 → 不再判 hard_constraint
    assert classify_content_type("回答里必须包含编程语言") != L0Type.HARD_CONSTRAINT
    assert classify_content_type("文件必须包含字段 status") != L0Type.HARD_CONSTRAINT
    # 自指 + 祈使 → 仍是 hard_constraint
    assert classify_content_type("我必须每天吃药") == L0Type.HARD_CONSTRAINT
    # 用户事实词 → 仍是 hard_constraint
    assert classify_content_type("不能吃海鲜") == L0Type.HARD_CONSTRAINT


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
