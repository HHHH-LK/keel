"""清理:把'新规则下不再是 hard_constraint 的 seed_promoted 卡'置 forgotten。"""
from my_agent_llms.memory.playbook.store import PlaybookStore
from my_agent_llms.memory.playbook.card import PlaybookCard, L0Type, L0Source
from my_agent_llms.memory.maintenance import purge_misclassified_seed_cards


def test_purge_removes_misclassified_seed_hard_constraints():
    store = PlaybookStore()   # 内存模式(path=None)
    bad = PlaybookCard(content="文件必须包含字段 status", type=L0Type.HARD_CONSTRAINT,
                       source=L0Source.SEED_PROMOTED)
    good = PlaybookCard(content="我对花生过敏", type=L0Type.HARD_CONSTRAINT,
                        source=L0Source.SEED_PROMOTED)
    explicit = PlaybookCard(content="回答必须含X", type=L0Type.HARD_CONSTRAINT,
                            source=L0Source.USER_EXPLICIT)  # 非 seed,不动
    store.add(bad)
    store.add(good)
    store.add(explicit)

    n = purge_misclassified_seed_cards(store)
    assert n == 1
    active = {c.content for c in store.all_active()}
    assert "文件必须包含字段 status" not in active      # 误判 → forgotten
    assert "我对花生过敏" in active                       # 真约束 → 保留
    assert "回答必须含X" in active                        # 非 seed → 不动(即便内容像指令)
