"""L0 来源补洞测试 —— 覆盖三处历史遗留:

1. 误标修复: 写入瞬间的种子分晋升应打 SEED_PROMOTED,而非 KG_PROMOTED
2. L1_GRADUATED: L1 项凭"反复访问 + pinned"的实绩毕业进 L0
3. LLM_REMEMBERED: LLM 主动调 remember 工具建卡
"""
from my_agent_llms.memory import (
    L0Source,
    L0Type,
    MemoryConfig,
    MemoryManager,
)


# ─────────────────────────────────────────────
# Fix 1: 误标修复 —— 种子分晋升的来源
# ─────────────────────────────────────────────

def test_seed_promotion_uses_seed_promoted_source():
    """写入瞬间靠关键词种子分晋升的卡,来源应是 SEED_PROMOTED(不是 KG)。"""
    mgr = MemoryManager(MemoryConfig())
    mgr.write("我对花生过敏", role="user")
    card = mgr.playbook.all_active()[0]
    assert card.source == L0Source.SEED_PROMOTED


# ─────────────────────────────────────────────
# Fix 2: L1_GRADUATED —— 实绩毕业
# ─────────────────────────────────────────────

def test_l1_item_graduates_after_repeated_access():
    """低种子分的 L1 项,靠 pinned + 反复访问的实绩毕业进 L0。"""
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    # 不含关键词 → 写入时不晋升
    item = mgr.write("项目代号是 Apollo", role="user")
    assert mgr.playbook.count_active() == 0

    # 模拟"久经考验": pin + 反复命中
    item.pinned = True
    for _ in range(6):
        item.touch()

    mgr.tick()
    actives = mgr.playbook.all_active()
    assert len(actives) == 1
    assert actives[0].source == L0Source.L1_GRADUATED
    assert "Apollo" in actives[0].content


def test_no_graduation_below_threshold():
    """访问次数不够,不毕业。"""
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    item = mgr.write("项目代号是 Apollo", role="user")
    item.pinned = True
    for _ in range(2):  # 低于阈值
        item.touch()
    mgr.tick()
    assert mgr.playbook.count_active() == 0


def test_no_graduation_when_not_pinned():
    """没 pinned(没被证明重要),即使访问够多也不毕业。"""
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    item = mgr.write("项目代号是 Apollo", role="user")
    # 不 pin
    for _ in range(6):
        item.touch()
    mgr.tick()
    assert mgr.playbook.count_active() == 0


def test_graduation_is_idempotent():
    """同一项不会每次 tick 都毕业一张新卡。"""
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    item = mgr.write("项目代号是 Apollo", role="user")
    item.pinned = True
    for _ in range(6):
        item.touch()
    mgr.tick()
    mgr.tick()
    assert mgr.playbook.count_active() == 1


def test_forgotten_item_does_not_regraduate():
    """已毕业的卡被 /forget 后,即使 L1 项仍 pinned+高访问,也不应再毕业一张新卡。

    回归保护: 毕业去重用 find_by_source_ref(查所有生命周期),
    forgotten 卡仍会挡住重新毕业 —— 尊重用户的显式遗忘。
    """
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    item = mgr.write("项目代号是 Apollo", role="user")
    item.pinned = True
    for _ in range(6):
        item.touch()
    mgr.tick()
    card = mgr.playbook.all_active()[0]

    # 用户显式忘记这张卡
    mgr.forget(card.id)
    assert mgr.playbook.count_active() == 0

    # 再 tick: item 仍在 L1、仍 pinned、仍高访问,但不应复活
    mgr.tick()
    assert mgr.playbook.count_active() == 0


def test_assistant_item_does_not_graduate():
    """assistant 自述不毕业(和写入晋升一致)。"""
    cfg = MemoryConfig(l0_graduate_min_hits=5)
    mgr = MemoryManager(cfg)
    item = mgr.write("我建议代号用 Apollo", role="assistant")
    item.pinned = True
    for _ in range(6):
        item.touch()
    mgr.tick()
    assert mgr.playbook.count_active() == 0


# ─────────────────────────────────────────────
# Fix 3: LLM_REMEMBERED —— LLM 主动 remember 工具
# ─────────────────────────────────────────────

def test_remember_tool_creates_llm_remembered_card():
    mgr = MemoryManager(MemoryConfig())
    from my_agent_llms.tools.builtin.remember import RememberTool
    tool = RememberTool(mgr)
    tool.run({"content": "用户偏好用 Python 写脚本"})
    actives = mgr.playbook.all_active()
    assert len(actives) == 1
    assert actives[0].source == L0Source.LLM_REMEMBERED
    assert "Python" in actives[0].content


def test_remember_tool_rejects_empty():
    mgr = MemoryManager(MemoryConfig())
    from my_agent_llms.tools.builtin.remember import RememberTool
    tool = RememberTool(mgr)
    tool.run({"content": "   "})
    assert mgr.playbook.count_active() == 0


def test_remember_tool_not_user_pinned():
    """LLM 记的卡不是用户显式 pin,可被衰减。"""
    mgr = MemoryManager(MemoryConfig())
    from my_agent_llms.tools.builtin.remember import RememberTool
    tool = RememberTool(mgr)
    tool.run({"content": "用户在做推荐系统"})
    card = mgr.playbook.all_active()[0]
    assert not card.user_pinned


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
