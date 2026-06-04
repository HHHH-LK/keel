from my_agent_llms.memory.context_engine import count_tokens


def test_count_tokens_positive_and_monotonic():
    # 空串 0;非空 >= 1;更长文本 token 不减
    assert count_tokens("") == 0
    assert count_tokens("hi") >= 1
    short = count_tokens("hello")
    long = count_tokens("hello hello hello hello hello")
    assert long >= short


from my_agent_llms.memory.context_engine import (
    ContextSegment, bigram_relevance, make_embedding_relevance,
)


def test_bigram_relevance_basic():
    assert bigram_relevance("用户对花生过敏", "花生") > 0.0
    assert bigram_relevance("完全无关", "xyz") == 0.0
    assert bigram_relevance("", "q") == 0.0


def test_embedding_relevance_falls_back_on_error():
    class BadProvider:
        def embed(self, text):
            raise RuntimeError("no backend")
    fn = make_embedding_relevance(BadProvider())
    # provider 抛错 → 自动回退 bigram,不抛异常
    assert fn("花生过敏", "花生") > 0.0


def test_embedding_relevance_cosine():
    # 假 provider:相同文本向量相同 → 余弦=1;正交 → 0
    vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    class FakeProvider:
        def embed(self, text):
            return vecs[text]
    fn = make_embedding_relevance(FakeProvider())
    assert abs(fn("a", "a") - 1.0) < 1e-6
    assert abs(fn("a", "b") - 0.0) < 1e-6


def test_segment_defaults():
    seg = ContextSegment(source="l1", role="user", content="hi",
                         priority=0.5, tokens=1, floor=False, order=6)
    assert seg.seq == 0
    assert seg.item_id is None


from my_agent_llms.memory.context_engine import ContextEngine


def _seg(source, content, *, priority=0.5, tokens=None, floor=False,
         order=6, seq=0, item_id=None, role="system"):
    return ContextSegment(
        source=source, role=role, content=content, priority=priority,
        tokens=tokens if tokens is not None else max(1, len(content) // 3),
        floor=floor, order=order, seq=seq, item_id=item_id,
    )


def test_dedup_exact_keeps_more_authoritative():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("l1", "用户对花生过敏", order=6, seq=1),
        _seg("l0-core", "用户对花生过敏", order=1, seq=0),
    ]
    kept = eng._dedup(segs)
    sources = {s.source for s in kept}
    assert "l0-core" in sources       # 权威的留下
    assert "l1" not in sources        # 重复的被删


def test_dedup_substring_drops_contained():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("l0-core", "喜欢猫", order=1, seq=0),
        _seg("recall", "用户说他非常喜欢猫还有狗", order=5, seq=1),
    ]
    kept = eng._dedup(segs)
    # "喜欢猫" 被更长的 recall 完全包含 → 删短的 l0-core
    assert any(s.source == "recall" for s in kept)
    assert all(s.source != "l0-core" for s in kept)


def test_dedup_item_id_prefers_l1():
    eng = ContextEngine(dedup=True)
    segs = [
        _seg("recall", "片段预览", order=5, seq=1, item_id="x1"),
        _seg("l1", "完整原文不同字面", order=6, seq=0, item_id="x1"),
    ]
    kept = eng._dedup(segs)
    assert any(s.source == "l1" for s in kept)
    assert all(s.source != "recall" for s in kept)


def test_dedup_disabled_keeps_all():
    eng = ContextEngine(dedup=False)
    segs = [_seg("l1", "同样的话", seq=0), _seg("l0-core", "同样的话", order=1, seq=1)]
    assert len(eng._dedup(segs)) == 2


def test_select_never_exceeds_budget():
    eng = ContextEngine(dedup=False)
    segs = [_seg("l1", "x" * 300, tokens=100, order=6, seq=i) for i in range(20)]
    chosen, report = eng._select(segs, budget=300)
    assert report.used <= 300


def test_select_floors_always_kept_even_low_priority():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("system", "sys", tokens=50, floor=True, order=0, seq=0),
        _seg("l0-core", "硬约束", tokens=50, floor=True, order=1, seq=1, priority=1.0),
        _seg("recall", "高分但非保底", tokens=50, floor=False, order=5, seq=2, priority=0.9),
        _seg("l2", "低分非保底", tokens=50, floor=False, order=2, seq=3, priority=0.1),
    ]
    chosen, report = eng._select(segs, budget=120)
    srcs = {s.source for s in chosen}
    assert "system" in srcs and "l0-core" in srcs   # 保底永在
    assert "recall" not in srcs and "l2" not in srcs   # 剩余预算不足,非保底全丢


def test_select_drops_lowest_priority_first():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("kg", "高分", tokens=40, floor=False, order=3, seq=0, priority=0.9),
        _seg("l2", "低分", tokens=40, floor=False, order=2, seq=1, priority=0.2),
    ]
    chosen, report = eng._select(segs, budget=40)  # 只装得下一个
    assert any(s.source == "kg" for s in chosen)    # 高分留下
    assert ("l2", 40) in report.dropped             # 低分被丢


def test_select_hard_cap_when_floors_exceed_budget():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("system", "sys", tokens=50, floor=True, order=0, seq=0),
        _seg("l1", "最近1", tokens=50, floor=True, order=6, seq=1),
        _seg("l1", "最近2", tokens=50, floor=True, order=6, seq=2),
    ]
    chosen, report = eng._select(segs, budget=80)
    assert report.used <= 80
    assert any(s.source == "system" for s in chosen)   # system 永不被砍


def test_build_orders_by_authority():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("l1", "最近对话", tokens=10, floor=True, order=6, seq=3, role="user"),
        _seg("system", "你是助手", tokens=10, floor=True, order=0, seq=0, role="system"),
        _seg("l0-core", "用户是工程师", tokens=10, floor=False, order=1, seq=1, priority=0.9),
    ]
    result = eng.build(segs, budget=1000)
    roles_contents = [(m["role"], m["content"]) for m in result.messages]
    # system 在最前,l1 在最后
    assert roles_contents[0][0] == "system"
    assert "你是助手" in roles_contents[0][1]
    assert roles_contents[-1][1] == "最近对话"


def test_build_groups_l0_core_under_one_heading():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("l0-core", "- 用户是工程师", tokens=10, order=1, seq=0, priority=0.9),
        _seg("l0-core", "- 用户喜欢猫", tokens=10, order=1, seq=1, priority=0.8),
    ]
    result = eng.build(segs, budget=1000)
    core_msgs = [m for m in result.messages if "核心信息" in m["content"]]
    assert len(core_msgs) == 1                        # 两张卡归并成一条 message
    assert "用户是工程师" in core_msgs[0]["content"]
    assert "用户喜欢猫" in core_msgs[0]["content"]


def test_build_report_records_drop():
    eng = ContextEngine(dedup=False)
    segs = [
        _seg("kg", "高分", tokens=40, order=3, seq=0, priority=0.9),
        _seg("l2", "低分", tokens=40, order=2, seq=1, priority=0.2),
    ]
    result = eng.build(segs, budget=40)
    assert result.report.used <= 40
    assert ("l2", 40) in result.report.dropped


from my_agent_llms.memory.config import MemoryConfig


def test_memory_config_context_defaults():
    cfg = MemoryConfig()
    assert cfg.context_budget_tokens == 12000
    assert cfg.context_dedup is True
    assert cfg.context_relevance == "embedding"


# ── Task 7: ContextEngine 接线进 MemoryManager 的集成测试 ──
from my_agent_llms.memory.manager import MemoryManager


def _mgr():
    # 内存后端、关冲突检测、关 tick:纯组装路径
    return MemoryManager(config=MemoryConfig(
        cold_backend="none", vector_backend="memory",
        conflict_strength="off", tick_mode="off",
        context_budget_tokens=2000,
    ))


def test_assemble_context_never_exceeds_budget():
    m = _mgr()
    for i in range(40):
        m.write(f"这是第{i}条较长的对话内容用来撑大上下文窗口" * 3, role="user")
    msgs = m.assemble_context("你是助手", query="对话")
    total = sum(count_tokens(x["content"]) for x in msgs)
    assert total <= 2000


def test_assemble_context_keeps_system_first():
    m = _mgr()
    m.write("你好", role="user")
    msgs = m.assemble_context("你是助手", query="你好")
    assert msgs[0]["role"] == "system"
    assert "你是助手" in msgs[0]["content"]


def test_assemble_context_reports_available():
    m = _mgr()
    m.write("一些内容", role="user")
    m.assemble_context("sys", query="内容")
    assert m._last_budget_report is not None
    assert m._last_budget_report.budget == 2000
